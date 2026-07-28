#!/usr/bin/env python3
"""
fetch_all_trials.py  —  MAIN SCRIPT
===================================

Give it a drug name; it queries every registry, writes one Excel file per
registry, and one combined master Excel with a `source` column.

    python fetch_all_trials.py "pembrolizumab"
    python fetch_all_trials.py "metformin" --outdir results --max-records 50
    python fetch_all_trials.py "aspirin" --only ClinicalTrials.gov CTRI
    python fetch_all_trials.py "semaglutide" --workers 4 --strict

Sources (exactly these values appear in the `source` column):
    EU Clinical Trials · ClinicalTrials.gov · CTRI · ChiCTR
    JRCT · ANZCTR · CRIS · ReBEC

Output in --outdir (default: ./<drug>_trials/):
    <drug>_ALL_REGISTRIES.xlsx     <- master, every source, `source` column
    <drug>_clinicaltrialsgov.xlsx
    <drug>_ctri.xlsx
    ... one per registry that returned rows

Requires: requests, beautifulsoup4, pandas, openpyxl, lxml (optional)
    pip install requests beautifulsoup4 pandas openpyxl
No LLM / AI service is used anywhere.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from registry_common import (
    ALLOWED_SOURCES, SRC_ANZCTR, SRC_CHICTR, SRC_CRIS, SRC_CTGOV, SRC_CTRI,
    SRC_EUCT, SRC_JRCT, SRC_REBEC, UNIFIED_COLUMNS, matches_drug, write_excel,
)

import anzctr_trials
import chictr_trials
import cris_trials
import ctgov_trials
import ctri_trials
import euct_trials
import gemini_enrich
import ictrp_trials
import jrct_trials
import rebec_trials

try:
    from gemini_summarizer import AI_COLUMNS, GeminiSummarizer
except Exception:                                          # noqa: BLE001
    AI_COLUMNS, GeminiSummarizer = [], None

# registry key -> (canonical source name, fetch function, short file slug)
REGISTRIES: Dict[str, tuple] = {
    "ctgov":   (SRC_CTGOV,  ctgov_trials.fetch,  "clinicaltrialsgov"),
    "euct":    (SRC_EUCT,   euct_trials.fetch,   "eu_clinical_trials"),
    "ctri":    (SRC_CTRI,   ctri_trials.fetch,   "ctri"),
    "chictr":  (SRC_CHICTR, chictr_trials.fetch, "chictr"),
    "jrct":    (SRC_JRCT,   jrct_trials.fetch,   "jrct"),
    "anzctr":  (SRC_ANZCTR, anzctr_trials.fetch, "anzctr"),
    "cris":    (SRC_CRIS,   cris_trials.fetch,   "cris"),
    "rebec":   (SRC_REBEC,  rebec_trials.fetch,  "rebec"),
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _enforce_sources(rows: List[Dict[str, Any]], expected: str
                     ) -> List[Dict[str, Any]]:
    """Guarantee the source column only ever holds an allowed value."""
    clean_rows = []
    for row in rows:
        src = row.get("source", "")
        if src not in ALLOWED_SOURCES:
            row["source"] = expected
        clean_rows.append(row)
    return clean_rows


def _run_one(key: str, drug: str, args) -> tuple:
    source, fetch_fn, slug = REGISTRIES[key]
    started = time.time()
    try:
        rows = fetch_fn(drug, max_records=args.max_records,
                        details=not args.no_details)
        rows = _enforce_sources(rows or [], source)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[{source}] FAILED: {exc}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        rows = []
    elapsed = time.time() - started
    return key, source, slug, rows, elapsed


def _fallback_via_ictrp(drug: str, missing: List[str], args
                        ) -> Dict[str, List[Dict[str, Any]]]:
    """Back-fill registries that returned nothing, using WHO ICTRP."""
    if not missing:
        return {}
    print(f"\n[ICTRP fallback] attempting back-fill for: {', '.join(missing)}",
          file=sys.stderr)
    try:
        rows = ictrp_trials.fetch(drug, max_records=None,
                                  only_sources=missing)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[ICTRP fallback] failed: {exc}", file=sys.stderr)
        return {}

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        src = row.get("source")
        if src in ALLOWED_SOURCES:
            grouped.setdefault(src, []).append(row)
    for src, rws in grouped.items():
        print(f"[ICTRP fallback] recovered {len(rws)} row(s) for {src}",
              file=sys.stderr)
    return grouped


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch clinical trials for a drug from all supported "
                    "registries and write per-registry + combined Excel files.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drug", help="drug / intervention name, e.g. 'pembrolizumab'")
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: ./<drug>_trials)")
    ap.add_argument("--only", nargs="*", default=None, metavar="SOURCE",
                    help=f"run only these sources. Choices: {ALLOWED_SOURCES}")
    ap.add_argument("--skip", nargs="*", default=None, metavar="SOURCE",
                    help="skip these sources")
    ap.add_argument("--max-records", type=int, default=None,
                    help="cap rows fetched per registry")
    ap.add_argument("--no-details", action="store_true",
                    help="skip per-trial detail pages (much faster, fewer fields)")
    ap.add_argument("--strict", action="store_true",
                    help="keep only trials naming the drug in an intervention field")
    ap.add_argument("--workers", type=int, default=3,
                    help="registries to query in parallel (default 3; "
                         "keep low to stay polite)")
    ap.add_argument("--no-fallback", action="store_true",
                    help="do not back-fill empty registries via WHO ICTRP")
    ap.add_argument("--traceback", action="store_true",
                    help="print full tracebacks on registry failure")

    g = ap.add_argument_group("Gemini 2.5 Flash key-findings summarisation")
    g.add_argument("--summarize", "--summarise", dest="summarize",
                   action="store_true",
                   help="add ai_* key-findings columns using Gemini 2.5 Flash "
                        "(needs GEMINI_API_KEY)")
    g.add_argument("--gemini-model", default="gemini-2.5-flash")
    g.add_argument("--gemini-workers", type=int, default=4)
    g.add_argument("--gemini-rpm", type=int, default=60,
                   help="requests-per-minute cap (default 60)")
    g.add_argument("--gemini-thinking", type=int, default=0,
                   help="thinking token budget (0 = off, cheapest)")
    g.add_argument("--grounded", action="store_true",
                   help="let Gemini use Google Search to find PUBLISHED results "
                        "for trials whose registry record has none; results land "
                        "in ai_published_findings + ai_sources")
    g.add_argument("--only-with-results", action="store_true",
                   help="only call Gemini for rows that already contain results "
                        "text (cheaper)")
    g.add_argument("--no-gemini-cache", action="store_true")

    m = ap.add_argument_group(
        "Metric extraction (entire_trial_information + weight/HbA1c/MASH/ALT)")
    m.add_argument("--metrics", action="store_true",
                   help="add entire_trial_information plus weight_reduction, "
                        "hba1c_reduction, mash and alt_reduction columns, "
                        "extracted by Gemini 2.5 Flash with grounding guards")
    m.add_argument("--metrics-text-only", action="store_true",
                   help="add entire_trial_information but do NOT call the LLM; "
                        "the four metric columns all read INFO N/A")
    m.add_argument("--metrics-key", default=None,
                   help="Gemini API key for metric extraction "
                        "(default: $GEMINI_API_KEY)")
    m.add_argument("--metrics-model", default=gemini_enrich.DEFAULT_MODEL)
    m.add_argument("--metrics-workers", type=int, default=4)
    m.add_argument("--metrics-cache", default=".gemini_metrics_cache.json",
                   help="cache file so re-runs do not re-pay for the same rows")
    args = ap.parse_args()

    drug_slug = _slug(args.drug)
    outdir = args.outdir or f"{drug_slug}_trials"
    os.makedirs(outdir, exist_ok=True)

    # decide which registries to run
    keys = list(REGISTRIES)
    if args.only:
        wanted = {s.lower() for s in args.only}
        keys = [k for k in keys
                if REGISTRIES[k][0].lower() in wanted or k.lower() in wanted]
        if not keys:
            print(f"No registry matched --only {args.only}. "
                  f"Valid: {ALLOWED_SOURCES}", file=sys.stderr)
            return 2
    if args.skip:
        unwanted = {s.lower() for s in args.skip}
        keys = [k for k in keys
                if REGISTRIES[k][0].lower() not in unwanted and k.lower() not in unwanted]

    print("=" * 70, file=sys.stderr)
    print(f"Drug          : {args.drug}", file=sys.stderr)
    print(f"Registries    : {', '.join(REGISTRIES[k][0] for k in keys)}",
          file=sys.stderr)
    print(f"Output folder : {os.path.abspath(outdir)}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    results: Dict[str, Dict[str, Any]] = {}

    if args.workers > 1 and len(keys) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one, k, args.drug, args): k for k in keys}
            for fut in as_completed(futures):
                key, source, slug, rows, elapsed = fut.result()
                results[key] = {"source": source, "slug": slug,
                                "rows": rows, "elapsed": elapsed}
                print(f"[{source}] {len(rows)} row(s) in {elapsed:.1f}s",
                      file=sys.stderr)
    else:
        for k in keys:
            key, source, slug, rows, elapsed = _run_one(k, args.drug, args)
            results[key] = {"source": source, "slug": slug,
                            "rows": rows, "elapsed": elapsed}
            print(f"[{source}] {len(rows)} row(s) in {elapsed:.1f}s",
                  file=sys.stderr)

    # ICTRP back-fill for empty registries
    if not args.no_fallback:
        missing = [results[k]["source"] for k in results if not results[k]["rows"]]
        # ClinicalTrials.gov and EU have solid direct routes; don't back-fill them
        missing = [m for m in missing if m not in (SRC_CTGOV, SRC_EUCT)]
        recovered = _fallback_via_ictrp(args.drug, missing, args)
        for key, meta in results.items():
            if not meta["rows"] and meta["source"] in recovered:
                meta["rows"] = _enforce_sources(recovered[meta["source"]],
                                                meta["source"])
                meta["via_ictrp"] = True

    # strict drug filter
    if args.strict:
        for meta in results.values():
            before = len(meta["rows"])
            meta["rows"] = [r for r in meta["rows"] if matches_drug(r, args.drug)]
            if before != len(meta["rows"]):
                print(f"[{meta['source']}] strict filter: {before} -> "
                      f"{len(meta['rows'])}", file=sys.stderr)

    # ── Gemini 2.5 Flash key-findings pass ───────────────────────────────────
    ai_cols: List[str] = []
    if args.summarize:
        if GeminiSummarizer is None:
            print("\n[Gemini] gemini_summarizer.py not importable – skipping.",
                  file=sys.stderr)
        else:
            all_rows = [r for k in keys for r in results.get(k, {}).get("rows", [])]
            if not all_rows:
                print("\n[Gemini] nothing to summarise.", file=sys.stderr)
            else:
                print(f"\n{'=' * 70}", file=sys.stderr)
                print(f"Gemini key-findings pass ({args.gemini_model}) over "
                      f"{len(all_rows)} trial(s)", file=sys.stderr)
                print("NOTE: ai_* columns are LLM-generated. Verify anything that "
                      "matters against the url column.", file=sys.stderr)
                print("=" * 70, file=sys.stderr)
                try:
                    sm = GeminiSummarizer(
                        model=args.gemini_model,
                        rpm=args.gemini_rpm,
                        thinking_budget=args.gemini_thinking,
                        grounded=args.grounded,
                        cache_path=None if args.no_gemini_cache
                        else os.path.join(outdir, ".gemini_cache.json"))
                    sm.summarize_rows(all_rows, workers=args.gemini_workers,
                                      only_with_results=args.only_with_results)
                    ai_cols = list(AI_COLUMNS)
                except RuntimeError as exc:
                    print(f"[Gemini] skipped: {exc}", file=sys.stderr)
                except Exception as exc:                       # noqa: BLE001
                    print(f"[Gemini] failed: {exc}", file=sys.stderr)
                    if args.traceback:
                        traceback.print_exc()

    out_columns = list(UNIFIED_COLUMNS) + list(ai_cols)

    # ── entire_trial_information + weight / HbA1c / MASH / ALT columns ───────
    if args.metrics or args.metrics_text_only:
        all_rows = [r for k in keys for r in results.get(k, {}).get("rows", [])]
        print(f"\nExtracting metric columns for {len(all_rows)} row(s) ...",
              file=sys.stderr)
        try:
            gemini_enrich.enrich_rows(
                all_rows,
                api_key=args.metrics_key,
                model=args.metrics_model,
                workers=args.metrics_workers,
                cache_path=(None if args.no_gemini_cache else args.metrics_cache),
                skip_llm=args.metrics_text_only,
            )
            out_columns = out_columns + [c for c in gemini_enrich.ENRICHED_COLUMNS
                                         if c not in out_columns]
        except Exception as exc:                               # noqa: BLE001
            print(f"[metrics] failed: {exc}", file=sys.stderr)
            if args.traceback:
                traceback.print_exc()

    # ── per-registry Excel files ─────────────────────────────────────────────
    print("\nWriting per-registry Excel files ...", file=sys.stderr)
    written: List[str] = []
    for key in keys:
        meta = results.get(key)
        if not meta or not meta["rows"]:
            continue
        path = os.path.join(outdir, f"{drug_slug}_{meta['slug']}.xlsx")
        if write_excel(meta["rows"], path, out_columns,
                       sheet_name=meta["source"][:31]):
            written.append(path)

    # ── combined master Excel ────────────────────────────────────────────────
    combined: List[Dict[str, Any]] = []
    for key in keys:
        meta = results.get(key)
        if meta:
            combined.extend(meta["rows"])

    # master keeps only the unified columns so the sheet stays readable
    slim = []
    for row in combined:
        slim.append({c: row.get(c, "") for c in out_columns})
    slim.sort(key=lambda r: (ALLOWED_SOURCES.index(r["source"])
                             if r["source"] in ALLOWED_SOURCES else 99,
                             r.get("trial_id", "")))

    master = os.path.join(outdir, f"{drug_slug}_ALL_REGISTRIES.xlsx")
    print("\nWriting combined master Excel ...", file=sys.stderr)
    write_excel(slim, master, out_columns, sheet_name="All Registries")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    for key in keys:
        meta = results.get(key, {})
        note = "  (via WHO ICTRP)" if meta.get("via_ictrp") else ""
        print(f"  {meta.get('source', key):<22} {len(meta.get('rows', [])):>6} "
              f"trial(s){note}", file=sys.stderr)
    print(f"  {'TOTAL':<22} {len(slim):>6} trial(s)", file=sys.stderr)

    if args.metrics or args.metrics_text_only:
        print("\nMetric coverage (rows with data, rest read INFO N/A):",
              file=sys.stderr)
        for m in gemini_enrich.METRICS:
            n = sum(1 for r in slim if r.get(m, gemini_enrich.NA) != gemini_enrich.NA)
            print(f"  {m:<22} {n:>6} / {len(slim)}", file=sys.stderr)
    if ai_cols:
        with_res = sum(1 for r in slim
                       if str(r.get("ai_has_registry_results", "")).lower() == "yes")
        pub = sum(1 for r in slim
                  if r.get("ai_published_findings") and
                  "No published results" not in str(r.get("ai_published_findings")))
        print(f"\n  findings from registry data : {with_res}", file=sys.stderr)
        if args.grounded:
            print(f"  findings from web search    : {pub}", file=sys.stderr)
        print(f"  no results reported         : {len(slim) - with_res}",
              file=sys.stderr)
    print(f"\nMaster file : {os.path.abspath(master)}", file=sys.stderr)
    for p in written:
        print(f"  per-source: {os.path.abspath(p)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
