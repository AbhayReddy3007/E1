#!/usr/bin/env python3
"""
enrich_ctgov_with_gemini.py
============================
Bridges the fast, authoritative registry scrapers — ctgov_trials.py
(ClinicalTrials.gov API v2) and euct_trials.py (EU CTIS + EudraCT,
via ctis_drug_trials.py / eudract_drug_trials.py) — with
gemini_extractor.py's Step 2 detail extraction (Gemini + Google Search —
used for efficacy fields that aren't in the raw registry data: HbA1c
change, weight loss %, ALT reduction, MASH resolution, dosage, company,
etc.).

Instead of letting gemini_extractor.py run its own Step 1 registry search,
this script hands Gemini the exact trial IDs already pulled from the
registry scrapers, jumps straight to Step 2 (get_trial_details), and
merges the enriched fields back onto the original rows — so nothing is
duplicated or re-discovered, only filled in.

Usage:
    python enrich_ctgov_with_gemini.py "semaglutide"
    python enrich_ctgov_with_gemini.py "semaglutide" --outdir results --max-records 50
    python enrich_ctgov_with_gemini.py "semaglutide" --sources ctgov          # US only
    python enrich_ctgov_with_gemini.py "semaglutide" --sources euct          # EU only
    python enrich_ctgov_with_gemini.py "semaglutide" --sources ctgov,euct    # both (default)

Requires (same as gemini_extractor.py):
    pip install google-genai aiohttp python-dotenv json-repair
    GOOGLE_API_KEY set in the environment or a .env file

Also requires ctgov_trials.py, euct_trials.py (and, for EU coverage,
ctis_drug_trials.py / eudract_drug_trials.py on PYTHONPATH) and
gemini_extractor.py to be importable. If a given source module can't be
imported, this script just skips that source with a warning rather than
failing outright. If registry_common.py is not available, this script
falls back to pandas for writing the output Excel file.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from typing import Any, Dict, List, Optional

from gemini_extractor import (
    MAX_WORKERS,
    RATE_LIMIT_DELAY,
    get_trial_details,
    fetch_ctgov_dates,
)

# Our own batching size for the FIRST pass — deliberately smaller than
# gemini_extractor's own BATCH_SIZE (8). Packing 8 trials x ~20 fields,
# including numeric outcome data (HbA1c change, weight loss %, ALT
# reduction, MASH resolution) that usually has to be dug out of a results
# page or a published paper rather than read off the registry front page,
# into one Gemini call measurably dilutes how much research depth each
# individual trial gets. We don't touch the imported BATCH_SIZE constant
# since other callers of gemini_extractor.py may rely on it.
FIRST_PASS_BATCH_SIZE = 4

# Batch size for the automatic retry pass on trials that came back with no
# efficacy data — one trial per call, so the model can spend its full
# search budget on just that trial instead of splitting it 4-8 ways.
RETRY_BATCH_SIZE = 1

# The efficacy-specific fields we actually care about recovering. If ALL
# of these are still empty after the first pass, the trial is a retry
# candidate. (We deliberately don't retry on missing trial_title/phase/etc
# — those are usually genuinely unavailable or already covered by the
# registry scraper, and retrying for them wastes API calls.)
EFFICACY_ROW_FIELDS = [
    "dosage", "hba1c_change_pct", "weight_change_pct",
    "alt_reduction_pct", "mash_resolution_pct", "company_name",
]


def _efficacy_missing(row: Dict[str, Any]) -> bool:
    return not any(str(row.get(f, "")).strip() for f in EFFICACY_ROW_FIELDS)


try:
    from registry_common import write_excel
except ImportError:
    write_excel = None

# Registry source modules. Each exposes fetch(drug, max_records=..., details=...)
# -> List[Dict] of rows sharing the same unified schema (trial_id, title,
# public_title, status, phase, sponsor, countries, url, ...) via registry_common.
# A source that fails to import is skipped (with a warning) rather than
# crashing the whole run, so this script still works if e.g. ctis/eudract
# scrapers aren't present on disk.
SOURCE_MODULES: Dict[str, Any] = {}

try:
    import ctgov_trials
    SOURCE_MODULES["ctgov"] = ctgov_trials
except ImportError as exc:
    print(f"  [warn] ctgov_trials.py not importable, skipping CT.gov: {exc}",
          file=sys.stderr)

try:
    import euct_trials
    SOURCE_MODULES["euct"] = euct_trials
except ImportError as exc:
    print(f"  [warn] euct_trials.py not importable, skipping EU (CTIS/EudraCT): {exc}",
          file=sys.stderr)

# Trial-ID prefixes that identify a row as CT.gov-native. Only these are
# safe to pass to fetch_ctgov_dates(), which calls the ClinicalTrials.gov
# API and won't recognise EU CTIS numbers (CT number, e.g. "2023-501234-56-00")
# or legacy EudraCT numbers (e.g. "2014-001234-56").
def _is_ctgov_id(trial_id: str) -> bool:
    return bool(re.match(r"^NCT\d+$", str(trial_id or ""), flags=re.IGNORECASE))


# Maps gemini_extractor's Step-2 output field names -> the column names on
# your ctgov_trials.py rows / final Excel sheet. Edit the right-hand side
# if your column names differ (e.g. to match FINAL_COLUMNS in
# fetch_all_trials.py).
GEMINI_FIELD_MAP = {
    "Dosage": "dosage",
    "Phase": "phase",
    "Status": "phase_status",
    "Trial Title": "trial_title",
    "Trial_Study_Type": "trial_study_type",
    "Size": "trial_size",
    "Primary Region": "trial_location",
    "Start Date": "trial_start_date",
    "Completion Date": "trial_completion_date",
    "HbA1c Change (%)": "hba1c_change_pct",
    "HbA1c Duration": "hba1c_duration",
    "Weight Loss (%)": "weight_change_pct",
    "Weight Duration": "weight_duration",
    "ALT Reduction (%)": "alt_reduction_pct",
    "ALT Duration": "alt_duration",
    "MASH Outcome (%)": "mash_resolution_pct",
    "MASH Duration": "mash_duration",
    "Company": "company_name",
    "Source URL": "source_url",
}


# The only columns that should appear in the final output, in this order.
FINAL_COLUMNS = [
    "molecule_name",
    "registry_source",
    "trial_id",
    "dosage",
    "phase",
    "trial_title",
    "trial_study_type",
    "trial_size",
    "trial_location",
    "trial_start_date",
    "trial_completion_date",
    "phase_status",
    "hba1c_change_pct",
    "hba1c_duration",
    "weight_change_pct",
    "weight_duration",
    "alt_reduction_pct",
    "alt_duration",
    "mash_resolution_pct",
    "mash_duration",
    "company_name",
    "source_url",
]


def normalise_phase(raw: str) -> str:
    """Normalise 'Phase III', 'PHASE3', 'Phase 2/3', etc. down to '1'/'2'/'3'/'4'
    (or '2/3' style for combined phases)."""
    if not raw:
        return ""
    s = re.sub(r"(?i)\bphase\s*", "phase ", str(raw))
    phase_map = {
        "one": "1", "two": "2", "three": "3", "four": "4",
        "i": "1", "ii": "2", "iii": "3", "iv": "4",
        "1": "1", "2": "2", "3": "3", "4": "4",
    }
    nums = []
    for tok in re.split(r"[/,;\s]+", s.lower()):
        tok = tok.strip("(). ")
        if tok in phase_map and phase_map[tok] not in nums:
            nums.append(phase_map[tok])
    return "/".join(nums) if nums else str(raw).strip()


def remap_row(row: Dict[str, Any], drug: str) -> Dict[str, Any]:
    """Collapse a merged registry+Gemini row down to exactly FINAL_COLUMNS.

    Falls back to the raw registry field name (e.g. 'title', 'countries',
    'status' — shared by ctgov_trials.py and euct_trials.py via
    registry_common) if the Gemini-named field wasn't filled in.
    """
    return {
        "molecule_name": drug,
        "registry_source": row.get("source", "") or row.get("registry_source", ""),
        "trial_id": row.get("trial_id", ""),
        "dosage": row.get("dosage", ""),
        "phase": normalise_phase(row.get("phase", "")),
        "trial_title": row.get("trial_title", "") or row.get("title", "")
                        or row.get("public_title", ""),
        "trial_study_type": row.get("trial_study_type", "") or row.get("study_type", ""),
        "trial_size": row.get("trial_size", "") or row.get("actual_enrollment", "")
                      or row.get("target_enrollment", ""),
        "trial_location": row.get("trial_location", "") or row.get("countries", ""),
        "trial_start_date": row.get("trial_start_date", "") or row.get("start_date", ""),
        "trial_completion_date": row.get("trial_completion_date", "")
                                  or row.get("completion_date", ""),
        "phase_status": row.get("phase_status", "") or row.get("status", ""),
        "hba1c_change_pct": row.get("hba1c_change_pct", ""),
        "hba1c_duration": row.get("hba1c_duration", ""),
        "weight_change_pct": row.get("weight_change_pct", ""),
        "weight_duration": row.get("weight_duration", ""),
        "alt_reduction_pct": row.get("alt_reduction_pct", ""),
        "alt_duration": row.get("alt_duration", ""),
        "mash_resolution_pct": row.get("mash_resolution_pct", ""),
        "mash_duration": row.get("mash_duration", ""),
        "company_name": row.get("company_name", "") or row.get("sponsor", ""),
        "source_url": row.get("source_url", "") or row.get("url", ""),
    }


def _to_gemini_trial_stub(row: Dict[str, Any]) -> Dict[str, str]:
    """gemini_extractor.get_trial_details() only needs an ID + Program_Name
    per trial to build its prompt — it looks up everything else itself.
    The field is still called "NCT_ID" because that's the key
    get_trial_details expects, but it works fine as a generic trial
    identifier — CT.gov NCT numbers, EU CTIS CT numbers, or legacy
    EudraCT numbers all get passed through and searched for as-is."""
    return {
        "NCT_ID": row.get("trial_id", ""),
        "Program_Name": row.get("public_title", "") or row.get("title", "") or "N/A",
    }


def _clean_trial_id(raw_id: str) -> str:
    """Gemini sometimes returns 'NCT01234567 (STEP 1)' style values."""
    return raw_id.split("(")[0].strip().split(" ")[0].strip()


def _is_combination_molecule(drug: str) -> bool:
    """Heuristic: does this molecule name look like a fixed-dose
    combination (e.g. "Cagrilintide+Semaglutide", "Cagrilintide + Semaglutide")
    rather than a single active ingredient?"""
    return bool(re.search(r"\+|\band\b", drug, flags=re.IGNORECASE))


# gemini_extractor.py's Step 2 prompt opens with a validation gate:
# "verify each trial is a {molecule} MONOTHERAPY trial ... SKIP trials
# testing combination drugs (e.g., CagriSema = Cagrilintide + Semaglutide
# should be EXCLUDED when searching for Semaglutide)". That's written for
# single-ingredient searches, where you WANT combination trials filtered
# out. If {molecule} is itself passed in as a combination name (e.g.
# "Cagrilintide+Semaglutide"), that same instruction — with your exact
# molecule spelled out as the example of what to exclude — pushes Gemini
# to under-extract or blank out efficacy data for the very trials you're
# asking about. This override is appended whenever the molecule name looks
# like a combination, telling Gemini the monotherapy filter doesn't apply.
COMBINATION_OVERRIDE = """
OVERRIDE ON THE MONOTHERAPY VALIDATION INSTRUCTION ABOVE:
The molecule under study here IS a fixed-dose combination product. The
earlier instruction to SKIP/exclude "combination drug" trials does NOT
apply in this case — every trial ID given to you below was already
confirmed to test this exact combination as its primary intervention.
Do not decline, blank out, or under-extract efficacy fields for these
trials on the basis that they involve a drug combination. Extract HbA1c
change, weight loss, ALT reduction, MASH outcome, dosage, etc. exactly as
you would for any other Phase 2/3 trial with the combination as the
tested product.
"""


# gemini_extractor.py's Step 2 prompt is written CT.gov-first: its own
# example for the "Trial ID" output field is an NCT number, its example
# "Source URL" is a clinicaltrials.gov URL, and its DATA SOURCES list
# mentions the legacy EudraCT register (clinicaltrialsregister.eu) but
# never CTIS at all. Left alone, that biases Gemini to either come up
# empty for EU CTIS/EudraCT trial IDs, or — worse — swap in an NCT number
# for the same molecule/trial instead of echoing the ID we gave it, which
# then fails the exact-match merge below and looks like "no data returned".
#
# We don't edit gemini_extractor.py itself (other callers may depend on
# its exact prompt); instead we use its extra_fields_prompt hook to patch
# in registry-appropriate guidance per batch, keyed by registry_source.
EXTRA_PROMPT_BY_SOURCE: Dict[str, str] = {
    "euct": """
ADDITIONAL SOURCE GUIDANCE FOR THIS BATCH (EU trials):
- These trial IDs come from the EU CTIS portal (euclinicaltrials.eu,
  format like "2023-501234-56-00") or the legacy EudraCT / EU-CTR
  register (clinicaltrialsregister.eu, format like "2014-001234-56").
  Search BOTH of those sites directly by ID, in addition to
  ClinicalTrials.gov and PubMed (an EU trial may also be registered
  there under its own NCT number as a secondary ID — use that only to
  find published efficacy results, not to replace the ID in your answer).
- "Trial ID" in your JSON response MUST be the EXACT identifier given to
  you above for that trial, character-for-character (e.g. "2023-501234-56-00"
  or "2014-001234-56") — do NOT substitute an NCT number or any other ID,
  even if you find one for the same trial.
""",
    "ctgov": """
ADDITIONAL SOURCE GUIDANCE FOR THIS BATCH (CT.gov trials):
- "Trial ID" in your JSON response MUST be the EXACT NCT identifier given
  to you above for that trial, character-for-character — do not alter it.
""",
}

# Appended on the retry pass (single trial per call) for trials that came
# back with no efficacy data on the first pass. Explicitly pushes the
# model past the registry front page toward results pages and publications,
# and gives it permission to say "genuinely not published" rather than
# leaving fields blank without trying.
DEEP_DIVE_SUFFIX = """
DEEP-DIVE MODE — this is a single-trial retry because the first pass
returned no efficacy data for this trial. Before answering:
1. Open the trial's own results page (e.g. clinicaltrials.gov/study/<ID>
   -> "Study Results" tab; or the EU registry's results page) — not just
   the summary/protocol page.
2. Search for the trial's Program Name / NCT ID alongside terms like
   "results", "topline", "primary endpoint", "HbA1c", "weight loss",
   "efficacy" — company press releases and conference abstracts often
   report efficacy numbers before the formal registry results page is
   updated.
3. Check for a linked publication (PubMed, NEJM, Lancet, Diabetes Care,
   etc.) — trial result papers usually report exact efficacy numbers even
   when the registry page only shows "N/A".
Only use "N/A" if you've checked the above and the trial genuinely has no
completed results yet (e.g. still recruiting, or terminated with no
posted data) — don't default to "N/A" just because the summary page alone
didn't show it.
"""


async def _fetch_source(name: str, module: Any, drug: str,
                         max_records: Optional[int]) -> List[Dict[str, Any]]:
    """Run one registry module's fetch() in a thread (they're synchronous /
    blocking HTTP scrapers) so multiple sources can be fetched concurrently."""
    label = {"ctgov": "ClinicalTrials.gov API", "euct": "EU (CTIS + EudraCT)"}.get(name, name)
    print(f"Fetching trials for '{drug}' from {label} ...", file=sys.stderr)
    try:
        rows = await asyncio.to_thread(module.fetch, drug, max_records=max_records,
                                        details=True)
    except Exception as exc:
        print(f"  [warn] {name} fetch failed: {exc}", file=sys.stderr)
        return []
    print(f"  Got {len(rows)} trial(s) from {name}", file=sys.stderr)
    for row in rows:
        row.setdefault("registry_source", name)
    return rows


async def _run_batches(drug: str, batches: List[tuple], extra_suffix: str = "",
                        label: str = "pass") -> List[Dict[str, Any]]:
    """Run a list of (source, [stub, ...]) batches through get_trial_details
    concurrently (bounded by MAX_WORKERS) and return the flattened trial list."""
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    # Applied to every batch regardless of source when the molecule name
    # looks like a fixed-dose combination — see COMBINATION_OVERRIDE above.
    combo_suffix = COMBINATION_OVERRIDE if _is_combination_molecule(drug) else ""

    async def run_batch(idx: int, source: str, batch: List[Dict[str, str]]):
        async with semaphore:
            stagger = (idx % MAX_WORKERS) * (RATE_LIMIT_DELAY / MAX_WORKERS)
            if stagger:
                await asyncio.sleep(stagger)
            print(f"  [{label}] Batch {idx + 1}/{len(batches)} [{source}] "
                  f"({len(batch)} trials) - starting", file=sys.stderr)
            extra_prompt = (EXTRA_PROMPT_BY_SOURCE.get(source, "")
                             + combo_suffix + extra_suffix) or None
            data = await get_trial_details(drug, batch, extra_fields_prompt=extra_prompt)
            trials = data.get("trials", [])
            print(f"  [{label}] Batch {idx + 1}/{len(batches)} [{source}] - done "
                  f"({len(trials)} trials)", file=sys.stderr)
            return trials

    results = await asyncio.gather(*[
        run_batch(i, source, batch) for i, (source, batch) in enumerate(batches)
    ])
    return [t for batch in results for t in batch if isinstance(t, dict)]


def _merge_enriched(enriched_trials: List[Dict[str, Any]],
                     rows_by_id: Dict[str, Dict[str, Any]]) -> set:
    """Merge Gemini's enriched fields back onto the original registry rows,
    matched by trial ID. Only overwrite when Gemini actually returned
    something usable — never blank out data the registry scraper already
    had. Returns the set of trial IDs that got a match."""
    matched_ids = set()
    for trial in enriched_trials:
        trial_id = _clean_trial_id(trial.get("Trial ID", ""))
        row = rows_by_id.get(trial_id)
        if row is None:
            continue
        for gemini_field, row_field in GEMINI_FIELD_MAP.items():
            value = trial.get(gemini_field)
            # Compare the STRING form, not the raw value — a numeric 0
            # (e.g. Gemini returning a bare 0 instead of "0") is falsy in
            # Python and would otherwise get silently skipped by `if value`,
            # even though "0" can be a legitimate (if rare) result.
            value_str = str(value).strip() if value is not None else ""
            if value_str and value_str not in ("N/A", "n/a", "None"):
                row[row_field] = value
        matched_ids.add(trial_id)
    return matched_ids


def _chunk_by_source(rows: List[Dict[str, Any]], batch_size: int) -> List[tuple]:
    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        rows_by_source.setdefault(r.get("registry_source", "ctgov"), []).append(r)
    batches: List[tuple] = []
    for source, source_rows in rows_by_source.items():
        stubs = [_to_gemini_trial_stub(r) for r in source_rows]
        for i in range(0, len(stubs), batch_size):
            batches.append((source, stubs[i:i + batch_size]))
    return batches


async def enrich(drug: str, max_records: Optional[int] = None,
                  sources: Optional[List[str]] = None,
                  retry_missing: bool = True) -> List[Dict[str, Any]]:
    sources = sources or list(SOURCE_MODULES.keys())
    active = [s for s in sources if s in SOURCE_MODULES]
    missing = [s for s in sources if s not in SOURCE_MODULES]
    if missing:
        print(f"  [warn] requested source(s) not available, skipping: {missing}",
              file=sys.stderr)
    if not active:
        print("  No registry sources available to fetch from.", file=sys.stderr)
        return []

    fetched = await asyncio.gather(*[
        _fetch_source(name, SOURCE_MODULES[name], drug, max_records)
        for name in active
    ])
    rows = [row for batch in fetched for row in batch]
    print(f"  Got {len(rows)} trial(s) total across source(s): {active}", file=sys.stderr)

    if not rows:
        return []

    if _is_combination_molecule(drug):
        print(f"  [info] '{drug}' looks like a fixed-dose combination — appending "
              f"COMBINATION_OVERRIDE to every Gemini batch so the built-in monotherapy "
              f"filter doesn't exclude/blank these trials.", file=sys.stderr)

    # trial_id namespaces don't overlap across sources (NCT######## vs.
    # CTIS CT numbers vs. EudraCT numbers), so a single dict keyed by
    # trial_id is safe to merge across sources.
    rows_by_id = {r["trial_id"]: r for r in rows if r.get("trial_id")}
    ctgov_ids = {tid for tid in rows_by_id if _is_ctgov_id(tid)}

    # --- Pass 1: normal batching, smaller batch size for research depth ---
    batches = _chunk_by_source(list(rows_by_id.values()), FIRST_PASS_BATCH_SIZE)
    print(f"  Sending {len(rows_by_id)} trial(s) to Gemini in {len(batches)} "
          f"batch(es) of up to {FIRST_PASS_BATCH_SIZE} ...", file=sys.stderr)
    enriched_trials = await _run_batches(drug, batches, label="pass1")
    matched_ids = _merge_enriched(enriched_trials, rows_by_id)

    print(f"  Pass 1: merged Gemini data into {len(matched_ids)}/{len(rows_by_id)} trial(s)",
          file=sys.stderr)

    unmatched = [tid for tid in rows_by_id if tid not in matched_ids]
    if unmatched:
        by_src = {}
        for tid in unmatched:
            src = rows_by_id[tid].get("registry_source", "?")
            by_src.setdefault(src, []).append(tid)
        for src, ids in by_src.items():
            sample = ", ".join(ids[:5])
            more = f" (+{len(ids) - 5} more)" if len(ids) > 5 else ""
            print(f"  [warn] {len(ids)} '{src}' trial(s) got no Gemini match at all: "
                  f"{sample}{more}", file=sys.stderr)

    # --- Pass 2: single-trial retry for rows still missing efficacy data ---
    # Covers both "matched but every efficacy field was N/A" and "no match
    # at all" — either way, give it one more focused shot before giving up.
    if retry_missing:
        retry_rows = [r for r in rows_by_id.values() if _efficacy_missing(r)]
        if retry_rows:
            print(f"  Pass 2: retrying {len(retry_rows)} trial(s) with no efficacy data, "
                  f"1 trial/call, deep-dive prompt ...", file=sys.stderr)
            retry_batches = _chunk_by_source(retry_rows, RETRY_BATCH_SIZE)
            retry_enriched = await _run_batches(drug, retry_batches,
                                                 extra_suffix=DEEP_DIVE_SUFFIX, label="pass2")
            retry_matched = _merge_enriched(retry_enriched, rows_by_id)
            still_missing = sum(1 for r in retry_rows if _efficacy_missing(r))
            print(f"  Pass 2: recovered efficacy data for "
                  f"{len(retry_rows) - still_missing}/{len(retry_rows)} previously-empty trial(s)",
                  file=sys.stderr)
        else:
            print("  Pass 2: skipped — no trials missing efficacy data.", file=sys.stderr)

    # Cross-check Start/Completion Date against the authoritative CT.gov API
    # (usually a no-op since ctgov_trials.py already used that API, but kept
    # for parity in case Gemini overwrote them with guesses). This ONLY
    # applies to CT.gov-native rows (NCT IDs) — the CT.gov API has no
    # knowledge of EU CTIS/EudraCT trial numbers, so EU rows keep whatever
    # dates their own registry scraper / Gemini already filled in.
    if ctgov_ids:
        dates_map = await fetch_ctgov_dates(list(ctgov_ids))
        for nct_id, dates in dates_map.items():
            row = rows_by_id.get(nct_id)
            if row:
                row["trial_start_date"] = dates["Start Date"]
                row["trial_completion_date"] = dates["Completion Date"]

    return list(rows_by_id.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drug", help="drug / intervention name, e.g. 'semaglutide'")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--max-records", type=int, default=None,
                     help="cap the number of trials fetched/enriched PER SOURCE")
    ap.add_argument("--sources", default="ctgov,euct",
                     help="comma-separated registry sources to pull from: "
                          "'ctgov' (ClinicalTrials.gov), 'euct' (EU CTIS + EudraCT), "
                          f"or both (default). Available on this machine: "
                          f"{', '.join(SOURCE_MODULES) or 'none'}")
    ap.add_argument("--no-retry", action="store_true",
                     help="skip the automatic single-trial retry pass for trials "
                          "that came back with no efficacy data (faster, fewer API calls, "
                          "lower fill rate)")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows = asyncio.run(enrich(args.drug, args.max_records, sources,
                               retry_missing=not args.no_retry))

    if not rows:
        print("No trials found / enriched.", file=sys.stderr)
        return 1

    still_empty = sum(1 for r in rows if _efficacy_missing(r))
    print(f"  Final: {len(rows) - still_empty}/{len(rows)} trial(s) have at least "
          f"some efficacy data; {still_empty} have none (likely no results published yet).",
          file=sys.stderr)

    final_rows = [remap_row(row, args.drug) for row in rows]

    os.makedirs(args.outdir, exist_ok=True)
    drug_slug = args.drug.lower().replace(" ", "_")
    src_tag = "_".join(sources)
    out_path = os.path.join(args.outdir, f"{drug_slug}_{src_tag}_gemini_enriched.xlsx")

    if write_excel is not None:
        write_excel(final_rows, out_path, FINAL_COLUMNS,
                    sheet_name="Registries + Gemini")
    else:
        import pandas as pd
        pd.DataFrame(final_rows, columns=FINAL_COLUMNS).to_excel(out_path, index=False)

    print(f"\nWrote {len(final_rows)} enriched trial(s) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())