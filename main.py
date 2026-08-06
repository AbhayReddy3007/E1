#!/usr/bin/env python3
"""
main.py – Full pipeline orchestrator.

Steps:
  1. Run main1.py  to fetch raw trials from all registries → <molecule>_trials.xlsx
  2. Run fetcher.py enrichment + scoring (no Excel written)
  3. Push enriched rows + score to BigQuery via push_to_bq.py
  4. Generate PDF efficacy report from BigQuery via generate_efficacy_report.py

Usage:
    python main.py <molecule_name> [options]

Examples:
    python main.py Semaglutide
    python main.py Cagrisema --max-records 50 --top-n 20 --workers 8
    python main.py Tirzepatide --no-score
    python main.py Orforglipron --skip-fetch --excel orforglipron_trials.xlsx
    python main.py Semaglutide --no-report
    python main.py Semaglutide --report-outdir ./reports
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Optional

# -- load .env -----------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==============================================================================
# HELPERS
# ==============================================================================

def _run_main1(
    molecule: str,
    max_records: Optional[int],
    top_n: Optional[int],
    out_xlsx: str,
) -> None:
    """Run main1.py to fetch raw trials. Exits on failure."""
    cmd = [sys.executable, "main1.py", molecule, "--no-enrich", "--out", out_xlsx]
    if max_records:
        cmd += ["--max-records", str(max_records)]
    if top_n:
        cmd += ["--top-n", str(top_n)]

    print(f"\n[MAIN] Step 1 – Fetching trials via main1.py ...", file=sys.stderr)
    print(f"  > {' '.join(cmd)}\n", file=sys.stderr)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("ERROR: main1.py failed. Aborting pipeline.")
    if not os.path.exists(out_xlsx):
        sys.exit(f"ERROR: main1.py ran but {out_xlsx} was not created.")
    print(f"[MAIN] Trials written to {out_xlsx}", file=sys.stderr)


# ==============================================================================
# PIPELINE
# ==============================================================================

def run_pipeline(
    molecule: str,
    max_records: Optional[int] = None,
    top_n: Optional[int] = None,
    workers: int = 6,
    no_score: bool = False,
    skip_fetch: bool = False,
    excel_path: Optional[str] = None,
    no_report: bool = False,
    report_outdir: Optional[str] = None,
) -> int:
    """
    Execute the full pipeline for one molecule.

    Returns 0 on success, non-zero on failure.
    """
    slug     = molecule.lower().replace(" ", "_")
    raw_xlsx = excel_path or f"{slug}_trials.xlsx"

    t_start = time.time()

    print(f"\n{'='*64}", file=sys.stderr)
    print(f"  PIPELINE  –  {molecule}", file=sys.stderr)
    print(f"{'='*64}", file=sys.stderr)

    # ── Step 1: Fetch raw trials (main1.py) ───────────────────────────────
    if skip_fetch:
        if not os.path.exists(raw_xlsx):
            print(f"ERROR: --skip-fetch set but {raw_xlsx} not found.", file=sys.stderr)
            return 1
        print(f"\n[MAIN] Step 1 – Skipped (using {raw_xlsx})", file=sys.stderr)
    else:
        _run_main1(molecule, max_records, top_n, raw_xlsx)

    # ── Step 2: Enrich + score (fetcher.py) ───────────────────────────────
    print(f"\n[MAIN] Step 2 – Enrichment + scoring (fetcher.py) ...", file=sys.stderr)
    try:
        import fetcher
    except ImportError:
        print("ERROR: fetcher.py not found in the same directory.", file=sys.stderr)
        return 1

    enriched_rows, score_result, score_rationale = fetcher.run_fetcher(
        molecule    = molecule,
        excel_path  = raw_xlsx,
        max_workers = workers,
        no_score    = no_score,
    )

    if not enriched_rows:
        print("[MAIN] No enriched rows returned. Aborting.", file=sys.stderr)
        return 1

    print(
        f"\n[MAIN] Enrichment complete: {len(enriched_rows)} trial(s)",
        file=sys.stderr,
    )
    if score_result:
        print(
            f"  Efficacy score : {score_result['weighted_score']} / 5.0\n"
            f"  Coverage       : {score_result['data_coverage']}",
            file=sys.stderr,
        )

    # ── Step 3: Push to BigQuery (push_to_bq.py) ──────────────────────────
    print(f"\n[MAIN] Step 3 – Pushing to BigQuery ...", file=sys.stderr)
    try:
        from push_to_bq import save_clinical_efficacy_to_bq
    except ImportError:
        print("ERROR: push_to_bq.py not found in the same directory.", file=sys.stderr)
        return 1

    save_clinical_efficacy_to_bq(
        molecule_name = molecule,
        trials        = enriched_rows,
        score_result  = score_result,
        rationale     = score_rationale,
    )

    # ── Step 4: Generate PDF report (generate_efficacy_report.py) ─────────
    if no_report:
        print(f"\n[MAIN] Step 4 – Skipped (--no-report)", file=sys.stderr)
    else:
        print(f"\n[MAIN] Step 4 – Generating efficacy report ...", file=sys.stderr)
        try:
            from generate_efficacy_report import generate_efficacy_report
        except ImportError:
            print(
                "  [WARN] generate_efficacy_report.py not found – skipping report.",
                file=sys.stderr,
            )
        else:
            try:
                report_paths = generate_efficacy_report(
                    molecules = [molecule],
                    outdir    = report_outdir,
                )
                if report_paths:
                    print(
                        f"[MAIN] Report(s) generated:\n"
                        + "\n".join(f"  {p}" for p in report_paths),
                        file=sys.stderr,
                    )
                else:
                    print(
                        "[MAIN] Report generation returned no output "
                        "(check GEMINI_API_KEY and BQ data).",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"  [WARN] Report generation failed: {exc}", file=sys.stderr)

    elapsed = time.time() - t_start
    print(
        f"\n{'='*64}\n"
        f"  DONE  –  {molecule}  ({elapsed:.1f}s)\n"
        f"{'='*64}\n",
        file=sys.stderr,
    )
    return 0


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Full pipeline: fetch trials (main1.py) → enrich + score (fetcher.py) "
            "→ push to BigQuery (push_to_bq.py) → generate PDF report "
            "(generate_efficacy_report.py)."
        )
    )
    ap.add_argument("molecule",
                    help="Molecule / drug name to process (e.g. Semaglutide)")
    ap.add_argument("--max-records",    type=int, default=None,
                    help="Max records per registry passed to main1.py (default: all)")
    ap.add_argument("--top-n",          type=int, default=None,
                    help="Keep only top N trials by completeness (passed to main1.py)")
    ap.add_argument("--workers",        type=int, default=6,
                    help="Concurrent Gemini enrichment workers (default: 6)")
    ap.add_argument("--no-score",       action="store_true",
                    help="Skip the clinical efficacy scoring step")
    ap.add_argument("--skip-fetch",     action="store_true",
                    help="Skip running main1.py; use an existing <molecule>_trials.xlsx")
    ap.add_argument("--excel",          default=None,
                    help="Explicit path to the raw trials Excel (implies --skip-fetch)")
    ap.add_argument("--no-report",      action="store_true",
                    help="Skip PDF report generation (Step 4)")
    ap.add_argument("--report-outdir",  default=None,
                    help="Directory to save the PDF report (default: current directory)")
    args = ap.parse_args()

    molecule   = args.molecule.strip()
    skip_fetch = args.skip_fetch or bool(args.excel)

    return run_pipeline(
        molecule      = molecule,
        max_records   = args.max_records,
        top_n         = args.top_n,
        workers       = args.workers,
        no_score      = args.no_score,
        skip_fetch    = skip_fetch,
        excel_path    = args.excel,
        no_report     = args.no_report,
        report_outdir = args.report_outdir,
    )


if __name__ == "__main__":
    sys.exit(main())