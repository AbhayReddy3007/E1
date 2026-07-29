#!/usr/bin/env python3
"""
enrich_ctgov_with_gemini.py
============================
Bridges ctgov_trials.py (fast, authoritative structural data straight from
the ClinicalTrials.gov API v2) with gemini_extractor.py's Step 2 detail
extraction (Gemini + Google Search — used for efficacy fields that aren't
in the raw API response: HbA1c change, weight loss %, ALT reduction, MASH
resolution, dosage, company, etc.).

Instead of letting gemini_extractor.py run its own Step 1 registry search,
this script hands Gemini the exact NCT IDs already pulled from
ctgov_trials.py, jumps straight to Step 2 (get_trial_details), and merges
the enriched fields back onto the original rows — so nothing is duplicated
or re-discovered, only filled in.

Usage:
    python enrich_ctgov_with_gemini.py "semaglutide"
    python enrich_ctgov_with_gemini.py "semaglutide" --outdir results --max-records 50

Requires (same as gemini_extractor.py):
    pip install google-genai aiohttp python-dotenv json-repair
    GOOGLE_API_KEY set in the environment or a .env file

Also requires ctgov_trials.py and gemini_extractor.py to be importable
(same directory or on PYTHONPATH). If registry_common.py (used by
ctgov_trials.py) is not available, this script falls back to pandas for
writing the output Excel file.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

import ctgov_trials
from gemini_extractor import (
    BATCH_SIZE,
    MAX_WORKERS,
    RATE_LIMIT_DELAY,
    get_trial_details,
    fetch_ctgov_dates,
)

try:
    from registry_common import write_excel
except ImportError:
    write_excel = None


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


def _to_gemini_trial_stub(row: Dict[str, Any]) -> Dict[str, str]:
    """gemini_extractor.get_trial_details() only needs NCT_ID + Program_Name
    per trial to build its prompt — it looks up everything else itself."""
    return {
        "NCT_ID": row.get("trial_id", ""),
        "Program_Name": row.get("public_title", "") or row.get("title", "") or "N/A",
    }


def _clean_nct_id(raw_id: str) -> str:
    """Gemini sometimes returns 'NCT01234567 (STEP 1)' style values."""
    return raw_id.split("(")[0].strip().split(" ")[0].strip()


async def enrich(drug: str, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    print(f"Fetching trials for '{drug}' from ClinicalTrials.gov API ...", file=sys.stderr)
    rows = ctgov_trials.fetch(drug, max_records=max_records, details=True)
    print(f"  Got {len(rows)} trial(s) from ctgov_trials.py", file=sys.stderr)

    if not rows:
        return []

    rows_by_id = {r["trial_id"]: r for r in rows if r.get("trial_id")}
    trial_stubs = [_to_gemini_trial_stub(r) for r in rows_by_id.values()]

    # Same batching strategy as gemini_extractor.py's Step 2, just skipping
    # its Step 1 (we already have real trial IDs from the ctgov API).
    batches = [
        trial_stubs[i:i + BATCH_SIZE]
        for i in range(0, len(trial_stubs), BATCH_SIZE)
    ]
    print(f"  Sending {len(trial_stubs)} trial(s) to Gemini in {len(batches)} "
          f"batch(es) of up to {BATCH_SIZE} ...", file=sys.stderr)

    semaphore = asyncio.Semaphore(MAX_WORKERS)

    async def run_batch(idx: int, batch: List[Dict[str, str]]):
        async with semaphore:
            stagger = (idx % MAX_WORKERS) * (RATE_LIMIT_DELAY / MAX_WORKERS)
            if stagger:
                await asyncio.sleep(stagger)
            print(f"  Batch {idx + 1}/{len(batches)} ({len(batch)} trials) - starting",
                  file=sys.stderr)
            data = await get_trial_details(drug, batch)
            trials = data.get("trials", [])
            print(f"  Batch {idx + 1}/{len(batches)} - done ({len(trials)} trials)",
                  file=sys.stderr)
            return trials

    results = await asyncio.gather(*[run_batch(i, b) for i, b in enumerate(batches)])
    enriched_trials = [t for batch in results for t in batch if isinstance(t, dict)]

    # Merge Gemini's enriched fields back onto the original ctgov rows,
    # matched by NCT ID. Only overwrite when Gemini actually returned
    # something usable — never blank out data ctgov_trials.py already had.
    merged_count = 0
    for trial in enriched_trials:
        nct_id = _clean_nct_id(trial.get("Trial ID", ""))
        row = rows_by_id.get(nct_id)
        if row is None:
            continue
        for gemini_field, row_field in GEMINI_FIELD_MAP.items():
            value = trial.get(gemini_field)
            if value and str(value).strip() not in ("", "N/A", "n/a"):
                row[row_field] = value
        merged_count += 1

    print(f"  Merged Gemini data into {merged_count}/{len(rows_by_id)} trial(s)",
          file=sys.stderr)

    # Cross-check Start/Completion Date against the authoritative CT.gov API
    # (usually a no-op here since ctgov_trials.py already used that API, but
    # kept for parity in case Gemini overwrote them with guesses).
    dates_map = await fetch_ctgov_dates(list(rows_by_id.keys()))
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
                     help="cap the number of ctgov trials fetched/enriched")
    args = ap.parse_args()

    rows = asyncio.run(enrich(args.drug, args.max_records))

    if not rows:
        print("No trials found / enriched.", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    drug_slug = args.drug.lower().replace(" ", "_")
    out_path = os.path.join(args.outdir, f"{drug_slug}_ctgov_gemini_enriched.xlsx")

    if write_excel is not None:
        columns = list(rows[0].keys())
        write_excel(rows, out_path, columns, sheet_name="ClinicalTrials.gov + Gemini")
    else:
        import pandas as pd
        pd.DataFrame(rows).to_excel(out_path, index=False)

    print(f"\nWrote {len(rows)} enriched trial(s) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())