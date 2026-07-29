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
import re
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


# The only columns that should appear in the final output, in this order.
FINAL_COLUMNS = [
    "molecule_name",
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
    """Collapse a merged ctgov+Gemini row down to exactly FINAL_COLUMNS.

    Falls back to the raw ctgov_trials.py field name (e.g. 'title',
    'countries', 'status') if the Gemini-named field wasn't filled in.
    """
    return {
        "molecule_name": drug,
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

    final_rows = [remap_row(row, args.drug) for row in rows]

    os.makedirs(args.outdir, exist_ok=True)
    drug_slug = args.drug.lower().replace(" ", "_")
    out_path = os.path.join(args.outdir, f"{drug_slug}_ctgov_gemini_enriched.xlsx")

    if write_excel is not None:
        write_excel(final_rows, out_path, FINAL_COLUMNS,
                    sheet_name="ClinicalTrials.gov + Gemini")
    else:
        import pandas as pd
        pd.DataFrame(final_rows, columns=FINAL_COLUMNS).to_excel(out_path, index=False)

    print(f"\nWrote {len(final_rows)} enriched trial(s) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())