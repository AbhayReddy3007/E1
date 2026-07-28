#!/usr/bin/env python3
"""
enrich_excel.py – Take an existing trials Excel and fill all columns
using Gemini + Google Search (source URL only), then add efficacy_score.

Usage:
    python enrich_excel.py cagrilintide_semaglutide_trials/cagrilintide_semaglutide_ALL_REGISTRIES.xlsx
    python enrich_excel.py my_trials.xlsx --drug "Semaglutide"
    python enrich_excel.py my_trials.xlsx --drug "Semaglutide" --out enriched.xlsx --workers 4
"""

from __future__ import annotations
import argparse
import os
import re
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemini_fill_columns
import efficacy_scorer
from registry_common import write_excel

FINAL_COLUMNS = [
    "source",
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
    "efficacy_score",
]


def normalise_phase(raw: str) -> str:
    if not raw or str(raw).strip().lower() in ("", "n/a", "nan", "none"):
        return ""
    s = re.sub(r"(?i)\bphase\s*", "phase ", str(raw))
    MAP = {"one": "1", "two": "2", "three": "3", "four": "4",
           "i": "1", "ii": "2", "iii": "3", "iv": "4",
           "1": "1", "2": "2", "3": "3", "4": "4"}
    nums = []
    for tok in re.split(r"[/,;\s]+", s.lower()):
        tok = tok.strip("(). ")
        if tok in MAP and MAP[tok] not in nums:
            nums.append(MAP[tok])
    return "/".join(nums) if nums else str(raw).strip()


def main():
    ap = argparse.ArgumentParser(
        description="Fill missing columns in a trials Excel using Gemini + "
                    "source URL, then add efficacy_score.")
    ap.add_argument("infile", help="Input .xlsx file")
    ap.add_argument("--drug", default=None,
                    help="Drug name (inferred from filename if not given)")
    ap.add_argument("--out", default=None,
                    help="Output .xlsx path (default: <infile>_enriched.xlsx)")
    ap.add_argument("--model", default=gemini_fill_columns.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=3,
                    help="Parallel Gemini workers (default 3)")
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--no-gemini", action="store_true",
                    help="Skip Gemini fill; only normalise + score")
    args = ap.parse_args()

    if not os.path.exists(args.infile):
        print(f"ERROR: file not found: {args.infile}", file=sys.stderr)
        return 1

    # Infer drug name from filename if not given
    drug = args.drug
    if not drug:
        stem = os.path.splitext(os.path.basename(args.infile))[0]
        drug = stem.replace("_", " ").replace("-", " ")
        # Strip trailing _ALL_REGISTRIES etc.
        drug = re.sub(r"\s*(all registries|trials|enriched).*$", "",
                      drug, flags=re.I).strip()
        print(f"  Drug inferred from filename: '{drug}'", file=sys.stderr)

    out_path = args.out or re.sub(r"\.xlsx$", "_enriched.xlsx",
                                  args.infile, flags=re.I)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\nLoading {args.infile} ...", file=sys.stderr)
    df = pd.read_excel(args.infile, dtype=str)
    df = df.fillna("")
    print(f"  {len(df)} rows, {len(df.columns)} columns", file=sys.stderr)

    # Ensure all 22 output columns exist (add missing ones as empty)
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Normalise phase
    df["phase"] = df["phase"].apply(normalise_phase)

    rows = df.to_dict("records")

    # ── Gemini fill ───────────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if args.no_gemini:
        print("\n[--no-gemini] Skipping Gemini fill.", file=sys.stderr)
    elif not api_key:
        print("\n[Gemini fill] No GEMINI_API_KEY found in .env – skipping.",
              file=sys.stderr)
    else:
        gemini_fill_columns.BATCH_SIZE = args.batch_size
        print(f"\n[Gemini fill] Filling {len(rows)} trials from source URLs ...",
              file=sys.stderr)
        rows = gemini_fill_columns.fill_columns(
            drug, rows,
            api_key=api_key,
            model=args.model,
            workers=args.workers,
        )

    # ── Efficacy scoring ──────────────────────────────────────────────────────
    print(f"\n[Efficacy] Scoring {len(rows)} trials ...", file=sys.stderr)
    efficacy_scorer.add_efficacy_column(rows)

    # ── Keep exactly FINAL_COLUMNS, nothing else ──────────────────────────────
    print(f"\nWriting {out_path} ...", file=sys.stderr)
    write_excel(rows, out_path, FINAL_COLUMNS, sheet_name="Trials")

    # ── Summary ───────────────────────────────────────────────────────────────
    NA = "n/a"
    def filled(col):
        return sum(1 for r in rows
                   if str(r.get(col, "") or "").strip().lower() not in
                   ("", NA, "none", "null"))

    print(f"\nColumn fill rate:", file=sys.stderr)
    metric_cols = ["dosage", "hba1c_change_pct", "weight_change_pct",
                   "alt_reduction_pct", "mash_resolution_pct",
                   "trial_title", "company_name"]
    for col in metric_cols:
        n = filled(col)
        print(f"  {col:25s} {n:4d}/{len(rows)}", file=sys.stderr)

    scored = sum(1 for r in rows if r.get("efficacy_score", 0))
    print(f"\n  efficacy_score            {scored:4d}/{len(rows)} trials scored",
          file=sys.stderr)
    print(f"\nDone → {os.path.abspath(out_path)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())