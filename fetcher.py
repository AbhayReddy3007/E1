#!/usr/bin/env python3
"""
fetcher.py – Post-process enrichment for main1.py Excel output.

Reads the Excel file produced by main1.py, then uses Gemini (with Google Search)
to fill in the following columns for every trial row:

    dosage
    hba1c_change_pct  hba1c_duration  hba1c_rationale  hba1c_confidence
    weight_change_pct weight_duration weight_rationale  weight_confidence
    alt_reduction_pct alt_duration    alt_rationale     alt_confidence
    mash_change_pct   mash_duration   mash_rationale    mash_confidence

Each *_rationale cell gets a 1-2 sentence explanation of where the value
came from and why that score was chosen.

Usage:
    python fetcher.py <molecule_name> [--excel path.xlsx] [--workers N] [--out enriched.xlsx]

    # Typical: molecule name only – auto-discovers <molecule>_trials.xlsx
    python fetcher.py Cagrisema

    # Explicit paths / parallelism
    python fetcher.py Cagrisema --excel cagrisema_trials.xlsx --workers 8 --out cagrisema_enriched.xlsx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

# ── third-party ────────────────────────────────────────────────────────────────
try:
    import openpyxl
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    sys.exit("ERROR: openpyxl not installed.  Run: pip install openpyxl")

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("ERROR: google-genai not installed.  Run: pip install google-genai")

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

# ── constants ──────────────────────────────────────────────────────────────────
MODEL          = "gemini-2.5-flash"
MAX_RETRIES    = 5
INITIAL_BACKOFF = 2.0
BATCH_SIZE     = 6          # trials per Gemini call
DEFAULT_WORKERS = 6         # max concurrent Gemini calls

# Columns that fetcher is responsible for filling
OUTCOME_COLS = [
    "dosage",
    "hba1c_change_pct", "hba1c_duration", "hba1c_rationale", "hba1c_confidence",
    "weight_change_pct", "weight_duration", "weight_rationale", "weight_confidence",
    "alt_reduction_pct", "alt_duration", "alt_rationale", "alt_confidence",
    "mash_change_pct", "mash_duration", "mash_rationale", "mash_confidence",
]

# All columns from main1.py (must match exactly)
ALL_COLUMNS = [
    "molecule_name", "registry_source", "trial_id", "acronym",
    "dosage", "phase", "trial_title", "trial_study", "trial_size",
    "trial_location", "trial_start_date", "trial_completion_date", "phase_status",
    "hba1c_change_pct", "hba1c_duration", "hba1c_rationale", "hba1c_confidence",
    "weight_change_pct", "weight_duration", "weight_rationale", "weight_confidence",
    "alt_reduction_pct", "alt_duration", "alt_rationale", "alt_confidence",
    "mash_change_pct", "mash_duration", "mash_rationale", "mash_confidence",
    "company_name", "source_url",
]

# ── Gemini client ──────────────────────────────────────────────────────────────
def _make_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: Set GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable.\n"
            "  export GOOGLE_API_KEY=your_key_here"
        )
    return genai.Client(api_key=api_key)

_client: Optional[genai.Client] = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _safe_parse(text: str) -> Any:
    """Parse JSON from Gemini response, with repair fallback."""
    if not text:
        return None
    text = text.strip()

    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    # Find start of JSON
    for i, ch in enumerate(text):
        if ch in "{[":
            text = text[i:]
            break

    if _HAS_JSON_REPAIR:
        try:
            return repair_json(text, return_objects=True)
        except Exception:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


# ── Gemini call with search + retry ───────────────────────────────────────────

def _sync_call(prompt: str) -> str:
    """Blocking Gemini call with Google Search enabled."""
    client = get_client()
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]
    config = types.GenerateContentConfig(
        tools=[types.Tool(googleSearch=types.GoogleSearch())],
    )
    out = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL, contents=contents, config=config
    ):
        if chunk.text:
            out += chunk.text
    return out.strip()


async def _gemini_call(prompt: str) -> str:
    """Async Gemini call with exponential backoff on rate-limit errors."""
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(_sync_call, prompt)
        except Exception as exc:
            err = str(exc).lower()
            if any(k in err for k in ("429", "rate limit", "quota", "resource exhausted")):
                if attempt == MAX_RETRIES:
                    print(f"  ✗ Max retries exceeded: {exc}", file=sys.stderr)
                    raise
                print(
                    f"  ⚠ Rate-limit hit – waiting {backoff:.0f}s "
                    f"(attempt {attempt+1}/{MAX_RETRIES})…",
                    file=sys.stderr,
                )
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                print(f"  ✗ Gemini error: {exc}", file=sys.stderr)
                raise
    return ""


# ── per-trial enrichment prompt ────────────────────────────────────────────────

def _build_prompt(molecule: str, batch: List[Dict[str, str]]) -> str:
    trial_lines = "\n".join(
        f"  - {t.get('trial_id','?')} | {t.get('trial_title','')[:80]} "
        f"| Phase {t.get('phase','?')} | {t.get('company_name','')} "
        f"| URL: {t.get('source_url','')}"
        for t in batch
    )

    return f"""You are a clinical data extraction engine with access to Google Search and live trial registries.

MOLECULE: {molecule}

TRIALS TO ENRICH ({len(batch)} total):
{trial_lines}

For EACH trial listed above, search ClinicalTrials.gov, PubMed, and published results to extract:

1. dosage          — Primary or highest dose tested (e.g. "2.4 mg OW", "15 mg QD").
                     If multiple doses, pick the highest. Format: "[amount] [unit] [frequency]"

2. hba1c_change_pct  — HbA1c reduction in percentage points (positive number, e.g. "1.8").
                        "N/A" if this is not a diabetes trial or data unavailable.
   hba1c_duration     — Timepoint of measurement (e.g. "26 wk"). "N/A" if unavailable.
   hba1c_rationale    — 1-2 sentences: state the exact source (paper / registry / result date)
                        and why this value was chosen (e.g. primary endpoint, highest-dose arm).
   hba1c_confidence   — "High" / "Medium" / "Low" reflecting data reliability.

3. weight_change_pct — Body weight loss percentage (positive number, e.g. "15.2").
                        "N/A" if unavailable.
   weight_duration    — Timepoint (e.g. "68 wk"). "N/A" if unavailable.
   weight_rationale   — 1-2 sentences citing source and reason for value chosen.
   weight_confidence  — "High" / "Medium" / "Low".

4. alt_reduction_pct — ALT enzyme reduction percentage (positive number). "N/A" if unavailable.
   alt_duration       — Timepoint (e.g. "24 wk"). "N/A" if unavailable.
   alt_rationale      — 1-2 sentences citing source and reason.
   alt_confidence     — "High" / "Medium" / "Low".

5. mash_change_pct   — MASH/NASH resolution rate or fibrosis improvement % (positive number).
                        "N/A" if not a liver trial.
   mash_duration      — Timepoint (e.g. "72 wk"). "N/A" if unavailable.
   mash_rationale     — 1-2 sentences citing source and reason.
   mash_confidence    — "High" / "Medium" / "Low".

RULES:
- Use actual published results where available; fall back to registry data.
- Report reductions as POSITIVE numbers.
- Use "N/A" for fields with genuinely no data.
- Each rationale MUST name the specific source (e.g. "NEJM 2023 STEP 1 paper", "ClinicalTrials.gov NCT03548935 results section").
- One JSON object per trial keyed by trial_id.

Return ONLY valid JSON, no markdown, no preamble:

{{
  "results": {{
    "<trial_id_1>": {{
      "dosage": "...",
      "hba1c_change_pct": "...",
      "hba1c_duration": "...",
      "hba1c_rationale": "...",
      "hba1c_confidence": "...",
      "weight_change_pct": "...",
      "weight_duration": "...",
      "weight_rationale": "...",
      "weight_confidence": "...",
      "alt_reduction_pct": "...",
      "alt_duration": "...",
      "alt_rationale": "...",
      "alt_confidence": "...",
      "mash_change_pct": "...",
      "mash_duration": "...",
      "mash_rationale": "...",
      "mash_confidence": "..."
    }},
    "<trial_id_2>": {{ ... }}
  }}
}}
"""


# ── batch enrichment ───────────────────────────────────────────────────────────

async def _enrich_batch(
    molecule: str,
    batch: List[Dict[str, str]],
    batch_idx: int,
    total_batches: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Dict[str, str]]:
    """Enrich one batch of trials; returns {trial_id: {field: value}}."""
    async with semaphore:
        ids = [t.get("trial_id", "?") for t in batch]
        print(
            f"  Batch {batch_idx+1}/{total_batches} → {ids}",
            file=sys.stderr,
        )
        prompt = _build_prompt(molecule, batch)
        try:
            raw = await _gemini_call(prompt)
        except Exception:
            return {}

        data = _safe_parse(raw)
        if not data:
            print(f"  ✗ Batch {batch_idx+1}: could not parse response", file=sys.stderr)
            return {}

        # Gemini should return {"results": {...}} but handle bare dict too
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        elif isinstance(data, dict):
            results = data
        else:
            print(f"  ✗ Batch {batch_idx+1}: unexpected JSON structure", file=sys.stderr)
            return {}

        print(
            f"  ✓ Batch {batch_idx+1}: enriched {len(results)} trial(s)",
            file=sys.stderr,
        )
        return results


async def enrich_all(
    molecule: str,
    rows: List[Dict[str, str]],
    max_workers: int = DEFAULT_WORKERS,
) -> List[Dict[str, str]]:
    """Enrich all rows in parallel batches."""
    # Split into batches
    batches = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    total = len(batches)
    print(
        f"\n🔬 Enriching {len(rows)} trial(s) across {total} batch(es) "
        f"(max {max_workers} concurrent)…\n",
        file=sys.stderr,
    )

    semaphore = asyncio.Semaphore(max_workers)

    tasks = [
        _enrich_batch(molecule, batch, idx, total, semaphore)
        for idx, batch in enumerate(batches)
    ]

    # stagger starts slightly to avoid burst rate-limiting
    async def _staggered(coro, delay: float):
        await asyncio.sleep(delay)
        return await coro

    staggered_tasks = [
        _staggered(task, i * 0.4)
        for i, task in enumerate(tasks)
    ]

    batch_results: List[Dict[str, Dict[str, str]]] = await asyncio.gather(
        *staggered_tasks, return_exceptions=False
    )

    # Merge all results keyed by trial_id
    merged: Dict[str, Dict[str, str]] = {}
    for br in batch_results:
        if isinstance(br, dict):
            merged.update(br)

    # Apply back to rows
    updated = 0
    for row in rows:
        tid = row.get("trial_id", "")
        enrichment = merged.get(tid, {})
        if not enrichment:
            # try case-insensitive / stripped match
            for k, v in merged.items():
                if k.strip().upper() == tid.strip().upper():
                    enrichment = v
                    break
        if enrichment:
            for col in OUTCOME_COLS:
                val = enrichment.get(col, "")
                if val and str(val).strip().lower() not in ("n/a", "null", "none", ""):
                    row[col] = str(val).strip()
                elif col not in row or not row[col]:
                    row[col] = "N/A"
            updated += 1
        else:
            # Ensure columns exist with N/A
            for col in OUTCOME_COLS:
                if col not in row or not row[col]:
                    row[col] = "N/A"

    print(f"\n✅ Enrichment done: {updated}/{len(rows)} trial(s) updated.\n", file=sys.stderr)
    return rows


# ── Excel I/O ──────────────────────────────────────────────────────────────────

def _read_excel(path: str) -> List[Dict[str, str]]:
    """Load rows from main1.py Excel output into a list of dicts."""
    wb = load_workbook(path)
    ws = wb.active

    headers = [str(c.value or "").strip() for c in ws[1]]
    rows: List[Dict[str, str]] = []

    for row in ws.iter_rows(min_row=2, values_only=True):
        record: Dict[str, str] = {}
        for h, v in zip(headers, row):
            record[h] = str(v) if v is not None else ""
        # Skip fully empty rows
        if any(record.values()):
            rows.append(record)

    print(f"📂 Loaded {len(rows)} row(s) from {path}", file=sys.stderr)
    return rows


def _write_excel(rows: List[Dict[str, str]], path: str) -> None:
    """Write enriched rows to a new Excel workbook matching main1.py styling."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Clinical Trials (Enriched)"

    HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
    HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    DATA_FONT   = Font(name="Arial", size=9)
    WRAP_ALIGN  = Alignment(wrap_text=True, vertical="top")
    THIN        = Side(style="thin", color="D0D0D0")
    BORDER      = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    ALT_FILL    = PatternFill("solid", fgColor="EBF5FB")
    ENRICH_FILL = PatternFill("solid", fgColor="E8F8E8")  # light green for enriched cols

    # Determine columns: use ALL_COLUMNS order, add any extras from data
    extra_cols = [c for c in (rows[0].keys() if rows else []) if c not in ALL_COLUMNS]
    columns = ALL_COLUMNS + extra_cols

    enriched_col_idxs = {i + 1 for i, c in enumerate(columns) if c in OUTCOME_COLS}

    # Header row
    ws.append(columns)
    for cell in ws[1]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = BORDER

    # Data rows
    for r_idx, row_data in enumerate(rows, start=2):
        values = [row_data.get(col, "") for col in columns]
        ws.append(values)
        alt = r_idx % 2 == 0
        for c_idx, cell in enumerate(ws[r_idx], start=1):
            cell.font      = DATA_FONT
            cell.alignment = WRAP_ALIGN
            cell.border    = BORDER
            if c_idx in enriched_col_idxs:
                cell.fill = ENRICH_FILL
            elif alt:
                cell.fill = ALT_FILL

    # Column widths
    WIDTHS = {
        "molecule_name": 16, "registry_source": 18, "trial_id": 18,
        "acronym": 18, "dosage": 18, "phase": 10, "trial_title": 40,
        "trial_study": 26, "trial_size": 12, "trial_location": 24,
        "trial_start_date": 16, "trial_completion_date": 18, "phase_status": 18,
        "hba1c_change_pct": 16, "hba1c_duration": 14, "hba1c_rationale": 45,
        "hba1c_confidence": 14,
        "weight_change_pct": 16, "weight_duration": 14, "weight_rationale": 45,
        "weight_confidence": 14,
        "alt_reduction_pct": 16, "alt_duration": 14, "alt_rationale": 45,
        "alt_confidence": 14,
        "mash_change_pct": 18, "mash_duration": 14, "mash_rationale": 45,
        "mash_confidence": 14,
        "company_name": 28, "source_url": 40,
    }
    for col_idx, col_name in enumerate(columns, start=1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = WIDTHS.get(col_name, 18)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    wb.save(path)
    print(f"\n💾 Wrote {len(rows)} row(s) → {path}", file=sys.stderr)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _resolve_input_excel(molecule: str, explicit: Optional[str]) -> str:
    """Find the Excel file to process."""
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"ERROR: File not found: {explicit}")
        return explicit

    # Auto-discover: <molecule>_trials.xlsx (same logic as main1.py)
    candidate = f"{molecule.lower().replace(' ', '_')}_trials.xlsx"
    if os.path.exists(candidate):
        return candidate

    # Also check current directory for any *_trials.xlsx
    candidates = [
        f for f in os.listdir(".")
        if f.endswith("_trials.xlsx") or f.endswith("_trials_enriched.xlsx")
    ]
    if len(candidates) == 1:
        print(
            f"  ℹ Auto-discovered input file: {candidates[0]}",
            file=sys.stderr,
        )
        return candidates[0]

    sys.exit(
        f"ERROR: Could not find input Excel file.\n"
        f"  Expected: {candidate}\n"
        f"  Or use:   --excel <path>"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Enrich clinical trial Excel (from main1.py) with efficacy outcomes "
            "and dosage using Gemini + Google Search."
        )
    )
    ap.add_argument("molecule", help="Molecule / drug name (e.g. Cagrisema)")
    ap.add_argument(
        "--excel", default=None,
        help="Input Excel file (default: <molecule>_trials.xlsx)"
    )
    ap.add_argument(
        "--out", default=None,
        help="Output Excel file (default: <molecule>_trials_enriched.xlsx)"
    )
    ap.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Max concurrent Gemini calls (default: {DEFAULT_WORKERS})"
    )
    ap.add_argument(
        "--run-main1", action="store_true",
        help="Run main1.py first to generate the input Excel, then enrich"
    )
    ap.add_argument(
        "--max-records", type=int, default=None,
        help="Passed to main1.py if --run-main1 is set"
    )
    ap.add_argument(
        "--top-n", type=int, default=None,
        help="Passed to main1.py if --run-main1 is set"
    )
    args = ap.parse_args()

    molecule = args.molecule.strip()
    slug     = molecule.lower().replace(" ", "_")
    out_path = args.out or f"{slug}_trials_enriched.xlsx"

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  FETCHER  –  {molecule}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # ── optionally run main1.py first ─────────────────────────────────────────
    if args.run_main1:
        import subprocess
        cmd = [sys.executable, "main1.py", molecule, "--no-enrich"]
        if args.max_records:
            cmd += ["--max-records", str(args.max_records)]
        if args.top_n:
            cmd += ["--top-n", str(args.top_n)]
        print(f"▶ Running: {' '.join(cmd)}\n", file=sys.stderr)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit("ERROR: main1.py failed.")

    # ── load the Excel produced by main1.py ───────────────────────────────────
    excel_path = _resolve_input_excel(molecule, args.excel)
    rows = _read_excel(excel_path)

    if not rows:
        print("No rows found in input Excel. Nothing to enrich.", file=sys.stderr)
        return 1

    # ── run async enrichment ──────────────────────────────────────────────────
    t0 = time.time()
    enriched_rows = asyncio.run(enrich_all(molecule, rows, max_workers=args.workers))
    elapsed = time.time() - t0

    print(f"⏱  Total enrichment time: {elapsed:.1f}s", file=sys.stderr)

    # ── write output ──────────────────────────────────────────────────────────
    _write_excel(enriched_rows, out_path)

    print(
        f"\n✅ Done!  Enriched file: {out_path}\n"
        f"   Rows: {len(enriched_rows)}  |  "
        f"Time: {elapsed:.1f}s\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())