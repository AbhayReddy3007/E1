#!/usr/bin/env python3
"""
main1.py – Unified clinical trial fetcher.

Queries ClinicalTrials.gov (CTGOV), EU CTIS, EudraCT, and CTRI for a given
molecule name, then writes a single Excel workbook with one row per trial and
the following standardised columns:

    molecule_name, registry_source, trial_id, acronym, dosage, phase,
    trial_title, trial_study, trial_size, trial_location, trial_start_date,
    trial_completion_date, phase_status,
    hba1c_change_pct, hba1c_duration, hba1c_rationale, hba1c_confidence,
    weight_change_pct, weight_duration, weight_rationale, weight_confidence,
    alt_reduction_pct, alt_duration, alt_rationale, alt_confidence,
    mash_change_pct, mash_duration, mash_rationale, mash_confidence,
    company_name, source_url

Usage:
    python main1.py <molecule_name> [--max-records N] [--out output.xlsx]
    python main1.py semaglutide --max-records 50 --out semaglutide_trials.xlsx
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Optional

import requests

# ── optional import of local registry modules ─────────────────────────────────
try:
    import ctgov_trials as _ctgov
except ImportError:
    _ctgov = None  # type: ignore

try:
    import ctis_drug_trials as _ctis
except ImportError:
    _ctis = None  # type: ignore

try:
    import eudract_drug_trials as _eudract
except ImportError:
    _eudract = None  # type: ignore

try:
    import ctri_trials as _ctri
except ImportError:
    _ctri = None  # type: ignore

try:
    from enrich_outcomes import enrich_trial_outcomes as _enrich
except ImportError:
    _enrich = None  # type: ignore

# ── output columns ─────────────────────────────────────────────────────────────
COLUMNS = [
    "molecule_name",
    "registry_source",
    "trial_id",
    "acronym",
    "dosage",
    "phase",
    "trial_title",
    "trial_study",
    "trial_size",
    "trial_location",
    "trial_start_date",
    "trial_completion_date",
    "phase_status",
    "hba1c_change_pct",
    "hba1c_duration",
    "hba1c_rationale",
    "hba1c_confidence",
    "weight_change_pct",
    "weight_duration",
    "weight_rationale",
    "weight_confidence",
    "alt_reduction_pct",
    "alt_duration",
    "alt_rationale",
    "alt_confidence",
    "mash_change_pct",
    "mash_duration",
    "mash_rationale",
    "mash_confidence",
    "company_name",
    "source_url",
]

# ── acronym fetch via ClinicalTrials.gov API ──────────────────────────────────

def _fetch_acronym_ctgov(trial_id: str, session: requests.Session) -> str:
    """
    Query the ClinicalTrials.gov v2 API for the acronym of a given NCT ID.
    Returns the acronym string or '' if not found.
    """
    if not trial_id or not trial_id.upper().startswith("NCT"):
        return ""
    url = f"https://clinicaltrials.gov/api/v2/studies/{trial_id}"
    try:
        resp = session.get(url, timeout=20)
        if not resp.ok:
            return ""
        data = resp.json()
        # path: protocolSection → identificationModule → acronym
        ps = data.get("protocolSection") or {}
        ident = ps.get("identificationModule") or {}
        return ident.get("acronym", "") or ""
    except Exception:
        return ""


def _fetch_acronym_euctr(trial_id: str, session: requests.Session) -> str:
    """
    Attempt to fetch an acronym for a CTIS trial via the public retrieve endpoint.
    """
    if not trial_id:
        return ""
    url = f"https://euclinicaltrials.eu/ctis-public-api/retrieve/{trial_id}"
    try:
        resp = session.get(url, timeout=20)
        if not resp.ok:
            return ""
        data = resp.json()
        # CTIS stores acronym in authorizedApplication → authorizedPartI → trialInformation
        p1 = (data.get("authorizedApplication") or {}).get("authorizedPartI") or {}
        trial_info = p1.get("trialInformation") or {}
        return trial_info.get("acronym") or trial_info.get("shortTitle") or ""
    except Exception:
        return ""


# ── per-registry mappers ───────────────────────────────────────────────────────

def _blank(molecule: str, source: str) -> Dict[str, str]:
    row: Dict[str, str] = {c: "" for c in COLUMNS}
    row["molecule_name"]   = molecule
    row["registry_source"] = source
    return row


def map_ctgov(raw: Dict[str, Any], molecule: str,
              session: requests.Session) -> Dict[str, str]:
    row = _blank(molecule, "ClinicalTrials.gov")

    trial_id = raw.get("trial_id", "")
    row["trial_id"]            = trial_id
    row["acronym"]             = _fetch_acronym_ctgov(trial_id, session)
    row["phase"]               = raw.get("phase", "")
    row["trial_title"]         = raw.get("title", "") or raw.get("public_title", "")
    row["trial_study"]         = raw.get("study_type", "") + (
                                    f" | {raw.get('study_design','')}"
                                    if raw.get("study_design") else "")
    row["trial_size"]          = raw.get("actual_enrollment") or raw.get("target_enrollment", "")
    row["trial_location"]      = raw.get("countries", "")
    row["trial_start_date"]    = raw.get("start_date", "")
    row["trial_completion_date"] = raw.get("completion_date", "")
    row["phase_status"]        = raw.get("status", "")
    row["company_name"]        = raw.get("sponsor", "")
    row["source_url"]          = raw.get("url", "")
    # dosage intentionally left blank

    return row


def map_ctis(raw: Dict[str, Any], molecule: str,
             session: requests.Session) -> Dict[str, str]:
    row = _blank(molecule, "EU CTIS")

    trial_id = raw.get("ct_number", "") or raw.get("trial_id", "")
    row["trial_id"]              = trial_id
    row["acronym"]               = _fetch_acronym_euctr(trial_id, session)
    row["phase"]                 = raw.get("phase", "")
    row["trial_title"]           = raw.get("title", "") or raw.get("short_title", "")
    row["trial_study"]           = raw.get("trial_design", "")
    row["trial_size"]            = (raw.get("planned_subjects_worldwide", "")
                                    or raw.get("enrolled", ""))
    row["trial_location"]        = raw.get("countries", "")
    row["trial_start_date"]      = raw.get("start_date", "")
    row["trial_completion_date"] = raw.get("end_date", "")
    row["phase_status"]          = raw.get("status", "")
    row["company_name"]          = raw.get("sponsors_full", "") or raw.get("sponsor", "")
    row["source_url"]            = raw.get("url", "")
    # dosage intentionally left blank

    return row


def map_eudract(raw: Dict[str, Any], molecule: str) -> Dict[str, str]:
    row = _blank(molecule, "EudraCT")

    trial_id = raw.get("eudract_number", "")
    row["trial_id"]              = trial_id
    # EudraCT has no dedicated acronym field; use sponsor protocol number as proxy
    row["acronym"]               = raw.get("sponsor_protocol_number", "")
    row["phase"]                 = raw.get("phase", "")
    row["trial_title"]           = (raw.get("full_title", "")
                                    or raw.get("title", "")
                                    or raw.get("lay_title", ""))
    design_parts = [raw.get("randomised", ""), raw.get("double_blind", ""),
                    raw.get("parallel_group", ""), raw.get("crossover", "")]
    row["trial_study"]           = " | ".join(p for p in design_parts if p)
    row["trial_size"]            = (raw.get("results_subjects_worldwide", "")
                                    or raw.get("subjects_worldwide", ""))
    row["trial_location"]        = raw.get("countries", "")
    row["trial_start_date"]      = raw.get("start_date", "")
    row["trial_completion_date"] = raw.get("global_end_date", "")
    row["phase_status"]          = (raw.get("end_of_trial_status", "")
                                    or raw.get("status", ""))
    row["company_name"]          = (raw.get("sponsor_name", "")
                                    or raw.get("sponsor", ""))
    row["source_url"]            = raw.get("url", "") or raw.get("results_url", "")
    # dosage intentionally left blank

    return row


def map_ctri(raw: Dict[str, Any], molecule: str) -> Dict[str, str]:
    row = _blank(molecule, "CTRI (India)")

    row["trial_id"]              = raw.get("trial_id", "")
    row["acronym"]               = ""   # CTRI does not expose an acronym field
    row["phase"]                 = raw.get("phase", "")
    row["trial_title"]           = raw.get("title", "") or raw.get("public_title", "")
    row["trial_study"]           = raw.get("study_type", "") + (
                                      f" | {raw.get('study_design', '')}"
                                      if raw.get("study_design") else "")
    row["trial_size"]            = (raw.get("actual_enrollment", "")
                                    or raw.get("target_enrollment", ""))
    row["trial_location"]        = raw.get("countries", "") or "India"
    row["trial_start_date"]      = raw.get("start_date", "")
    row["trial_completion_date"] = raw.get("completion_date", "")
    row["phase_status"]          = raw.get("status", "")
    row["company_name"]          = raw.get("sponsor", "")
    row["source_url"]            = raw.get("url", "")
    # dosage intentionally left blank

    return row


# ── session factory ────────────────────────────────────────────────────────────

def _make_session(extra_headers: Optional[Dict[str, str]] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/125.0 Safari/537.36"),
    })
    if extra_headers:
        s.headers.update(extra_headers)
    return s


# ── main fetch orchestration ───────────────────────────────────────────────────

def fetch_all(molecule: str, max_records: Optional[int] = None) -> List[Dict[str, str]]:
    unified: List[Dict[str, str]] = []

    # ── 1. ClinicalTrials.gov ─────────────────────────────────────────────────
    if _ctgov is not None:
        print(f"[CTGOV] Searching for '{molecule}' …", file=sys.stderr)
        try:
            session = _make_session({"Accept": "application/json"})
            ctgov_rows = _ctgov.fetch(molecule, max_records=max_records)
            print(f"[CTGOV] {len(ctgov_rows)} trial(s) found.", file=sys.stderr)
            for raw in ctgov_rows:
                try:
                    unified.append(map_ctgov(raw, molecule, session))
                except Exception as exc:
                    print(f"  [CTGOV] map error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"[CTGOV] fetch failed: {exc}", file=sys.stderr)
    else:
        print("[CTGOV] ctgov_trials.py not importable – skipping.", file=sys.stderr)

    # ── 2. EU CTIS ────────────────────────────────────────────────────────────
    if _ctis is not None:
        print(f"[CTIS]  Searching for '{molecule}' …", file=sys.stderr)
        try:
            session = _make_session(_ctis.HEADERS)
            ctis_rows: List[Dict[str, Any]] = []
            for summary in _ctis.search_trials(molecule, session,
                                               page_size=50,
                                               max_records=max_records):
                try:
                    details = _ctis.get_trial_details(
                        summary.get("ctNumber", ""), session)
                except Exception:
                    details = None
                ctis_rows.append(_ctis.flatten(summary, details))
                time.sleep(0.3)

            print(f"[CTIS]  {len(ctis_rows)} trial(s) found.", file=sys.stderr)
            for raw in ctis_rows:
                try:
                    unified.append(map_ctis(raw, molecule, session))
                except Exception as exc:
                    print(f"  [CTIS] map error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"[CTIS]  fetch failed: {exc}", file=sys.stderr)
    else:
        print("[CTIS]  ctis_drug_trials.py not importable – skipping.", file=sys.stderr)

    # ── 3. EudraCT ────────────────────────────────────────────────────────────
    if _eudract is not None:
        print(f"[EUCT]  Searching EudraCT for '{molecule}' …", file=sys.stderr)
        try:
            session = _make_session(_eudract.HEADERS)
            eudract_rows: List[Dict[str, Any]] = []
            for row in _eudract.search_trials(molecule, session,
                                              max_records=max_records):
                country = (row.get("countries", "").split(";")[0] or "GB").strip()
                try:
                    row.update(_eudract.get_trial_details(
                        row["eudract_number"], country, session))
                except Exception:
                    pass
                if row.get("results_available") == "Yes":
                    try:
                        row.update(_eudract.get_trial_results(
                            row["eudract_number"], session))
                    except Exception:
                        pass
                eudract_rows.append(row)
                time.sleep(_eudract.POLITE_DELAY)

            print(f"[EUCT]  {len(eudract_rows)} trial(s) found.", file=sys.stderr)
            for raw in eudract_rows:
                try:
                    unified.append(map_eudract(raw, molecule))
                except Exception as exc:
                    print(f"  [EUCT] map error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"[EUCT]  fetch failed: {exc}", file=sys.stderr)
    else:
        print("[EUCT]  eudract_drug_trials.py not importable – skipping.", file=sys.stderr)

    # ── 4. CTRI (India) ───────────────────────────────────────────────────────
    if _ctri is not None:
        print(f"[CTRI]  Searching CTRI for '{molecule}' …", file=sys.stderr)
        try:
            ctri_rows = _ctri.fetch(molecule, max_records=max_records)
            print(f"[CTRI]  {len(ctri_rows)} trial(s) found.", file=sys.stderr)
            for raw in ctri_rows:
                try:
                    unified.append(map_ctri(raw, molecule))
                except Exception as exc:
                    print(f"  [CTRI] map error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"[CTRI]  fetch failed: {exc}", file=sys.stderr)
    else:
        print("[CTRI]  ctri_trials.py not importable – skipping.", file=sys.stderr)

    return unified


# ── Excel writer ───────────────────────────────────────────────────────────────

def write_excel(rows: List[Dict[str, str]], path: str) -> None:
    """Write the unified rows to a formatted Excel workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import (Font, PatternFill, Alignment,
                                  Border, Side, numbers)
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Clinical Trials"

    # ── header style ──────────────────────────────────────────────────────────
    HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")   # dark blue
    HEADER_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    DATA_FONT    = Font(name="Arial", size=9)
    WRAP_ALIGN   = Alignment(wrap_text=True, vertical="top")
    THIN         = Side(style="thin", color="D0D0D0")
    BORDER       = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

    # alternate row fills
    ALT_FILL = PatternFill("solid", fgColor="EBF5FB")

    # header row
    ws.append(COLUMNS)
    for cell in ws[1]:
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center",
                                   horizontal="center")
        cell.border    = BORDER

    # data rows
    for r_idx, row_data in enumerate(rows, start=2):
        values = [row_data.get(col, "") for col in COLUMNS]
        ws.append(values)
        fill = ALT_FILL if r_idx % 2 == 0 else None
        for cell in ws[r_idx]:
            cell.font      = DATA_FONT
            cell.alignment = WRAP_ALIGN
            cell.border    = BORDER
            if fill:
                cell.fill = fill

    # ── column widths ─────────────────────────────────────────────────────────
    WIDTHS = {
        "molecule_name":         16,
        "registry_source":       18,
        "trial_id":              18,
        "acronym":               18,
        "dosage":                16,
        "phase":                 10,
        "trial_title":           40,
        "trial_study":           26,
        "trial_size":            12,
        "trial_location":        24,
        "trial_start_date":      16,
        "trial_completion_date": 18,
        "phase_status":          18,
        "hba1c_change_pct":      16,
        "hba1c_duration":        16,
        "hba1c_rationale":       35,
        "hba1c_confidence":      16,
        "weight_change_pct":     16,
        "weight_duration":       16,
        "weight_rationale":      35,
        "weight_confidence":     16,
        "alt_reduction_pct":     16,
        "alt_duration":          16,
        "alt_rationale":         35,
        "alt_confidence":        16,
        "mash_change_pct":       18,
        "mash_duration":         16,
        "mash_rationale":        35,
        "mash_confidence":       16,
        "company_name":          28,
        "source_url":            40,
    }
    for col_idx, col_name in enumerate(COLUMNS, start=1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = WIDTHS.get(col_name, 18)

    # freeze top row
    ws.freeze_panes = "A2"

    # auto-filter
    ws.auto_filter.ref = ws.dimensions

    wb.save(path)
    print(f"\nWrote {len(rows)} trial(s) → {path}", file=sys.stderr)


# ── top-N ranking ─────────────────────────────────────────────────────────────

def _rank_and_trim(rows: List[Dict[str, str]], top_n: int) -> List[Dict[str, str]]:
    """
    Score each trial by data completeness and return the top N.

    Scoring favours trials that have:
      - a recognised phase (Phase 3 > Phase 2 > Phase 1 > other)
      - an enrollment number (larger is better, capped)
      - start and completion dates
      - an acronym (published trials almost always have one)
      - a company name
    """

    def _phase_score(phase: str) -> int:
        p = (phase or "").lower()
        if "3" in p or "iii" in p:
            return 40
        if "4" in p or "iv" in p:
            return 35
        if "2" in p or "ii" in p:
            return 30
        if "1" in p or "i" in p:
            return 15
        return 0

    def _enrollment_score(size: str) -> int:
        try:
            n = int(str(size).replace(",", "").strip())
            return min(30, n // 10)           # up to 30 pts
        except (ValueError, TypeError):
            return 0

    def _score(row: Dict[str, str]) -> int:
        s = 0
        s += _phase_score(row.get("phase", ""))
        s += _enrollment_score(row.get("trial_size", ""))
        if row.get("trial_start_date"):
            s += 5
        if row.get("trial_completion_date"):
            s += 5
        if row.get("acronym"):
            s += 10
        if row.get("company_name"):
            s += 5
        if row.get("phase_status", "").lower() in ("completed", "active, not recruiting"):
            s += 10
        return s

    scored = sorted(rows, key=_score, reverse=True)
    return scored[:top_n]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch clinical trials for a molecule from multiple registries "
                    "and output a unified Excel file.")
    ap.add_argument("molecule", help="Molecule / drug name to search for")
    ap.add_argument("--max-records", type=int, default=None,
                    help="Maximum records per registry (default: all)")
    ap.add_argument("--out", default=None,
                    help="Output Excel file path (default: <molecule>_trials.xlsx)")
    ap.add_argument("--workers", type=int, default=6,
                    help="Concurrent trials during outcome enrichment "
                         "(default: 6; lower this if you hit Gemini rate limits)")
    ap.add_argument("--no-enrich", action="store_true",
                    help="Skip the outcome enrichment step entirely")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Only keep the top N trials (by completeness: "
                         "phase, enrollment, dates). Applied before enrichment "
                         "so enrichment runs only on the trials you want.")
    args = ap.parse_args()

    molecule  = args.molecule.strip()
    out_path  = args.out or f"{molecule.lower().replace(' ', '_')}_trials.xlsx"

    print(f"\n=== Fetching clinical trials for: {molecule} ===\n", file=sys.stderr)
    rows = fetch_all(molecule, max_records=args.max_records)

    if not rows:
        print("No trials found across all registries.", file=sys.stderr)
        return 1

    print(f"\nTotal trials collected: {len(rows)}", file=sys.stderr)

    # ── top-N filtering (before enrichment to save API calls) ─────────────
    if args.top_n and args.top_n > 0 and len(rows) > args.top_n:
        rows = _rank_and_trim(rows, args.top_n)
        print(f"Trimmed to top {args.top_n} trial(s) by completeness.",
              file=sys.stderr)

    # ── outcome enrichment (optional, requires GEMINI_API_KEY) ────────────
    if args.no_enrich:
        print("[ENRICH] --no-enrich set – skipping.", file=sys.stderr)
    elif _enrich is not None:
        rows = _enrich(rows, molecule, max_workers=args.workers)
    else:
        print("[ENRICH] enrich_outcomes.py not importable – skipping.",
              file=sys.stderr)

    write_excel(rows, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())