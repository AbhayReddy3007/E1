#!/usr/bin/env python3
"""
fetcher.py – Enrichment + clinical efficacy scoring for main1.py Excel output.

Step 1 – Gemini enrichment (parallel):
  Reads the Excel file produced by main1.py and uses Gemini (with Google Search)
  to fill in per-trial outcome columns:
    dosage
    hba1c_change_pct  hba1c_duration  hba1c_rationale  hba1c_confidence
    weight_change_pct weight_duration weight_rationale  weight_confidence
    alt_reduction_pct alt_duration    alt_rationale     alt_confidence
    mash_change_pct   mash_duration   mash_rationale    mash_confidence

Step 2 – Clinical Efficacy Scoring:
  Scores the molecule across four endpoints using a phase-anchored algorithm.
    Phase 3 -> no penalty  |  Phase 2 -> x0.85  |  Phase 1 -> x0.65
    >=22% -> 5  |  16-21.9% -> 4  |  10-15.9% -> 3  |  5-9.9% -> 2  |  <5% -> 1
    Weights: Weight Loss 40% | HbA1c 40% | MASH 10% | ALT 10%

Usage:
    python fetcher.py Cagrisema
    python fetcher.py Cagrisema --excel cagrisema_trials.xlsx --workers 8
    python fetcher.py Cagrisema --no-score
    python fetcher.py Cagrisema --no-excel        # skip writing output xlsx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# -- third-party ---------------------------------------------------------------
try:
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

# -- config from gcp_utils -----------------------------------------------------
from gcp_utils import GEMINI_API_KEY, GOOGLE_API_KEY, MODEL, RATIONALE_MODEL

# ==============================================================================
# SECTION 1 – CONSTANTS & COLUMN DEFINITIONS
# ==============================================================================

MAX_RETRIES     = 5
INITIAL_BACKOFF = 2.0
BATCH_SIZE      = 6
DEFAULT_WORKERS = 6

OUTCOME_COLS = [
    "dosage",
    "hba1c_change_pct",  "hba1c_duration",  "hba1c_rationale",  "hba1c_confidence",
    "weight_change_pct", "weight_duration",  "weight_rationale", "weight_confidence",
    "alt_reduction_pct", "alt_duration",     "alt_rationale",    "alt_confidence",
    "mash_change_pct",   "mash_duration",    "mash_rationale",   "mash_confidence",
]

ALL_COLUMNS = [
    "molecule_name", "registry_source", "trial_id", "acronym",
    "dosage", "phase", "trial_title", "trial_study", "trial_size",
    "trial_location", "trial_start_date", "trial_completion_date", "phase_status",
    "hba1c_change_pct",  "hba1c_duration",  "hba1c_rationale",  "hba1c_confidence",
    "weight_change_pct", "weight_duration",  "weight_rationale", "weight_confidence",
    "alt_reduction_pct", "alt_duration",     "alt_rationale",    "alt_confidence",
    "mash_change_pct",   "mash_duration",    "mash_rationale",   "mash_confidence",
    "company_name", "source_url",
    "efficacy_weighted_score", "efficacy_data_coverage",
    "efficacy_score_breakdown", "efficacy_narrative_rationale",
]

COLUMN_WIDTHS = {
    "molecule_name": 16, "registry_source": 18, "trial_id": 18,
    "acronym": 18, "dosage": 18, "phase": 10, "trial_title": 40,
    "trial_study": 26, "trial_size": 12, "trial_location": 24,
    "trial_start_date": 16, "trial_completion_date": 18, "phase_status": 18,
    "hba1c_change_pct": 16,  "hba1c_duration": 14, "hba1c_rationale": 45,  "hba1c_confidence": 14,
    "weight_change_pct": 16, "weight_duration": 14, "weight_rationale": 45, "weight_confidence": 14,
    "alt_reduction_pct": 16, "alt_duration": 14,   "alt_rationale": 45,    "alt_confidence": 14,
    "mash_change_pct": 18,   "mash_duration": 14,  "mash_rationale": 45,   "mash_confidence": 14,
    "company_name": 28, "source_url": 40,
    "efficacy_weighted_score": 20, "efficacy_data_coverage": 28,
    "efficacy_score_breakdown": 50, "efficacy_narrative_rationale": 60,
}

# ==============================================================================
# SECTION 2 – GEMINI CLIENT
# ==============================================================================

class _NoApiKeyError(RuntimeError):
    pass


def _make_client() -> genai.Client:
    api_key = GEMINI_API_KEY or GOOGLE_API_KEY
    if not api_key:
        raise _NoApiKeyError(
            "No Gemini API key found.\n"
            "  Set GEMINI_API_KEY in your .env file or environment."
        )
    return genai.Client(api_key=api_key)


_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


def _check_api_key_early() -> None:
    try:
        get_client()
    except _NoApiKeyError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        sys.exit(1)


# ==============================================================================
# SECTION 3 – JSON HELPERS
# ==============================================================================

def _safe_parse(text: str) -> Any:
    if not text:
        return None
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
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


# ==============================================================================
# SECTION 4 – GEMINI CALLS
# ==============================================================================

def _sync_call(prompt: str, use_search: bool = True) -> str:
    client = get_client()
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config_kwargs: Dict[str, Any] = {}
    if use_search:
        config_kwargs["tools"] = [types.Tool(googleSearch=types.GoogleSearch())]
    config = types.GenerateContentConfig(**config_kwargs)
    out = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL, contents=contents, config=config
    ):
        if chunk.text:
            out += chunk.text
    return out.strip()


async def _gemini_call(prompt: str, use_search: bool = True) -> str:
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(_sync_call, prompt, use_search)
        except _NoApiKeyError:
            raise
        except Exception as exc:
            err = str(exc).lower()
            if any(k in err for k in ("429", "rate limit", "quota", "resource exhausted")):
                if attempt == MAX_RETRIES:
                    print(f"  X Max retries exceeded: {exc}", file=sys.stderr)
                    raise
                print(f"  ! Rate-limit – waiting {backoff:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...", file=sys.stderr)
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                print(f"  X Gemini error: {exc}", file=sys.stderr)
                raise
    return ""


# ==============================================================================
# SECTION 5 – ENRICHMENT
# ==============================================================================

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

1. dosage          - Primary or highest dose tested (e.g. "2.4 mg OW", "15 mg QD").
                     If multiple doses, pick the highest. Format: "[amount] [unit] [frequency]"

2. hba1c_change_pct  - HbA1c reduction in percentage points (positive number, e.g. "1.8").
                        "N/A" if not a diabetes trial or data unavailable.
   hba1c_duration     - Timepoint of measurement (e.g. "26 wk"). "N/A" if unavailable.
   hba1c_rationale    - 1-2 sentences: state the exact source and why this value was chosen.
   hba1c_confidence   - "High" / "Medium" / "Low" reflecting data reliability.

3. weight_change_pct - Body weight loss percentage (positive number). "N/A" if unavailable.
   weight_duration    - Timepoint (e.g. "68 wk"). "N/A" if unavailable.
   weight_rationale   - 1-2 sentences citing source and reason for value chosen.
   weight_confidence  - "High" / "Medium" / "Low".

4. alt_reduction_pct - ALT enzyme reduction percentage (positive number). "N/A" if unavailable.
   alt_duration       - Timepoint. "N/A" if unavailable.
   alt_rationale      - 1-2 sentences citing source and reason.
   alt_confidence     - "High" / "Medium" / "Low".

5. mash_change_pct   - MASH/NASH resolution rate or fibrosis improvement % (positive number).
                        "N/A" if not a liver trial.
   mash_duration      - Timepoint. "N/A" if unavailable.
   mash_rationale     - 1-2 sentences citing source and reason.
   mash_confidence    - "High" / "Medium" / "Low".

RULES:
- Use actual published results where available; fall back to registry data.
- Report reductions as POSITIVE numbers.
- Use "N/A" for fields with genuinely no data.
- Each rationale MUST name the specific source (e.g. "NEJM 2023 STEP 1 paper").
- One JSON object per trial keyed by trial_id.

Return ONLY valid JSON, no markdown, no preamble:

{{
  "results": {{
    "<trial_id_1>": {{
      "dosage": "...",
      "hba1c_change_pct": "...", "hba1c_duration": "...", "hba1c_rationale": "...", "hba1c_confidence": "...",
      "weight_change_pct": "...", "weight_duration": "...", "weight_rationale": "...", "weight_confidence": "...",
      "alt_reduction_pct": "...", "alt_duration": "...", "alt_rationale": "...", "alt_confidence": "...",
      "mash_change_pct": "...", "mash_duration": "...", "mash_rationale": "...", "mash_confidence": "..."
    }},
    "<trial_id_2>": {{ ... }}
  }}
}}
"""


async def _enrich_batch(
    molecule: str,
    batch: List[Dict[str, str]],
    batch_idx: int,
    total_batches: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Dict[str, str]]:
    async with semaphore:
        ids = [t.get("trial_id", "?") for t in batch]
        print(f"  Batch {batch_idx+1}/{total_batches} -> {ids}", file=sys.stderr)
        try:
            raw = await _gemini_call(_build_prompt(molecule, batch), use_search=True)
        except Exception:
            return {}
        data = _safe_parse(raw)
        if not data:
            print(f"  X Batch {batch_idx+1}: could not parse response", file=sys.stderr)
            return {}
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        elif isinstance(data, dict):
            results = data
        else:
            print(f"  X Batch {batch_idx+1}: unexpected JSON structure", file=sys.stderr)
            return {}
        print(f"  OK Batch {batch_idx+1}: enriched {len(results)} trial(s)", file=sys.stderr)
        return results


async def enrich_all(
    molecule: str,
    rows: List[Dict[str, str]],
    max_workers: int = DEFAULT_WORKERS,
) -> List[Dict[str, str]]:
    batches = [rows[i: i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    total = len(batches)
    print(f"\n[ENRICH] {len(rows)} trial(s) across {total} batch(es) (max {max_workers} concurrent)...\n", file=sys.stderr)
    semaphore = asyncio.Semaphore(max_workers)

    async def _staggered(coro, delay: float):
        await asyncio.sleep(delay)
        return await coro

    staggered_tasks = [
        _staggered(_enrich_batch(molecule, batch, idx, total, semaphore), idx * 0.4)
        for idx, batch in enumerate(batches)
    ]
    batch_results = await asyncio.gather(*staggered_tasks, return_exceptions=False)

    merged: Dict[str, Dict[str, str]] = {}
    for br in batch_results:
        if isinstance(br, dict):
            merged.update(br)

    updated = 0
    for row in rows:
        tid = row.get("trial_id", "")
        enrichment = merged.get(tid, {})
        if not enrichment:
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
            for col in OUTCOME_COLS:
                if col not in row or not row[col]:
                    row[col] = "N/A"

    print(f"\n[ENRICH] Done: {updated}/{len(rows)} trial(s) updated.\n", file=sys.stderr)
    return rows


# ==============================================================================
# SECTION 6 – SCORING
# ==============================================================================

SCORE_TABLE = [(22.0, 5), (16.0, 4), (10.0, 3), (5.0, 2), (0.0, 1)]
ENDPOINT_WEIGHTS = {"weight_loss": 0.40, "hba1c": 0.40, "mash": 0.10, "alt": 0.10}
FIELD_MAP = {
    "weight_loss": "weight_change_pct",
    "hba1c":       "hba1c_change_pct",
    "mash":        "mash_change_pct",
    "alt":         "alt_reduction_pct",
}
PHASE_PENALTY = {3: 1.00, 2: 0.85, 1: 0.65}


def _parse_phase(raw) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip().upper().replace("PHASE", "").strip()
    if s.startswith("3"): return 3
    if s.startswith("2"): return 2
    if s.startswith("1"): return 1
    try:
        v = float(s)
        return 3 if v >= 3 else (2 if v >= 2 else 1)
    except ValueError:
        return None


def _parse_float(raw) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s.lower() in ("n/a", "", "0", "none", "null"):
        return None
    s = s.rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _pct_to_score(pct: float) -> int:
    for threshold, score in SCORE_TABLE:
        if pct >= threshold:
            return score
    return 1


def _score_endpoint(trials: List[Dict[str, str]], value_field: str) -> Dict[str, Any]:
    valid = []
    for t in trials:
        phase    = _parse_phase(t.get("phase"))
        value    = _parse_float(t.get(value_field))
        n        = _parse_float(t.get("trial_size")) or 0
        trial_id = t.get("trial_id") or t.get("Trial ID")
        if phase is None or value is None or n <= 0:
            continue
        if not trial_id:
            trial_id = f"__unknown_{id(t)}"
        valid.append({"phase": phase, "value": value, "n": n, "trial_id": trial_id, "full_trial": t})

    if not valid:
        return {"best_value": None, "raw_value": None, "phase_used": None,
                "penalty": 1.0, "score": None, "trial_details": {},
                "reason": "No valid data for this endpoint"}

    for target_phase in (3, 2, 1):
        phase_trials = [r for r in valid if r["phase"] == target_phase]
        if not phase_trials:
            continue
        trial_groups: Dict[str, list] = {}
        for t in phase_trials:
            trial_groups.setdefault(t["trial_id"], []).append(t)
        deduplicated = [max(arms, key=lambda x: x["value"]) for arms in trial_groups.values()]
        best = max(deduplicated, key=lambda r: r["value"])
        raw  = best["value"]
        pen  = PHASE_PENALTY[target_phase]
        adj  = raw * pen
        ft   = best.get("full_trial", {})
        return {
            "best_value":  round(adj, 4),
            "raw_value":   round(raw, 4),
            "phase_used":  target_phase,
            "penalty":     pen,
            "score":       _pct_to_score(adj),
            "trial_details": {
                "trial_id":       best["trial_id"],
                "dosage":         ft.get("dosage", "N/A"),
                "weight_duration": ft.get("weight_duration", "N/A"),
                "hba1c_duration":  ft.get("hba1c_duration", "N/A"),
                "mash_duration":   ft.get("mash_duration", "N/A"),
                "alt_duration":    ft.get("alt_duration", "N/A"),
            },
            "reason": f"Phase {target_phase} data used" + (f" (x{pen} penalty applied)" if pen < 1 else ""),
        }

    return {"best_value": None, "raw_value": None, "phase_used": None,
            "penalty": 1.0, "score": None, "trial_details": {}, "reason": "Unexpected state"}


def compute_clinical_efficacy_score(molecule: str, rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    endpoint_results = {ep: _score_endpoint(rows, field) for ep, field in FIELD_MAP.items()}

    score_sum, scored_eps, missing_eps = 0.0, [], []
    for ep, result in endpoint_results.items():
        w = ENDPOINT_WEIGHTS[ep]
        if result["score"] is not None:
            score_sum += result["score"] * w
            scored_eps.append(ep)
        else:
            missing_eps.append(ep)

    lines = []
    for ep, result in endpoint_results.items():
        w_pct = int(ENDPOINT_WEIGHTS[ep] * 100)
        if result["score"] is not None:
            lines.append(f"  {ep:12} | adj={result['best_value']:.2f}%  score={result['score']}  weight={w_pct}%  ({result['reason']})")
        else:
            lines.append(f"  {ep:12} | N/A  weight={w_pct}%  ({result['reason']})")

    coverage = (
        f"{len(scored_eps)}/4 endpoints scored"
        + (f" (missing: {', '.join(missing_eps)})" if missing_eps else "")
    )
    return {
        "molecule":        molecule,
        "total_trials":    total,
        "endpoints":       endpoint_results,
        "weighted_score":  round(score_sum, 3),
        "score_breakdown": "\n".join(lines),
        "data_coverage":   coverage,
    }


# ==============================================================================
# SECTION 7 – NARRATIVE RATIONALE
# ==============================================================================

def generate_score_rationale(molecule: str, score_result: Dict[str, Any]) -> str:
    print("\n[SCORE] Generating narrative rationale via Gemini...", file=sys.stderr)
    endpoints = score_result.get("endpoints", {})

    endpoint_summary = {
        "Weight Loss (40% weight)": {
            "score":      endpoints.get("weight_loss", {}).get("score"),
            "best_value": endpoints.get("weight_loss", {}).get("best_value"),
            "raw_value":  endpoints.get("weight_loss", {}).get("raw_value"),
            "phase_used": endpoints.get("weight_loss", {}).get("phase_used"),
            "trial_used": endpoints.get("weight_loss", {}).get("trial_details", {}).get("trial_id"),
            "dosage":     endpoints.get("weight_loss", {}).get("trial_details", {}).get("dosage"),
            "duration":   endpoints.get("weight_loss", {}).get("trial_details", {}).get("weight_duration"),
        },
        "HbA1c Reduction (40% weight)": {
            "score":      endpoints.get("hba1c", {}).get("score"),
            "best_value": endpoints.get("hba1c", {}).get("best_value"),
            "raw_value":  endpoints.get("hba1c", {}).get("raw_value"),
            "phase_used": endpoints.get("hba1c", {}).get("phase_used"),
            "trial_used": endpoints.get("hba1c", {}).get("trial_details", {}).get("trial_id"),
            "dosage":     endpoints.get("hba1c", {}).get("trial_details", {}).get("dosage"),
            "duration":   endpoints.get("hba1c", {}).get("trial_details", {}).get("hba1c_duration"),
        },
        "MASH Resolution (10% weight)": {
            "score":      endpoints.get("mash", {}).get("score"),
            "best_value": endpoints.get("mash", {}).get("best_value"),
            "raw_value":  endpoints.get("mash", {}).get("raw_value"),
            "phase_used": endpoints.get("mash", {}).get("phase_used"),
            "trial_used": endpoints.get("mash", {}).get("trial_details", {}).get("trial_id"),
            "dosage":     endpoints.get("mash", {}).get("trial_details", {}).get("dosage"),
            "duration":   endpoints.get("mash", {}).get("trial_details", {}).get("mash_duration"),
        },
        "ALT Reduction (10% weight)": {
            "score":      endpoints.get("alt", {}).get("score"),
            "best_value": endpoints.get("alt", {}).get("best_value"),
            "raw_value":  endpoints.get("alt", {}).get("raw_value"),
            "phase_used": endpoints.get("alt", {}).get("phase_used"),
            "trial_used": endpoints.get("alt", {}).get("trial_details", {}).get("trial_id"),
            "dosage":     endpoints.get("alt", {}).get("trial_details", {}).get("dosage"),
            "duration":   endpoints.get("alt", {}).get("trial_details", {}).get("alt_duration"),
        },
    }

    prompt = f"""You are a clinical pharmacology expert. Generate a concise, evidence-based rationale explaining the clinical efficacy score for {molecule}.

SCORING RESULTS:
- Clinical Efficacy Score: {score_result['weighted_score']} / 5
- Coverage: {score_result['data_coverage']}

ENDPOINT PERFORMANCE (EXACT trials used for scoring):
{json.dumps(endpoint_summary, indent=2)}

SCORING METHODOLOGY:
- Score ranges: 5 = >=22%, 4 = 16-21.9%, 3 = 10-15.9%, 2 = 5-9.9%, 1 = <5%
- Phase penalties: Phase 3 = no penalty, Phase 2 = x0.85, Phase 1 = x0.65
- Weighted average: Weight Loss (40%) + HbA1c (40%) + MASH (10%) + ALT (10%)

YOUR TASK:
Write a concise clinical rationale in EXACTLY 3 sentences:
1. State the overall clinical efficacy score and briefly summarise {molecule}'s performance across the scored endpoints.
2. Highlight the strongest endpoint(s) with specific percentage, dosage, duration, phase, and trial ID.
3. Note any missing endpoints or data gaps and state what the score reflects about the molecule's overall clinical profile.

WRITING GUIDELINES:
- EXACTLY 3 sentences — no more, no less
- Include specific numbers (percentages, trial IDs, phase info, dosage, duration) where available
- Plain text only — no markdown, no headers, no bullets
- Write as documentation for regulatory or pharma stakeholders

IMPORTANT: Use trial_id, dosage, and duration ONLY from the ENDPOINT PERFORMANCE section above.

Generate the rationale now:"""

    try:
        client = get_client()
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        config = types.GenerateContentConfig(temperature=0.3, response_mime_type="text/plain")
        response_text = ""
        for chunk in client.models.generate_content_stream(model=RATIONALE_MODEL, contents=contents, config=config):
            if chunk.text:
                response_text += chunk.text
        rationale = response_text.strip().replace("\n\n\n", "\n\n")
        print("[SCORE] Rationale generated.", file=sys.stderr)
        return rationale
    except Exception as exc:
        print(f"  [SCORE] Rationale generation failed: {exc}", file=sys.stderr)
        return (
            f"{molecule} received a clinical efficacy score of "
            f"{score_result['weighted_score']}/5 based on analysis of "
            f"{score_result['total_trials']} trials. {score_result['data_coverage']}."
        )


# ==============================================================================
# SECTION 8 – EXCEL I/O
# ==============================================================================

def _read_excel(path: str) -> List[Dict[str, str]]:
    wb = load_workbook(path)
    ws = wb.active
    headers = [str(c.value or "").strip() for c in ws[1]]
    rows: List[Dict[str, str]] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        record: Dict[str, str] = {h: (str(v) if v is not None else "") for h, v in zip(headers, row)}
        if any(record.values()):
            rows.append(record)
    print(f"[INPUT] Loaded {len(rows)} row(s) from {path}", file=sys.stderr)
    return rows


def _write_excel(
    rows: List[Dict[str, str]],
    path: str,
    score_result: Optional[Dict[str, Any]] = None,
    score_rationale: Optional[str] = None,
) -> None:
    from openpyxl import Workbook

    if score_result:
        for row_data in rows:
            row_data["efficacy_weighted_score"]      = str(score_result.get("weighted_score", ""))
            row_data["efficacy_data_coverage"]       = score_result.get("data_coverage", "")
            row_data["efficacy_score_breakdown"]     = score_result.get("score_breakdown", "")
            row_data["efficacy_narrative_rationale"] = score_rationale or ""

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
    ENRICH_FILL = PatternFill("solid", fgColor="E8F8E8")
    SCORE_FILL  = PatternFill("solid", fgColor="D4EDDA")

    SCORE_COLS = {"efficacy_weighted_score", "efficacy_data_coverage",
                  "efficacy_score_breakdown", "efficacy_narrative_rationale"}

    extra_cols = [c for c in (rows[0].keys() if rows else []) if c not in ALL_COLUMNS]
    columns = ALL_COLUMNS + extra_cols
    enriched_col_idxs = {i + 1 for i, c in enumerate(columns) if c in OUTCOME_COLS}
    score_col_idxs    = {i + 1 for i, c in enumerate(columns) if c in SCORE_COLS}

    ws.append(columns)
    for cell in ws[1]:
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border    = BORDER

    for r_idx, row_data in enumerate(rows, start=2):
        ws.append([row_data.get(col, "") for col in columns])
        alt = r_idx % 2 == 0
        for c_idx, cell in enumerate(ws[r_idx], start=1):
            cell.font      = DATA_FONT
            cell.alignment = WRAP_ALIGN
            cell.border    = BORDER
            if c_idx in score_col_idxs:
                cell.fill = SCORE_FILL
            elif c_idx in enriched_col_idxs:
                cell.fill = ENRICH_FILL
            elif alt:
                cell.fill = ALT_FILL

    for col_idx, col_name in enumerate(columns, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = COLUMN_WIDTHS.get(col_name, 18)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(path)
    print(f"\n[OUTPUT] Wrote {len(rows)} row(s) -> {path}", file=sys.stderr)
    if score_result:
        print(f"[OUTPUT] Efficacy score: {score_result.get('weighted_score','?')} / 5.0", file=sys.stderr)


# ==============================================================================
# SECTION 9 – PUBLIC API
# ==============================================================================

def run_fetcher(
    molecule: str,
    excel_path: str,
    max_workers: int = DEFAULT_WORKERS,
    no_score: bool = False,
) -> tuple:
    """
    Run enrichment + scoring on an already-generated trials Excel file.
    Returns: (enriched_rows, score_result, score_rationale)
    """
    _check_api_key_early()

    rows = _read_excel(excel_path)
    if not rows:
        print("[FETCHER] No rows found in input Excel.", file=sys.stderr)
        return [], None, None

    t0 = time.time()
    enriched_rows = asyncio.run(enrich_all(molecule, rows, max_workers=max_workers))
    print(f"[ENRICH] Time: {time.time() - t0:.1f}s", file=sys.stderr)

    score_result: Optional[Dict[str, Any]] = None
    score_rationale: Optional[str] = None

    if not no_score:
        print(f"\n[SCORE] Computing Clinical Efficacy Score...", file=sys.stderr)
        score_result = compute_clinical_efficacy_score(molecule, enriched_rows)
        print(f"  Weighted Score : {score_result['weighted_score']} / 5.0", file=sys.stderr)
        print(f"  Coverage       : {score_result['data_coverage']}", file=sys.stderr)
        print(f"  Breakdown:\n{score_result['score_breakdown']}", file=sys.stderr)
        score_rationale = generate_score_rationale(molecule, score_result)

    if score_result:
        for row_data in enriched_rows:
            row_data["efficacy_weighted_score"]      = str(score_result.get("weighted_score", ""))
            row_data["efficacy_data_coverage"]       = score_result.get("data_coverage", "")
            row_data["efficacy_score_breakdown"]     = score_result.get("score_breakdown", "")
            row_data["efficacy_narrative_rationale"] = score_rationale or ""

    return enriched_rows, score_result, score_rationale


# ==============================================================================
# SECTION 10 – CLI
# ==============================================================================

def _resolve_input_excel(molecule: str, explicit: Optional[str]) -> str:
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"ERROR: File not found: {explicit}")
        return explicit
    candidate = f"{molecule.lower().replace(' ', '_')}_trials.xlsx"
    if os.path.exists(candidate):
        return candidate
    candidates = [f for f in os.listdir(".") if f.endswith("_trials.xlsx") or f.endswith("_trials_enriched.xlsx")]
    if len(candidates) == 1:
        print(f"  i Auto-discovered: {candidates[0]}", file=sys.stderr)
        return candidates[0]
    sys.exit(f"ERROR: Could not find input Excel. Expected: {candidate}\nOr use: --excel <path>")


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich + score clinical trial Excel, then optionally write output.")
    ap.add_argument("molecule")
    ap.add_argument("--excel",       default=None)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--workers",     type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--run-main1",   action="store_true")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--top-n",       type=int, default=None)
    ap.add_argument("--no-score",    action="store_true")
    ap.add_argument("--no-excel",    action="store_true", help="Skip writing output xlsx")
    args = ap.parse_args()

    molecule = args.molecule.strip()
    slug     = molecule.lower().replace(" ", "_")
    out_path = args.out or f"{slug}_trials_enriched.xlsx"

    print(f"\n{'='*60}\n  FETCHER  -  {molecule}\n{'='*60}\n", file=sys.stderr)

    if args.run_main1:
        import subprocess
        cmd = [sys.executable, "main1.py", molecule, "--no-enrich"]
        if args.max_records: cmd += ["--max-records", str(args.max_records)]
        if args.top_n:       cmd += ["--top-n", str(args.top_n)]
        print(f"> Running: {' '.join(cmd)}\n", file=sys.stderr)
        if subprocess.run(cmd).returncode != 0:
            sys.exit("ERROR: main1.py failed.")

    excel_path = _resolve_input_excel(molecule, args.excel)
    t0 = time.time()
    enriched_rows, score_result, score_rationale = run_fetcher(
        molecule, excel_path, max_workers=args.workers, no_score=args.no_score
    )
    if not enriched_rows:
        return 1

    if not args.no_excel:
        _write_excel(enriched_rows, out_path, score_result=score_result, score_rationale=score_rationale)

    print(
        f"\nDone!\n  Rows       : {len(enriched_rows)}\n  Total time : {time.time()-t0:.1f}s\n"
        + (f"  Output     : {out_path}\n" if not args.no_excel else "  (no Excel written)\n"),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())