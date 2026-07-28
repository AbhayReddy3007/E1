#!/usr/bin/env python3
"""
gemini_search_trials.py – Two-step Gemini extraction for blocked registries.

Follows the same pattern as the reference gemini_extractor:
  Step 1: Gemini + Google Search → find trial IDs per registry
  Step 2: Gemini + Google Search → extract full trial details per batch

Only runs for registries that returned 0 from direct scraping.
Uses the REST API (no google-genai SDK needed).
"""

from __future__ import annotations
import json, os, re, sys, time
from typing import Any, Dict, List, Optional
import requests

from registry_common import (
    SRC_ANZCTR, SRC_CHICTR, SRC_CRIS, SRC_CTRI, SRC_JRCT, SRC_REBEC,
    UNIFIED_COLUMNS, blank_row, clean, make_session,
)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"

BATCH_SIZE = 8
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0

REGISTRIES = {
    SRC_CTRI: {"id_prefix": "CTRI", "url": "https://ctri.nic.in/", "region": "India",
               "name": "Clinical Trials Registry - India (CTRI)"},
    SRC_JRCT: {"id_prefix": "jRCT", "url": "https://jrct.mhlw.go.jp/", "region": "Japan",
               "name": "Japan Registry of Clinical Trials (jRCT)"},
    SRC_ANZCTR: {"id_prefix": "ACTRN", "url": "https://www.anzctr.org.au/", "region": "Australia/New Zealand",
                 "name": "Australian New Zealand Clinical Trials Registry (ANZCTR)"},
    SRC_CHICTR: {"id_prefix": "ChiCTR", "url": "https://www.chictr.org.cn/", "region": "China",
                 "name": "Chinese Clinical Trial Registry (ChiCTR)"},
    SRC_CRIS: {"id_prefix": "KCT", "url": "https://cris.nih.go.kr/", "region": "Korea",
               "name": "Clinical Research Information Service (CRIS)"},
    SRC_REBEC: {"id_prefix": "RBR", "url": "https://ensaiosclinicos.gov.br/", "region": "Brazil",
                "name": "Brazilian Clinical Trials Registry (ReBEC)"},
}

ID_VALIDATORS = {
    SRC_CTRI: re.compile(r"CTRI/\d{4}/\d{2,3}/\d+", re.I),
    SRC_JRCT: re.compile(r"jRCT[a-zA-Z0-9]*\d{6,}", re.I),
    SRC_ANZCTR: re.compile(r"ACTRN\d{10,}", re.I),
    SRC_CHICTR: re.compile(r"ChiCTR[-a-zA-Z0-9]*\d+", re.I),
    SRC_CRIS: re.compile(r"KCT\d{5,}", re.I),
    SRC_REBEC: re.compile(r"RBR-[0-9a-zA-Z]+", re.I),
}


# ── Gemini API call with google_search and retry ────────────────────────────
def _gemini_call(session: requests.Session, url: str, prompt: str,
                 timeout: int = 180) -> str:
    """Call Gemini with google_search tool, exponential backoff on rate limits."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192},
    }

    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(url, data=json.dumps(body), timeout=timeout)
            if r.status_code == 429:
                print(f"    Rate limit – waiting {backoff:.0f}s (attempt {attempt}/{MAX_RETRIES})",
                      file=sys.stderr)
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()

            payload = r.json()
            cands = payload.get("candidates") or []
            if not cands:
                return ""
            parts = (cands[0].get("content") or {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts).strip()

        except Exception as exc:
            if attempt == MAX_RETRIES:
                print(f"    Gemini call failed: {exc}", file=sys.stderr)
                return ""
            time.sleep(backoff)
            backoff *= 2
    return ""


def _parse_json(text: str) -> Any:
    """Extract and parse JSON from Gemini response text."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.M)
    text = re.sub(r"```\s*$", "", text, flags=re.M).strip()
    m = re.search(r"[\[{].*[\]}]", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Try json_repair if available
        try:
            from json_repair import repair_json
            return repair_json(m.group(0), return_objects=True)
        except Exception:
            pass
        # Last resort: find first complete { }
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(m.group(0))
            return obj
        except Exception:
            return None


# ── STEP 1: Search registry for trial IDs ───────────────────────────────────
def _step1_search(session: requests.Session, url: str, drug: str,
                  source: str, reg: dict) -> List[dict]:
    """Use Gemini+GoogleSearch to find all trial IDs on a registry."""

    prompt = f"""Search {reg['name']} ({reg['url']}) for ALL clinical trials of {drug}.

REGISTRY: {reg['name']}
URL: {reg['url']}
REGION: {reg['region']}
ID PREFIX: {reg['id_prefix']}

YOUR TASK: Find ALL {drug} trials registered on this specific registry.
Search Google for: site:{reg['url'].replace('https://', '').rstrip('/')} "{drug}"
Also search: {drug} clinical trial {reg['name']}

Include ALL phases, ALL indications (diabetes, obesity, MASH, cardiovascular, etc.),
completed/active/recruiting trials.

For each trial found, provide:
- trial_id: Registry ID (must start with {reg['id_prefix']})
- title: Trial title
- phase: Trial phase
- indication: Primary indication
- status: Recruitment status

Return as JSON:
{{
  "trials": [
    {{"trial_id": "{reg['id_prefix']}XXXXXXXX", "title": "...", "phase": "3", "indication": "...", "status": "..."}}
  ]
}}

If no trials found, return: {{"trials": []}}"""

    raw = _gemini_call(session, url, prompt)
    data = _parse_json(raw)
    if not data:
        return []

    trials = data.get("trials", []) if isinstance(data, dict) else data if isinstance(data, list) else []

    # Validate IDs
    validator = ID_VALIDATORS.get(source)
    valid = []
    for t in trials:
        tid = clean(t.get("trial_id", ""))
        if validator and validator.search(tid):
            t["trial_id"] = tid
            t["registry_source"] = source
            valid.append(t)

    return valid


# ── STEP 2: Extract full details per batch ──────────────────────────────────
def _step2_details(session: requests.Session, url: str, drug: str,
                   trial_batch: List[dict], source: str,
                   reg: dict) -> List[Dict[str, Any]]:
    """Use Gemini+GoogleSearch to extract detailed data for a batch of trials."""

    trial_list = "\n".join(
        f"- {t.get('trial_id', 'N/A')} ({t.get('title', '')[:60]})"
        for t in trial_batch)

    prompt = f"""You are a clinical data extraction engine. Extract detailed trial data
for these {len(trial_batch)} {drug} trials from {reg['name']}:

{trial_list}

DATA SOURCES (search ALL):
1. {reg['name']} - {reg['url']}
2. PubMed/published papers linked to the trial
3. ClinicalTrials.gov (cross-reference if trial is also registered there)
4. WHO ICTRP - https://trialsearch.who.int/

For EACH trial, extract ALL of these fields:
- trial_id: The registry ID
- title: Full official trial title
- dosage: Drug dosage(s) tested (e.g., "2.4 mg OW", "14 mg QD")
- phase: Trial phase (e.g., "1", "2", "3", "3b", "4")
- study_type: "Interventional", "Observational", or "Expanded Access"
- status: "Completed", "Active", "Recruiting", "Terminated"
- conditions: Disease/conditions studied
- interventions: All interventions
- drug_names: Drug names used
- sponsor: Sponsor company
- countries: Countries where trial conducted
- target_enrollment: Planned number of participants
- actual_enrollment: Actual number enrolled
- age_min: Minimum age
- age_max: Maximum age
- gender: Eligible gender
- primary_outcome: Primary outcome measure
- secondary_outcome: Secondary outcome measures
- start_date: Study start date (YYYY-MM-DD or YYYY-MM)
- completion_date: Primary completion date
- registration_date: Date registered
- results_available: "Yes" or "No"
- hba1c_change_pct: HbA1c reduction in percentage points (e.g., "-1.5") or "N/A"
- hba1c_duration: Timepoint for HbA1c measurement (e.g., "26 weeks") or "N/A"
- weight_change_pct: Body weight change percentage (e.g., "-15.2%") or "N/A"
- weight_duration: Timepoint for weight measurement (e.g., "68 weeks") or "N/A"
- alt_reduction_pct: ALT change percentage or absolute (e.g., "-40%") or "N/A"
- alt_duration: Timepoint for ALT measurement or "N/A"
- mash_resolution_pct: MASH/NASH resolution rate (e.g., "59%") or "N/A"
- mash_duration: Timepoint for MASH assessment or "N/A"
- findings: Any key results/conclusions mentioned
- url: Direct URL to the trial page on the registry
- inclusion_criteria: Key inclusion criteria
- exclusion_criteria: Key exclusion criteria
- contact: Contact person/email if available
- ethics_approval: Ethics committee info if available

IMPORTANT:
- Include ALL {len(trial_batch)} trials even if data is incomplete
- Use "N/A" for missing fields — NEVER invent data
- Report reductions as numbers (positive = reduction)
- Provide actual source URLs from the registry

Return JSON:
{{
  "trials": [
    {{
      "trial_id": "...",
      "title": "...",
      "dosage": "...",
      ... all fields ...
    }}
  ]
}}"""

    raw = _gemini_call(session, url, prompt)
    data = _parse_json(raw)
    if not data:
        return []

    trials = data.get("trials", []) if isinstance(data, dict) else data if isinstance(data, list) else []

    # Convert to unified rows
    rows = []
    for t in trials:
        if not isinstance(t, dict):
            continue
        row = blank_row(source)
        # Map every field
        for col in UNIFIED_COLUMNS:
            if col in t and t[col]:
                row[col] = clean(t[col])
        # Map extra fields that don't match unified names exactly
        row["trial_id"] = clean(t.get("trial_id", ""))
        row["source"] = source
        # Carry forward metric columns
        for key in ("dosage", "hba1c_change_pct", "hba1c_duration",
                    "weight_change_pct", "weight_duration", "alt_reduction_pct",
                    "alt_duration", "mash_resolution_pct", "mash_duration"):
            if key in t and clean(t[key]):
                row[key] = clean(t[key])
        # Map other fields
        if t.get("sponsor"):
            row["sponsor"] = clean(t["sponsor"])
        if t.get("url"):
            row["url"] = clean(t["url"])
        if t.get("title"):
            row["title"] = clean(t["title"])
        if t.get("findings"):
            row["findings"] = clean(t["findings"])
        rows.append(row)

    return rows


# ── Main entry point ─────────────────────────────────────────────────────────
def search_missing_registries(
    drug: str,
    missing_sources: List[str],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, List[Dict[str, Any]]]:
    """Two-step Gemini extraction for registries that returned 0 rows."""

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [Gemini search] no GEMINI_API_KEY – cannot search.",
              file=sys.stderr)
        return {}

    searchable = [s for s in missing_sources if s in REGISTRIES]
    if not searchable:
        return {}

    url = API_URL.format(model=model)
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    })

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    # ── Step 1: Find trial IDs per registry ──────────────────────────────────
    print(f"\n  [Gemini] Step 1: Searching {len(searchable)} registries for "
          f"'{drug}' trial IDs ...", file=sys.stderr)

    step1_results: Dict[str, List[dict]] = {}
    for source in searchable:
        reg = REGISTRIES[source]
        print(f"    Searching {reg['name']} ...", file=sys.stderr)
        trials = _step1_search(session, url, drug, source, reg)
        step1_results[source] = trials
        print(f"    {reg['name']}: {len(trials)} trial(s) found", file=sys.stderr)
        time.sleep(1)

    total_found = sum(len(t) for t in step1_results.values())
    if total_found == 0:
        print(f"  [Gemini] No trials found across any registry.", file=sys.stderr)
        return {}

    # ── Step 2: Extract full details in batches ──────────────────────────────
    print(f"\n  [Gemini] Step 2: Extracting details for {total_found} trials ...",
          file=sys.stderr)

    for source, trial_ids in step1_results.items():
        if not trial_ids:
            all_results[source] = []
            continue

        reg = REGISTRIES[source]
        rows: List[Dict[str, Any]] = []

        for batch_start in range(0, len(trial_ids), BATCH_SIZE):
            batch = trial_ids[batch_start:batch_start + BATCH_SIZE]
            print(f"    {reg['name']}: batch {batch_start // BATCH_SIZE + 1} "
                  f"({len(batch)} trials) ...", file=sys.stderr)

            batch_rows = _step2_details(session, url, drug, batch, source, reg)
            rows.extend(batch_rows)
            time.sleep(1)

        all_results[source] = rows
        print(f"  [Gemini] {source}: {len(rows)} trial(s) with details",
              file=sys.stderr)

    return all_results