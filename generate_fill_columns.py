#!/usr/bin/env python3
"""
gemini_fill_columns.py – Gemini Step 2 extraction for all trials.

Takes trial_id + source_url from already-fetched trials and uses
Gemini + Google Search to extract the 22 output columns.

Follows the EXACT same extraction pattern as the reference
gemini_extractor.py get_trial_details() function.

Only looks at the source URL / registry page — no PubMed, no other sources.
"""

from __future__ import annotations
import json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"
BATCH_SIZE = 6
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0

# The exact output fields we need (matches the 22-column output)
OUTPUT_FIELDS = [
    "dosage", "phase", "trial_title", "trial_study_type", "trial_size",
    "trial_location", "trial_start_date", "trial_completion_date",
    "phase_status", "hba1c_change_pct", "hba1c_duration",
    "weight_change_pct", "weight_duration", "alt_reduction_pct",
    "alt_duration", "mash_resolution_pct", "mash_duration", "company_name",
]

_print_lock = threading.Lock()


def _gemini_call(session, url, prompt, timeout=180):
    """Gemini API call with google_search tool and exponential backoff."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192},
    }
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(url, data=json.dumps(body), timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                with _print_lock:
                    print(f"    Rate limit/server error ({r.status_code}) – "
                          f"waiting {backoff:.0f}s (attempt {attempt}/{MAX_RETRIES})",
                          file=sys.stderr)
                time.sleep(backoff); backoff *= 2; continue
            r.raise_for_status()
            parts = (r.json().get("candidates") or [{}])[0] \
                .get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                with _print_lock:
                    print(f"    Gemini call failed: {exc}", file=sys.stderr)
                return ""
            time.sleep(backoff); backoff *= 2
    return ""


def _parse_json(text):
    """Parse JSON from Gemini response, handling markdown fences and truncation."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.M)
    text = re.sub(r"```\s*$", "", text, flags=re.M).strip()

    # Find JSON array or object
    m = re.search(r"[\[{].*[\]}]", text, flags=re.DOTALL)
    if not m:
        return None

    raw = m.group(0)

    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: json_repair library
    try:
        from json_repair import repair_json
        result = repair_json(raw, return_objects=True)
        if result is not None:
            return result
    except Exception:
        pass

    # Attempt 3: raw_decode (handles "Extra data" after valid JSON)
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(raw)
        return obj
    except Exception:
        pass

    return None


def _build_prompt(drug: str, batch: List[Dict[str, Any]]) -> str:
    """Build the Step 2 extraction prompt — mirrors gemini_extractor.py exactly."""

    trial_lines = []
    for row in batch:
        tid = row.get("trial_id", "")
        url = row.get("source_url", "")
        trial_lines.append(f"- {tid} (Source: {url})")

    return f"""You are a clinical data extraction engine with access to Google Search.

Task: Extract structured clinical trial data for these {len(batch)} {drug} trials.

For EACH trial below, go to the Source URL provided and extract data
ONLY from what is displayed on that specific registry page.
Do NOT use PubMed, news articles, or any other external source.

TRIALS TO EXTRACT:
{chr(10).join(trial_lines)}

For EACH trial, extract these fields from the registry page:
- trial_id: (keep exactly as provided above)
- trial_title: The FULL official title of the trial as listed on the registry page. Use "N/A" only if completely unavailable.
- dosage: The PRIMARY or HIGHEST dosage tested. Report ONLY ONE dosage value (e.g., "2.4 mg OW", "1.0 mg OW", "14 mg QD"). If multiple doses were tested, choose the highest dose. Format: "[amount] [unit] [frequency]"
- phase: Trial phase number (e.g., "3", "2", "3b", "4"). Just the number.
- trial_study_type: Type of trial — must be exactly one of: "Interventional", "Observational", "Expanded Access". Use "N/A" only if completely unavailable.
- trial_size: Total enrollment number (integer)
- trial_location: Countries where trial was conducted (comma-separated)
- trial_start_date: Study start date in format "YYYY-MM-DD" or "YYYY-MM"
- trial_completion_date: Primary completion date in format "YYYY-MM-DD" or "YYYY-MM"
- phase_status: "Completed", "Active", "Recruiting", "Terminated", "Withdrawn"
- hba1c_change_pct: HbA1c reduction in percentage points (positive number, e.g., "1.8"). Use "N/A" if not reported on the registry page.
- hba1c_duration: Timepoint for HbA1c measurement (e.g., "26 wk", "52 wk"). Use "N/A" if not reported.
- weight_change_pct: Body weight loss percentage (positive number, e.g., "15.2"). Use "N/A" if not reported on the registry page.
- weight_duration: Timepoint for weight measurement (e.g., "68 wk", "104 wk"). Use "N/A" if not reported.
- alt_reduction_pct: ALT enzyme reduction percentage (e.g., "35"). Use "N/A" if not reported on the registry page.
- alt_duration: Timepoint for ALT measurement. Use "N/A" if not reported.
- mash_resolution_pct: MASH/NASH resolution rate (e.g., "59"). Use "N/A" if not a MASH trial or not reported.
- mash_duration: Timepoint for MASH assessment. Use "N/A" if not reported.
- company_name: Sponsor company name as shown on the registry page
- source_url: (keep exactly as provided above)

CRITICAL RULES:
- Include ALL {len(batch)} trials even if data is incomplete
- Use "N/A" for fields that are genuinely not available on the registry page
- DO NOT fabricate or hallucinate any values — if it's not on the page, say "N/A"
- Report reductions as positive numbers (weight loss of 15% → "15")
- A few trials may not have results posted — that is OK, report "N/A" for those fields

Return JSON:
{{
  "trials": [
    {{
      "trial_id": "...",
      "trial_title": "...",
      "dosage": "...",
      "phase": "...",
      "trial_study_type": "...",
      "trial_size": 0,
      "trial_location": "...",
      "trial_start_date": "...",
      "trial_completion_date": "...",
      "phase_status": "...",
      "hba1c_change_pct": "...",
      "hba1c_duration": "...",
      "weight_change_pct": "...",
      "weight_duration": "...",
      "alt_reduction_pct": "...",
      "alt_duration": "...",
      "mash_resolution_pct": "...",
      "mash_duration": "...",
      "company_name": "...",
      "source_url": "..."
    }}
  ]
}}"""


def fill_columns(drug: str, rows: List[Dict[str, Any]],
                 api_key: Optional[str] = None,
                 model: str = DEFAULT_MODEL,
                 workers: int = 3) -> List[Dict[str, Any]]:
    """Fill missing columns for all rows using Gemini + Google Search.

    Only fills EMPTY fields — never overwrites existing data from APIs.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [Gemini fill] no GEMINI_API_KEY – columns left as-is.",
              file=sys.stderr)
        return rows

    if not rows:
        return rows

    url = API_URL.format(model=model)
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    })

    # Build batches
    batches = []
    for i in range(0, len(rows), BATCH_SIZE):
        batches.append((i, rows[i:i + BATCH_SIZE]))

    print(f"  [Gemini fill] Extracting data for {len(rows)} trials in "
          f"{len(batches)} batches (batch size {BATCH_SIZE}) ...",
          file=sys.stderr)

    def process_batch(batch_start, batch):
        # Stagger to avoid rate limits
        stagger = (batch_start // BATCH_SIZE) % workers
        if stagger > 0:
            time.sleep(stagger * 0.5)

        prompt = _build_prompt(drug, batch)
        raw = _gemini_call(session, url, prompt)
        data = _parse_json(raw)
        if not data:
            return batch_start, []

        if isinstance(data, dict):
            trials = data.get("trials", [])
        elif isinstance(data, list):
            trials = data
        else:
            trials = []

        return batch_start, [t for t in trials if isinstance(t, dict)]

    # Process batches with controlled concurrency
    results_map = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(process_batch, idx, batch): idx
                   for idx, batch in batches}
        done = 0
        for fut in as_completed(futures):
            batch_start, gemini_trials = fut.result()
            results_map[batch_start] = gemini_trials
            done += 1
            with _print_lock:
                print(f"  [Gemini fill] batch {done}/{len(batches)} "
                      f"→ {len(gemini_trials)} trial(s) extracted",
                      file=sys.stderr)

    # Merge Gemini results back into rows (only fill empty fields)
    filled_count = 0
    for batch_start, batch in batches:
        gemini_trials = results_map.get(batch_start, [])

        # Index Gemini results by trial_id for matching
        gemini_by_id = {}
        for gt in gemini_trials:
            tid = str(gt.get("trial_id", "")).strip()
            if tid:
                gemini_by_id[tid] = gt

        for row in batch:
            tid = str(row.get("trial_id", "")).strip()
            gt = gemini_by_id.get(tid)
            if not gt:
                continue

            # Fill only empty/missing fields — never overwrite existing data
            for field in OUTPUT_FIELDS:
                current = str(row.get(field, "") or "").strip()
                new_val = str(gt.get(field, "") or "").strip()

                if (not current or current.lower() in ("", "n/a", "none")) and \
                   new_val and new_val.lower() not in ("", "n/a", "none", "null"):
                    row[field] = new_val
                    filled_count += 1

    print(f"  [Gemini fill] Done — filled {filled_count} empty fields "
          f"across {len(rows)} trials", file=sys.stderr)
    return rows