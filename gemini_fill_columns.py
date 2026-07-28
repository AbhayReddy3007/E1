#!/usr/bin/env python3
"""
gemini_fill_columns.py – Gemini Step 2 extraction for all trials.

Follows the reference gemini_extractor.py pattern exactly.
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

OUTPUT_FIELDS = [
    "dosage", "phase", "trial_title", "trial_study_type", "trial_size",
    "trial_location", "trial_start_date", "trial_completion_date",
    "phase_status", "hba1c_change_pct", "hba1c_duration",
    "weight_change_pct", "weight_duration", "alt_reduction_pct",
    "alt_duration", "mash_resolution_pct", "mash_duration", "company_name",
]

_print_lock = threading.Lock()


def _gemini_call(session, url, prompt, timeout=180):
    """Gemini call with google_search. Returns full raw response text."""
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
                    print(f"    [{r.status_code}] waiting {backoff:.0f}s "
                          f"(attempt {attempt}/{MAX_RETRIES})", file=sys.stderr)
                time.sleep(backoff); backoff *= 2; continue
            r.raise_for_status()

            payload = r.json()
            cands = payload.get("candidates") or []
            if not cands:
                return ""

            # Collect ALL text parts (grounding wraps text in multiple parts)
            all_text = ""
            for part in (cands[0].get("content") or {}).get("parts") or []:
                if part.get("text"):
                    all_text += part["text"]
            return all_text.strip()

        except Exception as exc:
            if attempt == MAX_RETRIES:
                with _print_lock:
                    print(f"    Gemini call failed: {exc}", file=sys.stderr)
                return ""
            time.sleep(backoff); backoff *= 2
    return ""


def _parse_json(text: str) -> Optional[Any]:
    """
    Robustly extract JSON from Gemini response.
    Handles: markdown fences, grounding metadata, truncation, extra text.
    """
    if not text:
        return None

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # The reference safe_json_parse: find first { or [
    json_start = -1
    for i, c in enumerate(text):
        if c in "{[":
            json_start = i
            break
    if json_start < 0:
        return None
    text = text[json_start:]

    # Attempt 1: json_repair (reference uses this first)
    try:
        from json_repair import repair_json
        result = repair_json(text, return_objects=True)
        if result is not None:
            return result
    except Exception:
        pass

    # Attempt 2: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 3: raw_decode to handle trailing garbage
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj
    except Exception:
        pass

    # Attempt 4: find the last complete } and try up to there
    last_brace = text.rfind("}")
    if last_brace > 0:
        try:
            return json.loads(text[:last_brace + 1])
        except Exception:
            pass

    return None


def _extract_trials(data: Any) -> List[dict]:
    """Extract trials list from parsed JSON — handles dict or list."""
    if data is None:
        return []
    if isinstance(data, list):
        return [t for t in data if isinstance(t, dict)]
    if isinstance(data, dict):
        # Could be {"trials": [...]} or a single trial object
        if "trials" in data:
            t = data["trials"]
            return [x for x in t if isinstance(x, dict)] if isinstance(t, list) else []
        # Single trial object
        if any(k in data for k in ("trial_id", "Trial ID", "Molecule", "dosage")):
            return [data]
    return []


def _normalise_trial_id(tid: str) -> str:
    """Strip program name suffixes like 'NCT123 (STEP 1)' → 'NCT123'."""
    return tid.split("(")[0].strip().split(" ")[0].strip() if tid else ""


def _build_prompt(drug: str, batch: List[Dict[str, Any]]) -> str:
    trial_lines = []
    for row in batch:
        tid = row.get("trial_id", "")
        url = row.get("source_url", "")
        title = row.get("trial_title", "")[:60]
        trial_lines.append(f"- Trial ID: {tid} | URL: {url} | Title: {title}")

    return f"""You are a clinical data extraction engine. Extract data for these {len(batch)} {drug} trials.

For EACH trial, navigate to the Source URL provided and extract information ONLY from that registry page.
Do NOT use PubMed, Wikipedia, or any other source — only the registry page at the URL given.

TRIALS:
{chr(10).join(trial_lines)}

For EACH trial, extract:
- trial_id: (copy exactly from Trial ID above)
- trial_title: Full official title from the registry page
- dosage: Highest dosage tested (e.g., "2.4 mg OW", "14 mg QD"). "N/A" if not available.
- phase: Phase number only (e.g., "3", "2", "1"). "N/A" if not available.
- trial_study_type: "Interventional", "Observational", or "Expanded Access". "N/A" if not available.
- trial_size: Total enrollment number (integer). "N/A" if not available.
- trial_location: Countries where trial was conducted. "N/A" if not available.
- trial_start_date: Study start date (YYYY-MM-DD or YYYY-MM). "N/A" if not available.
- trial_completion_date: Primary completion date (YYYY-MM-DD or YYYY-MM). "N/A" if not available.
- phase_status: "Completed", "Active", "Recruiting", "Terminated", or "N/A"
- hba1c_change_pct: HbA1c reduction % as positive number (e.g., "1.8"). "N/A" if not on page.
- hba1c_duration: Timepoint (e.g., "26 wk"). "N/A" if not on page.
- weight_change_pct: Body weight loss % as positive number (e.g., "15.2"). "N/A" if not on page.
- weight_duration: Timepoint (e.g., "68 wk"). "N/A" if not on page.
- alt_reduction_pct: ALT reduction % (e.g., "35"). "N/A" if not on page.
- alt_duration: Timepoint. "N/A" if not on page.
- mash_resolution_pct: MASH/NASH resolution % (e.g., "59"). "N/A" if not on page.
- mash_duration: Timepoint. "N/A" if not on page.
- company_name: Sponsor/company name from the registry page.
- source_url: (copy exactly from URL above)

IMPORTANT:
- Return ALL {len(batch)} trials
- If a field is NOT present on the registry page, use "N/A" — do not guess
- Do not fabricate any values
- Return reductions as positive numbers

Return ONLY a JSON object with this structure:
{{
  "trials": [
    {{
      "trial_id": "...",
      "trial_title": "...",
      "dosage": "...",
      "phase": "...",
      "trial_study_type": "...",
      "trial_size": "...",
      "trial_location": "...",
      "trial_start_date": "...",
      "trial_completion_date": "...",
      "phase_status": "...",
      "hba1c_change_pct": "N/A",
      "hba1c_duration": "N/A",
      "weight_change_pct": "N/A",
      "weight_duration": "N/A",
      "alt_reduction_pct": "N/A",
      "alt_duration": "N/A",
      "mash_resolution_pct": "N/A",
      "mash_duration": "N/A",
      "company_name": "...",
      "source_url": "..."
    }}
  ]
}}"""


def fill_columns(drug: str, rows: List[Dict[str, Any]],
                 api_key: Optional[str] = None,
                 model: str = DEFAULT_MODEL,
                 workers: int = 3) -> List[Dict[str, Any]]:
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [Gemini fill] no GEMINI_API_KEY – skipping.", file=sys.stderr)
        return rows
    if not rows:
        return rows

    url = API_URL.format(model=model)
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    })

    batches = [(i, rows[i:i + BATCH_SIZE]) for i in range(0, len(rows), BATCH_SIZE)]
    print(f"  [Gemini fill] {len(rows)} trials in {len(batches)} batches ...",
          file=sys.stderr)

    def process_batch(batch_start, batch):
        stagger = (batch_start // BATCH_SIZE) % max(1, workers)
        if stagger > 0:
            time.sleep(stagger * 0.5)

        raw_text = _gemini_call(session, url, _build_prompt(drug, batch))
        if not raw_text:
            return batch_start, []

        data = _parse_json(raw_text)
        trials = _extract_trials(data)

        if not trials:
            # Log the raw response for diagnosis
            with _print_lock:
                print(f"    [parse fail] raw response preview: "
                      f"{raw_text[:200]!r}", file=sys.stderr)

        return batch_start, trials

    results_map = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(process_batch, idx, batch): idx
                   for idx, batch in batches}
        done = 0
        for fut in as_completed(futures):
            batch_start, trials = fut.result()
            results_map[batch_start] = trials
            done += 1
            with _print_lock:
                print(f"  [Gemini fill] batch {done}/{len(batches)} "
                      f"→ {len(trials)} extracted", file=sys.stderr)

    # Merge back: build lookup index with normalised IDs
    filled_count = 0
    for batch_start, batch in batches:
        gemini_trials = results_map.get(batch_start, [])

        # Index by both raw and normalised trial_id
        gemini_by_id: Dict[str, dict] = {}
        for gt in gemini_trials:
            raw_tid = str(gt.get("trial_id", "") or "").strip()
            if raw_tid:
                gemini_by_id[raw_tid] = gt
                # Also index by normalised (strips program name suffix)
                norm = _normalise_trial_id(raw_tid)
                if norm and norm != raw_tid:
                    gemini_by_id[norm] = gt

        for row in batch:
            raw_tid = str(row.get("trial_id", "") or "").strip()
            norm_tid = _normalise_trial_id(raw_tid)

            gt = gemini_by_id.get(raw_tid) or gemini_by_id.get(norm_tid)
            if not gt:
                continue

            for field in OUTPUT_FIELDS:
                current = str(row.get(field, "") or "").strip()
                new_val = str(gt.get(field, "") or "").strip()
                # Only fill if current is empty/N/A AND new value is real
                if (not current or current.lower() in ("", "n/a", "none", "null")) \
                   and new_val and new_val.lower() not in ("", "n/a", "none", "null"):
                    row[field] = new_val
                    filled_count += 1

    print(f"  [Gemini fill] Done — filled {filled_count} empty fields "
          f"across {len(rows)} trials", file=sys.stderr)
    return rows