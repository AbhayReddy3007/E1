#!/usr/bin/env python3
"""
gemini_search_trials.py – Gemini + Google Search fallback for blocked registries.

Uses Gemini 2.5 Flash with google_search grounding (no responseSchema — they
conflict, causing a 400). Instead we ask for JSON in the prompt and parse it.
"""

from __future__ import annotations
import json, os, re, sys, time
from typing import Any, Dict, List, Optional
import requests

from registry_common import (
    ALLOWED_SOURCES, SRC_ANZCTR, SRC_CHICTR, SRC_CRIS, SRC_CTRI,
    SRC_JRCT, SRC_REBEC, UNIFIED_COLUMNS, blank_row, clean, make_session,
)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"

REGISTRY_SITES = {
    SRC_CTRI: "ctri.nic.in",
    SRC_JRCT: "jrct.mhlw.go.jp",
    SRC_ANZCTR: "anzctr.org.au",
    SRC_CHICTR: "chictr.org.cn",
    SRC_CRIS: "cris.nih.go.kr",
    SRC_REBEC: "ensaiosclinicos.gov.br",
}

REGISTRY_FULL_NAMES = {
    SRC_CTRI: "Clinical Trials Registry - India (CTRI) at ctri.nic.in",
    SRC_JRCT: "Japan Registry of Clinical Trials (jRCT) at jrct.mhlw.go.jp",
    SRC_ANZCTR: "Australian New Zealand Clinical Trials Registry (ANZCTR) at anzctr.org.au",
    SRC_CHICTR: "Chinese Clinical Trial Registry (ChiCTR) at chictr.org.cn",
    SRC_CRIS: "Clinical Research Information Service (CRIS) at cris.nih.go.kr",
    SRC_REBEC: "Brazilian Clinical Trials Registry (ReBEC) at ensaiosclinicos.gov.br",
}

PROMPT_TEMPLATE = """Search for all clinical trials of the drug "{drug}" registered on {registry_name}.

Search Google for: site:{site} "{drug}"
Also search for: {drug} clinical trial {site}

For EACH trial you find, extract these fields:
- trial_id (the registry's own ID, e.g. CTRI/2023/04/052053, jRCT2031200001, ACTRN12621000123456, ChiCTR2300078000, KCT0009234, RBR-6qvdftm)
- title
- status (recruitment status)
- phase
- study_type
- conditions (disease/condition studied)
- interventions (all interventions including dosage)
- drug_names
- dosage (dose amounts mentioned)
- sponsor
- countries
- target_enrollment (planned number of participants)
- actual_enrollment
- age_min
- age_max
- gender
- primary_outcome
- secondary_outcome
- start_date
- completion_date
- registration_date
- results_available (Yes/No)
- findings (any results mentioned)
- url (direct link to the trial page on the registry)

RULES:
1. ONLY return trials you actually find in search results. NEVER invent trials.
2. Every trial MUST have a real registry ID.
3. If you find NO trials, return exactly: []
4. Return ONLY a JSON array, no markdown backticks, no explanation before or after.

Example format:
[{{"trial_id": "CTRI/2023/04/052053", "title": "...", "status": "Completed", "phase": "Phase 3", "url": "https://ctri.nic.in/..."}}]"""


class GeminiTrialSearcher:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 timeout: int = 180, retries: int = 3):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.url = API_URL.format(model=model)
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        })

    def search_registry(self, drug: str, source: str) -> List[Dict[str, Any]]:
        registry_name = REGISTRY_FULL_NAMES.get(source, source)
        site = REGISTRY_SITES.get(source, "")

        prompt = PROMPT_TEMPLATE.format(
            drug=drug, registry_name=registry_name, site=site)

        # google_search tool CANNOT be combined with responseSchema;
        # we ask for JSON in the prompt instead
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 8192,
            },
        }

        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.post(self.url, data=json.dumps(body),
                                      timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                if r.status_code == 400:
                    # Log the actual error for debugging
                    err = r.text[:500]
                    print(f"    [Gemini] 400 response: {err}", file=sys.stderr)
                    raise requests.HTTPError(f"HTTP 400: {err[:200]}")
                r.raise_for_status()
                payload = r.json()

                cands = payload.get("candidates") or []
                if not cands:
                    return []

                parts = (cands[0].get("content") or {}).get("parts") or []
                raw = ""
                for p in parts:
                    if p.get("text"):
                        raw += p["text"]
                raw = raw.strip()

                # Extract JSON array from the response
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.M)
                raw = re.sub(r"```\s*$", "", raw, flags=re.M)
                raw = raw.strip()

                # Find the JSON array in the response
                m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
                if not m:
                    if "[]" in raw or "no trial" in raw.lower():
                        return []
                    return []

                trials = json.loads(m.group(0))
                if isinstance(trials, list):
                    return trials
                return []

            except json.JSONDecodeError:
                return []
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)

        print(f"    [Gemini search] failed for {source}: {last_exc}",
              file=sys.stderr)
        return []


def _validate_trial_id(trial_id: str, source: str) -> bool:
    tid = clean(trial_id)
    if not tid or len(tid) < 5:
        return False
    patterns = {
        SRC_CTRI: r"CTRI/\d{4}/\d{2,3}/\d+",
        SRC_JRCT: r"jRCT[a-zA-Z0-9]*\d{6,}",
        SRC_ANZCTR: r"ACTRN\d{10,}",
        SRC_CHICTR: r"ChiCTR[-a-zA-Z0-9]*\d+",
        SRC_CRIS: r"KCT\d{5,}",
        SRC_REBEC: r"RBR-[0-9a-zA-Z]+",
    }
    pattern = patterns.get(source)
    if pattern:
        return bool(re.search(pattern, tid, re.I))
    return True


def _to_row(trial: Dict[str, Any], source: str) -> Dict[str, Any]:
    row = blank_row(source)
    for col in UNIFIED_COLUMNS:
        if col in trial and trial[col]:
            row[col] = clean(trial[col])
    # Also capture fields not in unified schema
    for k, v in trial.items():
        if k not in row and clean(v):
            row[k] = clean(v)
    row["source"] = source
    return row


def search_missing_registries(
    drug: str,
    missing_sources: List[str],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, List[Dict[str, Any]]]:
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [Gemini search] no GEMINI_API_KEY – cannot search.",
              file=sys.stderr)
        return {}

    searchable = [s for s in missing_sources if s in REGISTRY_SITES]
    if not searchable:
        return {}

    searcher = GeminiTrialSearcher(api_key, model)
    results: Dict[str, List[Dict[str, Any]]] = {}

    for source in searchable:
        print(f"  [Gemini search] searching {source} for '{drug}' ...",
              file=sys.stderr)
        try:
            trials = searcher.search_registry(drug, source)
        except Exception as exc:
            print(f"  [Gemini search] {source} failed: {exc}", file=sys.stderr)
            continue

        valid_rows: List[Dict[str, Any]] = []
        rejected = 0
        for trial in trials:
            tid = clean(trial.get("trial_id", ""))
            if not _validate_trial_id(tid, source):
                rejected += 1
                continue
            valid_rows.append(_to_row(trial, source))

        results[source] = valid_rows
        found = len(valid_rows)
        msg = f"  [Gemini search] {source}: {found} trial(s)"
        if rejected:
            msg += f" ({rejected} rejected – invalid IDs)"
        print(msg, file=sys.stderr)
        time.sleep(1)

    return results