#!/usr/bin/env python3
"""
gemini_search_trials.py
=======================

Uses Gemini 2.5 Flash + Google Search grounding to find clinical trials from
registries that block automated scraping (CTRI, jRCT, ANZCTR, ChiCTR, CRIS,
ReBEC).

Gemini searches the web for trials on a given registry, returns structured
JSON, and we map it into the unified schema. The `source` column is set to
the registry name (never "Gemini").

This is a FALLBACK — it only runs for registries that returned 0 rows from
direct scraping.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

from registry_common import (
    ALLOWED_SOURCES, SRC_ANZCTR, SRC_CHICTR, SRC_CRIS, SRC_CTRI,
    SRC_JRCT, SRC_REBEC, UNIFIED_COLUMNS, blank_row, clean, join,
    make_session,
)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"

# Registry site URLs for Gemini to search
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

SYSTEM_PROMPT = """You are a clinical trial search tool. You use Google Search to find
clinical trials registered on specific national trial registries.

RULES:
1. Search ONLY the specific registry website you are asked about.
2. Return ONLY trials you actually find via search. Never invent trials.
3. Every trial MUST have a real registry ID (e.g., CTRI/2023/04/052053, jRCT2031200001,
   ACTRN12621000123456, ChiCTR2300078000, KCT0009234, RBR-6qvdftm).
4. If you cannot find any trials for a drug on that registry, return an empty array [].
5. Include the source URL where you found each trial.
6. Extract as many fields as possible from the search results.
7. Return ONLY valid JSON, no markdown, no backticks, no explanation."""

TRIAL_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "trial_id": {"type": "STRING", "description": "Registry ID (e.g. CTRI/2023/04/052053)"},
            "title": {"type": "STRING"},
            "public_title": {"type": "STRING"},
            "status": {"type": "STRING", "description": "Recruitment status"},
            "phase": {"type": "STRING"},
            "study_type": {"type": "STRING"},
            "study_design": {"type": "STRING"},
            "conditions": {"type": "STRING"},
            "interventions": {"type": "STRING"},
            "drug_names": {"type": "STRING"},
            "sponsor": {"type": "STRING"},
            "countries": {"type": "STRING"},
            "target_enrollment": {"type": "STRING"},
            "actual_enrollment": {"type": "STRING"},
            "age_min": {"type": "STRING"},
            "age_max": {"type": "STRING"},
            "gender": {"type": "STRING"},
            "inclusion_criteria": {"type": "STRING"},
            "exclusion_criteria": {"type": "STRING"},
            "primary_outcome": {"type": "STRING"},
            "secondary_outcome": {"type": "STRING"},
            "start_date": {"type": "STRING"},
            "completion_date": {"type": "STRING"},
            "registration_date": {"type": "STRING"},
            "results_available": {"type": "STRING"},
            "findings": {"type": "STRING", "description": "Any results/findings mentioned"},
            "url": {"type": "STRING", "description": "Direct URL to the trial page"},
        },
        "required": ["trial_id", "title", "url"],
    },
}


class GeminiTrialSearcher:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 timeout: int = 120, retries: int = 3):
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
        """
        Ask Gemini to search a specific registry for trials of `drug`.
        Returns list of dicts with trial fields.
        """
        registry_name = REGISTRY_FULL_NAMES.get(source, source)
        site = REGISTRY_SITES.get(source, "")

        prompt = (
            f"Search for all clinical trials of the drug \"{drug}\" registered on "
            f"{registry_name}.\n\n"
            f"Search the site {site} specifically. Also try searching Google for: "
            f'site:{site} "{drug}"\n\n'
            f"For each trial you find, extract: trial ID, title, status, phase, "
            f"sponsor, conditions, interventions, start date, enrollment, and "
            f"the direct URL to the trial page.\n\n"
            f"Return a JSON array of trial objects. If you find no trials, "
            f"return an empty array []."
        )

        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json",
                "responseSchema": TRIAL_SCHEMA,
            },
        }

        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.post(self.url, data=json.dumps(body),
                                      timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                payload = r.json()

                cands = payload.get("candidates") or []
                if not cands:
                    return []

                parts = (cands[0].get("content") or {}).get("parts") or []
                raw = "".join(p.get("text", "") for p in parts).strip()
                raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()

                if not raw or raw == "[]":
                    return []

                trials = json.loads(raw)
                if isinstance(trials, list):
                    return trials
                return []

            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)

        print(f"    [Gemini search] failed for {source}: {last_exc}",
              file=sys.stderr)
        return []


def _validate_trial_id(trial_id: str, source: str) -> bool:
    """Check that the trial ID looks real for the given registry."""
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
    """Map Gemini's JSON output to a unified row."""
    row = blank_row(source)
    # Map directly – field names match unified schema
    for col in UNIFIED_COLUMNS:
        if col in trial and trial[col]:
            row[col] = clean(trial[col])
    row["source"] = source
    return row


def search_missing_registries(
    drug: str,
    missing_sources: List[str],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each source in `missing_sources`, use Gemini + Google Search to find
    trials. Returns {source_name: [rows]}.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [Gemini search] no GEMINI_API_KEY – cannot search missing "
              "registries.", file=sys.stderr)
        return {}

    # Only search registries we have site URLs for
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
            print(f"  [Gemini search] {source} failed: {exc}",
                  file=sys.stderr)
            continue

        # Validate and convert
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

        time.sleep(1)  # rate limit politeness

    return results