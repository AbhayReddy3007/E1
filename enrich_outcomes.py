#!/usr/bin/env python3
"""
enrich_outcomes.py – Clinical trial outcome enrichment module.

For each trial row produced by main1.py, searches for published outcome data
across four clinical endpoints (HbA1c, body weight, ALT, MASH) and uses
Gemini 3 Flash Preview to extract structured results + confidence scores.

Entry point:
    enrich_trial_outcomes(rows, molecule) -> list[dict]

Requires:
    - GEMINI_API_KEY in a .env file (or set as an environment variable)
    - requests, python-dotenv

    Create a .env file in the same directory as this script:
        GEMINI_API_KEY=your_key_here
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # falls back to os.environ as-is if python-dotenv is not installed

# ── configuration ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash-preview-05-20"
GEMINI_URL     = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

ENDPOINTS = [
    {
        "key":    "hba1c",
        "label":  "HbA1c",
        "col_pct":  "hba1c_change_pct",
        "col_dur":  "hba1c_duration",
        "col_rat":  "hba1c_rationale",
        "col_conf": "hba1c_confidence",
        "search_terms": [
            "HbA1c", "A1c", "glycated hemoglobin", "glycaemic control",
        ],
    },
    {
        "key":    "weight",
        "label":  "Body weight",
        "col_pct":  "weight_change_pct",
        "col_dur":  "weight_duration",
        "col_rat":  "weight_rationale",
        "col_conf": "weight_confidence",
        "search_terms": [
            "body weight", "weight loss", "weight change", "BMI reduction",
        ],
    },
    {
        "key":    "alt",
        "label":  "ALT (liver enzyme)",
        "col_pct":  "alt_reduction_pct",
        "col_dur":  "alt_duration",
        "col_rat":  "alt_rationale",
        "col_conf": "alt_confidence",
        "search_terms": [
            "ALT", "alanine aminotransferase", "liver enzyme",
        ],
    },
    {
        "key":    "mash",
        "label":  "MASH / NASH",
        "col_pct":  "mash_change_pct",
        "col_dur":  "mash_duration",
        "col_rat":  "mash_rationale",
        "col_conf": "mash_confidence",
        "search_terms": [
            "MASH", "NASH", "steatohepatitis", "MASH resolution",
            "fibrosis improvement",
        ],
    },
]

TAG = "[ENRICH]"

# simple in-memory cache keyed by trial_id
_CACHE: Dict[str, Dict[str, Any]] = {}

# ── helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"{TAG} {msg}", file=sys.stderr)


def _cache_key(trial_id: str) -> str:
    return trial_id.strip().upper()


def _gemini_call(prompt: str, system: str = "",
                 retries: int = 3, backoff: float = 2.0) -> Optional[str]:
    """
    Call Gemini 3 Flash Preview via REST and return the text response.
    Returns None on persistent failure.
    """
    if not GEMINI_API_KEY:
        _log("GEMINI_API_KEY not set – skipping LLM extraction.")
        return None

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    headers = {"Content-Type": "application/json"}
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 429:
                wait = backoff * attempt
                _log(f"  Rate-limited (429), retrying in {wait:.0f}s …")
                time.sleep(wait)
                continue
            if not resp.ok:
                _log(f"  Gemini HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(backoff * attempt)
                continue
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts)
            return None
        except Exception as exc:
            _log(f"  Gemini call error (attempt {attempt}): {exc}")
            time.sleep(backoff * attempt)

    return None


def _extract_json_from_response(text: str) -> Optional[Dict]:
    """Parse JSON from a Gemini response that may be wrapped in markdown fences."""
    if not text:
        return None
    # strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # try to find first { … } block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Step 1: Evidence gathering via web search ─────────────────────────────────

def _build_search_queries(row: Dict[str, str], molecule: str,
                          endpoint: Dict) -> List[str]:
    """
    Build a list of search queries to find outcome data for a given
    trial + endpoint combination.
    """
    trial_id    = row.get("trial_id", "")
    acronym     = row.get("acronym", "")
    company     = row.get("company_name", "")
    phase       = row.get("phase", "")
    ep_label    = endpoint["label"]
    ep_terms    = endpoint["search_terms"]

    queries: List[str] = []

    # primary: trial_id + endpoint
    if trial_id:
        queries.append(f"{trial_id} {ep_terms[0]} results")

    # acronym-based (often the public-facing name)
    if acronym and acronym != trial_id:
        queries.append(f"{acronym} {molecule} {ep_label} results")

    # molecule + endpoint + phase
    if phase:
        queries.append(f"{molecule} {ep_terms[0]} phase {phase} trial results")

    # company press release style
    if company:
        queries.append(
            f"{company} {molecule} {ep_label} clinical trial results"
        )

    # PubMed-style
    queries.append(f"{molecule} {ep_terms[0]} clinical trial PubMed")

    return queries[:4]  # cap at 4 queries per endpoint


def _search_pubmed(query: str, session: requests.Session,
                   max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search PubMed via the E-utilities API and return title + abstract snippets.
    """
    results: List[Dict[str, str]] = []
    try:
        # search
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        resp = session.get(search_url, params={
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json",
        }, timeout=20)
        if not resp.ok:
            return results
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return results

        # fetch abstracts
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        resp2 = session.get(fetch_url, params={
            "db": "pubmed", "id": ",".join(ids),
            "rettype": "abstract", "retmode": "text",
        }, timeout=20)
        if resp2.ok:
            results.append({
                "source": "PubMed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{ids[0]}/",
                "text": resp2.text[:3000],
            })
    except Exception:
        pass
    return results


def _search_ctgov_results(trial_id: str,
                          session: requests.Session) -> List[Dict[str, str]]:
    """
    Check if ClinicalTrials.gov has posted results for a given NCT ID.
    """
    results: List[Dict[str, str]] = []
    if not trial_id or not trial_id.upper().startswith("NCT"):
        return results
    try:
        url = f"https://clinicaltrials.gov/api/v2/studies/{trial_id}"
        resp = session.get(url, params={
            "fields": "ResultsSection",
        }, timeout=20)
        if not resp.ok:
            return results
        data = resp.json()
        res_section = data.get("resultsSection")
        if res_section:
            # flatten to text for LLM consumption
            text = json.dumps(res_section, indent=2)[:4000]
            results.append({
                "source": "ClinicalTrials.gov Results",
                "url": f"https://clinicaltrials.gov/study/{trial_id}?tab=results",
                "text": text,
            })
    except Exception:
        pass
    return results


def gather_evidence(row: Dict[str, str], molecule: str,
                    endpoint: Dict,
                    session: requests.Session) -> List[Dict[str, str]]:
    """
    Gather evidence snippets for one trial × one endpoint.
    Returns a list of dicts with keys: source, url, text.
    """
    snippets: List[Dict[str, str]] = []

    # 1. ClinicalTrials.gov results section (if NCT)
    trial_id = row.get("trial_id", "")
    ctgov_res = _search_ctgov_results(trial_id, session)
    snippets.extend(ctgov_res)

    # 2. PubMed search
    queries = _build_search_queries(row, molecule, endpoint)
    for q in queries[:2]:  # limit PubMed calls
        pm = _search_pubmed(q, session, max_results=2)
        snippets.extend(pm)
        if pm:
            break  # got something, move on
        time.sleep(0.3)

    return snippets


# ── Step 2: Structured extraction via Gemini ──────────────────────────────────

EXTRACTION_SYSTEM = """\
You are a clinical data extraction assistant. You will be given snippets from
medical literature, registry results, or press releases about a clinical trial,
along with trial metadata.

Your task is to extract a specific clinical endpoint result. Respond ONLY with
a JSON object (no markdown fences, no commentary) with these keys:

  {
    "change_pct": <number or null>,
    "duration": "<string or null>",
    "rationale": "<string or null>"
  }

Rules:
- change_pct: the numeric percentage change or reduction reported for the
  endpoint (e.g. -1.2 for HbA1c, -8.5 for weight, 37 for MASH resolution
  rate, -40 for ALT reduction). Use null if NO concrete number is found.
  NEVER fabricate a number.
- duration: the timepoint at which the result was measured (e.g. "24 weeks",
  "52 weeks", "6 months"). Use null if not stated.
- rationale: a concise 1–3 sentence summary of the finding in your own words,
  noting the source type (e.g. "peer-reviewed publication", "registry results",
  "company press release"). Use null if no relevant data was found.

If the snippets contain NO data for the requested endpoint, return:
  {"change_pct": null, "duration": null, "rationale": null}
"""


def _build_extraction_prompt(row: Dict[str, str], molecule: str,
                             endpoint: Dict,
                             snippets: List[Dict[str, str]]) -> str:
    meta = (
        f"Trial ID: {row.get('trial_id', 'N/A')}\n"
        f"Acronym: {row.get('acronym', 'N/A')}\n"
        f"Molecule: {molecule}\n"
        f"Phase: {row.get('phase', 'N/A')}\n"
        f"Sponsor: {row.get('company_name', 'N/A')}\n"
        f"Title: {row.get('trial_title', 'N/A')}\n"
    )
    ep_info = (
        f"Endpoint to extract: {endpoint['label']}\n"
        f"Related terms: {', '.join(endpoint['search_terms'])}\n"
    )
    snippet_text = ""
    for i, s in enumerate(snippets, 1):
        snippet_text += (
            f"\n--- Snippet {i} (source: {s['source']}, "
            f"url: {s['url']}) ---\n{s['text']}\n"
        )
    if not snippet_text:
        snippet_text = "\n[No evidence snippets available.]\n"

    return (
        f"## Trial Metadata\n{meta}\n"
        f"## Endpoint\n{ep_info}\n"
        f"## Evidence Snippets\n{snippet_text}\n"
        f"Extract the structured result for {endpoint['label']}. "
        f"Respond with JSON only."
    )


def extract_endpoint(row: Dict[str, str], molecule: str,
                     endpoint: Dict,
                     snippets: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Use Gemini to extract structured outcome data for one endpoint.
    Returns dict with keys: change_pct, duration, rationale.
    """
    empty = {"change_pct": None, "duration": None, "rationale": None}

    if not snippets:
        return empty

    prompt = _build_extraction_prompt(row, molecule, endpoint, snippets)
    raw = _gemini_call(prompt, system=EXTRACTION_SYSTEM)
    parsed = _extract_json_from_response(raw)

    if parsed and isinstance(parsed, dict):
        return {
            "change_pct": parsed.get("change_pct"),
            "duration":   parsed.get("duration"),
            "rationale":  parsed.get("rationale"),
        }
    return empty


# ── Step 3: Validation & confidence scoring ───────────────────────────────────

VALIDATION_SYSTEM = """\
You are a clinical data validation assistant. You will be given an extracted
result for a clinical endpoint along with the original evidence snippets.

Evaluate the extraction and respond ONLY with a JSON object:

  {
    "confidence": "High" or "Low",
    "reason": "<brief explanation>"
  }

Assign "High" confidence when ALL of the following are true:
  1. The rationale text actually supports the change_pct value (no contradiction
     or fabricated number).
  2. The source is credible and specific (peer-reviewed journal or registry-
     posted results are strongest; company press releases are acceptable;
     secondary aggregators or vague mentions are weak).
  3. The finding is unambiguous — a concrete number tied to a specific
     timepoint, not inferred or extrapolated.

Assign "Low" confidence if ANY of the above conditions is not met, or if
change_pct is null.
"""


def validate_endpoint(extracted: Dict[str, Any],
                      snippets: List[Dict[str, str]],
                      endpoint: Dict) -> str:
    """
    Run a validation pass on extracted data. Returns 'High' or 'Low'.
    """
    # if nothing was extracted, automatically Low
    if extracted.get("change_pct") is None:
        return "Low"

    snippet_text = ""
    for i, s in enumerate(snippets, 1):
        snippet_text += (
            f"\n--- Snippet {i} (source: {s['source']}) ---\n"
            f"{s['text'][:1500]}\n"
        )

    prompt = (
        f"## Extracted Result for {endpoint['label']}\n"
        f"change_pct: {extracted['change_pct']}\n"
        f"duration: {extracted['duration']}\n"
        f"rationale: {extracted['rationale']}\n\n"
        f"## Original Evidence\n{snippet_text}\n\n"
        f"Validate this extraction. Respond with JSON only."
    )

    raw = _gemini_call(prompt, system=VALIDATION_SYSTEM)
    parsed = _extract_json_from_response(raw)

    if parsed and isinstance(parsed, dict):
        conf = parsed.get("confidence", "Low")
        if conf in ("High", "Low"):
            return conf
    return "Low"


# ── Step 4: Per-trial enrichment ──────────────────────────────────────────────

def _enrich_single_trial(row: Dict[str, str], molecule: str,
                         session: requests.Session) -> Dict[str, str]:
    """
    Enrich a single trial row with outcome data for all four endpoints.
    Modifies and returns the row dict in-place.
    """
    trial_id = row.get("trial_id", "unknown")
    ck = _cache_key(trial_id)

    # check cache
    if ck in _CACHE:
        _log(f"  {trial_id} → using cached results")
        cached = _CACHE[ck]
        for ep in ENDPOINTS:
            row[ep["col_pct"]]  = cached.get(ep["col_pct"], "")
            row[ep["col_dur"]]  = cached.get(ep["col_dur"], "")
            row[ep["col_rat"]]  = cached.get(ep["col_rat"], "")
            row[ep["col_conf"]] = cached.get(ep["col_conf"], "")
        return row

    results_cache: Dict[str, str] = {}

    for ep in ENDPOINTS:
        _log(f"  {trial_id} → {ep['label']} …")

        # Step 1: gather evidence
        try:
            snippets = gather_evidence(row, molecule, ep, session)
        except Exception as exc:
            _log(f"    evidence gathering failed: {exc}")
            snippets = []

        # Step 2: extract via LLM
        try:
            extracted = extract_endpoint(row, molecule, ep, snippets)
        except Exception as exc:
            _log(f"    extraction failed: {exc}")
            extracted = {"change_pct": None, "duration": None, "rationale": None}

        # Step 3: validate
        try:
            confidence = validate_endpoint(extracted, snippets, ep)
        except Exception as exc:
            _log(f"    validation failed: {exc}")
            confidence = "Low"

        # Step 4: merge into row
        pct_val = extracted.get("change_pct")
        row[ep["col_pct"]]  = str(pct_val) if pct_val is not None else ""
        row[ep["col_dur"]]  = extracted.get("duration") or ""
        row[ep["col_rat"]]  = extracted.get("rationale") or ""
        row[ep["col_conf"]] = confidence

        # store for cache
        results_cache[ep["col_pct"]]  = row[ep["col_pct"]]
        results_cache[ep["col_dur"]]  = row[ep["col_dur"]]
        results_cache[ep["col_rat"]]  = row[ep["col_rat"]]
        results_cache[ep["col_conf"]] = row[ep["col_conf"]]

        # throttle between endpoints
        time.sleep(0.5)

    _CACHE[ck] = results_cache
    return row


# ── public entry point ────────────────────────────────────────────────────────

def enrich_trial_outcomes(rows: List[Dict[str, str]],
                          molecule: str) -> List[Dict[str, str]]:
    """
    Enrich all trial rows with outcome data for HbA1c, weight, ALT, and MASH.

    Parameters
    ----------
    rows : list of dict
        Trial row dicts as produced by main1.py's fetch_all().
    molecule : str
        The molecule / drug name being searched.

    Returns
    -------
    list of dict
        The same rows, with outcome columns populated where data was found.
    """
    if not GEMINI_API_KEY:
        _log("GEMINI_API_KEY not set — skipping outcome enrichment.")
        _log("Set the environment variable to enable this feature.")
        return rows

    _log(f"Starting outcome enrichment for {len(rows)} trial(s) …")

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        ),
    })

    for i, row in enumerate(rows, 1):
        trial_id = row.get("trial_id", "?")
        _log(f"[{i}/{len(rows)}] Enriching {trial_id} …")
        try:
            _enrich_single_trial(row, molecule, session)
        except Exception as exc:
            _log(f"  FAILED for {trial_id}: {exc}")
            # leave outcome columns blank, set confidence to Low
            for ep in ENDPOINTS:
                row.setdefault(ep["col_pct"], "")
                row.setdefault(ep["col_dur"], "")
                row.setdefault(ep["col_rat"], "")
                row[ep["col_conf"]] = "Low"

        # throttle between trials
        time.sleep(1.0)

    _log(f"Enrichment complete for {len(rows)} trial(s).")
    return rows