#!/usr/bin/env python3
"""
enrich_ctgov_with_gemini.py
============================
Bridges the fast, authoritative registry scrapers — ctgov_trials.py
(ClinicalTrials.gov API v2) and euct_trials.py (EU CTIS + EudraCT,
via ctis_drug_trials.py / eudract_drug_trials.py) — with
gemini_extractor.py's Step 2 detail extraction (Gemini + Google Search —
used for efficacy fields that aren't in the raw registry data: HbA1c
change, weight loss %, ALT reduction, MASH resolution, dosage, company,
etc.).

Instead of letting gemini_extractor.py run its own Step 1 registry search,
this script hands Gemini the exact trial IDs already pulled from the
registry scrapers, jumps straight to Step 2 (get_trial_details), and
merges the enriched fields back onto the original rows — so nothing is
duplicated or re-discovered, only filled in.

Usage:
    python enrich_ctgov_with_gemini.py "semaglutide"
    python enrich_ctgov_with_gemini.py "semaglutide" --outdir results --max-records 50
    python enrich_ctgov_with_gemini.py "semaglutide" --sources ctgov          # US only
    python enrich_ctgov_with_gemini.py "semaglutide" --sources euct          # EU only
    python enrich_ctgov_with_gemini.py "semaglutide" --sources ctgov,euct    # both (default)

Requires (same as gemini_extractor.py):
    pip install google-genai aiohttp python-dotenv json-repair
    GOOGLE_API_KEY set in the environment or a .env file

Also requires ctgov_trials.py, euct_trials.py (and, for EU coverage,
ctis_drug_trials.py / eudract_drug_trials.py on PYTHONPATH) and
gemini_extractor.py to be importable. If a given source module can't be
imported, this script just skips that source with a warning rather than
failing outright. If registry_common.py is not available, this script
falls back to pandas for writing the output Excel file.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from typing import Any, Dict, List, Optional

import aiohttp

from gemini_extractor import (
    MAX_WORKERS,
    RATE_LIMIT_DELAY,
    get_trial_details,
    fetch_ctgov_dates,
)

# Our own batching size for the FIRST pass — deliberately smaller than
# gemini_extractor's own BATCH_SIZE (8). Packing 8 trials x ~20 fields,
# including numeric outcome data (HbA1c change, weight loss %, ALT
# reduction, MASH resolution) that usually has to be dug out of a results
# page or a published paper rather than read off the registry front page,
# into one Gemini call measurably dilutes how much research depth each
# individual trial gets. We don't touch the imported BATCH_SIZE constant
# since other callers of gemini_extractor.py may rely on it.
FIRST_PASS_BATCH_SIZE = 2

# Batch size for the automatic retry pass on trials that came back with no
# efficacy data — one trial per call, so the model can spend its full
# search budget on just that trial instead of splitting it 4-8 ways.
RETRY_BATCH_SIZE = 1

# The efficacy-specific fields we actually care about recovering. If ALL
# of these are still empty after the first pass, the trial is a retry
# candidate. (We deliberately don't retry on missing trial_title/phase/etc
# — those are usually genuinely unavailable or already covered by the
# registry scraper, and retrying for them wastes API calls.)
EFFICACY_ROW_FIELDS = [
    "dosage", "hba1c_change_pct", "weight_change_pct",
    "alt_reduction_pct", "mash_resolution_pct", "company_name",
]


def _efficacy_missing(row: Dict[str, Any]) -> bool:
    return not any(str(row.get(f, "")).strip() for f in EFFICACY_ROW_FIELDS)


# Core, non-efficacy fields that the registry scraper OR Gemini should
# normally be able to fill in. Unlike EFFICACY_ROW_FIELDS (which can be
# legitimately empty — no results published yet), these are almost always
# obtainable, so a row missing any of them is also worth a retry pass rather
# than being accepted as final.
CORE_ROW_FIELDS = [
    "trial_title", "phase", "trial_size", "trial_location",
    "phase_status", "trial_start_date",
]


def _core_fields_missing(row: Dict[str, Any]) -> bool:
    """True if any core (non-efficacy) required field is still empty."""
    return any(not str(row.get(f, "")).strip() for f in CORE_ROW_FIELDS)


def _needs_retry(row: Dict[str, Any]) -> bool:
    """A row is a Pass-2 retry candidate if it's missing efficacy data OR
    missing a core field that should normally be recoverable."""
    return _efficacy_missing(row) or _core_fields_missing(row)


# ---------------------------------------------------------------------------
# Pre-fetch: pull real page/registry content for each trial before sending
# to Gemini, so we can feed dense, verified text (especially the results
# section) directly into the prompt instead of relying only on a search call
# per trial.
#
# IMPORTANT: clinicaltrials.gov's public site is a JavaScript SPA — a plain
# HTTP GET of the study page returns mostly navigation chrome, not the
# study/results content, because that's loaded client-side after the page
# boots. Scraping that HTML with regex silently produced near-empty or junk
# text for the majority of CT.gov trials while still being labeled the "most
# reliable source" in the prompt. CT.gov also happens to publish a real JSON
# API (v2) with the full protocol AND results sections, so for NCT-prefixed
# trials we use that instead of HTML scraping — it is both more reliable and
# cheaper than trying to parse rendered HTML.
#
# For non-CT.gov registries (EU CTIS / EudraCT) we don't have a documented
# public JSON API here, so we still fall back to fetching the HTML page, but
# with two fixes: (1) HTML tags are stripped BEFORE we search for a results
# section, since regexes like `Study\s+Results` never matched across tag
# boundaries in raw markup; (2) pages that look like an empty/near-empty SPA
# shell are detected and dropped rather than passed to Gemini as if they were
# real content.
# ---------------------------------------------------------------------------

# How many registry pages to fetch concurrently.
PREFETCH_CONCURRENCY = 10

# Max characters of page text to include per trial — enough for the results
# section without blowing up the prompt context window.
PREFETCH_MAX_CHARS = 6000

# Timeout per page fetch (seconds). Registry pages are usually fast; if
# one hangs we don't want to block the whole pipeline.
PREFETCH_TIMEOUT = 20

# CT.gov API v2 fields needed to reconstruct a results-bearing text block.
_CTGOV_API_FIELDS = (
    "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,EnrollmentCount,"
    "StartDateStruct,PrimaryCompletionDateStruct,LeadSponsorName,"
    "HasResults,OutcomeMeasureList"
)

# Keywords used (on already tag-stripped plain text) to find and window
# around the parts of an HTML page most likely to hold efficacy data.
# Operating on plain text avoids the tag-boundary-matching problem that a
# raw-HTML regex like `Study\s+Results` has.
_RESULT_KEYWORDS = [
    "study results", "results information", "primary outcome",
    "secondary outcome", "outcome measure", "hba1c", "weight loss",
    "weight change", "alt reduction", "mash resolution", "nash resolution",
    "change from baseline", "trial results", "efficacy",
]

# Markers that indicate we've been handed an SPA loading shell rather than
# real content (e.g. "Loading...", empty Angular root divs, cookie-consent
# only pages). If the tag-stripped text is short AND matches one of these,
# we discard it instead of forwarding near-empty text to Gemini as if it
# were reliable page content.
_SPA_SHELL_MARKERS = re.compile(
    r"(?i)\b(loading\.\.\.|please enable javascript|enable\s+javascript|"
    r"just a moment|checking your browser|access denied|cookie(s)? consent)\b"
)
_SPA_SHELL_MIN_CHARS = 300


def _strip_html_tags(html: str) -> str:
    """Strip script/style blocks and tags, decode common entities, collapse
    whitespace. Operates on the FULL page first so keyword search afterward
    isn't broken by tags sitting between words that belong together in the
    rendered text."""
    if not html:
        return ""
    text = re.sub(r"(?si)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (text
            .replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&nbsp;", " ")
            .replace("&#8209;", "-").replace("&#39;", "'")
            .replace("&quot;", '"'))
    return re.sub(r"\s+", " ", text).strip()


def _extract_page_text(html: str, max_chars: int = PREFETCH_MAX_CHARS) -> str:
    """Extract useful text from registry page HTML.

    Strips tags first (so keyword matching works across what used to be tag
    boundaries), detects SPA loading shells and returns "" for them instead
    of passing near-empty/misleading content downstream, and — if the page
    is long — keeps windows AROUND result-related keywords rather than
    blindly keeping only the head and tail of the document (which can throw
    away a results table sitting in the middle of the page).
    """
    if not html:
        return ""

    text = _strip_html_tags(html)

    if len(text) < _SPA_SHELL_MIN_CHARS and _SPA_SHELL_MARKERS.search(text):
        return ""
    if len(text) < 100:
        # Too short to be a real study page regardless of markers.
        return ""

    if len(text) <= max_chars:
        return text

    lower = text.lower()
    window = 900  # chars of context kept on each side of a keyword hit
    spans: List[tuple] = []
    for kw in _RESULT_KEYWORDS:
        start = 0
        while True:
            idx = lower.find(kw, start)
            if idx == -1:
                break
            spans.append((max(0, idx - window), min(len(text), idx + len(kw) + window)))
            start = idx + len(kw)

    if spans:
        # Merge overlapping windows, keep in document order.
        spans.sort()
        merged: List[list] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        pieces = [text[s:e] for s, e in merged]
        combined = " [...] ".join(pieces)
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        # Always keep a short head so trial identity/metadata isn't lost.
        head = text[:400]
        if not combined.startswith(head[:100]):
            combined = head + " [...] " + combined
        return combined[:max_chars]

    # No keyword hits found anywhere — fall back to head+tail, but flag it
    # clearly so the prompt-side logic doesn't over-trust this text.
    half = max_chars // 2
    return text[:half] + " [...PAGE TRUNCATED, NO RESULTS KEYWORDS FOUND...] " + text[-half:]


async def _fetch_ctgov_api_text(nct_id: str, session: aiohttp.ClientSession) -> str:
    """Pull a results-bearing text block for an NCT trial from CT.gov's real
    JSON API (v2), instead of scraping the JS-rendered HTML page which does
    not contain the results content in its initial response."""
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    try:
        async with session.get(url, params={"fields": _CTGOV_API_FIELDS},
                                timeout=aiohttp.ClientTimeout(total=PREFETCH_TIMEOUT)) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json()
    except Exception as exc:
        print(f"    [prefetch] {nct_id}: CT.gov API {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return ""

    protocol = data.get("protocolSection", {}) if isinstance(data, dict) else {}
    ident = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    outcomes = protocol.get("outcomesModule", {})
    has_results = bool(data.get("hasResults")) if isinstance(data, dict) else False

    parts = [
        f"Official Title: {ident.get('officialTitle', '')}",
        f"Overall Status: {status.get('overallStatus', '')}",
        f"Lead Sponsor: {sponsor.get('leadSponsor', {}).get('name', '')}",
        f"Has Results Posted: {has_results}",
    ]
    for om in outcomes.get("primaryOutcomes", []) or []:
        parts.append(f"Primary Outcome Measure: {om.get('measure', '')} — {om.get('description', '')}")
    for om in outcomes.get("secondaryOutcomes", []) or []:
        parts.append(f"Secondary Outcome Measure: {om.get('measure', '')} — {om.get('description', '')}")

    results_section = data.get("resultsSection", {}) if isinstance(data, dict) else {}
    if results_section:
        outcome_measures = results_section.get("outcomeMeasuresModule", {}).get("outcomeMeasures", [])
        for om in outcome_measures:
            title = om.get("title", "")
            om_type = om.get("type", "")
            desc = om.get("description", "")
            parts.append(f"Reported Outcome ({om_type}): {title} — {desc}")
            for group in om.get("classes", []):
                for cat in group.get("categories", []):
                    for measurement in cat.get("measurements", []):
                        parts.append(
                            f"  Group {measurement.get('groupId', '')}: "
                            f"{measurement.get('value', '')} {om.get('unitOfMeasure', '')}"
                        )

    text = "\n".join(p for p in parts if p and not p.endswith(": ") and not p.endswith("— "))
    return text[:PREFETCH_MAX_CHARS]


async def _prefetch_registry_pages(
    rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Fetch supporting content for each trial that has a trial ID / URL.

    Returns a dict mapping trial_id -> extracted text. NCT-prefixed trials
    use CT.gov's structured JSON API (reliable, includes results); other
    trials fall back to fetching and text-extracting their registry URL.
    """
    ctgov_ids: List[str] = []
    html_url_map: Dict[str, str] = {}  # trial_id -> url
    for row in rows:
        tid = row.get("trial_id", "")
        if not tid:
            continue
        if _is_ctgov_id(tid):
            ctgov_ids.append(tid)
            continue
        url = row.get("url", "") or row.get("source_url", "")
        if url and url.startswith("http"):
            html_url_map[tid] = url

    if not ctgov_ids and not html_url_map:
        return {}

    print(f"  Pre-fetching content for {len(ctgov_ids)} CT.gov (API) + "
          f"{len(html_url_map)} other (HTML) trial(s) ...", file=sys.stderr)

    results: Dict[str, str] = {}
    semaphore = asyncio.Semaphore(PREFETCH_CONCURRENCY)

    async def fetch_ctgov(tid: str, session: aiohttp.ClientSession):
        async with semaphore:
            text = await _fetch_ctgov_api_text(tid, session)
            results[tid] = text

    async def fetch_html(tid: str, url: str, session: aiohttp.ClientSession):
        async with semaphore:
            try:
                async with session.get(url, allow_redirects=True,
                                       headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    if resp.status == 200:
                        html = await resp.text(errors="replace")
                        results[tid] = _extract_page_text(html)
                    else:
                        print(f"    [prefetch] {tid}: HTTP {resp.status}",
                              file=sys.stderr)
                        results[tid] = ""
            except asyncio.TimeoutError:
                print(f"    [prefetch] {tid}: timeout after {PREFETCH_TIMEOUT}s",
                      file=sys.stderr)
                results[tid] = ""
            except Exception as exc:
                print(f"    [prefetch] {tid}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                results[tid] = ""

    timeout = aiohttp.ClientTimeout(total=PREFETCH_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(
            *[fetch_ctgov(tid, session) for tid in ctgov_ids],
            *[fetch_html(tid, url, session) for tid, url in html_url_map.items()],
        )

    fetched_count = sum(1 for v in results.values() if v)
    total = len(ctgov_ids) + len(html_url_map)
    print(f"  Pre-fetched {fetched_count}/{total} trial(s) with usable content "
          f"({len(ctgov_ids)} via CT.gov API, {len(html_url_map)} via HTML fetch)",
          file=sys.stderr)
    return results


try:
    from registry_common import write_excel
except ImportError:
    write_excel = None

# Registry source modules. Each exposes fetch(drug, max_records=..., details=...)
# -> List[Dict] of rows sharing the same unified schema (trial_id, title,
# public_title, status, phase, sponsor, countries, url, ...) via registry_common.
# A source that fails to import is skipped (with a warning) rather than
# crashing the whole run, so this script still works if e.g. ctis/eudract
# scrapers aren't present on disk.
SOURCE_MODULES: Dict[str, Any] = {}

try:
    import ctgov_trials
    SOURCE_MODULES["ctgov"] = ctgov_trials
except ImportError as exc:
    print(f"  [warn] ctgov_trials.py not importable, skipping CT.gov: {exc}",
          file=sys.stderr)

try:
    import euct_trials
    SOURCE_MODULES["euct"] = euct_trials
except ImportError as exc:
    print(f"  [warn] euct_trials.py not importable, skipping EU (CTIS/EudraCT): {exc}",
          file=sys.stderr)

# Trial-ID prefixes that identify a row as CT.gov-native. Only these are
# safe to pass to fetch_ctgov_dates(), which calls the ClinicalTrials.gov
# API and won't recognise EU CTIS numbers (CT number, e.g. "2023-501234-56-00")
# or legacy EudraCT numbers (e.g. "2014-001234-56").
def _is_ctgov_id(trial_id: str) -> bool:
    return bool(re.match(r"^NCT\d+$", str(trial_id or ""), flags=re.IGNORECASE))


# Maps gemini_extractor's Step-2 output field names -> the column names on
# your ctgov_trials.py rows / final Excel sheet. Edit the right-hand side
# if your column names differ (e.g. to match FINAL_COLUMNS in
# fetch_all_trials.py).
GEMINI_FIELD_MAP = {
    "Dosage": "dosage",
    "Phase": "phase",
    "Status": "phase_status",
    "Trial Title": "trial_title",
    "Trial_Study_Type": "trial_study_type",
    "Size": "trial_size",
    "Primary Region": "trial_location",
    "Start Date": "trial_start_date",
    "Completion Date": "trial_completion_date",
    "HbA1c Change (%)": "hba1c_change_pct",
    "HbA1c Duration": "hba1c_duration",
    "Weight Loss (%)": "weight_change_pct",
    "Weight Duration": "weight_duration",
    "ALT Reduction (%)": "alt_reduction_pct",
    "ALT Duration": "alt_duration",
    "MASH Outcome (%)": "mash_resolution_pct",
    "MASH Duration": "mash_duration",
    "Company": "company_name",
    "Source URL": "source_url",
}


# The only columns that should appear in the final output, in this order.
FINAL_COLUMNS = [
    "molecule_name",
    "registry_source",
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
]


def normalise_phase(raw: str) -> str:
    """Normalise 'Phase III', 'PHASE3', 'Phase 2/3', etc. down to '1'/'2'/'3'/'4'
    (or '2/3' style for combined phases)."""
    if not raw:
        return ""
    s = re.sub(r"(?i)\bphase\s*", "phase ", str(raw))
    phase_map = {
        "one": "1", "two": "2", "three": "3", "four": "4",
        "i": "1", "ii": "2", "iii": "3", "iv": "4",
        "1": "1", "2": "2", "3": "3", "4": "4",
    }
    nums = []
    for tok in re.split(r"[/,;\s]+", s.lower()):
        tok = tok.strip("(). ")
        if tok in phase_map and phase_map[tok] not in nums:
            nums.append(phase_map[tok])
    return "/".join(nums) if nums else str(raw).strip()


def remap_row(row: Dict[str, Any], drug: str) -> Dict[str, Any]:
    """Collapse a merged registry+Gemini row down to exactly FINAL_COLUMNS.

    Falls back to the raw registry field name (e.g. 'title', 'countries',
    'status' — shared by ctgov_trials.py and euct_trials.py via
    registry_common) if the Gemini-named field wasn't filled in.
    """
    return {
        "molecule_name": drug,
        "registry_source": row.get("source", "") or row.get("registry_source", ""),
        "trial_id": row.get("trial_id", ""),
        "dosage": row.get("dosage", ""),
        "phase": normalise_phase(row.get("phase", "")),
        "trial_title": (row.get("trial_title", "") or row.get("title", "")
                        or row.get("public_title", "")
                        or row.get("brief_title", "")),
        "trial_study_type": row.get("trial_study_type", "") or row.get("study_type", ""),
        "trial_size": (row.get("trial_size", "") or row.get("actual_enrollment", "")
                      or row.get("target_enrollment", "")
                      or row.get("enrollment", "")),
        "trial_location": row.get("trial_location", "") or row.get("countries", ""),
        "trial_start_date": row.get("trial_start_date", "") or row.get("start_date", ""),
        "trial_completion_date": row.get("trial_completion_date", "")
                                  or row.get("completion_date", ""),
        "phase_status": row.get("phase_status", "") or row.get("status", ""),
        "hba1c_change_pct": row.get("hba1c_change_pct", ""),
        "hba1c_duration": row.get("hba1c_duration", ""),
        "weight_change_pct": row.get("weight_change_pct", ""),
        "weight_duration": row.get("weight_duration", ""),
        "alt_reduction_pct": row.get("alt_reduction_pct", ""),
        "alt_duration": row.get("alt_duration", ""),
        "mash_resolution_pct": row.get("mash_resolution_pct", ""),
        "mash_duration": row.get("mash_duration", ""),
        "company_name": row.get("company_name", "") or row.get("sponsor", ""),
        "source_url": row.get("source_url", "") or row.get("url", ""),
    }


def _extract_trial_acronym(title: str) -> str:
    """Try to extract a trial program acronym from the title.

    Many trials embed their program name in parentheses at the end of the
    title, e.g. "... Lose Weight (REDUCE-1)" or have well-known naming
    patterns like REDEFINE, REIMAGINE, STEP, SURPASS, SURMOUNT, etc.
    The acronym is far more searchable than the NCT ID in press releases
    and publications.
    """
    if not title:
        return ""
    # Pattern 1: explicit parenthesized acronym at end of title
    m = re.search(r"\(([A-Z][A-Z0-9 _-]{1,30}(?:\s*\d+)?)\)\s*$", title)
    if m:
        return m.group(1).strip()
    # Pattern 2: known program names anywhere in title
    for prog in ("REDEFINE", "REIMAGINE", "REDUCE", "STEP", "SURPASS",
                 "SURMOUNT", "PIONEER", "SUSTAIN", "FLOW", "SELECT",
                 "SUMMIT", "ESSENCE"):
        pat = re.search(rf"\b({prog}\s*[-]?\s*\d*)\b", title, re.IGNORECASE)
        if pat:
            return pat.group(1).strip().upper()
    return ""


def _to_gemini_trial_stub(row: Dict[str, Any],
                          prefetched: Optional[Dict[str, str]] = None,
                          ) -> Dict[str, str]:
    """Build a context-rich stub for Gemini's Step 2 prompt.

    The field is still called "NCT_ID" because that's the key
    get_trial_details expects, but it works fine as a generic trial
    identifier — CT.gov NCT numbers, EU CTIS CT numbers, or legacy
    EudraCT numbers all get passed through and searched for as-is.

    We now also pack every piece of metadata the registry scraper already
    fetched into a "Registry_Context" field. This gives Gemini:
      1. Concrete search anchors (sponsor name, exact title, countries)
         so it can find publications and press releases faster.
      2. Known values it can skip re-searching, letting it spend its
         full search budget on the hard-to-find efficacy fields (HbA1c,
         weight loss, ALT, MASH, dosage).
      3. Disambiguation cues for common molecule names that return many
         unrelated trials in a bare search.

    If prefetched page text is available (from _prefetch_registry_pages),
    it is included as "Prefetched_Page_Content" so Gemini can extract
    efficacy data directly without needing to search for the page.
    """
    trial_id = row.get("trial_id", "")
    program_name = (row.get("public_title", "") or row.get("title", "")
                    or row.get("brief_title", "") or "N/A")

    # Extract trial acronym (e.g. "REDEFINE 1", "REIMAGINE 2") — far more
    # searchable in press releases and publications than the NCT ID.
    acronym = _extract_trial_acronym(program_name)
    if not acronym:
        # Try the other title fields too
        for alt_key in ("title", "public_title", "brief_title"):
            alt_title = row.get(alt_key, "")
            if alt_title and alt_title != program_name:
                acronym = _extract_trial_acronym(alt_title)
                if acronym:
                    break

    # Collect every useful field the registry already gave us.
    context_parts: List[str] = []

    if acronym:
        context_parts.append(f"Trial Acronym/Program: {acronym}")

    sponsor = row.get("sponsor", "") or row.get("lead_sponsor", "")
    if sponsor:
        context_parts.append(f"Sponsor/Company: {sponsor}")

    phase = row.get("phase", "")
    if phase:
        context_parts.append(f"Phase: {phase}")

    status = row.get("status", "") or row.get("overall_status", "")
    if status:
        context_parts.append(f"Status: {status}")

    enrollment = (row.get("actual_enrollment", "")
                  or row.get("target_enrollment", "")
                  or row.get("enrollment", ""))
    if enrollment:
        context_parts.append(f"Enrollment: {enrollment}")

    study_type = row.get("study_type", "")
    if study_type:
        context_parts.append(f"Study Type: {study_type}")

    countries = row.get("countries", "")
    if countries:
        context_parts.append(f"Countries: {countries}")

    start_date = row.get("start_date", "")
    if start_date:
        context_parts.append(f"Start Date: {start_date}")

    completion_date = row.get("completion_date", "")
    if completion_date:
        context_parts.append(f"Completion Date: {completion_date}")

    conditions = row.get("conditions", "") or row.get("condition", "")
    if conditions:
        context_parts.append(f"Conditions: {conditions}")

    interventions = row.get("interventions", "") or row.get("intervention", "")
    if interventions:
        context_parts.append(f"Interventions: {interventions}")

    url = row.get("url", "")
    if url:
        context_parts.append(f"Registry URL: {url}")

    source_url = row.get("source_url", "")
    if source_url and source_url != url:
        context_parts.append(f"Source URL: {source_url}")

    # Secondary IDs help Gemini cross-reference across registries
    secondary_ids = row.get("secondary_ids", "") or row.get("other_ids", "")
    if secondary_ids:
        context_parts.append(f"Secondary IDs: {secondary_ids}")

    # Has results posted? Tells Gemini whether to look at the results tab
    has_results = row.get("has_results", "")
    if has_results:
        context_parts.append(f"Results Posted: {has_results}")

    stub = {
        "NCT_ID": trial_id,
        "Program_Name": program_name,
    }

    if context_parts:
        stub["Registry_Context"] = "; ".join(context_parts)

    # Inject pre-fetched page text if available
    if prefetched and trial_id in prefetched and prefetched[trial_id]:
        stub["Prefetched_Page_Content"] = prefetched[trial_id]

    return stub


def _clean_trial_id(raw_id: str) -> str:
    """Gemini sometimes returns 'NCT01234567 (STEP 1)' style values."""
    return raw_id.split("(")[0].strip().split(" ")[0].strip()


def _is_combination_molecule(drug: str) -> bool:
    """Heuristic: does this molecule name look like a fixed-dose
    combination (e.g. "Cagrilintide+Semaglutide", "Cagrilintide + Semaglutide")
    rather than a single active ingredient?"""
    return bool(re.search(r"\+|\band\b", drug, flags=re.IGNORECASE))


# gemini_extractor.py's Step 2 prompt opens with a validation gate:
# "verify each trial is a {molecule} MONOTHERAPY trial ... SKIP trials
# testing combination drugs (e.g., CagriSema = Cagrilintide + Semaglutide
# should be EXCLUDED when searching for Semaglutide)". That's written for
# single-ingredient searches, where you WANT combination trials filtered
# out. If {molecule} is itself passed in as a combination name (e.g.
# "Cagrilintide+Semaglutide"), that same instruction — with your exact
# molecule spelled out as the example of what to exclude — pushes Gemini
# to under-extract or blank out efficacy data for the very trials you're
# asking about. This override is appended whenever the molecule name looks
# like a combination, telling Gemini the monotherapy filter doesn't apply.
COMBINATION_OVERRIDE = """
OVERRIDE ON THE MONOTHERAPY VALIDATION INSTRUCTION ABOVE:
The molecule under study here IS a fixed-dose combination product. The
earlier instruction to SKIP/exclude "combination drug" trials does NOT
apply in this case — every trial ID given to you below was already
confirmed to test this exact combination as its primary intervention.
Do not decline, blank out, or under-extract efficacy fields for these
trials on the basis that they involve a drug combination. Extract HbA1c
change, weight loss, ALT reduction, MASH outcome, dosage, etc. exactly as
you would for any other Phase 2/3 trial with the combination as the
tested product.
"""


# gemini_extractor.py's Step 2 prompt is written CT.gov-first: its own
# example for the "Trial ID" output field is an NCT number, its example
# "Source URL" is a clinicaltrials.gov URL, and its DATA SOURCES list
# mentions the legacy EudraCT register (clinicaltrialsregister.eu) but
# never CTIS at all. Left alone, that biases Gemini to either come up
# empty for EU CTIS/EudraCT trial IDs, or — worse — swap in an NCT number
# for the same molecule/trial instead of echoing the ID we gave it, which
# then fails the exact-match merge below and looks like "no data returned".
#
# We don't edit gemini_extractor.py itself (other callers may depend on
# its exact prompt); instead we use its extra_fields_prompt hook to patch
# in registry-appropriate guidance per batch, keyed by registry_source.
_CONTEXT_USAGE_BLOCK = """
REGISTRY CONTEXT — HOW TO USE IT:
Each trial below may include a "Registry_Context" field with metadata
already fetched from the official registry (sponsor, phase, status,
enrollment, conditions, interventions, countries, dates, registry URL).
USE this context to:
1. SKIP re-searching for fields you already have (phase, status, sponsor,
   enrollment, study type, countries, dates) — trust the registry values
   and spend your search budget on the EFFICACY fields instead.
2. BUILD BETTER SEARCHES — use the sponsor/company name + trial title +
   conditions as search terms to find press releases, conference
   abstracts, and publications that report HbA1c change, weight loss,
   ALT reduction, and MASH resolution data.
3. OPEN THE REGISTRY URL directly if provided — go to the Results tab
   (not just the summary/protocol page) for posted efficacy data.
4. CROSS-REFERENCE — if Secondary IDs are provided (e.g. an NCT number
   for an EU trial), search for those too to find linked publications.
FOCUS YOUR SEARCH EFFORT on: dosage, HbA1c change (%), weight change (%),
ALT reduction (%), MASH resolution (%), and company name. These are the
fields most likely to require digging into results pages, press releases,
and published papers. The rest are usually already filled in from the
registry metadata.

TRIAL ACRONYM SEARCH STRATEGY:
If "Trial Acronym/Program" is provided (e.g. "REDEFINE 1", "REIMAGINE 2",
"STEP 1", "REDUCE-1"), ALWAYS search for the acronym — it is FAR more
discoverable than the NCT ID in press releases and publications:
- Search: "<Acronym>" + "<molecule>" + "results" (e.g. "REDEFINE 1 CagriSema results")
- Search: "<Acronym>" + "weight loss" / "HbA1c" / "efficacy"
- Search: "<Acronym>" + "phase 3" + "topline"
- Search: "<Acronym>" + site:nejm.org OR site:thelancet.com OR site:pubmed.ncbi.nlm.nih.gov
Press releases almost always reference the trial acronym, not the NCT ID.
Do NOT skip the acronym search even if the registry page had no results —
efficacy data is often in press releases and journal articles that only
use the program name.

ALSO SEARCH USING THE SOURCE URL:
If a "Source URL" or "Registry URL" is provided pointing to a
clinicaltrials.gov or EU registry page, visit it directly (especially
the Results tab) BEFORE falling back to text searches. Many completed
trials have structured results data posted on their registry page that
is not in the summary view.

PRE-FETCHED PAGE CONTENT:
Some trials include a "Prefetched_Page_Content" field containing text
already extracted from the trial's official registry page. When present:
- EXTRACT efficacy data DIRECTLY from this text FIRST — it contains the
  actual page content including any posted results, outcome measures,
  and study details. This is the MOST RELIABLE source.
- Look for keywords like "primary outcome", "HbA1c", "weight", "ALT",
  "MASH", "resolution", "reduction", "change from baseline", "placebo",
  "dose", "mg" in the prefetched text to find efficacy numbers.
- You do NOT need to search for or visit this page — it has already been
  fetched for you. Save your search budget for finding publications,
  press releases, and conference abstracts that may have ADDITIONAL data
  not on the registry page.
- If the prefetched text shows "[...PAGE TRUNCATED...]", the full page
  was too long; you still have the most relevant beginning and end
  sections — extract what you can and only search if key efficacy fields
  are still missing.
"""

EXTRA_PROMPT_BY_SOURCE: Dict[str, str] = {
    "euct": _CONTEXT_USAGE_BLOCK + """
ADDITIONAL SOURCE GUIDANCE FOR THIS BATCH (EU trials):
- These trial IDs come from the EU CTIS portal (euclinicaltrials.eu,
  format like "2023-501234-56-00") or the legacy EudraCT / EU-CTR
  register (clinicaltrialsregister.eu, format like "2014-001234-56").
  Search BOTH of those sites directly by ID, in addition to
  ClinicalTrials.gov and PubMed (an EU trial may also be registered
  there under its own NCT number as a secondary ID — use that only to
  find published efficacy results, not to replace the ID in your answer).
- "Trial ID" in your JSON response MUST be the EXACT identifier given to
  you above for that trial, character-for-character (e.g. "2023-501234-56-00"
  or "2014-001234-56") — do NOT substitute an NCT number or any other ID,
  even if you find one for the same trial.
""",
    "ctgov": _CONTEXT_USAGE_BLOCK + """
ADDITIONAL SOURCE GUIDANCE FOR THIS BATCH (CT.gov trials):
- "Trial ID" in your JSON response MUST be the EXACT NCT identifier given
  to you above for that trial, character-for-character — do not alter it.
- If "Registry_Context" includes "Results Posted: True" or "Results Posted:
  Yes", go directly to clinicaltrials.gov/study/<ID> → "Study Results" tab
  to extract efficacy endpoints — don't stop at the summary page.
""",
}

# Appended on the retry pass (single trial per call) for trials that came
# back with no efficacy data on the first pass. Explicitly pushes the
# model past the registry front page toward results pages and publications,
# and gives it permission to say "genuinely not published" rather than
# leaving fields blank without trying.
DEEP_DIVE_SUFFIX = """
DEEP-DIVE MODE — this is a single-trial retry because the first pass
returned no efficacy data for this trial. Before answering:
1. If a "Trial Acronym/Program" is in Registry_Context (e.g. REDEFINE 1,
   REIMAGINE 2), START HERE — search for:
   - "<Acronym> results" (e.g. "REDEFINE 1 results")
   - "<Acronym> <molecule> weight loss HbA1c"
   - "<Acronym> topline" or "<Acronym> headline results"
   - "<Acronym> NEJM" or "<Acronym> Lancet" or "<Acronym> published"
   These searches almost always find press releases and publications that
   contain the exact efficacy numbers.
2. Open the trial's own results page (e.g. clinicaltrials.gov/study/<ID>
   -> "Study Results" tab; or the EU registry's results page) — not just
   the summary/protocol page. Also try the Source URL if different from
   the Registry URL.
3. USE THE REGISTRY_CONTEXT provided below to build precise searches:
   - Search for: "<Sponsor Name>" + "<Program Name>" + "results" / "topline"
   - Search for: "<Sponsor Name>" + "<Conditions>" + "phase <N>" + "efficacy"
   - Search for: "<Trial ID>" + "HbA1c" / "weight loss" / "ALT" / "MASH"
   - Search for: "<Sponsor Name>" + "<Program Name>" + "press release"
   - Search for: "<Sponsor Name>" + "<Program Name>" + site:pubmed.ncbi.nlm.nih.gov
   Company press releases and conference abstracts (ADA, EASD, AASLD,
   ENDO) often report efficacy numbers before the formal registry results
   page is updated.
4. Check for a linked publication (PubMed, NEJM, Lancet, Diabetes Care,
   Hepatology, JAMA, etc.) — trial result papers usually report exact
   efficacy numbers even when the registry page only shows "N/A".
5. If the trial has secondary IDs listed in Registry_Context, search for
   those IDs too — some publications cite the secondary ID rather than the
   primary one.
6. Try the sponsor's investor relations / pipeline page — many companies
   post topline data there before peer-reviewed publication.
Only use "N/A" if you've checked the above and the trial genuinely has no
completed results yet (e.g. still recruiting, or terminated with no
posted data) — don't default to "N/A" just because the summary page alone
didn't show it.
"""


async def _fetch_source(name: str, module: Any, drug: str,
                         max_records: Optional[int]) -> List[Dict[str, Any]]:
    """Run one registry module's fetch() in a thread (they're synchronous /
    blocking HTTP scrapers) so multiple sources can be fetched concurrently."""
    label = {"ctgov": "ClinicalTrials.gov API", "euct": "EU (CTIS + EudraCT)"}.get(name, name)
    print(f"Fetching trials for '{drug}' from {label} ...", file=sys.stderr)
    try:
        rows = await asyncio.to_thread(module.fetch, drug, max_records=max_records,
                                        details=True)
    except Exception as exc:
        print(f"  [warn] {name} fetch failed: {exc}", file=sys.stderr)
        return []
    print(f"  Got {len(rows)} trial(s) from {name}", file=sys.stderr)
    for row in rows:
        row.setdefault("registry_source", name)
    return rows


async def _run_batches(drug: str, batches: List[tuple], extra_suffix: str = "",
                        label: str = "pass") -> List[Dict[str, Any]]:
    """Run a list of (source, [stub, ...]) batches through get_trial_details
    concurrently (bounded by MAX_WORKERS) and return the flattened trial list."""
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    # Applied to every batch regardless of source when the molecule name
    # looks like a fixed-dose combination — see COMBINATION_OVERRIDE above.
    combo_suffix = COMBINATION_OVERRIDE if _is_combination_molecule(drug) else ""

    async def run_batch(idx: int, source: str, batch: List[Dict[str, str]]):
        async with semaphore:
            stagger = (idx % MAX_WORKERS) * (RATE_LIMIT_DELAY / MAX_WORKERS)
            if stagger:
                await asyncio.sleep(stagger)
            print(f"  [{label}] Batch {idx + 1}/{len(batches)} [{source}] "
                  f"({len(batch)} trials) - starting", file=sys.stderr)
            extra_prompt = (EXTRA_PROMPT_BY_SOURCE.get(source, "")
                             + combo_suffix + extra_suffix) or None
            data = await get_trial_details(drug, batch, extra_fields_prompt=extra_prompt)
            trials = data.get("trials", [])
            print(f"  [{label}] Batch {idx + 1}/{len(batches)} [{source}] - done "
                  f"({len(trials)} trials)", file=sys.stderr)
            # Flag when the response has fewer trials than we sent — this is
            # the signature of Gemini's JSON output getting truncated
            # (max-output-tokens) mid-batch, which otherwise looks identical
            # to "Gemini searched and found nothing" once merged.
            sent_ids = {stub.get("NCT_ID", "") for stub in batch if stub.get("NCT_ID")}
            returned_ids = {_clean_trial_id(t.get("Trial ID", "")) for t in trials if isinstance(t, dict)}
            if len(trials) < len(batch):
                dropped = sent_ids - returned_ids
                if dropped:
                    print(f"  [{label}] Batch {idx + 1}/{len(batches)} [{source}] - "
                          f"WARNING: sent {len(batch)} trial(s) but only got {len(trials)} back; "
                          f"likely dropped by output truncation: {', '.join(sorted(dropped))}",
                          file=sys.stderr)
            return trials

    results = await asyncio.gather(*[
        run_batch(i, source, batch) for i, (source, batch) in enumerate(batches)
    ])
    return [t for batch in results for t in batch if isinstance(t, dict)]


def _normalise_trial_id(trial_id: str) -> str:
    """Loose form of a trial ID for fallback matching: strip everything but
    letters/digits and uppercase. Used only as a fallback when the exact
    string Gemini returned doesn't match — e.g. it added a trailing space,
    changed case, or left in a stray character — so a real match isn't
    silently discarded over formatting noise."""
    return re.sub(r"[^A-Za-z0-9]", "", str(trial_id or "")).upper()


def _merge_enriched(enriched_trials: List[Dict[str, Any]],
                     rows_by_id: Dict[str, Dict[str, Any]]) -> set:
    """Merge Gemini's enriched fields back onto the original registry rows,
    matched by trial ID. Only overwrite when Gemini actually returned
    something usable — never blank out data the registry scraper already
    had. Returns the set of trial IDs that got a match.

    Matching tries an exact ID match first, then falls back to a normalised
    (letters/digits only, uppercased) comparison so minor formatting
    differences in what Gemini echoes back don't cause a real match to be
    silently dropped.
    """
    norm_lookup = {_normalise_trial_id(tid): tid for tid in rows_by_id}
    matched_ids = set()
    unmatched_returned = []
    for trial in enriched_trials:
        raw_returned_id = trial.get("Trial ID", "")
        trial_id = _clean_trial_id(raw_returned_id)
        row = rows_by_id.get(trial_id)
        if row is None:
            fallback_id = norm_lookup.get(_normalise_trial_id(trial_id))
            if fallback_id:
                trial_id = fallback_id
                row = rows_by_id.get(trial_id)
        if row is None:
            if raw_returned_id:
                unmatched_returned.append(raw_returned_id)
            continue
        for gemini_field, row_field in GEMINI_FIELD_MAP.items():
            value = trial.get(gemini_field)
            # Compare the STRING form, not the raw value — a numeric 0
            # (e.g. Gemini returning a bare 0 instead of "0") is falsy in
            # Python and would otherwise get silently skipped by `if value`,
            # even though "0" can be a legitimate (if rare) result.
            value_str = str(value).strip() if value is not None else ""
            if value_str and value_str not in ("N/A", "n/a", "None"):
                row[row_field] = value
        matched_ids.add(trial_id)
    if unmatched_returned:
        sample = ", ".join(unmatched_returned[:5])
        more = f" (+{len(unmatched_returned) - 5} more)" if len(unmatched_returned) > 5 else ""
        print(f"  [warn] Gemini returned {len(unmatched_returned)} trial(s) whose "
              f"'Trial ID' matched none of the requested IDs (even after "
              f"normalisation) — discarded: {sample}{more}", file=sys.stderr)
    return matched_ids


def _normalise_url(url: str) -> str:
    """Normalise a registry URL for matching: strip trailing slash,
    lowercase, remove protocol and www prefix."""
    if not url:
        return ""
    url = url.strip().rstrip("/").lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url


def _propagate_efficacy_by_url(rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    """Copy efficacy data between trials that share the same source_url.

    Many EU CTIS/EudraCT entries and CT.gov entries point to the same
    underlying clinicaltrials.gov study page. If one row has efficacy data
    and another sharing the same URL doesn't, propagate the data across.
    This is free — no Gemini calls needed.
    """
    # Build URL -> list of trial_ids mapping
    url_to_ids: Dict[str, List[str]] = {}
    for tid, row in rows_by_id.items():
        for url_key in ("url", "source_url"):
            raw_url = row.get(url_key, "")
            norm = _normalise_url(raw_url)
            if norm:
                url_to_ids.setdefault(norm, []).append(tid)

    propagated = 0
    for norm_url, tids in url_to_ids.items():
        if len(tids) < 2:
            continue
        # Find a "donor" row that has efficacy data
        donor = None
        for tid in tids:
            if not _efficacy_missing(rows_by_id[tid]):
                donor = rows_by_id[tid]
                break
        if donor is None:
            continue
        # Propagate to recipients that are missing efficacy data
        for tid in tids:
            recipient = rows_by_id[tid]
            if _efficacy_missing(recipient) and recipient is not donor:
                for field in EFFICACY_ROW_FIELDS:
                    donor_val = str(donor.get(field, "")).strip()
                    recip_val = str(recipient.get(field, "")).strip()
                    if donor_val and not recip_val:
                        recipient[field] = donor[field]
                # Also propagate duration fields
                for dur_field in ("hba1c_duration", "weight_duration",
                                  "alt_duration", "mash_duration"):
                    donor_val = str(donor.get(dur_field, "")).strip()
                    recip_val = str(recipient.get(dur_field, "")).strip()
                    if donor_val and not recip_val:
                        recipient[dur_field] = donor[dur_field]
                propagated += 1

    if propagated:
        print(f"  URL propagation: copied efficacy data to {propagated} "
              f"trial(s) sharing URLs with data-rich siblings", file=sys.stderr)


def _chunk_by_source(rows: List[Dict[str, Any]], batch_size: int,
                     prefetched: Optional[Dict[str, str]] = None) -> List[tuple]:
    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        rows_by_source.setdefault(r.get("registry_source", "ctgov"), []).append(r)
    batches: List[tuple] = []
    for source, source_rows in rows_by_source.items():
        stubs = [_to_gemini_trial_stub(r, prefetched=prefetched) for r in source_rows]
        for i in range(0, len(stubs), batch_size):
            batches.append((source, stubs[i:i + batch_size]))
    return batches


async def enrich(drug: str, max_records: Optional[int] = None,
                  sources: Optional[List[str]] = None,
                  retry_missing: bool = True) -> List[Dict[str, Any]]:
    sources = sources or list(SOURCE_MODULES.keys())
    active = [s for s in sources if s in SOURCE_MODULES]
    missing = [s for s in sources if s not in SOURCE_MODULES]
    if missing:
        print(f"  [warn] requested source(s) not available, skipping: {missing}",
              file=sys.stderr)
    if not active:
        print("  No registry sources available to fetch from.", file=sys.stderr)
        return []

    fetched = await asyncio.gather(*[
        _fetch_source(name, SOURCE_MODULES[name], drug, max_records)
        for name in active
    ])
    rows = [row for batch in fetched for row in batch]
    print(f"  Got {len(rows)} trial(s) total across source(s): {active}", file=sys.stderr)

    if not rows:
        return []

    if _is_combination_molecule(drug):
        print(f"  [info] '{drug}' looks like a fixed-dose combination — appending "
              f"COMBINATION_OVERRIDE to every Gemini batch so the built-in monotherapy "
              f"filter doesn't exclude/blank these trials.", file=sys.stderr)

    # trial_id namespaces don't overlap across sources (NCT######## vs.
    # CTIS CT numbers vs. EudraCT numbers), so a single dict keyed by
    # trial_id is safe to merge across sources.
    rows_by_id = {r["trial_id"]: r for r in rows if r.get("trial_id")}
    ctgov_ids = {tid for tid in rows_by_id if _is_ctgov_id(tid)}

    # --- Pre-fetch: hit each trial's registry URL to grab page content ---
    # This runs concurrently and gives Gemini the actual page text so it
    # can extract efficacy data directly instead of spending a search call
    # to re-find a page we already have the URL for.
    prefetched = await _prefetch_registry_pages(list(rows_by_id.values()))

    # --- Pass 1: normal batching, smaller batch size for research depth ---
    batches = _chunk_by_source(list(rows_by_id.values()), FIRST_PASS_BATCH_SIZE,
                                prefetched=prefetched)
    print(f"  Sending {len(rows_by_id)} trial(s) to Gemini in {len(batches)} "
          f"batch(es) of up to {FIRST_PASS_BATCH_SIZE} ...", file=sys.stderr)
    enriched_trials = await _run_batches(drug, batches, label="pass1")
    matched_ids = _merge_enriched(enriched_trials, rows_by_id)

    print(f"  Pass 1: merged Gemini data into {len(matched_ids)}/{len(rows_by_id)} trial(s)",
          file=sys.stderr)

    unmatched = [tid for tid in rows_by_id if tid not in matched_ids]
    if unmatched:
        by_src = {}
        for tid in unmatched:
            src = rows_by_id[tid].get("registry_source", "?")
            by_src.setdefault(src, []).append(tid)
        for src, ids in by_src.items():
            sample = ", ".join(ids[:5])
            more = f" (+{len(ids) - 5} more)" if len(ids) > 5 else ""
            print(f"  [warn] {len(ids)} '{src}' trial(s) got no Gemini match at all: "
                  f"{sample}{more}", file=sys.stderr)

    # --- Pass 2: single-trial retry for rows still missing efficacy data ---
    # Covers both "matched but every efficacy field was N/A" and "no match
    # at all" — either way, give it one more focused shot before giving up.
    if retry_missing:
        # Retry any row missing efficacy data OR a core field that should
        # normally be recoverable (title, phase, size, location, status,
        # start date) — not just efficacy, since those aren't the only
        # required fields that can come back empty.
        retry_rows = [r for r in rows_by_id.values() if _needs_retry(r)]
        if retry_rows:
            efficacy_only = sum(1 for r in retry_rows if _efficacy_missing(r) and not _core_fields_missing(r))
            core_missing = sum(1 for r in retry_rows if _core_fields_missing(r))
            print(f"  Pass 2: retrying {len(retry_rows)} trial(s) "
                  f"({efficacy_only} missing only efficacy data, "
                  f"{core_missing} missing core fields), "
                  f"1 trial/call, deep-dive prompt ...", file=sys.stderr)
            retry_batches = _chunk_by_source(retry_rows, RETRY_BATCH_SIZE,
                                              prefetched=prefetched)
            retry_enriched = await _run_batches(drug, retry_batches,
                                                 extra_suffix=DEEP_DIVE_SUFFIX, label="pass2")
            retry_matched = _merge_enriched(retry_enriched, rows_by_id)
            still_missing = sum(1 for r in retry_rows if _needs_retry(r))
            print(f"  Pass 2: recovered data for "
                  f"{len(retry_rows) - still_missing}/{len(retry_rows)} previously-incomplete trial(s)",
                  file=sys.stderr)
        else:
            print("  Pass 2: skipped — no trials missing efficacy or core data.", file=sys.stderr)

    # --- Pass 3 (no API calls): cross-URL efficacy propagation ---
    # Many EU CTIS/EudraCT entries share the same source_url as a CT.gov
    # trial (e.g. both point to the same clinicaltrials.gov/study/NCTxxx
    # page). If the CT.gov row already has efficacy data but the EU row
    # doesn't, copy it across. This is free — no Gemini calls needed.
    _propagate_efficacy_by_url(rows_by_id)

    # Cross-check Start/Completion Date against the authoritative CT.gov API
    # (usually a no-op since ctgov_trials.py already used that API, but kept
    # for parity in case Gemini overwrote them with guesses). This ONLY
    # applies to CT.gov-native rows (NCT IDs) — the CT.gov API has no
    # knowledge of EU CTIS/EudraCT trial numbers, so EU rows keep whatever
    # dates their own registry scraper / Gemini already filled in.
    if ctgov_ids:
        dates_map = await fetch_ctgov_dates(list(ctgov_ids))
        for nct_id, dates in dates_map.items():
            row = rows_by_id.get(nct_id)
            if not row:
                continue
            # Only overwrite when the API actually returned something —
            # don't blank out a good existing value (from the registry
            # scraper or Gemini) just because this particular field came
            # back empty/"N/A" from the API.
            start = dates.get("Start Date", "")
            completion = dates.get("Completion Date", "")
            if start and str(start).strip().upper() not in ("N/A", ""):
                row["trial_start_date"] = start
            if completion and str(completion).strip().upper() not in ("N/A", ""):
                row["trial_completion_date"] = completion

    return list(rows_by_id.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drug", help="drug / intervention name, e.g. 'semaglutide'")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--max-records", type=int, default=None,
                     help="cap the number of trials fetched/enriched PER SOURCE")
    ap.add_argument("--sources", default="ctgov,euct",
                     help="comma-separated registry sources to pull from: "
                          "'ctgov' (ClinicalTrials.gov), 'euct' (EU CTIS + EudraCT), "
                          f"or both (default). Available on this machine: "
                          f"{', '.join(SOURCE_MODULES) or 'none'}")
    ap.add_argument("--no-retry", action="store_true",
                     help="skip the automatic single-trial retry pass for trials "
                          "that came back with no efficacy data (faster, fewer API calls, "
                          "lower fill rate)")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows = asyncio.run(enrich(args.drug, args.max_records, sources,
                               retry_missing=not args.no_retry))

    if not rows:
        print("No trials found / enriched.", file=sys.stderr)
        return 1

    still_empty = sum(1 for r in rows if _efficacy_missing(r))
    still_core_missing = sum(1 for r in rows if _core_fields_missing(r))
    print(f"  Final: {len(rows) - still_empty}/{len(rows)} trial(s) have at least "
          f"some efficacy data; {still_empty} have none (likely no results published yet).",
          file=sys.stderr)
    if still_core_missing:
        print(f"  Final: {still_core_missing}/{len(rows)} trial(s) are still missing "
              f"a core field (title/phase/size/location/status/start date) after "
              f"both passes — check these rows.", file=sys.stderr)

    final_rows = [remap_row(row, args.drug) for row in rows]

    os.makedirs(args.outdir, exist_ok=True)
    drug_slug = args.drug.lower().replace(" ", "_")
    src_tag = "_".join(sources)
    out_path = os.path.join(args.outdir, f"{drug_slug}_{src_tag}_gemini_enriched.xlsx")

    if write_excel is not None:
        write_excel(final_rows, out_path, FINAL_COLUMNS,
                    sheet_name="Registries + Gemini")
    else:
        import pandas as pd
        pd.DataFrame(final_rows, columns=FINAL_COLUMNS).to_excel(out_path, index=False)

    print(f"\nWrote {len(final_rows)} enriched trial(s) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())