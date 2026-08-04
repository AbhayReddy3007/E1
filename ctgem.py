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
FIRST_PASS_BATCH_SIZE = 4

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


# ---------------------------------------------------------------------------
# Pre-fetch: hit each trial's registry URL before sending to Gemini so we
# can feed the raw page text (especially the results section) directly into
# the prompt. This saves Gemini a search call per trial and gives it data
# that Google Search sometimes can't reach (dynamic JS-rendered results
# tabs, pages behind cookie walls, etc.).
# ---------------------------------------------------------------------------

# How many registry pages to fetch concurrently.
PREFETCH_CONCURRENCY = 10

# Max characters of page text to include per trial — enough for the results
# section without blowing up the prompt context window.
PREFETCH_MAX_CHARS = 6000

# Timeout per page fetch (seconds). Registry pages are usually fast; if
# one hangs we don't want to block the whole pipeline.
PREFETCH_TIMEOUT = 20

# Sections / keywords we try to isolate from the fetched HTML. If we find
# a results section we prefer that over the full page — it's denser and
# more relevant.
_RESULTS_SECTION_PATTERNS = [
    # CT.gov results tab content
    re.compile(r"(?si)(Study\s+Results.*?)(?=<footer|<div[^>]*id=[\"']footer|$)"),
    # CT.gov outcome measures
    re.compile(r"(?si)(Primary\s+Outcome\s+Measures?.*?)(?=Secondary\s+Outcome|<footer|$)"),
    re.compile(r"(?si)(Secondary\s+Outcome\s+Measures?.*?)(?=<footer|$)"),
    # EU CTIS results
    re.compile(r"(?si)(Results\s+information.*?)(?=<footer|$)"),
    # Generic "results" heading
    re.compile(r"(?si)(<h[1-4][^>]*>.*?results.*?</h[1-4]>.*?)(?=<footer|$)"),
]


def _extract_page_text(html: str, max_chars: int = PREFETCH_MAX_CHARS) -> str:
    """Extract useful text from registry page HTML.

    Tries to isolate the results section first; falls back to the full
    page body. Strips HTML tags and collapses whitespace.
    """
    if not html:
        return ""

    # Try to find a results-specific section first
    results_text = ""
    for pattern in _RESULTS_SECTION_PATTERNS:
        m = pattern.search(html)
        if m:
            results_text = m.group(1)
            break

    # Use results section if found, otherwise fall back to <body> or full HTML
    if results_text:
        text_source = results_text
    else:
        # Extract body content
        body_match = re.search(r"(?si)<body[^>]*>(.*?)</body>", html)
        text_source = body_match.group(1) if body_match else html

    # Strip script/style blocks
    text_source = re.sub(r"(?si)<(script|style|noscript)[^>]*>.*?</\1>", " ", text_source)
    # Strip HTML tags
    text_source = re.sub(r"<[^>]+>", " ", text_source)
    # Decode common HTML entities
    text_source = (text_source
                   .replace("&amp;", "&").replace("&lt;", "<")
                   .replace("&gt;", ">").replace("&nbsp;", " ")
                   .replace("&#8209;", "-").replace("&#39;", "'")
                   .replace("&quot;", '"'))
    # Collapse whitespace
    text_source = re.sub(r"\s+", " ", text_source).strip()

    if len(text_source) > max_chars:
        # Keep the beginning (trial metadata) and the end (often has
        # results/outcome data) with a marker in the middle.
        half = max_chars // 2
        text_source = (text_source[:half]
                       + " [...PAGE TRUNCATED...] "
                       + text_source[-half:])

    return text_source


async def _prefetch_registry_pages(
    rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Fetch registry page content for each trial that has a URL.

    Returns a dict mapping trial_id -> extracted page text.
    Only fetches pages that have a plausible registry URL (http/https).
    """
    url_map: Dict[str, str] = {}  # trial_id -> url
    for row in rows:
        tid = row.get("trial_id", "")
        url = row.get("url", "") or row.get("source_url", "")
        if tid and url and url.startswith("http"):
            url_map[tid] = url

    if not url_map:
        return {}

    print(f"  Pre-fetching {len(url_map)} registry page(s) for page content ...",
          file=sys.stderr)

    results: Dict[str, str] = {}
    semaphore = asyncio.Semaphore(PREFETCH_CONCURRENCY)

    async def fetch_one(tid: str, url: str):
        async with semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=PREFETCH_TIMEOUT)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Some registries redirect HTTP -> HTTPS or to a
                    # results-specific URL — follow redirects.
                    async with session.get(url, allow_redirects=True,
                                           headers={"User-Agent": "Mozilla/5.0"}) as resp:
                        if resp.status == 200:
                            html = await resp.text(errors="replace")
                            text = _extract_page_text(html)
                            if text and len(text) > 100:
                                results[tid] = text
                            else:
                                # Page was mostly empty / JS-rendered
                                results[tid] = ""
                        else:
                            print(f"    [prefetch] {tid}: HTTP {resp.status}",
                                  file=sys.stderr)
            except asyncio.TimeoutError:
                print(f"    [prefetch] {tid}: timeout after {PREFETCH_TIMEOUT}s",
                      file=sys.stderr)
            except Exception as exc:
                print(f"    [prefetch] {tid}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)

    await asyncio.gather(*[fetch_one(tid, url) for tid, url in url_map.items()])

    fetched_count = sum(1 for v in results.values() if v)
    print(f"  Pre-fetched {fetched_count}/{len(url_map)} page(s) with usable content",
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
    "trial_acronym",
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
        "trial_acronym": row.get("trial_acronym", ""),
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


async def _fetch_ctgov_acronyms(nct_ids: List[str]) -> Dict[str, str]:
    """Fetch official acronyms from the ClinicalTrials.gov API v2.

    The API exposes protocolSection.identificationModule.acronym which is
    the authoritative trial acronym (e.g. "REDEFINE 1", "REIMAGINE 2").
    We batch up to 50 IDs per request using the filter query.
    Returns a dict mapping NCT ID -> acronym (only for IDs that have one).
    """
    result: Dict[str, str] = {}
    if not nct_ids:
        return result

    batch_size = 50
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        for i in range(0, len(nct_ids), batch_size):
            batch = nct_ids[i:i + batch_size]
            ids_filter = " OR ".join(batch)
            params = {
                "query.id": ids_filter,
                "fields": "protocolSection.identificationModule.nctId,"
                          "protocolSection.identificationModule.acronym,"
                          "protocolSection.identificationModule.officialTitle,"
                          "protocolSection.identificationModule.briefTitle",
                "pageSize": str(len(batch)),
            }
            try:
                async with session.get(base_url, params=params) as resp:
                    if resp.status != 200:
                        print(f"    [acronym] CT.gov API returned {resp.status}",
                              file=sys.stderr)
                        continue
                    data = await resp.json()
                    for study in data.get("studies", []):
                        id_mod = (study.get("protocolSection", {})
                                       .get("identificationModule", {}))
                        nct_id = id_mod.get("nctId", "")
                        acronym = id_mod.get("acronym", "")
                        if not acronym:
                            # Fall back to extracting from titles
                            for title_key in ("officialTitle", "briefTitle"):
                                acronym = _extract_trial_acronym(
                                    id_mod.get(title_key, ""))
                                if acronym:
                                    break
                        if nct_id and acronym:
                            result[nct_id] = acronym.strip()
            except Exception as exc:
                print(f"    [acronym] CT.gov API error: {exc}", file=sys.stderr)

    return result


def _assign_acronyms(rows_by_id: Dict[str, Dict[str, Any]],
                     api_acronyms: Dict[str, str]) -> None:
    """Assign trial_acronym to every row using multiple strategies:
    1. CT.gov API acronym (most authoritative)
    2. Extract from title fields
    3. Extract from source_url's NCT ID -> API acronym mapping
    """
    # Build a map from source_url NCT IDs to acronyms for EU trial cross-ref
    url_nct_map: Dict[str, str] = {}
    for nct_id, acronym in api_acronyms.items():
        url_nct_map[nct_id.upper()] = acronym

    for tid, row in rows_by_id.items():
        acronym = ""

        # Strategy 1: direct API match (CT.gov rows)
        if tid in api_acronyms:
            acronym = api_acronyms[tid]

        # Strategy 2: extract NCT ID from source_url and look up
        if not acronym:
            source_url = row.get("source_url", "") or row.get("url", "")
            url_nct = _extract_nct_from_url(source_url)
            if url_nct and url_nct.upper() in url_nct_map:
                acronym = url_nct_map[url_nct.upper()]

        # Strategy 3: extract from title fields
        if not acronym:
            for title_key in ("public_title", "title", "brief_title",
                              "trial_title"):
                title = row.get(title_key, "")
                if title:
                    acronym = _extract_trial_acronym(title)
                    if acronym:
                        break

        row["trial_acronym"] = acronym


def _extract_nct_from_url(url: str) -> str:
    """Extract an NCT ID from a clinicaltrials.gov URL."""
    if not url:
        return ""
    m = re.search(r"(NCT\d{8,})", url, re.IGNORECASE)
    return m.group(1).upper() if m else ""


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
            return trials

    results = await asyncio.gather(*[
        run_batch(i, source, batch) for i, (source, batch) in enumerate(batches)
    ])
    return [t for batch in results for t in batch if isinstance(t, dict)]


def _merge_enriched(enriched_trials: List[Dict[str, Any]],
                     rows_by_id: Dict[str, Dict[str, Any]]) -> set:
    """Merge Gemini's enriched fields back onto the original registry rows,
    matched by trial ID. Only overwrite when Gemini actually returned
    something usable — never blank out data the registry scraper already
    had. Returns the set of trial IDs that got a match."""
    matched_ids = set()
    for trial in enriched_trials:
        trial_id = _clean_trial_id(trial.get("Trial ID", ""))
        row = rows_by_id.get(trial_id)
        if row is None:
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
    """Copy efficacy data between trials that share the same source_url
    or whose source_url points to the same NCT ID.

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

    # Also group by NCT ID extracted from source_url — this catches EU
    # trials whose source_url is a clinicaltrials.gov link
    nct_to_ids: Dict[str, List[str]] = {}
    for tid, row in rows_by_id.items():
        # The row itself might be an NCT row
        if _is_ctgov_id(tid):
            nct_to_ids.setdefault(tid.upper(), []).append(tid)
        # Also extract NCT from URL fields
        for url_key in ("url", "source_url"):
            nct_from_url = _extract_nct_from_url(row.get(url_key, ""))
            if nct_from_url:
                nct_to_ids.setdefault(nct_from_url.upper(), []).append(tid)

    propagated = 0

    # Propagate by URL
    for norm_url, tids in url_to_ids.items():
        if len(set(tids)) < 2:
            continue
        unique_tids = list(dict.fromkeys(tids))  # dedupe preserving order
        donor = None
        for tid in unique_tids:
            if not _efficacy_missing(rows_by_id[tid]):
                donor = rows_by_id[tid]
                break
        if donor is None:
            continue
        for tid in unique_tids:
            recipient = rows_by_id[tid]
            if _efficacy_missing(recipient) and recipient is not donor:
                propagated += _copy_efficacy(donor, recipient)

    # Propagate by NCT ID
    for nct_id, tids in nct_to_ids.items():
        if len(set(tids)) < 2:
            continue
        unique_tids = list(dict.fromkeys(tids))
        donor = None
        for tid in unique_tids:
            if not _efficacy_missing(rows_by_id[tid]):
                donor = rows_by_id[tid]
                break
        if donor is None:
            continue
        for tid in unique_tids:
            recipient = rows_by_id[tid]
            if _efficacy_missing(recipient) and recipient is not donor:
                propagated += _copy_efficacy(donor, recipient)

    if propagated:
        print(f"  URL/NCT propagation: copied efficacy data to {propagated} "
              f"trial(s) sharing URLs/NCT IDs with data-rich siblings",
              file=sys.stderr)


def _copy_efficacy(donor: Dict[str, Any], recipient: Dict[str, Any]) -> int:
    """Copy efficacy + duration fields from donor to recipient.
    Returns 1 if any field was copied, 0 otherwise."""
    copied = False
    for field in EFFICACY_ROW_FIELDS:
        donor_val = str(donor.get(field, "")).strip()
        recip_val = str(recipient.get(field, "")).strip()
        if donor_val and not recip_val:
            recipient[field] = donor[field]
            copied = True
    for dur_field in ("hba1c_duration", "weight_duration",
                      "alt_duration", "mash_duration"):
        donor_val = str(donor.get(dur_field, "")).strip()
        recip_val = str(recipient.get(dur_field, "")).strip()
        if donor_val and not recip_val:
            recipient[dur_field] = donor[dur_field]
            copied = True
    return 1 if copied else 0


def _is_completed_or_has_results(row: Dict[str, Any]) -> bool:
    """Check whether a trial is completed or likely has results available."""
    status = str(row.get("status", "") or row.get("phase_status", "")
                 or row.get("overall_status", "")).lower()
    if any(kw in status for kw in ("completed", "complete", "active",
                                     "terminated", "results")):
        return True
    if str(row.get("has_results", "")).lower() in ("true", "yes", "1"):
        return True
    return False


def _build_acronym_search_suffix(rows: List[Dict[str, Any]]) -> str:
    """Build a Gemini prompt suffix that lists each trial's acronym and
    NCT ID, instructing Gemini to search specifically by acronym."""
    lines = []
    for r in rows:
        acronym = r.get("trial_acronym", "")
        tid = r.get("trial_id", "")
        nct_from_url = _extract_nct_from_url(
            r.get("source_url", "") or r.get("url", ""))
        ref_id = tid if _is_ctgov_id(tid) else (nct_from_url or tid)
        if acronym:
            lines.append(f"  - {tid}: acronym={acronym}, ref_nct={ref_id}")

    return f"""
ACRONYM-TARGETED SEARCH MODE — these trials have known program acronyms
that are highly searchable in press releases, journal publications, and
conference abstracts. For each trial below, you MUST search using BOTH
the trial acronym AND the NCT/trial ID:

{chr(10).join(lines)}

REQUIRED SEARCH STRATEGY (do ALL of these for each trial):
1. Search: "<acronym> results" (e.g. "REDEFINE 1 results")
2. Search: "<acronym> <molecule> weight loss" OR "<acronym> HbA1c"
3. Search: "<acronym> topline" OR "<acronym> headline results"
4. Search: "<acronym> NEJM" OR "<acronym> Lancet" OR "<acronym> published"
5. Search: "<ref_nct> results" if different from trial_id
6. Search: "<molecule> <acronym> press release Novo Nordisk"

These searches will find press releases (GlobeNewsWire, BioSpace),
journal publications (NEJM, Lancet, Diabetes Care), and conference
abstracts (ADA, EASD, AASLD) that contain exact efficacy numbers.
The numbers ARE published for these trials — if you return N/A it means
you didn't search by acronym. DO NOT skip the acronym searches.

IMPORTANT: Report the TREATMENT POLICY ESTIMAND (real-world adherence)
weight/HbA1c values when available, not the trial product estimand.
If both are reported, prefer treatment policy estimand values.
"""


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

    # --- Acronym resolution: fetch from CT.gov API + title extraction ---
    # Do this early so acronyms are available for all Gemini passes.
    # Also resolve acronyms for NCT IDs found in EU trial source_urls.
    url_nct_ids = set()
    for row in rows_by_id.values():
        for uk in ("url", "source_url"):
            nct_from_url = _extract_nct_from_url(row.get(uk, ""))
            if nct_from_url:
                url_nct_ids.add(nct_from_url)
    all_nct_ids = list(ctgov_ids | url_nct_ids)
    print(f"  Fetching acronyms for {len(all_nct_ids)} NCT ID(s) from CT.gov API ...",
          file=sys.stderr)
    api_acronyms = await _fetch_ctgov_acronyms(all_nct_ids)
    print(f"  Got {len(api_acronyms)} acronym(s) from CT.gov API", file=sys.stderr)
    _assign_acronyms(rows_by_id, api_acronyms)

    # Log acronym coverage
    with_acronym = sum(1 for r in rows_by_id.values() if r.get("trial_acronym"))
    print(f"  Acronyms assigned: {with_acronym}/{len(rows_by_id)} trial(s)",
          file=sys.stderr)

    # --- Pre-fetch: hit each trial's registry URL to grab page content ---
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
    if retry_missing:
        retry_rows = [r for r in rows_by_id.values() if _efficacy_missing(r)]
        if retry_rows:
            print(f"  Pass 2: retrying {len(retry_rows)} trial(s) with no efficacy data, "
                  f"1 trial/call, deep-dive prompt ...", file=sys.stderr)
            retry_batches = _chunk_by_source(retry_rows, RETRY_BATCH_SIZE,
                                              prefetched=prefetched)
            retry_enriched = await _run_batches(drug, retry_batches,
                                                 extra_suffix=DEEP_DIVE_SUFFIX, label="pass2")
            retry_matched = _merge_enriched(retry_enriched, rows_by_id)
            still_missing = sum(1 for r in retry_rows if _efficacy_missing(r))
            print(f"  Pass 2: recovered efficacy data for "
                  f"{len(retry_rows) - still_missing}/{len(retry_rows)} previously-empty trial(s)",
                  file=sys.stderr)
        else:
            print("  Pass 2: skipped — no trials missing efficacy data.", file=sys.stderr)

    # --- Pass 3 (no API calls): cross-URL efficacy propagation ---
    _propagate_efficacy_by_url(rows_by_id)

    # --- Pass 4: acronym-targeted Gemini search for COMPLETED trials ---
    # After propagation, some completed trials may still lack efficacy.
    # For each one that has an acronym, do a single focused Gemini call
    # with the acronym explicitly injected into the prompt as the primary
    # search term.
    if retry_missing:
        acronym_retry_rows = [
            r for r in rows_by_id.values()
            if (_efficacy_missing(r)
                and r.get("trial_acronym")
                and _is_completed_or_has_results(r))
        ]
        if acronym_retry_rows:
            print(f"  Pass 4 (acronym search): {len(acronym_retry_rows)} completed "
                  f"trial(s) with acronyms still missing efficacy data ...",
                  file=sys.stderr)
            acronym_batches = _chunk_by_source(acronym_retry_rows, RETRY_BATCH_SIZE,
                                                prefetched=prefetched)
            acronym_enriched = await _run_batches(
                drug, acronym_batches,
                extra_suffix=_build_acronym_search_suffix(acronym_retry_rows),
                label="pass4-acronym")
            acronym_matched = _merge_enriched(acronym_enriched, rows_by_id)
            still_missing = sum(1 for r in acronym_retry_rows if _efficacy_missing(r))
            print(f"  Pass 4: recovered efficacy for "
                  f"{len(acronym_retry_rows) - still_missing}/{len(acronym_retry_rows)} "
                  f"trial(s)", file=sys.stderr)
        else:
            print("  Pass 4: skipped — no completed+acronym trials missing efficacy.",
                  file=sys.stderr)

    # --- Final propagation pass after acronym search may have filled donors ---
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
            if row:
                row["trial_start_date"] = dates["Start Date"]
                row["trial_completion_date"] = dates["Completion Date"]

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
    print(f"  Final: {len(rows) - still_empty}/{len(rows)} trial(s) have at least "
          f"some efficacy data; {still_empty} have none (likely no results published yet).",
          file=sys.stderr)

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