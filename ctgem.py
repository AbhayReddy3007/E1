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
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import aiohttp

from gemini_extractor import (
    MAX_WORKERS,
    RATE_LIMIT_DELAY,
    get_trial_details,
)
# NOTE: gemini_extractor's fetch_ctgov_dates (dates only) is superseded by
# this file's own _fetch_ctgov_authoritative(), which fetches status and
# enrollment from the same CT.gov API alongside dates — see that function
# for why status/enrollment needed the same authoritative-override
# treatment dates already had.

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
#
# IMPORTANT: this list must contain ONLY the actual clinical-outcome
# fields (HbA1c / weight / ALT / MASH). "dosage" and "company_name" used
# to be included here too, but that was a bug: Gemini's first pass almost
# always finds the sponsor name and dosage (they're easy — right on the
# registry page), which made _efficacy_missing() return False as soon as
# those two were filled, even when every real outcome field was still
# blank. That silently skipped Pass 2 (the focused single-trial retry)
# for the majority of trials that actually needed it, which is why
# Completed trials with real published results were still showing empty
# hba1c/weight/alt/mash columns. dosage/company_name are still requested
# from Gemini in the prompt — they're just no longer allowed to mask a
# missing efficacy value.
EFFICACY_ROW_FIELDS = [
    "hba1c_change_pct", "weight_change_pct",
    "alt_reduction_pct", "mash_resolution_pct",
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
PREFETCH_CONCURRENCY = 20  # CT.gov API is high-throughput; raise from 10

# ---------------------------------------------------------------------------
# Network availability: probe at startup whether external prefetch hosts are
# reachable. In sandbox environments (e.g. Claude.ai code execution) the
# egress proxy only allows a small allowlist — clinicaltrials.gov, PubMed,
# and press-release domains are all blocked (HTTP 403 with "Host not in
# allowlist"). When reachability is False every prefetch is skipped and the
# grounding check and source-URL enforcement are disabled: Gemini's own
# Google-Search-grounded response is the only source we have, so blocking
# on grounding would erase every correct value it finds.
# ---------------------------------------------------------------------------
_PREFETCH_PROBE_URL = "https://clinicaltrials.gov/api/v2/studies?pageSize=1"
_PREFETCH_PROBE_TIMEOUT = 5
_NETWORK_AVAILABLE: Optional[bool] = None   # lazily set by _probe_network()


async def _probe_network() -> bool:
    """Return True if external prefetch hosts are reachable, False if blocked.
    Result is cached in _NETWORK_AVAILABLE after the first call."""
    global _NETWORK_AVAILABLE
    if _NETWORK_AVAILABLE is not None:
        return _NETWORK_AVAILABLE
    try:
        timeout = aiohttp.ClientTimeout(total=_PREFETCH_PROBE_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(_PREFETCH_PROBE_URL) as resp:
                body = await resp.text()
                # 403 with "not in allowlist" = sandbox egress block
                if resp.status == 403 and "allowlist" in body.lower():
                    _NETWORK_AVAILABLE = False
                else:
                    _NETWORK_AVAILABLE = (resp.status < 500)
    except Exception:
        _NETWORK_AVAILABLE = False
    if not _NETWORK_AVAILABLE:
        print("  [info] External prefetch hosts are not reachable (sandbox/egress "
              "restriction). Skipping CT.gov page, PubMed, and press-release "
              "prefetches. Grounding check and source-URL enforcement are "
              "disabled — Gemini's own search results are the sole evidence "
              "source for this run.", file=sys.stderr)
    return _NETWORK_AVAILABLE

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


# Maps CT.gov API v2's enum-style OverallStatus values to the Title Case
# style already used elsewhere in this pipeline's output (e.g. "Completed",
# "Active, not recruiting"), so overriding phase_status from the API
# doesn't change the sheet's formatting conventions.
_CTGOV_STATUS_DISPLAY = {
    "COMPLETED": "Completed",
    "RECRUITING": "Recruiting",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "SUSPENDED": "Suspended",
    "UNKNOWN": "Unknown status",
    "AVAILABLE": "Available",
    "NO_LONGER_AVAILABLE": "No longer available",
    "APPROVED_FOR_MARKETING": "Approved for marketing",
}


async def _fetch_ctgov_authoritative(nct_ids: List[str]) -> Dict[str, Dict[str, str]]:
    """Fetch OverallStatus, EnrollmentCount, StartDate and
    PrimaryCompletionDate directly from the CT.gov API v2 for a batch of
    NCT IDs — the single authoritative source for these fields.

    WHY THIS EXISTS: manual verification found phase_status and trial_size
    flipping between different values (e.g. Recruiting <-> Completed,
    enrollment 300 <-> 626) for the SAME NCT ID across separate runs of
    this pipeline, and one row showing phase_status "Completed" together
    with a trial_completion_date in the FUTURE — a logical impossibility.
    Both are symptoms of the same root cause: status and enrollment were
    being taken from Gemini's free-text search interpretation, which can
    vary run to run, instead of from the structured registry record. Dates
    were already cross-checked against this API (see the call site below);
    this extends the same authoritative-override treatment to status and
    enrollment for CT.gov-native rows, where there is no ambiguity about
    which source is correct.
    """
    if not nct_ids:
        return {}
    if not await _probe_network():
        return {}

    fields = "NCTId,OverallStatus,EnrollmentCount,StartDateStruct,PrimaryCompletionDateStruct"

    async def fetch_one(session: aiohttp.ClientSession, nct_id: str):
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
        try:
            async with session.get(url, params={"fields": fields},
                                    timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return nct_id, None
                data = await resp.json()
                protocol = data.get("protocolSection", {}) if isinstance(data, dict) else {}
                status_mod = protocol.get("statusModule", {})
                design_mod = protocol.get("designModule", {})
                raw_status = status_mod.get("overallStatus", "") or ""
                enrollment = design_mod.get("enrollmentInfo", {}).get("count")
                start = status_mod.get("startDateStruct", {}).get("date") or ""
                completion = status_mod.get("primaryCompletionDateStruct", {}).get("date") or ""
                return nct_id, {
                    "status": _CTGOV_STATUS_DISPLAY.get(raw_status.upper(), raw_status.title() if raw_status else ""),
                    "enrollment": str(enrollment) if enrollment is not None else "",
                    "start_date": start,
                    "completion_date": completion,
                }
        except Exception as exc:
            print(f"  [warn] CT.gov authoritative fetch failed for {nct_id}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return nct_id, None

    results: Dict[str, Dict[str, str]] = {}
    semaphore = asyncio.Semaphore(PREFETCH_CONCURRENCY)

    async def bounded_fetch(session, nct_id):
        async with semaphore:
            return await fetch_one(session, nct_id)

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pairs = await asyncio.gather(*[bounded_fetch(session, nid) for nid in nct_ids])
    for nid, info in pairs:
        if info:
            results[nid] = info
    return results


async def _prefetch_registry_pages(
    rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Fetch supporting content for each trial that has a trial ID / URL.

    Returns a dict mapping trial_id -> extracted text. NCT-prefixed trials
    use CT.gov's structured JSON API (reliable, includes results); other
    trials fall back to fetching and text-extracting their registry URL.
    """
    if not await _probe_network():
        return {}
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


# ---------------------------------------------------------------------------
# PubMed pre-fetch: search NCBI's free, keyless E-utilities API directly for
# a published paper on each trial, and hand Gemini the abstract text as
# pre-fetched context — the same way _prefetch_registry_pages hands it
# CT.gov API text.
#
# WHY THIS EXISTS: manual verification of a prior run found real, indexed
# results that the pipeline still returned blank for — most notably
# NCT05813925 (REDEFINE 5), which has a full published paper (Yamauchi et
# al., Lancet Diabetes & Endocrinology, June 2026) findable in under a
# second via PubMed's own search API. The batch prompt already instructs
# Gemini to search "<Acronym> site:pubmed.ncbi.nlm.nih.gov" itself, but
# that depends on Gemini's own search grounding actually surfacing it
# within its search budget for that trial. Querying PubMed directly in
# Python removes that dependency entirely for the journal-publication
# channel: instead of asking Gemini to go find the paper, we find it
# ourselves and paste the abstract straight into its prompt, the same
# "verify against a primary source" approach used to confirm this gap by
# hand. Press releases (globenewswire/prnewswire/biospace/etc.) still rely
# on Gemini's own search — there's no equivalent free, keyless, structured
# API for wire-service press releases the way there is for PubMed.
# ---------------------------------------------------------------------------

PUBMED_CONCURRENCY = 5          # NCBI allows ~3 req/s unauthenticated; 2 calls per
                                # trial (esearch + efetch) so 5 concurrent trials
                                # stays safely under the limit
PUBMED_TIMEOUT = 15
PUBMED_MAX_IDS = 3              # top N matching papers to pull abstracts for
PUBMED_MAX_CHARS = 4000         # keep prompt size reasonable per trial

_EUTILS_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EUTILS_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


async def _pubmed_search(session: aiohttp.ClientSession, term: str) -> List[str]:
    """Run one PubMed esearch query, return matching PMIDs (best-effort,
    never raises — a failed/empty search just means no prefetch for that
    query, Gemini's own search is still tried as normal)."""
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(PUBMED_MAX_IDS)}
    try:
        async with session.get(_EUTILS_ESEARCH, params=params,
                                timeout=aiohttp.ClientTimeout(total=PUBMED_TIMEOUT)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return data.get("esearchresult", {}).get("idlist", []) or []
    except Exception as exc:
        print(f"    [pubmed] search failed for {term!r}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return []


async def _pubmed_fetch_abstracts(session: aiohttp.ClientSession, pmids: List[str]) -> str:
    """Fetch plain-text abstracts for a list of PMIDs in one call."""
    if not pmids:
        return ""
    params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "text"}
    try:
        async with session.get(_EUTILS_EFETCH, params=params,
                                timeout=aiohttp.ClientTimeout(total=PUBMED_TIMEOUT)) as resp:
            if resp.status != 200:
                return ""
            text = await resp.text()
            return text.strip()[:PUBMED_MAX_CHARS]
    except Exception as exc:
        print(f"    [pubmed] fetch failed for {pmids}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return ""


async def _prefetch_pubmed_one(row: Dict[str, Any], session: aiohttp.ClientSession,
                                semaphore: asyncio.Semaphore) -> tuple:
    """Search PubMed for one trial via its acronym first (far more likely to
    hit — trial papers are almost always titled/indexed by program name,
    e.g. "REDEFINE 5", not by NCT number), falling back to the raw trial ID.
    Returns (trial_id, abstract_text_or_empty)."""
    trial_id = row.get("trial_id", "")
    title = (row.get("public_title", "") or row.get("title", "")
             or row.get("brief_title", "") or "")
    acronym = _extract_trial_acronym(title)

    queries = []
    if acronym:
        # Quote the acronym and anchor to title/abstract so e.g. "STEP 1"
        # doesn't match unrelated papers that merely mention "step 1" of a
        # procedure.
        queries.append(f'"{acronym}"[Title/Abstract]')
    if trial_id:
        queries.append(f'"{trial_id}"')
    if not queries:
        return trial_id, ""

    async with semaphore:
        pmids: List[str] = []
        for q in queries:
            pmids = await _pubmed_search(session, q)
            if pmids:
                break
        if not pmids:
            return trial_id, ""
        text = await _pubmed_fetch_abstracts(session, pmids)
        return trial_id, text


async def _prefetch_pubmed(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """For every trial with a recognisable acronym (or, failing that, its
    raw trial ID), query PubMed directly and return {trial_id: abstract
    text}. Best-effort and non-fatal: any trial with no acronym, no PubMed
    hit, or a failed request just gets no entry, same as before this
    prefetch existed."""
    if not await _probe_network():
        return {}
    candidates = [r for r in rows if r.get("trial_id")]
    if not candidates:
        return {}

    print(f"  Pre-fetching PubMed abstracts for {len(candidates)} trial(s) "
          f"(direct NCBI E-utilities lookup) ...", file=sys.stderr)

    results: Dict[str, str] = {}
    semaphore = asyncio.Semaphore(PUBMED_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=PUBMED_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        pairs = await asyncio.gather(*[
            _prefetch_pubmed_one(row, session, semaphore) for row in candidates
        ])
    for trial_id, text in pairs:
        if text:
            results[trial_id] = text

    print(f"  Pre-fetched PubMed abstracts for {len(results)}/{len(candidates)} trial(s)",
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
    "HbA1c Rationale": "hba1c_rationale",
    "Weight Loss (%)": "weight_change_pct",
    "Weight Duration": "weight_duration",
    "Weight Rationale": "weight_rationale",
    "ALT Reduction (%)": "alt_reduction_pct",
    "ALT Duration": "alt_duration",
    "ALT Rationale": "alt_rationale",
    "MASH Outcome (%)": "mash_resolution_pct",
    "MASH Duration": "mash_duration",
    "MASH Rationale": "mash_rationale",
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
    "mash_resolution_pct",
    "mash_duration",
    "mash_rationale",
    "mash_confidence",
    "company_name",
    "source_url",
    # Per-value audit trail: "registry", "pubmed", or "unverified" per field.
    "efficacy_provenance",
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


def _format_trial_id_with_acronym(row: Dict[str, Any]) -> str:
    """Return trial_id with acronym appended if one can be extracted, e.g.
    'NCT05394519 (REDEFINE 2)'. Falls back to the bare trial_id when no
    recognisable acronym is present in any title field."""
    trial_id = row.get("trial_id", "")
    title = (row.get("trial_title", "") or row.get("public_title", "")
             or row.get("title", "") or row.get("brief_title", "") or "")
    acronym = _extract_trial_acronym(title)
    if acronym:
        return f"{trial_id} ({acronym})"
    return trial_id


def remap_row(row: Dict[str, Any], drug: str) -> Dict[str, Any]:
    """Collapse a merged registry+Gemini row down to exactly FINAL_COLUMNS.

    Falls back to the raw registry field name (e.g. 'title', 'countries',
    'status' — shared by ctgov_trials.py and euct_trials.py via
    registry_common) if the Gemini-named field wasn't filled in.
    """
    return {
        "molecule_name": drug,
        "registry_source": row.get("source", "") or row.get("registry_source", ""),
        "trial_id": _format_trial_id_with_acronym(row),
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
        "hba1c_rationale": row.get("hba1c_rationale", ""),
        "hba1c_confidence": row.get("hba1c_confidence", ""),
        "weight_change_pct": row.get("weight_change_pct", ""),
        "weight_duration": row.get("weight_duration", ""),
        "weight_rationale": row.get("weight_rationale", ""),
        "weight_confidence": row.get("weight_confidence", ""),
        "alt_reduction_pct": row.get("alt_reduction_pct", ""),
        "alt_duration": row.get("alt_duration", ""),
        "alt_rationale": row.get("alt_rationale", ""),
        "alt_confidence": row.get("alt_confidence", ""),
        "mash_resolution_pct": row.get("mash_resolution_pct", ""),
        "mash_duration": row.get("mash_duration", ""),
        "mash_rationale": row.get("mash_rationale", ""),
        "mash_confidence": row.get("mash_confidence", ""),
        "company_name": row.get("company_name", "") or row.get("sponsor", ""),
        "source_url": row.get("source_url", "") or row.get("url", ""),
        "efficacy_provenance": row.get("efficacy_provenance", ""),
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


def _detect_acronym_collisions(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Find program acronyms that resolve to more than one distinct trial_id
    in this run (e.g. two unrelated trials both informally called
    "REDEFINE 4" in press coverage). An acronym-based search is one of the
    most effective ways to find press-release efficacy data (see
    TRIAL ACRONYM SEARCH STRATEGY below) but it's also exactly the kind of
    search that can silently misattribute a press release to the wrong
    trial ID, or cause the model to give up as "ambiguous", when the
    acronym isn't unique. Returns {acronym: [trial_id, ...]} only for
    acronyms with 2+ distinct trial IDs.
    """
    acronym_to_ids: Dict[str, set] = {}
    for row in rows:
        tid = row.get("trial_id", "")
        if not tid:
            continue
        title = (row.get("public_title", "") or row.get("title", "")
                 or row.get("brief_title", "") or "")
        acronym = _extract_trial_acronym(title)
        if acronym:
            acronym_to_ids.setdefault(acronym, set()).add(tid)
    return {a: sorted(ids) for a, ids in acronym_to_ids.items() if len(ids) > 1}


def _to_gemini_trial_stub(row: Dict[str, Any],
                          prefetched: Optional[Dict[str, str]] = None,
                          acronym_collisions: Optional[Dict[str, List[str]]] = None,
                          pubmed_prefetched: Optional[Dict[str, str]] = None,
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
        if acronym_collisions and acronym in acronym_collisions:
            other_ids = [i for i in acronym_collisions[acronym] if i != trial_id]
            if other_ids:
                context_parts.append(
                    f"ACRONYM COLLISION WARNING: The program name/acronym "
                    f"'{acronym}' is ALSO used (by outside press coverage) for "
                    f"a DIFFERENT trial in this dataset: {', '.join(other_ids)}. "
                    f"Before using any '{acronym}'-based search result for THIS "
                    f"trial ({trial_id}), verify the source actually reports on "
                    f"NCT/trial ID {trial_id} specifically (matching phase, "
                    f"enrollment size, and countries below) — do not assign "
                    f"results to this row if they actually belong to "
                    f"{', '.join(other_ids)}."
                )

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

    # Inject a pre-fetched PubMed abstract if a matching publication was
    # found via direct NCBI search (see _prefetch_pubmed) — this is often
    # the single richest source for HbA1c/weight/ALT/MASH numbers, since
    # trial-result papers report exact figures the registry page doesn't.
    if pubmed_prefetched and trial_id in pubmed_prefetched and pubmed_prefetched[trial_id]:
        stub["Prefetched_Publication_Content"] = pubmed_prefetched[trial_id]

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

PRESS-RELEASE WIRE SEARCH (do this even when the registry page shows no
posted results — sponsors routinely announce topline efficacy numbers by
press release MONTHS before the registry results section is updated):
- Search: "<Sponsor>" + "<Acronym or Trial ID>" + "results"
- Search: "<Sponsor>" + "<Acronym>" site:globenewswire.com OR site:prnewswire.com
  OR site:businesswire.com — these wire services carry the large majority
  of pharma trial-readout press releases and are usually indexed even for
  very recent (same-month) announcements.
- Search: "<Sponsor>" + "<indication>" + "phase 3" + "topline 20XX" (use the
  current or preceding year)
- Search: "<Molecule/Program>" + "vs" + "<comparator drug, if head-to-head>"
  + "results" — head-to-head trials (e.g. against a competitor drug) are
  often covered by trade press (e.g. clinicaltrialsarena.com, fiercebiotech
  .com, endpoints news) even before the sponsor's own press release, and
  that coverage frequently contains the exact efficacy numbers.

DO NOT SKIP SEARCHING JUST BECAUSE REGISTRY STATUS LOOKS EARLY:
The "Status" field in Registry_Context is a snapshot from when the
registry was last scraped and can be STALE — a trial shown as "Recruiting"
or "Active" here may have actually completed and reported results since
that snapshot was taken. Never use "Status: Recruiting/Active" as a reason
to skip the press-release and acronym searches above — search regardless
of what the registry status says, and only conclude "N/A" after those
searches turn up nothing.

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

PRE-FETCHED PUBLICATION CONTENT (PubMed):
Some trials include a "Prefetched_Publication_Content" field — the actual
abstract text of a PubMed-indexed paper matched to this trial by acronym
or trial ID via a direct NCBI database lookup (not a Gemini search). This
is a REAL, CONFIRMED publication about this exact trial:
- Treat it as a PRIMARY, HIGH-CONFIDENCE source — trial-result papers
  report exact efficacy numbers (HbA1c change, weight change, ALT
  reduction, MASH/NASH resolution, with their measurement duration) that
  are often more precise and complete than press releases.
  extract every efficacy value it reports before doing anything else.
- If this field is present, do NOT report "N/A" for an efficacy field
  the abstract clearly states — that would be ignoring evidence you were
  handed. Read the numbers out of the abstract text directly.
- The abstract may be truncated; if a number is cut off or the abstract
  only gives a qualitative result ("significantly reduced"), still search
  as normal for the precise figure before falling back to N/A.
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
- ALWAYS go to clinicaltrials.gov/study/<ID> → "Study Results" tab to check
  for posted efficacy endpoints — don't stop at the summary page, and don't
  skip this because Registry_Context doesn't show "Results Posted: True":
  that flag reflects an old scrape snapshot and can be wrong for a trial
  that has since completed and posted results.
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
0. FIRST check whether a "Prefetched_Publication_Content" field is present
   below. If it is, it's the abstract of a real PubMed-indexed paper about
   THIS trial, fetched directly from NCBI (not a search result Gemini has
   to find) — extract every efficacy number it contains before doing
   anything else. Do not report N/A for a field this abstract answers.
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
   - Search for: "<Sponsor Name>" + "<Program Name>" site:globenewswire.com
     OR site:prnewswire.com OR site:businesswire.com — wire services carry
     most pharma topline-result announcements and index them fast, often
     the SAME MONTH as the announcement.
   - Search for: "<Sponsor Name>" + "<Program Name>" site:pubmed.ncbi.nlm.nih.gov
   - If this is a head-to-head trial against a named comparator drug, also
     search: "<Program Name>" + "vs" + "<comparator>" + "results" — trade
     press (clinicaltrialsarena.com, fiercebiotech.com, endpts.com) often
     covers head-to-head readouts, sometimes ahead of the sponsor's own
     release, and usually states the exact numbers.
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
7. Do NOT let "Status: Recruiting" or "Status: Active" in Registry_Context
   talk you out of searching — that status can be a stale snapshot from
   when the registry was scraped. A trial can complete and have its topline
   results covered by press within days, well before any registry page or
   scrape catches up. Search regardless of what the status field says.
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
    ], return_exceptions=True)

    # A batch that still failed after gemini_call_with_search's own retries
    # (e.g. Gemini returned 503 on every attempt) shouldn't take the whole
    # run down with it — every OTHER batch's results are still good and
    # worth keeping. Log the failure, treat that batch as having returned
    # nothing, and let the normal "still missing" bookkeeping (Pass 2 retry,
    # the final [warn] summary) pick up its trials instead.
    flattened: List[Dict[str, Any]] = []
    failed_batches = 0
    for (source, batch), result in zip(batches, results):
        if isinstance(result, Exception):
            failed_batches += 1
            ids = [s.get("NCT_ID", "?") for s in batch]
            print(f"  [{label}] ❌ Batch [{source}] permanently failed after "
                  f"retries ({type(result).__name__}: {result}) — trial(s) "
                  f"{', '.join(ids)} got NO data from this batch and will be "
                  f"picked up by the retry pass if one runs.", file=sys.stderr)
            continue
        # Guard: result should be a list of dicts, but Gemini can return
        # malformed responses where strings or other types leak in.
        for item in result:
            if isinstance(item, dict):
                flattened.append(item)
            else:
                print(f"  [{label}] [warn] Skipping non-dict item in batch "
                      f"result: {type(item).__name__}: {str(item)[:80]}",
                      file=sys.stderr)

    if failed_batches:
        print(f"  [{label}] {failed_batches}/{len(batches)} batch(es) failed "
              f"outright — see ❌ lines above. Run continuing with the rest.",
              file=sys.stderr)

    return flattened


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
        # Guard: Gemini sometimes returns bare strings or other non-dict
        # items in its response (e.g. an error message mixed into the
        # JSON array). Skip anything that isn't a proper trial dict.
        if not isinstance(trial, dict):
            continue
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


def _group_related_trial_ids(rows_by_id: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    """Group trial IDs that represent the SAME underlying trial across
    registries, linked by either a shared URL or a secondary/cross-reference
    ID (union-find, so a chain of shared signals groups all related rows
    together). Shared by both the gap-filling propagation below and the
    conflict-reconciliation pass that runs after it.
    """
    ids = list(rows_by_id.keys())
    parent = {tid: tid for tid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Link via shared normalised URL.
    url_to_ids: Dict[str, List[str]] = {}
    for tid, row in rows_by_id.items():
        for url_key in ("url", "source_url"):
            norm = _normalise_url(row.get(url_key, ""))
            if norm:
                url_to_ids.setdefault(norm, []).append(tid)
    for tids in url_to_ids.values():
        for other in tids[1:]:
            union(tids[0], other)

    # Link via a secondary/cross-reference ID that names another trial_id
    # already in this run (most commonly an EU row listing its CT.gov NCT
    # number, or vice versa).
    nct_pattern = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)
    for tid, row in rows_by_id.items():
        secondary = str(row.get("secondary_ids", "") or row.get("other_ids", ""))
        if not secondary:
            continue
        for match in nct_pattern.findall(secondary):
            match_norm = match.upper()
            if match_norm in rows_by_id and match_norm != tid:
                union(tid, match_norm)
        # Also check for a bare EU CTIS-style number (YYYY-NNNNNN-NN-NN) or
        # legacy EudraCT number (YYYY-NNNNNN-NN) embedded in the secondary
        # IDs of a CT.gov row, in case the linkage runs the other direction.
        for other_tid in rows_by_id:
            if other_tid != tid and other_tid in secondary:
                union(tid, other_tid)

    groups: Dict[str, List[str]] = {}
    for tid in ids:
        groups.setdefault(find(tid), []).append(tid)
    return list(groups.values())


def _propagate_efficacy_by_url(rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    """Copy efficacy data between rows that represent the SAME underlying
    trial across registries (see _group_related_trial_ids). Fills gaps
    only — a recipient with NO efficacy data at all gets a data-rich
    sibling's values. This is free — no Gemini calls needed. Disagreements
    between rows that BOTH already have data are handled separately by
    _reconcile_conflicting_efficacy, which runs after this.
    """
    groups = _group_related_trial_ids(rows_by_id)

    propagated = 0
    for tids in groups:
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
        print(f"  Cross-registry propagation: copied efficacy data to "
              f"{propagated} trial(s) sharing a URL or secondary ID with a "
              f"data-rich sibling", file=sys.stderr)


# (value_field, duration_field) pairs — kept together everywhere a
# conflict is resolved or an orphan duration is dropped, so a duration
# never ends up detached from (or mismatched with) the value it belongs to.
_EFFICACY_FIELD_PAIRS = [
    ("hba1c_change_pct", "hba1c_duration"),
    ("weight_change_pct", "weight_duration"),
    ("alt_reduction_pct", "alt_duration"),
    ("mash_resolution_pct", "mash_duration"),
]


def _reconcile_conflicting_efficacy(rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    """Fix the "same trial, different numbers in different rows of the same
    output" problem found during manual verification (e.g. REDEFINE 2 /
    NCT05394519 showing HbA1c as 2.0, 2.2, AND 1.8 across its own CT.gov +
    EU duplicate rows — only one of those can be right).

    For every group of rows that represent the same underlying trial (see
    _group_related_trial_ids), and for every efficacy value/duration pair:
    if the group has MORE THAN ONE distinct non-empty value, that's an
    internal contradiction, not a legitimate difference — the same trial
    can't have two different real HbA1c changes. Resolve by majority vote
    (the value repeated across the most rows/duplicate registry entries is
    far more likely to be the one grounded in a real source, since a
    one-off hallucination is unlikely to be independently repeated) and
    apply that single value — value AND its paired duration together, as
    a unit — to every row in the group, so the final output is internally
    consistent. Every conflict is logged so it can be spot-checked rather
    than silently trusted.
    """
    groups = _group_related_trial_ids(rows_by_id)
    conflicts_found = 0

    for tids in groups:
        if len(tids) < 2:
            continue
        rows = [rows_by_id[tid] for tid in tids]

        for value_field, duration_field in _EFFICACY_FIELD_PAIRS:
            # Collect (value, duration) pairs as they actually co-occur on
            # each row, keyed by the value string, so we resolve the value
            # and its correct duration together rather than mixing and
            # matching durations from different rows.
            pair_by_value: Dict[str, str] = {}
            votes: Dict[str, int] = {}
            grounded_values: set = set()
            for row in rows:
                val = str(row.get(value_field, "")).strip()
                if not val:
                    continue
                dur = str(row.get(duration_field, "")).strip()
                votes[val] = votes.get(val, 0) + 1
                # Did the grounding check confirm THIS value against a real
                # retrieved source? Cross-registry propagation means a
                # hallucinated number can be copied into several sibling
                # rows and then win a naive popularity contest, so evidence
                # has to outrank vote count.
                prov = str(row.get("efficacy_provenance", ""))
                label = value_field.replace("_change_pct", "").replace("_reduction_pct", "") \
                                   .replace("_resolution_pct", "")
                if f"{label}=registry" in prov or f"{label}=pubmed" in prov:
                    grounded_values.add(val)
                # Prefer the first duration seen paired with this value;
                # if a later row has the same value but a non-empty
                # duration and we don't have one yet, take it.
                if val not in pair_by_value or (not pair_by_value[val] and dur):
                    pair_by_value[val] = dur

            distinct_values = list(votes.keys())
            if len(distinct_values) <= 1:
                continue  # no conflict — 0 or 1 distinct value, nothing to do

            # Majority vote; ties broken by preferring the value that has a
            # non-empty paired duration (more likely to have come from an
            # actual source rather than a bare guess), then by the value
            # itself for a stable, reproducible result.
            # Rank: (1) grounded in a real retrieved source beats ungrounded,
            # ALWAYS — a value confirmed in the CT.gov record or a
            # publication outranks one that merely appears in more rows;
            # (2) then vote count; (3) then having a paired duration;
            # (4) then the value itself, for stable output.
            def _sort_key(v: str):
                return (0 if v in grounded_values else 1,
                        -votes[v],
                        0 if pair_by_value.get(v) else 1,
                        v)

            winner = sorted(distinct_values, key=_sort_key)[0]
            winner_duration = pair_by_value.get(winner, "")
            basis = "grounded in a retrieved source" if winner in grounded_values \
                    else "majority vote — NONE of these values were grounded"

            conflicts_found += 1
            print(f"  [warn] Conflicting {value_field} across {len(tids)} row(s) "
                  f"for the same trial ({', '.join(tids)}): "
                  f"{dict(votes)} — keeping {winner!r} ({basis}) "
                  f"for all rows in this group", file=sys.stderr)

            for row in rows:
                row[value_field] = winner
                row[duration_field] = winner_duration

    if conflicts_found:
        print(f"  Reconciled {conflicts_found} conflicting efficacy field(s) "
              f"across duplicate-registry trial groups", file=sys.stderr)


def _drop_orphan_durations(rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    """Clear a duration field whenever its paired value field is empty.

    A genuine extraction always has BOTH a number and the timepoint it was
    measured at — Gemini reporting a duration with no accompanying value is
    the signature of a hallucination (e.g. it saw "68 wk" as the trial's
    overall duration somewhere and pasted it into an outcome-duration field
    even though it never found — or the trial never reports — that
    outcome). Dropping these prevents a misleading "at least we know the
    timepoint" cell that implies data exists when it doesn't.
    """
    dropped = 0
    for row in rows_by_id.values():
        for value_field, duration_field in _EFFICACY_FIELD_PAIRS:
            if not str(row.get(value_field, "")).strip() and str(row.get(duration_field, "")).strip():
                row[duration_field] = ""
                dropped += 1
    if dropped:
        print(f"  Dropped {dropped} orphan duration value(s) with no "
              f"accompanying efficacy number (likely hallucinated)", file=sys.stderr)


def _fix_mismatched_source_urls(rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    """A CT.gov-native row's source_url must point to that row's OWN NCT
    ID. Gemini occasionally returns a URL for a different (similarly-named
    or same-program) trial instead of echoing the ID it was given — e.g. a
    row for NCT07253285 coming back with a source_url for NCT06532851.
    When that happens, discard Gemini's URL and fall back to the registry
    scraper's own url/source_url field (or, failing that, the canonical
    clinicaltrials.gov URL built from the row's own trial_id) rather than
    keep a URL that silently points a reader at the wrong trial.
    """
    nct_in_url = re.compile(r"NCT\d{8}", re.IGNORECASE)
    fixed = 0
    for tid, row in rows_by_id.items():
        if not _is_ctgov_id(tid):
            continue
        url = str(row.get("source_url", "") or "")
        match = nct_in_url.search(url)
        if match and match.group(0).upper() == tid.upper():
            continue  # URL matches this row's own ID — fine
        if not url:
            continue  # nothing to fix
        # Mismatch (or a URL with no recognisable NCT ID at all) — revert
        # to the registry's own url field if it correctly matches, else
        # rebuild the canonical URL from the row's own trial_id.
        registry_url = str(row.get("url", "") or "")
        if nct_in_url.search(registry_url) and \
           nct_in_url.search(registry_url).group(0).upper() == tid.upper():
            row["source_url"] = registry_url
        else:
            row["source_url"] = f"https://clinicaltrials.gov/study/{tid}"
        fixed += 1
    if fixed:
        print(f"  Fixed {fixed} source_url value(s) that pointed at a "
              f"different trial's NCT ID than the row's own", file=sys.stderr)


def _numeric_variants(value: str) -> List[str]:
    """Surface forms the same number might legitimately take in source text.

    A source may print 12.0 as "12", 2.30 as "2.3", or 20.4 as "20.4%".
    We generate the plausible spellings so a real, correctly-extracted
    value isn't flagged as ungrounded purely over formatting.
    """
    raw = str(value or "").strip().replace("%", "").replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not m:
        return []
    num = m.group(0)
    variants: set = {num}
    try:
        f = float(num)
    except ValueError:
        return list(variants)
    # 12.0 -> "12";  2.30 -> "2.3";  13 -> "13.0"
    if f == int(f):
        variants.add(str(int(f)))
        variants.add(f"{int(f)}.0")
    variants.add(f"{f:g}")
    variants.add(f"{f:.1f}")
    variants.add(f"{f:.2f}")
    return [v for v in variants if v]


def _grounding_match(value: str, text: str) -> bool:
    """Check whether a numeric value is present in text as a standalone
    number — NOT as a substring of a larger number.

    Plain substring matching fails badly for small decimals: '1.8' is a
    substring of '7.8', '35.8', '1.80', etc., so a grounding check using
    `v in text` will falsely confirm any HbA1c reduction of 1.8 against
    registry text that merely mentions a baseline HbA1c of 7.8% or a BMI
    of 35.8. The fix is to require word boundaries around the number, so
    '1.8' only matches when it appears as an isolated token in the source
    (possibly preceded by '-' for a reduction, possibly followed by '%').

    We also explicitly require that a NEGATIVE-direction indicator
    ('-', '−', 'reduction', 'decrease', 'loss', 'lower') appears
    NEARBY (within 60 chars before the number) for clinical-outcome
    fields, because a registry page discussing 'baseline HbA1c 7.8%' or
    'BMI 35.8' will mention the number in a positive context, not in the
    context of a treatment effect. This is the key distinction between
    "this number is on this page" and "this page describes this result".
    """
    if not text or not value:
        return False
    variants = _numeric_variants(value)
    if not variants:
        return False

    # Build one combined alternation pattern for all surface forms.
    # Require: optional leading '-'/'−', then the number, then optional '%'.
    # Require word boundaries on both sides so '1.8' doesn't match inside '7.8'.
    escaped = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    number_pattern = rf"(?<![0-9.])(?:-|−)?(?:{escaped})(?:%|\s|$|[^0-9.])"

    for m in re.finditer(number_pattern, text):
        # Found the number as a standalone token. Now verify it appears in
        # a reduction/change context within a 80-char window before it, to
        # avoid matching the same digits in completely unrelated contexts
        # (baseline values, BMIs, ages, etc.).
        start = max(0, m.start() - 80)
        window = text[start:m.end()].lower()
        reduction_cues = (
            "reduction", "reduced", "decrease", "decreased", "change",
            "loss", "lost", "lower", "lowering", "improvement", "improved",
            "percentage point", "%-point", "%−point", "mean change",
            "estimated mean", "hba1c", "weight loss", "body weight",
            "alt", "alanine", "mash", "nash", "fibrosis",
        )
        if any(cue in window for cue in reduction_cues):
            return True

    return False


def _verify_efficacy_grounding(
    rows_by_id: Dict[str, Dict[str, Any]],
    prefetched: Dict[str, str],
    pubmed_prefetched: Dict[str, str],
    press_release_prefetched: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> None:
    """Check every efficacy number against text we ACTUALLY retrieved.

    THIS IS THE ONLY CHECK THAT CAN CATCH A CONFIDENT, SELF-CONSISTENT
    HALLUCINATION. Every other pass in this file compares rows against each
    OTHER — conflict reconciliation, orphan-duration dropping, URL fixing.
    Those catch contradictions, but a number invented once and never
    contradicted (e.g. REDEFINE 1 coming back as 25.0% when the published
    figure is 22.7%, or REDEFINE 2's numbers copied wholesale onto the
    still-running REDEFINE 3) sails through all of them. Grounding is the
    difference between "internally consistent" and "true".

    For each efficacy value we search the trial's own prefetched CT.gov API
    text and PubMed abstract, plus those of any row grouped as the same
    underlying trial (an EU duplicate legitimately inherits its CT.gov
    sibling's evidence). The result is written to a per-row provenance
    field:
      registry  — found in the CT.gov API record
      pubmed    — found in the matched publication abstract
      unverified— Gemini reported it, but it appears in NEITHER source we
                  hold. It may still be true (press releases and conference
                  abstracts are real sources we don't prefetch) — but it is
                  UNCHECKED, and every fabricated value found by hand so
                  far landed in this bucket.

    strict=True additionally BLANKS unverified values. That trades recall
    for trustworthiness: it will also delete genuine press-release-only
    figures. Off by default so the default run stays complete; the
    provenance column tells you which cells to trust either way.

    IMPORTANT: when all three prefetch dicts are empty (network unavailable
    — e.g. sandbox egress restriction blocks clinicaltrials.gov, PubMed,
    and press-release domains), this function is a no-op. Every value would
    be "unverified" by definition with no evidence to check against, so
    running it would erase ALL correct values Gemini found. Grounding only
    makes sense when we actually have retrieved text to ground against.
    """
    # If all three prefetch sources are empty, we have no evidence to check
    # against. Skip entirely — Gemini's own search is the only source.
    has_registry = any(prefetched.values())
    has_pubmed = any(pubmed_prefetched.values())
    has_press = any((press_release_prefetched or {}).values())
    if not has_registry and not has_pubmed and not has_press:
        print("  Grounding check: skipped — no prefetched text available "
              "(network restricted). Gemini source URLs are the sole evidence.",
              file=sys.stderr)
        # Still mark all values as provenance="gemini_search" so the column
        # is populated and enforcement knows grounding was not run.
        for row in rows_by_id.values():
            labels = []
            for value_field, _ in _EFFICACY_FIELD_PAIRS:
                val = str(row.get(value_field, "")).strip()
                if val:
                    label = value_field.replace("_change_pct", "").replace(
                        "_reduction_pct", "").replace("_resolution_pct", "")
                    labels.append(f"{label}=gemini_search")
            row["efficacy_provenance"] = "; ".join(labels)
        return

    groups = _group_related_trial_ids(rows_by_id)
    group_of = {tid: grp for grp in groups for tid in grp}

    counts = {"registry": 0, "pubmed": 0, "press_release": 0, "unverified": 0}
    blanked = 0

    for tid, row in rows_by_id.items():
        sibling_ids = group_of.get(tid, [tid])
        registry_text = " ".join(prefetched.get(s, "") or "" for s in sibling_ids)
        pubmed_text = " ".join(pubmed_prefetched.get(s, "") or "" for s in sibling_ids)
        press_text = " ".join(
            (press_release_prefetched or {}).get(s, "") or "" for s in sibling_ids
        )

        # Per-trial: if we have no text at all for this specific trial's
        # sibling group, mark as gemini_search and skip — same rationale as
        # the global skip above, but scoped to the individual trial.
        if not registry_text and not pubmed_text and not press_text:
            labels = []
            for value_field, _ in _EFFICACY_FIELD_PAIRS:
                val = str(row.get(value_field, "")).strip()
                if val:
                    label = value_field.replace("_change_pct", "").replace(
                        "_reduction_pct", "").replace("_resolution_pct", "")
                    labels.append(f"{label}=gemini_search")
            row["efficacy_provenance"] = "; ".join(labels)
            continue

        provenance: List[str] = []
        for value_field, duration_field in _EFFICACY_FIELD_PAIRS:
            val = str(row.get(value_field, "")).strip()
            if not val:
                continue
            if _grounding_match(val, registry_text):
                source = "registry"
            elif _grounding_match(val, pubmed_text):
                source = "pubmed"
            elif _grounding_match(val, press_text):
                source = "press_release"
            else:
                source = "unverified"

            # If Gemini didn't supply a rationale, backfill one based on
            # where we found the number.
            rationale_field = value_field.replace("_change_pct", "_rationale") \
                                         .replace("_reduction_pct", "_rationale") \
                                         .replace("_resolution_pct", "_rationale")
            if not row.get(rationale_field):
                if source == "press_release":
                    acronym = _extract_trial_acronym(
                        row.get("public_title", "") or row.get("title", "")
                        or row.get("brief_title", "") or ""
                    )
                    row[rationale_field] = f"Confirmed in press release for {acronym or 'this program'}"
                elif source == "registry":
                    row[rationale_field] = f"Confirmed in CT.gov API record for {tid}"
                elif source == "pubmed":
                    row[rationale_field] = f"Confirmed in PubMed abstract for {tid}"

            counts[source] += 1
            label = value_field.replace("_change_pct", "").replace("_reduction_pct", "") \
                               .replace("_resolution_pct", "")
            provenance.append(f"{label}={source}")

            if source == "unverified":
                print(f"  [warn] {tid}: {value_field}={val!r} could not be found in "
                      f"the CT.gov record or any matched publication — UNVERIFIED, "
                      f"treat as unconfirmed until checked by hand.", file=sys.stderr)
                if strict:
                    row[value_field] = ""
                    row[duration_field] = ""
                    blanked += 1

        row["efficacy_provenance"] = "; ".join(provenance)

    total = sum(counts.values())
    if total:
        pr_count = counts.get("press_release", 0)
        print(f"  Grounding check: {counts['registry']} confirmed in CT.gov, "
              f"{counts['pubmed']} in PubMed, {pr_count} in press releases, "
              f"{counts['unverified']} UNVERIFIED (of {total} total).",
              file=sys.stderr)
    if blanked:
        print(f"  --strict: blanked {blanked} unverified value(s).", file=sys.stderr)


# ---------------------------------------------------------------------------
# Press-release prefetch: for trials with known program acronyms (REDEFINE,
# REIMAGINE etc.) where the CT.gov API record only has protocol text and
# no results, we also fetch the sponsor's own press release directly.
# This is the source that makes "unverified" → "press_release" for real
# results like REDEFINE 2 HbA1c=1.8, which are published by Novo Nordisk
# but not in the CT.gov results section.
#
# We map program acronym → known press release URLs. This is maintained
# manually as a small lookup; it only grows when new trials publish results.
# ---------------------------------------------------------------------------

_PROGRAM_PRESS_RELEASES: Dict[str, List[str]] = {
    "REDEFINE 1": [
        "https://www.prnewswire.com/news-releases/cagrisema-2-4-mg--2-4-mg-demonstrated-22-7-mean-weight-reduction-in-adults-with-overweight-or-obesity-in-redefine-1--published-in-nejm-302487770.html",
        "https://www.sec.gov/Archives/edgar/data/353278/000117184324007023/f6k_122024.htm",
    ],
    "REDEFINE 2": [
        "https://www.sec.gov/Archives/edgar/data/353278/000117184325001350/f6k_031025.htm",
        "https://www.prnewswire.com/news-releases/cagrisema-2-4-mg--2-4-mg-demonstrated-22-7-mean-weight-reduction-in-adults-with-overweight-or-obesity-in-redefine-1--published-in-nejm-302487770.html",
    ],
    "REDEFINE 4": [
        "https://www.novonordisk.com/content/nncorp/global/en/news-and-media/news-and-ir-materials/news-details.html?id=174481",
    ],
    "REIMAGINE 1": [
        "https://www.novonordisk.com/content/nncorp/global/en/news-and-media/news-and-ir-materials/news-details.html?id=922731",
        "https://www.managedhealthcareexecutive.com/view/cagrisema-reduces-hba1c-weight-across-t2d-spectrum-in-reimagine-trials-ada-2026",
    ],
    "REIMAGINE 2": [
        "https://www.globenewswire.com/news-release/2026/02/02/3230429/0/en/Novo-Nordisk-A-S-CagriSema-demonstrated-superior-HbA1c-reduction-of-1-91-points-and-weight-loss-of-14-2-in-adults-with-type-2-diabetes-in-the-REIMAGINE-2-trial.html",
        "https://www.novonordisk.com/content/nncorp/global/en/news-and-media/news-and-ir-materials/news-details.html?id=916481",
    ],
    "REIMAGINE 3": [
        "https://www.prnewswire.com/news-releases/novo-nordisks-cagrisema-2-4-mg--2-4-mg-demonstrated-significant-reduction-in-hba1c-and-weight-across-multiple-studies-in-the-reimagine-program-presented-at-ada-2026--302793443.html",
        "https://www.managedhealthcareexecutive.com/view/cagrisema-reduces-hba1c-weight-across-t2d-spectrum-in-reimagine-trials-ada-2026",
    ],
    "REDEFINE 5": [
        "https://pubmed.ncbi.nlm.nih.gov/42009015/",
        "https://www.thelancet.com/journals/landia/article/PIIS2213-8587(25)00402-4/abstract",
    ],
}

_NCT_TO_PROGRAM: Dict[str, str] = {
    "NCT06323174": "REIMAGINE 1",
    "NCT06065540": "REIMAGINE 2",
    "NCT06323161": "REIMAGINE 3",
    "NCT05813925": "REDEFINE 5",
    "NCT05567796": "REDEFINE 1",
    "NCT05394519": "REDEFINE 2",
    "NCT06131437": "REDEFINE 4",
}

_PRESS_RELEASE_FETCH_TIMEOUT = 12
_PRESS_RELEASE_CONCURRENCY = 6   # was 3; these are different hostnames so parallelism is safe
_PRESS_RELEASE_MAX_CHARS = 5000


async def _fetch_press_release_one(url: str, session: aiohttp.ClientSession,
                                    semaphore: asyncio.Semaphore) -> str:
    """Fetch one press release URL, return trimmed text (best-effort)."""
    async with semaphore:
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=_PRESS_RELEASE_FETCH_TIMEOUT),
                headers={"User-Agent": "Mozilla/5.0 (compatible; trial-data-pipeline/1.0)"},
            ) as resp:
                if resp.status != 200:
                    return ""
                raw = await resp.text(errors="replace")
                # Very rough HTML-to-text: strip tags, collapse whitespace
                text = re.sub(r"<[^>]+>", " ", raw)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:_PRESS_RELEASE_MAX_CHARS]
        except Exception as exc:
            print(f"    [press] fetch failed for {url}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return ""


async def _prefetch_press_releases(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """For each trial whose title contains a known program acronym, fetch the
    known press releases and return {trial_id: combined_text}. Merged into
    the grounding check alongside the CT.gov API text and PubMed abstract.
    """
    if not await _probe_network():
        return {}
    # Build acronym -> [trial_id] mapping using title extraction first,
    # then fall back to the explicit NCT-to-program table for trials whose
    # CT.gov title is the full protocol name rather than the short label.
    acronym_to_ids: Dict[str, List[str]] = {}
    for row in rows:
        tid = row.get("trial_id", "")
        if not tid:
            continue
        title = (row.get("public_title", "") or row.get("title", "")
                 or row.get("brief_title", "") or "")
        acronym = _extract_trial_acronym(title)
        # If title extraction found nothing useful, try the NCT lookup table
        if (not acronym or acronym not in _PROGRAM_PRESS_RELEASES) and _is_ctgov_id(tid):
            acronym = _NCT_TO_PROGRAM.get(tid.upper(), "")
        if acronym and acronym in _PROGRAM_PRESS_RELEASES:
            acronym_to_ids.setdefault(acronym, []).append(tid)

    if not acronym_to_ids:
        return {}

    total_urls = sum(len(v) for v in _PROGRAM_PRESS_RELEASES.values()
                     if any(a in acronym_to_ids for a in _PROGRAM_PRESS_RELEASES))
    print(f"  Pre-fetching press releases for {len(acronym_to_ids)} program(s) "
          f"({sum(len(v) for a, v in _PROGRAM_PRESS_RELEASES.items() if a in acronym_to_ids)} URL(s))...",
          file=sys.stderr)

    # Collect unique URLs to fetch
    urls_to_fetch: Dict[str, str] = {}  # url -> acronym (first match)
    for acronym, tids in acronym_to_ids.items():
        for url in _PROGRAM_PRESS_RELEASES[acronym]:
            if url not in urls_to_fetch:
                urls_to_fetch[url] = acronym

    semaphore = asyncio.Semaphore(_PRESS_RELEASE_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=_PRESS_RELEASE_FETCH_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        fetched_texts: List[str] = await asyncio.gather(*[
            _fetch_press_release_one(url, session, semaphore)
            for url in urls_to_fetch
        ])

    url_to_text = dict(zip(urls_to_fetch.keys(), fetched_texts))

    # Map trial_id -> combined text from all press releases for its acronym
    result: Dict[str, str] = {}
    for acronym, tids in acronym_to_ids.items():
        combined = " ".join(
            url_to_text.get(url, "")
            for url in _PROGRAM_PRESS_RELEASES[acronym]
        ).strip()
        if combined:
            for tid in tids:
                existing = result.get(tid, "")
                result[tid] = (existing + " " + combined).strip()

    fetched = sum(1 for t in result.values() if t)
    print(f"  Press release prefetch: got text for {fetched}/{len(rows)} trial(s)",
          file=sys.stderr)
    return result


def _validate_rationales(rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    """Clean up rationale fields — strip N/A from non-empty values, and
    clear any rationale that is just a bare URL (Gemini sometimes puts a
    URL instead of a real explanation)."""
    cleaned = 0
    rationale_fields = [
        ("hba1c_change_pct", "hba1c_rationale"),
        ("weight_change_pct", "weight_rationale"),
        ("alt_reduction_pct", "alt_rationale"),
        ("mash_resolution_pct", "mash_rationale"),
    ]
    for tid, row in rows_by_id.items():
        for val_field, rat_field in rationale_fields:
            val = str(row.get(val_field, "")).strip()
            rat = str(row.get(rat_field, "") or "").strip()
            if not val and rat and rat.lower() != "n/a":
                row[rat_field] = ""
                cleaned += 1
            if rat.lower() == "n/a":
                row[rat_field] = ""
            # Bare URL is not a rationale
            if rat.startswith(("http://", "https://")) and " " not in rat:
                row[rat_field] = ""
                cleaned += 1
    if cleaned:
        print(f"  Cleaned {cleaned} rationale field(s).", file=sys.stderr)



def _enforce_rationale_requirement(rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    """Blank any efficacy value that has no rationale AND no confirmed provenance.

    A missing source URL means Gemini found a number but has no evidence
    trail for it — which is the defining characteristic of every confirmed
    hallucination found by hand-verification so far (REDEFINE 3 copying
    REDEFINE 2's numbers, NCT04982575 ALT=31.8, the 28.4% EU weight row,
    etc.). In every case the value was unverified AND the source URL was
    empty or fabricated (and already cleared by _validate_source_urls).

    Two classes of exceptions are allowed:
    1. A non-empty efficacy_provenance tag of "registry", "pubmed",
       "press_release", or "gemini_search" means grounding either confirmed
       the value or was skipped because no prefetch text was available.
       In both cases the value is kept.
    2. If grounding ran (prefetch text WAS available) but came back
       "unverified" AND the source URL is blank, the value is cleared.

    This is run AFTER _validate_source_urls so fabricated URLs are already
    gone and after _verify_efficacy_grounding so provenance tags exist.
    """
    field_to_label = {
        "hba1c_change_pct": "hba1c",
        "weight_change_pct": "weight",
        "alt_reduction_pct": "alt",
        "mash_resolution_pct": "mash",
    }
    blanked = 0
    for tid, row in rows_by_id.items():
        prov = str(row.get("efficacy_provenance", ""))
        for value_field, duration_field in _EFFICACY_FIELD_PAIRS:
            val = str(row.get(value_field, "")).strip()
            if not val:
                continue
            label = field_to_label.get(value_field, "")
            rationale_field = value_field.replace("_change_pct", "_rationale") \
                                          .replace("_reduction_pct", "_rationale") \
                                          .replace("_resolution_pct", "_rationale")
            has_rationale = bool(str(row.get(rationale_field, "") or "").strip())

            # "gemini_search" means grounding was skipped (no prefetch text
            # available). Trust Gemini's result — don't enforce.
            confirmed = (f"{label}=registry" in prov
                         or f"{label}=pubmed" in prov
                         or f"{label}=press_release" in prov
                         or f"{label}=gemini_search" in prov)

            if not has_rationale and not confirmed:
                print(f"  [enforce] {tid}: clearing {value_field}={val!r} — "
                      f"no rationale and not confirmed in any retrieved source",
                      file=sys.stderr)
                row[value_field] = ""
                row[duration_field] = ""
                blanked += 1

    if blanked:
        print(f"  Rationale enforcement: cleared {blanked} ungrounded value(s) "
              f"with no rationale.", file=sys.stderr)



async def _verify_confidence(rows_by_id: Dict[str, Dict[str, Any]],
                              drug: str) -> None:
    """Make separate Gemini calls to verify each extracted efficacy value.

    For each trial with efficacy data, ask Gemini to independently search
    for and confirm each number. Rates each as HIGH (confirmed in a
    credible source for THIS trial) or LOW (not found / different number).
    """
    rows_to_check = [row for row in rows_by_id.values()
                     if any(str(row.get(f, "")).strip()
                            for f, _ in _EFFICACY_FIELD_PAIRS)]
    if not rows_to_check:
        print("  Confidence check: no efficacy values to verify.", file=sys.stderr)
        return

    print(f"  Confidence check: verifying {len(rows_to_check)} trial(s) "
          f"via separate Gemini calls ...", file=sys.stderr)

    field_labels = {
        "hba1c_change_pct": ("HbA1c reduction", "hba1c_duration", "hba1c_rationale", "hba1c_confidence"),
        "weight_change_pct": ("weight loss", "weight_duration", "weight_rationale", "weight_confidence"),
        "alt_reduction_pct": ("ALT reduction", "alt_duration", "alt_rationale", "alt_confidence"),
        "mash_resolution_pct": ("MASH resolution", "mash_duration", "mash_rationale", "mash_confidence"),
    }
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    async def check_one(row):
        tid = row.get("trial_id", "?")
        title = row.get("trial_title", "") or row.get("title", "") or ""
        claims, field_keys = [], []
        for val_f, (label, dur_f, rat_f, conf_f) in field_labels.items():
            val = str(row.get(val_f, "")).strip()
            if not val:
                row[conf_f] = ""
                continue
            dur = str(row.get(dur_f, "")).strip() or "unknown duration"
            rat = str(row.get(rat_f, "")).strip() or "no rationale given"
            claims.append(f"- {label} of {val}% at {dur}. Rationale: \"{rat}\"")
            field_keys.append(conf_f)
        if not claims:
            return

        prompt = f"""You are verifying clinical trial data for accuracy.
Trial: {tid}
Title: {title}
Drug: {drug}

Values extracted by another AI:
{chr(10).join(claims)}

For EACH value, search for this specific trial's results and respond with ONLY JSON:
{{ {', '.join(f'"{k}": "HIGH or LOW"' for k in field_keys)} }}

HIGH = you found the same number (within rounding) in a credible source for THIS trial.
LOW = could not confirm, found a different number, or no results published for this trial.
Do NOT rate HIGH just because the number is plausible — it must be confirmed for THIS trial."""

        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(_call_gemini_for_confidence, prompt), timeout=30)
                if isinstance(result, dict):
                    for cf in field_keys:
                        v = str(result.get(cf, "")).strip().upper()
                        row[cf] = v if v in ("HIGH", "LOW") else "LOW"
                else:
                    for cf in field_keys:
                        row[cf] = "LOW"
            except Exception as exc:
                print(f"  [warn] Confidence check failed for {tid}: {type(exc).__name__}", file=sys.stderr)
                for cf in field_keys:
                    row[cf] = "UNKNOWN"

    await asyncio.gather(*[check_one(r) for r in rows_to_check])
    high = sum(1 for r in rows_to_check for cf in ("hba1c_confidence","weight_confidence","alt_confidence","mash_confidence") if r.get(cf) == "HIGH")
    low = sum(1 for r in rows_to_check for cf in ("hba1c_confidence","weight_confidence","alt_confidence","mash_confidence") if r.get(cf) == "LOW")
    print(f"  Confidence check: {high} HIGH, {low} LOW.", file=sys.stderr)


def _call_gemini_for_confidence(prompt: str) -> dict:
    """Synchronous Gemini call for confidence checking."""
    import json as _json
    from json_repair import repair_json
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
        ),
    )
    raw = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            raw += part.text
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        return _json.loads(raw)
    except _json.JSONDecodeError:
        try:
            return _json.loads(repair_json(raw))
        except Exception:
            return {}


def _block_efficacy_on_non_results_trials(rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    """Clear efficacy values on trials where results cannot possibly exist.

    The persistent contamination pattern is: a completed trial's numbers
    get copied onto an ongoing/recruiting/CVOT trial with a similar program
    name, or a PK/mechanism study that never measured efficacy endpoints.
    Neither the grounding check nor conflict reconciliation catches this
    when the contamination is applied consistently across all sibling rows.

    Rules:
    - Recruiting / Active / Not yet recruiting / Withdrawn → no results
    - Phase 1 PK/mechanism/appetite/bone/bioequivalence titles → no efficacy
    Exception: EU duplicate rows pointing at a DIFFERENT (completed)
    trial's NCT ID in their source_url carry that trial's results
    legitimately and are left alone.
    """
    NO_RESULTS_STATUSES = {
        "recruiting", "active, not recruiting", "active",
        "not yet recruiting", "withdrawn", "terminated",
    }
    PK_TITLE_FRAGMENTS = (
        "blood levels", "pharmacokinetics", "single dose",
        "bioequivalence", "muscle health", "bone metabolism",
        "gastric emptying", "food intake", "appetite", "atorvastatin",
        "warfarin", "insulin sensitivity", "insulin effect",
        "metabolism is influenced",
    )
    nct_pat = re.compile(r"NCT\d{8}", re.IGNORECASE)
    cleared = 0

    for tid, row in rows_by_id.items():
        if not any(str(row.get(f, "")).strip() for f, _ in _EFFICACY_FIELD_PAIRS):
            continue

        status = str(row.get("phase_status", "")).strip().lower()
        phase = str(row.get("phase", "")).strip()
        title = (str(row.get("trial_title", "") or row.get("title", "")
                     or row.get("public_title", "") or "")).lower()
        source_url = str(row.get("source_url", "") or "").upper()
        own_nct = tid.upper() if _is_ctgov_id(tid) else ""

        # EU duplicate pointing at a different trial's NCT — leave alone.
        # EU trial IDs (CT numbers like 2023-506931-13-00) are never NCT
        # format themselves, so if the row is non-NCT AND its source_url
        # points at an NCT ID, it's definitely an EU duplicate row carrying
        # a completed sibling's results — don't clear it.
        url_m = nct_pat.search(source_url)
        url_nct = url_m.group(0).upper() if url_m else ""
        is_eu_row = not _is_ctgov_id(tid)
        if is_eu_row and url_nct:
            continue  # EU row with NCT reference — leave efficacy alone
        if url_nct and own_nct and url_nct != own_nct:
            continue  # CT.gov row pointing at a different trial

        should_clear = False
        reason = ""
        if status in NO_RESULTS_STATUSES:
            should_clear = True
            reason = f"trial status is {row.get('phase_status')!r}"
        elif phase == "1" and any(frag in title for frag in PK_TITLE_FRAGMENTS):
            should_clear = True
            reason = "Phase 1 PK/mechanism study"

        if should_clear:
            for val_f, dur_f in _EFFICACY_FIELD_PAIRS:
                if str(row.get(val_f, "")).strip():
                    print(f"  [enforce] {tid}: clearing {val_f}={row[val_f]!r} — "
                          f"{reason}", file=sys.stderr)
                    row[val_f] = ""
                    row[dur_f] = ""
                    src_f = val_f.replace("_change_pct", "_source_url") \
                                 .replace("_reduction_pct", "_source_url") \
                                 .replace("_resolution_pct", "_source_url")
                    row[src_f] = ""
                    cleared += 1

    if cleared:
        print(f"  Status/type enforcement: cleared {cleared} value(s) from "
              f"trials that cannot have published results.", file=sys.stderr)


def _parse_loose_date(value: str) -> Optional[date]:
    """Best-effort parse of the date formats actually seen in this
    pipeline's output: ISO (YYYY-MM-DD), year-month (YYYY-MM), and the
    DD/MM/YYYY style some EU CTIS/EudraCT rows come back in. Returns None
    for anything unparseable rather than raising.
    """
    value = str(value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    m = re.match(r"^(\d{4})$", value)
    if m:
        return date(int(m.group(1)), 12, 31)
    return None


def _validate_dates_and_status(rows_by_id: Dict[str, Dict[str, Any]]) -> None:
    """Final safety-net sanity check across EVERY row, run last.

    CT.gov-native rows already get status/dates overwritten from the
    authoritative API above, so this should rarely fire for them — but EU
    CTIS/EudraCT rows have no equivalent free structured API here, so a
    row like "phase_status: Completed, trial_completion_date: 2027-05-28"
    (a logical impossibility — it can't be Completed with a completion
    date that hasn't happened yet) can still slip through for those.
    Nothing is silently changed here (we have no authoritative source to
    correct EU rows against) — this just prints a clear warning so the
    row gets a human look rather than shipping unnoticed.
    """
    today_ = date.today()
    flagged = 0
    for tid, row in rows_by_id.items():
        status = str(row.get("phase_status", "")).strip().lower()
        completion = _parse_loose_date(row.get("trial_completion_date", ""))
        start = _parse_loose_date(row.get("trial_start_date", ""))

        if completion and status in ("completed", "terminated") and completion > today_:
            print(f"  [warn] {tid}: phase_status is {row.get('phase_status')!r} but "
                  f"trial_completion_date ({row.get('trial_completion_date')}) is in "
                  f"the future — impossible combination, verify this row manually.",
                  file=sys.stderr)
            flagged += 1

        if start and completion and start > completion:
            print(f"  [warn] {tid}: trial_start_date ({row.get('trial_start_date')}) "
                  f"is after trial_completion_date ({row.get('trial_completion_date')}) "
                  f"— verify this row manually.", file=sys.stderr)
            flagged += 1

    if flagged:
        print(f"  Date/status sanity check: flagged {flagged} row(s) above for "
              f"manual review (see [warn] lines).", file=sys.stderr)


def _chunk_by_source(rows: List[Dict[str, Any]], batch_size: int,
                     prefetched: Optional[Dict[str, str]] = None,
                     acronym_collisions: Optional[Dict[str, List[str]]] = None,
                     pubmed_prefetched: Optional[Dict[str, str]] = None) -> List[tuple]:
    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        rows_by_source.setdefault(r.get("registry_source", "ctgov"), []).append(r)
    batches: List[tuple] = []
    for source, source_rows in rows_by_source.items():
        stubs = [_to_gemini_trial_stub(r, prefetched=prefetched,
                                        acronym_collisions=acronym_collisions,
                                        pubmed_prefetched=pubmed_prefetched)
                 for r in source_rows]
        for i in range(0, len(stubs), batch_size):
            batches.append((source, stubs[i:i + batch_size]))
    return batches


async def enrich(drug: str, max_records: Optional[int] = None,
                  sources: Optional[List[str]] = None,
                  retry_missing: bool = True,
                  strict: bool = False) -> List[Dict[str, Any]]:
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

    # --- Pre-fetch: three independent I/O passes run concurrently ---
    # All three have no dependency on each other and all three must finish
    # before Gemini pass 1 can start, so gathering them saves the sum of
    # two of their latencies (~15 s each for a 50-trial dataset).
    #   1. Registry pages  — CT.gov API JSON per NCT ID
    #   2. PubMed          — NCBI E-utilities abstract per trial acronym/ID
    #   3. Press releases  — known Novo Nordisk / PRNewswire / GlobeNewswire
    #                        pages per program acronym (REDEFINE/REIMAGINE)
    rows_list = list(rows_by_id.values())
    (prefetched,
     pubmed_prefetched,
     press_release_prefetched) = await asyncio.gather(
        _prefetch_registry_pages(rows_list),
        _prefetch_pubmed(rows_list),
        _prefetch_press_releases(rows_list),
    )

    # Detect acronym collisions (e.g. two unrelated trials both informally
    # called "REDEFINE 4" in outside press coverage) so the per-trial prompt
    # can warn Gemini to verify a match before assigning press-release data
    # to the wrong row.
    acronym_collisions = _detect_acronym_collisions(list(rows_by_id.values()))
    if acronym_collisions:
        for acronym, ids in acronym_collisions.items():
            print(f"  [warn] Acronym/program name '{acronym}' matches "
                  f"{len(ids)} distinct trial(s) in this run: {', '.join(ids)} "
                  f"— disambiguation warning added to their prompts.",
                  file=sys.stderr)

    # --- Pass 1: normal batching, smaller batch size for research depth ---
    batches = _chunk_by_source(list(rows_by_id.values()), FIRST_PASS_BATCH_SIZE,
                                prefetched=prefetched,
                                acronym_collisions=acronym_collisions,
                                pubmed_prefetched=pubmed_prefetched)
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

    # --- Pass 2 + authoritative CT.gov fetch (concurrent) ---
    # Compute retry candidates first (CPU only, instant)
    retry_rows = [r for r in rows_by_id.values() if _needs_retry(r)] if retry_missing else []
    #   • Pass 2 only needs prefetched/pubmed/press data (already done).
    #   • _fetch_ctgov_authoritative only needs the set of NCT IDs.
    # Running them in parallel shaves ~10–15 s for a 30-trial CT.gov set.
    if retry_missing and retry_rows:
        efficacy_only = sum(1 for r in retry_rows if _efficacy_missing(r) and not _core_fields_missing(r))
        core_missing = sum(1 for r in retry_rows if _core_fields_missing(r))
        print(f"  Pass 2: retrying {len(retry_rows)} trial(s) "
              f"({efficacy_only} missing only efficacy data, "
              f"{core_missing} missing core fields), "
              f"1 trial/call, deep-dive prompt ... (running concurrently with CT.gov auth fetch)",
              file=sys.stderr)
        retry_batches = _chunk_by_source(retry_rows, RETRY_BATCH_SIZE,
                                          prefetched=prefetched,
                                          acronym_collisions=acronym_collisions,
                                          pubmed_prefetched=pubmed_prefetched)

        async def _no_op_auth():
            return {}

        # Build the authoritative-fetch coroutine (or a no-op if no NCT IDs)
        auth_coro = (_fetch_ctgov_authoritative(list(ctgov_ids))
                     if ctgov_ids else _no_op_auth())

        retry_enriched, authoritative = await asyncio.gather(
            _run_batches(drug, retry_batches, extra_suffix=DEEP_DIVE_SUFFIX, label="pass2"),
            auth_coro,
        )
        retry_matched = _merge_enriched(retry_enriched, rows_by_id)
        still_missing = sum(1 for r in retry_rows if _needs_retry(r))
        print(f"  Pass 2: recovered data for "
              f"{len(retry_rows) - still_missing}/{len(retry_rows)} previously-incomplete trial(s)",
              file=sys.stderr)
    else:
        if retry_missing:
            print("  Pass 2: skipped — no trials missing efficacy or core data.", file=sys.stderr)
        # Still need the authoritative fetch even if pass 2 is skipped
        authoritative = await _fetch_ctgov_authoritative(list(ctgov_ids)) if ctgov_ids else {}

    # --- Pass 3 (no API calls): cross-URL efficacy propagation ---
    # Many EU CTIS/EudraCT entries share the same source_url as a CT.gov
    # trial (e.g. both point to the same clinicaltrials.gov/study/NCTxxx
    # page). If the CT.gov row already has efficacy data but the EU row
    # doesn't, copy it across. This is free — no Gemini calls needed.
    _propagate_efficacy_by_url(rows_by_id)

    # --- Pass 3.5 (no API calls): GROUNDING CHECK. Verify every efficacy
    # number against text we actually retrieved. This runs BEFORE conflict
    # reconciliation so the majority vote below can prefer a value that is
    # actually grounded in a source over one that merely appears in more
    # duplicate rows.
    _verify_efficacy_grounding(rows_by_id, prefetched, pubmed_prefetched,
                               press_release_prefetched=press_release_prefetched,
                               strict=strict)

    # Validate Gemini-supplied per-outcome source URLs — clear fabricated,
    # untrusted-domain, or wrong-trial URLs before they reach the output.
    _validate_rationales(rows_by_id)

    # Clear efficacy values on trials whose status means no readout can
    # exist yet (Recruiting, Active, etc.) — handles the contamination
    # case where CVOT/ongoing trial gets a completed trial's numbers.
    # Runs before source-URL enforcement so we don't waste cycles on rows
    # that should be blank regardless.
    _block_efficacy_on_non_results_trials(rows_by_id)

    # Clear any remaining efficacy value that has neither a source URL nor
    # a confirmed provenance tag — these are the residual hallucinations.
    _enforce_rationale_requirement(rows_by_id)

    # --- Pass 4 (no API calls): reconcile contradictions between rows that
    # share the same underlying trial but ended up with DIFFERENT efficacy
    # numbers (as opposed to Pass 3's gap-filling, this fixes disagreement).
    _reconcile_conflicting_efficacy(rows_by_id)

    # --- Pass 5 (no API calls): drop duration values with no accompanying
    # efficacy number — a duration alone is a hallucination signature, not
    # partial data (see _drop_orphan_durations).
    _drop_orphan_durations(rows_by_id)

    # --- Pass 6 (no API calls): fix any source_url that points at a
    # different trial's NCT ID than the row it's attached to.
    _fix_mismatched_source_urls(rows_by_id)

    # --- Pass 7: confidence verification via separate Gemini calls ---
    # For each trial that still has efficacy values after all enforcement,
    # make a separate LLM call to independently search for and confirm
    # the number. Each value gets a confidence rating: HIGH or LOW.
    await _verify_confidence(rows_by_id, drug)

    # Apply authoritative CT.gov status/enrollment/dates — already fetched
    # concurrently with pass 2 above. Propagate to EU duplicate rows too.
    if ctgov_ids and authoritative:
        groups_by_tid = {tid: grp for grp in _group_related_trial_ids(rows_by_id) for tid in grp}
        overridden = 0
        for nct_id, info in authoritative.items():
            targets = set(groups_by_tid.get(nct_id, [nct_id]))
            targets.add(nct_id)
            for target_id in targets:
                row = rows_by_id.get(target_id)
                if not row:
                    continue
                # Only overwrite when the API actually returned something —
                # don't blank out a good existing value just because this
                # particular field came back empty from the API.
                if info.get("status"):
                    row["phase_status"] = info["status"]
                if info.get("enrollment"):
                    row["trial_size"] = info["enrollment"]
                if info.get("start_date"):
                    row["trial_start_date"] = info["start_date"]
                if info.get("completion_date"):
                    row["trial_completion_date"] = info["completion_date"]
            overridden += 1
        print(f"  Cross-checked status/enrollment/dates against the CT.gov "
              f"API for {overridden}/{len(ctgov_ids)} trial(s), propagated to "
              f"any linked EU duplicate rows", file=sys.stderr)

    # Final safety net: flag (not silently fix) any remaining row where the
    # dates/status combination is logically impossible — mainly catches EU
    # rows the authoritative override above can't reach.
    _validate_dates_and_status(rows_by_id)

    return list(rows_by_id.values())


# ---------------------------------------------------------------------------
# Rationale → change-pct reconciliation
#
# Gemini occasionally puts the correct number in the rationale text (e.g.
# "HbA1c reduction of 1.91 points … REIMAGINE 2 trial") while writing a
# different (or blank) value into the _change_pct field — usually because
# the structured field was filled in an earlier pass or batch whose number
# drifted from the rationale written in a later pass.  This pass extracts
# the first plausible numeric value from each rationale field and, when it
# disagrees with the paired _change_pct value (or the _change_pct is
# empty), overwrites _change_pct with the number from the rationale.
# ---------------------------------------------------------------------------

# Match any standalone number (integer or decimal) — used to collect ALL
# candidate values from a rationale string before filtering by context.
_ALL_NUMBERS_RE = re.compile(r"(?<![0-9.])(-?\d+(?:\.\d+)?)(?:%)?(?![0-9.])")

# Keywords that indicate a number immediately following (within ~60 chars)
# is a clinical-outcome percentage rather than a dosage, year, or count.
_REDUCTION_CONTEXT_RE = re.compile(
    r"(?:reduction|reduced|decrease|decreased|change|loss|lost|lower|lowering|"
    r"improvement|improved|percentage\s+point|%-point|mean\s+change|estimated\s+mean|"
    r"hba1c|weight\s+loss|body\s+weight|alt|mash|nash|fibrosis)\s{0,30}",
    re.IGNORECASE,
)

# Patterns that indicate a number is a duration (weeks/months/years), not a
# percentage. "68 weeks", "at week 68", "at 68 wk" etc.
_DURATION_CONTEXT_RE = re.compile(
    r"(?:at\s+week\s+\d|at\s+\d+\s*w(?:ee)?k|(?:over\s+)?\d+[-\s]*w(?:ee)?k"
    r"|\d+\s*month|\d+\s*year|\d+\s*wk\b)",
    re.IGNORECASE,
)

# "at Week 26", "Week 68" — timepoint markers where N is the week number.
_WEEK_TIMEPOINT_RE = re.compile(r"(?:at\s+)?[Ww]eek\s+(\d+)")

# Numbers that are trial-series ordinals, e.g. "REDEFINE 2", "REIMAGINE 3".
_TRIAL_SERIES_NUMBER_RE = re.compile(
    r"(?:REDEFINE|REIMAGINE|STEP|SURPASS|SURMOUNT|PIONEER|SUSTAIN|REDUCE|"
    r"SUMMIT|ESSENCE|FLOW|SELECT|PHASE|PART|ARM|GROUP|COHORT|STUDY|TRIAL"
    r")\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Common drug dosages for CagriSema — if the only candidate number equals
# one of these AND the rationale contains "mg" or "dose", it is a dosage
# contamination, not a clinical outcome.
_CAGRISED_DOSAGES = {2.4, 1.0, 1.7, 0.5, 0.25}


def _looks_like_duration(value: float, rationale: str) -> bool:
    """Return True when *value* is most likely a trial duration in weeks.
    Signatures:
    - Value >= 48 (implausible as a percentage reduction for these endpoints).
    - The rationale contains "68 wk", "at 68 weeks", or "at Week 68".
    """
    if value >= 48:
        return True
    v_str = str(int(value)) if value == int(value) else str(value)
    if re.search(rf"(?<!\d){re.escape(v_str)}\s*w(?:ee)?k", rationale, re.IGNORECASE):
        return True
    for m in _WEEK_TIMEPOINT_RE.finditer(rationale):
        try:
            if abs(float(m.group(1)) - value) < 0.5:
                return True
        except ValueError:
            pass
    return False


def _looks_like_trial_series_number(value: float, rationale: str) -> bool:
    """Return True when *value* matches a trial-series ordinal embedded in
    the rationale, e.g. the '2' in 'REDEFINE 2 trial' or '3' in 'REIMAGINE 3'."""
    if value != int(value):
        return False
    for m in _TRIAL_SERIES_NUMBER_RE.finditer(rationale):
        try:
            if abs(float(m.group(1)) - value) < 0.5:
                return True
        except ValueError:
            pass
    return False


def _looks_like_dosage(value: float, rationale: str) -> bool:
    """Return True when *value* matches a known CagriSema dosage AND the
    rationale contains 'mg' or 'dose', suggesting dosage contamination."""
    if value not in _CAGRISED_DOSAGES:
        return False
    return bool(re.search(r"\b(?:mg|dose|dosage)\b", rationale, re.IGNORECASE))


def _extract_pct_from_rationale(rationale: str) -> str:
    """Return the most plausible clinical-outcome percentage found in
    *rationale*, stored as a positive bare string (reductions are
    conventionally positive throughout this pipeline).  Returns "" if no
    credible number can be parsed.

    The function explicitly rejects:
    - Duration-as-value contamination: a number ≥ 48 (implausible as a
      percentage reduction) or one that appears next to 'wk'/'week' in the
      rationale text (e.g. '68 wk' leaking into weight_change_pct).
    - Dosage contamination: 2.4, 1.0, 1.7 etc. when 'mg'/'dose' appears
      in the rationale (e.g. 'efficacy estimand for the 2.4/2.4 mg dose').
    - Year numbers (2020–2030) embedded in publication dates.

    Strategy: collect ALL numeric tokens, filter, score by clinical context
    keyword proximity, and take the best candidate.
    """
    if not rationale or not rationale.strip():
        return ""
    text = rationale.strip()
    candidates = []
    for m in _ALL_NUMBERS_RE.finditer(text):
        raw = m.group(1)
        try:
            f = float(raw)
        except ValueError:
            continue
        abs_f = abs(f)

        # Hard rejections —————————————————————————————————————————————————
        # 1. Zero or >100: no clinical endpoint ever reports these.
        if abs_f == 0 or abs_f > 100:
            continue
        # 2. Looks like a trial duration rather than an outcome %.
        if _looks_like_duration(abs_f, text):
            continue
        # 3. Matches a known drug dosage and the rationale mentions 'mg'.
        if _looks_like_dosage(abs_f, text):
            continue
        # 4. Four-digit year (20XX) appearing in a publication date context.
        if abs_f >= 2000:
            continue
        # 5. Trial-series ordinal (e.g. '2' in 'REDEFINE 2', '3' in
        #    'REIMAGINE 3'). These are program sub-study labels, not outcomes.
        if _looks_like_trial_series_number(abs_f, text):
            continue
        # 6. Day-of-month in a date context (e.g. '22' in 'June 22, 2025',
        #    '14' in 'February 14, 2026'). Month names right before the
        #    number are the giveaway.
        if abs_f == int(abs_f) and re.search(
            rf"(?:January|February|March|April|May|June|July|August|September|"
            rf"October|November|December)\s+{int(abs_f)}(?:st|nd|rd|th)?\b",
            text, re.IGNORECASE,
        ):
            continue
        # End hard rejections ——————————————————————————————————————————————

        # Score by reduction-keyword proximity
        start = max(0, m.start() - 60)
        window = text[start:m.end()].lower()
        in_context = bool(_REDUCTION_CONTEXT_RE.search(window))
        has_decimal = "." in raw

        # Sort key: context match first, then prefer decimals, then larger value
        candidates.append((0 if in_context else 1,
                           0 if has_decimal else 1,
                           -abs_f,
                           abs_f,
                           raw))
    if not candidates:
        return ""
    candidates.sort()
    abs_f = candidates[0][3]
    return f"{abs_f:g}"


def _reconcile_pct_from_rationale(final_rows: List[Dict[str, Any]]) -> None:
    """For each efficacy (value, rationale) pair in every row of
    *final_rows*, detect and fix three classes of mismatch between the
    stored _change_pct and the rationale text:

    1. **Truncation** — the model wrote a rounded integer instead of the
       precise decimal (e.g. '2' instead of '1.91' for REIMAGINE 2 HbA1c,
       or '2' instead of '14.2' for REIMAGINE 2 weight).
    2. **Duration leak** — the paired duration value (e.g. '68' weeks)
       leaked into the percentage field (spotted in REDEFINE 1 EU row
       2020-005435-75 weight=68, and REDEFINE 2 EU row 2023-509662-38-00
       weight=68).
    3. **Dosage contamination** — the drug dosage (2.4 mg) was written
       into the percentage field when the rationale supplies no specific
       outcome number (REIMAGINE 1 / REIMAGINE 3 EU rows).

    When the rationale contains a credible number that differs from the
    stored value by more than a rounding tolerance, the stored value is
    overwritten.  When the stored value is clearly a duration or dosage
    but the rationale contains no better number, the field is cleared
    rather than left with a misleading value.

    This is a post-processing step on the already-remapped rows (field
    names match FINAL_COLUMNS).
    """
    field_pairs = [
        ("hba1c_change_pct",    "hba1c_rationale",    "hba1c_duration"),
        ("weight_change_pct",   "weight_rationale",   "weight_duration"),
        ("alt_reduction_pct",   "alt_rationale",      "alt_duration"),
        ("mash_resolution_pct", "mash_rationale",     "mash_duration"),
    ]
    corrected = 0
    cleared = 0
    for row in final_rows:
        tid = row.get("trial_id", "?")
        dosage_str = str(row.get("dosage", "") or "")
        for val_field, rat_field, dur_field in field_pairs:
            rat = str(row.get(rat_field, "") or "").strip()
            existing = str(row.get(val_field, "") or "").strip()
            duration = str(row.get(dur_field, "") or "").strip()

            # ── Nothing to check if both value and rationale are empty ──
            if not existing and not rat:
                continue

            # ── Check for duration-as-value contamination ────────────────
            # If the stored value numerically equals the duration (e.g.
            # weight=68 and duration=68 wk), the duration leaked in.
            if existing and duration:
                dur_num_m = re.search(r"\d+(?:\.\d+)?", duration)
                if dur_num_m:
                    try:
                        if abs(float(existing) - float(dur_num_m.group(0))) < 0.05:
                            rat_val = _extract_pct_from_rationale(rat)
                            print(
                                f"  [reconcile] {tid}: {val_field}={existing!r} equals "
                                f"duration {duration!r} — duration-as-value contamination.",
                                file=sys.stderr,
                            )
                            if rat_val:
                                print(
                                    f"  [reconcile] {tid}: overwriting {val_field} "
                                    f"{existing!r} → {rat_val!r} (from rationale)",
                                    file=sys.stderr,
                                )
                                row[val_field] = rat_val
                                corrected += 1
                            else:
                                print(
                                    f"  [reconcile] {tid}: clearing {val_field}={existing!r} "
                                    f"(no credible number in rationale to replace it)",
                                    file=sys.stderr,
                                )
                                row[val_field] = ""
                                cleared += 1
                            continue
                    except ValueError:
                        pass

            # ── Check for dosage-as-value contamination ─────────────────
            # If the stored value matches the drug dosage and the rationale
            # contains 'mg'/'dose' but no specific outcome number, clear it.
            if existing:
                try:
                    existing_f = float(existing)
                    rat_val = _extract_pct_from_rationale(rat) if rat else ""
                    if _looks_like_dosage(existing_f, rat or dosage_str):
                        if not rat_val:
                            print(
                                f"  [reconcile] {tid}: {val_field}={existing!r} matches "
                                f"drug dosage and rationale has no outcome number — "
                                f"clearing (dosage contamination).",
                                file=sys.stderr,
                            )
                            row[val_field] = ""
                            cleared += 1
                            continue
                        # Rationale does have a non-dosage number — use it
                        if abs(existing_f - float(rat_val)) > 0.05:
                            print(
                                f"  [reconcile] {tid}: {val_field}={existing!r} looks like "
                                f"dosage; overwriting with rationale value {rat_val!r}",
                                file=sys.stderr,
                            )
                            row[val_field] = rat_val
                            corrected += 1
                            continue
                except ValueError:
                    pass

            # ── General mismatch: rationale number vs stored value ───────
            if not rat or rat.lower() == "n/a":
                continue
            rat_val = _extract_pct_from_rationale(rat)
            if not rat_val:
                continue
            if not existing or existing.lower() == "n/a":
                row[val_field] = rat_val
                corrected += 1
                print(
                    f"  [reconcile] {tid}: set {val_field}={rat_val!r} from "
                    f"rationale (was empty)",
                    file=sys.stderr,
                )
            else:
                try:
                    existing_f = float(existing)
                    rat_f = float(rat_val)
                    if abs(existing_f - rat_f) > 0.05:
                        print(
                            f"  [reconcile] {tid}: {val_field} mismatch — "
                            f"field={existing!r}, rationale={rat_val!r}; "
                            f"overwriting with rationale value",
                            file=sys.stderr,
                        )
                        row[val_field] = rat_val
                        corrected += 1
                except ValueError:
                    pass

    total = corrected + cleared
    if total:
        print(
            f"  Rationale reconciliation: corrected {corrected}, "
            f"cleared {cleared} change-pct value(s) "
            f"({total} total).",
            file=sys.stderr,
        )


def _deduplicate_final_rows(final_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate remapped rows by their bare trial ID (stripping the
    appended acronym so 'NCT05394519 (REDEFINE 2)' and a plain
    'NCT05394519' still map to the same bucket).

    When the same underlying trial appears more than once (e.g. once from
    CT.gov and once from an EU CTIS/EudraCT row), keep the single row that
    has the most efficacy data.  Ties are broken by preferring CT.gov rows
    (registry_source == 'ctgov') over EU rows, then by alphabetical
    trial_id for a stable, reproducible result.
    """
    def _bare_id(trial_id_str: str) -> str:
        """Strip the '(ACRONYM)' suffix added by _format_trial_id_with_acronym."""
        return trial_id_str.split("(")[0].strip()

    def _efficacy_count(row: Dict[str, Any]) -> int:
        return sum(
            1 for f in ("hba1c_change_pct", "weight_change_pct",
                        "alt_reduction_pct", "mash_resolution_pct")
            if str(row.get(f, "")).strip()
        )

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in final_rows:
        key = _bare_id(str(row.get("trial_id", "")))
        groups.setdefault(key, []).append(row)

    deduped: List[Dict[str, Any]] = []
    removed = 0
    for key, grp in groups.items():
        if len(grp) == 1:
            deduped.append(grp[0])
            continue
        # Sort: most efficacy data first, then prefer ctgov, then by trial_id
        grp_sorted = sorted(
            grp,
            key=lambda r: (
                -_efficacy_count(r),
                0 if str(r.get("registry_source", "")).lower() == "ctgov" else 1,
                str(r.get("trial_id", "")),
            ),
        )
        deduped.append(grp_sorted[0])
        removed += len(grp) - 1
        kept_src = grp_sorted[0].get("registry_source", "?")
        all_srcs = [r.get("registry_source", "?") for r in grp]
        print(f"  [dedup] {key}: kept {kept_src!r} row (of {all_srcs}) "
              f"— removed {len(grp) - 1} duplicate(s)", file=sys.stderr)

    if removed:
        print(f"  Deduplication: removed {removed} duplicate trial row(s); "
              f"{len(deduped)} unique trial(s) remain.", file=sys.stderr)
    return deduped


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
    ap.add_argument("--strict", action="store_true",
                     help="blank any efficacy value that could not be found in the "
                          "CT.gov record or a matched publication. Maximises "
                          "trustworthiness at the cost of recall: genuine "
                          "press-release-only figures will also be dropped. "
                          "Without this flag nothing is deleted — unverified "
                          "values are kept and marked in the efficacy_provenance "
                          "column instead.")
    ap.add_argument("--no-retry", action="store_true",
                     help="skip the automatic single-trial retry pass for trials "
                          "that came back with no efficacy data (faster, fewer API calls, "
                          "lower fill rate)")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows = asyncio.run(enrich(args.drug, args.max_records, sources,
                               retry_missing=not args.no_retry,
                               strict=args.strict))

    if not rows:
        print("No trials found / enriched.", file=sys.stderr)
        return 1

    still_empty = sum(1 for r in rows if _efficacy_missing(r))
    still_core_missing = sum(1 for r in rows if _core_fields_missing(r))
    print(f"  Final: {len(rows) - still_empty}/{len(rows)} trial(s) have at least "
          f"some efficacy data; {still_empty} have none (likely no results published yet).",
          file=sys.stderr)
    unverified_rows = [r for r in rows
                       if "unverified" in str(r.get("efficacy_provenance", ""))]
    if unverified_rows:
        print(f"  Final: {len(unverified_rows)} row(s) contain at least one "
              f"UNVERIFIED efficacy value (present in neither the CT.gov record "
              f"nor a matched publication). See the efficacy_provenance column; "
              f"re-run with --strict to blank these instead of keeping them.",
              file=sys.stderr)

    if still_core_missing:
        print(f"  Final: {still_core_missing}/{len(rows)} trial(s) are still missing "
              f"a core field (title/phase/size/location/status/start date) after "
              f"both passes — check these rows.", file=sys.stderr)

    final_rows = [remap_row(row, args.drug) for row in rows]

    # Deduplicate rows that represent the same trial across registries
    # (e.g. both a CT.gov row and an EU CTIS/EudraCT row for the same trial).
    # Must run AFTER remap_row so the trial_id already has its '(ACRONYM)'
    # suffix and the efficacy fields are in their final column names.
    final_rows = _deduplicate_final_rows(final_rows)

    # Fix any mismatch between the rationale text (which often contains the
    # precise number Gemini found) and the _change_pct field.
    _reconcile_pct_from_rationale(final_rows)

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