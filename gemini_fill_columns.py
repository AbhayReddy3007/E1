#!/usr/bin/env python3
"""
gemini_fill_columns.py – Fetch source URL content, then extract efficacy with LLM.

Two-step approach:
  1. FETCH: Download the actual page from each source_url (requests + BeautifulSoup)
     - ClinicalTrials.gov → use API v2 for structured JSON
     - Other registries → fetch HTML, extract text
  2. EXTRACT: Send page text to Gemini as context → extract efficacy metrics

This way Gemini reads REAL page content, not search snippets.
"""

from __future__ import annotations
import json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
CTGOV_API = "https://clinicaltrials.gov/api/v2/studies/{nct_id}"
DEFAULT_MODEL = "gemini-2.5-flash"
BATCH_SIZE = 4          # smaller batches = more context per trial
MAX_RETRIES = 4
INITIAL_BACKOFF = 2.0
MAX_PAGE_CHARS = 15000  # cap per trial page to fit in context

EFFICACY_FIELDS = [
    "dosage",
    "hba1c_change_pct", "hba1c_duration",
    "weight_change_pct", "weight_duration",
    "alt_reduction_pct", "alt_duration",
    "mash_resolution_pct", "mash_duration",
]

_print_lock = threading.Lock()
_http = requests.Session()
_http.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
})


# ── Step 1: Fetch actual page content ────────────────────────────────────────

def _fetch_ctgov(nct_id: str) -> str:
    """Fetch structured data from ClinicalTrials.gov API v2 and format as text."""
    try:
        r = _http.get(CTGOV_API.format(nct_id=nct_id), timeout=20)
        if r.status_code != 200:
            return ""
        data = r.json()

        ps = data.get("protocolSection") or {}
        rs = data.get("resultsSection") or {}

        lines = [f"=== ClinicalTrials.gov: {nct_id} ==="]

        # Protocol
        ident = ps.get("identificationModule") or {}
        lines.append(f"Title: {ident.get('officialTitle', ident.get('briefTitle', ''))}")

        status = ps.get("statusModule") or {}
        lines.append(f"Status: {status.get('overallStatus', '')}")
        lines.append(f"Start Date: {(status.get('startDateStruct') or {}).get('date', '')}")
        lines.append(f"Completion Date: {(status.get('completionDateStruct') or {}).get('date', '')}")

        design = ps.get("designModule") or {}
        lines.append(f"Phase: {', '.join(design.get('phases') or [])}")
        enroll = design.get("enrollmentInfo") or {}
        lines.append(f"Enrollment: {enroll.get('count', '')} ({enroll.get('type', '')})")

        sponsor = ps.get("sponsorCollaboratorsModule") or {}
        lines.append(f"Sponsor: {(sponsor.get('leadSponsor') or {}).get('name', '')}")

        arms = ps.get("armsInterventionsModule") or {}
        for intv in (arms.get("interventions") or []):
            lines.append(f"Intervention: {intv.get('type', '')}: {intv.get('name', '')} "
                         f"— {intv.get('description', '')[:200]}")

        outcomes = ps.get("outcomesModule") or {}
        for o in (outcomes.get("primaryOutcomes") or []):
            lines.append(f"Primary Outcome: {o.get('measure', '')} [{o.get('timeFrame', '')}]")

        desc = ps.get("descriptionModule") or {}
        lines.append(f"Brief Summary: {desc.get('briefSummary', '')[:500]}")

        # Results (if posted)
        if rs:
            lines.append("\n--- RESULTS ---")
            for om in (rs.get("outcomeMeasuresModule", {}).get("outcomeMeasures") or []):
                otype = om.get("type", "")
                title = om.get("title", "")
                unit = om.get("unitOfMeasure", "")
                desc_r = om.get("description", "")[:200]
                lines.append(f"\n[{otype}] {title} (unit: {unit})")
                if desc_r:
                    lines.append(f"  Description: {desc_r}")

                for cls in (om.get("classes") or [])[:5]:
                    cls_title = cls.get("title", "")
                    for cat in (cls.get("categories") or [])[:5]:
                        for meas in (cat.get("measurements") or [])[:8]:
                            grp = meas.get("groupId", "")
                            val = meas.get("value", "")
                            spread = meas.get("spread", "")
                            lo = meas.get("lowerLimit", "")
                            hi = meas.get("upperLimit", "")
                            line = f"  {cls_title} {grp}: {val}"
                            if spread:
                                line += f" ±{spread}"
                            if lo or hi:
                                line += f" ({lo}-{hi})"
                            lines.append(line)

                for an in (om.get("analyses") or [])[:3]:
                    pv = an.get("pValue", "")
                    pm = an.get("paramValue", "")
                    pt = an.get("paramType", "")
                    ci_lo = an.get("ciLowerLimit", "")
                    ci_hi = an.get("ciUpperLimit", "")
                    bits = []
                    if pt or pm: bits.append(f"{pt}={pm}")
                    if ci_lo or ci_hi: bits.append(f"95%CI {ci_lo}-{ci_hi}")
                    if pv: bits.append(f"p={pv}")
                    if bits:
                        lines.append(f"  Stats: {', '.join(bits)}")

        return "\n".join(lines)[:MAX_PAGE_CHARS]

    except Exception as exc:
        return ""


def _fetch_html_page(url: str) -> str:
    """Fetch any registry HTML page and extract readable text."""
    if not url or url.lower() in ("n/a", "none", ""):
        return ""
    try:
        r = _http.get(url, timeout=25, allow_redirects=True)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        # Clean excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:MAX_PAGE_CHARS]
    except Exception:
        return ""


def _fetch_page_content(trial_id: str, source_url: str) -> str:
    """Fetch page content — uses API for ClinicalTrials.gov, HTML for others."""
    tid = (trial_id or "").strip()

    # ClinicalTrials.gov — use the structured API
    nct_match = re.match(r"(NCT\d{6,})", tid, re.I)
    if nct_match:
        content = _fetch_ctgov(nct_match.group(1))
        if content:
            return content

    # All other registries — fetch HTML
    return _fetch_html_page(source_url)


# ── Step 2: Gemini extraction from page content ─────────────────────────────

def _gemini_call(session, url, prompt, timeout=120):
    """Plain Gemini call (NO google_search tool — we provide the content)."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(url, data=json.dumps(body), timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff); backoff *= 2; continue
            r.raise_for_status()
            cands = r.json().get("candidates") or []
            if not cands:
                return ""
            text = ""
            for part in (cands[0].get("content") or {}).get("parts") or []:
                if part.get("text"):
                    text += part["text"]
            return text.strip()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                return ""
            time.sleep(backoff); backoff *= 2
    return ""


def _parse_json(text: str) -> Optional[Any]:
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text).strip()
    json_start = -1
    for i, c in enumerate(text):
        if c in "{[":
            json_start = i
            break
    if json_start < 0:
        return None
    text = text[json_start:]
    try:
        from json_repair import repair_json
        result = repair_json(text, return_objects=True)
        if result is not None:
            return result
    except Exception:
        pass
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        return obj
    except Exception:
        pass
    return None


def _extract_trials(data: Any) -> List[dict]:
    if data is None:
        return []
    if isinstance(data, list):
        return [t for t in data if isinstance(t, dict)]
    if isinstance(data, dict):
        if "trials" in data:
            t = data["trials"]
            return [x for x in t if isinstance(x, dict)] if isinstance(t, list) else []
        if any(k in data for k in ("trial_id", "dosage", "hba1c_change_pct")):
            return [data]
    return []


def _normalise_tid(tid: str) -> str:
    return tid.split("(")[0].strip().split(" ")[0].strip() if tid else ""


def _build_prompt(drug: str, batch_with_content: List[dict]) -> str:
    """Build extraction prompt with ACTUAL page content as context."""

    trial_sections = []
    for item in batch_with_content:
        tid = item["trial_id"]
        content = item["content"]
        trial_sections.append(
            f"=== TRIAL: {tid} ===\n"
            f"PAGE CONTENT:\n{content}\n"
            f"=== END {tid} ===\n"
        )

    return f"""You are a clinical data extraction engine.

Below is the ACTUAL content fetched from registry pages for {len(batch_with_content)} {drug} trials.
Extract efficacy data ONLY from the page content provided below.
Do NOT use any external knowledge — extract only what is written in the text.

{"".join(trial_sections)}

For EACH trial, extract:
- trial_id: (the trial ID from the header above)
- dosage: The HIGHEST dosage tested (e.g., "2.4 mg once weekly", "14 mg daily"). "N/A" if not found in text.
- hba1c_change_pct: HbA1c reduction in percentage points as positive number (e.g., "1.8"). "N/A" if not in text.
- hba1c_duration: Timepoint for HbA1c measurement (e.g., "26 wk", "52 wk"). "N/A" if not in text.
- weight_change_pct: Body weight loss % as positive number (e.g., "15.2"). "N/A" if not in text.
- weight_duration: Timepoint for weight measurement (e.g., "68 wk"). "N/A" if not in text.
- alt_reduction_pct: ALT reduction % as positive number. "N/A" if not in text.
- alt_duration: Timepoint for ALT. "N/A" if not in text.
- mash_resolution_pct: MASH/NASH resolution rate %. "N/A" if not in text.
- mash_duration: Timepoint for MASH. "N/A" if not in text.

RULES:
- Return ALL {len(batch_with_content)} trials
- If a value is NOT in the page content, use "N/A" — NEVER guess or fabricate
- Report reductions as POSITIVE numbers
- Many trials will have "N/A" for most efficacy fields — that is expected

Return JSON:
{{
  "trials": [
    {{"trial_id": "...", "dosage": "...", "hba1c_change_pct": "N/A", "hba1c_duration": "N/A", "weight_change_pct": "N/A", "weight_duration": "N/A", "alt_reduction_pct": "N/A", "alt_duration": "N/A", "mash_resolution_pct": "N/A", "mash_duration": "N/A"}}
  ]
}}"""


# ── Main entry point ─────────────────────────────────────────────────────────

def fill_columns(drug: str, rows: List[Dict[str, Any]],
                 api_key: Optional[str] = None,
                 model: str = DEFAULT_MODEL,
                 workers: int = 4) -> List[Dict[str, Any]]:
    """
    1. Fetch page content from each trial's source_url
    2. Send content to Gemini as context → extract efficacy metrics
    3. Fill only empty efficacy fields — never overwrite existing data
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [Gemini fill] no GEMINI_API_KEY – skipping.", file=sys.stderr)
        return rows
    if not rows:
        return rows

    gemini_url = GEMINI_URL.format(model=model)
    gemini_session = requests.Session()
    gemini_session.headers.update({
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    })

    # ── Step 1: Fetch page content for all trials in parallel ────────────────
    print(f"  [Step 1] Fetching page content for {len(rows)} trials ...",
          file=sys.stderr)

    page_contents: Dict[int, str] = {}

    def fetch_one(idx):
        row = rows[idx]
        content = _fetch_page_content(
            row.get("trial_id", ""), row.get("source_url", ""))
        return idx, content

    fetched, failed = 0, 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_one, i): i for i in range(len(rows))}
        for fut in as_completed(futures):
            idx, content = fut.result()
            page_contents[idx] = content
            if content:
                fetched += 1
            else:
                failed += 1

    print(f"  [Step 1] Fetched {fetched} pages ({failed} failed/empty)",
          file=sys.stderr)

    # ── Step 2: Send content to Gemini in batches ────────────────────────────
    # Build batches with content
    items = []
    for i, row in enumerate(rows):
        content = page_contents.get(i, "")
        if not content:
            content = "(No page content available — return N/A for all fields)"
        items.append({
            "idx": i,
            "trial_id": row.get("trial_id", ""),
            "content": content,
        })

    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    print(f"  [Step 2] Extracting efficacy via Gemini ({len(batches)} batches) ...",
          file=sys.stderr)

    all_gemini: Dict[str, dict] = {}  # trial_id → gemini result

    def process_batch(batch_idx, batch):
        stagger = batch_idx % max(1, workers)
        if stagger > 0:
            time.sleep(stagger * 0.5)
        prompt = _build_prompt(drug, batch)
        raw = _gemini_call(gemini_session, gemini_url, prompt)
        data = _parse_json(raw)
        trials = _extract_trials(data)
        if not trials and raw:
            with _print_lock:
                print(f"    [parse fail] {raw[:150]!r}", file=sys.stderr)
        return batch_idx, trials

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(process_batch, bi, b): bi
                   for bi, b in enumerate(batches)}
        done = 0
        for fut in as_completed(futures):
            batch_idx, trials = fut.result()
            for gt in trials:
                tid = str(gt.get("trial_id", "") or "").strip()
                if tid:
                    all_gemini[tid] = gt
                    norm = _normalise_tid(tid)
                    if norm != tid:
                        all_gemini[norm] = gt
            done += 1
            with _print_lock:
                print(f"  [Step 2] batch {done}/{len(batches)} → "
                      f"{len(trials)} extracted", file=sys.stderr)

    # ── Merge: only fill empty efficacy fields ───────────────────────────────
    filled = 0
    for row in rows:
        tid = str(row.get("trial_id", "") or "").strip()
        norm = _normalise_tid(tid)
        gt = all_gemini.get(tid) or all_gemini.get(norm)
        if not gt:
            continue
        for field in EFFICACY_FIELDS:
            current = str(row.get(field, "") or "").strip()
            new_val = str(gt.get(field, "") or "").strip()
            if (not current or current.lower() in ("", "n/a", "none", "null")) \
               and new_val and new_val.lower() not in ("", "n/a", "none", "null"):
                row[field] = new_val
                filled += 1

    print(f"  [Gemini fill] Done — filled {filled} efficacy fields",
          file=sys.stderr)
    return rows