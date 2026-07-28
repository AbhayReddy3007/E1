#!/usr/bin/env python3
"""
gemini_fill_columns.py – Two-pass efficacy extraction.

Pass 1: Fetch source URL content (API for ClinicalTrials.gov, HTML for others)
        → extract efficacy metrics from page content using Gemini
Pass 2: For trials STILL missing efficacy data after Pass 1,
        use Gemini + Google Search to find published results

Only extracts: dosage, hba1c, weight, alt, mash (+ durations)
Never touches: trial_title, company_name, phase, etc.
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
BATCH_SIZE = 4
SEARCH_BATCH_SIZE = 6
MAX_RETRIES = 4
INITIAL_BACKOFF = 2.0
MAX_PAGE_CHARS = 15000

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


# ── Page fetching ────────────────────────────────────────────────────────────

def _fetch_ctgov(nct_id: str) -> str:
    try:
        r = _http.get(CTGOV_API.format(nct_id=nct_id), timeout=20)
        if r.status_code != 200:
            return ""
        data = r.json()
        ps = data.get("protocolSection") or {}
        rs = data.get("resultsSection") or {}

        lines = [f"=== ClinicalTrials.gov: {nct_id} ==="]
        ident = ps.get("identificationModule") or {}
        lines.append(f"Title: {ident.get('officialTitle', ident.get('briefTitle', ''))}")

        status = ps.get("statusModule") or {}
        lines.append(f"Status: {status.get('overallStatus', '')}")

        design = ps.get("designModule") or {}
        lines.append(f"Phase: {', '.join(design.get('phases') or [])}")

        sponsor = ps.get("sponsorCollaboratorsModule") or {}
        lines.append(f"Sponsor: {(sponsor.get('leadSponsor') or {}).get('name', '')}")

        arms = ps.get("armsInterventionsModule") or {}
        for intv in (arms.get("interventions") or []):
            lines.append(f"Intervention: {intv.get('type','')}: {intv.get('name','')} "
                         f"— {intv.get('description','')[:300]}")

        desc = ps.get("descriptionModule") or {}
        lines.append(f"Summary: {desc.get('briefSummary', '')[:500]}")

        outcomes = ps.get("outcomesModule") or {}
        for o in (outcomes.get("primaryOutcomes") or []):
            lines.append(f"Primary Outcome: {o.get('measure','')} [{o.get('timeFrame','')}]")

        if rs:
            lines.append("\n--- POSTED RESULTS ---")
            for om in (rs.get("outcomeMeasuresModule", {}).get("outcomeMeasures") or []):
                otype = om.get("type", "")
                title = om.get("title", "")
                unit = om.get("unitOfMeasure", "")
                desc_r = om.get("description", "")[:300]
                lines.append(f"\n[{otype}] {title} (unit: {unit})")
                if desc_r:
                    lines.append(f"  Description: {desc_r}")
                for cls in (om.get("classes") or [])[:5]:
                    for cat in (cls.get("categories") or [])[:5]:
                        for meas in (cat.get("measurements") or [])[:8]:
                            val = meas.get("value", "")
                            spread = meas.get("spread", "")
                            line = f"  {cls.get('title','')} {meas.get('groupId','')}: {val}"
                            if spread: line += f" ±{spread}"
                            lines.append(line)
                for an in (om.get("analyses") or [])[:3]:
                    bits = []
                    if an.get("paramType") or an.get("paramValue"):
                        bits.append(f"{an.get('paramType','')}={an.get('paramValue','')}")
                    if an.get("ciLowerLimit") or an.get("ciUpperLimit"):
                        bits.append(f"95%CI {an.get('ciLowerLimit','')}-{an.get('ciUpperLimit','')}")
                    if an.get("pValue"): bits.append(f"p={an['pValue']}")
                    if bits: lines.append(f"  Stats: {', '.join(bits)}")
        else:
            lines.append("\n(No results posted on registry)")

        return "\n".join(lines)[:MAX_PAGE_CHARS]
    except Exception:
        return ""


def _fetch_html_page(url: str) -> str:
    if not url or url.lower() in ("n/a", "none", ""):
        return ""
    try:
        r = _http.get(url, timeout=25, allow_redirects=True)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        return re.sub(r"\n{3,}", "\n\n", text)[:MAX_PAGE_CHARS]
    except Exception:
        return ""


def _fetch_page(trial_id: str, source_url: str) -> str:
    nct = re.match(r"(NCT\d{6,})", (trial_id or "").strip(), re.I)
    if nct:
        content = _fetch_ctgov(nct.group(1))
        if content:
            return content
    return _fetch_html_page(source_url)


# ── Gemini calls ─────────────────────────────────────────────────────────────

def _gemini_call(session, url, prompt, timeout=120, use_search=False):
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4096},
    }
    if use_search:
        body["tools"] = [{"google_search": {}}]
    else:
        body["generationConfig"]["responseMimeType"] = "application/json"

    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(url, data=json.dumps(body), timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff); backoff *= 2; continue
            r.raise_for_status()
            cands = r.json().get("candidates") or []
            if not cands: return ""
            text = ""
            for part in (cands[0].get("content") or {}).get("parts") or []:
                if part.get("text"): text += part["text"]
            return text.strip()
        except Exception as exc:
            if attempt == MAX_RETRIES: return ""
            time.sleep(backoff); backoff *= 2
    return ""


def _parse_json(text):
    if not text: return None
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text).strip()
    for i, c in enumerate(text):
        if c in "{[":
            text = text[i:]
            break
    else:
        return None
    try:
        from json_repair import repair_json
        r = repair_json(text, return_objects=True)
        if r is not None: return r
    except Exception: pass
    try: return json.loads(text)
    except Exception: pass
    try:
        d = json.JSONDecoder()
        obj, _ = d.raw_decode(text)
        return obj
    except Exception: pass
    return None


def _extract_trials(data):
    if data is None: return []
    if isinstance(data, list): return [t for t in data if isinstance(t, dict)]
    if isinstance(data, dict):
        if "trials" in data:
            t = data["trials"]
            return [x for x in t if isinstance(x, dict)] if isinstance(t, list) else []
        if any(k in data for k in ("trial_id", "dosage")): return [data]
    return []


def _norm_tid(tid): return tid.split("(")[0].strip().split(" ")[0].strip() if tid else ""


# ── Pass 1: Extract from fetched page content ────────────────────────────────

def _pass1_prompt(drug, batch):
    sections = []
    for item in batch:
        sections.append(
            f"=== TRIAL: {item['trial_id']} ===\n"
            f"{item['content']}\n"
            f"=== END ===\n")
    return f"""Extract clinical efficacy data for {len(batch)} {drug} trials from the registry page content below.

{"".join(sections)}

For EACH trial, extract ONLY from the text above:
- trial_id: (copy from header)
- dosage: Highest dosage tested (e.g., "2.4 mg OW"). "N/A" if not in text.
- hba1c_change_pct: HbA1c reduction in percentage points, positive number (e.g., "1.8"). "N/A" if not in text.
- hba1c_duration: Timepoint (e.g., "26 wk"). "N/A" if not in text.
- weight_change_pct: Weight loss %, positive number (e.g., "15.2"). "N/A" if not in text.
- weight_duration: Timepoint (e.g., "68 wk"). "N/A" if not in text.
- alt_reduction_pct: ALT reduction %. "N/A" if not in text.
- alt_duration: Timepoint. "N/A" if not in text.
- mash_resolution_pct: MASH/NASH resolution %. "N/A" if not in text.
- mash_duration: Timepoint. "N/A" if not in text.

RULES: Return ALL {len(batch)} trials. Use "N/A" if NOT in the text. NEVER fabricate.
Report reductions as POSITIVE numbers.

Return JSON: {{"trials": [{{"trial_id":"...","dosage":"...","hba1c_change_pct":"N/A",...}}]}}"""


# ── Pass 2: Google Search for trials still missing data ──────────────────────

def _pass2_prompt(drug, batch):
    trial_lines = []
    for row in batch:
        tid = row.get("trial_id", "")
        title = row.get("trial_title", "")[:80]
        trial_lines.append(f"- {tid}: {title}")
    return f"""Search Google for published clinical trial results for these {drug} trials.
Find the primary efficacy outcomes that were published or presented.

TRIALS (search for each one):
{chr(10).join(trial_lines)}

For EACH trial, search for its NCT/registry ID and find published results.
Extract:
- trial_id: (keep as provided)
- dosage: Highest dosage tested. "N/A" if not found.
- hba1c_change_pct: HbA1c reduction in percentage points, positive (e.g., "1.8"). "N/A" if not found.
- hba1c_duration: Timepoint (e.g., "52 wk"). "N/A" if not found.
- weight_change_pct: Weight loss %, positive (e.g., "15.2"). "N/A" if not found.
- weight_duration: Timepoint (e.g., "68 wk"). "N/A" if not found.
- alt_reduction_pct: ALT reduction %. "N/A" if not found.
- alt_duration: Timepoint. "N/A" if not found.
- mash_resolution_pct: MASH resolution %. "N/A" if not found.
- mash_duration: Timepoint. "N/A" if not found.

RULES: Return ALL {len(batch)} trials. Use "N/A" if genuinely not found. NEVER fabricate.
Only report values you find in actual search results.

Return JSON: {{"trials": [{{"trial_id":"...","dosage":"...","hba1c_change_pct":"N/A",...}}]}}"""


# ── Main ─────────────────────────────────────────────────────────────────────

def fill_columns(drug: str, rows: List[Dict[str, Any]],
                 api_key: Optional[str] = None,
                 model: str = DEFAULT_MODEL,
                 workers: int = 4) -> List[Dict[str, Any]]:
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("  [Gemini fill] no GEMINI_API_KEY – skipping.", file=sys.stderr)
        return rows
    if not rows:
        return rows

    gemini_url = GEMINI_URL.format(model=model)
    gs = requests.Session()
    gs.headers.update({"Content-Type": "application/json", "x-goog-api-key": api_key})

    # ── Pass 1: Fetch pages + extract from content ───────────────────────────
    print(f"\n  [Pass 1] Fetching {len(rows)} source pages ...", file=sys.stderr)
    pages: Dict[int, str] = {}

    def fetch_one(i):
        return i, _fetch_page(rows[i].get("trial_id", ""), rows[i].get("source_url", ""))

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, content in pool.map(lambda i: fetch_one(i), range(len(rows))):
            pages[i] = content

    fetched = sum(1 for c in pages.values() if c)
    print(f"  [Pass 1] {fetched}/{len(rows)} pages fetched", file=sys.stderr)

    # Build batches with content
    items = [{"idx": i, "trial_id": rows[i].get("trial_id", ""),
              "content": pages.get(i, "") or "(No content)"} for i in range(len(rows))]
    batches = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

    print(f"  [Pass 1] Extracting efficacy from page content ({len(batches)} batches) ...",
          file=sys.stderr)

    all_results: Dict[str, dict] = {}

    def run_pass1(bi, batch):
        time.sleep((bi % workers) * 0.3)
        raw = _gemini_call(gs, gemini_url, _pass1_prompt(drug, batch))
        return bi, _extract_trials(_parse_json(raw))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_pass1, bi, b): bi for bi, b in enumerate(batches)}
        done = 0
        for fut in as_completed(futs):
            bi, trials = fut.result()
            for gt in trials:
                tid = str(gt.get("trial_id", "")).strip()
                if tid:
                    all_results[tid] = gt
                    all_results[_norm_tid(tid)] = gt
            done += 1
            with _print_lock:
                print(f"  [Pass 1] batch {done}/{len(batches)} → {len(trials)}",
                      file=sys.stderr)

    # Merge Pass 1
    p1_filled = _merge(rows, all_results)
    print(f"  [Pass 1] Filled {p1_filled} fields from registry pages", file=sys.stderr)

    # ── Pass 2: Google Search for trials still missing ALL efficacy data ──────
    missing = [row for row in rows
               if not _has_any_efficacy(row)]
    if missing:
        print(f"\n  [Pass 2] {len(missing)} trials still missing efficacy data — "
              f"searching Google ...", file=sys.stderr)

        search_batches = [missing[i:i+SEARCH_BATCH_SIZE]
                          for i in range(0, len(missing), SEARCH_BATCH_SIZE)]
        pass2_results: Dict[str, dict] = {}

        def run_pass2(bi, batch):
            time.sleep((bi % workers) * 0.5)
            raw = _gemini_call(gs, gemini_url, _pass2_prompt(drug, batch),
                               use_search=True)
            return bi, _extract_trials(_parse_json(raw))

        with ThreadPoolExecutor(max_workers=max(1, workers - 1)) as pool:
            futs = {pool.submit(run_pass2, bi, b): bi
                    for bi, b in enumerate(search_batches)}
            done = 0
            for fut in as_completed(futs):
                bi, trials = fut.result()
                for gt in trials:
                    tid = str(gt.get("trial_id", "")).strip()
                    if tid:
                        pass2_results[tid] = gt
                        pass2_results[_norm_tid(tid)] = gt
                done += 1
                with _print_lock:
                    print(f"  [Pass 2] batch {done}/{len(search_batches)} → "
                          f"{len(trials)}", file=sys.stderr)

        p2_filled = _merge(rows, pass2_results)
        print(f"  [Pass 2] Filled {p2_filled} fields from Google Search",
              file=sys.stderr)
    else:
        print(f"\n  [Pass 2] All trials have some efficacy data — skipping.",
              file=sys.stderr)

    total_with = sum(1 for r in rows if _has_any_efficacy(r))
    print(f"\n  [Gemini fill] {total_with}/{len(rows)} trials have efficacy data",
          file=sys.stderr)
    return rows


def _has_any_efficacy(row: Dict[str, Any]) -> bool:
    for f in ("hba1c_change_pct", "weight_change_pct", "alt_reduction_pct",
              "mash_resolution_pct"):
        v = str(row.get(f, "") or "").strip().lower()
        if v and v not in ("", "n/a", "none", "null"):
            return True
    return False


def _merge(rows, gemini_map):
    filled = 0
    for row in rows:
        tid = str(row.get("trial_id", "")).strip()
        gt = gemini_map.get(tid) or gemini_map.get(_norm_tid(tid))
        if not gt: continue
        for field in EFFICACY_FIELDS:
            cur = str(row.get(field, "") or "").strip()
            new = str(gt.get(field, "") or "").strip()
            if (not cur or cur.lower() in ("", "n/a", "none", "null")) \
               and new and new.lower() not in ("", "n/a", "none", "null"):
                row[field] = new
                filled += 1
    return filled