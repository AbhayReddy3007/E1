#!/usr/bin/env python3
"""
eudract_drug_trials.py – Fetch EU clinical trials from the legacy EU-CTR
(EudraCT) register, including full protocol details and trial results/findings.
"""

import argparse, csv, html as html_mod, json, re, sys, time
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote_plus
import requests
from bs4 import BeautifulSoup, Tag

BASE = "https://www.clinicaltrialsregister.eu"
SEARCH_URL = BASE + "/ctr-search/search?query={query}&page={page}"
TRIAL_URL = BASE + "/ctr-search/trial/{eudract}/{country}"
RESULTS_URL = BASE + "/ctr-search/trial/{eudract}/results"

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
}
PER_PAGE = 20; TIMEOUT = 60; RETRIES = 3; RETRY_BACKOFF = 3; POLITE_DELAY = 0.6



# ── HTTP ─────────────────────────────────────────────────────────────────────
def _get(session, url):
    last_exc = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_exc = exc
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise RuntimeError(f"GET {url} failed after {RETRIES} attempts: {last_exc}")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _clean(text):
    return re.sub(r"\s+", " ", html_mod.unescape(text or "")).strip()

def _field(block_text, label, stop_labels):
    stops = "|".join(re.escape(s) for s in stop_labels)
    pattern = re.escape(label) + r"\s*:?\s*(.*?)(?=" + (stops + "|$" if stops else "$") + ")"
    m = re.search(pattern, block_text, flags=re.IGNORECASE | re.DOTALL)
    return _clean(m.group(1)) if m else ""


LABELS = [
    "EudraCT Number", "Sponsor Protocol Number", "Start Date", "Sponsor Name",
    "Full Title", "Medical condition", "Disease", "Population Age", "Gender",
    "Trial protocol", "Trial results",
]


# ── Search result parsing ────────────────────────────────────────────────────
def parse_result_block(block):
    text = _clean(block.get_text(" ", strip=True))
    if "EudraCT Number" not in text:
        return None
    def f(label):
        return _field(text, label, [l for l in LABELS if l != label])

    eudract = f("EudraCT Number")
    m = re.match(r"(\d{4}-\d{6}-\d{2})", eudract)
    eudract = m.group(1) if m else eudract
    if not eudract:
        return None

    countries, statuses = [], []
    for a in block.find_all("a", href=True):
        cm = re.search(r"/ctr-search/trial/[\d-]+/([A-Za-z0-9]+)", a["href"])
        if cm and "results" not in a["href"]:
            code = cm.group(1)
            if code not in countries:
                countries.append(code)
    for sm in re.finditer(r"\(([^()]{3,40})\)", _field(text, "Trial protocol", ["Trial results"])):
        status = sm.group(1).strip()
        if status not in statuses:
            statuses.append(status)

    has_results = "view results" in text.lower()
    start_raw = f("Start Date").lstrip("*: ").strip()

    return {
        "eudract_number": eudract,
        "sponsor_protocol_number": f("Sponsor Protocol Number"),
        "start_date": start_raw,
        "sponsor": f("Sponsor Name"),
        "title": f("Full Title"),
        "medical_condition": f("Medical condition"),
        "disease_meddra": f("Disease"),
        "population_age": f("Population Age"),
        "gender": f("Gender"),
        "countries": "; ".join(countries),
        "status": "; ".join(statuses),
        "results_available": "Yes" if has_results else "No",
        "results_url": RESULTS_URL.format(eudract=eudract) if has_results else "",
        "url": TRIAL_URL.format(eudract=eudract,
                                country=countries[0] if countries else "GB"),
    }


def total_results(soup):
    m = re.search(r"([\d,]+)\s+result\(s\)\s+found", soup.get_text(" ", strip=True))
    return int(m.group(1).replace(",", "")) if m else 0


# ── Search ───────────────────────────────────────────────────────────────────
def search_trials(drug, session, max_records=None, verbose=True):
    page, yielded, total = 1, 0, None
    query = quote_plus(drug)
    while True:
        html_text = _get(session, SEARCH_URL.format(query=query, page=page))
        soup = BeautifulSoup(html_text, "html.parser")
        if total is None:
            total = total_results(soup)
            if verbose:
                print(f"  EU-CTR reports {total} matching trial(s).", file=sys.stderr)
            if total == 0:
                return
        blocks = soup.select("table.result") or soup.find_all("table")
        found = 0
        for block in blocks:
            row = parse_result_block(block)
            if not row:
                continue
            found += 1
            yield row
            yielded += 1
            if max_records and yielded >= max_records:
                return
        if found == 0 or yielded >= total:
            return
        page += 1
        time.sleep(POLITE_DELAY)


# ── Protocol detail page ─────────────────────────────────────────────────────
def _table_value(soup, code_prefix):
    """Find a table row whose first cell matches code_prefix and return the value cell.
    Matching: first cell text starts with code_prefix (stripped), and the next char
    (if any) is not a digit or dot, to avoid E.5.1 matching E.5.1.1."""
    prefix = code_prefix.rstrip()
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) >= 2:
            first = _clean(cells[0].get_text(" ", strip=True))
            if first == prefix:
                return _clean(cells[-1].get_text(" ", strip=True))
            if first.startswith(prefix):
                rest = first[len(prefix):]
                # Allow match if remainder is whitespace/description, not a sub-code
                if rest and rest[0] in ".0123456789":
                    continue
                return _clean(cells[-1].get_text(" ", strip=True))
    return ""


def get_trial_details(eudract, country, session):
    page = _get(session, TRIAL_URL.format(eudract=eudract, country=country))
    soup = BeautifulSoup(page, "html.parser")
    text = _clean(soup.get_text(" ", strip=True))

    out = {}

    # A – Protocol
    out["full_title"] = _table_value(soup, "A.3 ") or _table_value(soup, "A.3")
    out["lay_title"] = _table_value(soup, "A.3.1")
    out["sponsor_protocol_code"] = _table_value(soup, "A.4")
    out["trial_is_pip"] = _table_value(soup, "A.7")

    # B – Sponsor
    out["sponsor_name"] = _table_value(soup, "B.1.1")
    out["sponsor_country"] = _table_value(soup, "B.1.3.4")
    out["sponsor_status"] = _table_value(soup, "B.3")

    # D – IMP
    out["product_name"] = _table_value(soup, "D.3.1")
    out["product_code"] = _table_value(soup, "D.3.2")
    out["pharmaceutical_form"] = _table_value(soup, "D.3.4 ")
    out["route"] = _table_value(soup, "D.3.7")
    out["inn"] = _table_value(soup, "D.3.8")
    out["cas_number"] = _table_value(soup, "D.3.9.1")
    out["strength"] = _table_value(soup, "D.3.10.3")
    out["strength_unit"] = _table_value(soup, "D.3.10.1")
    out["orphan_drug"] = _table_value(soup, "D.2.5 ")
    out["has_marketing_auth"] = _table_value(soup, "D.2.1")
    out["imp_role"] = _table_value(soup, "D.1.2")

    # E – General
    out["medical_condition_full"] = _table_value(soup, "E.1.1 ")
    out["condition_lay"] = _table_value(soup, "E.1.1.1")
    out["therapeutic_area"] = _table_value(soup, "E.1.1.2")
    out["rare_disease"] = _table_value(soup, "E.1.3")
    out["main_objective"] = _table_value(soup, "E.2.1")
    out["secondary_objectives"] = _table_value(soup, "E.2.2")
    out["primary_endpoint"] = _table_value(soup, "E.5.1 ")
    out["primary_endpoint_timeframe"] = _table_value(soup, "E.5.1.1")
    out["secondary_endpoint"] = _table_value(soup, "E.5.2 ")
    out["secondary_endpoint_timeframe"] = _table_value(soup, "E.5.2.1")
    out["inclusion_criteria"] = _table_value(soup, "E.3")
    out["exclusion_criteria"] = _table_value(soup, "E.4")

    # E.6 – Scope flags
    for label, key in [("E.6.1", "scope_diagnosis"), ("E.6.2", "scope_prophylaxis"),
                       ("E.6.3", "scope_therapy"), ("E.6.4", "scope_safety"),
                       ("E.6.5", "scope_efficacy"), ("E.6.6", "scope_pk"),
                       ("E.6.7", "scope_pd"), ("E.6.8", "scope_bioequivalence")]:
        out[key] = _table_value(soup, label)

    # E.7 – Phase: keep the raw label from the source
    phases = []
    for desc_fragment, label in [
        ("Human pharmacology", "Phase I"),
        ("Therapeutic exploratory", "Phase II"),
        ("Therapeutic confirmatory", "Phase III"),
        ("Therapeutic use", "Phase IV"),
    ]:
        for tr in soup.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) >= 2:
                row_text = _clean(tr.get_text(" ", strip=True))
                if desc_fragment in row_text:
                    last_val = _clean(cells[-1].get_text(" ", strip=True)).lower()
                    if last_val.startswith("yes"):
                        phases.append(label)
                    break
    out["phase"] = ", ".join(phases)

    # E.8 – Design
    for label, key in [("E.8.1 ", "controlled"), ("E.8.1.1", "randomised"),
                       ("E.8.1.2", "open_label"), ("E.8.1.3", "single_blind"),
                       ("E.8.1.4", "double_blind"), ("E.8.1.5", "parallel_group"),
                       ("E.8.1.6", "crossover"),
                       ("E.8.2.1", "comparator_other_product"),
                       ("E.8.2.2", "comparator_placebo"),
                       ("E.8.2.4", "number_of_arms"),
                       ("E.8.4.1", "sites_in_member_state"),
                       ("E.8.5.1", "sites_in_eea"),
                       ("E.8.7", "has_dmc")]:
        out[key] = _table_value(soup, label)

    # F – Population
    out["subjects_member_state"] = _table_value(soup, "F.4.1")
    out["subjects_eea"] = _table_value(soup, "F.4.2.1")
    out["subjects_worldwide"] = _table_value(soup, "F.4.2.2")
    out["healthy_volunteers"] = _table_value(soup, "F.3.1")

    # N – Regulatory
    out["ca_decision"] = ""
    out["ca_decision_date"] = ""
    out["ethics_opinion"] = ""
    out["ethics_opinion_date"] = ""
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) >= 2:
            label = _clean(cells[0].get_text())
            val = _clean(cells[-1].get_text())
            if "Competent Authority Decision" in label and "Date" not in label:
                out["ca_decision"] = val
            elif "Date of Competent Authority" in label:
                out["ca_decision_date"] = val
            elif "Ethics Committee Opinion" in label and "Date" not in label and "Reason" not in label:
                out["ethics_opinion"] = val
            elif "Date of Ethics Committee" in label:
                out["ethics_opinion_date"] = val

    # P – End of trial
    out["end_of_trial_status"] = ""
    out["global_end_date"] = ""
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) >= 2:
            label = _clean(cells[0].get_text())
            val = _clean(cells[-1].get_text())
            if "End of Trial Status" in label:
                out["end_of_trial_status"] = val
            elif "global end of the trial" in label.lower():
                out["global_end_date"] = val

    return out


# ── Results page (findings) ──────────────────────────────────────────────────
def get_trial_results(eudract, session):
    """Parse the results page for endpoint findings, adverse events, etc."""
    try:
        page = _get(session, RESULTS_URL.format(eudract=eudract))
    except Exception:
        return {}
    soup = BeautifulSoup(page, "html.parser")
    text = _clean(soup.get_text(" ", strip=True))

    out = {}

    # global end date & recruitment date
    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) >= 2:
            label = _clean(cells[0].get_text())
            val = _clean(cells[-1].get_text())
            if "Global end of trial date" in label:
                out["results_global_end_date"] = val
            elif "Actual start date of recruitment" in label:
                out["results_recruitment_start"] = val
            elif "Worldwide total number of subjects" in label:
                out["results_subjects_worldwide"] = val
            elif "EEA total number of subjects" in label:
                out["results_subjects_eea"] = val
            elif "Main objective" in label:
                out["results_main_objective"] = val
            elif "Analysis stage" in label and "Date" not in label:
                out["results_analysis_stage"] = val
            elif "Date of interim/final analysis" in label:
                out["results_analysis_date"] = val

    # Endpoint titles and descriptions
    findings_parts = []
    # Look for sections with "Primary:" or "Secondary:" in h3/h4 or in bold text
    for section in soup.find_all(["td", "th"]):
        sec_text = _clean(section.get_text(" ", strip=True))
        if ("End point title" in sec_text or "end point title" in sec_text.lower()):
            # the title is typically in the next row or in the value cell
            pass

    # Broader approach: scan for endpoint blocks in the text
    endpoint_blocks = re.findall(
        r"((?:Primary|Secondary)\s*:\s*.+?)(?=(?:Primary|Secondary)\s*:|Adverse events|More Information|$)",
        text, flags=re.IGNORECASE | re.DOTALL
    )
    for block in endpoint_blocks:
        # Extract title
        title_m = re.search(r"End point title\s+(.+?)(?:End point description|End point type|$)",
                            block, re.IGNORECASE | re.DOTALL)
        title = _clean(title_m.group(1)) if title_m else ""

        desc_m = re.search(r"End point description\s+(.+?)(?:End point type|End point timeframe|$)",
                           block, re.IGNORECASE | re.DOTALL)
        desc = _clean(desc_m.group(1)) if desc_m else ""

        # Extract statistical results
        stat_parts = []
        for pat in [
            r"p-value\s*[=:]\s*([\d.<>eE\-]+)",
            r"Confidence interval\s*[:]?\s*([\d\.\-\sto]+)",
            r"(?:mean|median)\s+difference\s*[=:]\s*([\d\.\-\sto]+)",
            r"(?:hazard|odds|risk)\s+ratio\s*[=:]\s*([\d\.\-\sto]+)",
            r"Statistical analysis\s+(.{10,300}?)(?:Notes|End point|$)",
        ]:
            for m in re.finditer(pat, block, re.IGNORECASE):
                stat_parts.append(m.group(0).strip()[:200])

        line = " | ".join(p for p in [title, desc] + stat_parts if p)
        if line:
            findings_parts.append(line[:500])

    # Also grab the "Notes" with justifications (often has termination reasons)
    notes = re.findall(r"Justification:\s*(.{10,500}?)(?:\n|Notes|$)", text, re.IGNORECASE)
    for n in notes:
        findings_parts.append(f"Note: {_clean(n)[:300]}")

    # Adverse events summary
    ae_m = re.search(
        r"Adverse event reporting additional description\s+(.{10,500}?)(?:Assessment type|Dictionary|$)",
        text, re.IGNORECASE | re.DOTALL)
    if ae_m:
        out["adverse_events_summary"] = _clean(ae_m.group(1))[:500]

    # Limitations
    lim_m = re.search(
        r"Limitations of the trial\s+(?:such as.*?\.)\s*(.{10,800}?)(?:For support|$)",
        text, re.IGNORECASE | re.DOTALL)
    if lim_m:
        out["limitations"] = _clean(lim_m.group(1))[:500]

    out["findings"] = " ;; ".join(findings_parts) if findings_parts else ""
    return out


def mentions_drug(row, drug):
    needle = drug.lower()
    fields = ("product_name", "inn", "title", "full_title", "trade_name", "product_code")
    return any(needle in str(row.get(f, "")).lower() for f in fields)


# ── Output ───────────────────────────────────────────────────────────────────
def write_csv(rows, path):
    if not rows: return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Fetch EU clinical trials from the EU-CTR (EudraCT) register.")
    ap.add_argument("drug")
    ap.add_argument("--details", action="store_true",
                    help="Fetch full protocol page (sections A-P)")
    ap.add_argument("--results", action="store_true",
                    help="Fetch results/findings page (endpoints, adverse events)")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--format", choices=("csv", "json", "both"), default="both")
    args = ap.parse_args()
    prefix = args.out or f"{args.drug.lower().replace(' ', '_')}_eudract_trials"
    session = requests.Session()
    session.headers.update(HEADERS)
    print(f"Searching EU-CTR for '{args.drug}' ...", file=sys.stderr)
    rows = []
    try:
        for i, row in enumerate(search_trials(args.drug, session,
                                              max_records=args.max_records), start=1):
            if args.details:
                country = (row["countries"].split(";")[0] or "GB").strip()
                try:
                    row.update(get_trial_details(row["eudract_number"], country, session))
                except Exception as exc:
                    print(f"  ! details failed for {row['eudract_number']}: {exc}",
                          file=sys.stderr)
                time.sleep(POLITE_DELAY)

            if args.results and row.get("results_available") == "Yes":
                try:
                    row.update(get_trial_results(row["eudract_number"], session))
                except Exception as exc:
                    print(f"  ! results failed for {row['eudract_number']}: {exc}",
                          file=sys.stderr)
                time.sleep(POLITE_DELAY)

            if args.strict and not mentions_drug(row, args.drug):
                continue
            rows.append(row)
            if i % 20 == 0:
                print(f"  ...{i} trials processed", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if not rows: return 1
    if not rows:
        print("No trials found.", file=sys.stderr); return 0
    if args.format in ("csv", "both"):
        write_csv(rows, f"{prefix}.csv")
        print(f"Wrote {len(rows)} trials -> {prefix}.csv", file=sys.stderr)
    if args.format in ("json", "both"):
        write_json(rows, f"{prefix}.json")
        print(f"Wrote {len(rows)} trials -> {prefix}.json", file=sys.stderr)
    for row in rows[:5]:
        ph = row.get("phase", "")
        print(f"  {row['eudract_number']}  [Phase {ph}] [{row.get('status','')}]  "
              f"{row.get('title','')[:70]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
