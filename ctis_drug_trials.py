#!/usr/bin/env python3
"""
ctis_drug_trials.py – Fetch EU clinical trials from the CTIS public portal.
"""

import argparse, csv, json, re, sys, time
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote
import requests

BASE = "https://euclinicaltrials.eu"
SEARCH_URL = f"{BASE}/ctis-public-api/search"
RETRIEVE_URL = f"{BASE}/ctis-public-api/retrieve/{{ct_number}}"
TRIAL_PAGE_URL = f"{BASE}/ctis-public/search?lang=en&EUCT={{ct_number}}"

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Origin": BASE,
    "Referer": f"{BASE}/ctis-public/search",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
}
TIMEOUT = 60; RETRIES = 3; RETRY_BACKOFF = 3



# ── HTTP ─────────────────────────────────────────────────────────────────────
def _request(session, method, url, **kw):
    last_exc = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.request(method, url, timeout=TIMEOUT, **kw)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise RuntimeError(f"Request to {url} failed after {RETRIES} attempts: {last_exc}")


def _search_page(session, criteria, page, size):
    body = {
        "pagination": {"page": page, "size": size},
        "sort": {"property": "decisionDate", "direction": "DESC"},
        "searchCriteria": criteria,
    }
    resp = _request(session, "POST", SEARCH_URL, data=json.dumps(body))
    if resp.ok:
        return resp.json()
    url = (f"{SEARCH_URL}"
           f"?paging={quote(json.dumps({'page': page, 'size': size}))}"
           f"&searchCriteria={quote(json.dumps(criteria))}"
           f"&sort={quote(json.dumps({'property': 'decisionDate', 'direction': 'DESC'}))}")
    resp = _request(session, "GET", url)
    resp.raise_for_status()
    return resp.json()


# ── Search ───────────────────────────────────────────────────────────────────
def search_trials(drug, session, page_size=50, max_records=None, verbose=True):
    criteria = {
        "containAll": drug, "containAny": None, "containNot": None,
        "title": None, "number": None, "status": None,
        "medicalCondition": None, "sponsor": None, "productName": None,
        "trialPhaseCode": None, "msc": None, "ageGroupCode": None,
        "therapeuticAreaCode": None, "gender": None, "eudraCtCode": None,
        "trialRegion": None,
    }
    page, yielded = 1, 0
    while True:
        payload = _search_page(session, criteria, page, page_size)
        records = payload.get("data") or []
        pagination = payload.get("pagination") or {}
        if verbose and page == 1:
            print(f"  CTIS reports {pagination.get('totalRecords', len(records))} "
                  f"matching trial(s).", file=sys.stderr)
        for rec in records:
            yield rec
            yielded += 1
            if max_records and yielded >= max_records:
                return
        if not pagination.get("nextPage") or not records:
            return
        page += 1
        time.sleep(0.4)


def get_trial_details(ct_number, session):
    resp = _request(session, "GET", RETRIEVE_URL.format(ct_number=ct_number))
    resp.raise_for_status()
    return resp.json()


# ── Flatten ──────────────────────────────────────────────────────────────────
def _as_list(val):
    """Ensure a value is a list – API fields sometimes return str/int/None."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [val] if val else []
    return [str(val)]


def _safe(d, *keys, default=""):
    v = d
    for k in keys:
        if isinstance(v, dict):
            v = v.get(k)
        else:
            return default
    return v if v is not None else default


def _part_one(details):
    return _safe(details, "authorizedApplication", "authorizedPartI", default={})


def extract_products(details):
    out = []
    for prod in _safe(_part_one(details), "products", default=[]) or []:
        info = prod.get("productDictionaryInfo") or {}
        out.append({
            "product_name": info.get("prodName") or prod.get("productName") or "",
            "active_substance": info.get("activeSubstanceName") or "",
            "atc_code": info.get("atcCode") or "",
            "pharmaceutical_form": info.get("pharmForm") or "",
            "route": ", ".join(_as_list(prod.get("routes"))),
            "strength": info.get("strength") or "",
            "imp_role": prod.get("impRole") or "",
            "orphan_drug": str(prod.get("orphanDrug", "")),
            "has_marketing_auth": str(prod.get("hasMarketingAuth", "")),
        })
    return out


def extract_sponsors(details):
    out = []
    for sp in _safe(_part_one(details), "sponsors", default=[]) or []:
        org = sp.get("organisation") or {}
        out.append({
            "name": org.get("name", ""),
            "country": org.get("countryName", ""),
            "status": sp.get("sponsorType", ""),
        })
    return out


def _extract_results_summary(details):
    """Pull trial results / findings from the CTIS detail record."""
    results = _safe(details, "results", default={}) or {}
    if not results:
        results = _safe(details, "authorizedApplication", "results", default={}) or {}
    summaries = []
    for ep in _safe(results, "endpoints", default=[]) or []:
        title = ep.get("title") or ep.get("name") or ""
        ep_type = ep.get("type") or ""
        desc = ep.get("description") or ""
        stat = ep.get("statisticalAnalysis") or ep.get("result") or ""
        conclusion = ep.get("conclusion") or ""
        parts = [p for p in [f"[{ep_type}]" if ep_type else "",
                             title, desc, stat, conclusion] if p]
        if parts:
            summaries.append(" | ".join(parts))
    overall = _safe(results, "summaryOfResults") or _safe(results, "overallConclusion") or ""
    if overall:
        summaries.insert(0, f"Overall: {overall}")
    return " ;; ".join(summaries)


def flatten(summary, details=None):
    ct_number = summary.get("ctNumber", "")

    row = {
        "ct_number": ct_number,
        "eudract_number": summary.get("eudraCtCode") or "",
        "title": summary.get("ctTitle", ""),
        "short_title": summary.get("shortTitle", ""),
        "status": summary.get("ctStatus", ""),
        "phase": summary.get("trialPhase", ""),
        "conditions": summary.get("conditions", ""),
        "sponsor": summary.get("sponsor", ""),
        "sponsor_type": summary.get("sponsorType", ""),
        "product": summary.get("product", ""),
        "countries": "; ".join(_as_list(summary.get("trialCountries"))),
        "therapeutic_areas": "; ".join(_as_list(summary.get("therapeuticAreas"))),
        "age_group": summary.get("ageGroup", ""),
        "gender": summary.get("gender", ""),
        "enrolled": summary.get("totalNumberEnrolled", ""),
        "primary_endpoint": summary.get("primaryEndPoint", ""),
        "secondary_endpoint": summary.get("secondaryEndPoint", ""),
        "decision_date": summary.get("decisionDateOverall", ""),
        "start_date": summary.get("startDate") or summary.get("startDateEU")
                      or summary.get("decisionDateOverall") or "",
        "end_date": summary.get("endDate") or "",
        "results_first_received": summary.get("resultsFirstReceived", ""),
        "last_updated": summary.get("lastUpdated", ""),
        "trial_region": "; ".join(_as_list(summary.get("trialRegion"))),
        "msc": summary.get("msc", ""),
        "url": TRIAL_PAGE_URL.format(ct_number=ct_number),
    }

    if details:
        products = extract_products(details)
        sponsors = extract_sponsors(details)

        row["product_names"] = "; ".join(
            sorted({p["product_name"] for p in products if p["product_name"]}))
        row["active_substances"] = "; ".join(
            sorted({p["active_substance"] for p in products if p["active_substance"]}))
        row["atc_codes"] = "; ".join(
            sorted({p["atc_code"] for p in products
                    if p["atc_code"] and p["atc_code"] != "-"}))
        row["routes"] = "; ".join(sorted({p["route"] for p in products if p["route"]}))
        row["pharmaceutical_forms"] = "; ".join(
            sorted({p["pharmaceutical_form"] for p in products if p["pharmaceutical_form"]}))
        row["strengths"] = "; ".join(
            sorted({p["strength"] for p in products if p["strength"]}))
        row["imp_roles"] = "; ".join(
            sorted({p["imp_role"] for p in products if p["imp_role"]}))
        row["orphan_drug"] = "; ".join(
            sorted({p["orphan_drug"] for p in products if p["orphan_drug"]}))
        row["has_marketing_auth"] = "; ".join(
            sorted({p["has_marketing_auth"] for p in products if p["has_marketing_auth"]}))
        row["sponsors_full"] = "; ".join(
            sorted({s["name"] for s in sponsors if s["name"]}))
        row["sponsor_countries"] = "; ".join(
            sorted({s["country"] for s in sponsors if s["country"]}))
        row["start_date_eu"] = details.get("startDateEU") or ""
        row["end_date_eu"] = details.get("endDateEU") or ""
        if not row["start_date"]:
            row["start_date"] = row["start_date_eu"]

        # trial design fields from part I
        p1 = _part_one(details)
        row["main_objective"] = _safe(p1, "trialInformation", "mainObjective")
        row["secondary_objectives"] = _safe(p1, "trialInformation", "secondaryObjectives")
        row["inclusion_criteria"] = _safe(p1, "trialInformation", "inclusionCriteria")
        row["exclusion_criteria"] = _safe(p1, "trialInformation", "exclusionCriteria")
        row["trial_design"] = _safe(p1, "trialInformation", "trialDesign")
        row["comparator"] = _safe(p1, "trialInformation", "comparator")
        row["planned_subjects_eea"] = _safe(p1, "trialInformation", "numberSubjectsEEA")
        row["planned_subjects_worldwide"] = _safe(p1, "trialInformation", "numberSubjectsWorldwide")

        # results / findings
        row["findings"] = _extract_results_summary(details)

    return row


def mentions_drug(row, drug):
    needle = drug.lower()
    fields = ("product", "product_names", "active_substances", "atc_codes")
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
        description="Fetch EU/EEA clinical trials from the CTIS public portal.")
    ap.add_argument("drug")
    ap.add_argument("--details", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--format", choices=("csv", "json", "both"), default="both")
    args = ap.parse_args()
    prefix = args.out or f"{args.drug.lower().replace(' ', '_')}_ctis_trials"
    session = requests.Session()
    session.headers.update(HEADERS)
    print(f"Searching CTIS for '{args.drug}' ...", file=sys.stderr)
    rows = []
    try:
        for i, summary in enumerate(
                search_trials(args.drug, session, page_size=args.page_size,
                              max_records=args.max_records), start=1):
            details = None
            if args.details:
                try:
                    details = get_trial_details(summary.get("ctNumber", ""), session)
                except Exception as exc:
                    print(f"  ! details failed for {summary.get('ctNumber')}: {exc}",
                          file=sys.stderr)
                time.sleep(0.3)
            row = flatten(summary, details)
            if args.strict and not mentions_drug(row, args.drug):
                continue
            rows.append(row)
            if i % 25 == 0:
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
        print(f"  {row['ct_number']}  [Phase {row['phase']}] [{row['status']}]  "
              f"{row['title'][:70]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
