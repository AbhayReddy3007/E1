#!/usr/bin/env python3
"""
ctgov_trials.py – ClinicalTrials.gov (USA), via the official REST API v2.

    API docs : https://clinicaltrials.gov/data-api/api
    Endpoint : https://clinicaltrials.gov/api/v2/studies

This is a fully documented public API, no key required. Because the API
returns the complete study record as JSON, every field is captured by
flattening the whole record into dotted columns.
"""

from __future__ import annotations
import sys
from typing import Any, Dict, List, Optional

from registry_common import (
    SRC_CTGOV, UNIFIED_COLUMNS, blank_row, clean, first_nonempty,
    flatten_json, http_get, join, make_session, run_cli,
)

API = "https://clinicaltrials.gov/api/v2/studies"
TRIAL_URL = "https://clinicaltrials.gov/study/{nct}"
PAGE_SIZE = 100


def _s(study: Dict[str, Any], *path, default=None):
    """Safe nested getter."""
    cur: Any = study
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return default
    return cur if cur is not None else default


def _map(study: Dict[str, Any]) -> Dict[str, Any]:
    ps = _s(study, "protocolSection", default={}) or {}
    ident = ps.get("identificationModule", {}) or {}
    status_m = ps.get("statusModule", {}) or {}
    design = ps.get("designModule", {}) or {}
    arms = ps.get("armsInterventionsModule", {}) or {}
    conds = ps.get("conditionsModule", {}) or {}
    elig = ps.get("eligibilityModule", {}) or {}
    outcomes = ps.get("outcomesModule", {}) or {}
    sponsor_m = ps.get("sponsorCollaboratorsModule", {}) or {}
    contacts = ps.get("contactsLocationsModule", {}) or {}
    oversight = ps.get("oversightModule", {}) or {}
    desc = ps.get("descriptionModule", {}) or {}

    nct = ident.get("nctId", "")
    interventions = arms.get("interventions", []) or []
    drug_names = [i.get("name", "") for i in interventions
                  if str(i.get("type", "")).upper() in ("DRUG", "BIOLOGICAL")]
    all_interventions = [f"{i.get('type', '')}: {i.get('name', '')}".strip(": ")
                         for i in interventions]

    locations = contacts.get("locations", []) or []
    countries = join(l.get("country", "") for l in locations)
    sites = join(f"{l.get('facility', '')} ({l.get('city', '')})".strip()
                 for l in locations[:40])

    secondary = [x.get("id", "") for x in (ident.get("secondaryIdInfos") or [])]
    if ident.get("orgStudyIdInfo", {}).get("id"):
        secondary.insert(0, ident["orgStudyIdInfo"]["id"])

    enroll = design.get("enrollmentInfo", {}) or {}
    central = contacts.get("centralContacts", []) or []

    row = blank_row(SRC_CTGOV)
    row.update({
        "trial_id": nct,
        "secondary_ids": join(secondary),
        "title": first_nonempty(ident.get("officialTitle"), ident.get("briefTitle")),
        "public_title": clean(ident.get("briefTitle")),
        "status": clean(status_m.get("overallStatus")),
        "phase": join(design.get("phases") or []),
        "study_type": clean(design.get("studyType")),
        "study_design": join([
            _s(design, "designInfo", "allocation", default=""),
            _s(design, "designInfo", "interventionModel", default=""),
            _s(design, "designInfo", "primaryPurpose", default=""),
            _s(design, "designInfo", "maskingInfo", "masking", default=""),
        ]),
        "conditions": join(conds.get("conditions") or []),
        "interventions": join(all_interventions),
        "drug_names": join(drug_names),
        "sponsor": _s(sponsor_m, "leadSponsor", "name", default=""),
        "sponsor_type": _s(sponsor_m, "leadSponsor", "class", default=""),
        "collaborators": join(c.get("name", "") for c in
                              (sponsor_m.get("collaborators") or [])),
        "countries": countries,
        "sites": sites,
        "target_enrollment": (str(enroll.get("count", ""))
                              if str(enroll.get("type", "")).upper() == "ESTIMATED" else ""),
        "actual_enrollment": (str(enroll.get("count", ""))
                              if str(enroll.get("type", "")).upper() == "ACTUAL" else ""),
        "age_min": clean(elig.get("minimumAge")),
        "age_max": clean(elig.get("maximumAge")),
        "gender": clean(elig.get("sex")),
        "healthy_volunteers": str(elig.get("healthyVolunteers", "")),
        "inclusion_criteria": clean(elig.get("eligibilityCriteria")),
        "exclusion_criteria": "",   # CTGOV merges both into eligibilityCriteria
        "primary_objective": clean(desc.get("briefSummary")),
        "primary_outcome": join(
            f"{o.get('measure', '')} [{o.get('timeFrame', '')}]"
            for o in (outcomes.get("primaryOutcomes") or [])),
        "secondary_outcome": join(
            f"{o.get('measure', '')} [{o.get('timeFrame', '')}]"
            for o in (outcomes.get("secondaryOutcomes") or [])),
        "start_date": _s(status_m, "startDateStruct", "date", default=""),
        "completion_date": _s(status_m, "completionDateStruct", "date", default=""),
        "registration_date": _s(status_m, "studyFirstSubmitDate", default=""),
        "last_updated": _s(status_m, "lastUpdatePostDateStruct", "date", default=""),
        "results_available": "Yes" if study.get("hasResults") else "No",
        "findings": _extract_findings(study),
        "contact": join(f"{c.get('name', '')} {c.get('email', '')}".strip()
                        for c in central),
        "ethics_approval": clean(oversight.get("oversightHasDmc")),
        "url": TRIAL_URL.format(nct=nct) if nct else "",
    })

    # every remaining API field, flattened
    for k, v in flatten_json(study).items():
        col = "ctgov." + k
        if col not in row:
            row[col] = v
    return row


def _extract_findings(study: Dict[str, Any]) -> str:
    """Build a findings string from the resultsSection when present."""
    rs = study.get("resultsSection") or {}
    if not rs:
        return ""
    parts: List[str] = []

    for og in (_s(rs, "outcomeMeasuresModule", "outcomeMeasures", default=[]) or []):
        title = clean(og.get("title"))
        otype = clean(og.get("type"))
        unit = clean(og.get("unitOfMeasure"))
        desc = clean(og.get("description"))
        values = []
        for cls in (og.get("classes") or [])[:3]:
            for cat in (cls.get("categories") or [])[:3]:
                for m in (cat.get("measurements") or [])[:6]:
                    v = clean(m.get("value"))
                    lo, hi = clean(m.get("lowerLimit")), clean(m.get("upperLimit"))
                    if v:
                        values.append(f"{v}" + (f" ({lo}-{hi})" if lo or hi else ""))
        stats = []
        for an in (og.get("analyses") or [])[:3]:
            pv = clean(an.get("pValue"))
            pm = clean(an.get("paramValue"))
            pt = clean(an.get("paramType"))
            ci_lo, ci_hi = clean(an.get("ciLowerLimit")), clean(an.get("ciUpperLimit"))
            bits = []
            if pt or pm:
                bits.append(f"{pt}={pm}".strip("="))
            if ci_lo or ci_hi:
                bits.append(f"95%CI {ci_lo}-{ci_hi}")
            if pv:
                bits.append(f"p={pv}")
            if bits:
                stats.append(", ".join(bits))
        line = " | ".join(x for x in [
            f"[{otype}]" if otype else "", title, desc,
            f"unit={unit}" if unit else "",
            "values: " + ", ".join(values[:8]) if values else "",
            "stats: " + "; ".join(stats) if stats else "",
        ] if x)
        if line:
            parts.append(line[:600])

    ae = _s(rs, "adverseEventsModule", "frequencyThreshold", default="")
    serious = _s(rs, "adverseEventsModule", "seriousEvents", default=[]) or []
    if serious:
        top = "; ".join(clean(e.get("term")) for e in serious[:5])
        parts.append(f"Serious AEs (top): {top}")
    if ae:
        parts.append(f"AE frequency threshold: {ae}")

    lim = _s(rs, "moreInfoModule", "limitationsAndCaveats", "description", default="")
    if lim:
        parts.append(f"Limitations: {clean(lim)[:400]}")

    return " ;; ".join(parts)


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True) -> List[Dict[str, Any]]:
    """Fetch ClinicalTrials.gov studies matching `drug`."""
    session = make_session({"Accept": "application/json"})
    rows: List[Dict[str, Any]] = []
    token = None

    while True:
        params = {
            "query.intr": drug,
            "pageSize": str(min(PAGE_SIZE, max_records or PAGE_SIZE)),
            "format": "json",
            "countTotal": "true",
        }
        if token:
            params["pageToken"] = token

        payload = http_get(session, API, params=params, expect_json=True)
        studies = payload.get("studies") or []
        if not rows and payload.get("totalCount") is not None:
            print(f"  ClinicalTrials.gov reports {payload['totalCount']} study(ies).",
                  file=sys.stderr)

        for study in studies:
            rows.append(_map(study))
            if max_records and len(rows) >= max_records:
                return rows

        token = payload.get("nextPageToken")
        if not token or not studies:
            break

    return rows


if __name__ == "__main__":
    sys.exit(run_cli(fetch, SRC_CTGOV, "Fetch trials from ClinicalTrials.gov (API v2)."))
