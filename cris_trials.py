#!/usr/bin/env python3
"""
cris_trials.py – Clinical Research Information Service (CRIS, South Korea).

Two routes, tried in order:

1. **Official open API** on the Korean government data portal
   (https://www.data.go.kr – service "국립보건연구원_임상연구정보서비스").
   This needs a free service key. Export it before running:

       export CRIS_API_KEY="your-decoded-service-key"

   Endpoint: https://apis.data.go.kr/1352000/ODMS_CLINIC_1/callClinicSvc1

2. **HTML fallback** against https://cris.nih.go.kr search + detail pages,
   used automatically when no key is set or the API call fails.

Both routes return the same unified rows plus every extra field found.
"""

from __future__ import annotations
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from registry_common import (
    SRC_CRIS, blank_row, clean, extract_label_value_pairs, find_field,
    first_nonempty, flatten_xml, http_get, join, make_session, run_cli,
)

API_URL = "https://apis.data.go.kr/1352000/ODMS_CLINIC_1/callClinicSvc1"
BASE = "https://cris.nih.go.kr"
SEARCH_URL = BASE + "/cris/search/listDetail.do"
SEARCH_ALT = BASE + "/cris/en/search/search_result_st01.jsp"
SEARCH_KO = BASE + "/cris/search/search_result_st01.jsp"
DETAIL_URL_EN = BASE + "/cris/en/search/search_result_st01_en.jsp?seq={seq}"
DETAIL_URL_KO = BASE + "/cris/search/search_result_st01.jsp?seq={seq}"
SEQ_RE = re.compile(r"[?&]seq=(\d{3,})", re.I)  # require 3+ digits to avoid false matches
KCT_RE = re.compile(r"(KCT\d{7,})", re.I)
DELAY = 1.0


# ── Route 1: official open API ───────────────────────────────────────────────
def _fetch_api(drug: str, session, max_records: Optional[int]) -> List[Dict[str, Any]]:
    key = os.environ.get("CRIS_API_KEY", "").strip()
    if not key:
        return []

    rows: List[Dict[str, Any]] = []
    page, page_size = 1, 100
    while True:
        try:
            text = http_get(session, API_URL, params={
                "serviceKey": key,
                "pageNo": str(page),
                "numOfRows": str(page_size),
                "resultType": "xml",
                "searchWord": drug,
            })
        except Exception as exc:
            print(f"  [CRIS] open API call failed: {exc}", file=sys.stderr)
            return rows

        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            print("  [CRIS] open API did not return XML (check CRIS_API_KEY).",
                  file=sys.stderr)
            return rows

        items = [el for el in root.iter() if el.tag.split("}")[-1] == "item"]
        if not items:
            break

        for item in items:
            flat = flatten_xml(item)

            def g(*needles: str) -> str:
                for n in needles:
                    for k, v in flat.items():
                        if k.lower().endswith(n.lower()):
                            return v
                for n in needles:
                    for k, v in flat.items():
                        if n.lower() in k.lower():
                            return v
                return ""

            kct = g("researchRegNo", "regNo", "kctNo")
            row = blank_row(SRC_CRIS)
            row.update({
                "trial_id": kct,
                "secondary_ids": g("otherRegNo", "secondaryNo"),
                "title": first_nonempty(g("researchTitle"), g("scientificTitle"),
                                        g("titleEn")),
                "public_title": g("publicTitle", "titleKo"),
                "status": g("recruitStatus", "progressStatus"),
                "phase": g("researchPhase", "phase"),
                "study_type": g("researchType", "studyType"),
                "study_design": join([g("studyDesign"), g("allocation"),
                                      g("blinding"), g("maskingType")]),
                "conditions": g("targetDisease", "conditionName"),
                "interventions": g("intervention", "interventionName"),
                "drug_names": g("drugName", "intervention"),
                "sponsor": g("sponsorName", "institutionName", "leadOrg"),
                "sponsor_type": g("sponsorType", "fundingType"),
                "collaborators": g("collaborator", "cooperation"),
                "countries": first_nonempty(g("country"), "Republic of Korea"),
                "sites": g("institutionName", "siteName"),
                "target_enrollment": g("targetSize", "targetNumber"),
                "actual_enrollment": g("actualSize", "enrolledNumber"),
                "age_min": g("minAge", "ageMin"),
                "age_max": g("maxAge", "ageMax"),
                "gender": g("gender", "sex"),
                "healthy_volunteers": g("healthyVolunteer"),
                "inclusion_criteria": g("inclusionCriteria"),
                "exclusion_criteria": g("exclusionCriteria"),
                "primary_objective": g("objective", "purpose"),
                "primary_outcome": g("primaryOutcome", "primaryEndpoint"),
                "secondary_outcome": g("secondaryOutcome", "secondaryEndpoint"),
                "start_date": g("studyStartDate", "firstEnrollDate"),
                "completion_date": g("studyEndDate", "completionDate"),
                "registration_date": g("registerDate", "regDate"),
                "last_updated": g("updateDate", "modifyDate"),
                "results_available": g("resultYn", "hasResult"),
                "findings": join([g("resultSummary"), g("conclusion"),
                                  g("publicationInfo")]),
                "contact": join([g("contactName"), g("contactEmail")]),
                "ethics_approval": join([g("irbName"), g("irbApprovalDate"),
                                         g("irbStatus")]),
                "url": f"{BASE}/cris/search/detailSearch.do?seq={g('seq')}"
                       if g("seq") else f"{BASE}/cris/en/search/search.jsp",
            })
            for k, v in flat.items():
                col = "cris." + k
                if col not in row:
                    row[col] = v
            rows.append(row)
            if max_records and len(rows) >= max_records:
                return rows

        if len(items) < page_size:
            break
        page += 1
        time.sleep(DELAY)
    return rows


# ── Route 2: HTML fallback ───────────────────────────────────────────────────
def _search_seqs(drug: str, session, max_records: Optional[int]) -> List[str]:
    seqs: List[str] = []
    seen = set()
    for url, params in [
        (SEARCH_ALT, {"searchWord": drug, "search_page": "M"}),
        (SEARCH_KO, {"searchWord": drug, "search_page": "M"}),
        (SEARCH_URL, {"searchWord": drug}),
    ]:
        try:
            html = http_get(session, url, params=params)
        except Exception as exc:
            print(f"  [CRIS] {url} -> {exc}", file=sys.stderr)
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            m = SEQ_RE.search(a["href"])
            if not m:
                continue
            seq = m.group(1)
            if seq in seen:
                continue
            seen.add(seq)
            seqs.append(seq)
            if max_records and len(seqs) >= max_records:
                return seqs
        if seqs:
            break
        time.sleep(DELAY)
    return seqs


def _map_detail(html: str, seq: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    pairs = extract_label_value_pairs(soup)
    text = clean(soup.get_text(" ", strip=True))

    kct = find_field(pairs, "Unique Protocol ID", "CRIS Registration Number",
                     "Registration number")
    if not kct:
        m = KCT_RE.search(text)
        kct = m.group(1) if m else ""

    row = blank_row(SRC_CRIS)
    row.update({
        "trial_id": kct,
        "secondary_ids": join([find_field(pairs, "Secondary IDs", "Other IDs"),
                               find_field(pairs, "Unique Protocol ID")]),
        "title": first_nonempty(find_field(pairs, "Scientific Title",
                                           "Official Title"),
                                find_field(pairs, "Public/Brief Title",
                                           "Public Title")),
        "public_title": find_field(pairs, "Public/Brief Title", "Public Title"),
        "status": find_field(pairs, "Recruitment Status", "Study Status"),
        "phase": find_field(pairs, "Phase"),
        "study_type": find_field(pairs, "Study Type"),
        "study_design": join([find_field(pairs, "Study Design"),
                              find_field(pairs, "Allocation"),
                              find_field(pairs, "Blinding/Masking", "Masking"),
                              find_field(pairs, "Intervention Model"),
                              find_field(pairs, "Purpose")]),
        "conditions": join([find_field(pairs, "Condition", "Target Disease"),
                            find_field(pairs, "Health Condition")]),
        "interventions": join([find_field(pairs, "Intervention"),
                               find_field(pairs, "Arm/Group")]),
        "drug_names": join([find_field(pairs, "Intervention Name", "Drug"),
                            find_field(pairs, "Intervention")]),
        "sponsor": join([find_field(pairs, "Primary Sponsor", "Sponsor"),
                         find_field(pairs, "Organization Name")]),
        "sponsor_type": find_field(pairs, "Funding Source", "Sponsor Type"),
        "collaborators": join([find_field(pairs, "Secondary Sponsor"),
                               find_field(pairs, "Collaborator")]),
        "countries": first_nonempty(find_field(pairs, "Countries of Recruitment"),
                                    "Republic of Korea"),
        "sites": find_field(pairs, "Study Site", "Institution", "Facility"),
        "target_enrollment": find_field(pairs, "Target Number of Participant",
                                        "Target Sample Size"),
        "actual_enrollment": find_field(pairs, "Actual Number of Participant",
                                        "Enrollment"),
        "age_min": find_field(pairs, "Min Age", "Age Minimum"),
        "age_max": find_field(pairs, "Max Age", "Age Maximum"),
        "gender": find_field(pairs, "Gender", "Sex"),
        "healthy_volunteers": find_field(pairs, "Healthy Volunteers"),
        "inclusion_criteria": find_field(pairs, "Inclusion Criteria"),
        "exclusion_criteria": find_field(pairs, "Exclusion Criteria"),
        "primary_objective": find_field(pairs, "Brief Summary", "Objective",
                                        "Purpose of the study"),
        "primary_outcome": find_field(pairs, "Primary Outcome"),
        "secondary_outcome": find_field(pairs, "Secondary Outcome"),
        "start_date": join([find_field(pairs, "Date of First Enrollment",
                                       "Study Start Date"),
                            find_field(pairs, "Target First Enrollment Date")]),
        "completion_date": find_field(pairs, "Study Completion Date",
                                      "Primary Completion Date"),
        "registration_date": find_field(pairs, "Date of Registration",
                                        "Registered Date"),
        "last_updated": find_field(pairs, "Date of Last Update", "Last Update"),
        "results_available": find_field(pairs, "Results Registered",
                                        "Result Upload"),
        "findings": join([find_field(pairs, "Result Summary", "Study Results"),
                          find_field(pairs, "Conclusion"),
                          find_field(pairs, "Publication", "Related Publication")]),
        "contact": join([find_field(pairs, "Contact Person", "Name of Contact"),
                         find_field(pairs, "Email")]),
        "ethics_approval": join([find_field(pairs, "IRB Name", "Institutional Review Board"),
                                 find_field(pairs, "IRB Approval Date"),
                                 find_field(pairs, "IRB Approval Number")]),
        "url": url,
    })

    for k, v in pairs.items():
        col = "cris." + re.sub(r"\s+", "_", k)[:80]
        if col not in row:
            row[col] = v
    return row


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True) -> List[Dict[str, Any]]:
    session = make_session()

    rows = _fetch_api(drug, session, max_records)
    if rows:
        print(f"  CRIS open API returned {len(rows)} trial(s).", file=sys.stderr)
        return rows

    print("  [CRIS] falling back to HTML scrape "
          "(set CRIS_API_KEY to use the official open API).", file=sys.stderr)
    seqs = _search_seqs(drug, session, max_records)
    print(f"  CRIS search returned {len(seqs)} record(s).", file=sys.stderr)

    for i, seq in enumerate(seqs, start=1):
        row = None
        for url in [DETAIL_URL_EN.format(seq=seq), DETAIL_URL_KO.format(seq=seq)]:
            try:
                row = _map_detail(http_get(session, url), seq, url)
                break
            except Exception:
                continue
        if row:
            rows.append(row)
        else:
            print(f"  ! CRIS detail failed for seq={seq}", file=sys.stderr)
        if i % 10 == 0:
            print(f"  ...CRIS {i}/{len(seqs)}", file=sys.stderr)
        time.sleep(DELAY)
    return rows


if __name__ == "__main__":
    sys.exit(run_cli(fetch, SRC_CRIS, "Fetch trials from CRIS (South Korea)."))
