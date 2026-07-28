#!/usr/bin/env python3
"""
anzctr_trials.py – Australian New Zealand Clinical Trials Registry.

    Search : https://www.anzctr.org.au/TrialSearch.aspx
    Trial  : https://www.anzctr.org.au/Trial/Registration/TrialReview.aspx?id=<n>

ANZCTR has no JSON API, but each trial record can be retrieved as XML by
appending &isXml=true to the TrialReview URL. The XML is very complete, so we
flatten the whole document to capture every field. If the XML route fails we
fall back to parsing the HTML review page.
"""

from __future__ import annotations
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from registry_common import (
    SRC_ANZCTR, blank_row, clean, extract_label_value_pairs, find_field,
    first_nonempty, flatten_xml, http_get, join, make_session, run_cli,
)

BASE = "https://www.anzctr.org.au"
SEARCH_URL = BASE + "/TrialSearch.aspx?searchTxt={q}&isBasic=True"
SEARCH_ALT = BASE + "/BasicSearch.aspx?searchTxt={q}"
REVIEW_URL = BASE + "/Trial/Registration/TrialReview.aspx?id={tid}"
REVIEW_XML = BASE + "/Trial/Registration/TrialReview.aspx?id={tid}&isXml=true"
ACTRN_URL = BASE + "/Trial/Registration/TrialReview.aspx?ACTRN={actrn}"
DELAY = 1.0


def _search_ids(drug: str, session, max_records: Optional[int]) -> List[Dict[str, str]]:
    """Walk the search pages and collect (internal id, ACTRN) pairs."""
    found: List[Dict[str, str]] = []
    seen = set()
    page = 1
    while True:
        html = ""
        for url in [SEARCH_URL.format(q=quote_plus(drug)),
                     SEARCH_ALT.format(q=quote_plus(drug))]:
            try:
                html = http_get(session, url)
                if "TrialReview" in html or "ACTRN" in html:
                    break
            except Exception:
                continue
        soup = BeautifulSoup(html, "html.parser")

        page_hits = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"TrialReview\.aspx\?(?:id=(\d+)|ACTRN=([A-Za-z0-9]+))", href)
            if not m:
                continue
            tid, actrn = m.group(1), m.group(2)
            key = tid or actrn
            if key in seen:
                continue
            seen.add(key)
            found.append({"id": tid or "", "actrn": actrn or "",
                          "label": clean(a.get_text())})
            page_hits += 1
            if max_records and len(found) >= max_records:
                return found

        if page_hits == 0:
            break
        page += 1
        if page > 200:
            break
        time.sleep(DELAY)
    return found


def _from_xml(xml_text: str) -> Optional[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    flat = flatten_xml(root)

    def g(*needles: str) -> str:
        for needle in needles:
            for k, v in flat.items():
                if k.lower().endswith(needle.lower()):
                    return v
        for needle in needles:
            for k, v in flat.items():
                if needle.lower() in k.lower():
                    return v
        return ""

    actrn = g("actrnumber", "trialID", "anzctrId")
    row = blank_row(SRC_ANZCTR)
    row.update({
        "trial_id": actrn,
        "secondary_ids": join([g("secondaryId"), g("universalTrialNumber"),
                               g("nctId")]),
        "title": first_nonempty(g("scientificTitle"), g("publicTitle")),
        "public_title": g("publicTitle"),
        "status": first_nonempty(g("recruitmentStatus"), g("trialStatus")),
        "phase": g("phase"),
        "study_type": g("studyType"),
        "study_design": join([g("purpose"), g("allocation"), g("masking"),
                              g("assignment"), g("studyDesign")]),
        "conditions": join([g("healthCondition"), g("conditionCode"),
                            g("conditionCategory")]),
        "interventions": g("interventionDescription", "intervention"),
        "drug_names": g("interventionDescription", "intervention"),
        "sponsor": first_nonempty(g("primarySponsorName"), g("sponsorName")),
        "sponsor_type": g("primarySponsorType", "sponsorType"),
        "collaborators": join([g("secondarySponsorName"), g("fundingSource")]),
        "countries": join([g("countryOfRecruitment"), g("recruitmentCountry")]),
        "sites": g("recruitmentHospital", "hospital"),
        "target_enrollment": g("targetSampleSize", "anticipatedSampleSize"),
        "actual_enrollment": g("actualSampleSize"),
        "age_min": g("minimumAge"),
        "age_max": g("maximumAge"),
        "gender": g("gender", "sex"),
        "healthy_volunteers": g("healthyVolunteers"),
        "inclusion_criteria": g("inclusiveCriteria", "inclusionCriteria"),
        "exclusion_criteria": g("exclusionCriteria"),
        "primary_objective": first_nonempty(g("briefSummary"), g("trialPurpose")),
        "primary_outcome": g("primaryOutcome"),
        "secondary_outcome": g("secondaryOutcome"),
        "start_date": first_nonempty(g("anticipatedStartDate"),
                                     g("actualStartDate"), g("dateOfFirstEnrolment")),
        "completion_date": first_nonempty(g("actualEndDate"),
                                          g("anticipatedEndDate"),
                                          g("dateOfLastEnrolment")),
        "registration_date": g("dateRegistered", "submissionDate"),
        "last_updated": g("dateLastUpdated", "lastUpdated"),
        "results_available": g("hasResults", "resultsPublished"),
        "findings": join([g("summaryResults"), g("resultsSummary"),
                          g("keyResults"), g("publicationDetails")]),
        "contact": join([g("principalInvestigatorName"), g("contactName"),
                         g("contactEmail")]),
        "ethics_approval": join([g("ethicsCommitteeName"), g("ethicsApprovalDate"),
                                 g("ethicsApprovalStatus")]),
        "url": ACTRN_URL.format(actrn=actrn) if actrn else "",
    })

    for k, v in flat.items():
        col = "anzctr." + k
        if col not in row:
            row[col] = v
    return row


def _from_html(html: str, tid: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    pairs = extract_label_value_pairs(soup)

    actrn = find_field(pairs, "ACTRN", "Trial registration number", "Registration number")
    if not actrn:
        m = re.search(r"(ACTRN\d{11,})", soup.get_text(" ", strip=True))
        actrn = m.group(1) if m else ""

    row = blank_row(SRC_ANZCTR)
    row.update({
        "trial_id": actrn,
        "secondary_ids": find_field(pairs, "Secondary ID", "Universal Trial Number",
                                    "Trial acronym"),
        "title": first_nonempty(find_field(pairs, "Scientific title"),
                                find_field(pairs, "Public title")),
        "public_title": find_field(pairs, "Public title"),
        "status": find_field(pairs, "Recruitment status", "Trial status"),
        "phase": find_field(pairs, "Phase"),
        "study_type": find_field(pairs, "Study type"),
        "study_design": join([find_field(pairs, "Purpose of the study"),
                              find_field(pairs, "Allocation"),
                              find_field(pairs, "Masking"),
                              find_field(pairs, "Assignment")]),
        "conditions": join([find_field(pairs, "Health condition"),
                            find_field(pairs, "Condition category")]),
        "interventions": find_field(pairs, "Intervention", "Description of intervention"),
        "drug_names": find_field(pairs, "Intervention", "Description of intervention"),
        "sponsor": find_field(pairs, "Primary sponsor", "Sponsor name"),
        "sponsor_type": find_field(pairs, "Primary sponsor type", "Sponsor type"),
        "collaborators": join([find_field(pairs, "Secondary sponsor"),
                               find_field(pairs, "Funding source")]),
        "countries": find_field(pairs, "Countries of recruitment", "Recruitment outside"),
        "sites": find_field(pairs, "Recruitment hospital", "Recruitment state"),
        "target_enrollment": find_field(pairs, "Target sample size",
                                        "Anticipated sample size"),
        "actual_enrollment": find_field(pairs, "Actual sample size"),
        "age_min": find_field(pairs, "Minimum age"),
        "age_max": find_field(pairs, "Maximum age"),
        "gender": find_field(pairs, "Gender", "Sex"),
        "healthy_volunteers": find_field(pairs, "Healthy volunteers", "Can healthy"),
        "inclusion_criteria": find_field(pairs, "Inclusion criteria", "Key inclusion"),
        "exclusion_criteria": find_field(pairs, "Exclusion criteria", "Key exclusion"),
        "primary_objective": find_field(pairs, "Brief summary", "Study purpose"),
        "primary_outcome": find_field(pairs, "Primary outcome"),
        "secondary_outcome": find_field(pairs, "Secondary outcome"),
        "start_date": find_field(pairs, "Anticipated date of first participant",
                                 "Actual date of first participant", "Date of first"),
        "completion_date": find_field(pairs, "Actual date of last",
                                      "Anticipated date of last", "Date of last"),
        "registration_date": find_field(pairs, "Date registered", "Date submitted"),
        "last_updated": find_field(pairs, "Date last updated"),
        "results_available": find_field(pairs, "Results", "Has results"),
        "findings": join([find_field(pairs, "Summary results"),
                          find_field(pairs, "Results summary"),
                          find_field(pairs, "Key results"),
                          find_field(pairs, "Publication details")]),
        "contact": join([find_field(pairs, "Principal investigator"),
                         find_field(pairs, "Contact person"),
                         find_field(pairs, "Email")]),
        "ethics_approval": join([find_field(pairs, "Ethics committee name"),
                                 find_field(pairs, "Ethics approval date"),
                                 find_field(pairs, "Ethics application status")]),
        "url": REVIEW_URL.format(tid=tid) if tid else
               (ACTRN_URL.format(actrn=actrn) if actrn else ""),
    })

    for k, v in pairs.items():
        col = "anzctr." + re.sub(r"\s+", "_", k)[:80]
        if col not in row:
            row[col] = v
    return row


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True) -> List[Dict[str, Any]]:
    session = make_session()
    hits = _search_ids(drug, session, max_records)
    print(f"  ANZCTR search returned {len(hits)} trial link(s).", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for i, hit in enumerate(hits, start=1):
        tid, actrn = hit["id"], hit["actrn"]
        row: Optional[Dict[str, Any]] = None

        if details and tid:
            try:
                row = _from_xml(http_get(session, REVIEW_XML.format(tid=tid)))
            except Exception:
                row = None
            if row is None:
                try:
                    row = _from_html(http_get(session, REVIEW_URL.format(tid=tid)), tid)
                except Exception as exc:
                    print(f"  ! ANZCTR detail failed for {tid or actrn}: {exc}",
                          file=sys.stderr)
        if row is None:
            row = blank_row(SRC_ANZCTR)
            row["trial_id"] = actrn or tid
            row["title"] = hit["label"]
            row["url"] = (ACTRN_URL.format(actrn=actrn) if actrn
                          else REVIEW_URL.format(tid=tid))

        rows.append(row)
        if i % 20 == 0:
            print(f"  ...ANZCTR {i}/{len(hits)}", file=sys.stderr)
        time.sleep(DELAY)
    return rows


if __name__ == "__main__":
    sys.exit(run_cli(fetch, SRC_ANZCTR, "Fetch trials from ANZCTR."))
