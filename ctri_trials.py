#!/usr/bin/env python3
"""
ctri_trials.py – Clinical Trials Registry - India (CTRI).

    Search : https://ctri.nic.in/Clinicaltrials/advsearch.php
    Trial  : https://ctri.nic.in/Clinicaltrials/pmaindet2.php?EncHid=<hash>

CTRI has no public API. The site is a classic PHP app whose trial pages are
label/value tables, so we harvest every label/value pair on the page rather
than hard-coding a field list. CTRI is slow and rate-sensitive, so requests
are throttled.
"""

from __future__ import annotations
import re
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup

from registry_common import (
    SRC_CTRI, blank_row, clean, extract_label_value_pairs, find_field,
    first_nonempty, http_get, http_post, join, make_session, run_cli,
)

BASE = "https://ctri.nic.in"
SEARCH_URL = BASE + "/Clinicaltrials/advancesearchmain.php"
SEARCH_GET = BASE + "/Clinicaltrials/showallp.php?mid1=&EncHid=&userName={q}"
DETAIL_RE = re.compile(r"(?:showallp|pmaindet2)\.php\?(?:mid1=\d*&)?EncHid=([^&\"'\s]*)", re.I)
DETAIL_ALT = re.compile(r"showallp\.php\?mid1=(\d+)", re.I)
DELAY = 1.5


def _search_links(drug: str, session, max_records: Optional[int]) -> List[str]:
    """Collect detail-page hrefs for trials matching the drug."""
    links: List[str] = []
    seen = set()

    # CTRI's public keyword search endpoint
    candidates = [
        SEARCH_GET.format(q=quote_plus(drug)),
        f"{BASE}/Clinicaltrials/showallp.php?mid1=&EncHid=&userName={quote_plus(drug)}",
    ]

    html = ""
    for url in candidates:
        try:
            html = http_get(session, url)
            if DETAIL_RE.search(html):
                break
        except Exception as exc:
            print(f"  [CTRI] {url} -> {exc}", file=sys.stderr)

    if not html or not DETAIL_RE.search(html):
        # POST form fallback
        try:
            html = http_post(session, SEARCH_URL,
                             data={"drug": drug, "search": "Search",
                                   "public_title": drug})
        except Exception as exc:
            print(f"  [CTRI] search POST failed: {exc}", file=sys.stderr)
            return links

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        m = DETAIL_RE.search(a["href"]) or DETAIL_ALT.search(a["href"])
        if not m:
            continue
        href = urljoin(BASE + "/Clinicaltrials/", a["href"])
        if href in seen:
            continue
        seen.add(href)
        links.append(href)
        if max_records and len(links) >= max_records:
            break
    return links


def _map_detail(html: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    pairs = extract_label_value_pairs(soup)
    text = clean(soup.get_text(" ", strip=True))

    ctri_no = find_field(pairs, "CTRI Number", "CTRI No")
    if not ctri_no:
        m = re.search(r"(CTRI/\d{4}/\d{2}/\d+)", text)
        ctri_no = m.group(1) if m else ""

    row = blank_row(SRC_CTRI)
    row.update({
        "trial_id": ctri_no,
        "secondary_ids": join([find_field(pairs, "Secondary IDs if Any",
                                          "Secondary ID"),
                               find_field(pairs, "Protocol Number"),
                               find_field(pairs, "UTN")]),
        "title": first_nonempty(find_field(pairs, "Scientific Title of Study",
                                           "Scientific Title"),
                                find_field(pairs, "Public Title of Study",
                                           "Public Title")),
        "public_title": find_field(pairs, "Public Title of Study", "Public Title",
                                   "Brief Summary"),
        "status": find_field(pairs, "Recruitment Status of Trial",
                             "Recruitment Status", "Trial Status"),
        "phase": find_field(pairs, "Phase of Trial", "Phase"),
        "study_type": find_field(pairs, "Type of Study", "Study Type",
                                 "Type of Trial"),
        "study_design": join([find_field(pairs, "Study Design"),
                              find_field(pairs, "Method of generating random sequence"),
                              find_field(pairs, "Method of Concealment"),
                              find_field(pairs, "Blinding/Masking")]),
        "conditions": join([find_field(pairs, "Health Condition", "Condition"),
                            find_field(pairs, "Health Type")]),
        "interventions": join([find_field(pairs, "Intervention", "Intervention/Comparator Agent"),
                               find_field(pairs, "Comparator Agent")]),
        "drug_names": join([find_field(pairs, "Intervention"),
                            find_field(pairs, "Comparator Agent")]),
        "sponsor": find_field(pairs, "Primary Sponsor", "Name of Primary Sponsor"),
        "sponsor_type": find_field(pairs, "Type of Sponsor", "Sponsor Type"),
        "collaborators": join([find_field(pairs, "Secondary Sponsor"),
                               find_field(pairs, "Source of Monetary or Material Support"),
                               find_field(pairs, "Details of Secondary Sponsor")]),
        "countries": first_nonempty(find_field(pairs, "Countries of Recruitment"),
                                    "India"),
        "sites": join([find_field(pairs, "Sites of Study", "Site of Study"),
                       find_field(pairs, "Name of the Site")]),
        "target_enrollment": join([find_field(pairs, "Target Sample Size"),
                                   find_field(pairs, "Total Sample Size"),
                                   find_field(pairs, "Sample Size from India")]),
        "actual_enrollment": find_field(pairs, "Final Enrollment numbers achieved",
                                        "Actual Sample Size"),
        "age_min": find_field(pairs, "Age From", "Minimum Age"),
        "age_max": find_field(pairs, "Age To", "Maximum Age"),
        "gender": find_field(pairs, "Gender", "Sex"),
        "healthy_volunteers": find_field(pairs, "Healthy Volunteers"),
        "inclusion_criteria": find_field(pairs, "Inclusion Criteria"),
        "exclusion_criteria": find_field(pairs, "Exclusion Criteria"),
        "primary_objective": find_field(pairs, "Brief Summary", "Objective"),
        "primary_outcome": find_field(pairs, "Primary Outcome"),
        "secondary_outcome": find_field(pairs, "Secondary Outcome"),
        "start_date": join([find_field(pairs, "Date of First Enrollment (India)",
                                       "Date of First Enrollment"),
                            find_field(pairs, "Date of Study Commencement")]),
        "completion_date": join([find_field(pairs, "Date of Study Completion"),
                                 find_field(pairs, "Estimated Duration of Trial")]),
        "registration_date": find_field(pairs, "Date of Registration",
                                        "Registered on"),
        "last_updated": find_field(pairs, "Last Modified On", "Modified On"),
        "results_available": find_field(pairs, "Publication Details",
                                        "Results Available"),
        "findings": join([find_field(pairs, "Summary of Results", "Brief Results"),
                          find_field(pairs, "Publication Details"),
                          find_field(pairs, "Outcome of the trial")]),
        "contact": join([find_field(pairs, "Contact Person (Scientific Query)"),
                         find_field(pairs, "Contact Person (Public Query)"),
                         find_field(pairs, "Email")]),
        "ethics_approval": join([find_field(pairs, "Ethics Committee"),
                                 find_field(pairs, "Status of Ethics Committee"),
                                 find_field(pairs, "Approval Status"),
                                 find_field(pairs, "Regulatory Clearance Status")]),
        "url": url,
    })

    for k, v in pairs.items():
        col = "ctri." + re.sub(r"\s+", "_", k)[:80]
        if col not in row:
            row[col] = v
    return row


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True) -> List[Dict[str, Any]]:
    session = make_session({"Referer": BASE + "/Clinicaltrials/login.php"})
    links = _search_links(drug, session, max_records)
    print(f"  CTRI search returned {len(links)} trial link(s).", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for i, url in enumerate(links, start=1):
        try:
            rows.append(_map_detail(http_get(session, url), url))
        except Exception as exc:
            print(f"  ! CTRI detail failed for {url}: {exc}", file=sys.stderr)
        if i % 10 == 0:
            print(f"  ...CTRI {i}/{len(links)}", file=sys.stderr)
        time.sleep(DELAY)
    return rows


if __name__ == "__main__":
    sys.exit(run_cli(fetch, SRC_CTRI, "Fetch trials from CTRI (India)."))
