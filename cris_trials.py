#!/usr/bin/env python3
"""
ctri_trials.py – Clinical Trials Registry - India (CTRI).

CTRI's search page uses a CAPTCHA for its main search form, which blocks
automated access. We therefore use two routes:

1. The "pubview.php" keyword search which does NOT require a CAPTCHA and
   returns an HTML result page with trial links.
2. A direct URL search via advsearch.php with a POST body (works for some
   keywords without the CAPTCHA being triggered).

If both fail (e.g. bot detection), results will be 0. CTRI data is also
available via WHO ICTRP which is used as the fallback in fetch_all_trials.py.
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
# pubview.php = public keyword search, no CAPTCHA
SEARCH_PUBVIEW = BASE + "/Clinicaltrials/pubview.php"
# advsearch.php = POST-based search result page
SEARCH_ADVSEARCH = BASE + "/Clinicaltrials/advsearch.php"
# Direct detail URL using internal numeric ID
DETAIL_URL = BASE + "/Clinicaltrials/pmaindet2.php?EncHid={enc}&modid={mid}"
# CTRI number-based detail URL (used when we only have the CTRI number)
DETAIL_CTRI = BASE + "/Clinicaltrials/pmaindet2.php?trialid={ctri}"

DETAIL_RE = re.compile(
    r"pmaindet2\.php\?(?:[^\"'\s]*&)?EncHid=([^&\"'\s]+)(?:&modid=(\d+))?",
    re.I
)
CTRI_ID_RE = re.compile(r"(CTRI/\d{4}/\d{2,3}/\d+)", re.I)
DELAY = 1.5


def _search_links(drug: str, session, max_records: Optional[int]) -> List[str]:
    """Try multiple CTRI search methods and return unique detail-page URLs."""
    links: List[str] = []
    seen: set = set()

    def harvest(html: str):
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = DETAIL_RE.search(href)
            if m:
                full = urljoin(BASE + "/Clinicaltrials/", href)
                if full not in seen:
                    seen.add(full)
                    links.append(full)
                    if max_records and len(links) >= max_records:
                        return
            # Also catch plain CTRI number links
            cm = CTRI_ID_RE.search(href)
            if cm:
                url = DETAIL_CTRI.format(ctri=quote_plus(cm.group(1)))
                if url not in seen:
                    seen.add(url)
                    links.append(url)
                    if max_records and len(links) >= max_records:
                        return

    # Method 1: pubview.php GET with keyword (no CAPTCHA on this endpoint)
    try:
        html = http_get(session, SEARCH_PUBVIEW,
                        params={"searchtype": "2", "searchkey": drug})
        harvest(html)
        if links:
            return links
    except Exception as exc:
        print(f"  [CTRI] pubview search failed: {exc}", file=sys.stderr)

    # Method 2: advsearch.php POST
    try:
        html = http_post(session, SEARCH_ADVSEARCH,
                         data={"searchtype": "2", "searchkey": drug,
                               "search": "Search"})
        harvest(html)
        if links:
            return links
    except Exception as exc:
        print(f"  [CTRI] advsearch POST failed: {exc}", file=sys.stderr)

    # Method 3: try the advancesearchmain page with a GET
    try:
        html = http_get(session,
                        BASE + "/Clinicaltrials/advancesearchmain.php",
                        params={"searchtype": "2", "searchkey": drug})
        harvest(html)
    except Exception as exc:
        print(f"  [CTRI] advancesearchmain failed: {exc}", file=sys.stderr)

    return links


def _map_detail(html: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    pairs = extract_label_value_pairs(soup)
    text = clean(soup.get_text(" ", strip=True))

    ctri_no = find_field(pairs, "CTRI Number", "CTRI No", "Trial ID")
    if not ctri_no:
        m = CTRI_ID_RE.search(text)
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
        "public_title": find_field(pairs, "Public Title of Study",
                                   "Public Title", "Brief Summary"),
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
        "interventions": join([find_field(pairs, "Intervention",
                                          "Intervention/Comparator Agent"),
                               find_field(pairs, "Comparator Agent")]),
        "drug_names": join([find_field(pairs, "Intervention"),
                            find_field(pairs, "Comparator Agent")]),
        "sponsor": find_field(pairs, "Primary Sponsor",
                              "Name of Primary Sponsor"),
        "sponsor_type": find_field(pairs, "Type of Sponsor", "Sponsor Type"),
        "collaborators": join([find_field(pairs, "Secondary Sponsor"),
                               find_field(pairs,
                                          "Source of Monetary or Material Support"),
                               find_field(pairs, "Details of Secondary Sponsor")]),
        "countries": first_nonempty(find_field(pairs, "Countries of Recruitment"),
                                    "India"),
        "sites": join([find_field(pairs, "Sites of Study", "Site of Study"),
                       find_field(pairs, "Name of the Site")]),
        "target_enrollment": join([find_field(pairs, "Target Sample Size"),
                                   find_field(pairs, "Total Sample Size"),
                                   find_field(pairs, "Sample Size from India")]),
        "actual_enrollment": find_field(pairs,
                                        "Final Enrollment numbers achieved",
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
        "start_date": join([
            find_field(pairs, "Date of First Enrollment (India)",
                       "Date of First Enrollment"),
            find_field(pairs, "Date of Study Commencement")]),
        "completion_date": join([
            find_field(pairs, "Date of Study Completion"),
            find_field(pairs, "Estimated Duration of Trial")]),
        "registration_date": find_field(pairs, "Date of Registration",
                                        "Registered on"),
        "last_updated": find_field(pairs, "Last Modified On", "Modified On"),
        "results_available": find_field(pairs, "Publication Details",
                                        "Results Available"),
        "findings": join([find_field(pairs, "Summary of Results",
                                     "Brief Results"),
                          find_field(pairs, "Publication Details"),
                          find_field(pairs, "Outcome of the trial")]),
        "contact": join([
            find_field(pairs, "Contact Person (Scientific Query)"),
            find_field(pairs, "Contact Person (Public Query)"),
            find_field(pairs, "Email")]),
        "ethics_approval": join([
            find_field(pairs, "Ethics Committee"),
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