#!/usr/bin/env python3
"""
chictr_trials.py – Chinese Clinical Trial Registry (ChiCTR).

    Search : https://www.chictr.org.cn/searchproj.html
    Trial  : https://www.chictr.org.cn/showproj.html?proj=<id>

ChiCTR has no public API and is the hardest of these registries to reach
programmatically: it sits behind bot protection, frequently rate-limits, and
serves bilingual pages whose markup changes. This module therefore:

  * tries the site's own search endpoints (HTML + the searchlist JSON used by
    the page's own JavaScript),
  * harvests every label/value pair from the bilingual detail table,
  * returns an empty list rather than raising if the site blocks the request,
    so the orchestrator can fall back to WHO ICTRP for ChiCTR records.
"""

from __future__ import annotations
import re
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup

from registry_common import (
    SRC_CHICTR, blank_row, clean, extract_label_value_pairs, find_field,
    first_nonempty, http_get, http_post, join, make_session, run_cli,
)

BASE = "https://www.chictr.org.cn"
# ChiCTR restructured circa 2024-2025; try multiple endpoints
SEARCH_HTML = BASE + "/enindex.html"
SEARCH_LIST = BASE + "/searchprojen.html"
SEARCH_API = BASE + "/searchproj.html"
SEARCH_LEGACY = BASE + "/searchprojEN.html"
DETAIL_EN = BASE + "/showprojEN.html?proj={proj}"
DETAIL_CN = BASE + "/showproj.html?proj={proj}"
PROJ_RE = re.compile(r"showproj(?:EN)?\.html\?proj=(\d+)", re.I)
CHICTR_ID_RE = re.compile(r"(ChiCTR[-A-Za-z0-9]*\d+)", re.I)
DELAY = 2.0


def _search_projs(drug: str, session, max_records: Optional[int]) -> List[str]:
    projs: List[str] = []
    seen = set()

    attempts = [
        ("GET", SEARCH_LIST, {"title": drug, "intervention": drug, "page": "1"}),
        ("GET", SEARCH_HTML, {"title": drug, "intervention": drug, "page": "1"}),
        ("GET", SEARCH_LEGACY, {"title": drug, "officialname": "",
                                "intervention": drug, "page": "1"}),
        ("GET", SEARCH_API, {"title": drug, "intervention": drug, "page": "1"}),
        ("POST", SEARCH_API, {"title": drug, "intervention": drug, "page": "1"}),
    ]

    for method, url, params in attempts:
        try:
            html = (http_get(session, url, params=params) if method == "GET"
                    else http_post(session, url, data=params))
        except Exception as exc:
            print(f"  [ChiCTR] {url} -> {exc}", file=sys.stderr)
            continue

        for m in PROJ_RE.finditer(html):
            proj = m.group(1)
            if proj in seen:
                continue
            seen.add(proj)
            projs.append(proj)
            if max_records and len(projs) >= max_records:
                return projs
        if projs:
            break
        time.sleep(DELAY)

    return projs


def _map_detail(html: str, proj: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    pairs = extract_label_value_pairs(soup)
    text = clean(soup.get_text(" ", strip=True))

    reg_no = find_field(pairs, "Registration number", "Registration Number",
                        "注册号")
    if not reg_no:
        m = CHICTR_ID_RE.search(text)
        reg_no = m.group(1) if m else ""

    row = blank_row(SRC_CHICTR)
    row.update({
        "trial_id": reg_no,
        "secondary_ids": join([find_field(pairs, "Secondary registration number",
                                          "Secondary ID"),
                               find_field(pairs, "Sub-project")]),
        "title": first_nonempty(
            find_field(pairs, "Scientific title", "Public title",
                       "研究课题的正式科学名称"),
            find_field(pairs, "Study title")),
        "public_title": find_field(pairs, "Public title", "研究课题的公开名称"),
        "status": find_field(pairs, "Recruiting status", "Recruitment status",
                             "征募研究对象状态"),
        "phase": find_field(pairs, "Study phase", "研究阶段", "Phase"),
        "study_type": find_field(pairs, "Study type", "研究类型"),
        "study_design": join([find_field(pairs, "Study design", "研究设计"),
                              find_field(pairs, "Randomization Procedure"),
                              find_field(pairs, "Blinding", "盲法"),
                              find_field(pairs, "Allocation")]),
        "conditions": join([find_field(pairs, "Target disease", "目标疾病"),
                            find_field(pairs, "Target disease code")]),
        "interventions": join([find_field(pairs, "Interventions", "干预措施"),
                               find_field(pairs, "Intervention")]),
        "drug_names": join([find_field(pairs, "Interventions", "干预措施"),
                            find_field(pairs, "Drug")]),
        "sponsor": join([find_field(pairs, "Primary sponsor", "申请注册联系人"),
                         find_field(pairs, "Applicant", "Sponsor")]),
        "sponsor_type": find_field(pairs, "Source(s) of funding",
                                   "Sponsor type", "经费或物资来源"),
        "collaborators": join([find_field(pairs, "Secondary sponsor"),
                               find_field(pairs, "Source(s) of funding")]),
        "countries": first_nonempty(find_field(pairs, "Countries of recruitment",
                                               "Country"), "China"),
        "sites": join([find_field(pairs, "Countries of recruitment and research settings"),
                       find_field(pairs, "Study execute time"),
                       find_field(pairs, "Institution")]),
        "target_enrollment": join([find_field(pairs, "Target size of sample",
                                              "Sample size", "研究实施时间"),
                                   find_field(pairs, "Anticipated total sample size")]),
        "actual_enrollment": find_field(pairs, "Actual sample size",
                                        "Recruited sample size"),
        "age_min": find_field(pairs, "Age minimum", "Applicant age", "最小年龄"),
        "age_max": find_field(pairs, "Age maximum", "最大年龄"),
        "gender": find_field(pairs, "Gender", "性别"),
        "healthy_volunteers": find_field(pairs, "Healthy volunteers"),
        "inclusion_criteria": find_field(pairs, "Inclusion criteria", "纳入标准"),
        "exclusion_criteria": find_field(pairs, "Exclusion criteria", "排除标准"),
        "primary_objective": find_field(pairs, "Objectives of Study",
                                        "Study objective", "研究目的"),
        "primary_outcome": find_field(pairs, "Primary outcome", "主要指标"),
        "secondary_outcome": find_field(pairs, "Secondary outcome", "次要指标"),
        "start_date": join([find_field(pairs, "Study execute time",
                                       "Date of first enrollment"),
                            find_field(pairs, "Applicant time")]),
        "completion_date": find_field(pairs, "Date of completion",
                                      "Study completion"),
        "registration_date": find_field(pairs, "Date of Registration",
                                        "Registration time", "注册时间"),
        "last_updated": find_field(pairs, "Date of Last Refreshed on",
                                   "Last updated"),
        "results_available": find_field(pairs, "Results", "Result published"),
        "findings": join([find_field(pairs, "Summary results", "研究结果"),
                          find_field(pairs, "Results and conclusion"),
                          find_field(pairs, "Published results"),
                          find_field(pairs, "Conclusion")]),
        "contact": join([find_field(pairs, "Applicant name", "Contact Name"),
                         find_field(pairs, "Applicant E-mail", "E-mail")]),
        "ethics_approval": join([find_field(pairs, "Name of the ethic committee",
                                            "Ethics committee", "伦理委员会"),
                                 find_field(pairs, "Date of approved by ethic committee"),
                                 find_field(pairs, "Approved No. of ethic committee")]),
        "url": url,
    })

    for k, v in pairs.items():
        col = "chictr." + re.sub(r"\s+", "_", k)[:80]
        if col not in row:
            row[col] = v
    return row


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True) -> List[Dict[str, Any]]:
    session = make_session({
        "Referer": BASE + "/searchprojEN.html",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    try:
        projs = _search_projs(drug, session, max_records)
    except Exception as exc:
        print(f"  [ChiCTR] search failed entirely: {exc}", file=sys.stderr)
        return []

    if not projs:
        print("  [ChiCTR] no results (the site may be blocking automated "
              "requests – the orchestrator can back-fill via WHO ICTRP).",
              file=sys.stderr)
        return []

    print(f"  ChiCTR search returned {len(projs)} project id(s).", file=sys.stderr)
    rows: List[Dict[str, Any]] = []
    for i, proj in enumerate(projs, start=1):
        url = DETAIL_EN.format(proj=proj)
        try:
            html = http_get(session, url)
        except Exception:
            url = DETAIL_CN.format(proj=proj)
            try:
                html = http_get(session, url)
            except Exception as exc:
                print(f"  ! ChiCTR detail failed for {proj}: {exc}", file=sys.stderr)
                continue
        rows.append(_map_detail(html, proj, url))
        if i % 10 == 0:
            print(f"  ...ChiCTR {i}/{len(projs)}", file=sys.stderr)
        time.sleep(DELAY)
    return rows


if __name__ == "__main__":
    sys.exit(run_cli(fetch, SRC_CHICTR, "Fetch trials from ChiCTR (China)."))
