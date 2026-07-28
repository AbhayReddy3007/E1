#!/usr/bin/env python3
"""
jrct_trials.py – Japan Registry of Clinical Trials (jRCT).

URL migration (March 25, 2025):
  OLD: https://jrct.niph.go.jp/   (dead)
  NEW: https://jrct.mhlw.go.jp/

JPRN cross-search portal migration (March 25, 2025):
  OLD: https://rctportal.niph.go.jp/  (dead)
  NEW: https://rctportal.mhlw.go.jp/

The public search page is at https://jrct.mhlw.go.jp/search (Japanese) or
the English top page at https://jrct.mhlw.go.jp/en-top. The search form at
/search accepts a POST with field 'freeword'. The detail pages at
/en-latest-detail/<jRCT-id> are in English.

>>> TERMS-OF-USE NOTICE <<<
jRCT's site asks users NOT to perform bulk automated downloading beyond
personal use. This module sleeps DELAY seconds between requests and caps
at DEFAULT_CAP records unless overridden. Set JRCT_DELAY env var to
slow it further.
"""

from __future__ import annotations
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup

from registry_common import (
    SRC_JRCT, blank_row, clean, extract_label_value_pairs, find_field,
    first_nonempty, http_get, http_post, join, make_session, run_cli,
)

BASE = "https://jrct.mhlw.go.jp"
SEARCH_URL = BASE + "/search"       # Japanese search (POST freeword=)
DETAIL_EN = BASE + "/en-latest-detail/{jid}"
DETAIL_JA = BASE + "/latest-detail/{jid}"

DEFAULT_CAP = 100
DELAY = float(os.environ.get("JRCT_DELAY", "2.0"))
JRCT_ID_RE = re.compile(r"(jRCT[a-zA-Z0-9]*\d{8,})", re.I)


def _search_ids(drug: str, session, max_records: Optional[int]) -> List[str]:
    cap = max_records or DEFAULT_CAP
    ids: List[str] = []
    seen: set = set()

    def harvest(html: str):
        for m in JRCT_ID_RE.finditer(html):
            jid = m.group(1)
            if jid not in seen:
                seen.add(jid)
                ids.append(jid)
                if len(ids) >= cap:
                    return

    # Method 1: POST to /search with freeword
    try:
        html = http_post(session, SEARCH_URL,
                         data={"freeword": drug})
        harvest(html)
        if ids:
            return ids
    except Exception as exc:
        print(f"  [jRCT] POST search failed: {exc}", file=sys.stderr)

    # Method 2: GET /search
    try:
        html = http_get(session, SEARCH_URL,
                        params={"freeword": drug})
        harvest(html)
        if ids:
            return ids
    except Exception as exc:
        print(f"  [jRCT] GET search failed: {exc}", file=sys.stderr)

    # Method 3: GET /en-top page and extract any IDs present
    try:
        html = http_get(session, BASE + "/en-top",
                        params={"freeword": drug})
        harvest(html)
    except Exception as exc:
        print(f"  [jRCT] en-top search failed: {exc}", file=sys.stderr)

    return ids


def _map_detail(html: str, jid: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    pairs = extract_label_value_pairs(soup)

    row = blank_row(SRC_JRCT)
    row.update({
        "trial_id": first_nonempty(find_field(pairs, "Trial ID", "jRCT ID",
                                              "研究計画書ID"), jid),
        "secondary_ids": join([find_field(pairs, "Secondary ID", "他の登録機関",
                                          "Other study ID"),
                               find_field(pairs, "Japic", "NCT")]),
        "title": first_nonempty(
            find_field(pairs, "Title of the study", "Scientific Title",
                       "Official title", "研究の名称"),
            find_field(pairs, "Title", "Public title")),
        "public_title": find_field(pairs, "Public title",
                                   "Title for public disclosure"),
        "status": find_field(pairs, "Recruitment status", "Study status",
                             "募集状況", "進捗状況"),
        "phase": find_field(pairs, "Phase", "Development phase",
                            "試験のフェーズ"),
        "study_type": find_field(pairs, "Study type", "研究の種類"),
        "study_design": join([find_field(pairs, "Study design", "試験デザイン"),
                              find_field(pairs, "Allocation"),
                              find_field(pairs, "Blinding"),
                              find_field(pairs, "Control")]),
        "conditions": join([find_field(pairs, "Health condition", "Condition",
                                       "対象疾患名"),
                            find_field(pairs, "Target disease")]),
        "interventions": join([find_field(pairs, "Interventions/Control",
                                          "Intervention", "介入の内容"),
                               find_field(pairs, "Investigational drug")]),
        "drug_names": join([find_field(pairs, "Investigational drug",
                                       "Name of drug", "医薬品等の名称"),
                            find_field(pairs, "Interventions/Control",
                                       "Intervention")]),
        "sponsor": join([find_field(pairs, "Name of lead principal investigator",
                                    "Sponsor", "研究責任医師"),
                         find_field(pairs, "Organization", "実施医療機関")]),
        "sponsor_type": find_field(pairs, "Source of funding", "資金提供者"),
        "collaborators": find_field(pairs, "Collaborator"),
        "countries": first_nonempty(find_field(pairs, "Region", "実施国"),
                                    "Japan"),
        "sites": find_field(pairs, "Institution", "Name of institution",
                            "実施医療機関"),
        "target_enrollment": find_field(pairs, "Target sample size",
                                        "Planned sample size", "予定症例数"),
        "actual_enrollment": find_field(pairs, "Actual sample size"),
        "age_min": find_field(pairs, "Age minimum", "Age Minimum", "年齢下限"),
        "age_max": find_field(pairs, "Age maximum", "Age Maximum", "年齢上限"),
        "gender": find_field(pairs, "Gender", "Sex", "性別"),
        "healthy_volunteers": find_field(pairs, "Healthy volunteers"),
        "inclusion_criteria": find_field(pairs, "Inclusion criteria", "選択基準"),
        "exclusion_criteria": find_field(pairs, "Exclusion criteria", "除外基準"),
        "primary_objective": find_field(pairs, "Objectives", "Purpose",
                                        "Brief summary", "研究の目的"),
        "primary_outcome": find_field(pairs, "Primary outcome",
                                      "Primary endpoint", "主要評価項目"),
        "secondary_outcome": find_field(pairs, "Secondary outcome",
                                        "Secondary endpoint", "副次評価項目"),
        "start_date": join([find_field(pairs, "Date of first enrollment",
                                       "Study start date", "試験開始日"),
                            find_field(pairs, "Date of protocol enactment")]),
        "completion_date": find_field(pairs, "Study completion date",
                                      "Date of completion", "試験終了日"),
        "registration_date": find_field(pairs, "Date of registration",
                                        "Registered date", "公表日"),
        "last_updated": find_field(pairs, "Last modified on", "Date of update",
                                   "最終更新日"),
        "results_available": find_field(pairs, "Result posted", "Results",
                                        "結果の公表"),
        "findings": join([find_field(pairs, "Summary of results",
                                     "Result summary", "結果の概要"),
                          find_field(pairs, "Primary outcome result"),
                          find_field(pairs, "Conclusion"),
                          find_field(pairs, "Publication", "公表論文")]),
        "contact": join([find_field(pairs, "Contact for public queries",
                                    "Contact person", "問合せ先"),
                         find_field(pairs, "E-mail", "Email")]),
        "ethics_approval": join([
            find_field(pairs, "Name of certified review board",
                       "IRB", "認定臨床研究審査委員会"),
            find_field(pairs, "Approval date", "Date of approval")]),
        "url": url,
    })

    for k, v in pairs.items():
        col = "jrct." + re.sub(r"\s+", "_", k)[:80]
        if col not in row:
            row[col] = v
    return row


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True) -> List[Dict[str, Any]]:
    print("  [jRCT] NOTE: jRCT asks users not to bulk-download automatically. "
          "Keep runs small and personal.", file=sys.stderr)
    session = make_session()
    ids = _search_ids(drug, session, max_records)
    print(f"  jRCT search returned {len(ids)} trial id(s).", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for i, jid in enumerate(ids, start=1):
        url = DETAIL_EN.format(jid=jid)
        try:
            html = http_get(session, url)
        except Exception:
            url = DETAIL_JA.format(jid=jid)
            try:
                html = http_get(session, url)
            except Exception as exc:
                print(f"  ! jRCT detail failed for {jid}: {exc}",
                      file=sys.stderr)
                continue
        rows.append(_map_detail(html, jid, url))
        if i % 10 == 0:
            print(f"  ...jRCT {i}/{len(ids)}", file=sys.stderr)
        time.sleep(DELAY)
    return rows


if __name__ == "__main__":
    sys.exit(run_cli(fetch, SRC_JRCT, "Fetch trials from jRCT (Japan)."))