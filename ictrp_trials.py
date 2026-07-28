#!/usr/bin/env python3
"""
ictrp_trials.py – WHO ICTRP fallback provider.

ICTRP (https://trialsearch.who.int) aggregates records contributed by all the
national registries. It is used here ONLY as a fallback for registries whose
own site is unreachable, blocked, or rate-limiting.

IMPORTANT: records retrieved via ICTRP are re-labelled with the *originating*
registry name, so the `source` column never says "ICTRP" — it says ChiCTR,
CTRI, JRCT, ANZCTR, CRIS, ReBEC, ClinicalTrials.gov or EU Clinical Trials.
Records from any other registry are discarded.
"""

from __future__ import annotations
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from registry_common import (
    ALLOWED_SOURCES, SRC_ANZCTR, SRC_CHICTR, SRC_CRIS, SRC_CTGOV, SRC_CTRI,
    SRC_EUCT, SRC_JRCT, SRC_REBEC, UNIFIED_COLUMNS, blank_row, clean,
    first_nonempty, flatten_xml, http_get, http_post, join, make_session,
    write_excel,
)

SEARCH_URL = "https://trialsearch.who.int/api/Trial/Search"
EXPORT_URL = "https://trialsearch.who.int/Trial2.aspx"
PORTAL_URL = "https://trialsearch.who.int/Trial2.aspx?TrialID={tid}"

# ── Map ICTRP register labels / ID prefixes onto our canonical source names ──
REGISTER_NAME_MAP = {
    "chictr": SRC_CHICTR,
    "chinese clinical trial register": SRC_CHICTR,
    "ctri": SRC_CTRI,
    "clinical trials registry - india": SRC_CTRI,
    "clinical trials registry-india": SRC_CTRI,
    "jrct": SRC_JRCT,
    "japan registry of clinical trials": SRC_JRCT,
    "jprn": SRC_JRCT,
    "anzctr": SRC_ANZCTR,
    "australian new zealand clinical trials registry": SRC_ANZCTR,
    "cris": SRC_CRIS,
    "clinical research information service": SRC_CRIS,
    "rebec": SRC_REBEC,
    "brazilian clinical trials registry": SRC_REBEC,
    "registro brasileiro de ensaios clinicos": SRC_REBEC,
    "clinicaltrials.gov": SRC_CTGOV,
    "nct": SRC_CTGOV,
    "euctr": SRC_EUCT,
    "eu clinical trials register": SRC_EUCT,
    "ctis": SRC_EUCT,
}

ID_PREFIX_MAP = [
    (re.compile(r"^ChiCTR", re.I), SRC_CHICTR),
    (re.compile(r"^CTRI/", re.I), SRC_CTRI),
    (re.compile(r"^jRCT", re.I), SRC_JRCT),
    (re.compile(r"^JPRN-jRCT", re.I), SRC_JRCT),
    (re.compile(r"^ACTRN", re.I), SRC_ANZCTR),
    (re.compile(r"^KCT", re.I), SRC_CRIS),
    (re.compile(r"^RBR-", re.I), SRC_REBEC),
    (re.compile(r"^NCT\d", re.I), SRC_CTGOV),
    (re.compile(r"^\d{4}-\d{6}-\d{2}", re.I), SRC_EUCT),
    (re.compile(r"^EUCTR", re.I), SRC_EUCT),
]


def canonical_source(register_label: str, trial_id: str) -> Optional[str]:
    """Return one of ALLOWED_SOURCES, or None if the record should be dropped."""
    label = clean(register_label).lower()
    for needle, src in REGISTER_NAME_MAP.items():
        if needle in label:
            return src
    tid = clean(trial_id)
    for pattern, src in ID_PREFIX_MAP:
        if pattern.search(tid):
            return src
    return None


def _text(el, *tags) -> str:
    for tag in tags:
        found = el.find(tag)
        if found is not None and clean(found.text):
            return clean(found.text)
    return ""


def _map_trial(el) -> Optional[Dict[str, Any]]:
    trial_id = _text(el, "TrialID", "trialid", "main_id")
    register = _text(el, "Source_Register", "register", "Register")
    source = canonical_source(register, trial_id)
    if source is None:
        return None    # not one of the requested registries -> drop

    row = blank_row(source)
    row.update({
        "trial_id": trial_id,
        "secondary_ids": _text(el, "Secondary_ID", "SecondaryIDs"),
        "title": first_nonempty(_text(el, "Scientific_title"),
                                _text(el, "Public_title")),
        "public_title": _text(el, "Public_title"),
        "status": _text(el, "Recruitment_Status", "Recruitment_status"),
        "phase": _text(el, "Phase"),
        "study_type": _text(el, "Study_type"),
        "study_design": _text(el, "Study_design"),
        "conditions": _text(el, "Condition"),
        "interventions": _text(el, "Intervention"),
        "drug_names": _text(el, "Intervention"),
        "sponsor": _text(el, "Primary_sponsor"),
        "collaborators": _text(el, "Secondary_Sponsor", "Source_Support"),
        "countries": _text(el, "Countries", "Recruitment_Country"),
        "target_enrollment": _text(el, "Target_size", "Target_sample_size"),
        "age_min": _text(el, "Inclusion_agemin", "Agemin"),
        "age_max": _text(el, "Inclusion_agemax", "Agemax"),
        "gender": _text(el, "Inclusion_gender", "Gender"),
        "inclusion_criteria": _text(el, "Inclusion_Criteria", "Inclusion_criteria"),
        "exclusion_criteria": _text(el, "Exclusion_Criteria", "Exclusion_criteria"),
        "primary_outcome": _text(el, "Primary_outcome"),
        "secondary_outcome": _text(el, "Secondary_outcome",
                                   "Secondary_outcomes"),
        "start_date": _text(el, "Date_enrollement", "Date_enrolment",
                            "Study_start_date"),
        "registration_date": _text(el, "Date_registration",
                                   "Date_registration3"),
        "last_updated": _text(el, "Last_Refreshed_on", "Export_date"),
        "results_available": _text(el, "results_yes_no", "Results_available"),
        "findings": first_nonempty(
            _text(el, "results_summary"),
            _text(el, "Results_summary"),
            _text(el, "results_date_completed"),
        ),
        "contact": join([_text(el, "Contact_Firstname") + " " +
                         _text(el, "Contact_Lastname"),
                         _text(el, "Contact_Email")]),
        "ethics_approval": _text(el, "Ethics_review_status",
                                 "Ethics_Review_Status"),
        "url": first_nonempty(_text(el, "web_address", "Web_address"),
                              PORTAL_URL.format(tid=trial_id)),
    })

    for k, v in flatten_xml(el).items():
        col = "ictrp." + k
        if col not in row:
            row[col] = v
    return row


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True,
          only_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Query ICTRP for `drug`. `only_sources` restricts output to the given
    canonical source names (e.g. [SRC_CHICTR]) so this can back-fill a single
    registry without duplicating ones already fetched directly.
    """
    session = make_session({"Accept": "application/xml, text/xml, text/csv, */*"})
    rows: List[Dict[str, Any]] = []

    # Try XML export, then CSV export as fallback
    xml_text = None
    csv_text = None
    attempts = [
        ("xml", EXPORT_URL, {"SearchTermStat": drug, "ExportMethod": "XML",
                             "SearchTermFlag": "1"}),
        ("xml", "https://trialsearch.who.int/AdvSearch.aspx",
         {"SearchTermStat": drug, "ExportMethod": "XML",
          "SearchTermFlag": "1", "recruitmentStat": "ALL"}),
        ("csv", EXPORT_URL, {"SearchTermStat": drug, "ExportMethod": "CSV",
                             "SearchTermFlag": "1"}),
    ]
    for fmt, url, params in attempts:
        try:
            resp = http_get(session, url, params=params)
            if not resp or len(resp.strip()) < 50:
                continue
            if fmt == "xml" and resp.strip().startswith("<"):
                xml_text = resp
                break
            elif fmt == "csv" and ("TrialID" in resp or "Source_Register" in resp):
                csv_text = resp
                break
        except Exception:
            continue

    if xml_text is None and csv_text is None:
        print("  [ICTRP] all search endpoints failed or returned HTML.",
              file=sys.stderr)
        return rows

    # CSV path
    if csv_text and not xml_text:
        import csv as csv_mod, io
        reader = csv_mod.DictReader(io.StringIO(csv_text))
        for rec in reader:
            tid = clean(rec.get("TrialID") or rec.get("trial_id", ""))
            reg = clean(rec.get("Source_Register") or rec.get("register", ""))
            source = canonical_source(reg, tid)
            if source is None:
                continue
            if only_sources and source not in only_sources:
                continue
            row = blank_row(source)
            row.update({
                "trial_id": tid,
                "title": first_nonempty(clean(rec.get("Scientific_title", "")),
                                        clean(rec.get("Public_title", ""))),
                "public_title": clean(rec.get("Public_title", "")),
                "status": clean(rec.get("Recruitment_Status", "")),
                "phase": clean(rec.get("Phase", "")),
                "study_type": clean(rec.get("Study_type", "")),
                "conditions": clean(rec.get("Condition", "")),
                "interventions": clean(rec.get("Intervention", "")),
                "drug_names": clean(rec.get("Intervention", "")),
                "sponsor": clean(rec.get("Primary_sponsor", "")),
                "countries": clean(rec.get("Countries", "")),
                "target_enrollment": clean(rec.get("Target_size", "")),
                "start_date": clean(rec.get("Date_enrollement", "")),
                "registration_date": clean(rec.get("Date_registration", "")),
                "last_updated": clean(rec.get("Last_Refreshed_on", "")),
                "results_available": clean(rec.get("results_yes_no", "")),
                "url": first_nonempty(clean(rec.get("web_address", "")),
                                      PORTAL_URL.format(tid=tid)),
            })
            for k, v in rec.items():
                col = "ictrp." + clean(k).replace(" ", "_")[:80]
                if col not in row and clean(v):
                    row[col] = clean(v)
            rows.append(row)
            if max_records and len(rows) >= max_records:
                break
        return rows

    # XML path
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        print(f"  [ICTRP] response was not parseable XML ({exc}); "
              f"the portal likely returned an HTML page instead.", file=sys.stderr)
        return rows

    for el in root.iter():
        if el.tag.split("}")[-1].lower() != "trial":
            continue
        row = _map_trial(el)
        if row is None:
            continue
        if only_sources and row["source"] not in only_sources:
            continue
        rows.append(row)
        if max_records and len(rows) >= max_records:
            break

    return rows


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="WHO ICTRP fallback fetcher (records re-labelled to their "
                    "originating registry).")
    ap.add_argument("drug")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"restrict to these sources: {ALLOWED_SOURCES}")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    slug = re.sub(r"[^a-z0-9]+", "_", args.drug.lower()).strip("_")
    out = args.out or f"{slug}_ictrp_fallback.xlsx"
    rows = fetch(args.drug, max_records=args.max_records, only_sources=args.only)
    print(f"[ICTRP] {len(rows)} trial(s) mapped to allowed sources", file=sys.stderr)
    write_excel(rows, out, UNIFIED_COLUMNS, sheet_name="ICTRP")
    return 0


if __name__ == "__main__":
    sys.exit(main())