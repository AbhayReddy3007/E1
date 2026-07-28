#!/usr/bin/env python3
"""
euct_trials.py – adapter that plugs the existing EU scripts into this package.

Reuses ctis_drug_trials.py and eudract_drug_trials.py (CTIS + EudraCT), then
maps their rows onto the shared unified schema with source = "EU Clinical
Trials". Copy those two files next to this one, or put them on PYTHONPATH.
"""

from __future__ import annotations
import re
import sys
import time
from typing import Any, Dict, List, Optional

from registry_common import (
    SRC_EUCT, blank_row, clean, first_nonempty, join, make_session, run_cli,
)

try:
    import ctis_drug_trials as ctis
except Exception:                                     # noqa: BLE001
    ctis = None
try:
    import eudract_drug_trials as eudract
except Exception:                                     # noqa: BLE001
    eudract = None


def _from_ctis(flat: Dict[str, Any]) -> Dict[str, Any]:
    row = blank_row(SRC_EUCT)
    row.update({
        "trial_id": flat.get("ct_number", ""),
        "secondary_ids": flat.get("eudract_number", ""),
        "title": flat.get("title", ""),
        "public_title": flat.get("short_title", ""),
        "status": flat.get("status", ""),
        "phase": flat.get("phase", ""),
        "study_type": "Interventional",
        "study_design": flat.get("trial_design", ""),
        "conditions": flat.get("conditions", ""),
        "interventions": first_nonempty(flat.get("product_names"),
                                        flat.get("product")),
        "drug_names": first_nonempty(flat.get("active_substances"),
                                     flat.get("product_names"),
                                     flat.get("product")),
        "sponsor": first_nonempty(flat.get("sponsors_full"), flat.get("sponsor")),
        "sponsor_type": flat.get("sponsor_type", ""),
        "collaborators": "",
        "countries": flat.get("countries", ""),
        "sites": flat.get("msc", ""),
        "target_enrollment": flat.get("planned_subjects_worldwide", ""),
        "actual_enrollment": flat.get("enrolled", ""),
        "age_min": flat.get("age_group", ""),
        "gender": flat.get("gender", ""),
        "inclusion_criteria": flat.get("inclusion_criteria", ""),
        "exclusion_criteria": flat.get("exclusion_criteria", ""),
        "primary_objective": flat.get("main_objective", ""),
        "primary_outcome": flat.get("primary_endpoint", ""),
        "secondary_outcome": flat.get("secondary_endpoint", ""),
        "start_date": flat.get("start_date", ""),
        "completion_date": flat.get("end_date", ""),
        "registration_date": flat.get("decision_date", ""),
        "last_updated": flat.get("last_updated", ""),
        "results_available": flat.get("results_first_received", ""),
        "findings": flat.get("findings", ""),
        "url": flat.get("url", ""),
    })
    for k, v in flat.items():
        col = "euct.ctis." + k
        if col not in row:
            row[col] = v
    return row


def _from_eudract(flat: Dict[str, Any]) -> Dict[str, Any]:
    row = blank_row(SRC_EUCT)
    row.update({
        "trial_id": flat.get("eudract_number", ""),
        "secondary_ids": flat.get("sponsor_protocol_number", ""),
        "title": first_nonempty(flat.get("full_title"), flat.get("title")),
        "public_title": flat.get("lay_title", ""),
        "status": first_nonempty(flat.get("end_of_trial_status"),
                                 flat.get("status")),
        "phase": flat.get("phase", ""),
        "study_type": "Interventional",
        "study_design": join([flat.get("randomised"), flat.get("double_blind"),
                              flat.get("parallel_group"), flat.get("crossover")]),
        "conditions": first_nonempty(flat.get("medical_condition_full"),
                                     flat.get("medical_condition")),
        "interventions": join([flat.get("product_name"), flat.get("product_code")]),
        "drug_names": first_nonempty(flat.get("inn"), flat.get("product_name")),
        "sponsor": first_nonempty(flat.get("sponsor_name"), flat.get("sponsor")),
        "sponsor_type": flat.get("sponsor_status", ""),
        "countries": flat.get("countries", ""),
        "target_enrollment": flat.get("subjects_worldwide", ""),
        "actual_enrollment": flat.get("results_subjects_worldwide", ""),
        "age_min": flat.get("population_age", ""),
        "gender": flat.get("gender", ""),
        "healthy_volunteers": flat.get("healthy_volunteers", ""),
        "inclusion_criteria": flat.get("inclusion_criteria", ""),
        "exclusion_criteria": flat.get("exclusion_criteria", ""),
        "primary_objective": flat.get("main_objective", ""),
        "primary_outcome": flat.get("primary_endpoint", ""),
        "secondary_outcome": flat.get("secondary_endpoint", ""),
        "start_date": flat.get("start_date", ""),
        "completion_date": flat.get("global_end_date", ""),
        "registration_date": flat.get("ca_decision_date", ""),
        "results_available": flat.get("results_available", ""),
        "findings": flat.get("findings", ""),
        "ethics_approval": join([flat.get("ethics_opinion"),
                                 flat.get("ethics_opinion_date")]),
        "url": flat.get("url", ""),
    })
    for k, v in flat.items():
        col = "euct.eudract." + k
        if col not in row:
            row[col] = v
    return row


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if ctis is not None:
        try:
            session = make_session(ctis.HEADERS)
            for summary in ctis.search_trials(drug, session, page_size=50,
                                              max_records=max_records):
                d = None
                if details:
                    try:
                        d = ctis.get_trial_details(summary.get("ctNumber", ""), session)
                    except Exception:
                        d = None
                rows.append(_from_ctis(ctis.flatten(summary, d)))
        except Exception as exc:
            print(f"  [EU/CTIS] {exc}", file=sys.stderr)
    else:
        print("  [EU] ctis_drug_trials.py not importable – skipping CTIS.",
              file=sys.stderr)

    if eudract is not None:
        try:
            session = make_session(eudract.HEADERS)
            for flat in eudract.search_trials(drug, session,
                                              max_records=max_records):
                if details:
                    country = (flat.get("countries", "").split(";")[0] or "GB").strip()
                    try:
                        flat.update(eudract.get_trial_details(
                            flat["eudract_number"], country, session))
                    except Exception:
                        pass
                    if flat.get("results_available") == "Yes":
                        try:
                            flat.update(eudract.get_trial_results(
                                flat["eudract_number"], session))
                        except Exception:
                            pass
                    time.sleep(eudract.POLITE_DELAY)
                rows.append(_from_eudract(flat))
        except Exception as exc:
            print(f"  [EU/EudraCT] {exc}", file=sys.stderr)
    else:
        print("  [EU] eudract_drug_trials.py not importable – skipping EudraCT.",
              file=sys.stderr)

    # merge CTIS + EudraCT rows describing the same trial
    by_eudract: Dict[str, Dict[str, Any]] = {}
    merged: List[Dict[str, Any]] = []
    for row in rows:
        key = ""
        if row.get("trial_id", "").count("-") == 2 and len(row.get("trial_id", "")) == 14:
            key = row["trial_id"]
        elif re.match(r"^\d{4}-\d{6}-\d{2}$", row.get("secondary_ids", "") or ""):
            key = row["secondary_ids"]
        if key and key in by_eudract:
            target = by_eudract[key]
            for k, v in row.items():
                if not target.get(k) and v:
                    target[k] = v
            continue
        if key:
            by_eudract[key] = row
        merged.append(row)
    return merged


if __name__ == "__main__":
    sys.exit(run_cli(fetch, SRC_EUCT, "Fetch EU trials (CTIS + EudraCT)."))
