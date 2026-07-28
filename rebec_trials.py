#!/usr/bin/env python3
"""
rebec_trials.py – Registro Brasileiro de Ensaios Clínicos (ReBEC, Brazil).

    Search : https://ensaiosclinicos.gov.br/rg?q=<drug>
    Trial  : https://ensaiosclinicos.gov.br/rg/<RBR-id>
    XML    : https://ensaiosclinicos.gov.br/rg/<RBR-id>/xml/ictrp

The canonical domain is ensaiosclinicos.gov.br (no www prefix).
The /rg search accepts a ?q= GET parameter.
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
    SRC_REBEC, blank_row, clean, extract_label_value_pairs, find_field,
    first_nonempty, flatten_xml, http_get, join, make_session, run_cli,
)

# Canonical domain – no www prefix
BASE = "https://ensaiosclinicos.gov.br"
# /rg no longer has a search; /list shows all trials, /rg/RBR-xxx for detail
SEARCH_URL = BASE + "/list"
TRIAL_URL = BASE + "/rg/{rbr}"
TRIAL_XML = BASE + "/rg/{rbr}/xml/ictrp"

RBR_RE = re.compile(r"(RBR-[0-9a-zA-Z]+)", re.I)
DELAY = 1.0


def _search_ids(drug: str, session, max_records: Optional[int]) -> List[str]:
    ids: List[str] = []
    seen: set = set()
    page = 1
    while True:
        try:
            # Try /list with search param, then /rg as legacy fallback
            html = None
            for url, params in [
                (SEARCH_URL, {"q": drug, "page": str(page)}),
                (BASE + "/rg", {"q": drug, "page": str(page)}),
                (BASE + "/search", {"q": drug, "page": str(page)}),
            ]:
                try:
                    html = http_get(session, url, params=params)
                    if "RBR-" in html:
                        break
                except Exception:
                    continue
            if not html:
                print(f"  [ReBEC] search failed on all URLs", file=sys.stderr)
                break
        except Exception as exc:
            print(f"  [ReBEC] search failed: {exc}", file=sys.stderr)
            break

        soup = BeautifulSoup(html, "html.parser")
        hits = 0
        for a in soup.find_all("a", href=True):
            m = RBR_RE.search(a["href"])
            if not m:
                m = RBR_RE.search(a.get_text())
            if not m:
                continue
            rbr = m.group(1).lower()
            if rbr in seen:
                continue
            seen.add(rbr)
            ids.append(rbr)
            hits += 1
            if max_records and len(ids) >= max_records:
                return ids

        if hits == 0:
            break
        page += 1
        if page > 100:
            break
        time.sleep(DELAY)
    return ids


def _map_xml(xml_text: str, rbr: str) -> Optional[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    flat = flatten_xml(root)

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

    trial_id = first_nonempty(g("trial_id", "TrialID"), rbr.upper())
    row = blank_row(SRC_REBEC)
    row.update({
        "trial_id": trial_id,
        "secondary_ids": join([g("secondary_id", "Secondary_ID"), g("utrn")]),
        "title": first_nonempty(g("scientific_title", "Scientific_title"),
                                g("public_title", "Public_title")),
        "public_title": g("public_title", "Public_title"),
        "status": g("recruitment_status", "Recruitment_Status"),
        "phase": g("phase", "Phase"),
        "study_type": g("study_type", "Study_type"),
        "study_design": g("study_design", "Study_design"),
        "conditions": g("health_condition", "condition", "Condition"),
        "interventions": g("intervention", "Intervention"),
        "drug_names": g("intervention", "Intervention"),
        "sponsor": g("primary_sponsor", "Primary_sponsor"),
        "sponsor_type": g("sponsor_type", "source_support"),
        "collaborators": join([g("secondary_sponsor", "Secondary_Sponsor"),
                               g("source_support", "Source_Support")]),
        "countries": first_nonempty(g("countries", "Countries"), "Brazil"),
        "sites": g("recruitment_site", "site"),
        "target_enrollment": g("target_size", "Target_size"),
        "actual_enrollment": g("actual_size", "results_actual_enrolment"),
        "age_min": g("inclusion_agemin", "agemin"),
        "age_max": g("inclusion_agemax", "agemax"),
        "gender": g("inclusion_gender", "gender"),
        "healthy_volunteers": g("healthy_volunteers"),
        "inclusion_criteria": g("inclusion_criteria", "Inclusion_Criteria"),
        "exclusion_criteria": g("exclusion_criteria", "Exclusion_Criteria"),
        "primary_objective": g("brief_summary", "objective"),
        "primary_outcome": g("primary_outcome", "Primary_outcome"),
        "secondary_outcome": g("secondary_outcome", "Secondary_outcome"),
        "start_date": g("date_enrollement", "date_enrolment",
                        "study_start_date"),
        "completion_date": g("results_date_completed", "date_completed"),
        "registration_date": g("date_registration", "Date_registration"),
        "last_updated": g("last_refreshed_on", "export_date"),
        "results_available": g("results_yes_no", "results_available"),
        "findings": join([g("results_summary"), g("results_url_link"),
                          g("results_date_posted"), g("conclusion")]),
        "contact": join([g("contact_firstname"), g("contact_lastname"),
                         g("contact_email")]),
        "ethics_approval": join([g("ethics_review_status"),
                                 g("ethics_review_approval_date"),
                                 g("ethics_review_contact_name")]),
        "url": TRIAL_URL.format(rbr=rbr),
    })

    for k, v in flat.items():
        col = "rebec." + k
        if col not in row:
            row[col] = v
    return row


def _map_html(html: str, rbr: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    pairs = extract_label_value_pairs(soup)

    row = blank_row(SRC_REBEC)
    row.update({
        "trial_id": first_nonempty(
            find_field(pairs, "Trial identifying number", "UTN",
                       "Registration number"), rbr.upper()),
        "secondary_ids": find_field(pairs, "Secondary identifying numbers",
                                    "Other identifying"),
        "title": first_nonempty(
            find_field(pairs, "Scientific title", "Título científico"),
            find_field(pairs, "Public title", "Título público")),
        "public_title": find_field(pairs, "Public title", "Título público"),
        "status": find_field(pairs, "Recruitment status",
                             "Status de recrutamento"),
        "phase": find_field(pairs, "Study phase", "Fase", "Phase"),
        "study_type": find_field(pairs, "Study type", "Tipo de estudo"),
        "study_design": join([find_field(pairs, "Study design", "Desenho"),
                              find_field(pairs, "Intervention assignment"),
                              find_field(pairs, "Masking", "Mascaramento")]),
        "conditions": join([
            find_field(pairs, "Health condition", "Condição de saúde"),
            find_field(pairs, "General descriptors")]),
        "interventions": find_field(pairs, "Interventions", "Intervenções"),
        "drug_names": find_field(pairs, "Interventions", "Intervention"),
        "sponsor": find_field(pairs, "Primary sponsor",
                              "Patrocinador principal"),
        "sponsor_type": find_field(pairs, "Institution type"),
        "collaborators": join([find_field(pairs, "Secondary sponsor"),
                               find_field(pairs, "Support source",
                                          "Fonte de financiamento")]),
        "countries": first_nonempty(
            find_field(pairs, "Countries of recruitment"), "Brazil"),
        "sites": find_field(pairs, "Recruitment site",
                            "Centro de recrutamento"),
        "target_enrollment": find_field(pairs, "Target sample size",
                                        "Tamanho da amostra"),
        "actual_enrollment": find_field(pairs, "Actual sample size"),
        "age_min": find_field(pairs, "Minimum age", "Idade mínima"),
        "age_max": find_field(pairs, "Maximum age", "Idade máxima"),
        "gender": find_field(pairs, "Gender", "Sexo"),
        "healthy_volunteers": find_field(pairs, "Healthy volunteers"),
        "inclusion_criteria": find_field(pairs, "Inclusion criteria",
                                         "Critérios de inclusão"),
        "exclusion_criteria": find_field(pairs, "Exclusion criteria",
                                         "Critérios de exclusão"),
        "primary_objective": find_field(pairs, "Brief summary", "Objetivo"),
        "primary_outcome": find_field(pairs, "Primary outcome",
                                      "Desfecho primário"),
        "secondary_outcome": find_field(pairs, "Secondary outcome",
                                        "Desfecho secundário"),
        "start_date": find_field(pairs, "Date of first enrollment",
                                 "Data do primeiro recrutamento"),
        "completion_date": find_field(pairs, "Date of completion",
                                      "Data de conclusão"),
        "registration_date": find_field(pairs, "Date of registration",
                                        "Data de registro"),
        "last_updated": find_field(pairs, "Last update", "Última atualização"),
        "results_available": find_field(pairs, "Results", "Resultados"),
        "findings": join([find_field(pairs, "Results summary",
                                     "Resumo dos resultados"),
                          find_field(pairs, "Conclusion", "Conclusão"),
                          find_field(pairs, "Publication")]),
        "contact": join([find_field(pairs, "Contact for public queries"),
                         find_field(pairs, "E-mail", "Email")]),
        "ethics_approval": join([
            find_field(pairs, "Ethics committee", "Comitê de ética"),
            find_field(pairs, "Approval date"),
            find_field(pairs, "CAAE")]),
        "url": url,
    })

    for k, v in pairs.items():
        col = "rebec." + re.sub(r"\s+", "_", k)[:80]
        if col not in row:
            row[col] = v
    return row


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True) -> List[Dict[str, Any]]:
    session = make_session()
    ids = _search_ids(drug, session, max_records)
    print(f"  ReBEC search returned {len(ids)} trial id(s).", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for i, rbr in enumerate(ids, start=1):
        row = None
        if details:
            try:
                row = _map_xml(http_get(session, TRIAL_XML.format(rbr=rbr)),
                               rbr)
            except Exception:
                row = None
        if row is None:
            url = TRIAL_URL.format(rbr=rbr)
            try:
                row = _map_html(http_get(session, url), rbr, url)
            except Exception as exc:
                print(f"  ! ReBEC detail failed for {rbr}: {exc}",
                      file=sys.stderr)
                continue
        rows.append(row)
        if i % 10 == 0:
            print(f"  ...ReBEC {i}/{len(ids)}", file=sys.stderr)
        time.sleep(DELAY)
    return rows


if __name__ == "__main__":
    sys.exit(run_cli(fetch, SRC_REBEC, "Fetch trials from ReBEC (Brazil)."))