"""
generate_efficacy_report.py
───────────────────────────
Reads clinical trial data from BigQuery `clinical_efficacy` table and generates
one professional PDF report **per molecule** using Gemini for narrative generation.

The report aggregates per-trial data into endpoint-level summaries
(Weight Loss, HbA1c, MASH Resolution, ALT Reduction), uses the stored
efficacy score and rationale, and enriches via Gemini+Search if data is thin.

Report structure (single-molecule, business-facing — max 2 pages):
  - Executive Summary
  - Endpoint Performance Overview
  - Scoring Methodology & Rationale
  - Strategic Implications
  - Scoring reference table (end of document)

Usage:
    python generate_efficacy_report.py
    python generate_efficacy_report.py --molecule Semaglutide
    python generate_efficacy_report.py --molecule "Semaglutide,Tirzepatide"
    python generate_efficacy_report.py --outdir ./reports
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config (all values sourced from gcp_utils) ────────────────────────────────
from gcp_utils import (
    GEMINI_API_KEY,
    GOOGLE_APPLICATION_CREDENTIALS,
    MODEL,
    RATIONALE_MODEL,
    PROJECT_ID,
    BQ_DATASET_ID,
    CLINICAL_EFFICACY_TABLE,
    GCS_BUCKET,
    GCS_REPORT_BASE_PATH,
    GCS_MEDICAL_POTENTIAL_SUBFOLDER,
)

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

from google import genai as genai_client
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY          = GEMINI_API_KEY
CREDENTIALS_PATH = GOOGLE_APPLICATION_CREDENTIALS

BQ_PROJECT_ID = PROJECT_ID
BQ_LOCATION   = os.environ.get("BQ_LOCATION", "asia-south1")
BQ_TABLE      = CLINICAL_EFFICACY_TABLE

GCS_BASE_PATH = f"{GCS_REPORT_BASE_PATH}/{GCS_MEDICAL_POTENTIAL_SUBFOLDER}"
GCS_FILE_NAME = "Clinical_Efficacy_Analysis.pdf"

REPORT_TITLE = "Clinical Efficacy Analysis"

MIN_RATIONALE_LENGTH = 80

# Scoring constants (from clinical_efficacy_scorer.py)
SCORE_TABLE = [(22.0, 5), (16.0, 4), (10.0, 3), (5.0, 2), (0.0, 1)]
ENDPOINT_WEIGHTS = {"weight_loss": 0.40, "hba1c": 0.40, "mash": 0.10, "alt": 0.10}
ENDPOINT_LABELS = {
    "weight_loss": "Weight Loss",
    "hba1c": "HbA1c Reduction",
    "mash": "MASH Resolution",
    "alt": "ALT Reduction",
}
PHASE_PENALTY = {3: 1.00, 2: 0.85, 1: 0.65}

# ── Colors ────────────────────────────────────────────────────────────────────
DARK_BLUE = colors.HexColor("#1F3864")
LIGHT_BLUE_BG = colors.HexColor("#E8EDF3")
ACCENT_BLUE = colors.HexColor("#2E5FA3")
WHITE = colors.white
LIGHT_GRAY = colors.HexColor("#666666")
DIVIDER_COLOR = colors.HexColor("#D0D7E3")

SCORE_COLORS = {
    5: colors.HexColor("#008000"),
    4: colors.HexColor("#4CAF50"),
    3: colors.HexColor("#CC9900"),
    2: colors.HexColor("#E65100"),
    1: colors.HexColor("#CC0000"),
}

SCORE_LABEL = {
    5: "Exceptional",
    4: "Strong",
    3: "Moderate",
    2: "Weak",
    1: "Poor",
}


# ── Gemini helpers ────────────────────────────────────────────────────────────

def call_gemini(prompt: str, use_search: bool = False) -> str:
    client = genai_client.Client(api_key=API_KEY)
    config_kwargs = {"temperature": 0.3}
    if use_search:
        config_kwargs["tools"] = [types.Tool(googleSearch=types.GoogleSearch())]
    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(model=MODEL, contents=prompt, config=config)
    return response.text.strip() if response.text else ""


def _extract_json(text: str):
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def _get_credentials():
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None


def _bq_client():
    credentials = _get_credentials()
    return bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials, location=BQ_LOCATION)


def load_from_bigquery(molecules: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Load ALL trial rows per molecule from the clinical_efficacy table.
    Returns a dict: {molecule_name: DataFrame of trials}.
    """
    client = _bq_client()

    molecule_filter = ""
    if molecules:
        escaped = ", ".join(f"'{m.strip()}'" for m in molecules)
        molecule_filter = f"AND molecule_name IN ({escaped})"

    query = f"""
    SELECT *
    FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE}`
    WHERE molecule_name IS NOT NULL {molecule_filter}
    ORDER BY molecule_name, phase DESC, created_at DESC
    """
    print(f"Loading clinical efficacy data from {BQ_TABLE}...")
    df = client.query(query).to_dataframe()
    if df.empty:
        print("  No data found.")
        return {}

    grouped = {name: group for name, group in df.groupby("molecule_name")}
    print(f"  Loaded data for {len(grouped)} molecule(s), {len(df)} total trial rows.")
    return grouped


# ── Endpoint computation from trial data ─────────────────────────────────────

def _parse_float(raw) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s.lower() in ("n/a", "", "0", "none", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_phase(raw) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip().upper().replace("PHASE", "").strip()
    if s.startswith("3"):
        return 3
    if s.startswith("2"):
        return 2
    if s.startswith("1"):
        return 1
    try:
        v = float(s)
        return 3 if v >= 3 else (2 if v >= 2 else 1)
    except ValueError:
        return None


def _pct_to_score(pct: float) -> int:
    for threshold, score in SCORE_TABLE:
        if pct >= threshold:
            return score
    return 1


def _compute_endpoint(trials_df: pd.DataFrame, value_col: str, duration_col: str) -> dict:
    """Compute best value for a single endpoint using phase-anchored selection."""
    best = {"raw_value": None, "adj_value": None, "phase_used": None,
            "penalty": 1.0, "score": None, "trial_id": "N/A",
            "dosage": "N/A", "duration": "N/A", "reason": "No valid data"}

    candidates = []
    for _, row in trials_df.iterrows():
        phase = _parse_phase(row.get("phase"))
        value = _parse_float(row.get(value_col))
        size = _parse_float(row.get("trial_size"))
        if phase is None or value is None or (size is not None and size <= 0):
            continue
        candidates.append({
            "phase": phase, "value": value,
            "trial_id": str(row.get("trial_id", "N/A")),
            "dosage": str(row.get("dosage", "N/A")),
            "duration": str(row.get(duration_col, "N/A")),
        })

    if not candidates:
        return best

    for target_phase in (3, 2, 1):
        phase_cands = [c for c in candidates if c["phase"] == target_phase]
        if not phase_cands:
            continue
        top = max(phase_cands, key=lambda c: c["value"])
        pen = PHASE_PENALTY[target_phase]
        adj = top["value"] * pen
        return {
            "raw_value": round(top["value"], 2),
            "adj_value": round(adj, 2),
            "phase_used": target_phase,
            "penalty": pen,
            "score": _pct_to_score(adj),
            "trial_id": top["trial_id"],
            "dosage": top["dosage"],
            "duration": top["duration"],
            "reason": f"Phase {target_phase}" + (f" (x{pen} penalty)" if pen < 1 else ""),
        }
    return best


# ── Extract molecule stats ────────────────────────────────────────────────────

def extract_molecule_stats(molecule_name: str, trials_df: pd.DataFrame) -> dict:
    """Aggregate trial data into a structured stats dict for report generation."""
    # Get stored score/rationale from first row (same across rows)
    first = trials_df.iloc[0]
    efficacy_score = _parse_float(first.get("efficacy_weighted_score"))
    data_coverage = str(first.get("efficacy_data_coverage", "N/A")) if pd.notna(first.get("efficacy_data_coverage")) else "N/A"
    rationale = str(first.get("efficacy_narrative_rationale", "")) if pd.notna(first.get("efficacy_narrative_rationale")) else ""

    score_int = None
    if efficacy_score is not None:
        score_int = max(1, min(5, round(efficacy_score)))

    # Compute per-endpoint breakdowns
    endpoints = {
        "weight_loss": _compute_endpoint(trials_df, "weight_change_pct", "weight_duration"),
        "hba1c": _compute_endpoint(trials_df, "hba1c_change_pct", "hba1c_duration"),
        "mash": _compute_endpoint(trials_df, "mash_change_pct", "mash_duration"),
        "alt": _compute_endpoint(trials_df, "alt_reduction_pct", "alt_duration"),
    }

    # Recompute weighted score from endpoints
    weighted_score = 0.0
    scored_count = 0
    for ep, result in endpoints.items():
        w = ENDPOINT_WEIGHTS[ep]
        if result["score"] is not None:
            weighted_score += result["score"] * w
            scored_count += 1

    # Prefer stored score, fall back to recomputed
    display_score = efficacy_score if efficacy_score is not None else round(weighted_score, 3)

    total_trials = len(trials_df)
    phases = trials_df["phase"].dropna().apply(lambda x: str(x).strip()).value_counts().to_dict()

    # Company name
    company = "N/A"
    for _, row in trials_df.iterrows():
        c = row.get("company_name")
        if pd.notna(c) and str(c).strip():
            company = str(c).strip()
            break

    return {
        "molecule_name": molecule_name,
        "efficacy_score": display_score,
        "score_int": score_int,
        "score_label": SCORE_LABEL.get(score_int, "N/A") if score_int else "N/A",
        "data_coverage": data_coverage,
        "rationale": rationale,
        "total_trials": total_trials,
        "phase_distribution": phases,
        "company_name": company,
        "endpoints": endpoints,
        "scored_endpoints": scored_count,
    }


# ── Data enrichment ───────────────────────────────────────────────────────────

def _is_data_sufficient(stats: dict) -> bool:
    rationale_ok = len(stats.get("rationale", "")) >= MIN_RATIONALE_LENGTH
    has_any_endpoint = stats.get("scored_endpoints", 0) >= 1
    return rationale_ok and has_any_endpoint


def enrich_molecule_data(stats: dict) -> dict:
    """If BQ data is insufficient, call Gemini with Google Search to fetch
    additional clinical efficacy information."""
    if _is_data_sufficient(stats):
        print(f"  BQ data is sufficient for {stats['molecule_name']} — skipping enrichment.")
        return stats

    print(f"  BQ data is thin for {stats['molecule_name']} — enriching via Gemini + Google Search...")

    prompt = f"""You are a pharmaceutical research analyst. Research the clinical efficacy data
for the following drug using the latest clinical trial publications, FDA labels, and registries.

Drug: {stats['molecule_name']}
Company: {stats['company_name']}
Current Score: {stats['efficacy_score']}/5

Find the best available data for these endpoints from Phase 2/3 trials:
1. Weight Loss (% body weight reduction)
2. HbA1c Reduction (% point reduction)
3. MASH/NASH Resolution (% of patients achieving resolution)
4. ALT Reduction (% reduction in ALT levels)

Return ONLY a valid JSON object:
{{
    "weight_loss_pct": "Best % weight loss from pivotal trials (e.g. 15.2%)",
    "weight_loss_trial": "Trial name/ID and dosage (e.g. STEP 1, 2.4mg, 68 weeks)",
    "hba1c_pct": "Best % HbA1c reduction (e.g. 1.8%)",
    "hba1c_trial": "Trial name/ID and dosage",
    "mash_pct": "Best MASH resolution % (or N/A if not studied)",
    "mash_trial": "Trial name/ID if available",
    "alt_pct": "Best ALT reduction % (or N/A)",
    "alt_trial": "Trial name/ID if available",
    "efficacy_narrative": "5-6 sentence comprehensive clinical efficacy summary covering all endpoints, key trial results, dosages, durations, and clinical significance",
    "key_trials": ["List of key trial names/IDs: STEP 1, SURPASS-1, etc."]
}}"""

    try:
        text = call_gemini(prompt, use_search=True)
        enrichment = _extract_json(text)
        if enrichment:
            enriched = dict(stats)
            if len(stats.get("rationale", "")) < MIN_RATIONALE_LENGTH and enrichment.get("efficacy_narrative"):
                enriched["rationale"] = enrichment["efficacy_narrative"]
            enriched["_enrichment"] = enrichment
            print(f"  Enrichment complete for {stats['molecule_name']}")
            return enriched
    except Exception as e:
        print(f"  [WARN] Enrichment failed for {stats['molecule_name']}: {e}")

    return stats


# ── LLM narrative (single molecule) ──────────────────────────────────────────

def generate_efficacy_narrative(stats: dict) -> dict:
    """Generate a business-focused clinical efficacy report narrative for Medical Affairs."""

    # Build endpoint summary block
    ep_lines = []
    for ep_key, ep_label in ENDPOINT_LABELS.items():
        ep = stats["endpoints"].get(ep_key, {})
        if ep.get("score") is not None:
            ep_lines.append(
                f"- {ep_label}: Best result {ep['raw_value']}% "
                f"(Phase {ep['phase_used']}, Trial {ep['trial_id']}, "
                f"Dosage {ep['dosage']}, Duration {ep['duration']})"
            )
        else:
            ep_lines.append(f"- {ep_label}: No valid data available")

    enrichment = stats.get("_enrichment", {})
    enrichment_block = ""
    if enrichment:
        enrich_parts = []
        for k in ["weight_loss_pct", "weight_loss_trial", "hba1c_pct", "hba1c_trial",
                   "mash_pct", "mash_trial", "alt_pct", "alt_trial", "efficacy_narrative"]:
            val = enrichment.get(k, "")
            if val and val != "N/A":
                enrich_parts.append(f"{k.replace('_', ' ').title()}: {val}")
        if enrich_parts:
            enrichment_block = "\n".join(enrich_parts)

    prompt = f"""You are a business-focused medical insights analyst.

Goal: Create a concise, 2-page Clinical Efficacy report for {stats['molecule_name']} that
highlights key efficacy findings and insights derived from the provided data outputs. The
report is intended for a Medical Affairs business audience.

Context: The data comes from structured outputs containing clinical trial data and extracted
efficacy-related data points (e.g., HbA1c reduction, weight change, liver enzyme changes,
response rates, treatment duration, trial size, etc.). The audience is non-technical and not
familiar with internal analytical frameworks, scoring methodologies, or internal jargon.

Source: Use only the provided data outputs as the source of truth. Focus specifically on
clinical efficacy-related fields and trial outcomes. Do not introduce external assumptions
unless clearly derived from the data.

=== PROVIDED DATA ===
- Product Name: {stats['molecule_name']}
- Company: {stats['company_name']}
- Efficacy Score: {stats['efficacy_score']} / 5 ({stats['score_label']})
- Data Coverage: {stats['data_coverage']}
- Total Trials Analyzed: {stats['total_trials']}
- Phase Distribution: {json.dumps(stats['phase_distribution'])}
- Endpoints Scored: {stats['scored_endpoints']}/4

ENDPOINT RESULTS:
{chr(10).join(ep_lines)}

STORED ANALYSIS RATIONALE:
{stats.get('rationale', 'Not available')}

ADDITIONAL RESEARCH DATA:
{enrichment_block if enrichment_block else 'Not available'}

=== INSTRUCTIONS ===

1. Start with: "Key Clinical Efficacy Findings for {stats['molecule_name']}"
   - Summarize the most important efficacy outcomes observed across trials
   - Highlight key metrics such as:
     - Improvements in disease-specific endpoints (HbA1c reduction, weight loss, liver markers)
     - Duration over which efficacy is observed
     - Consistency of results across studies or populations
   - Focus on what the data shows, not how it was calculated

2. Follow with: "Insights and Implications"
   - Translate clinical findings into business-relevant insights
   - Highlight:
     - Strength of efficacy across different endpoints
     - Any standout outcomes or differentiating factors
     - Gaps, inconsistencies, or limitations in efficacy data
     - Potential implications for positioning, adoption, or patient segments
   - Keep insights simple, clear, and actionable

3. Include: "Efficacy Profile Summary"
   - Provide a high-level synthesis of the molecule's overall clinical effectiveness
   - Comment on breadth of efficacy across multiple endpoints or indications
   - Mention the overall efficacy score briefly and in simple terms without explaining
     the scoring methodology

Language and Style:
- Do NOT use internal jargon, technical modeling terms, or scoring framework names
- Avoid statistical or methodological explanations unless essential
- Use clear, simple, and business-friendly language
- Translate clinical metrics into plain-language meaning wherever possible

Tone: Professional, objective, and insight-driven. Focus on clarity, relevance, and
business impact rather than technical depth.

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{{
  "key_findings": {{
    "summary_bullets": [
      "Key finding 1 about the most significant efficacy outcomes",
      "Key finding 2 about weight loss or metabolic improvements",
      "Key finding 3 about HbA1c reduction and glycemic control",
      "Key finding 4 about liver-related endpoints if available",
      "Key finding 5 about consistency across studies or notable trial results"
    ],
    "weight_loss_detail": "3-4 sentences in plain language about weight loss results — what percentage of weight loss was achieved, over what timeframe, at what dose, and what this means clinically for patients",
    "hba1c_detail": "3-4 sentences in plain language about HbA1c improvements — how much reduction was achieved, over what timeframe, and what this means for diabetes management",
    "liver_endpoints_detail": "2-3 sentences about MASH resolution and ALT reduction results if available, or a brief note on why data is limited for these endpoints"
  }},
  "insights_implications": {{
    "efficacy_strength": "2-3 sentences on the overall strength of efficacy and any standout results that differentiate this molecule",
    "gaps_limitations": "2-3 sentences on gaps or inconsistencies in the efficacy data, including missing endpoints or limited data",
    "positioning_impact": "2-3 sentences on what the efficacy profile implies for competitive positioning, adoption by physicians, and patient segments",
    "differentiation": "2-3 sentences on key differentiating factors compared to other treatments in the class"
  }},
  "profile_summary": {{
    "overall_assessment": "3-4 sentences providing a high-level synthesis of the molecule's clinical effectiveness, written for a business audience",
    "endpoint_breadth": "1-2 sentences commenting on the breadth of efficacy across multiple endpoints",
    "score_context": "1-2 sentences briefly mentioning the efficacy score in simple terms (e.g., 'The molecule demonstrates strong/moderate/limited clinical efficacy') WITHOUT explaining how the score was calculated"
  }}
}}"""

    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result

    # Fallback
    return {
        "key_findings": {
            "summary_bullets": [
                f"{stats['molecule_name']} shows efficacy across {stats['scored_endpoints']}/4 endpoints evaluated.",
                f"Data coverage: {stats['data_coverage']} across {stats['total_trials']} trials.",
            ],
            "weight_loss_detail": stats.get("rationale", "See detailed analysis."),
            "hba1c_detail": "See detailed analysis.",
            "liver_endpoints_detail": "See detailed analysis.",
        },
        "insights_implications": {
            "efficacy_strength": "Refer to detailed analysis.",
            "gaps_limitations": "Refer to detailed analysis.",
            "positioning_impact": "Refer to detailed analysis.",
            "differentiation": "Refer to detailed analysis.",
        },
        "profile_summary": {
            "overall_assessment": f"{stats['molecule_name']} received an efficacy score of {stats['efficacy_score']}/5 ({stats['score_label']}).",
            "endpoint_breadth": f"{stats['scored_endpoints']} out of 4 endpoints had evaluable data.",
            "score_context": f"The molecule demonstrates {stats['score_label'].lower()} clinical efficacy.",
        },
    }


# ── Style helpers ─────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Normal"],
            fontSize=20, leading=26, textColor=DARK_BLUE,
            alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=2,
        ),
        "molecule_name_title": ParagraphStyle(
            "MoleculeNameTitle", parent=base["Normal"],
            fontSize=14, leading=18, textColor=ACCENT_BLUE,
            alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"],
            fontSize=9, leading=12, textColor=LIGHT_GRAY,
            alignment=TA_CENTER, fontName="Helvetica", spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Normal"],
            fontSize=13, leading=16, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Normal"],
            fontSize=11, leading=14, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, leading=14, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=9, leading=13, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", spaceAfter=3, leftIndent=18, bulletIndent=6,
        ),
        "methodology_note": ParagraphStyle(
            "MethodologyNote", parent=base["Normal"],
            fontSize=8, leading=12, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=base["Normal"],
            fontSize=7, leading=10, textColor=colors.HexColor("#999999"),
            fontName="Helvetica", alignment=TA_CENTER, spaceBefore=10,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=base["Normal"],
            fontSize=8, leading=11, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader", parent=base["Normal"],
            fontSize=8, leading=11, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "section_label": ParagraphStyle(
            "SectionLabel", parent=base["Normal"],
            fontSize=10, leading=13, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceAfter=2, leftIndent=6,
        ),
    }


def _score_color(score_val):
    try:
        return SCORE_COLORS.get(int(float(score_val)), colors.black)
    except (ValueError, TypeError):
        return colors.black


def _render_bullets(items: list, styles: dict, story: list):
    for item in items:
        if item and str(item).strip():
            story.append(Paragraph(f"&#8226; {item}", styles["bullet"]))


def _scoring_framework_table(styles: dict) -> Table:
    """Render the clinical efficacy scoring reference table."""
    framework = [
        ("5", "Exceptional", "Adjusted efficacy value >= 22% (class-leading performance)"),
        ("4", "Strong",      "Adjusted efficacy value 16-21.9% (above-average efficacy)"),
        ("3", "Moderate",    "Adjusted efficacy value 10-15.9% (clinically meaningful)"),
        ("2", "Weak",        "Adjusted efficacy value 5-9.9% (below expectations)"),
        ("1", "Poor",        "Adjusted efficacy value < 5% (minimal clinical benefit)"),
    ]
    header = [
        Paragraph("Score", styles["cell_header"]),
        Paragraph("Label", styles["cell_header"]),
        Paragraph("Description", styles["cell_header"]),
    ]
    rows = [header]
    for sc, lbl, desc in framework:
        rows.append([
            Paragraph(sc, ParagraphStyle("FWS", parent=styles["cell"], fontName="Helvetica")),
            Paragraph(lbl, ParagraphStyle("FWL", parent=styles["cell"], fontName="Helvetica")),
            Paragraph(desc, ParagraphStyle("FWD", parent=styles["cell"], alignment=TA_LEFT)),
        ])

    tbl = Table(rows, colWidths=[0.5 * inch, 1.2 * inch, 5.0 * inch])
    row_bgs = [
        ("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE_BG if i % 2 == 0 else WHITE)
        for i in range(1, len(rows))
    ]
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",      (0, 0), (1, -1), "CENTER"),
        ("ALIGN",      (2, 1), (2, -1), "LEFT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        *row_bgs,
    ]))
    return tbl


def _endpoint_summary_table(stats: dict, styles: dict) -> Table:
    """Render the 4-endpoint performance summary table."""
    header = [
        Paragraph("Endpoint", styles["cell_header"]),
        Paragraph("Weight", styles["cell_header"]),
        Paragraph("Raw Value", styles["cell_header"]),
        Paragraph("Adjusted", styles["cell_header"]),
        Paragraph("Score", styles["cell_header"]),
        Paragraph("Phase", styles["cell_header"]),
        Paragraph("Trial / Dosage", styles["cell_header"]),
    ]
    rows = [header]
    for ep_key, ep_label in ENDPOINT_LABELS.items():
        ep = stats["endpoints"].get(ep_key, {})
        weight_pct = f"{int(ENDPOINT_WEIGHTS[ep_key] * 100)}%"
        if ep.get("score") is not None:
            sc = ep["score"]
            sc_color = _score_color(sc)
            score_style = ParagraphStyle("EPS", parent=styles["cell"], textColor=sc_color, fontName="Helvetica-Bold")
            rows.append([
                Paragraph(ep_label, ParagraphStyle("EPL", parent=styles["cell"], alignment=TA_LEFT)),
                Paragraph(weight_pct, styles["cell"]),
                Paragraph(f"{ep['raw_value']}%", styles["cell"]),
                Paragraph(f"{ep['adj_value']}%", styles["cell"]),
                Paragraph(f"{sc}/5", score_style),
                Paragraph(f"P{ep['phase_used']}", styles["cell"]),
                Paragraph(f"{ep['trial_id']} / {ep['dosage']}", ParagraphStyle("EPT", parent=styles["cell"], alignment=TA_LEFT, fontSize=7)),
            ])
        else:
            rows.append([
                Paragraph(ep_label, ParagraphStyle("EPL2", parent=styles["cell"], alignment=TA_LEFT)),
                Paragraph(weight_pct, styles["cell"]),
                Paragraph("N/A", styles["cell"]),
                Paragraph("N/A", styles["cell"]),
                Paragraph("N/A", styles["cell"]),
                Paragraph("—", styles["cell"]),
                Paragraph("No valid data", ParagraphStyle("EPT2", parent=styles["cell"], alignment=TA_LEFT, fontSize=7)),
            ])

    tbl = Table(rows, colWidths=[1.1*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.5*inch, 0.5*inch, 2.7*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        *[("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE_BG if i % 2 == 0 else WHITE)
          for i in range(1, len(rows))],
    ]))
    return tbl


# ── Single-molecule report builder ────────────────────────────────────────────

def build_single_molecule_report(stats: dict, narrative: dict, output_path: str):
    """Build and save a detailed PDF report for one molecule."""
    styles = build_styles()
    story = []

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        title=f"{REPORT_TITLE} — {stats['molecule_name']}",
        author="Clinical Efficacy Scorer",
    )

    # ── Title block ───────────────────────────────────────────────────────
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(stats["molecule_name"], styles["molecule_name_title"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))

    # ══════════════════════════════════════════════════════════════════════
    # Key Clinical Efficacy Findings
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph(
        f"Key Clinical Efficacy Findings for {stats['molecule_name']}", styles["h2"]
    ))

    key_findings = narrative.get("key_findings", {})
    if isinstance(key_findings, dict):
        # Bullet summary
        bullets = key_findings.get("summary_bullets", [])
        if isinstance(bullets, list):
            _render_bullets(bullets, styles, story)
        story.append(Spacer(1, 6))

        # Endpoint summary table (compact visual)
        story.append(_endpoint_summary_table(stats, styles))
        story.append(Spacer(1, 8))

        wl_detail = key_findings.get("weight_loss_detail", "")
        if wl_detail:
            story.append(Paragraph("<b>Weight Loss:</b>", styles["section_label"]))
            story.append(Paragraph(wl_detail, styles["body"]))
            story.append(Spacer(1, 4))

        hba1c_detail = key_findings.get("hba1c_detail", "")
        if hba1c_detail:
            story.append(Paragraph("<b>HbA1c Reduction:</b>", styles["section_label"]))
            story.append(Paragraph(hba1c_detail, styles["body"]))
            story.append(Spacer(1, 4))

        liver_detail = key_findings.get("liver_endpoints_detail", "")
        if liver_detail:
            story.append(Paragraph("<b>Liver Endpoints (MASH / ALT):</b>", styles["section_label"]))
            story.append(Paragraph(liver_detail, styles["body"]))
    elif isinstance(key_findings, str):
        story.append(Paragraph(key_findings, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # Insights and Implications
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Insights and Implications", styles["h2"]))

    insights = narrative.get("insights_implications", {})
    if isinstance(insights, dict):
        efficacy_str = insights.get("efficacy_strength", "")
        if efficacy_str:
            story.append(Paragraph(f"<b>Efficacy Strength:</b> {efficacy_str}", styles["body"]))
            story.append(Spacer(1, 4))

        gaps = insights.get("gaps_limitations", "")
        if gaps:
            story.append(Paragraph(f"<b>Gaps &amp; Limitations:</b> {gaps}", styles["body"]))
            story.append(Spacer(1, 4))

        positioning = insights.get("positioning_impact", "")
        if positioning:
            story.append(Paragraph(f"<b>Positioning &amp; Adoption:</b> {positioning}", styles["body"]))
            story.append(Spacer(1, 4))

        differentiation = insights.get("differentiation", "")
        if differentiation:
            story.append(Paragraph(f"<b>Differentiation:</b> {differentiation}", styles["body"]))
    elif isinstance(insights, str):
        story.append(Paragraph(insights, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # Efficacy Profile Summary
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Efficacy Profile Summary", styles["h2"]))

    profile_summary = narrative.get("profile_summary", {})
    if isinstance(profile_summary, dict):
        overall = profile_summary.get("overall_assessment", "")
        if overall:
            story.append(Paragraph(overall, styles["body"]))
            story.append(Spacer(1, 4))

        breadth = profile_summary.get("endpoint_breadth", "")
        if breadth:
            story.append(Paragraph(breadth, styles["body"]))
            story.append(Spacer(1, 4))

        score_ctx = profile_summary.get("score_context", "")
        if score_ctx:
            story.append(Paragraph(score_ctx, styles["body"]))
    elif isinstance(profile_summary, str):
        story.append(Paragraph(profile_summary, styles["body"]))
    story.append(Spacer(1, 8))

    # ── Scoring reference table ───────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Clinical Efficacy Scoring Reference", styles["h2"]))
    story.append(_scoring_framework_table(styles))
    story.append(Spacer(1, 8))

    legend_text = "  |  ".join(f"{k} = {v}" for k, v in SCORE_LABEL.items())
    story.append(Paragraph(
        f"<b>Score Legend:</b>  {legend_text}",
        ParagraphStyle("Legend", parent=styles["body"], fontSize=8, textColor=LIGHT_GRAY),
    ))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceBefore=14))
    story.append(Paragraph(
        "This report was auto-generated from Clinical Efficacy analysis output. "
        "For internal use only.",
        styles["footer"],
    ))

    doc.build(story)
    print(f"  Report saved -> {output_path}")


# ── GCS Upload ────────────────────────────────────────────────────────────────

def upload_to_gcs(local_path: str, molecule_name: str) -> str:
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage required. pip install google-cloud-storage")

    credentials = _get_credentials()
    client = storage.Client(project=BQ_PROJECT_ID, credentials=credentials)
    bucket = client.bucket(GCS_BUCKET)

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(molecule_name))
    blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_FILE_NAME}"
    gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"

    print(f"  Uploading to GCS: {gcs_uri}")
    bucket.blob(blob_name).upload_from_filename(local_path, content_type="application/pdf")
    return gcs_uri


# ── Public entry point ────────────────────────────────────────────────────────

def generate_efficacy_report(
    molecules: list[str] | None = None,
    outdir: str | None = None,
) -> list[str]:
    if not API_KEY:
        print("[Efficacy Report] GEMINI_API_KEY not set — skipping report generation.")
        return []

    out_root = Path(outdir) if outdir else Path(".")
    out_root.mkdir(parents=True, exist_ok=True)

    molecule_data = load_from_bigquery(molecules)
    if not molecule_data:
        print("[Efficacy Report] No data found — skipping.")
        return []

    output_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for molecule_name, trials_df in molecule_data.items():
        print(f"\nProcessing: {molecule_name}")

        stats = extract_molecule_stats(molecule_name, trials_df)
        stats = enrich_molecule_data(stats)

        print("  Generating narrative with Gemini...")
        narrative = generate_efficacy_narrative(stats)

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", molecule_name)
        output_path = str(out_root / f"efficacy_report_{safe_name}_{ts}.pdf")

        build_single_molecule_report(stats, narrative, output_path)
        output_paths.append(output_path)

        try:
            gcs_uri = upload_to_gcs(output_path, molecule_name)
            print(f"  GCS: {gcs_uri}")
        except Exception as e:
            print(f"  [WARN] GCS upload failed for '{molecule_name}': {e}")

    print(f"\nDone. {len(output_paths)} report(s) generated.")
    return output_paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate one Clinical Efficacy PDF per molecule from BigQuery data."
    )
    parser.add_argument("--molecule", "-m", default=None,
        help="Comma-separated molecule name(s). Omit for all.")
    parser.add_argument("--outdir", "-o", default=None,
        help="Output directory for PDFs (default: current directory).")
    args = parser.parse_args()

    mols = None
    if args.molecule:
        mols = [m.strip() for m in args.molecule.split(",") if m.strip()]

    generate_efficacy_report(molecules=mols, outdir=args.outdir)


if __name__ == "__main__":
    main()