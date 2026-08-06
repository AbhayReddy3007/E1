#!/usr/bin/env python3
"""
gcp_utils.py – Central configuration for the clinical trials pipeline.

Fill in your values below. Every other script imports from this file.

Quick check:
    python gcp_utils.py
"""

from __future__ import annotations
import os

# ==============================================================================
# GCP / PROJECT
# ==============================================================================

PROJECT_ID: str = os.getenv("PROJECT_ID", "")
"""Your GCP project ID — e.g. "my-project-123" """

GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
"""Absolute path to a service-account JSON key file.
   Leave blank to use Application Default Credentials (gcloud auth login)."""

# ==============================================================================
# BIGQUERY
# ==============================================================================

BQ_DATASET_ID: str = os.getenv("BQ_DATASET_ID", "clinical_trials")
"""BigQuery dataset that holds all pipeline tables."""

CLINICAL_EFFICACY_TABLE: str = os.getenv("CLINICAL_EFFICACY_TABLE", "clinical_efficacy")
"""BQ table written by push_to_bq.py and read by generate_efficacy_report.py."""

DIM_SCORES_TABLE: str = os.getenv("DIM_SCORES_TABLE", "dim_scores")
"""BQ table for dimension-level scores (reserved for future use)."""

BQ_LOCATION: str = os.getenv("BQ_LOCATION", "US")
"""BigQuery dataset location — e.g. "US", "EU", "asia-south1" """

# ==============================================================================
# GOOGLE CLOUD STORAGE
# ==============================================================================

GCS_BUCKET: str = os.getenv("GCS_BUCKET", "")
"""GCS bucket name — no gs:// prefix — e.g. "my-data-bucket" """

GCS_REPORT_BASE_PATH: str = os.getenv("GCS_REPORT_BASE_PATH", "reports")
"""Base folder inside GCS_BUCKET where PDF reports are uploaded."""

GCS_MEDICAL_POTENTIAL_SUBFOLDER: str = os.getenv("GCS_MEDICAL_POTENTIAL_SUBFOLDER", "medical_potential")
"""Sub-folder under GCS_REPORT_BASE_PATH for efficacy PDF reports."""

GCS_PIPELINE_CACHE_BASE_PATH: str = os.getenv("GCS_PIPELINE_CACHE_BASE_PATH", "pipeline_cache")
"""Base folder for caching intermediate pipeline artefacts (reserved)."""

# ==============================================================================
# GEMINI / GOOGLE AI
# ==============================================================================

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
"""Google API key — fallback if GEMINI_API_KEY is not set."""

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or GOOGLE_API_KEY
"""Gemini API key — get one at https://aistudio.google.com/app/apikey"""

MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
"""Gemini model used for trial enrichment prompts."""

RATIONALE_MODEL: str = os.getenv("GEMINI_RATIONALE_MODEL", "gemini-2.5-flash")
"""Gemini model used for narrative rationale generation."""

# ==============================================================================
# CONVENIENCE HELPERS  (no need to edit below this line)
# ==============================================================================

def get_active_api_key() -> str:
    """Return whichever API key is set."""
    return GEMINI_API_KEY or GOOGLE_API_KEY


def full_bq_table(table_name: str) -> str:
    """Return a fully-qualified BQ table ID: project.dataset.table."""
    return f"{PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"


def gcs_path(*parts: str) -> str:
    """Join GCS path segments cleanly."""
    return "/".join(p.strip("/") for p in parts if p)


def validate_config(raise_on_error: bool = True) -> list[str]:
    """Check that all required variables are set."""
    required = {
        "PROJECT_ID":                      PROJECT_ID,
        "GEMINI_API_KEY / GOOGLE_API_KEY": get_active_api_key(),
        "GCS_BUCKET":                      GCS_BUCKET,
    }
    missing = [name for name, val in required.items() if not val]
    if missing and raise_on_error:
        raise ValueError(
            "Missing required values in gcp_utils.py:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
    return missing


# ==============================================================================
# SELF-CHECK  —  python gcp_utils.py
# ==============================================================================

if __name__ == "__main__":
    import sys

    W = 38
    print("\n── gcp_utils – Current Configuration ──────────────────────────")
    display = {
        "PROJECT_ID":                      PROJECT_ID                      or "(not set)",
        "GOOGLE_APPLICATION_CREDENTIALS":  GOOGLE_APPLICATION_CREDENTIALS  or "(using ADC / gcloud auth)",
        "BQ_DATASET_ID":                   BQ_DATASET_ID,
        "CLINICAL_EFFICACY_TABLE":         CLINICAL_EFFICACY_TABLE,
        "DIM_SCORES_TABLE":                DIM_SCORES_TABLE,
        "BQ_LOCATION":                     BQ_LOCATION,
        "GCS_BUCKET":                      GCS_BUCKET                      or "(not set)",
        "GCS_REPORT_BASE_PATH":            GCS_REPORT_BASE_PATH,
        "GCS_MEDICAL_POTENTIAL_SUBFOLDER": GCS_MEDICAL_POTENTIAL_SUBFOLDER,
        "GCS_PIPELINE_CACHE_BASE_PATH":    GCS_PIPELINE_CACHE_BASE_PATH,
        "GOOGLE_API_KEY":                  "***set***" if GOOGLE_API_KEY   else "(not set)",
        "GEMINI_API_KEY":                  "***set***" if GEMINI_API_KEY   else "(not set)",
        "MODEL":                           MODEL,
        "RATIONALE_MODEL":                 RATIONALE_MODEL,
    }
    for k, v in display.items():
        print(f"  {k:<{W}} {v}")
    print()

    missing = validate_config(raise_on_error=False)
    if missing:
        print("⚠  Missing required vars:", ", ".join(missing))
        sys.exit(1)
    print("✓  All required variables are set.\n")