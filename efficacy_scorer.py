#!/usr/bin/env python3
"""
efficacy_scorer.py – Phase-Anchored Clinical Efficacy Scoring.

Scoring rules (from reference):
  Phase 3 → no penalty | Phase 2 → x0.85 | Phase 1 → x0.65
  Score table: >=22% → 5 | 16-21.9% → 4 | 10-15.9% → 3 | 5-9.9% → 2 | <5% → 1
  Weights: Weight Loss 0.40 | HbA1c 0.40 | MASH 0.10 | ALT 0.10
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

SCORE_TABLE = [(22.0, 5), (16.0, 4), (10.0, 3), (5.0, 2), (0.0, 1)]

ENDPOINT_WEIGHTS = {
    "weight_loss": 0.40,
    "hba1c": 0.40,
    "mash": 0.10,
    "alt": 0.10,
}

FIELD_MAP = {
    "weight_loss": "weight_change_pct",
    "hba1c": "hba1c_change_pct",
    "mash": "mash_resolution_pct",
    "alt": "alt_reduction_pct",
}

PHASE_PENALTY = {3: 1.00, 4: 1.00, 2: 0.85, 1: 0.65}


def _parse_phase(raw) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip().upper().replace("PHASE", "").strip()
    if s.startswith("3"):
        return 3
    if s.startswith("4"):
        return 3  # Phase 4 treated same as Phase 3
    if s.startswith("2"):
        return 2
    if s.startswith("1"):
        return 1
    try:
        v = float(s)
        if v >= 3: return 3
        if v >= 2: return 2
        return 1
    except ValueError:
        return None


def _parse_float(raw) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip().rstrip("%")
    if s.lower() in ("n/a", "", "0", "none", "info n/a"):
        return None
    # Handle negative signs (weight loss reported as negative)
    s = s.lstrip("-")
    try:
        return abs(float(s))
    except ValueError:
        return None


def _pct_to_score(pct: float) -> int:
    for threshold, score in SCORE_TABLE:
        if pct >= threshold:
            return score
    return 1


def score_endpoint(trials: List[Dict[str, Any]], value_field: str) -> dict:
    """Score a single endpoint. Prefer Phase 3, fall back to 2 then 1."""
    valid = []
    for t in trials:
        phase = _parse_phase(t.get("phase"))
        value = _parse_float(t.get(value_field))
        size = _parse_float(t.get("trial_size")) or 0
        tid = t.get("trial_id", "")
        if phase is None or value is None or size <= 0:
            continue
        valid.append({"phase": phase, "value": value, "n": size,
                      "trial_id": tid, "dosage": t.get("dosage", "N/A")})

    if not valid:
        return {"score": None, "raw_value": None, "best_value": None,
                "phase_used": None, "penalty": 1.0,
                "reason": "No valid data"}

    for target in (3, 2, 1):
        phase_trials = [r for r in valid if r["phase"] == target]
        if not phase_trials:
            continue
        # Deduplicate by trial_id, keep highest value
        groups: Dict[str, list] = {}
        for t in phase_trials:
            groups.setdefault(t["trial_id"], []).append(t)
        deduped = [max(g, key=lambda x: x["value"]) for g in groups.values()]
        best = max(deduped, key=lambda r: r["value"])
        pen = PHASE_PENALTY[target]
        adj = best["value"] * pen
        return {
            "score": _pct_to_score(adj),
            "raw_value": round(best["value"], 2),
            "best_value": round(adj, 2),
            "phase_used": target,
            "penalty": pen,
            "trial_id": best["trial_id"],
            "dosage": best["dosage"],
            "reason": f"Phase {target}" + (f" (x{pen})" if pen < 1 else ""),
        }

    return {"score": None, "raw_value": None, "best_value": None,
            "phase_used": None, "penalty": 1.0, "reason": "No valid phase"}


def compute_efficacy_score(trials: List[Dict[str, Any]]) -> dict:
    """Compute weighted efficacy score across all 4 endpoints."""
    endpoint_results = {}
    for ep, field in FIELD_MAP.items():
        endpoint_results[ep] = score_endpoint(trials, field)

    score_sum = 0.0
    for ep, result in endpoint_results.items():
        w = ENDPOINT_WEIGHTS[ep]
        score_sum += (result["score"] or 0) * w

    return {
        "weighted_score": round(score_sum, 3),
        "endpoints": endpoint_results,
    }


def add_efficacy_column(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add an efficacy_score column to each row.

    The score is per-trial: we score each row individually based on its own
    endpoint values, phase, and size.
    """
    for row in rows:
        result = compute_efficacy_score([row])
        row["efficacy_score"] = result["weighted_score"]
    return rows