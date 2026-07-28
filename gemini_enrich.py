#!/usr/bin/env python3
"""
gemini_enrich.py – add trial text + Gemini-extracted metric columns.

Adds these columns to every row:

    entire_trial_information   every non-empty field of the record, as text
    weight_reduction           weight / body-weight reduction findings
    hba1c_reduction            HbA1c reduction findings
    mash                       MASH / NASH findings
    alt_reduction              ALT reduction findings

The four metric columns are filled by **Gemini 2.5 Flash** reading ONLY the
`entire_trial_information` text for that row. When a metric is not present,
the column reads exactly:  INFO N/A

Anti-hallucination design
-------------------------
The model is not trusted on its own. Four independent guards apply:

 1. `temperature=0` and `thinkingBudget=0` — deterministic, no free reasoning.
 2. A strict JSON `responseSchema` forces the model to return, for every
    metric, both a `finding` AND the `evidence` — a verbatim span copied from
    the source text.
 3. **Evidence verification (programmatic).** The returned `evidence` must
    appear verbatim in the source text (whitespace-normalised). If it does
    not, the finding is discarded and replaced with INFO N/A.
 4. **Number verification (programmatic).** Every number appearing in the
    `finding` must also appear in the source text. A fabricated percentage or
    p-value therefore cannot survive, even if the evidence span was real.

Guards 3 and 4 run in Python, so a hallucinated value is dropped by code, not
by trusting the model's own claim. Rows dropped by a guard are counted and
reported at the end of the run.

Setup
-----
    pip install requests
    export GEMINI_API_KEY="your-key"        # https://aistudio.google.com/apikey

Standalone use on an existing Excel file:

    python gemini_enrich.py trials.xlsx -o trials_enriched.xlsx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"
NA = "INFO N/A"

# columns this module appends, in order
ENRICHED_COLUMNS = [
    "entire_trial_information",
    "weight_reduction",
    "hba1c_reduction",
    "mash",
    "alt_reduction",
]

METRICS = {
    "weight_reduction": (
        "Weight reduction / body-weight change. Includes absolute weight loss "
        "(kg, lb), percent body-weight change, BMI change, waist circumference "
        "change, or weight-loss responder rates."
    ),
    "hba1c_reduction": (
        "HbA1c (glycated haemoglobin, A1c) reduction or change, in percentage "
        "points or mmol/mol, including HbA1c target-attainment rates."
    ),
    "mash": (
        "MASH (metabolic dysfunction-associated steatohepatitis) or its former "
        "name NASH, including MASH/NASH resolution, fibrosis improvement, "
        "steatosis, liver fat fraction, or MASLD/NAFLD outcomes."
    ),
    "alt_reduction": (
        "ALT (alanine aminotransferase / SGPT) reduction or change, in U/L or "
        "IU/L, or ALT normalisation rates."
    ),
}

SYSTEM_INSTRUCTION = """You are a strict information-extraction tool for clinical trial records.

RULES — follow exactly:
1. Use ONLY the trial text supplied by the user. You have no other knowledge.
   Never use anything you know about the drug, the sponsor, or the disease.
2. Never infer, estimate, calculate, or generalise a value. Only report what
   is literally written in the text.
3. For each metric, if the text does not explicitly discuss it, set
   "finding" to exactly "INFO N/A" and "evidence" to "".
4. If a metric is only named as a planned outcome measure with NO result
   value reported, say so plainly, e.g. "Listed as a primary endpoint; no
   result value reported." That is a valid finding.
5. "evidence" MUST be an exact, character-for-character span copied from the
   supplied text that supports the finding. Do not paraphrase it, do not
   correct its spelling, do not join separate sentences. If you cannot copy an
   exact supporting span, the finding must be "INFO N/A".
6. Every number in "finding" must also appear in the supplied text. Never
   write a number that is not in the text.
7. Keep each "finding" under 300 characters."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        m: {
            "type": "OBJECT",
            "properties": {
                "finding": {"type": "STRING",
                            "description": f"What the text says about: {desc} "
                                           f"Or exactly 'INFO N/A'."},
                "evidence": {"type": "STRING",
                             "description": "Exact verbatim span copied from "
                                            "the supplied text, or ''."},
            },
            "required": ["finding", "evidence"],
            "propertyOrdering": ["finding", "evidence"],
        }
        for m, desc in METRICS.items()
    },
    "required": list(METRICS),
    "propertyOrdering": list(METRICS),
}

MAX_CHARS = 120_000          # per-row cap sent to the model
_print_lock = threading.Lock()


# ── Build the "entire trial information" text ────────────────────────────────
def build_trial_text(row: Dict[str, Any], max_chars: int = MAX_CHARS) -> str:
    """
    Flatten every non-empty field of a row into readable 'Label: value' lines.
    Unified columns come first, then registry-specific extras.
    """
    from registry_common import UNIFIED_COLUMNS

    lines: List[str] = []
    used = set()

    for col in UNIFIED_COLUMNS:
        val = str(row.get(col, "") or "").strip()
        if val and col not in ENRICHED_COLUMNS:
            label = col.replace("_", " ").title()
            lines.append(f"{label}: {val}")
            used.add(col)

    extras = [k for k in row
              if k not in used and k not in ENRICHED_COLUMNS
              and str(row.get(k, "") or "").strip()]
    for col in sorted(extras):
        val = str(row[col]).strip()
        label = col.replace("_", " ").replace(".", " > ")
        lines.append(f"{label}: {val}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[TRUNCATED]"
    return text


# ── Verification guards ──────────────────────────────────────────────────────
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _numbers(text: str) -> List[str]:
    return [n.replace(",", ".").rstrip("0").rstrip(".") if "." in n.replace(",", ".")
            else n.replace(",", "")
            for n in NUM_RE.findall(text or "")]


def verify(finding: str, evidence: str, source: str) -> tuple:
    """
    Returns (safe_finding, reason_rejected_or_empty).
    Rejects anything not grounded in `source`.
    """
    finding = (finding or "").strip()
    evidence = (evidence or "").strip()

    if not finding or finding.upper().replace("\\", "/") in (
            "INFO N/A", "INFO NA", "N/A", "NA", "NONE"):
        return NA, ""

    src_norm = _norm(source)

    # Guard 3: evidence must be a verbatim span of the source
    if not evidence:
        return NA, "no evidence supplied"
    ev_norm = _norm(evidence)
    if len(ev_norm) < 8:
        return NA, "evidence too short to verify"
    if ev_norm not in src_norm:
        return NA, "evidence not found verbatim in source"

    # Guard 4: every number in the finding must exist in the source
    src_numbers = set(_numbers(source))
    for num in _numbers(finding):
        if num not in src_numbers:
            return NA, f"number '{num}' not present in source"

    return finding, ""


# ── Gemini call ──────────────────────────────────────────────────────────────
class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 timeout: int = 120, retries: int = 4):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.url = API_BASE.format(model=model)
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        })

    def extract(self, trial_text: str) -> Dict[str, Dict[str, str]]:
        prompt = (
            "Extract the four metrics below from the CLINICAL TRIAL TEXT.\n"
            "Use only that text. If a metric is absent, return exactly "
            "'INFO N/A'.\n\n"
            + "\n".join(f"- {m}: {d}" for m, d in METRICS.items())
            + "\n\n=== CLINICAL TRIAL TEXT START ===\n"
            + trial_text
            + "\n=== CLINICAL TRIAL TEXT END ===\n"
        )
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "topP": 1,
                "candidateCount": 1,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
                # 2.5-series: disable thinking for determinism + cost
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        last = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.post(self.url, data=json.dumps(body),
                                      timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                payload = r.json()
                cands = payload.get("candidates") or []
                if not cands:
                    raise RuntimeError(f"no candidates: {str(payload)[:200]}")
                parts = (cands[0].get("content") or {}).get("parts") or []
                raw = "".join(p.get("text", "") for p in parts).strip()
                raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
                return json.loads(raw)
            except Exception as exc:                       # noqa: BLE001
                last = exc
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini call failed after {self.retries} tries: {last}")


# ── Cache ────────────────────────────────────────────────────────────────────
class Cache:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.data: Dict[str, Any] = {}
        self.lock = threading.Lock()
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    self.data = json.load(fh)
            except Exception:
                self.data = {}

    @staticmethod
    def key(text: str, model: str) -> str:
        return hashlib.sha256((model + "\x00" + text).encode("utf-8")).hexdigest()

    def get(self, k: str):
        with self.lock:
            return self.data.get(k)

    def put(self, k: str, value):
        with self.lock:
            self.data[k] = value

    def save(self):
        if not self.path:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh)
        except Exception as exc:
            print(f"  ! cache save failed: {exc}", file=sys.stderr)


# ── Main entry point ─────────────────────────────────────────────────────────
def enrich_rows(rows: List[Dict[str, Any]], api_key: Optional[str] = None,
                model: str = DEFAULT_MODEL, workers: int = 4,
                cache_path: Optional[str] = None,
                skip_llm: bool = False) -> List[Dict[str, Any]]:
    """
    Add `entire_trial_information` + the four metric columns to every row.
    If skip_llm (or no API key), the text column is still added and the four
    metric columns are all set to INFO N/A.
    """
    for row in rows:
        row["entire_trial_information"] = build_trial_text(row)
        for m in METRICS:
            row.setdefault(m, NA)

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if skip_llm or not api_key:
        if not skip_llm:
            print("  [Gemini] no GEMINI_API_KEY set – metric columns left as "
                  f"'{NA}'.", file=sys.stderr)
        return rows

    client = GeminiClient(api_key, model)
    cache = Cache(cache_path)
    stats = {"cached": 0, "called": 0, "failed": 0, "rejected": 0}
    stat_lock = threading.Lock()

    def work(idx_row):
        idx, row = idx_row
        text = row["entire_trial_information"]
        if not text.strip():
            return idx, {m: NA for m in METRICS}

        ckey = Cache.key(text, model)
        cached = cache.get(ckey)
        if cached is not None:
            with stat_lock:
                stats["cached"] += 1
            return idx, cached

        try:
            raw = client.extract(text)
        except Exception as exc:                           # noqa: BLE001
            with _print_lock:
                print(f"  ! Gemini failed on row {idx}: {exc}", file=sys.stderr)
            with stat_lock:
                stats["failed"] += 1
            return idx, {m: NA for m in METRICS}

        result: Dict[str, str] = {}
        for m in METRICS:
            item = raw.get(m) or {}
            safe, reason = verify(item.get("finding", ""),
                                  item.get("evidence", ""), text)
            if reason:
                with stat_lock:
                    stats["rejected"] += 1
                with _print_lock:
                    print(f"  [guard] row {idx} {m}: rejected ({reason})",
                          file=sys.stderr)
            result[m] = safe

        cache.put(ckey, result)
        with stat_lock:
            stats["called"] += 1
        return idx, result

    print(f"  [Gemini] extracting metrics for {len(rows)} row(s) "
          f"with {model} ...", file=sys.stderr)

    todo = list(enumerate(rows))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, item) for item in todo]
        done = 0
        for fut in as_completed(futures):
            idx, result = fut.result()
            rows[idx].update(result)
            done += 1
            if done % 25 == 0:
                print(f"  [Gemini] {done}/{len(rows)}", file=sys.stderr)

    cache.save()
    print(f"  [Gemini] done — {stats['called']} API call(s), "
          f"{stats['cached']} cached, {stats['failed']} failed, "
          f"{stats['rejected']} finding(s) rejected by grounding guards",
          file=sys.stderr)
    return rows


# ── Standalone CLI: enrich an existing Excel/CSV ─────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Add entire_trial_information + Gemini-extracted metric "
                    "columns to an existing trials Excel/CSV file.")
    ap.add_argument("infile", help=".xlsx or .csv produced by fetch_all_trials.py")
    ap.add_argument("-o", "--outfile", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default=".gemini_cache.json")
    ap.add_argument("--no-llm", action="store_true",
                    help="only build entire_trial_information; leave metrics INFO N/A")
    args = ap.parse_args()

    import pandas as pd
    from registry_common import UNIFIED_COLUMNS, write_excel

    df = (pd.read_csv(args.infile) if args.infile.lower().endswith(".csv")
          else pd.read_excel(args.infile))
    df = df.fillna("")
    rows = df.to_dict("records")

    rows = enrich_rows(rows, model=args.model, workers=args.workers,
                       cache_path=args.cache, skip_llm=args.no_llm)

    out = args.outfile or re.sub(r"\.(xlsx|csv)$", "_enriched.xlsx",
                                 args.infile, flags=re.I)
    cols = [c for c in UNIFIED_COLUMNS if c in df.columns] + ENRICHED_COLUMNS
    write_excel(rows, out, cols, sheet_name="Trials")

    filled = {m: sum(1 for r in rows if r.get(m) != NA) for m in METRICS}
    print("\nMetric coverage:", file=sys.stderr)
    for m, n in filled.items():
        print(f"  {m:20s} {n}/{len(rows)} row(s) with data", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
