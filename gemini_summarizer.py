#!/usr/bin/env python3
"""
gemini_summarizer.py – key-findings extraction with Gemini 2.5 Flash.

    Model    : gemini-2.5-flash
    Endpoint : https://generativelanguage.googleapis.com/v1beta/models/
               gemini-2.5-flash:generateContent

Setup
-----
    export GEMINI_API_KEY="your-key"        # or GOOGLE_API_KEY

Two modes
---------
* **grounded-in-registry (default)** — Gemini only summarises text that the
  registry itself published for that trial. If a trial reported no results,
  the output says so. It is not allowed to fill gaps from its own knowledge.

* **--grounded (opt-in)** — additionally lets Gemini use Google Search to
  find *published* results (papers, press releases) for trials whose registry
  record has no results. Findings from this route land in separate
  `ai_published_*` columns with source URLs, so you can always tell registry
  fact from web-sourced claim.

>>> ACCURACY WARNING <<<
An LLM can misread or fabricate clinical numbers. Every row carries
`ai_evidence_basis` telling you what the summary was built from, and
`ai_has_registry_results` (Yes/No). Treat `ai_*` columns as a reading aid and
verify anything that matters against the `url` column. Do not use these
summaries for clinical or regulatory decisions without checking the source.
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

from registry_common import UNIFIED_COLUMNS, clean, write_excel

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
            "{model}:generateContent")

# columns this module adds
AI_COLUMNS = [
    "ai_key_findings",
    "ai_primary_result",
    "ai_secondary_results",
    "ai_safety_findings",
    "ai_conclusion",
    "ai_summary",
    "ai_has_registry_results",
    "ai_evidence_basis",
    "ai_published_findings",
    "ai_sources",
    "ai_model",
]

# fields fed to the model as the trial's factual basis
CONTEXT_FIELDS = [
    "source", "trial_id", "secondary_ids", "title", "public_title", "status",
    "phase", "study_type", "study_design", "conditions", "interventions",
    "drug_names", "sponsor", "countries", "target_enrollment",
    "actual_enrollment", "age_min", "age_max", "gender",
    "primary_objective", "primary_outcome", "secondary_outcome",
    "start_date", "completion_date", "results_available", "findings",
]

# registry-specific extras worth including when present
RESULT_HINTS = ("result", "finding", "outcome", "conclusion", "endpoint",
                "adverse", "efficacy", "publication", "summary", "limitation")

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_registry_results": {
            "type": "string",
            "enum": ["Yes", "No"],
            "description": "Yes only if the supplied record contains actual "
                           "reported results/outcome data.",
        },
        "key_findings": {
            "type": "string",
            "description": "All key findings as short clauses separated by ' | '. "
                           "Keep every number, percentage, ratio, CI and p-value "
                           "exactly as written in the record. If the record has no "
                           "results, write exactly: No results reported in registry record",
        },
        "primary_result": {
            "type": "string",
            "description": "Result for the primary endpoint with its numbers, "
                           "or empty string if not reported.",
        },
        "secondary_results": {
            "type": "string",
            "description": "Results for secondary endpoints with numbers, "
                           "or empty string.",
        },
        "safety_findings": {
            "type": "string",
            "description": "Adverse events / safety signals reported, or empty string.",
        },
        "conclusion": {
            "type": "string",
            "description": "The conclusion stated in the record, or empty string. "
                           "Do not invent a conclusion.",
        },
        "summary": {
            "type": "string",
            "description": "Two to three sentence plain summary of what the trial "
                           "tested and what it found.",
        },
        "evidence_basis": {
            "type": "string",
            "description": "Which supplied fields the findings came from, e.g. "
                           "'findings, primary_outcome' or 'no results fields present'.",
        },
    },
    "required": ["has_registry_results", "key_findings", "summary", "evidence_basis"],
}

SYSTEM_RULES = """You extract clinical trial findings from registry records.

HARD RULES:
1. Use ONLY the record text supplied in the user message. You have no other
   knowledge of this trial. Never use your training knowledge to add, complete,
   guess or "correct" any result.
2. Copy every number exactly: percentages, means, medians, hazard/odds/risk
   ratios, confidence intervals, p-values, sample sizes. Never round, rescale,
   recompute or infer a number that is not written in the record.
3. A registry record often contains NO results — only a protocol (what the
   trial plans to measure). Planned endpoints are NOT findings. In that case set
   has_registry_results to "No" and key_findings to exactly:
   No results reported in registry record
4. Never state a treatment worked, was safe, was superior, or was
   well-tolerated unless the record says so. Report what is written, nothing more.
5. If the record is ambiguous, say so in the field rather than resolving it.
6. Output valid JSON matching the schema. No markdown, no commentary."""

GROUNDED_RULES = """You research published results for a clinical trial.

You are given a registry record whose own results section is empty. Use Google
Search to find published results for THIS EXACT TRIAL, matched by its
registration ID (e.g. NCT number, CTRI number, EudraCT number, jRCT ID) or by
an exact title match plus matching sponsor and phase.

HARD RULES:
1. Only report findings you can attribute to a specific search result you
   actually retrieved. Give the URL for each claim.
2. If you cannot confidently match a publication to THIS trial, return
   published_findings as exactly: No published results located
   Do not substitute results from a similar or related trial.
3. Copy numbers exactly as the source states them. Never estimate.
4. Never use your training knowledge as the source of a finding.
5. Reply as JSON only, with keys: published_findings (string, key findings with
   numbers, separated by ' | '), sources (string, URLs separated by ' | ').
   No markdown fences."""


class RateLimiter:
    """Simple min-interval limiter shared across threads."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self._interval


class GeminiSummarizer:
    def __init__(self, api_key: Optional[str] = None, model: str = MODEL,
                 rpm: int = 60, thinking_budget: int = 0,
                 cache_path: Optional[str] = ".gemini_cache.json",
                 grounded: bool = False, timeout: int = 120,
                 retries: int = 4):
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY")
                        or os.environ.get("GOOGLE_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError(
                "No Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        self.model = model
        self.url = ENDPOINT.format(model=model)
        self.limiter = RateLimiter(rpm)
        self.thinking_budget = thinking_budget
        self.grounded = grounded
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        })
        self.cache_path = cache_path
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self.calls = 0
        self.cached_hits = 0
        self.failures = 0
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except Exception:
                self._cache = {}

    # ── cache ────────────────────────────────────────────────────────────────
    def _key(self, payload_text: str, tag: str) -> str:
        h = hashlib.sha256((self.model + "|" + tag + "|" + payload_text)
                           .encode("utf-8")).hexdigest()
        return h[:32]

    def save_cache(self):
        if not self.cache_path:
            return
        with self._cache_lock:
            try:
                with open(self.cache_path, "w", encoding="utf-8") as fh:
                    json.dump(self._cache, fh)
            except Exception as exc:
                print(f"  ! cache write failed: {exc}", file=sys.stderr)

    # ── API ──────────────────────────────────────────────────────────────────
    def _call(self, prompt: str, system: str, *,
              use_schema: bool, use_search: bool) -> Optional[str]:
        body: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 2048,
                "thinkingConfig": {"thinkingBudget": self.thinking_budget},
            },
        }
        # NOTE: Gemini does not allow responseSchema together with tools, so
        # the grounded path asks for JSON in the prompt and parses it manually.
        if use_search:
            body["tools"] = [{"google_search": {}}]
        elif use_schema:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = RESPONSE_SCHEMA

        last = None
        for attempt in range(1, self.retries + 1):
            self.limiter.wait()
            try:
                r = self.session.post(self.url, data=json.dumps(body),
                                      timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                self.calls += 1

                cands = data.get("candidates") or []
                if not cands:
                    fb = data.get("promptFeedback", {})
                    raise RuntimeError(f"no candidates (feedback={fb})")

                cand = cands[0]
                parts = (cand.get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts).strip()

                if not text and cand.get("finishReason") == "MAX_TOKENS":
                    raise RuntimeError("truncated at MAX_TOKENS")
                if not text:
                    raise RuntimeError(f"empty text (finish={cand.get('finishReason')})")

                # attach grounding URLs when present
                gm = cand.get("groundingMetadata") or {}
                chunks = gm.get("groundingChunks") or []
                urls = []
                for ch in chunks:
                    uri = ((ch.get("web") or {}).get("uri") or "")
                    if uri and uri not in urls:
                        urls.append(uri)
                if urls:
                    text += "\n<<<GROUNDING_URLS>>>" + " | ".join(urls[:10])
                return text
            except Exception as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 30))
        self.failures += 1
        print(f"  ! Gemini call failed: {last}", file=sys.stderr)
        return None

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        grounding = ""
        if "<<<GROUNDING_URLS>>>" in text:
            text, grounding = text.split("<<<GROUNDING_URLS>>>", 1)
        body = text.strip()
        body = re.sub(r"^```(?:json)?\s*", "", body)
        body = re.sub(r"\s*```$", "", body).strip()
        try:
            obj = json.loads(body)
        except Exception:
            m = re.search(r"\{.*\}", body, re.DOTALL)
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return None
        if isinstance(obj, list):
            obj = obj[0] if obj else {}
        if not isinstance(obj, dict):
            return None
        if grounding.strip():
            obj["_grounding_urls"] = grounding.strip()
        return obj

    # ── prompt construction ──────────────────────────────────────────────────
    @staticmethod
    def build_record_text(row: Dict[str, Any], max_chars: int = 14000) -> str:
        lines: List[str] = []
        for f in CONTEXT_FIELDS:
            v = clean(row.get(f, ""))
            if v:
                lines.append(f"{f}: {v[:2500]}")

        extras: List[str] = []
        for k, v in row.items():
            if k in CONTEXT_FIELDS or k.startswith("ai_"):
                continue
            kl = k.lower()
            if any(h in kl for h in RESULT_HINTS):
                vv = clean(v)
                if vv:
                    extras.append(f"{k}: {vv[:1500]}")

        text = "\n".join(lines)
        if extras:
            text += "\n\n--- additional registry fields ---\n" + "\n".join(extras)
        return text[:max_chars]

    @staticmethod
    def has_result_text(row: Dict[str, Any]) -> bool:
        """Cheap pre-check: does this row contain any results text at all?"""
        NEGATIVE = {"", "no", "n", "none", "false", "0", "nan", "not available",
                    "no results available", "na", "n/a", "-"}

        if clean(row.get("findings", "")).lower() not in NEGATIVE:
            return True
        if str(row.get("results_available", "")).strip().lower() in (
                "yes", "y", "true", "1"):
            return True
        for k, v in row.items():
            kl = k.lower()
            if k.startswith("ai_") or kl in ("results_available", "findings"):
                continue
            if ("result" in kl or "finding" in kl or "conclusion" in kl) \
                    and clean(v).lower() not in NEGATIVE:
                return True
        return False

    # ── per-row work ─────────────────────────────────────────────────────────
    def summarize_row(self, row: Dict[str, Any]) -> Dict[str, str]:
        out = {c: "" for c in AI_COLUMNS}
        out["ai_model"] = self.model

        record = self.build_record_text(row)
        if not record.strip():
            out["ai_key_findings"] = "No registry data available to summarise"
            out["ai_evidence_basis"] = "empty record"
            out["ai_has_registry_results"] = "No"
            return out

        prompt = ("Extract the findings from this clinical trial registry "
                  "record.\n\n=== RECORD START ===\n" + record +
                  "\n=== RECORD END ===")
        ck = self._key(record, "extract")

        with self._cache_lock:
            cached = self._cache.get(ck)
        if cached is not None:
            self.cached_hits += 1
            obj = cached
        else:
            text = self._call(prompt, SYSTEM_RULES,
                              use_schema=True, use_search=False)
            obj = self._parse_json(text) if text else None
            if obj is None:
                out["ai_key_findings"] = "[Gemini extraction failed]"
                out["ai_evidence_basis"] = "API error or unparseable response"
                return out
            with self._cache_lock:
                self._cache[ck] = obj

        out["ai_has_registry_results"] = clean(obj.get("has_registry_results"))
        out["ai_key_findings"] = clean(obj.get("key_findings"))
        out["ai_primary_result"] = clean(obj.get("primary_result"))
        out["ai_secondary_results"] = clean(obj.get("secondary_results"))
        out["ai_safety_findings"] = clean(obj.get("safety_findings"))
        out["ai_conclusion"] = clean(obj.get("conclusion"))
        out["ai_summary"] = clean(obj.get("summary"))
        out["ai_evidence_basis"] = clean(obj.get("evidence_basis"))

        # optional web-grounded pass for trials with no registry results
        if self.grounded and out["ai_has_registry_results"] != "Yes":
            self._grounded_pass(row, out)
        return out

    def _grounded_pass(self, row: Dict[str, Any], out: Dict[str, str]):
        ident = " / ".join(x for x in [clean(row.get("trial_id")),
                                       clean(row.get("secondary_ids"))] if x)
        prompt = (
            f"Trial registration ID(s): {ident}\n"
            f"Registry: {clean(row.get('source'))}\n"
            f"Title: {clean(row.get('title'))}\n"
            f"Sponsor: {clean(row.get('sponsor'))}\n"
            f"Phase: {clean(row.get('phase'))}\n"
            f"Condition: {clean(row.get('conditions'))}\n"
            f"Intervention: {clean(row.get('interventions'))}\n"
            f"Registry URL: {clean(row.get('url'))}\n\n"
            "Search for published results for this exact trial and report them "
            "as JSON with keys published_findings and sources."
        )
        ck = self._key(prompt, "grounded")
        with self._cache_lock:
            cached = self._cache.get(ck)
        if cached is not None:
            self.cached_hits += 1
            obj = cached
        else:
            text = self._call(prompt, GROUNDED_RULES,
                              use_schema=False, use_search=True)
            obj = self._parse_json(text) if text else None
            if obj is None:
                out["ai_published_findings"] = "[grounded search failed]"
                return
            with self._cache_lock:
                self._cache[ck] = obj

        out["ai_published_findings"] = clean(obj.get("published_findings"))
        urls: List[str] = []
        for blob in (clean(obj.get("sources")), clean(obj.get("_grounding_urls"))):
            for u in re.split(r"[\s|,;]+", blob):
                u = u.strip().rstrip(".,)")
                if u and u not in urls:
                    urls.append(u)
        out["ai_sources"] = " | ".join(urls)

    # ── batch ────────────────────────────────────────────────────────────────
    def summarize_rows(self, rows: List[Dict[str, Any]], workers: int = 4,
                       only_with_results: bool = False,
                       progress_every: int = 10) -> List[Dict[str, Any]]:
        targets = list(range(len(rows)))
        if only_with_results:
            targets = [i for i in targets if self.has_result_text(rows[i])]
            skipped = len(rows) - len(targets)
            for i in range(len(rows)):
                if i not in set(targets):
                    rows[i].update({c: "" for c in AI_COLUMNS})
                    rows[i]["ai_key_findings"] = \
                        "No results reported in registry record"
                    rows[i]["ai_has_registry_results"] = "No"
                    rows[i]["ai_evidence_basis"] = \
                        "skipped: no results text in record (--only-with-results)"
                    rows[i]["ai_model"] = self.model
            print(f"  skipping {skipped} row(s) with no results text",
                  file=sys.stderr)

        print(f"  summarising {len(targets)} trial(s) with {self.model} "
              f"({workers} workers)...", file=sys.stderr)

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self.summarize_row, rows[i]): i
                       for i in targets}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    rows[i].update(fut.result())
                except Exception as exc:
                    print(f"  ! row {i} failed: {exc}", file=sys.stderr)
                    rows[i].update({c: "" for c in AI_COLUMNS})
                    rows[i]["ai_key_findings"] = "[error]"
                done += 1
                if done % progress_every == 0:
                    print(f"  ...{done}/{len(targets)}", file=sys.stderr)

        self.save_cache()
        print(f"  Gemini: {self.calls} API call(s), {self.cached_hits} cache hit(s), "
              f"{self.failures} failure(s)", file=sys.stderr)
        return rows


# ── standalone CLI: summarise an existing Excel ───────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Add Gemini 2.5 Flash key-findings columns to a trials Excel.")
    ap.add_argument("excel", help="input .xlsx produced by fetch_all_trials.py")
    ap.add_argument("--out", default=None, help="output .xlsx (default: *_summarized.xlsx)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=int, default=60, help="requests per minute cap")
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="Gemini thinking tokens (0 = off, cheapest)")
    ap.add_argument("--grounded", action="store_true",
                    help="also Google-Search for published results when the "
                         "registry record has none")
    ap.add_argument("--only-with-results", action="store_true",
                    help="only call Gemini for rows that contain results text")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_excel(args.excel).fillna("")
    rows = df.to_dict("records")
    print(f"loaded {len(rows)} row(s) from {args.excel}", file=sys.stderr)

    try:
        sm = GeminiSummarizer(model=args.model, rpm=args.rpm,
                              thinking_budget=args.thinking_budget,
                              grounded=args.grounded,
                              cache_path=None if args.no_cache
                              else ".gemini_cache.json")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rows = sm.summarize_rows(rows, workers=args.workers,
                             only_with_results=args.only_with_results)

    out = args.out or re.sub(r"\.xlsx$", "", args.excel) + "_summarized.xlsx"
    cols = [c for c in UNIFIED_COLUMNS if c in rows[0]] + AI_COLUMNS
    write_excel(rows, out, cols, sheet_name="Trials + Findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
