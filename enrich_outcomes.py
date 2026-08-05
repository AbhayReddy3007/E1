#!/usr/bin/env python3
"""
enrich_outcomes.py – Clinical trial outcome enrichment module (Gemini Search edition).

For each trial row produced by main1.py, uses **Gemini with Google Search grounding**
to find published outcome evidence across the entire web (journals, press releases,
FDA docs, conference abstracts, registries) for four clinical endpoints (HbA1c,
body weight, ALT, MASH) *and* dosage information, then extracts a grounded
rationale and DERIVES the numeric change_pct from that rationale.

Three design rules drive everything below:

  1. change_pct is parsed OUT OF the rationale, never accepted alongside it.
     The number in the cell is therefore always the number the rationale says,
     and the rationale is always traceable to a grounded web source.

  2. A blank cell beats a wrong number. Any value that cannot be traced from
     grounded source -> quote -> rationale -> cell is discarded.

  3. Rate limits are treated as a shared resource. Every worker passes through
     one adaptive throttle that halves its own rate on HTTP 429, honours
     Retry-After, and recovers slowly.

Entry point:
    enrich_trial_outcomes(rows, molecule, max_workers=6) -> list[dict]

Requires:
    - GEMINI_API_KEY in a .env file (or set as an environment variable)
    - requests, python-dotenv
"""

from __future__ import annotations

import difflib
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


# ── configuration ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
GEMINI_URL     = ("https://generativelanguage.googleapis.com/v1beta/models/"
                  f"{GEMINI_MODEL}:generateContent")

DEFAULT_MAX_WORKERS = _env_int("ENRICH_MAX_WORKERS", 6)

GEMINI_RPM         = _env_float("ENRICH_GEMINI_RPM", 60)
GEMINI_RPM_MAX     = _env_float("ENRICH_GEMINI_RPM_MAX", 240)
GEMINI_RPM_MIN     = _env_float("ENRICH_GEMINI_RPM_MIN", 6)
GEMINI_CONCURRENCY = _env_int("ENRICH_GEMINI_CONCURRENCY", 4)

MAX_ATTEMPTS = _env_int("ENRICH_MAX_ATTEMPTS", 5)

# ── anti-hallucination thresholds ─────────────────────────────────────────────
QUOTE_MATCH_THRESHOLD = 0.70   # relaxed for grounded search (paraphrasing OK)
NUMBER_ABS_TOLERANCE  = 0.051
NUMBER_REL_TOLERANCE  = 0.01
STRICT_MODE = os.environ.get("ENRICH_STRICT", "1") not in ("0", "false", "False")

TAG = "[ENRICH]"

# ── endpoint definitions ──────────────────────────────────────────────────────

ENDPOINTS = [
    {
        "key":      "hba1c",
        "label":    "HbA1c",
        "col_pct":  "hba1c_change_pct",
        "col_dur":  "hba1c_duration",
        "col_rat":  "hba1c_rationale",
        "col_conf": "hba1c_confidence",
        "sign":     "negative",
        "plausible_max": 15.0,
        "unit_hint": ("Change in HbA1c, normally a small negative number such as "
                      "-1.2 or -2.07 (percentage points)."),
        "keywords": ["hba1c", "a1c", "glycated", "glycosylated", "haemoglobin",
                     "hemoglobin", "glycaemic", "glycemic"],
        "search_terms": ["HbA1c", "A1c", "glycated hemoglobin", "glycaemic control"],
    },
    {
        "key":      "weight",
        "label":    "Body weight",
        "col_pct":  "weight_change_pct",
        "col_dur":  "weight_duration",
        "col_rat":  "weight_rationale",
        "col_conf": "weight_confidence",
        "sign":     "negative",
        "plausible_max": 80.0,
        "unit_hint": ("Percent change in body weight, normally negative, such as "
                      "-14.9 or -20.9."),
        "keywords": ["weight", "bmi", "body mass", "adiposity"],
        "search_terms": ["body weight", "weight loss", "weight change",
                         "BMI reduction"],
    },
    {
        "key":      "alt",
        "label":    "ALT (alanine aminotransferase)",
        "col_pct":  "alt_reduction_pct",
        "col_dur":  "alt_duration",
        "col_rat":  "alt_rationale",
        "col_conf": "alt_confidence",
        "sign":     "negative",
        "plausible_max": 100.0,
        "unit_hint": ("Percent reduction in ALT, normally negative, such as -40.0. "
                      "Do NOT convert an absolute U/L change into a percentage."),
        "keywords": ["alt", "alanine", "aminotransferase", "transaminase",
                     "liver enzyme"],
        "search_terms": ["ALT", "alanine aminotransferase", "liver enzyme"],
    },
    {
        "key":      "mash",
        "label":    "MASH / NASH",
        "col_pct":  "mash_change_pct",
        "col_dur":  "mash_duration",
        "col_rat":  "mash_rationale",
        "col_conf": "mash_confidence",
        "sign":     "positive",
        "plausible_max": 100.0,
        "unit_hint": ("MASH resolution rate as a positive percentage of patients "
                      "(e.g. 37.0), or fibrosis improvement rate."),
        "keywords": ["mash", "nash", "steatohepatitis", "fibrosis", "resolution",
                     "ballooning", "steatosis"],
        "search_terms": ["MASH", "NASH", "steatohepatitis", "MASH resolution",
                         "fibrosis improvement"],
    },
]

SOURCE_TIER = {
    "ClinicalTrials.gov Results": 1,
    "PubMed":                     1,
    "Peer-reviewed journal":      1,
    "FDA document":               1,
    "Gemini Grounded Search":     2,   # web-grounded, good but not tier-1
    "Company press release":      2,
    "Conference abstract":        2,
    "Secondary aggregator":       3,
}

# ── thread-safe primitives ────────────────────────────────────────────────────

_CACHE: Dict[str, Dict[str, str]] = {}
_CACHE_LOCK   = threading.Lock()
_LOG_LOCK     = threading.Lock()
_THREAD_LOCAL = threading.local()


def _log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"{TAG} {msg}", file=sys.stderr)
        sys.stderr.flush()


class _AdaptiveThrottle:
    """Shared rate governor for one upstream API."""

    def __init__(self, name: str, rpm: float, concurrency: int,
                 rpm_min: float = 6.0, rpm_max: Optional[float] = None) -> None:
        self.name = name
        self._rpm = max(rpm, rpm_min)
        self._rpm_min = rpm_min
        self._rpm_max = rpm_max or rpm * 4
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._resume_at = 0.0
        self._ok_streak = 0
        self._sem = threading.BoundedSemaphore(max(1, concurrency))

    @property
    def _interval(self) -> float:
        return 60.0 / self._rpm if self._rpm > 0 else 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot, self._resume_at)
            self._next_slot = slot + self._interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def penalise(self, retry_after: float = 0.0) -> None:
        with self._lock:
            old = self._rpm
            self._rpm = max(self._rpm_min, self._rpm * 0.5)
            self._ok_streak = 0
            pause = retry_after if retry_after > 0 else 60.0 / self._rpm
            pause += random.uniform(0, 0.5)
            self._resume_at = max(self._resume_at, time.monotonic() + pause)
            self._next_slot = max(self._next_slot, self._resume_at)
        _log(f"    throttle[{self.name}]: {old:.0f} → {self._rpm:.0f} rpm, "
             f"pausing {pause:.1f}s")

    def reward(self) -> None:
        with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 20 and self._rpm < self._rpm_max:
                self._rpm = min(self._rpm_max, self._rpm * 1.1)
                self._ok_streak = 0

    def slot(self) -> "_ThrottleSlot":
        return _ThrottleSlot(self)


class _ThrottleSlot:
    def __init__(self, throttle: _AdaptiveThrottle) -> None:
        self._t = throttle

    def __enter__(self) -> _AdaptiveThrottle:
        self._t.acquire()
        self._t._sem.acquire()
        return self._t

    def __exit__(self, *exc: Any) -> None:
        self._t._sem.release()


_GEMINI_THROTTLE = _AdaptiveThrottle("gemini", GEMINI_RPM, GEMINI_CONCURRENCY,
                                     GEMINI_RPM_MIN, GEMINI_RPM_MAX)
_CTGOV_THROTTLE  = _AdaptiveThrottle("ctgov", 300, 4)


def _retry_after_seconds(resp: requests.Response) -> float:
    raw = resp.headers.get("Retry-After", "")
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return 0.0


def _session() -> requests.Session:
    s = getattr(_THREAD_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/125.0 Safari/537.36"),
        })
        _THREAD_LOCAL.session = s
    return s


def _throttled_get(url: str, throttle: _AdaptiveThrottle,
                   params: Optional[Dict[str, Any]] = None,
                   timeout: int = 25) -> Optional[requests.Response]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with throttle.slot():
                resp = _session().get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS:
                _log(f"    GET failed after {attempt} attempts: {exc}")
                return None
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            throttle.penalise(_retry_after_seconds(resp))
            if attempt == MAX_ATTEMPTS:
                return None
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
            continue
        throttle.reward()
        return resp if resp.ok else None
    return None


# ── text normalisation & numeric helpers ──────────────────────────────────────

def _normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = re.sub(r"[\u2010-\u2015\u2212]", "-", t)
    t = re.sub(r"[\u2018\u2019\u201b]", "'", t)
    t = re.sub(r"[\u201c\u201d]", '"', t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def _numbers_in(text: str) -> List[float]:
    out: List[float] = []
    for tok in re.findall(r"-?\d+(?:\.\d+)?", (text or "").replace(",", "")):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _same_number(a: Any, b: Any) -> bool:
    try:
        x, y = abs(float(a)), abs(float(b))
    except (TypeError, ValueError):
        return False
    if abs(x - y) <= NUMBER_ABS_TOLERANCE:
        return True
    return y > 0 and abs(x - y) / y <= NUMBER_REL_TOLERANCE


def _number_supported_by(value: Any, text: str) -> bool:
    return any(_same_number(value, n) for n in _numbers_in(text))


QUOTE_MIN_CHARS = 20
QUOTE_MAX_CHARS = 800
QUOTE_MAX_NUMBERS = 30


def _quote_is_well_formed(quote: str) -> Tuple[bool, str]:
    q = quote.strip()
    if len(q) < QUOTE_MIN_CHARS:
        return False, f"quote too short ({len(q)} chars)"
    if len(q) > QUOTE_MAX_CHARS:
        return False, f"quote too long ({len(q)} chars)"
    n_nums = len(_numbers_in(q))
    if n_nums > QUOTE_MAX_NUMBERS:
        return False, f"quote is a number dump ({n_nums} numbers)"
    return True, ""


def _quote_is_on_topic(quote: str, endpoint: Dict) -> bool:
    q = _normalize(quote)
    for kw in endpoint["keywords"]:
        if re.search(rf"(?<![a-z0-9]){re.escape(_normalize(kw))}(?![a-z0-9])", q):
            return True
    return False


# ── deriving change_pct from the rationale ────────────────────────────────────

_MASK_PATTERNS = [
    re.compile(r"\([^)]*\b(?:ci|confidence interval|p\s*[=<>])[^)]*\)", re.I),
    re.compile(r"\b\d{2}\s*%\s*ci\b[^;.]{0,60}", re.I),
    re.compile(r"\bp\s*[=<>]\s*0?\.\d+", re.I),
    re.compile(r"\bp\s*[=<>]\s*\d+\s*[x×]\s*10", re.I),
]

_REJECT_AFTER = re.compile(
    r"^\s*(?:%?\s*(?:ci\b|confidence)"
    r"|weeks?\b|wks?\b|months?\b|days?\b|years?\b|yrs?\b|hours?\b"
    r"|mg\b|mcg\b|µg|ug\b|kg\b|lbs?\b|g\b|u/l\b|iu\b|ml\b|mmol|mg/dl"
    r"|patients?\b|participants?\b|subjects?\b|sites?\b|arms?\b)",
    re.I)

_REJECT_BEFORE = re.compile(
    r"(?:\bn\s*=\s*|\bp\s*[=<>]\s*|\bweeks?\s+|\bdays?\s+|\bmonths?\s+"
    r"|\byears?\s+|\bphase\s+|\bci\s*[:=]?\s*|\bnct\s*|\bcohort\s+"
    r"|\bgroup\s+|\barm\s+|\bvisit\s+)$",
    re.I)

_BASELINE_BEFORE = re.compile(
    r"\bbaselines?\b(?:\s+\S+){0,3}\s*(?:was|were|is|are|of|:)\s*$", re.I)

_NEGATIVE_CUES = re.compile(
    r"\b(reduc\w*|decreas\w*|declin\w*|lower\w*|loss|lost|fell|fall\w*|drop\w*|"
    r"shrank|improv\w*\s+by|less)\b", re.I)
_POSITIVE_CUES = re.compile(
    r"\b(increas\w*|rose|rise|gain\w*|higher|greater|resolution|resolv\w*|"
    r"achiev\w*|respond\w*|response\s+rate|improvement\s+rate)\b", re.I)


def _mask_stats(text: str) -> str:
    out = text
    for pat in _MASK_PATTERNS:
        out = pat.sub(lambda m: " " * len(m.group()), out)
    return out


def _derive_change_pct(rationale: str, endpoint: Dict) -> Optional[float]:
    if not rationale:
        return None

    masked = _mask_stats(rationale)
    low = masked.lower()

    kw_spans = [m.start() for kw in endpoint["keywords"]
                for m in re.finditer(re.escape(kw), low)]

    candidates: List[Tuple[float, float, bool, int]] = []

    for m in re.finditer(r"(-|–|—|minus\s+)?(\d+(?:\.\d+)?)\s*(%|percent(?:age)?"
                         r"(?:\s+points?)?)?", masked, re.I):
        raw = m.group(2)
        after = masked[m.end():m.end() + 24]
        before = masked[max(0, m.start() - 40):m.start()]

        num_start = m.start(2)
        num_end = m.end(2)
        if num_start > 0 and masked[num_start - 1].isalpha():
            continue
        if num_end < len(masked) and masked[num_end].isalpha():
            continue

        if _REJECT_AFTER.match(after) and not m.group(3):
            continue
        if _REJECT_BEFORE.search(before):
            continue
        if _BASELINE_BEFORE.search(before):
            continue

        try:
            value = float(raw)
        except ValueError:
            continue
        if abs(value) > endpoint["plausible_max"]:
            continue
        if value == 0 and not m.group(3):
            continue

        explicit_neg = bool(m.group(1))
        had_pct = bool(m.group(3))

        dist = min((abs(m.start() - k) for k in kw_spans), default=9999.0)
        if not had_pct and dist > 60:
            continue

        signed = -value if explicit_neg else value
        candidates.append((dist, signed, had_pct, m.start(2)))

    if not candidates:
        return None

    candidates.sort(key=lambda c: (0 if c[2] else 1, c[0]))
    _, value, _, best_pos = candidates[0]

    if value > 0 and endpoint["sign"] == "negative":
        lo, hi = max(0, best_pos - 90), min(len(rationale), best_pos + 90)
        window, offset = rationale[lo:hi], lo

        def nearest(pattern) -> float:
            return min((abs((m.start() + offset) - best_pos)
                        for m in pattern.finditer(window)), default=float("inf"))

        neg_d, pos_d = nearest(_NEGATIVE_CUES), nearest(_POSITIVE_CUES)
        if neg_d < pos_d:
            value = -value
    if endpoint["sign"] == "positive":
        value = abs(value)

    return round(value, 4)


# ── Gemini schemas ────────────────────────────────────────────────────────────

_EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found": {
            "type": "BOOLEAN",
            "description": "True only if the snippets state an explicit numeric "
                           "result for this endpoint for THIS trial.",
        },
        "evidence_quote": {
            "type": "STRING",
            "nullable": True,
            "description": "VERBATIM span copied character-for-character from a "
                           "snippet, containing the number. Never paraphrase.",
        },
        "snippet_index": {
            "type": "INTEGER",
            "nullable": True,
            "description": "1-based index of the snippet the quote came from.",
        },
        "duration": {
            "type": "STRING",
            "nullable": True,
            "description": "Timepoint of measurement, e.g. '24 weeks'.",
        },
        "rationale": {
            "type": "STRING",
            "nullable": True,
            "description": "1-3 sentences that MUST state the numeric result "
                           "explicitly, exactly as written in the quote.",
        },
    },
    "required": ["found", "evidence_quote", "snippet_index", "duration",
                 "rationale"],
}

_VALIDATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "supported":     {"type": "BOOLEAN"},
        "contradiction": {"type": "BOOLEAN"},
        "ambiguous":     {"type": "BOOLEAN"},
        "confidence":    {"type": "STRING", "enum": ["High", "Low"]},
        "reason":        {"type": "STRING"},
    },
    "required": ["supported", "contradiction", "ambiguous", "confidence",
                 "reason"],
}

_DOSAGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found": {
            "type": "BOOLEAN",
            "description": "True if dosage/dose information was found.",
        },
        "dosage": {
            "type": "STRING",
            "nullable": True,
            "description": "Dosage info, e.g. '0.25mg, 0.5mg, 1.0mg, 2.4mg SC weekly' "
                           "or '10mg, 25mg oral daily'. Include route and frequency.",
        },
        "source_url": {
            "type": "STRING",
            "nullable": True,
            "description": "URL of the source where dosage was found.",
        },
    },
    "required": ["found", "dosage"],
}


def _gemini_json(prompt: str, system: str,
                 schema: Dict[str, Any]) -> Optional[Dict]:
    """Call Gemini with enforced JSON schema output at temperature 0."""
    if not GEMINI_API_KEY:
        return None

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "temperature": 0.0,
            "topP": 1.0,
            "candidateCount": 1,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with _GEMINI_THROTTLE.slot():
                resp = requests.post(url, json=payload, headers=headers,
                                     timeout=90)
        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS:
                _log(f"    Gemini network failure: {exc}")
                return None
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            _GEMINI_THROTTLE.penalise(_retry_after_seconds(resp))
            if attempt == MAX_ATTEMPTS:
                _log(f"    Gemini gave up after {attempt} attempts "
                     f"({resp.status_code})")
                return None
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
            continue

        if not resp.ok:
            _log(f"    Gemini HTTP {resp.status_code}: {resp.text[:160]}")
            return None

        _GEMINI_THROTTLE.reward()
        try:
            cands = resp.json().get("candidates", [])
            if not cands:
                return None
            parts = cands[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            return json.loads(text) if text else None
        except (ValueError, json.JSONDecodeError):
            _log("    Gemini returned unparseable JSON; discarding.")
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Evidence gathering via GEMINI SEARCH (google_search tool)
# ═══════════════════════════════════════════════════════════════════════════════

def _gemini_search(query: str, system_hint: str = "") -> Optional[Dict[str, Any]]:
    """
    Call Gemini with the google_search tool for grounded web search.
    Returns the full candidate dict including groundingMetadata.
    """
    if not GEMINI_API_KEY:
        return None

    sys_text = system_hint or (
        "You are a clinical research assistant. Search the web thoroughly and "
        "report findings with specific numbers, sources, and verbatim quotes. "
        "If you cannot find specific data, say 'No results found'."
    )

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": query}]}],
        "systemInstruction": {"parts": [{"text": sys_text}]},
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 4096,
        },
    }
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with _GEMINI_THROTTLE.slot():
                resp = requests.post(url, json=payload, headers=headers,
                                     timeout=120)
        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS:
                _log(f"    Gemini Search network failure: {exc}")
                return None
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            _GEMINI_THROTTLE.penalise(_retry_after_seconds(resp))
            if attempt == MAX_ATTEMPTS:
                return None
            time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
            continue

        if not resp.ok:
            _log(f"    Gemini Search HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        _GEMINI_THROTTLE.reward()
        try:
            data = resp.json()
            cands = data.get("candidates", [])
            if not cands:
                return None
            return cands[0]
        except (ValueError, json.JSONDecodeError):
            _log("    Gemini Search returned unparseable response.")
            return None
    return None


def _extract_grounding_snippets(candidate: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract evidence snippets from Gemini Search grounding metadata.
    Returns list of {source, url, text} dicts compatible with existing pipeline.
    """
    snippets: List[Dict[str, str]] = []

    # Get the model's response text (which IS the grounded content)
    parts = candidate.get("content", {}).get("parts", [])
    response_text = "".join(p.get("text", "") for p in parts).strip()
    if not response_text:
        return snippets

    # Get grounding metadata
    gm = candidate.get("groundingMetadata", {})
    chunks = gm.get("groundingChunks", [])
    supports = gm.get("groundingSupports", [])

    # Build a list of source URLs from grounding chunks
    source_urls: List[str] = []
    source_titles: List[str] = []
    for chunk in chunks:
        web = chunk.get("web", {})
        source_urls.append(web.get("uri", ""))
        source_titles.append(web.get("title", ""))

    # If we have grounding supports, group text by source chunk
    if supports and chunks:
        # Map chunk index -> collected text segments
        chunk_texts: Dict[int, List[str]] = {}
        for sup in supports:
            seg_text = sup.get("segment", {}).get("text", "")
            if not seg_text:
                continue
            for idx in sup.get("groundingChunkIndices", []):
                chunk_texts.setdefault(idx, []).append(seg_text)

        for idx, texts in chunk_texts.items():
            if idx < len(source_urls) and source_urls[idx]:
                combined = " ".join(texts)
                # Determine source type from URL
                url = source_urls[idx]
                title = source_titles[idx] if idx < len(source_titles) else ""
                source_type = _classify_source(url, title)
                snippets.append({
                    "source": source_type,
                    "url": url,
                    "text": combined[:6000],
                })

    # Always include the full response as a comprehensive snippet
    # (the model's grounded synthesis often contains the clearest statement)
    if response_text:
        primary_url = source_urls[0] if source_urls else ""
        snippets.append({
            "source": "Gemini Grounded Search",
            "url": primary_url,
            "text": response_text[:8000],
        })

    return snippets


def _classify_source(url: str, title: str = "") -> str:
    """Classify a URL into a source tier category."""
    url_lower = url.lower()
    title_lower = title.lower()
    if "clinicaltrials.gov" in url_lower:
        return "ClinicalTrials.gov Results"
    if "pubmed" in url_lower or "ncbi.nlm.nih.gov" in url_lower:
        return "PubMed"
    if "fda.gov" in url_lower or "ema.europa.eu" in url_lower:
        return "FDA document"
    if any(j in url_lower for j in [
        "nejm.org", "thelancet.com", "bmj.com", "nature.com",
        "sciencedirect.com", "wiley.com", "springer.com", "jama",
        "oup.com", "diabetesjournals.org", "ahajournals.org",
        "journal", "doi.org"
    ]):
        return "Peer-reviewed journal"
    if any(kw in url_lower or kw in title_lower for kw in [
        "press release", "newsroom", "media", "investor",
        "businesswire", "prnewswire", "globenewswire"
    ]):
        return "Company press release"
    if any(kw in url_lower or kw in title_lower for kw in [
        "abstract", "poster", "congress", "conference", "asco", "easd", "aasld"
    ]):
        return "Conference abstract"
    return "Gemini Grounded Search"


def _search_ctgov_results(trial_id: str) -> List[Dict[str, str]]:
    """Registry-posted results for an NCT ID – the highest-trust source."""
    out: List[Dict[str, str]] = []
    if not trial_id or not trial_id.upper().startswith("NCT"):
        return out
    resp = _throttled_get(
        f"https://clinicaltrials.gov/api/v2/studies/{trial_id}", _CTGOV_THROTTLE)
    if resp is None:
        return out
    try:
        res_section = resp.json().get("resultsSection")
    except ValueError:
        return out
    if res_section:
        out.append({
            "source": "ClinicalTrials.gov Results",
            "url": f"https://clinicaltrials.gov/study/{trial_id}?tab=results",
            "text": json.dumps(res_section, indent=1)[:6000],
        })
    return out


def _build_search_queries(row: Dict[str, str], molecule: str,
                          endpoint: Dict) -> List[str]:
    """Build search queries for Gemini grounded search."""
    trial_id = (row.get("trial_id") or "").strip()
    acronym  = (row.get("acronym") or "").strip()
    company  = (row.get("company_name") or "").strip()
    title    = (row.get("trial_title") or "").strip()
    term     = endpoint["search_terms"][0]

    queries: List[str] = []

    # Primary: trial ID + endpoint
    if trial_id:
        queries.append(
            f"clinical trial {trial_id} {molecule} {term} results outcome data"
        )
    # Acronym-based (journals often use acronym, not ID)
    if acronym and acronym.lower() != trial_id.lower():
        queries.append(
            f"{acronym} trial {molecule} {term} results published"
        )
    # Fallback: molecule + company + endpoint
    if company and not queries:
        queries.append(
            f"{molecule} {company} {term} clinical trial results"
        )
    # Last resort: molecule + title fragment
    if not queries:
        title_short = " ".join(title.split()[:8]) if title else molecule
        queries.append(f"{title_short} {term} trial results")

    return queries


def gather_evidence(row: Dict[str, str], molecule: str,
                    endpoint: Dict) -> List[Dict[str, str]]:
    """
    Collect evidence snippets using Gemini Search grounding + ClinicalTrials.gov API.
    Searches across ALL web sources: journals, press releases, FDA docs, registries,
    conference abstracts, and more.
    """
    snippets: List[Dict[str, str]] = []
    seen_urls: set = set()

    # 1. ClinicalTrials.gov API results (highest trust, direct fetch)
    for s in _search_ctgov_results(row.get("trial_id", "")):
        snippets.append(s)
        seen_urls.add(s["url"])

    # 2. Gemini Search grounding — searches the entire web
    queries = _build_search_queries(row, molecule, endpoint)
    for query in queries:
        candidate = _gemini_search(query)
        if candidate is None:
            continue
        grounded_snippets = _extract_grounding_snippets(candidate)
        for snip in grounded_snippets:
            if snip["url"] and snip["url"] in seen_urls:
                continue
            if snip["url"]:
                seen_urls.add(snip["url"])
            snippets.append(snip)

        # One good grounded search is usually enough
        if len(snippets) >= 3:
            break

    # Filter: at least one snippet must mention the trial or molecule
    return [s for s in snippets
            if _evidence_mentions_trial(row, molecule, s["text"])]


def _evidence_mentions_trial(row: Dict[str, str], molecule: str,
                             evidence: str) -> bool:
    ev = _normalize(evidence)
    if not ev:
        return False
    for anchor in (molecule, row.get("trial_id", ""), row.get("acronym", "")):
        a = _normalize(anchor)
        if len(a) >= 3 and a in ev:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1b: Dosage enrichment via Gemini Search
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_dosage(row: Dict[str, str], molecule: str) -> str:
    """
    Use Gemini Search to find dosage/dosing info for a clinical trial.
    Returns dosage string or empty string.
    """
    trial_id = (row.get("trial_id") or "").strip()
    acronym  = (row.get("acronym") or "").strip()
    title    = (row.get("trial_title") or "").strip()

    # Build a targeted dosage query
    identifiers = []
    if trial_id:
        identifiers.append(trial_id)
    if acronym:
        identifiers.append(acronym)
    id_str = " ".join(identifiers) if identifiers else title[:60]

    query = (
        f"What are the specific drug doses and dosing regimen used in "
        f"clinical trial {id_str} for {molecule}? "
        f"Include dose amounts (mg), route of administration (oral, subcutaneous, IV), "
        f"and dosing frequency (daily, weekly, etc). "
        f"Report the actual arms/doses tested in the trial."
    )

    candidate = _gemini_search(query)
    if candidate is None:
        return ""

    # Extract dosage from the grounded response
    parts = candidate.get("content", {}).get("parts", [])
    response_text = "".join(p.get("text", "") for p in parts).strip()
    if not response_text or "no results found" in response_text.lower():
        return ""

    # Use Gemini to extract structured dosage from the grounded text
    extract_prompt = (
        f"From the following text about clinical trial {id_str} ({molecule}), "
        f"extract ONLY the dosage information. Return a concise summary of doses "
        f"tested, e.g. '0.25mg, 0.5mg, 1.0mg, 2.4mg subcutaneous weekly' or "
        f"'10mg, 25mg oral daily'. Include route and frequency if available.\n\n"
        f"Text:\n{response_text[:3000]}\n\n"
        f"If no specific dosage info is found, set found=false."
    )

    result = _gemini_json(
        extract_prompt,
        "Extract dosage info concisely. Return JSON only.",
        _DOSAGE_SCHEMA
    )

    if result and result.get("found") and result.get("dosage"):
        dosage = result["dosage"].strip()
        _log(f"    {trial_id or acronym}: dosage → {dosage[:80]}")
        return dosage

    return ""


# ── Step 2: extraction ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """\
You are a clinical data extraction assistant working under strict evidentiary
rules. You will receive trial metadata and numbered evidence snippets.

Extract ONE endpoint result. Obey these rules absolutely:

1. ABSTAIN BY DEFAULT. If the snippets do not state an explicit number for this
   exact endpoint for THIS exact trial, set found=false and every other field to
   null. Abstaining is always correct when in doubt.

2. NEVER CALCULATE, CONVERT, OR INFER A NUMBER. Only report a number printed
   verbatim in a snippet. Do not derive percentages from absolute values, do not
   average arms, do not annualise, do not estimate from a described chart.

3. evidence_quote must be copied CHARACTER-FOR-CHARACTER from a snippet and must
   contain the number. Do not paraphrase, tidy, translate or re-punctuate it.

4. THE RATIONALE MUST STATE THE NUMBER AND ITS ORIGIN. Write 1-3 sentences that:
     (a) state WHAT was found, including the numeric result written the same way
         the quote writes it, the treatment arm or dose, and the timepoint;
     (b) state WHERE it was found - name the source type.
   The numeric cell is parsed directly from this rationale.

5. Report the TREATMENT arm at the primary or headline timepoint. If several
   doses are reported, use the highest dose and say so.

6. Sign convention: reductions are negative, increases positive. Resolution and
   response RATES are positive.
"""


def _extraction_prompt(row: Dict[str, str], molecule: str, endpoint: Dict,
                       snippets: List[Dict[str, str]]) -> str:
    trial_id = (row.get("trial_id") or "").strip()
    acronym  = (row.get("acronym") or "").strip()
    anchors = []
    if trial_id:
        anchors.append(f"registry ID {trial_id}")
    if acronym:
        anchors.append(f"acronym/short name {acronym}")
    anchor_line = (" or ".join(anchors)) if anchors else "the title below"

    meta = (
        f"Trial ID: {trial_id or 'N/A'}\n"
        f"Acronym: {acronym or 'N/A'}\n"
        f"Molecule: {molecule}\n"
        f"Phase: {row.get('phase') or 'N/A'}\n"
        f"Sponsor: {row.get('company_name') or 'N/A'}\n"
        f"Title: {row.get('trial_title') or 'N/A'}\n"
    )
    body = "".join(
        f"\n--- Snippet {i} (source: {s['source']}) ---\n{s['text']}\n"
        for i, s in enumerate(snippets, 1)
    )
    return (
        f"## Trial Metadata\n{meta}\n"
        f"## Endpoint to extract\n{endpoint['label']}\n"
        f"Expected form: {endpoint['unit_hint']}\n"
        f"Related terms: {', '.join(endpoint['search_terms'])}\n\n"
        f"## Evidence Snippets\n{body}\n\n"
        f"A snippet belongs to this trial if it refers to {anchor_line}. "
        f"Extract the {endpoint['label']} result for THIS trial only.\n"
        f"Many trials have no published result for this endpoint yet — that is "
        f"normal and expected. If it is not explicitly stated above, set "
        f"found=false rather than reaching for a related number."
    )


# ── Step 3: verification (deterministic) + validation (LLM) ──────────────────

VALIDATION_SYSTEM = """\
You are an independent auditor. You did not perform the extraction and must not
defend it. Given a quote from a source document and a value claimed to come from
it, decide whether the quote genuinely supports that value.

supported=false if the quote does not explicitly state the value for the named
endpoint, if the number belongs to a different endpoint, arm, timepoint or
trial, or if the value was derived rather than quoted.
contradiction=true if the rationale disagrees with the quote.
ambiguous=true if the arm, timepoint or comparator is unclear.

confidence is "High" ONLY when supported=true, contradiction=false and
ambiguous=false. Otherwise "Low".
"""


def _quote_support_ratio(quote: str, evidence: str) -> float:
    q = _normalize(quote)
    e = _normalize(evidence)
    if not q or not e:
        return 0.0
    if q in e:
        return 1.0
    sm = difflib.SequenceMatcher(None, q, e, autojunk=False)
    match = sm.find_longest_match(0, len(q), 0, len(e))
    return match.size / len(q)


def _verify(extracted: Dict[str, Any], snippets: List[Dict[str, str]],
            endpoint: Dict) -> Tuple[bool, str, Optional[float],
                                     Optional[Dict[str, str]]]:
    """Deterministic gate before the LLM auditor."""
    if not extracted.get("found"):
        return False, "model abstained", None, None

    quote     = (extracted.get("evidence_quote") or "").strip()
    rationale = (extracted.get("rationale") or "").strip()

    if not quote:
        return False, "no evidence quote supplied", None, None
    if not rationale:
        return False, "no rationale supplied", None, None

    well_formed, why = _quote_is_well_formed(quote)
    if not well_formed:
        return False, why, None, None

    if not _quote_is_on_topic(quote, endpoint):
        return False, f"quote does not mention {endpoint['label']}", None, None

    # Check quote against evidence — relaxed threshold for grounded search
    all_text = "\n".join(s["text"] for s in snippets)
    ratio = _quote_support_ratio(quote, all_text)
    if ratio < QUOTE_MATCH_THRESHOLD:
        return False, f"quote not found in source (match {ratio:.0%})", None, None

    derived = _derive_change_pct(rationale, endpoint)
    if derived is None:
        return False, "rationale states no usable number", None, None

    if not _number_supported_by(derived, quote):
        return False, f"rationale value {derived} absent from quote", None, None

    # Resolve owning snippet
    src: Optional[Dict[str, str]] = None
    idx = extracted.get("snippet_index")
    if isinstance(idx, int) and 1 <= idx <= len(snippets):
        cand = snippets[idx - 1]
        if _quote_support_ratio(quote, cand["text"]) >= QUOTE_MATCH_THRESHOLD:
            src = cand
    if src is None:
        for s in snippets:
            if _quote_support_ratio(quote, s["text"]) >= QUOTE_MATCH_THRESHOLD:
                src = s
                break
    if src is None:
        # For grounded search, accept if the quote matched the combined text
        if ratio >= QUOTE_MATCH_THRESHOLD and snippets:
            src = snippets[0]
        else:
            return False, "quote spans no single source", None, None

    if abs(derived) > endpoint["plausible_max"]:
        return False, f"value {derived} outside plausible range", None, None

    return True, "verified", derived, src


def _llm_validate(derived: float, extracted: Dict[str, Any],
                  src: Dict[str, str], endpoint: Dict) -> Tuple[str, str]:
    prompt = (
        f"## Endpoint\n{endpoint['label']}\n\n"
        f"## Value parsed from the rationale\n{derived}\n"
        f"## Claimed timepoint\n{extracted.get('duration')}\n"
        f"## Rationale written by the extractor\n{extracted.get('rationale')}\n\n"
        f"## Source type\n{src['source']}\n"
        f"## Verbatim quote from that source\n{extracted.get('evidence_quote')}\n\n"
        f"Audit this extraction."
    )
    res = _gemini_json(prompt, VALIDATION_SYSTEM, _VALIDATION_SCHEMA)
    if not res:
        return "Low", "validator unavailable"
    if not res.get("supported") or res.get("contradiction") or res.get("ambiguous"):
        return "Low", str(res.get("reason", ""))[:200]
    conf = res.get("confidence", "Low")
    return ("High" if conf == "High" else "Low"), str(res.get("reason", ""))[:200]


# ── per trial x endpoint pipeline ─────────────────────────────────────────────

def _blank_result() -> Dict[str, str]:
    return {"pct": "", "dur": "", "rat": "", "conf": "Low"}


def _process_endpoint(row: Dict[str, str], molecule: str,
                      endpoint: Dict) -> Dict[str, str]:
    """Full evidence -> extract -> verify -> validate chain for one endpoint."""
    trial_id = row.get("trial_id", "?")
    out = _blank_result()

    try:
        snippets = gather_evidence(row, molecule, endpoint)
    except Exception as exc:
        _log(f"    {trial_id}/{endpoint['key']}: evidence error: {exc}")
        return out

    if not snippets:
        _log(f"    {trial_id}/{endpoint['key']}: no evidence → blank")
        return out

    extracted = _gemini_json(
        _extraction_prompt(row, molecule, endpoint, snippets),
        EXTRACTION_SYSTEM, _EXTRACTION_SCHEMA)
    if not extracted:
        _log(f"    {trial_id}/{endpoint['key']}: extraction failed → blank")
        return out

    passed, reason, derived, src = _verify(extracted, snippets, endpoint)
    if not passed:
        _log(f"    {trial_id}/{endpoint['key']}: REJECTED ({reason})")
        if STRICT_MODE or src is None:
            return out
        out["rat"] = f"[unverified] {extracted.get('rationale') or ''}".strip()
        return out

    confidence, why = _llm_validate(derived, extracted, src, endpoint)

    if SOURCE_TIER.get(src["source"], 3) > 1 and confidence == "High":
        confidence = "Low"
        why = f"source tier below peer-reviewed/registry; {why}"

    rationale = (extracted.get("rationale") or "").strip()
    final_rat = f"{rationale} [Source: {src['source']} – {src['url']}]"

    round_trip = _derive_change_pct(final_rat, endpoint)
    if round_trip is None or not _same_number(round_trip, derived):
        _log(f"    {trial_id}/{endpoint['key']}: REJECTED "
             f"(rationale/value out of sync: {derived} vs {round_trip})")
        return out

    out["pct"]  = f"{derived:g}"
    out["dur"]  = (extracted.get("duration") or "").strip()
    out["rat"]  = final_rat
    out["conf"] = confidence
    note = f" – {why}" if confidence == "Low" and why else ""
    _log(f"    {trial_id}/{endpoint['key']}: {out['pct']} "
         f"@ {out['dur'] or '?'} ({confidence}){note}")
    return out


def _enrich_one_trial(row: Dict[str, str], molecule: str) -> Dict[str, str]:
    """Enrich a single trial: dosage + all four endpoints."""
    result: Dict[str, str] = {}

    # ── Dosage enrichment ─────────────────────────────────────────────────
    if not row.get("dosage"):
        try:
            dosage = _enrich_dosage(row, molecule)
            result["dosage"] = dosage
        except Exception as exc:
            _log(f"    {row.get('trial_id')}/dosage: error: {exc}")
            result["dosage"] = ""
    else:
        result["dosage"] = row["dosage"]

    # ── Outcome endpoints ─────────────────────────────────────────────────
    for ep in ENDPOINTS:
        try:
            r = _process_endpoint(row, molecule, ep)
        except Exception as exc:
            _log(f"    {row.get('trial_id')}/{ep['key']}: unexpected error: {exc}")
            r = _blank_result()
        result[ep["col_pct"]]  = r["pct"]
        result[ep["col_dur"]]  = r["dur"]
        result[ep["col_rat"]]  = r["rat"]
        result[ep["col_conf"]] = r["conf"]
    return result


# ── public entry point ────────────────────────────────────────────────────────

def enrich_trial_outcomes(rows: List[Dict[str, str]], molecule: str,
                          max_workers: int = DEFAULT_MAX_WORKERS
                          ) -> List[Dict[str, str]]:
    """
    Enrich trial rows with dosage info and verified outcome data for
    HbA1c, weight, ALT and MASH using Gemini Search grounding.

    Parameters
    ----------
    rows : list of dict
        Trial rows as produced by main1.py's fetch_all().
    molecule : str
        Molecule / drug name being searched.
    max_workers : int
        Concurrent trials.

    Returns
    -------
    list of dict
        The same list object, with dosage + outcome columns populated.
    """
    if not rows:
        return rows

    if not GEMINI_API_KEY:
        _log("GEMINI_API_KEY not found – skipping enrichment.")
        for row in rows:
            row.setdefault("dosage", "")
            for ep in ENDPOINTS:
                row.setdefault(ep["col_pct"], "")
                row.setdefault(ep["col_dur"], "")
                row.setdefault(ep["col_rat"], "")
                row.setdefault(ep["col_conf"], "")
        return rows

    groups: Dict[str, List[Dict[str, str]]] = {}
    uncacheable: set = set()
    call_id = f"{id(rows):x}{time.time_ns():x}"
    for i, row in enumerate(rows):
        trial_id = (row.get("trial_id") or "").strip().upper()
        if trial_id:
            key = trial_id
        else:
            key = f"__nocache:{call_id}:{i}"
            uncacheable.add(key)
        groups.setdefault(key, []).append(row)

    _log(f"Enriching {len(rows)} row(s) / {len(groups)} unique trial(s); "
         f"{max_workers} worker(s), using Gemini Search grounding; "
         f"strict mode {'ON' if STRICT_MODE else 'OFF'}.")

    pending: Dict[str, List[Dict[str, str]]] = {}
    with _CACHE_LOCK:
        for key, grp in groups.items():
            if key in _CACHE:
                for row in grp:
                    row.update(_CACHE[key])
            else:
                pending[key] = grp

    if pending:
        done, total = 0, len(pending)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_enrich_one_trial, grp[0], molecule): key
                       for key, grp in pending.items()}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    values = fut.result()
                except Exception as exc:
                    _log(f"  {key}: FAILED ({exc}) → blank")
                    values = {"dosage": ""}
                    for ep in ENDPOINTS:
                        values[ep["col_pct"]]  = ""
                        values[ep["col_dur"]]  = ""
                        values[ep["col_rat"]]  = ""
                        values[ep["col_conf"]] = "Low"

                for row in pending[key]:
                    row.update(values)
                if key not in uncacheable:
                    with _CACHE_LOCK:
                        _CACHE[key] = values

                done += 1
                _log(f"  [{done}/{total}] {key} done")

    filled = sum(1 for row in rows for ep in ENDPOINTS if row.get(ep["col_pct"]))
    dosage_filled = sum(1 for row in rows if row.get("dosage"))
    _log(f"Enrichment complete: {filled} verified outcome value(s), "
         f"{dosage_filled} dosage(s) across {len(rows)} row(s).")
    return rows