#!/usr/bin/env python3
"""
enrich_outcomes.py – Clinical trial outcome enrichment module.

For each trial row produced by main1.py, gathers published outcome evidence for
four clinical endpoints (HbA1c, body weight, ALT, MASH), extracts a grounded
rationale with Gemini 3 Flash Preview, and then DERIVES the numeric change_pct
from that rationale.

Three design rules drive everything below:

  1. change_pct is parsed OUT OF the rationale, never accepted alongside it.
     The number in the cell is therefore always the number the rationale says,
     and the rationale is always traceable to a verbatim source quote.

  2. A blank cell beats a wrong number. Any value that cannot be traced from
     source text -> quote -> rationale -> cell is discarded, not flagged.

  3. Rate limits are treated as a shared resource. Every worker passes through
     one adaptive throttle that halves its own rate on HTTP 429, honours
     Retry-After, and recovers slowly. Concurrency raises throughput without
     raising request rate.

Entry point:
    enrich_trial_outcomes(rows, molecule, max_workers=6) -> list[dict]

Requires:
    - GEMINI_API_KEY in a .env file (or set as an environment variable)
    - requests, python-dotenv

    Create a .env file next to this script:
        GEMINI_API_KEY=your_key_here
        # optional tuning
        GEMINI_MODEL=gemini-2.5-flash-preview-05-20
        ENRICH_MAX_WORKERS=6
        ENRICH_GEMINI_RPM=60
        ENRICH_GEMINI_CONCURRENCY=4
        NCBI_API_KEY=          # optional, raises PubMed limit 3/s -> 10/s
        ENRICH_STRICT=1        # 0 = keep unverified values, marked Low
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
    pass  # falls back to os.environ as-is if python-dotenv is not installed


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
NCBI_API_KEY   = os.environ.get("NCBI_API_KEY", "")

DEFAULT_MAX_WORKERS = _env_int("ENRICH_MAX_WORKERS", 6)

# Requests per minute, and how many may be in flight at once.
# Start conservative: the throttle raises the rate itself when nothing 429s.
GEMINI_RPM         = _env_float("ENRICH_GEMINI_RPM", 60)
GEMINI_RPM_MAX     = _env_float("ENRICH_GEMINI_RPM_MAX", 240)
GEMINI_RPM_MIN     = _env_float("ENRICH_GEMINI_RPM_MIN", 6)
GEMINI_CONCURRENCY = _env_int("ENRICH_GEMINI_CONCURRENCY", 4)

# NCBI allows 3 req/s anonymously, 10 req/s with a key.
NCBI_RPM         = 600 if NCBI_API_KEY else 150
NCBI_CONCURRENCY = 6 if NCBI_API_KEY else 3
CTGOV_RPM        = 300
CTGOV_CONCURRENCY = 4

MAX_ATTEMPTS = _env_int("ENRICH_MAX_ATTEMPTS", 5)

# ── anti-hallucination thresholds ─────────────────────────────────────────────
QUOTE_MATCH_THRESHOLD = 0.85   # how literally a quote must appear in the source
NUMBER_ABS_TOLERANCE  = 0.051  # numeric equality slack, absolute
NUMBER_REL_TOLERANCE  = 0.01   # numeric equality slack, relative
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
        "sign":     "negative",   # reductions expected
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
        "sign":     "positive",   # resolution / response RATES
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
    "ClinicalTrials.gov Results": 1,   # registry-posted
    "PubMed":                     1,   # peer-reviewed
    "Company press release":      2,
    "Conference abstract":        2,
    "Secondary aggregator":       3,   # never High
}

# ── thread-safe primitives ────────────────────────────────────────────────────

_CACHE: Dict[str, Dict[str, str]] = {}
_CACHE_LOCK   = threading.Lock()
_LOG_LOCK     = threading.Lock()
_THREAD_LOCAL = threading.local()


def _log(msg: str) -> None:
    """Thread-safe stderr logging in the style of main1.py's registry tags."""
    with _LOG_LOCK:
        print(f"{TAG} {msg}", file=sys.stderr)
        sys.stderr.flush()


class _AdaptiveThrottle:
    """
    Shared rate governor for one upstream API.

    Combines three mechanisms so that adding workers never adds request rate:
      * a reservation clock, so calls are evenly spaced across all threads;
      * a semaphore capping simultaneous in-flight requests;
      * AIMD adaptation - halve the rate on 429/503, creep back up on success.

    A 429 also parks every thread until Retry-After has elapsed, so one thread
    hitting the wall stops the rest from walking into it.
    """

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
        """Reserve the next send slot, then sleep outside the lock."""
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot, self._resume_at)
            self._next_slot = slot + self._interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def penalise(self, retry_after: float = 0.0) -> None:
        """Called on 429/503: halve the rate and park all threads."""
        with self._lock:
            old = self._rpm
            self._rpm = max(self._rpm_min, self._rpm * 0.5)
            self._ok_streak = 0
            pause = retry_after if retry_after > 0 else 60.0 / self._rpm
            pause += random.uniform(0, 0.5)          # de-synchronise threads
            self._resume_at = max(self._resume_at, time.monotonic() + pause)
            self._next_slot = max(self._next_slot, self._resume_at)
        _log(f"    throttle[{self.name}]: {old:.0f} → {self._rpm:.0f} rpm, "
             f"pausing {pause:.1f}s")

    def reward(self) -> None:
        """Called on success: creep the rate back up after a clean streak."""
        with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 20 and self._rpm < self._rpm_max:
                self._rpm = min(self._rpm_max, self._rpm * 1.1)
                self._ok_streak = 0

    def slot(self) -> "_ThrottleSlot":
        return _ThrottleSlot(self)


class _ThrottleSlot:
    """Context manager holding one concurrency permit for a throttle."""

    def __init__(self, throttle: _AdaptiveThrottle) -> None:
        self._t = throttle

    def __enter__(self) -> _AdaptiveThrottle:
        # Wait for the send slot FIRST, then take a permit. Doing it the other
        # way round holds a permit for the whole rate-limit sleep, so the
        # in-flight cap throttles throughput a second time on top of the clock.
        self._t.acquire()
        self._t._sem.acquire()
        return self._t

    def __exit__(self, *exc: Any) -> None:
        self._t._sem.release()


_GEMINI_THROTTLE = _AdaptiveThrottle("gemini", GEMINI_RPM, GEMINI_CONCURRENCY,
                                     GEMINI_RPM_MIN, GEMINI_RPM_MAX)
_NCBI_THROTTLE   = _AdaptiveThrottle("ncbi", NCBI_RPM, NCBI_CONCURRENCY)
_CTGOV_THROTTLE  = _AdaptiveThrottle("ctgov", CTGOV_RPM, CTGOV_CONCURRENCY)


def _retry_after_seconds(resp: requests.Response) -> float:
    """Parse a Retry-After header, seconds form or HTTP-date form."""
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
    """One requests.Session per thread – Sessions are not safe to share."""
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
    """GET with adaptive throttling, retry, and exponential backoff + jitter."""
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
    """Normalise unicode, dashes, quotes and whitespace for robust matching."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = re.sub(r"[\u2010-\u2015\u2212]", "-", t)   # dash variants incl. minus
    t = re.sub(r"[\u2018\u2019\u201b]", "'", t)
    t = re.sub(r"[\u201c\u201d]", '"', t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def _quote_support_ratio(quote: str, evidence: str) -> float:
    """How much of `quote` actually appears in `evidence`, 0.0–1.0."""
    q = _normalize(quote)
    e = _normalize(evidence)
    if not q or not e:
        return 0.0
    if q in e:
        return 1.0
    sm = difflib.SequenceMatcher(None, q, e, autojunk=False)
    match = sm.find_longest_match(0, len(q), 0, len(e))
    return match.size / len(q)


def _numbers_in(text: str) -> List[float]:
    """All numeric tokens in a string, comma separators removed."""
    out: List[float] = []
    for tok in re.findall(r"-?\d+(?:\.\d+)?", (text or "").replace(",", "")):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _same_number(a: Any, b: Any) -> bool:
    """Numeric equality on absolute values, with tolerance."""
    try:
        x, y = abs(float(a)), abs(float(b))
    except (TypeError, ValueError):
        return False
    if abs(x - y) <= NUMBER_ABS_TOLERANCE:
        return True
    return y > 0 and abs(x - y) / y <= NUMBER_REL_TOLERANCE


def _number_supported_by(value: Any, text: str) -> bool:
    """True only if `value` literally appears somewhere in `text`."""
    return any(_same_number(value, n) for n in _numbers_in(text))


# A quote has to be small enough that "the number appears in it" still means
# something. A 4000-character dump of a registry results table contains almost
# every plausible number, so matching against it proves nothing.
QUOTE_MIN_CHARS = 25
QUOTE_MAX_CHARS = 600
QUOTE_MAX_NUMBERS = 25


def _quote_is_well_formed(quote: str) -> Tuple[bool, str]:
    """Reject quotes too short to be evidence or too broad to be probative."""
    q = quote.strip()
    if len(q) < QUOTE_MIN_CHARS:
        return False, f"quote too short ({len(q)} chars, need {QUOTE_MIN_CHARS})"
    if len(q) > QUOTE_MAX_CHARS:
        return False, f"quote too long ({len(q)} chars, max {QUOTE_MAX_CHARS})"
    n_nums = len(_numbers_in(q))
    if n_nums > QUOTE_MAX_NUMBERS:
        return False, f"quote is a number dump ({n_nums} numbers)"
    return True, ""


def _quote_is_on_topic(quote: str, endpoint: Dict) -> bool:
    """
    The quote must actually mention the endpoint it is being used to support.

    Without this, a model can pair a genuine quote about one endpoint with a
    rationale about another and pass every other check - the quote is real, the
    number is in it, the source resolves. Only the topic is wrong.
    """
    q = _normalize(quote)
    for kw in endpoint["keywords"]:
        if re.search(rf"(?<![a-z0-9]){re.escape(_normalize(kw))}(?![a-z0-9])", q):
            return True
    return False


# ── deriving change_pct from the rationale ────────────────────────────────────
#
# The rationale is the single source of truth for the numeric cell. We mask out
# statistical furniture (CIs, p-values), reject numbers that are clearly
# timepoints, doses or counts, then pick the candidate closest to an endpoint
# keyword and apply the endpoint's sign convention.

_MASK_PATTERNS = [
    re.compile(r"\([^)]*\b(?:ci|confidence interval|p\s*[=<>])[^)]*\)", re.I),
    re.compile(r"\b\d{2}\s*%\s*ci\b[^;.]{0,60}", re.I),
    re.compile(r"\bp\s*[=<>]\s*0?\.\d+", re.I),
    re.compile(r"\bp\s*[=<>]\s*\d+\s*[x×]\s*10", re.I),
]

# Units that mean "this number is not our endpoint value"
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

# "Baseline HbA1c was 8.2%" is a starting value, not a result. Note this only
# fires when 'baseline' PRECEDES the number; "reduced 38.5% from baseline"
# puts it after, and stays valid.
_BASELINE_BEFORE = re.compile(
    r"\bbaselines?\b(?:\s+\S+){0,3}\s*(?:was|were|is|are|of|:)\s*$", re.I)

_NEGATIVE_CUES = re.compile(
    r"\b(reduc\w*|decreas\w*|declin\w*|lower\w*|loss|lost|fell|fall\w*|drop\w*|"
    r"shrank|improv\w*\s+by|less)\b", re.I)
_POSITIVE_CUES = re.compile(
    r"\b(increas\w*|rose|rise|gain\w*|higher|greater|resolution|resolv\w*|"
    r"achiev\w*|respond\w*|response\s+rate|improvement\s+rate)\b", re.I)


def _mask_stats(text: str) -> str:
    """Blank out CI/p-value spans, preserving character offsets."""
    out = text
    for pat in _MASK_PATTERNS:
        out = pat.sub(lambda m: " " * len(m.group()), out)
    return out


def _derive_change_pct(rationale: str, endpoint: Dict) -> Optional[float]:
    """
    Parse the endpoint's numeric value out of the rationale text.

    This is what fills the change_pct cell: the number is taken from the
    sentence that explains it, so cell and rationale can never disagree.
    Returns None when the rationale states no usable number.
    """
    if not rationale:
        return None

    masked = _mask_stats(rationale)
    low = masked.lower()

    # positions of endpoint keywords, used to disambiguate multiple numbers
    kw_spans = [m.start() for kw in endpoint["keywords"]
                for m in re.finditer(re.escape(kw), low)]

    # (distance to keyword, signed value, had explicit %, position in text)
    candidates: List[Tuple[float, float, bool, int]] = []

    for m in re.finditer(r"(-|–|—|minus\s+)?(\d+(?:\.\d+)?)\s*(%|percent(?:age)?"
                         r"(?:\s+points?)?)?", masked, re.I):
        raw = m.group(2)
        after = masked[m.end():m.end() + 24]
        before = masked[max(0, m.start() - 40):m.start()]

        # digits welded into a word are part of a name, not a measurement:
        # the "1" in "HbA1c", the "2" in "COVID19", the "3" in "GLP1R3"
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
        # "no change" is a real finding, but only trust a bare 0 when it is
        # explicitly written as a percentage
        if value == 0 and not m.group(3):
            continue

        explicit_neg = bool(m.group(1))
        had_pct = bool(m.group(3))

        # a bare integer with no % and no keyword nearby is probably a count
        dist = min((abs(m.start() - k) for k in kw_spans), default=9999.0)
        if not had_pct and dist > 60:
            continue

        signed = -value if explicit_neg else value
        candidates.append((dist, signed, had_pct, m.start(2)))

    if not candidates:
        return None

    # prefer a number carrying an explicit % and sitting nearest a keyword
    candidates.sort(key=lambda c: (0 if c[2] else 1, c[0]))
    _, value, _, best_pos = candidates[0]

    # Apply the endpoint's sign convention using only the words NEAR the number.
    # Scanning the whole rationale lets an unrelated clause flip the sign -
    # "weight decreased 12%; MASH resolution was also reported" would otherwise
    # see "resolution" as a positive cue for the weight value.
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


def _gemini_json(prompt: str, system: str,
                 schema: Dict[str, Any]) -> Optional[Dict]:
    """
    Call Gemini with enforced JSON schema output at temperature 0.
    Routed through the adaptive throttle; returns None on persistent failure.
    """
    if not GEMINI_API_KEY:
        return None

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "temperature": 0.0,          # no creative latitude
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


# ── Step 1: evidence gathering ────────────────────────────────────────────────

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


def _search_pubmed(query: str, max_results: int = 2) -> List[Dict[str, str]]:
    """Peer-reviewed abstracts via NCBI E-utilities."""
    out: List[Dict[str, str]] = []
    base = {"db": "pubmed"}
    if NCBI_API_KEY:
        base["api_key"] = NCBI_API_KEY

    r1 = _throttled_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        _NCBI_THROTTLE,
        params={**base, "term": query, "retmax": max_results, "retmode": "json"})
    if r1 is None:
        return out
    try:
        ids = r1.json().get("esearchresult", {}).get("idlist", [])
    except ValueError:
        return out
    if not ids:
        return out

    r2 = _throttled_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        _NCBI_THROTTLE,
        params={**base, "id": ",".join(ids),
                "rettype": "abstract", "retmode": "text"})
    if r2 is not None and r2.text.strip():
        out.append({
            "source": "PubMed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{ids[0]}/",
            "text": r2.text[:6000],
        })
    return out


def _evidence_mentions_trial(row: Dict[str, str], molecule: str,
                             evidence: str) -> bool:
    """Reject evidence that is about a different trial or drug."""
    ev = _normalize(evidence)
    if not ev:
        return False
    for anchor in (molecule, row.get("trial_id", ""), row.get("acronym", "")):
        a = _normalize(anchor)
        if len(a) >= 4 and a in ev:
            return True
    return False


def _build_queries(row: Dict[str, str], molecule: str,
                   endpoint: Dict) -> List[Tuple[str, str]]:
    """
    Build (anchor_kind, query) pairs. Trial ID and acronym are both used:
    registries index by ID, but journals and abstracts almost always publish
    under the acronym, so either one alone misses a large slice of the evidence.
    """
    trial_id = (row.get("trial_id") or "").strip()
    acronym  = (row.get("acronym") or "").strip()
    company  = (row.get("company_name") or "").strip()
    term     = endpoint["search_terms"][0]

    queries: List[Tuple[str, str]] = []
    if trial_id:
        queries.append(("trial_id", f"{trial_id} {term}"))
        queries.append(("trial_id", f"{trial_id}[si] {term}"))
    if acronym and acronym.lower() != trial_id.lower():
        queries.append(("acronym", f"{acronym} {molecule} {term}"))
        queries.append(("acronym", f"{acronym} trial {term}"))
    if company:
        queries.append(("sponsor", f"{molecule} {company} {term}"))
    queries.append(("molecule", f"{molecule} {term} randomized trial"))
    return queries


def gather_evidence(row: Dict[str, str], molecule: str,
                    endpoint: Dict) -> List[Dict[str, str]]:
    """
    Collect evidence snippets for one trial x one endpoint.

    Evidence is accumulated across anchors rather than stopping at the first
    hit, because the registry entry and the journal publication often carry
    different endpoints for the same trial. Duplicates are removed by URL.
    """
    snippets: List[Dict[str, str]] = _search_ctgov_results(row.get("trial_id", ""))
    seen = {s["url"] for s in snippets}

    anchors_used = {"registry"} if snippets else set()
    for kind, query in _build_queries(row, molecule, endpoint):
        # one PubMed hit per anchor kind is enough; keep the call budget sane
        if kind in anchors_used:
            continue
        for snip in _search_pubmed(query):
            if snip["url"] in seen:
                continue
            seen.add(snip["url"])
            snippets.append(snip)
            anchors_used.add(kind)
        if len(snippets) >= 4:
            break

    return [s for s in snippets
            if _evidence_mentions_trial(row, molecule, s["text"])]


# ── Step 2: extraction ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """\
You are a clinical data extraction assistant working under strict evidentiary
rules. You will receive trial metadata and numbered evidence snippets.

Extract ONE endpoint result. Obey these rules absolutely:

1. ABSTAIN BY DEFAULT. If the snippets do not state an explicit number for this
   exact endpoint for THIS exact trial, set found=false and every other field to
   null. Abstaining is always correct when in doubt. You are never penalised for
   abstaining; you are heavily penalised for guessing.

2. NEVER CALCULATE, CONVERT, OR INFER A NUMBER. Only report a number printed
   verbatim in a snippet. Do not derive percentages from absolute values, do not
   average arms, do not annualise, do not estimate from a described chart, and
   do not use anything you remember about this drug or trial from training. If
   the number is not on the page, it does not exist.

3. evidence_quote must be copied CHARACTER-FOR-CHARACTER from a snippet and must
   contain the number. Do not paraphrase, tidy, translate or re-punctuate it.
   This quote is checked automatically against the source text; a quote that
   does not appear verbatim causes the whole extraction to be discarded.

4. THE RATIONALE MUST STATE THE NUMBER AND ITS ORIGIN. Write 1-3 sentences in
   your own words that:
     (a) state WHAT was found, including the numeric result written the same way
         the quote writes it (for example "-14.9%" or "37% of patients"), the
         treatment arm or dose, and the timepoint;
     (b) state WHERE it was found - name the source type and the publication or
         registry it came from (for example "reported in the registry results
         posted for NCT03548935" or "reported in the peer-reviewed publication
         of the STEP 1 trial").
   The numeric cell is parsed directly from this rationale, so a rationale
   without a number produces an empty cell, and a rationale whose number differs
   from the quote is discarded entirely. Do not put confidence intervals or
   p-values in the rationale.

5. Report the TREATMENT arm at the primary or headline timepoint. If several
   doses are reported, use the highest dose and say so. If you cannot tell which
   arm a number belongs to, abstain.

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


# ── Step 3: verification (deterministic) + validation (LLM) ───────────────────

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
ambiguous=false. Otherwise "Low". When uncertain, answer "Low".
"""


def _verify(extracted: Dict[str, Any], snippets: List[Dict[str, str]],
            endpoint: Dict) -> Tuple[bool, str, Optional[float],
                                     Optional[Dict[str, str]]]:
    """
    Deterministic gate, run before the LLM auditor.
    Returns (passed, reason, derived_value, source_snippet).

    The chain enforced here is:
        fetched text -> verbatim quote -> rationale -> parsed number
    Break any link and the value is discarded.
    """
    if not extracted.get("found"):
        return False, "model abstained", None, None

    quote     = (extracted.get("evidence_quote") or "").strip()
    rationale = (extracted.get("rationale") or "").strip()

    if not quote:
        return False, "no evidence quote supplied", None, None
    if not rationale:
        return False, "no rationale supplied", None, None

    # 1. the quote must be a usable size - not a fragment, not a data dump
    well_formed, why = _quote_is_well_formed(quote)
    if not well_formed:
        return False, why, None, None

    # 2. the quote must be about the endpoint it is supporting
    if not _quote_is_on_topic(quote, endpoint):
        return False, f"quote does not mention {endpoint['label']}", None, None

    # 3. the quote must genuinely appear in the text we downloaded
    all_text = "\n".join(s["text"] for s in snippets)
    ratio = _quote_support_ratio(quote, all_text)
    if ratio < QUOTE_MATCH_THRESHOLD:
        return False, f"quote not found in source (match {ratio:.0%})", None, None

    # 4. the cell value is parsed FROM THE RATIONALE
    derived = _derive_change_pct(rationale, endpoint)
    if derived is None:
        return False, "rationale states no usable number", None, None

    # 5. that number must also be present in the verbatim quote
    if not _number_supported_by(derived, quote):
        return False, f"rationale value {derived} absent from quote", None, None

    # 6. resolve the owning snippet, so the URL is ours and never the model's
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
        return False, "quote spans no single source", None, None

    # 7. plausibility bound – catches unit confusion (ALT U/L read as percent)
    if abs(derived) > endpoint["plausible_max"]:
        return False, f"value {derived} outside plausible range", None, None

    return True, "verified", derived, src


def _llm_validate(derived: float, extracted: Dict[str, Any],
                  src: Dict[str, str], endpoint: Dict) -> Tuple[str, str]:
    """Independent second-opinion pass. Returns (confidence, reason)."""
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

    # a press release or aggregator cannot reach High on its own
    if SOURCE_TIER.get(src["source"], 3) > 1 and confidence == "High":
        confidence = "Low"
        why = f"source tier below peer-reviewed/registry; {why}"

    rationale = (extracted.get("rationale") or "").strip()
    final_rat = f"{rationale} [Source: {src['source']} – {src['url']}]"

    # Final sync guard: re-parse the rationale exactly as it will be written to
    # the sheet. If appending the source line perturbed the parse, the cell and
    # its explanation would disagree — drop the pair rather than ship a mismatch.
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
    """
    Enrich a single trial across all four endpoints.
    Endpoints run sequentially within a trial; trials run in parallel.
    """
    result: Dict[str, str] = {}
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
    Enrich trial rows with verified outcome data for HbA1c, weight, ALT and MASH.

    Trials run concurrently behind a shared adaptive throttle, so raising
    max_workers raises throughput without raising request rate. Rows sharing a
    trial_id are resolved once and the result fanned out.

    Each change_pct cell is parsed from its own rationale, and that rationale is
    checked against a verbatim quote from fetched source text. Anything that
    fails the chain is left blank with confidence "Low".

    Parameters
    ----------
    rows : list of dict
        Trial rows as produced by main1.py's fetch_all().
    molecule : str
        Molecule / drug name being searched.
    max_workers : int
        Concurrent trials. Request rate is governed separately by the throttle.

    Returns
    -------
    list of dict
        The same list object, with outcome columns populated in place.
    """
    if not rows:
        return rows

    if not GEMINI_API_KEY:
        _log("GEMINI_API_KEY not found in .env or environment "
             "– skipping outcome enrichment.")
        for row in rows:
            for ep in ENDPOINTS:
                row.setdefault(ep["col_pct"], "")
                row.setdefault(ep["col_dur"], "")
                row.setdefault(ep["col_rat"], "")
                row.setdefault(ep["col_conf"], "")
        return rows

    # Group rows by trial so duplicates across registries cost nothing.
    # Rows with no trial_id get a per-call unique key and are NEVER cached:
    # a positional key like "__row0" would otherwise be reused by an unrelated
    # trial on a later call and hand it the wrong outcome data.
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
         f"{max_workers} worker(s), {GEMINI_RPM:.0f} rpm start, "
         f"{GEMINI_CONCURRENCY} in flight; strict mode "
         f"{'ON' if STRICT_MODE else 'OFF'}.")

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
                    values = {}
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
    _log(f"Enrichment complete: {filled} verified value(s) across "
         f"{len(rows)} row(s).")
    return rows