# Multi-registry clinical trial fetcher

Give a drug name, get every clinical trial for it from eight registries —
one Excel per registry plus one combined master Excel with a `source` column.

Optionally, Gemini 2.5 Flash extracts the key findings for every trial into
`ai_*` columns.

## Install

```bash
pip install requests beautifulsoup4 pandas openpyxl
export GEMINI_API_KEY="your-key"      # only needed for --summarize
```

## Run

```bash
python fetch_all_trials.py "pembrolizumab"
```

Output lands in `./pembrolizumab_trials/`:

```
pembrolizumab_ALL_REGISTRIES.xlsx     <- master, all sources, `source` column
pembrolizumab_clinicaltrialsgov.xlsx
pembrolizumab_eu_clinical_trials.xlsx
pembrolizumab_ctri.xlsx
pembrolizumab_chictr.xlsx
pembrolizumab_jrct.xlsx
pembrolizumab_anzctr.xlsx
pembrolizumab_cris.xlsx
pembrolizumab_rebec.xlsx
```

### Useful flags

| Flag | What it does |
|---|---|
| `--outdir DIR` | where to write (default `./<drug>_trials`) |
| `--only SOURCE...` | run just some registries, e.g. `--only ClinicalTrials.gov CTRI` |
| `--skip SOURCE...` | skip some registries |
| `--max-records N` | cap rows per registry (good for a smoke test) |
| `--no-details` | skip per-trial detail pages — much faster, fewer fields |
| `--strict` | keep only trials naming the drug in an intervention field |
| `--workers N` | registries queried in parallel (default 3; keep it low) |
| `--no-fallback` | don't back-fill empty registries via WHO ICTRP |
| `--traceback` | full tracebacks when a registry fails |

### Gemini key-findings flags

| Flag | What it does |
|---|---|
| `--summarize` | add `ai_*` key-findings columns via Gemini 2.5 Flash |
| `--grounded` | also Google-Search for *published* results when the registry record has none |
| `--only-with-results` | only call Gemini for rows that already contain results text (cheaper) |
| `--gemini-model M` | default `gemini-2.5-flash` |
| `--gemini-workers N` | parallel Gemini calls (default 4) |
| `--gemini-rpm N` | requests-per-minute cap (default 60) |
| `--gemini-thinking N` | thinking token budget (default 0 = off, cheapest) |
| `--no-gemini-cache` | disable the on-disk response cache |

Each registry script also runs standalone:

```bash
python ctgov_trials.py "semaglutide" --max-records 50
python rebec_trials.py "metformin" --out rebec.xlsx
```

And the summariser can be run over an Excel you already have:

```bash
python gemini_summarizer.py metformin_trials/metformin_ALL_REGISTRIES.xlsx
python gemini_summarizer.py trials.xlsx --grounded --workers 8
```

## Gemini 2.5 Flash key findings

```bash
python fetch_all_trials.py "semaglutide" --summarize
python fetch_all_trials.py "semaglutide" --summarize --grounded
```

Adds eleven columns:

| Column | Contents |
|---|---|
| `ai_key_findings` | all key findings, `\|`-separated, numbers verbatim — or `No results reported in registry record` |
| `ai_primary_result` | primary endpoint result with its numbers |
| `ai_secondary_results` | secondary endpoint results |
| `ai_safety_findings` | adverse events / safety signals |
| `ai_conclusion` | the conclusion **as stated in the record** |
| `ai_summary` | 2–3 sentence plain summary |
| `ai_has_registry_results` | `Yes`/`No` — did the record contain real results? |
| `ai_evidence_basis` | which fields the findings were drawn from |
| `ai_published_findings` | `--grounded` only: findings from published literature |
| `ai_sources` | `--grounded` only: URLs backing those findings |
| `ai_model` | model that produced the row |

### How hallucination is constrained

- **Registry text only.** The system prompt forbids using training knowledge;
  the model sees only that trial's registry fields.
- **Numbers verbatim.** No rounding, rescaling, recomputing or inferring.
- **Planned ≠ observed.** A protocol-only record lists endpoints it *intends*
  to measure. The prompt treats those as not-findings and returns
  `No results reported in registry record`.
- **No unearned conclusions.** The model may not say a drug worked, was safe or
  was superior unless the record says so.
- **Structured output.** `responseMimeType: application/json` + `responseSchema`
  so replies can't drift into prose. `temperature: 0`.
- **Auditable.** `ai_evidence_basis` names the source fields;
  `ai_has_registry_results` separates real results from empty records; web
  findings stay in separate `ai_published_*` columns with URLs.
- **Cached.** Responses are cached by content hash in
  `<outdir>/.gemini_cache.json`, so re-runs don't re-bill.

> **Still verify.** An LLM can misread a table or transpose a number. These
> columns are a reading aid over the raw registry fields, not a substitute.
> Check anything that matters against the `url` column. Don't use them for
> clinical or regulatory decisions unverified.

### Cost note

Roughly one call per trial at ~2–4k input tokens. Thinking is off by default.
`--only-with-results` skips protocol-only records, which is usually most of
them. `--grounded` adds a second call per resultless trial and is the expensive
mode.

## How each registry is reached

| Source (`source` column) | Registry | Method |
|---|---|---|
| `ClinicalTrials.gov` | USA | **Official REST API v2** — `clinicaltrials.gov/api/v2/studies`, no key |
| `EU Clinical Trials` | CTIS + EudraCT | CTIS JSON API + EU-CTR HTML (reuses `ctis_drug_trials.py`, `eudract_drug_trials.py`) |
| `ReBEC` | Brazil | **Per-trial ICTRP XML** — `/rg/<id>/xml/ictrp`, effectively a free API |
| `CRIS` | South Korea | **Official open API** (data.go.kr, needs a key) → HTML fallback |
| `ANZCTR` | Australia/NZ | Per-trial XML (`&isXml=true`) → HTML fallback |
| `CTRI` | India | HTML scraping (no API exists) |
| `JRCT` | Japan | HTML scraping (no API exists) — see the notice below |
| `ChiCTR` | China | HTML scraping (no API, heavy bot protection) |

### CRIS open API key (optional but better)

CRIS's official API lives on the Korean government data portal. Register free
at data.go.kr for the "국립보건연구원_임상연구정보서비스" service, then:

```bash
export CRIS_API_KEY="your-decoded-service-key"
```

Without it, `cris_trials.py` automatically falls back to HTML scraping.

### jRCT terms of use

jRCT's site notice asks users not to bulk-download via automated programs
beyond personal use. `jrct_trials.py` therefore caps at 100 records by
default, sleeps 2s between requests, and runs single-threaded. Raise
`JRCT_DELAY` to slow it further. For bulk needs use the JPRN portal
(rctportal.mhlw.go.jp) or contact jRCT.

## WHO ICTRP fallback

If a registry returns nothing (ChiCTR blocking, a site outage), the
orchestrator back-fills from WHO ICTRP, which mirrors all of these registries.

**Records recovered this way are re-labelled with their originating registry.**
The `source` column never says "ICTRP" — it says ChiCTR, CTRI, JRCT, ANZCTR,
CRIS, ReBEC, ClinicalTrials.gov or EU Clinical Trials. Any ICTRP record from a
registry not in that list (DRKS, IRCT, TCTR, PACTR…) is discarded. Disable
with `--no-fallback`. Rows recovered this way are flagged in the run summary.

## Output schema

The master Excel has 37 unified columns, `source` first:

```
source, trial_id, secondary_ids, title, public_title, status, phase,
study_type, study_design, conditions, interventions, drug_names, sponsor,
sponsor_type, collaborators, countries, sites, target_enrollment,
actual_enrollment, age_min, age_max, gender, healthy_volunteers,
inclusion_criteria, exclusion_criteria, primary_objective, primary_outcome,
secondary_outcome, start_date, completion_date, registration_date,
last_updated, results_available, findings, contact, ethics_approval, url
```

Per-registry files carry those same columns **plus every extra field** that
registry publishes, namespaced (`ctgov.*`, `ctri.*`, `anzctr.*`, …). Field
capture is generic — JSON and XML records are fully flattened, and HTML pages
are harvested for every label/value pair — so new registry fields appear
automatically rather than needing code changes.

Values are passed through **exactly as the source publishes them**. Phase from
ClinicalTrials.gov reads `PHASE3`, from CTRI `Phase 3`, from EudraCT
`Phase III`. Nothing is normalised across registries.

## Caveats worth knowing

- **HTML scrapers are brittle by nature.** CTRI, ChiCTR, jRCT and the HTML
  fallbacks parse live pages; a site redesign breaks them. The generic
  label/value harvester limits the damage, but the `find_field(...)` label
  lists in each module are what you'd adjust.
- **ChiCTR blocks automated requests aggressively.** Expect it to return
  nothing and rely on the ICTRP path.
- **These scripts could not be tested against the live sites** from my
  environment (no outbound network). Logic is unit-tested against realistic
  mocked payloads; the URL patterns and API shapes come from each registry's
  current documentation and pages. Run with `--max-records 5` first and check
  the output before a full run.
- **Be polite.** Every scraper throttles. Don't raise `--workers` much, and
  don't remove the sleeps.
- **The Gemini path is mock-tested, not live-tested.** Request/response shapes
  match Google's current `generateContent` docs (verified: model id
  `gemini-2.5-flash`, `responseSchema`, `thinkingConfig.thinkingBudget`,
  `google_search` tool), but I had no key or network to call it for real. Run
  `--summarize --max-records 3` first.
- **`--grounded` cannot combine with structured output.** Gemini disallows
  `responseSchema` alongside tools, so the grounded pass asks for JSON in the
  prompt and parses it defensively. Its output is less reliably shaped than the
  registry-only pass.

---

## Metric extraction with Gemini 2.5 Flash

Adds five columns:

| Column | Contents |
|---|---|
| `entire_trial_information` | every non-empty field of the record as `Label: value` text |
| `weight_reduction` | weight / BMI / waist findings, or `INFO N/A` |
| `hba1c_reduction` | HbA1c change findings, or `INFO N/A` |
| `mash` | MASH / NASH / fibrosis / liver-fat findings, or `INFO N/A` |
| `alt_reduction` | ALT change findings, or `INFO N/A` |

```bash
export GEMINI_API_KEY="your-key"
python fetch_all_trials.py "tirzepatide" --metrics
```

Gemini reads **only** `entire_trial_information` for that row — never the web,
never its own knowledge of the drug.

| Flag | Effect |
|---|---|
| `--metrics` | run the extraction |
| `--metrics-text-only` | build the text column but skip the LLM (all four read `INFO N/A`) |
| `--metrics-key KEY` | API key (else `$GEMINI_API_KEY`) |
| `--metrics-model` | default `gemini-2.5-flash` |
| `--metrics-workers N` | parallel calls, default 4 |
| `--metrics-cache FILE` | re-runs don't re-pay for unchanged rows |

Standalone on an existing file:

```bash
python gemini_enrich.py tirzepatide_trials/tirzepatide_ALL_REGISTRIES.xlsx
```

### How hallucination is prevented

The model's answer is **not trusted**. Four guards apply, two of them enforced
by Python rather than by asking the model nicely:

1. `temperature=0`, `thinkingBudget=0` — deterministic.
2. A JSON `responseSchema` forces every metric to carry both a `finding` and
   an `evidence` span.
3. **Evidence verification (code).** The `evidence` must appear verbatim in
   the row's source text, whitespace-normalised. If not → `INFO N/A`.
4. **Number verification (code).** Every number in the `finding` must also
   appear in the source text. A fabricated percentage or p-value is dropped
   even when the evidence span was genuine.

Rejections are logged live (`[guard] row 12 mash: rejected (...)`) and counted
in the run summary, so you can see how often the model tried to overreach.

Tested against deliberately hallucinating mock responses: fabricated numbers,
fabricated evidence spans, missing evidence, and invented MASH results were
all rejected and became `INFO N/A`.

### Honest limits

- Guards catch **fabrication**, not **misreading**. If a trial reports a
  placebo-arm number and the model attributes it to the treatment arm, both
  the number and the evidence span are real, so it passes. Spot-check rows
  that matter against the `url` column.
- "`INFO N/A`" means *not found in the registry record* — not *the trial did
  not measure it*. Registries frequently list an endpoint with no posted
  result.
- Costs money per row. Use `--max-records` first, and keep the cache file.

### Two separate Gemini features

This package now has two independent LLM steps, with separate flags:

| Flag | Module | Columns |
|---|---|---|
| `--summarize` | `gemini_summarizer.py` | general `ai_*` key-findings columns |
| `--metrics` | `gemini_enrich.py` | `entire_trial_information` + the four metric columns |

They compose — run both and you get both sets of columns.
