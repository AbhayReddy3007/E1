"""Two-Step Gemini Clinical Trial Extraction - Core Function

This module provides the core extraction logic used by the market potential agent.
"""
import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv
from google import genai
from google.genai import types
from json_repair import repair_json

load_dotenv()


def safe_json_parse(json_str: str, context: str = "") -> dict | list | None:
    """Parse JSON with automatic repair using json_repair library.

    Args:
        json_str: The JSON string to parse
        context: Optional context for error messages

    Returns:
        Parsed JSON object/array, or None if parsing fails
    """
    if not json_str or not json_str.strip():
        return None

    # Clean up common issues
    json_str = json_str.strip()

    # Remove markdown code blocks if present
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        parts = json_str.split("```")
        if len(parts) >= 2:
            json_str = parts[1].strip()

    # Find first { or [ if there's text before JSON
    json_start = -1
    for i, c in enumerate(json_str):
        if c in '{[':
            json_start = i
            break
    if json_start > 0:
        json_str = json_str[json_start:]

    # Try json_repair first (handles most issues automatically)
    try:
        repaired = repair_json(json_str, return_objects=True)
        if repaired is not None:
            return repaired
    except Exception as e:
        if context:
            print(f"  ⚠️ json_repair failed for {context}: {e}")

    # Fallback to standard json.loads
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None

# Initialize Gemini
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Configuration
BATCH_SIZE = 8
MAX_WORKERS = 12          # was 8 — Gemini Flash can sustain more concurrent calls
RATE_LIMIT_DELAY = 1.0    # was 2.0 — halved; backoff handles actual rate limits
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0

# Models
SEARCH_MODEL = "gemini-3-flash-preview"
EXTRACTION_MODEL = "gemini-3-flash-preview"

# Registry definitions - each will be searched individually
REGISTRIES = [
    {
        "name": "ClinicalTrials.gov",
        "id_prefix": "NCT",
        "url": "https://clinicaltrials.gov/",
        "region": "US/Global",
        "description": "US National Library of Medicine - primary source for most clinical trials"
    },
    {
        "name": "ChiCTR",
        "id_prefix": "ChiCTR",
        "url": "https://www.chictr.org.cn/",
        "region": "China",
        "description": "Chinese Clinical Trial Registry"
    },
    {
        "name": "EU Clinical Trials Register",
        "id_prefix": "EUCTR",
        "url": "https://www.clinicaltrialsregister.eu/",
        "region": "Europe",
        "description": "European Union Clinical Trials Register"
    },
    {
        "name": "CTRI",
        "id_prefix": "CTRI",
        "url": "https://ctri.nic.in/",
        "region": "India",
        "description": "Clinical Trials Registry - India"
    },
    {
        "name": "JRCT",
        "id_prefix": "jRCT",
        "url": "https://rctportal.niph.go.jp/en/",
        "region": "Japan",
        "description": "Japan Registry of Clinical Trials"
    },
    {
        "name": "ANZCTR",
        "id_prefix": "ACTRN",
        "url": "https://www.anzctr.org.au/",
        "region": "Australia/New Zealand",
        "description": "Australian New Zealand Clinical Trials Registry"
    },
    {
        "name": "CRIS",
        "id_prefix": "KCT",
        "url": "https://cris.nih.go.kr/",
        "region": "Korea",
        "description": "Clinical Research Information Service - Korea"
    },
    {
        "name": "ReBEC",
        "id_prefix": "RBR",
        "url": "https://ensaiosclinicos.gov.br/",
        "region": "Brazil",
        "description": "Brazilian Clinical Trials Registry"
    },
]


async def gemini_call_with_search(prompt: str, model: str, response_mime_type: str = None) -> str:
    """Make a Gemini call with Google Search enabled and exponential backoff for rate limits (async).

    Args:
        prompt: The prompt to send to Gemini
        model: Model name (e.g., "gemini-2.5-flash")
        response_mime_type: Optional mime type (e.g., "application/json").
                           Note: Using response_mime_type with Google Search tools may cause errors.
    """

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]

    # Build config - only include response_mime_type if specified
    config_params = {"tools": tools}
    if response_mime_type:
        config_params["response_mime_type"] = response_mime_type

    config = types.GenerateContentConfig(**config_params)

    retry_count = 0
    backoff_delay = INITIAL_BACKOFF

    while retry_count <= MAX_RETRIES:
        try:
            # Run synchronous genai call in thread pool
            response_text = await asyncio.to_thread(_sync_gemini_call, model, contents, config)
            return response_text

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limit errors
            if any(err in error_str for err in ["429", "rate limit", "quota", "resource exhausted"]):
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    print(f"\n  ❌ Max retries ({MAX_RETRIES}) exceeded for rate limit. Error: {e}")
                    raise

                print(f"\n  ⚠️  Rate limit hit - waiting {backoff_delay:.1f}s before retry {retry_count}/{MAX_RETRIES}...")
                await asyncio.sleep(backoff_delay)
                backoff_delay *= 2  # Exponential backoff: 2s, 4s, 8s, 16s, 32s
            else:
                # Non-rate-limit error, raise immediately
                print(f"  ❌ Gemini API error: {e}")
                raise

    return ""  # Fallback (should not reach here)


def _sync_gemini_call(model: str, contents: list, config) -> str:
    """Synchronous wrapper for Gemini API call (to run in thread pool)."""
    response_text = ""
    for chunk in genai_client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.text:
            response_text += chunk.text
    return response_text.strip()


def _extract_first_json_object(text: str) -> str:
    """Extract the first complete JSON object from text.

    Handles cases where Gemini returns extra data after valid JSON.
    """
    # Find the first opening brace
    start = text.find('{')
    if start == -1:
        # Maybe it's an array
        start = text.find('[')
        if start == -1:
            return ""
        closing = ']'
        opening = '['
    else:
        closing = '}'
        opening = '{'

    # Track brace depth to find matching close
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start:i+1]

    # No complete JSON found, return from start to end
    return text[start:]


def _repair_truncated_json(response: str) -> str:
    """Attempt to repair truncated/malformed JSON by removing incomplete trials and closing structures.

    Strategy:
    1. Find the last complete trial object (ends with })
    2. Remove everything after it
    3. Close the trials array and outer object
    """
    if not response:
        return response

    # Find the "trials" array start
    trials_start = response.find('"trials"')
    if trials_start == -1:
        return response

    # Find the [ that starts the trials array
    array_start = response.find('[', trials_start)
    if array_start == -1:
        return response

    # Find all complete trial objects (pattern: },\n or }\n])
    # We want to find the last } that completes a trial object before truncation

    # Strategy: Find all positions where we have a complete trial (} followed by comma or array close)
    last_complete_pos = -1
    depth = 0
    in_string = False
    escape_next = False

    for i in range(array_start + 1, len(response)):
        char = response[i]

        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        # Track brace depth
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            # If depth is 0, we've closed a complete trial object
            if depth == 0:
                # Check if next non-whitespace is comma or array close
                next_idx = i + 1
                while next_idx < len(response) and response[next_idx] in ' \n\r\t':
                    next_idx += 1
                if next_idx < len(response) and response[next_idx] in ',]':
                    last_complete_pos = i

    # If we found at least one complete trial, cut there
    if last_complete_pos != -1:
        # Cut right after the last complete }
        response = response[:last_complete_pos + 1]

        # Add closing for trials array and outer object
        response += '\n  ]\n}'
        return response

    # Fallback: try to salvage what we can
    # Find last complete } at any depth
    last_brace = response.rfind('}')
    if last_brace != -1 and last_brace > array_start:
        response = response[:last_brace + 1]
        response += '\n  ]\n}'
        return response

    # Last resort: just close everything
    open_braces = response.count('{') - response.count('}')
    open_brackets = response.count('[') - response.count(']')
    response += ']' * open_brackets
    response += '}' * open_braces

    return response


def _parse_trial_json(response: str) -> list:
    """Helper to parse JSON response from trial search with repair attempts."""
    if not response or not response.strip():
        return []

    # Clean code blocks
    response = response.strip()
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()

    # Extract just the first JSON object
    json_str = _extract_first_json_object(response)
    if not json_str:
        return []

    # First attempt: parse as-is
    try:
        data = json.loads(json_str)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("trials", [])
        else:
            return []

    except json.JSONDecodeError as e:
        error_msg = str(e)
        print(f"  ⚠️  JSON parse error: {e}")

        # Handle "Extra data" error - extract just the first complete JSON
        if "Extra data" in error_msg:
            print(f"  Attempting to extract first valid JSON object...")
            try:
                decoder = json.JSONDecoder()
                data, end_idx = decoder.raw_decode(json_str)

                if isinstance(data, list):
                    print(f"  ✓ Extracted JSON - recovered {len(data)} trials (ignored extra data)")
                    return data
                elif isinstance(data, dict):
                    trials = data.get("trials", [])
                    print(f"  ✓ Extracted JSON - recovered {len(trials)} trials (ignored extra data)")
                    return trials
                else:
                    return []
            except Exception as e2:
                print(f"  ❌ Extraction failed: {e2}")
                return []

        # Handle truncation errors - try to repair
        print(f"  Attempting to repair truncated JSON...")
        try:
            repaired = _repair_truncated_json(json_str)
            data = json.loads(repaired)

            if isinstance(data, list):
                print(f"  ✓ Repaired JSON - recovered {len(data)} trials")
                return data
            elif isinstance(data, dict):
                trials = data.get("trials", [])
                print(f"  ✓ Repaired JSON - recovered {len(trials)} trials")
                return trials
            else:
                return []

        except json.JSONDecodeError as e2:
            print(f"  ❌ Repair failed: {e2}")
            print(f"  Trying last-resort extraction...")

            # Last resort: extract any complete trial objects manually
            trials = []
            # Find all complete {...} objects that look like trials
            import re
            # Match trial objects with NCT_ID and Program_Name fields
            trial_pattern = r'\{[^{}]*"NCT_ID"[^{}]*"Program_Name"[^{}]*\}'
            matches = re.finditer(trial_pattern, json_str, re.DOTALL)

            for match in matches:
                try:
                    trial_obj = json.loads(match.group(0))
                    trials.append(trial_obj)
                except:
                    continue

            if trials:
                print(f"  ✓ Last-resort extraction - recovered {len(trials)} trials")
                return trials

            print(f"  ❌ All recovery attempts failed.")
            return []


def _phase_to_number(phase: str) -> float:
    """Convert phase string to numeric value for comparison."""
    if not phase or phase.upper() == "N/A":
        return 0.0

    phase = str(phase).strip().upper().replace("PHASE", "").strip()

    phase_map = {
        "1": 1.0,
        "1B": 1.5,
        "2": 2.0,
        "2A": 2.3,
        "2B": 2.5,
        "3": 3.0,
        "3B": 3.5,
        "4": 4.0,
    }

    return phase_map.get(phase, 0.0)


def _deduplicate_by_phase(trials: list) -> list:
    """Deduplicate trials by ID, keeping the one with highest phase.

    If same trial ID appears multiple times with different phases,
    only keep the entry with the latest (highest) phase.
    """
    if not trials:
        return []

    # Group by trial ID
    trial_groups = {}
    for trial in trials:
        trial_id = trial.get("NCT_ID", "").strip()
        if not trial_id:
            continue

        if trial_id not in trial_groups:
            trial_groups[trial_id] = []
        trial_groups[trial_id].append(trial)

    # For each group, select the trial with highest phase
    deduplicated = []
    duplicates_found = 0

    for trial_id, group in trial_groups.items():
        if len(group) > 1:
            duplicates_found += len(group) - 1
            # Sort by phase (highest first) and take the first one
            group_sorted = sorted(group, key=lambda t: _phase_to_number(t.get("Phase", "0")), reverse=True)
            selected = group_sorted[0]

            phases = [t.get("Phase", "N/A") for t in group]
            print(f"  ℹ️  {trial_id}: Multiple phases found {phases}, selected Phase {selected.get('Phase', 'N/A')}")
        else:
            selected = group[0]

        deduplicated.append(selected)

    if duplicates_found > 0:
        print(f"\n  Resolved {duplicates_found} duplicate(s) by selecting highest phase")

    return deduplicated


async def search_single_registry(molecule: str, registry: dict) -> list:
    """Search a single registry for all trials of a molecule."""

    prompt = f"""Search {registry['name']} ({registry['url']}) for ALL clinical trials of {molecule}.

REGISTRY: {registry['name']}
URL: {registry['url']}
REGION: {registry['region']}
ID PREFIX: {registry['id_prefix']}

YOUR TASK: Find ALL {molecule} trials registered on this specific registry.

SEARCH SCOPE - Include ALL of these:
- All phases (1, 2, 3, 3b, 4)
- All indications: Type 2 Diabetes, Obesity, MASH/NASH, Cardiovascular, CKD, Heart Failure, etc.
- All dosage forms: injection (subcutaneous), oral tablets, etc.
- Innovator trials (Novo Nordisk, Eli Lilly, etc.)
- Generic/biosimilar trials
- Completed, active, recruiting, and terminated trials

CRITICAL VERIFICATION - MUST EXCLUDE:
- ❌ EXCLUDE combination drugs where {molecule} is combined with another active molecule (e.g., CagriSema = Cagrilintide + Semaglutide should be EXCLUDED when searching for Semaglutide)
- ❌ EXCLUDE trials where {molecule} is only used as background therapy or comparator
- ❌ EXCLUDE trials testing a different drug in patients taking {molecule}
- ✅ ONLY include trials where {molecule} alone is the PRIMARY intervention being tested
- ✅ {molecule} + placebo is OK (monotherapy trial design)
- ✅ {molecule} vs other drugs is OK (if {molecule} is being tested)
- If the trial name includes multiple drug names (e.g., "Drug A + Drug B"), DO NOT include it
- If uncertain whether {molecule} is the primary intervention, DO NOT include it

For each trial found, provide:
- NCT_ID: Registry ID (must start with {registry['id_prefix']} for this registry)
- Program_Name: Official program name (e.g., "STEP 1", "SUSTAIN 6") or "N/A" if not named
- Indication: Primary indication (e.g., "Obesity", "Type 2 Diabetes", "MASH")
- Phase: Trial phase (e.g., "1", "2", "3", "3b", "4") - if available from registry

IMPORTANT:
- ONLY return trials from {registry['name']} that are verified to relate to {molecule}
- Every trial ID must be a valid {registry['id_prefix']} number
- Return as many trials as you can find

Return as JSON:

{{
  "trials": [
    {{
      "NCT_ID": "{registry['id_prefix']}XXXXXXXX",
      "Program_Name": "PROGRAM NAME",
      "Indication": "INDICATION",
      "Phase": "3"
    }}
  ]
}}

If no trials are found on this registry, return: {{"trials": []}}
"""

    response = await gemini_call_with_search(prompt, model=SEARCH_MODEL, response_mime_type="application/json")
    trials = _parse_trial_json(response)

    # Add registry source to each trial
    for trial in trials:
        trial["registry_source"] = registry["name"]

    return trials


async def get_all_trials_parallel(molecule: str) -> list:
    """Step 1: Search ALL registries in parallel and combine results."""

    print(f"\n{'='*80}")
    print(f"STEP 1: Searching ALL registries INDIVIDUALLY for {molecule}")
    print(f"Using model: {SEARCH_MODEL}")
    print(f"Searching {len(REGISTRIES)} registries in parallel")
    print(f"{'='*80}\n")

    # Create tasks for all registries
    tasks = []
    for registry in REGISTRIES:
        print(f"  📋 Queuing: {registry['name']} ({registry['region']})")
        tasks.append(search_single_registry(molecule, registry))

    print(f"\n🚀 Launching {len(tasks)} parallel searches...\n")

    # Run all searches in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect and display results per registry
    all_trials = []
    registry_counts = {}

    for i, (registry, result) in enumerate(zip(REGISTRIES, results)):
        if isinstance(result, Exception):
            print(f"  ❌ {registry['name']}: Error - {result}")
            registry_counts[registry['name']] = 0
        else:
            count = len(result)
            registry_counts[registry['name']] = count
            if count > 0:
                print(f"  ✓ {registry['name']}: Found {count} trials")
                all_trials.extend([t for t in result if isinstance(t, dict)])
            else:
                print(f"  ○ {registry['name']}: No trials found")

    # Deduplicate by trial ID and phase - keep highest phase
    unique_trials = _deduplicate_by_phase(all_trials)

    print(f"\n{'='*80}")
    print(f"STEP 1 SUMMARY")
    print(f"{'='*80}")
    print(f"  Total trials found: {len(all_trials)}")
    print(f"  After phase deduplication: {len(unique_trials)} unique trials")
    print(f"\n  Breakdown by registry:")
    for registry_name, count in registry_counts.items():
        print(f"    {registry_name}: {count}")

    # Display sample
    print(f"\n  Trial IDs (showing first 20):")
    for trial in unique_trials[:20]:
        nct = trial.get("NCT_ID", "")
        program = trial.get("Program_Name", "")
        phase = trial.get("Phase", "N/A")
        source = trial.get("registry_source", "")
        print(f"    {nct:25} {program:20} Phase {phase:3} [{source}]")

    if len(unique_trials) > 20:
        print(f"    ... and {len(unique_trials) - 20} more")

    return unique_trials


async def get_trial_details(molecule: str, trial_batch: list,
                            extra_fields_prompt: str = None,
                            extra_fields_json: str = None) -> dict:
    """Step 2: Get detailed efficacy data for a batch of trials (async).

    Args:
        molecule: Molecule name
        trial_batch: List of trial dicts from Step 1
        extra_fields_prompt: Optional additional prompt for extra fields (e.g., tolerability)
        extra_fields_json: Optional additional JSON fields to include in schema
    """

    # Filter out trials without NCT_ID to avoid None in join
    nct_ids = [t.get("NCT_ID") for t in trial_batch if t.get("NCT_ID")]
    nct_list = ", ".join(nct_ids)

    print(f"\nQuerying details for: {nct_list[:80]}...")

    prompt = f"""You are a clinical data extraction engine with access to MULTIPLE international trial registries.

IMPORTANT VALIDATION: Before extracting data, verify each trial is a {molecule} MONOTHERAPY trial.
- ❌ SKIP trials testing combination drugs (e.g., CagriSema = Cagrilintide + Semaglutide)
- ❌ SKIP trials where {molecule} is only background therapy
- ✅ ONLY extract data for trials where {molecule} alone is the primary intervention

Task: Extract structured Phase 2, 3, and 3b interventional clinical trial efficacy data for these {len(trial_batch)} {molecule} MONOTHERAPY trials:

{chr(10).join([f"- {t.get('NCT_ID', 'N/A')} ({t.get('Program_Name', 'N/A')})" for t in trial_batch])}

DATA SOURCES (search ALL for complete data):
1. ClinicalTrials.gov - https://clinicaltrials.gov/study/[TRIAL_ID]
2. PubMed/published papers linked to the trial
3. ChiCTR (China) - https://www.chictr.org.cn/ (for Chinese trials)
4. EU Clinical Trials Register - https://www.clinicaltrialsregister.eu/ (for European trials)
5. CTRI (India) - https://ctri.nic.in/ (for Indian trials)
6. JRCT (Japan) - https://rctportal.niph.go.jp/en/ (for Japanese trials)
7. Other international registries as available

MULTI-SOURCE STRATEGY:
- If a field is missing from ClinicalTrials.gov, check other registries
- Cross-reference data across sources for accuracy
- Prioritize: Published papers > Primary registry > Secondary registries

For EACH trial, extract:
- Molecule: {molecule}
- Trial Title: The FULL official title of the trial as listed on the trial website (e.g., "Research Study Investigating How Well Semaglutide Works in People Suffering From Overweight or Obesity"). Use "N/A" only if completely unavailable.
- Dosage: The PRIMARY or HIGHEST dosage tested in this trial. Report ONLY ONE dosage value (e.g., "2.4 mg OW", "1.0 mg OW", "14 mg QD"). If multiple doses were tested, choose the highest dose. Format: "[amount] [unit] [frequency]"
- Trial ID: NCT number with program name (e.g., "NCT03548935 (STEP 1)")
- Phase: (e.g., "3", "2", "3b")
- Size: Total enrollment number
- Primary Region: The TOP 1 or 2 geographic regions where the trial has the most sites/countries. Use ONLY these region codes: "US", "EU", "China", "India", "Japan", "Korea", "Latin America", "Middle East", "Asia-Pacific". MAXIMUM 2 regions combined with "/" — pick the 2 with the most country representation. If clearly dominated by one region, use just 1. Examples: US + many EU countries → "US/EU". EU-only → "EU". China-only → "China". Global trial (US, EU, Japan, Korea, Latin America) → "US/EU" (top 2 only).
- Secondary Countries: ALL specific countries where the trial was conducted, comma-separated (e.g., "United States, Canada, Mexico" or "Germany, France, Spain, Italy, Netherlands"). List every country with trial sites. Use "N/A" only if country information is unavailable.
- Start Date: Full "Study Start" date in format "YYYY-MM-DD" or "YYYY-MM" or "YYYY". For NCT trials this will be overridden by the CT.gov API — provide best effort for non-NCT registries. Use "N/A" if unavailable.
- Completion Date: Full "Primary Completion (Actual)" date in format "YYYY-MM-DD" or "YYYY-MM" or "YYYY". For NCT trials this will be overridden by the CT.gov API — provide best effort for non-NCT registries. Use "N/A" if ongoing or unavailable.
- Status: "Completed", "Active", "Recruiting"
- MASH Outcome (%): MASH/NASH resolution rate (or "N/A" if not a MASH trial)
- HbA1c Change (%): HbA1c reduction in percentage points (positive number, or "N/A")
- HbA1c Duration: Timepoint for HbA1c measurement (e.g., "12 wk", "26 wk", or "N/A")
- HbA1c Rationale: One sentence explaining exactly where you found this HbA1c number (e.g. "From NEJM 2025 paper by Davies et al., Table 2, treatment-policy estimand" or "From Novo Nordisk press release dated Feb 2, 2026, topline results"). Use "N/A" if HbA1c is N/A.
- Weight Loss (%): Body weight loss percentage (positive number, or "N/A")
- Weight Duration: Timepoint for weight measurement (e.g., "68 wk", "104 wk", or "N/A")
- Weight Rationale: Same format as HbA1c Rationale but for the weight loss number. Use "N/A" if weight is N/A.
- ALT Reduction (%): ALT enzyme reduction percentage (or "N/A")
- ALT Duration: Timepoint for ALT measurement (e.g., "24 wk", "72 wk", or "N/A")
- ALT Rationale: Same format. Use "N/A" if ALT is N/A.
- MASH Duration: Timepoint for MASH/NASH assessment (e.g., "72 wk", "96 wk", or "N/A")
- MASH Rationale: Same format. Use "N/A" if MASH is N/A.
- Company: Sponsor company (include generic/pharma context: e.g., "Novo Nordisk" or "Generic manufacturer")
- Trial_Study_Type: Type of clinical trial — must be exactly one of: "Expanded Access", "Interventional", "Observational". Use "N/A" only if completely unavailable.
- Source URL: https://clinicaltrials.gov/study/NCTXXXXXXXX
{extra_fields_prompt if extra_fields_prompt else ""}
IMPORTANT:
- One entry per trial
- Include ALL {len(trial_batch)} trials even if data is incomplete
- Use "N/A" for missing fields
- Report reductions as positive numbers
- Use actual Source URL from ClinicalTrials.gov

Return JSON:

{{
  "trials": [
    {{
      "Molecule": "{molecule}",
      "Trial Title": "",
      "Dosage": "",
      "Trial ID": "",
      "Phase": "",
      "Size": 0,
      "Primary Region": "",
      "Secondary Countries": "",
      "Start Date": "",
      "Completion Date": "",
      "Status": "",
      "MASH Outcome (%)": "",
      "MASH Duration": "",
      "MASH Rationale": "",
      "HbA1c Change (%)": "",
      "HbA1c Duration": "",
      "HbA1c Rationale": "",
      "Weight Loss (%)": "",
      "Weight Duration": "",
      "Weight Rationale": "",
      "ALT Reduction (%)": "",
      "ALT Duration": "",
      "ALT Rationale": "",
      "Company": "",
      "Trial_Study_Type": "",
      "Source URL": ""{extra_fields_json if extra_fields_json else ""}
    }}
  ]
}}

EXAMPLE of correct Primary Region format (MAX 2 regions):
- Trial in US only → Primary Region: "US"
- Trial in US + Canada + Mexico → Primary Region: "US"
- Trial in Germany, France, Spain, Italy → Primary Region: "EU"
- Trial in US + Germany + France + Spain → Primary Region: "US/EU"
- Trial in China only → Primary Region: "China"
- Trial in Japan + Korea + Singapore → Primary Region: "Japan"
- Global trial (US, EU, Japan, Korea, Latin America) → Primary Region: "US/EU" (top 2 only — drop Japan/Korea/LatAm)
- Global trial (US, EU, China, India, Japan) → Primary Region: "US/EU" (top 2 by site count)
- Trial mostly in EU with some Asian sites → Primary Region: "EU"
- Trial in Japan + many Asian countries → Primary Region: "Japan/Asia-Pacific"
"""

    response = await gemini_call_with_search(prompt, model=EXTRACTION_MODEL, response_mime_type="application/json")

    if not response or not response.strip():
        print(f"  ❌ Empty response received for batch")
        return {"trials": []}

    # FIRST ATTEMPT: Use json_repair library (handles most issues automatically)
    data = safe_json_parse(response, context="batch response")

    if data is not None:
        # Ensure it returns the expected structure
        if isinstance(data, dict) and "trials" in data:
            print(f"  ✓ json_repair successful - {len(data.get('trials', []))} trials")
            return data
        elif isinstance(data, list):
            print(f"  ✓ json_repair successful - {len(data)} trials (list)")
            return {"trials": data}
        elif isinstance(data, dict):
            # Check if it's a single trial object (has Trial ID or Molecule)
            if any(k in data for k in ["Trial ID", "Molecule", "trial_id", "molecule"]):
                print(f"  ℹ️  Single trial object returned, wrapping in list")
                return {"trials": [data]}
            # Check if keys look like trial data fields
            if any(k in data for k in ["Phase", "Size", "Dosage", "Status"]):
                print(f"  ℹ️  Single trial-like object returned, wrapping in list")
                return {"trials": [data]}
            # Log what we got for debugging
            print(f"  ⚠️  Unexpected dict structure. Keys: {list(data.keys())[:5]}")
            # Fall through to manual methods
        else:
            print(f"  ⚠️  Unexpected response type: {type(data)}")
            # Fall through to manual methods

    # SECOND ATTEMPT: Manual cleanup and parsing
    print(f"  🔧 json_repair returned unexpected structure, trying manual methods...")

    response = response.strip()
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()

    # Find the first { that starts valid JSON
    json_start = response.find('{')
    if json_start > 0:
        response = response[json_start:]

    # Sometimes Gemini adds google_search arrays - try to extract just the trials array
    if '"google_search"' in response:
        trials_match = response.find('"trials"')
        if trials_match != -1:
            bracket_start = response.find('[', trials_match)
            if bracket_start != -1:
                depth = 0
                for i, char in enumerate(response[bracket_start:], start=bracket_start):
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            trials_json = response[bracket_start:i+1]
                            parsed = safe_json_parse(trials_json, "trials array")
                            if parsed is not None:
                                if isinstance(parsed, list):
                                    return {"trials": parsed}

    # Extract just the first JSON object
    json_str = _extract_first_json_object(response)
    if json_str:
        parsed = safe_json_parse(json_str, "first JSON object")
        if parsed is not None:
            if isinstance(parsed, dict) and "trials" in parsed:
                return parsed
            elif isinstance(parsed, list):
                return {"trials": parsed}

    # THIRD ATTEMPT: Last resort field-by-field extraction
    print(f"  🔧 Attempting field-by-field extraction...")
    import re
    trials = []

    # Method 1: Try to find complete trial objects
    trial_pattern = r'\{\s*"Molecule"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(trial_pattern, response, re.DOTALL)

    for match in matches:
        try:
            trial_obj = safe_json_parse(match.group(0), "trial object")
            if trial_obj and (trial_obj.get("Trial ID") or trial_obj.get("NCT_ID")):
                trials.append(trial_obj)
        except:
            continue

    if trials:
        print(f"  ✓ Last-resort extraction (complete objects) - recovered {len(trials)} trials")
        return {"trials": trials}

    # Method 2: Extract individual fields from truncated JSON
    # This handles cases where even a single trial is truncated mid-field
    print(f"  🔧 Attempting field-by-field extraction...")
    field_patterns = {
        # Base clinical efficacy fields
        "Molecule": r'"Molecule"\s*:\s*"([^"]*)"',
        "Trial ID": r'"Trial ID"\s*:\s*"([^"]*)"',
        "Trial Title": r'"Trial Title"\s*:\s*"([^"]*)"',
        "Phase": r'"Phase"\s*:\s*"?([^",}]*)"?',
        "Size": r'"Size"\s*:\s*(\d+)',
        "Dosage": r'"Dosage"\s*:\s*"([^"]*)"',
        "Status": r'"Status"\s*:\s*"([^"]*)"',
        "Primary Region": r'"Primary Region"\s*:\s*"([^"]*)"',
        "Secondary Countries": r'"Secondary Countries"\s*:\s*"([^"]*)"',
        "HbA1c Change (%)": r'"HbA1c Change \(%\)"\s*:\s*"?([^",}]*)"?',
        "HbA1c Duration": r'"HbA1c Duration"\s*:\s*"([^"]*)"',
        "Weight Loss (%)": r'"Weight Loss \(%\)"\s*:\s*"?([^",}]*)"?',
        "Weight Duration": r'"Weight Duration"\s*:\s*"([^"]*)"',
        "MASH Outcome (%)": r'"MASH Outcome \(%\)"\s*:\s*"?([^",}]*)"?',
        "MASH Duration": r'"MASH Duration"\s*:\s*"([^"]*)"',
        "ALT Reduction (%)": r'"ALT Reduction \(%\)"\s*:\s*"?([^",}]*)"?',
        "ALT Duration": r'"ALT Duration"\s*:\s*"([^"]*)"',
        "HbA1c Rationale": r'"HbA1c Rationale"\s*:\s*"([^"]*)"',
        "Weight Rationale": r'"Weight Rationale"\s*:\s*"([^"]*)"',
        "ALT Rationale": r'"ALT Rationale"\s*:\s*"([^"]*)"',
        "MASH Rationale": r'"MASH Rationale"\s*:\s*"([^"]*)"',
        "Start Date": r'"Start Date"\s*:\s*"([^"]*)"',
        "Completion Date": r'"Completion Date"\s*:\s*"([^"]*)"',
        "Company": r'"Company"\s*:\s*"([^"]*)"',
        "Trial_Study_Type": r'"Trial_Study_Type"\s*:\s*"([^"]*)"',
        "Source URL": r'"Source URL"\s*:\s*"([^"]*)"',
        # Tolerability-specific fields
        "Drug Arm Size (N)": r'"Drug Arm Size \(N\)"\s*:\s*(\d+)',
        "Placebo Arm Size (N)": r'"Placebo Arm Size \(N\)"\s*:\s*(\d+)',
        "Discontinued AE Drug (N)": r'"Discontinued AE Drug \(N\)"\s*:\s*(\d+)',
        "Discontinued AE Placebo (N)": r'"Discontinued AE Placebo \(N\)"\s*:\s*(\d+)',
        "Comparator Name": r'"Comparator Name"\s*:\s*"([^"]*)"',
        "Comparator Arm Size (N)": r'"Comparator Arm Size \(N\)"\s*:\s*(\d+)',
        "Discontinued AE Comparator (N)": r'"Discontinued AE Comparator \(N\)"\s*:\s*(\d+)',
        "Common AEs": r'"Common AEs"\s*:\s*"([^"]*)"',
        "AE Severity": r'"AE Severity"\s*:\s*"([^"]*)"',
        "AE Persistence": r'"AE Persistence"\s*:\s*"([^"]*)"',
        "Management Required": r'"Management Required"\s*:\s*"([^"]*)"',
    }

    # Split JSON by trial boundaries (look for "Molecule" as trial start marker)
    trial_chunks = re.split(r'(?=\{\s*"Molecule")', response)

    for chunk in trial_chunks:
        if '"Molecule"' not in chunk:
            continue

        trial_obj = {}
        for field, pattern in field_patterns.items():
            match = re.search(pattern, chunk)
            if match:
                val = match.group(1).strip()
                if val and val not in ["N/A", "null", ""]:
                    trial_obj[field] = val

        # Only add if we have at least Trial ID or Molecule
        if trial_obj.get("Trial ID") or trial_obj.get("Molecule"):
            trials.append(trial_obj)

    if trials:
        print(f"  ✓ Last-resort extraction (field-by-field) - recovered {len(trials)} partial trials")
        return {"trials": trials}

    print(f"  ❌ All recovery attempts failed")
    return {"trials": []}


async def fetch_ctgov_dates(nct_ids: list) -> dict:
    """Fetch Study Start and Primary Completion dates from ClinicalTrials.gov API v2.

    Uses the authoritative CT.gov API instead of relying on Gemini for date accuracy.

    Returns:
        dict: {nct_id: {"Start Date": "YYYY-MM-DD", "Completion Date": "YYYY-MM-DD"}}
    """
    if not nct_ids:
        return {}

    async def fetch_one(session: aiohttp.ClientSession, nct_id: str):
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
        params = {"fields": "NCTId,StartDate,PrimaryCompletionDate"}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    status_mod = data.get("protocolSection", {}).get("statusModule", {})
                    start = status_mod.get("startDateStruct", {}).get("date") or "N/A"
                    completion = status_mod.get("primaryCompletionDateStruct", {}).get("date") or "N/A"
                    return nct_id, {"Start Date": start, "Completion Date": completion}
                else:
                    print(f"  ⚠️  CT.gov API {resp.status} for {nct_id}")
        except Exception as e:
            print(f"  ⚠️  CT.gov API error for {nct_id}: {e}")
        return nct_id, None

    print(f"\n🌐 Fetching dates from CT.gov API for {len(nct_ids)} NCT trials...")
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, nct_id) for nct_id in nct_ids]
        results = await asyncio.gather(*tasks)

    dates_map = {nct_id: dates for nct_id, dates in results if dates}
    print(f"  ✓ Retrieved dates for {len(dates_map)}/{len(nct_ids)} trials")
    return dates_map


async def extract_clinical_trial_data(molecule_name: str, max_trials: int = None,
                                       extra_fields_prompt: str = None,
                                       extra_fields_json: str = None) -> dict:
    """Main extraction function - Two-step Gemini extraction with multi-source search.

    Args:
        molecule_name: Name of the molecule to search for
        max_trials: Optional limit on number of trials to process (None = all)
        extra_fields_prompt: Optional additional prompt text for extracting extra fields
                            (e.g., tolerability fields for Dimension 6)
        extra_fields_json: Optional additional JSON schema fields to include in response

    Returns:
        dict with keys:
            - trials: list of trial data dictionaries
            - total_trials: total number of trials found
            - completeness: percentage of filled fields
    """
    import time
    start_time = time.time()

    print(f"\n{'='*80}")
    print("TWO-STEP GEMINI EXTRACTION")
    print("(Individual Registry Search)")
    print(f"{'='*80}")
    print(f"\nMolecule: {molecule_name}")
    print(f"Models:")
    print(f"  Search (Step 1):     {SEARCH_MODEL}")
    print(f"  Extraction (Step 2): {EXTRACTION_MODEL}")
    print(f"Strategy:")
    print(f"  Step 1: Search EACH registry individually (parallel)")
    print(f"  Step 2: Query detailed data in batches of {BATCH_SIZE} (accurate extraction)")
    print(f"  Registries: {len(REGISTRIES)}")
    print(f"  Max concurrent workers: {MAX_WORKERS}")

    # Step 1: Get trials from ALL registries in parallel
    step1_start = time.time()
    trial_list = await get_all_trials_parallel(molecule_name)
    step1_time = time.time() - step1_start
    print(f"\n⏱️  Step 1 time: {step1_time:.1f}s")

    if not trial_list:
        print("\n❌ No trials found across any registry")
        return {"trials": [], "total_trials": 0, "completeness": 0}

    # Optional: Filter to top N trials for faster processing
    if max_trials and len(trial_list) > max_trials:
        print(f"\n⚡ Limiting to top {max_trials} trials (out of {len(trial_list)}) for speed")
        trial_list = trial_list[:max_trials]

    # Step 2: Get details in batches (ASYNC)
    print(f"\n{'='*80}")
    print(f"STEP 2: Getting detailed data for {len(trial_list)} trials")
    print(f"Using model: {EXTRACTION_MODEL} (optimized for accurate extraction)")
    print(f"Processing ASYNCHRONOUSLY with up to {MAX_WORKERS} concurrent batches")
    print(f"{'='*80}\n")

    step2_start = time.time()
    all_trials = []
    num_batches = (len(trial_list) + BATCH_SIZE - 1) // BATCH_SIZE

    # Create all batches
    batches = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(trial_list))
        batch = trial_list[start_idx:end_idx]
        batches.append((batch_idx, batch))

    async def process_batch_async(batch_idx: int, batch: list):
        """Process a single batch asynchronously with rate-limit staggering."""
        # Stagger starts to respect rate limits
        stagger_delay = (batch_idx % MAX_WORKERS) * (RATE_LIMIT_DELAY / MAX_WORKERS)
        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)

        batch_start = time.time()
        print(f"\nBatch {batch_idx + 1}/{num_batches} ({len(batch)} trials) - starting")

        batch_data = await get_trial_details(molecule_name, batch, extra_fields_prompt, extra_fields_json)
        batch_trials = batch_data.get("trials", [])

        batch_time = time.time() - batch_start
        print(f"  ✓ Batch {batch_idx + 1}/{num_batches} - completed ({len(batch_trials)} trials) in {batch_time:.1f}s")
        return batch_trials

    # Process batches with controlled concurrency using semaphore
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    async def process_with_semaphore(batch_idx: int, batch: list):
        async with semaphore:
            return await process_batch_async(batch_idx, batch)

    # Run all batches concurrently with semaphore limit
    tasks = [process_with_semaphore(batch_idx, batch) for batch_idx, batch in batches]
    results = await asyncio.gather(*tasks)

    # Flatten results (filter out non-dict entries from malformed JSON)
    for batch_trials in results:
        all_trials.extend([t for t in batch_trials if isinstance(t, dict)])

    step2_time = time.time() - step2_start
    print(f"\n⏱️  Step 2 time: {step2_time:.1f}s ({num_batches} batches, {step2_time/num_batches:.1f}s avg/batch)")

    # Step 2.5: Override Start Date / Completion Date from CT.gov API v2
    # Gemini often guesses dates incorrectly — the API is the authoritative source
    nct_ids = [
        t.get("Trial ID", "").split("(")[0].strip().split(" ")[0].strip()
        for t in all_trials
        if isinstance(t, dict) and t.get("Trial ID", "").upper().startswith("NCT")
    ]
    nct_ids = [nid for nid in nct_ids if nid]  # remove empty strings

    if nct_ids:
        dates_map = await fetch_ctgov_dates(nct_ids)
        overrides = 0
        for trial in all_trials:
            if not isinstance(trial, dict):
                continue
            raw_id = trial.get("Trial ID", "")
            nct_id = raw_id.split("(")[0].strip().split(" ")[0].strip()
            if nct_id in dates_map:
                trial["Start Date"] = dates_map[nct_id]["Start Date"]
                trial["Completion Date"] = dates_map[nct_id]["Completion Date"]
                overrides += 1
        print(f"  ✓ Overrode dates for {overrides} trials from CT.gov API")

    # Calculate completeness - count non-N/A values across all fields
    # Exclude metadata fields that are always present
    metadata_fields = ["Molecule", "Trial ID", "Trial Title", "Source URL"]

    total_fields = 0
    filled_fields = 0

    for trial in all_trials:
        if not isinstance(trial, dict):
            continue
        for field, value in trial.items():
            # Skip metadata fields
            if field in metadata_fields:
                continue

            total_fields += 1

            # Count as filled if not N/A, not empty, and not 0
            if value and str(value).strip() not in ["N/A", "n/a", "", "0"]:
                filled_fields += 1

    completeness = (filled_fields / total_fields * 100) if total_fields > 0 else 0

    print(f"\n📊 Data Completeness: {completeness:.1f}%")
    print(f"   Filled: {filled_fields}/{total_fields} fields (excluding Molecule, Trial ID, Source URL)")
    print(f"   N/A or empty: {total_fields - filled_fields} fields")

    # Final timing summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ COMPLETED IN {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"{'='*80}")
    print(f"Timing Breakdown:")
    print(f"  Step 1 (parallel registry search): {step1_time:.1f}s")
    print(f"  Step 2 (batch extraction):         {step2_time:.1f}s")
    print(f"  Total:                             {total_time:.1f}s")
    print(f"{'='*80}\n")

    return {
        "trials": all_trials,
        "total_trials": len(all_trials),
        "completeness": completeness,
        "extraction_time": total_time
    }