"""
AI Matcher Module

Core resume-to-job matching logic with multi-strategy requirement checking.
"""

import json
import re
from datetime import datetime
from typing import Optional, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import time

# Import unified configuration
from app.config.app_config import (
    ScoringConfig,
    TextProcessingConfig,
    PatternConfig
)

# Import LLM factory (no more module-level instantiation!)
from app.processors.llm_factory import get_llm_client

# Import logging
from app.utils.logger import get_logger, log_analysis_start, log_analysis_complete

logger = get_logger(__name__)

# --- Helpers for JD parsing and normalization ---

def _extract_bullets(text: str, header: str) -> list[str]:
    lines = text.splitlines()
    out, grab = [], False
    header = header.strip().lower() + ":"
    for ln in lines:
        s = ln.strip()
        if not s:
            if grab: break
            continue
        low = s.lower()
        if low.startswith(header):
            grab = True
            continue
        if grab and re.match(r"^[A-Za-z][^:]{0,60}:\s*$", s):
            break
        if grab and (s.startswith("-") or s.startswith("*") or s.startswith("•")):
            out.append(s[1:].strip())

    return out

def _window(text: str, start: int, end: int, pad: int = None) -> str:
    """
    Extract a text window around a match with context padding.
    
    Args:
        text: Full text to extract from
        start: Start index of match
        end: End index of match
        pad: Padding characters before/after (uses config default if None)
    
    Returns:
        Snippet with context
    """
    if pad is None:
        pad = TextProcessingConfig.TEXT_WINDOW_PADDING
    
    s = max(0, start - pad)
    e = min(len(text), end + pad)
    snippet = text[s:e].replace("\n", " ")
    snippet = re.sub(r"^\S*\s", "", snippet)
    snippet = re.sub(r"\s\S*$", "", snippet)
    return snippet

# define clean_text BEFORE using it anywhere
def clean_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s+./-]", "", text.lower().strip()).split())

# === canonical regex patterns for must-haves / nice-to-haves ===
REQ_PATTERNS = {
    "proficiency with node.js": [r"\bnode\.?js\b"],
    "proficiency with react": [r"\breact\b"],
    "proficiency with mysql": [r"\bmy\s*sql\b|\bmysql\b"],
    "proficiency with html/css": [r"\bhtml\b", r"\bcss\b"],
    "proficiency with github": [r"\bgithub\b|\bgit\s*hub\b"],  # removed \bgit\b
}

PREF_PATTERNS = {
    "bootstrap": [r"\bbootstrap\b"],
    "ci/cd (github actions/gitlab ci)": [r"\bci/?cd\b|\bgithub actions\b|\bgitlab ci\b|\bjenkins\b|\bazure devops\b"],
    "google cloud platform": [r"\bgoogle cloud platform\b|\bgoogle cloud\b|\bgcp\b"],
    "mocha/chai (tdd)": [r"\bmocha\b|\bchai\b|\btdd\b"],
    "ios/android app development": [r"\bios\b|\bandroid\b|\breact native\b"],
    "object-oriented programming (oop)": [r"\boop\b|\bobject[- ]oriented\b"],
    "agile/scrum": [r"\bagile\b|\bscrum\b"],
}

NEG_KEYWORDS = {
    "proficiency with node.js": ["node.js", "nodejs", "node"],
    "proficiency with react": ["react"],
    "proficiency with mysql": ["mysql", "my sql"],
    "proficiency with html/css": ["html", "css"],
    "proficiency with github": ["github", "git hub"],

    "bootstrap": ["bootstrap"],
    "ci/cd (github actions/gitlab ci)": ["ci/cd", "github actions", "gitlab ci", "jenkins", "azure devops"],
    "google cloud platform": ["google cloud platform", "google cloud", "gcp"],
    "mocha/chai (tdd)": ["mocha", "chai", "tdd"],
    "ios/android app development": ["ios", "android", "react native"],
    "object-oriented programming (oop)": ["oop", "object oriented"],
    "agile/scrum": ["agile", "scrum"],
}

# --- Configuration References ---
# These are now loaded from app_config, but kept as module-level variables for backward compatibility
NON_BLOCKING_MUST_PHRASES = PatternConfig.NON_BLOCKING_MUST_PHRASES
SOFT_KEYWORDS = PatternConfig.SOFT_KEYWORDS
NEGATION_TOKENS = TextProcessingConfig.NEGATION_TOKENS
NEGATION_NEAR_GAP = TextProcessingConfig.NEGATION_DETECTION_GAP

# Compile technical cues pattern
TECH_CUES = re.compile(PatternConfig.TECH_CUES_PATTERN, re.I)


def _is_negated(snippet: str, keywords: list[str]) -> bool:
    s = snippet.lower()
    for kw in keywords:
        # allow a wider gap between the negation token and the keyword
        if re.search(
            rf"{NEGATION_TOKENS}.{{0,{NEGATION_NEAR_GAP}}}\b{re.escape(kw.lower())}\b",
            s, flags=re.I
        ):
            return True
    return False

def is_non_blocking_requirement(req: str) -> bool:
    rl = (req or "").lower()
    return any(p in rl for p in NON_BLOCKING_MUST_PHRASES)

def _normalize_req_name(name: str) -> str:
    return clean_text(name).strip()

def _auto_is_soft_requirement(req: str) -> bool:
    r = req.lower()
    has_soft = any(k in r for k in SOFT_KEYWORDS)
    has_tech = bool(TECH_CUES.search(r))
    return has_soft and not has_tech

def _is_non_blocking_requirement(req: str) -> bool:
    return SOFT_MUSTS_NONBLOCKING and _auto_is_soft_requirement(req)

def _keywords_from_requirement(req: str) -> list[str]:
    """
    Extract keywords from a requirement string.
    
    Args:
        req: Requirement string
    
    Returns:
        List of keywords (unique, order-preserved, capped by config)
    """
    # keep capitalized tech words & tokens with letters+digits or length >=3
    toks = re.findall(r"[A-Za-z0-9\.\+\-/]+", req)
    stop = {"and","or","with","in","of","for","to","a","the","on","using","experience","proficiency"}
    kws = [t for t in toks if t.lower() not in stop and (re.search(r"\d", t) or len(t) >= 3)]
    # Cap to configured maximum
    max_keywords = TextProcessingConfig.MAX_KEYWORDS_PER_REQUIREMENT
    return list(dict.fromkeys(kws))[:max_keywords]

def _simple_keyword_evidence(text: str, req: str) -> Optional[str]:
    """
    Find simple keyword evidence in text.
    
    Args:
        text: Full text to search
        req: Requirement string
    
    Returns:
        Evidence snippet or None
    """
    kws = _keywords_from_requirement(req)
    if not kws:
        return None
    # Find a line/snippet that contains at least one keyword (non-negated)
    for kw in kws:
        m = re.search(re.escape(kw), text, flags=re.I)
        if not m:
            continue
        snippet = _window(text, m.start(), m.end())  # Uses config default
        # if snippet contains explicit negation + kw, skip
        if re.search(rf"{NEGATION_TOKENS}.{{0,{NEGATION_NEAR_GAP}}}\b{re.escape(kw)}\b", snippet, flags=re.I):
            continue
        return snippet
    return None


def _snippet_is_negated(snippet: str, req: str) -> bool:
    kws = _keywords_from_requirement(req)
    s = snippet.lower()
    for kw in kws:
        if re.search(
            rf"{NEGATION_TOKENS}.{{0,{NEGATION_NEAR_GAP}}}\b{re.escape(kw.lower())}\b",
            s, flags=re.I
        ):
            return True
    return False

def _text_has_negated_keyword(text: str, kw: str) -> bool:
    s = text.lower()
    k = re.escape(kw.lower())
    # “no experience with/in X”, “… without X …”, “lacking X”, “X … no experience”
    if re.search(rf"\bno\s+experience\s+(with|in)\s+{k}\b", s): return True
    if re.search(rf"\bwithout\b.{0,50}\b{k}\b", s, flags=re.S): return True
    if re.search(rf"\black(?:ing)?\s+{k}\b", s): return True
    if re.search(rf"\b{k}\b.{0,50}\bno\s+experience\b", s, flags=re.S): return True
    # general proximity with your NEGATION_TOKENS window
    if re.search(rf"{NEGATION_TOKENS}.{{0,{NEGATION_NEAR_GAP}}}\b{k}\b", s, flags=re.I): return True
    return False

# Build normalized-key lookups so punctuation/case differences still match
REQ_KEYS_NORM = None
PREF_KEYS_NORM = None
def _build_norm_maps():
    global REQ_KEYS_NORM, PREF_KEYS_NORM
    REQ_KEYS_NORM = { _normalize_req_name(k): k for k in REQ_PATTERNS.keys() }
    PREF_KEYS_NORM = { _normalize_req_name(k): k for k in PREF_PATTERNS.keys() }

_build_norm_maps()

def analyze_resume_with_job(
    resume_text: str,
    job_description: str,
    llm = None
) -> dict:
    """
    Analyze resume against job description.
    
    Args:
        resume_text: Full resume text
        job_description: Full job description text
        llm: LLM client instance (uses factory default if None)
    
    Returns:
        dict with match results, requirements analysis, and score
    """
    try:
        from app.config.app_config import LLMConfig
        
        # Track timing
        start_time = time.time()
        
        # Log analysis start
        log_analysis_start(len(resume_text), len(job_description))
        
        # Use provided LLM or get from factory
        if llm is None:
            llm = get_llm_client()
        
        prompt = create_matching_prompt(resume_text, job_description)
        logger.info("Sending prompt to LLM...")
        
        llm_start = time.time()
        raw_response = llm.ask(prompt, max_tokens=LLMConfig.MAX_TOKENS)
        llm_duration = (time.time() - llm_start) * 1000
        
        logger.debug(f"LLM response received in {llm_duration:.2f}ms")

        parsed = safely_parse_json(raw_response)
        if "error" in parsed:
            logger.error(f"Error parsing LLM response: {parsed['error']}")
            logger.debug(f"Raw response: {raw_response[:500]}...")
            return {"error": parsed["error"], "match_percentage": 0.0}

        parsed.setdefault("skills", [])
        parsed.setdefault("certifications", [])
        parsed.setdefault("must_haves", [])
        parsed.setdefault("nice_to_haves", [])

        musts_from_jd = _extract_bullets(job_description, "Qualifications") or _extract_bullets(job_description, "Required")
        prefs_from_jd = (
            _extract_bullets(job_description, "Preferred")
            or _extract_bullets(job_description, "Nice to have")
            or _extract_bullets(job_description, "Nice-to-haves")
            or _extract_bullets(job_description, "Nice-to-have")
        )

        if musts_from_jd:
            parsed["must_haves"] = musts_from_jd
        if prefs_from_jd:
            parsed["nice_to_haves"] = prefs_from_jd

        experience = safe_int_convert(parsed.get("work_experience_years"))
        if experience is None or experience < 1:
            experience = calculate_experience_years(resume_text)

        must_have_results = []
        for req in parsed["must_haves"]:
            met, evidence = check_requirement(req, parsed["skills"] + parsed["certifications"], experience, resume_text)
            must_have_results.append({"requirement": req, "met": bool(met), "evidence": evidence})

        nice_have_results = []
        for req in parsed["nice_to_haves"]:
            met, evidence = check_requirement(req, parsed["skills"] + parsed["certifications"], experience, resume_text)
            nice_have_results.append({"requirement": req, "met": bool(met), "evidence": evidence})

        match_percentage = calculate_match_score(must_have_results, nice_have_results)
        
        # Log completion
        total_duration = (time.time() - start_time) * 1000
        log_analysis_complete(match_percentage, total_duration)
        
        logger.info(
            f"Analysis complete | Match: {match_percentage}% | "
            f"Must-haves: {len(must_have_results)} | "
            f"Nice-to-haves: {len(nice_have_results)}"
        )

        return {
            "name": parsed.get("name"),
            "location": format_location(parsed),
            "degree": parsed.get("degree"),
            "work_experience_years": experience,
            "work_permit": determine_work_permit(parsed),
            "skills": parsed["skills"],
            "certifications": parsed["certifications"],
            "must_have_results": must_have_results,
            "nice_have_results": nice_have_results,
            "match_percentage": match_percentage,
            "summary": parsed.get("summary", "No summary provided"),
            "error": None
        }

    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}", exc_info=True)
        return {
            "error": f"Processing failed: {str(e)}",
            "match_percentage": 0.0,
            "must_have_results": [],
            "nice_have_results": []
        }

def check_requirement(requirement: str, candidate_items: List[str], experience: Optional[int], resume_text: str) -> tuple:
    met_llm_logic = is_requirement_met(requirement, candidate_items, experience, resume_text)

    # regex maps (Node.js/React/MySQL/etc.) stay as before…
    norm = _normalize_req_name(requirement)
    neg_hit = False
    for kw in _keywords_from_requirement(requirement):
        if _text_has_negated_keyword(resume_text, kw):
            neg_hit = True
            break
    if neg_hit:
        # wipe any optimistic model signal
        met_llm_logic = False
    if REQ_KEYS_NORM and norm in REQ_KEYS_NORM:
        key = REQ_KEYS_NORM[norm]
        met_regex, ev = _has_evidence_regex(resume_text, REQ_PATTERNS.get(key, []), negation_keywords=NEG_KEYWORDS.get(key, []))
        if met_regex and not _snippet_is_negated(ev, requirement):
            return True, ev
    if PREF_KEYS_NORM and norm in PREF_KEYS_NORM:
        key = PREF_KEYS_NORM[norm]
        met_regex, ev = _has_evidence_regex(resume_text, PREF_PATTERNS.get(key, []), negation_keywords=NEG_KEYWORDS.get(key, []))
        if met_regex and not _snippet_is_negated(ev, requirement):
            return True, ev


    # semantic best-line
    evidence = find_evidence_for_requirement(requirement, resume_text)
    
    if evidence:
        evl = evidence.lower()
        kws = _keywords_from_requirement(requirement)
        for kw in kws:
            k = kw.lower()
            if (
                re.search(rf"\bno\s+experience\s+(with|in)\s+{re.escape(k)}\b", evl)
                or re.search(rf"\bwithout\b.*\b{re.escape(k)}\b", evl)
                or re.search(rf"\black(?:ing)?\s+{re.escape(k)}\b", evl)
                or re.search(rf"\b{re.escape(k)}\b.*\bno\s+experience\b", evl)
            ):
                evidence = ""  # treat as negative evidence
                break
    
    if evidence and _snippet_is_negated(evidence, requirement):
        evidence = ""  # discard negated “evidence”

    # if model said True but we have no clean evidence, try a simple keyword search
    if met_llm_logic and (not evidence or "no direct evidence" in evidence.lower()):
        kw_ev = _simple_keyword_evidence(resume_text, requirement)
        if kw_ev:
            return True, kw_ev

    # final guard
    if not evidence or "no direct evidence" in evidence.lower():
        return False, "No direct evidence found"
    return bool(met_llm_logic), evidence


def create_matching_prompt(resume_text: str, job_description: str) -> str:
    return f"""Analyze this resume against the job description and return STRICT JSON only.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Output JSON with these EXACT fields:
{{
  "name": "extracted name or null",
  "city": "extracted city or null",
  "country": "extracted country or null",
  "degree": "highest degree mentioned",
  "work_experience_years": "total years experience (number)",
  "work_permit": "only if explicitly mentioned",
  "skills": ["all technical skills from RESUME only"],
  "certifications": ["only formal certs from RESUME"],
  "must_haves": ["absolute requirements from JOB DESCRIPTION"],
  "nice_to_haves": ["preferred qualifications from JOB DESCRIPTION"],
  "summary": "2–3 lines summarizing the RESUME only"
}}

CRITICAL RULES:
- Output a SINGLE JSON object. No markdown, no backticks, no code, no prefixes/suffixes.
- Summarize the RESUME only. Do NOT summarize or paraphrase the job description.
- Extract must_haves and nice_to_haves from the JOB DESCRIPTION only.
- If any field cannot be determined, use null.
"""


def calculate_match_score(must_have_results, nice_have_results) -> float:
    """
    Calculate final match score using configured weights and thresholds.
    
    Args:
        must_have_results: List of must-have requirement check results
        nice_have_results: List of nice-to-have requirement check results
    
    Returns:
        Match percentage (0.0-100.0)
    """
    if not must_have_results:
        return 0.0

    def strength(ev: str) -> float:
        """
        Determine evidence strength based on configured multipliers.
        
        Args:
            ev: Evidence string
        
        Returns:
            Strength multiplier (0.0, 0.5, or 1.0 by default)
        """
        evl = (ev or "").lower()
        if "no direct evidence found" in evl or not evl.strip():
            return ScoringConfig.NO_EVIDENCE_MULTIPLIER
        if any(w in evl for w in ScoringConfig.WEAK_EVIDENCE_PATTERNS):
            return ScoringConfig.WEAK_EVIDENCE_MULTIPLIER
        return ScoringConfig.STRONG_EVIDENCE_MULTIPLIER

    # Exclude soft musts (e.g., "curiosity") from gating/counting
    hard_musts = [r for r in must_have_results if not _is_non_blocking_requirement(r["requirement"])]

    # Compute ratio of satisfied hard musts
    if hard_musts:
        hard_strengths = [strength(r["evidence"]) if r.get("met") else 0.0 for r in hard_musts]

        # Allow 0% only when zero hard musts are satisfied
        if sum(1 for s in hard_strengths if s > 0.0) == 0:
            return 0.0

        must_ratio = (sum(1 for s in hard_strengths if s > 0.0)) / len(hard_musts)
    else:
        must_ratio = 1.0  # no hard musts listed => treat as fully satisfied

    # Niceties ratio (proportion satisfied)
    nice_strengths = [strength(r["evidence"]) if r.get("met") else 0.0 for r in nice_have_results]
    nice_ratio = (sum(1 for s in nice_strengths if s > 0.0) / len(nice_have_results)) if nice_have_results else 0.0

    # --- Curved scoring for musts ---
    # Uses configurable exponent for non-linear scoring
    # Higher exponent = steeper curve (rewards complete matches more)
    hard_score = (
        ScoringConfig.MAX_MUST_HAVE_SCORE * (must_ratio ** ScoringConfig.SCORING_EXPONENT)
        if must_ratio > 0 else 0.0
    )

    # Nice-to-haves add bonus points
    nice_score = ScoringConfig.MAX_NICE_TO_HAVE_SCORE * nice_ratio

    total = hard_score + nice_score

    # --- Low-coverage floor ---
    # If at least one hard must is met but not all, ensure a minimal dynamic signal
    has_any_hard = bool(hard_musts) and must_ratio > 0.0
    all_hard_perfect = must_ratio >= 1.0

    if has_any_hard and not all_hard_perfect:
        total = max(total, ScoringConfig.LOW_COVERAGE_FLOOR)

    return round(min(ScoringConfig.MAX_TOTAL_SCORE, total), 2)


def is_requirement_met(requirement: str, candidate_items: List[str], experience: Optional[int], resume_text: str) -> bool:
    req_lower = requirement.lower()
    full_text = " ".join(candidate_items + [resume_text]).lower()

    if any(term in req_lower for term in ["ph.d.", "phd", "doctorate"]):
        if any(term in full_text for term in ["ph.d.", "phd", "doctorate"]):
            return True

    year_match = re.search(r"(\d+)\+? years?", req_lower)
    if year_match and "experience" in req_lower and experience is not None:
        return experience >= int(year_match.group(1))

    # Tools derived from requirement
    tools = extract_tools_from_requirement(requirement)
    if tools:
        combined = " ".join([clean_text(i) for i in candidate_items + [resume_text]])
        for t in tools:
            if t.lower() in combined and not _text_has_negated_keyword(combined, t):
                return True

    # Python/programming explicit
    if ("programming" in req_lower or "python" in req_lower) and "python" in full_text:
        if not _text_has_negated_keyword(full_text, "python"):
            return True

    if "intellectual curiosity" in req_lower or "drive to excel" in req_lower:
        return False

    return semantic_match_with_context(requirement, candidate_items + [resume_text])

def semantic_match_with_context(requirement: str, items: List[str], threshold: float = None) -> bool:
    """
    Check if requirement matches any item using semantic similarity.
    
    Args:
        requirement: Requirement string
        items: List of candidate items/text
        threshold: Similarity threshold (uses config default if None)
    
    Returns:
        True if any item exceeds threshold
    """
    if threshold is None:
        threshold = ScoringConfig.SEMANTIC_MATCH_THRESHOLD
    
    from app.processors.model_factory import get_semantic_model
    model = get_semantic_model()
    req_emb = model.encode([clean_text(requirement)])
    item_embs = model.encode([clean_text(item) for item in items])
    similarities = cosine_similarity(req_emb, item_embs)[0]
    return bool(np.max(similarities) > threshold) if similarities.size > 0 else False

def find_evidence_for_requirement(requirement: str, resume_text: str) -> str:
    """
    Find best evidence line for a requirement using semantic similarity.
    
    Args:
        requirement: Requirement string
        resume_text: Full resume text
    
    Returns:
        Best matching line or "No direct evidence found"
    """
    from app.processors.model_factory import get_semantic_model
    model = get_semantic_model()
    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]
    req_emb = model.encode([clean_text(requirement)])
    line_embs = model.encode([clean_text(line) for line in lines])
    similarities = cosine_similarity(req_emb, line_embs)[0]
    if similarities.size > 0:
        best_idx = np.argmax(similarities)
        if similarities[best_idx] > ScoringConfig.EVIDENCE_EXTRACTION_THRESHOLD:
            return lines[best_idx]
    return "No direct evidence found"

def _has_evidence_regex(resume_text: str, patterns: List[str], negation_keywords: Optional[List[str]] = None) -> Tuple[bool, str]:
    for pat in patterns or []:
        m = re.search(pat, resume_text, flags=re.I)
        if m:
            snippet = _window(resume_text, m.start(), m.end(), pad=80)
            # If caller provided keywords and the snippet negates them, treat as no evidence
            if negation_keywords and _is_negated(snippet, negation_keywords):
                continue
            return True, snippet
    return False, ""

def extract_tools_from_requirement(text: str) -> List[str]:
    tools = []

    # Match common phrases like "proficiency with X, Y and Z"
    matches = re.findall(r"(?:with|using|proficiency in)\s+([A-Za-z0-9,\s]+)", text, re.IGNORECASE)
    for match in matches:
        tools.extend([t.strip() for t in re.split(r",|\band\b", match)])

    # Add direct mentions of known tools from requirement text
    tools += [tool for tool in KNOWN_TOOLS if tool.lower() in text.lower()]
    return list(set(t for t in tools if t))

def safely_parse_json(raw_response: str) -> dict:
    if not raw_response.strip():
        return {"error": "Empty response from DeepSeek"}
    try:
        parsed = json.loads(raw_response.strip())
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")
        return parsed
    except json.JSONDecodeError:
        try:
            start = raw_response.find('{')
            end = raw_response.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            return json.loads(raw_response[start:end])
        except Exception as e:
            return {"error": f"JSON parsing failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected parsing error: {str(e)}"}

def safe_int_convert(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            if re.fullmatch(r"\d+", value.strip()):
                return int(value.strip())
            return None
        return int(value)
    except (ValueError, TypeError):
        return None

def calculate_experience_years(resume_text: str) -> Optional[int]:
    current_year = datetime.now().year
    year_ranges = re.findall(r'(19\d{2}|20\d{2})\s*[\u2013\-]\s*(19\d{2}|20\d{2}|present|Present)', resume_text)
    total_years = 0
    for start, end in year_ranges:
        try:
            start_year = int(start)
            end_year = current_year if end.lower() == "present" else int(end)
            if start_year <= end_year <= current_year:
                total_years += (end_year - start_year)
        except:
            continue
    return total_years if total_years > 0 else None

def determine_work_permit(parsed: dict) -> Optional[str]:
    permit = parsed.get("work_permit")
    if permit and is_valid_value(permit) and permit.strip().lower() not in ["canada", "united states", "us", "uk"]:
        return permit.strip()
    country = parsed.get("country")
    if country and country.lower() == "canada":
        return "Authorized to work in Canada (inferred)"
    return None

def format_location(parsed: dict) -> Optional[str]:
    city = parsed.get("city")
    country = parsed.get("country")
    return f"{city}, {country}" if city and country else city or country or None

def is_valid_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        value = value.strip().lower()
        return value and not value.startswith("only if") and value not in ["n/a", "none", "null", "not mentioned", "no certifications mentioned"]
    return True


# ============================================================================
# NEW AI-DRIVEN SCORING LOGIC
# ============================================================================

def create_ai_scoring_system_prompt() -> str:
    """
    Get the AI scoring system prompt from configuration.
    
    Returns:
        System prompt for AI-driven scoring
    """
    from app.config.app_config import AI_SCORING_SYSTEM_PROMPT
    return AI_SCORING_SYSTEM_PROMPT


def create_ai_scoring_user_prompt(resume_text: str, job_description: str) -> str:
    """
    Create the user prompt for AI-driven scoring.
    
    Args:
        resume_text: Full resume text
        job_description: Full job description text
    
    Returns:
        Formatted user prompt
    """
    return f"""**RESUME:**
{resume_text}

**JOB DESCRIPTION:**
{job_description}

Analyze the resume against the job description and return the scoring JSON following ALL the rules in the system prompt."""


def normalize_analyze_ai_response(ai: dict) -> dict:
    # Guarantee all mandatory keys/structure exist for client contract
    summary = ai.get("summary") or ""
    ci = ai.get("candidate_info", {})
    ab = ai.get("analysis_breakdown", ai)
    # Accept field present on top-level or under .analysis_breakdown (for LLM flexibility)
    def get_nested(d, key, default):
        return d.get(key) if key in d else ai.get(key, default)

    candidate_info = {
        "name": str(get_nested(ci, "name", "")),
        "location": str(get_nested(ci, "location", "")),
        "work_experience_years": int(get_nested(ci, "work_experience_years", get_nested(ai, "work_experience_years", 0)))
    }
    analysis_breakdown = {
        "location_match"     : bool(get_nested(ab, "location_match", False)),
        "must_haves_total"   : int(get_nested(ab, "must_haves_total", 0)),
        "must_haves_met"     : int(get_nested(ab, "must_haves_met", 0)),
        "nice_to_haves_total": int(get_nested(ab, "nice_to_haves_total", 0)),
        "nice_to_haves_met"  : int(get_nested(ab, "nice_to_haves_met", 0)),
    }
    result = {
        "match_percentage": float(get_nested(ai, "match_percentage", 0)),
        "summary": summary,
        "candidate_info": candidate_info,
        "analysis_breakdown": analysis_breakdown,
        "error": ai.get("error") # could be None/null or a string
    }
    return result


def analyze_resume_with_job_ai(
    resume_text: str,
    job_description: str,
    llm = None
) -> dict:
    """
    Analyze resume against job description using AI-driven scoring logic.
    
    This function uses the new AI scoring system that:
    - Applies location caps
    - Enforces zero-tolerance must-haves
    - Recognizes synonyms
    - Ignores soft skills
    - Calculates dates for experience
    - Adds nice-to-have bonuses
    
    Args:
        resume_text: Full resume text
        job_description: Full job description text
        llm: LLM client instance (uses factory default if None)
    
    Returns:
        dict with AI-driven analysis results and breakdown
    """
    try:
        from app.config.app_config import LLMConfig
        
        # Track timing
        start_time = time.time()
        
        # Log analysis start
        log_analysis_start(len(resume_text), len(job_description))
        logger.info("Using NEW AI-driven scoring logic")
        
        # Use provided LLM or get from factory
        if llm is None:
            llm = get_llm_client()
        
        # Create prompts
        system_prompt = create_ai_scoring_system_prompt()
        user_prompt = create_ai_scoring_user_prompt(resume_text, job_description)
        
        # Combine into single prompt for DeepSeek (which doesn't support separate system/user messages)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        logger.info("Sending AI scoring prompt to LLM...")
        
        llm_start = time.time()
        raw_response = llm.ask(full_prompt, max_tokens=LLMConfig.MAX_TOKENS)
        llm_duration = (time.time() - llm_start) * 1000
        
        logger.debug(f"LLM response received in {llm_duration:.2f}ms")
        logger.debug(f"Raw LLM response (first 500 chars): {raw_response[:500]}")
        
        # Parse the AI's JSON response
        parsed = safely_parse_json(raw_response)
        
        if "error" in parsed:
            logger.error(f"Error parsing AI scoring response: {parsed['error']}")
            logger.debug(f"Full raw response: {raw_response}")
            # Return normalized structure—empty contract
            return normalize_analyze_ai_response({"error": parsed["error"]})
        
        # Validate required fields
        # required_fields = [
        #     "match_percentage", "location_match", "must_haves_total", 
        #     "must_haves_met", "nice_to_haves_total", "nice_to_haves_met", 
        #     "work_experience_years", "summary"
        # ]
        
        # missing_fields = [field for field in required_fields if field not in parsed]
        # if missing_fields:
        #     logger.error(f"AI response missing required fields: {missing_fields}")
        #     return {
        #         "error": f"AI response incomplete. Missing: {', '.join(missing_fields)}",
        #         "match_percentage": 0,
        #         "candidate_info": {},
        #         "analysis_breakdown": {},
        #         "summary": "AI response did not follow the required format"
        #     }
        
        # Extract candidate info from resume (basic parsing)
        # candidate_info = {
        #     "name": None,  # Could extract with NER if needed
        #     "location": None,  # Could extract with NER if needed
        #     "work_experience_years": parsed.get("work_experience_years")
        # }
        
        # Structure the analysis breakdown
        # analysis_breakdown = {
        #     "location_match": parsed["location_match"],
        #     "must_haves_total": parsed["must_haves_total"],
        #     "must_haves_met": parsed["must_haves_met"],
        #     "nice_to_haves_total": parsed["nice_to_haves_total"],
        #     "nice_to_haves_met": parsed["nice_to_haves_met"]
        # }
        
        normalized_ai = normalize_analyze_ai_response(parsed)
        
        # Log completion
        total_duration = (time.time() - start_time) * 1000
        log_analysis_complete(
            normalized_ai["match_percentage"],
            total_duration,
        )
        
        logger.info(
            f"AI Scoring complete | Match: {normalized_ai['match_percentage']}% | "
            f"Must-haves: {normalized_ai['analysis_breakdown']['must_haves_met']}/{normalized_ai['analysis_breakdown']['must_haves_total']} | "
            f"Nice-to-haves: {normalized_ai['analysis_breakdown']['nice_to_haves_met']}/{normalized_ai['analysis_breakdown']['nice_to_haves_total']} | "
            f"Location match: {normalized_ai['analysis_breakdown']['location_match']}"
        )
        
        return normalized_ai
        
    except Exception as e:
        logger.error(f"Unexpected error during AI scoring analysis: {str(e)}", exc_info=True)
        return {
            "error": f"AI scoring processing failed: {str(e)}",
            "match_percentage": 0,
            "candidate_info": {},
            "analysis_breakdown": {},
            "summary": "System error occurred during analysis"
        }


# Reference to known tools (used in extract_tools_from_requirement)
KNOWN_TOOLS = ScoringConfig.KNOWN_TOOLS
# Reference to soft musts nonblocking flag
SOFT_MUSTS_NONBLOCKING = ScoringConfig.SOFT_MUSTS_NONBLOCKING