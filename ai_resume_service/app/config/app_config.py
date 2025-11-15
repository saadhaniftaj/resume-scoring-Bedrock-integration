"""
Unified Application Configuration Module

This module provides centralized configuration management for the entire application.
All configurable parameters are loaded from environment variables with sensible defaults.

Usage:
    from app.config.app_config import LLMConfig, ScoringConfig

    # Access configuration
    max_tokens = LLMConfig.MAX_TOKENS
    must_have_weight = ScoringConfig.MUST_HAVE_WEIGHT
"""

import os
import json
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _parse_list(env_var: str, default: List[str], delimiter: str = ",") -> List[str]:
    """
    Parse comma-separated environment variable into list.
    
    Args:
        env_var: Environment variable name
        default: Default list if env var not set
        delimiter: Delimiter character (default: comma)
    
    Returns:
        List of stripped strings
    """
    value = os.getenv(env_var)
    if value:
        return [item.strip() for item in value.split(delimiter) if item.strip()]
    return default


def _parse_bool(env_var: str, default: bool = False) -> bool:
    """
    Parse boolean environment variable.
    
    Args:
        env_var: Environment variable name
        default: Default boolean value
    
    Returns:
        Boolean value
    """
    value = os.getenv(env_var, str(default)).lower()
    return value in ("1", "true", "yes", "on")


def _parse_json_list(env_var: str, default: List[str]) -> List[str]:
    """
    Parse JSON array from environment variable.
    
    Args:
        env_var: Environment variable name
        default: Default list
    
    Returns:
        Parsed list or default
    """
    value = os.getenv(env_var)
    if value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return default


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

class LLMConfig:
    """
    Configuration for Large Language Model (LLM) interactions.
    
    Controls model selection, generation parameters, and behavior.
    """
    
    # Model identification
    MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "deepseek-ai/deepseek-llm-7b-chat")
    MODEL_TYPE: str = os.getenv("LLM_MODEL_TYPE", "deepseek")  # deepseek, claude, gpt
    
    # Generation parameters
    MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "128"))
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    TOP_P: float = float(os.getenv("LLM_TOP_P", "1.0"))
    DO_SAMPLE: bool = _parse_bool("LLM_DO_SAMPLE", False)
    
    # Stop sequences
    STOP_SEQUENCES: List[str] = _parse_json_list("LLM_STOP_SEQUENCES", [])
    
    # Prompt format
    PROMPT_TEMPLATE: str = os.getenv("LLM_PROMPT_TEMPLATE", "<|user|>\n{prompt}\n<|assistant|>\n")
    RESPONSE_DELIMITER: str = os.getenv("LLM_RESPONSE_DELIMITER", "<|assistant|>")
    
    # Model storage (for local models)
    OFFLOAD_PATH: str = os.getenv("LLM_OFFLOAD_PATH", "./deepseek_offload")
    DEVICE_MAP: str = os.getenv("LLM_DEVICE_MAP", "auto")
    TORCH_DTYPE: str = os.getenv("LLM_TORCH_DTYPE", "float32")
    
    @classmethod
    def validate(cls) -> None:
        """Validate LLM configuration."""
        if cls.MAX_TOKENS < 1 or cls.MAX_TOKENS > 8192:
            raise ValueError(f"LLM_MAX_TOKENS must be between 1 and 8192, got {cls.MAX_TOKENS}")
        
        if not (0.0 <= cls.TEMPERATURE <= 2.0):
            raise ValueError(f"LLM_TEMPERATURE must be between 0.0 and 2.0, got {cls.TEMPERATURE}")
        
        if not (0.0 <= cls.TOP_P <= 1.0):
            raise ValueError(f"LLM_TOP_P must be between 0.0 and 1.0, got {cls.TOP_P}")


# ============================================================================
# SCORING CONFIGURATION
# ============================================================================

class ScoringConfig:
    """
    Configuration for resume matching scoring algorithm.
    
    Controls how candidate-job matches are calculated and weighted.
    """
    
    # Primary weights (must sum to 100)
    MUST_HAVE_WEIGHT: float = float(os.getenv("MUST_HAVE_WEIGHT", "75.0"))
    NICE_TO_HAVE_WEIGHT: float = float(os.getenv("NICE_TO_HAVE_WEIGHT", "25.0"))
    
    # Scoring floors
    MUST_PASS_FLOOR: float = float(os.getenv("MUST_PASS_FLOOR", "60.0"))
    LOW_COVERAGE_FLOOR: float = float(os.getenv("LOW_COVERAGE_FLOOR", "10.0"))
    
    # Score calculation parameters
    SCORING_EXPONENT: float = float(os.getenv("SCORING_EXPONENT", "1.65"))
    MAX_MUST_HAVE_SCORE: float = float(os.getenv("MAX_MUST_HAVE_SCORE", "90.0"))
    MAX_NICE_TO_HAVE_SCORE: float = float(os.getenv("MAX_NICE_TO_HAVE_SCORE", "10.0"))
    MAX_TOTAL_SCORE: float = float(os.getenv("MAX_TOTAL_SCORE", "100.0"))
    
    # Evidence strength multipliers
    NO_EVIDENCE_MULTIPLIER: float = float(os.getenv("NO_EVIDENCE_MULTIPLIER", "0.0"))
    WEAK_EVIDENCE_MULTIPLIER: float = float(os.getenv("WEAK_EVIDENCE_MULTIPLIER", "0.5"))
    STRONG_EVIDENCE_MULTIPLIER: float = float(os.getenv("STRONG_EVIDENCE_MULTIPLIER", "1.0"))
    
    # Semantic matching thresholds
    SEMANTIC_MATCH_THRESHOLD: float = float(os.getenv("SEMANTIC_MATCH_THRESHOLD", "0.6"))
    EVIDENCE_EXTRACTION_THRESHOLD: float = float(os.getenv("EVIDENCE_EXTRACTION_THRESHOLD", "0.5"))
    SEMANTIC_MATCHER_THRESHOLD: float = float(os.getenv("SEMANTIC_MATCHER_THRESHOLD", "0.65"))
    
    # Soft requirements handling
    SOFT_MUSTS_NONBLOCKING: bool = _parse_bool("SOFT_MUSTS_NONBLOCKING", True)
    
    # Weak evidence patterns (phrases indicating weak/limited experience)
    WEAK_EVIDENCE_PATTERNS: List[str] = _parse_list(
        "WEAK_EVIDENCE_PATTERNS",
        default=[
            "data science", "basic scripting", "occasional", "cleanup",
            "not direct", "data cleanup", "bioinformatics",
            "limited", "minimal", "partial", "assisting", "exposure to"
        ]
    )
    
    # Known tools for dynamic matching
    KNOWN_TOOLS: List[str] = _parse_list(
        "KNOWN_TOOLS",
        default=[
            "Schrodinger", "GROMACS", "Ambertools",
            "RDKit", "pandas", "seaborn", "SQL"
        ]
    )
    
    @classmethod
    def validate(cls) -> None:
        """Validate scoring configuration."""
        # Check weights sum approximately to 100
        total_weight = cls.MUST_HAVE_WEIGHT + cls.NICE_TO_HAVE_WEIGHT
        if not (99.0 <= total_weight <= 101.0):
            raise ValueError(
                f"MUST_HAVE_WEIGHT + NICE_TO_HAVE_WEIGHT should sum to ~100, "
                f"got {total_weight}"
            )
        
        # Check thresholds are in valid range
        if not (0.0 <= cls.SEMANTIC_MATCH_THRESHOLD <= 1.0):
            raise ValueError(
                f"SEMANTIC_MATCH_THRESHOLD must be between 0.0 and 1.0, "
                f"got {cls.SEMANTIC_MATCH_THRESHOLD}"
            )
        
        if not (0.0 <= cls.EVIDENCE_EXTRACTION_THRESHOLD <= 1.0):
            raise ValueError(
                f"EVIDENCE_EXTRACTION_THRESHOLD must be between 0.0 and 1.0, "
                f"got {cls.EVIDENCE_EXTRACTION_THRESHOLD}"
            )
        
        # Check exponent is positive
        if cls.SCORING_EXPONENT <= 0:
            raise ValueError(f"SCORING_EXPONENT must be positive, got {cls.SCORING_EXPONENT}")


# ============================================================================
# TEXT PROCESSING CONFIGURATION
# ============================================================================

class TextProcessingConfig:
    """
    Configuration for text processing and pattern matching.
    
    Controls window sizes, distances, and text manipulation parameters.
    """
    
    # Text window parameters
    TEXT_WINDOW_PADDING: int = int(os.getenv("TEXT_WINDOW_PADDING", "80"))
    NEGATION_DETECTION_GAP: int = int(os.getenv("NEGATION_DETECTION_GAP", "50"))
    MAX_KEYWORDS_PER_REQUIREMENT: int = int(os.getenv("MAX_KEYWORDS_PER_REQUIREMENT", "8"))
    
    # Regex patterns for negation detection
    NEGATION_TOKENS: str = os.getenv(
        "NEGATION_TOKENS",
        r"(?:^|\b)(?:no|not|without|lack(?:ing)?|lack of|intentionally no)\b"
    )
    
    # Job description section headers
    QUALIFICATIONS_HEADERS: List[str] = _parse_list(
        "QUALIFICATIONS_HEADERS",
        default=["Qualifications", "Required"]
    )
    
    PREFERRED_HEADERS: List[str] = _parse_list(
        "PREFERRED_HEADERS",
        default=["Preferred", "Nice to have", "Nice-to-haves", "Nice-to-have"]
    )
    
    # Bullet point characters
    BULLET_CHARACTERS: List[str] = _parse_list(
        "BULLET_CHARACTERS",
        default=["-", "*", "•"]
    )
    
    # Header detection pattern
    HEADER_PATTERN: str = os.getenv("HEADER_PATTERN", r"^[A-Za-z][^:]{0,60}:\s*$")
    
    @classmethod
    def validate(cls) -> None:
        """Validate text processing configuration."""
        if cls.TEXT_WINDOW_PADDING < 0:
            raise ValueError(f"TEXT_WINDOW_PADDING must be non-negative, got {cls.TEXT_WINDOW_PADDING}")
        
        if cls.NEGATION_DETECTION_GAP < 0:
            raise ValueError(f"NEGATION_DETECTION_GAP must be non-negative, got {cls.NEGATION_DETECTION_GAP}")
        
        if cls.MAX_KEYWORDS_PER_REQUIREMENT < 1:
            raise ValueError(
                f"MAX_KEYWORDS_PER_REQUIREMENT must be at least 1, "
                f"got {cls.MAX_KEYWORDS_PER_REQUIREMENT}"
            )


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

class ModelConfig:
    """
    Configuration for ML models (semantic matching, NLP, etc).
    
    Controls which pre-trained models are loaded for various tasks.
    """
    
    # Sentence transformer model for semantic matching
    SEMANTIC_TRANSFORMER_MODEL: str = os.getenv(
        "SEMANTIC_TRANSFORMER_MODEL",
        "all-MiniLM-L6-v2"
    )
    
    # spaCy model for NLP tasks
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    
    # Model caching
    CACHE_MODELS: bool = _parse_bool("CACHE_MODELS", True)
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./.model_cache")
    
    @classmethod
    def validate(cls) -> None:
        """Validate model configuration."""
        # Basic validation - models will fail at load time if invalid
        if not cls.SEMANTIC_TRANSFORMER_MODEL:
            raise ValueError("SEMANTIC_TRANSFORMER_MODEL cannot be empty")
        
        if not cls.SPACY_MODEL:
            raise ValueError("SPACY_MODEL cannot be empty")


# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

class ServerConfig:
    """
    Configuration for FastAPI server.
    
    Controls host, port, and server behavior.
    """
    
    HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    
    # CORS settings
    CORS_ENABLED: bool = _parse_bool("CORS_ENABLED", False)
    CORS_ORIGINS: List[str] = _parse_json_list("CORS_ORIGINS", ["*"])
    
    # API settings
    API_VERSION: str = os.getenv("API_VERSION", "v1")
    API_PREFIX: str = os.getenv("API_PREFIX", "")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> None:
        """Validate server configuration."""
        if not (1 <= cls.PORT <= 65535):
            raise ValueError(f"SERVER_PORT must be between 1 and 65535, got {cls.PORT}")
        
        if cls.LOG_LEVEL not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"LOG_LEVEL must be valid level, got {cls.LOG_LEVEL}")


# ============================================================================
# AI SCORING SYSTEM PROMPT
# ============================================================================

AI_SCORING_SYSTEM_PROMPT = """You are an expert AI recruiter tasked with scoring contractor resumes against job descriptions with EXTREME precision.

**CRITICAL SCORING RULES (MUST FOLLOW EXACTLY):**

1. **LOCATION CAP (ABSOLUTE PRIORITY):**
   - If the job specifies a location AND the candidate is in a DIFFERENT location → **MAXIMUM score is 5%**, regardless of skills.
   - If no location is specified in the job, OR the candidate's location matches → proceed with normal scoring.

2. **MUST-HAVES (ZERO TOLERANCE):**
   - Count ALL technical must-haves from the job description (ignore soft skills like "communication", "curiosity", "team player").
   - If the candidate has **ZERO must-haves** → **score is 0%**.
   - If the candidate has **ALL must-haves** → base score is **90%** (can reach 100% with nice-to-haves).
   - If the candidate has **some but not all must-haves** → score between 10% and 89% (proportional, using a curved formula).

3. **SYNONYM INTELLIGENCE:**
   - Recognize technical synonyms: "SQL" = "PostgreSQL" OR "MySQL" OR "SQL Server"
   - "JavaScript" = "JS" = "Node.js" (for backend roles)
   - "AWS" = "Amazon Web Services" = specific AWS services (e.g., "EC2", "S3")
   - "Python" matches if ANY Python framework is present (Django, Flask, Pandas, etc.)

4. **SOFT SKILLS EXCLUSION:**
   - Do NOT count these as must-haves for scoring: "communication", "leadership", "time management", "team player", "self-starter", "curiosity", "passion", "motivated".
   - These are always considered "met" but do NOT affect the numeric score.

5. **DATE CALCULATION:**
   - If a job requires "X+ years of experience", calculate the candidate's years by:
     - Parsing all date ranges in the resume (e.g., "Jan 2018 - Present", "2015-2017")
     - Summing the total years (use current year for "Present")
     - If total years ≥ X, the experience requirement is met.

6. **NICE-TO-HAVES BONUS:**
   - Each nice-to-have met adds bonus points (up to 10% total).
   - If all must-haves are met (90%) + all nice-to-haves are met (10%) → 100% score.

7. **EDUCATION:**
   - Treat education requirements (e.g., "Bachelor's degree", "PhD") as must-haves.
   - "Bachelor's" matches "B.Sc.", "B.A.", "B.Eng."
   - "Master's" matches "M.Sc.", "M.A.", "M.Eng."
   - "PhD" matches "Ph.D.", "Doctorate"

**OUTPUT FORMAT (STRICT JSON):**
You MUST return a single JSON object with this exact structure:

{
  "match_percentage": <integer from 0 to 100>,
  "location_match": <true/false>,
  "must_haves_total": <integer: count of technical must-haves>,
  "must_haves_met": <integer: how many the candidate has>,
  "nice_to_haves_total": <integer>,
  "nice_to_haves_met": <integer>,
  "work_experience_years": <integer or null>,
  "summary": "<3-sentence explanation of the score>"
}

**EXAMPLE:**

Job: "Must-Haves: Python, AWS, 5+ years experience. Location: New York, NY. Nice-to-Haves: Docker."
Resume: "Python developer with 7 years. Expert in AWS. Located in London, UK. Knows Docker."

Output:
{
  "match_percentage": 5,
  "location_match": false,
  "must_haves_total": 3,
  "must_haves_met": 3,
  "nice_to_haves_total": 1,
  "nice_to_haves_met": 1,
  "work_experience_years": 7,
  "summary": "Candidate has all required skills (Python, AWS, 7 years experience) and the nice-to-have (Docker). However, they are located in London, UK while the job is in New York, NY, resulting in a maximum score of 5% due to location mismatch."
}
"""


# ============================================================================
# SOFT SKILLS & PATTERNS
# ============================================================================

class PatternConfig:
    """
    Configuration for pattern matching (soft skills, tech cues, etc).
    
    These are kept in code for now but can be externalized to JSON files.
    """
    
    # Non-blocking must-have phrases (soft skills that shouldn't zero the score)
    NON_BLOCKING_MUST_PHRASES: List[str] = [
        "intellectual curiosity",
        "drive to excel",
        "curiosity",
        "strong communication",
        "communication skills",
        "time-management",
        "time management",
        "team player",
        "ability to work independently",
        "self-starter",
        "passion",
        "motivated",
        "enthusiasm",
    ]
    
    # Soft skill keywords
    SOFT_KEYWORDS: set = {
        "communication", "time management", "time-management", "leadership",
        "team player", "teamwork", "collaboration", "self-starter",
        "motivated", "passion", "curiosity", "drive to excel",
        "ownership", "problem solving", "work independently",
        "attention to detail"
    }
    
    # Technical cues pattern
    TECH_CUES_PATTERN: str = (
        r"(years?|experience|b\.?sc|m\.?sc|ph\.?d|certificat|aws|gcp|azure|sql|"
        r"mysql|node\.?js|react|docker|kubernetes|schrodinger|gromacs|amber|"
        r"ambertools|rdkit|pandas|seaborn|tensorflow|pytorch)"
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_all_configs() -> None:
    """
    Validate all configuration classes.
    
    Raises:
        ValueError: If any configuration is invalid
    """
    LLMConfig.validate()
    ScoringConfig.validate()
    TextProcessingConfig.validate()
    ModelConfig.validate()
    ServerConfig.validate()


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config_summary() -> None:
    """Print configuration summary for debugging."""
    print("=" * 60)
    print("APPLICATION CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print("\n[LLM Configuration]")
    print(f"  Model: {LLMConfig.MODEL_NAME}")
    print(f"  Max Tokens: {LLMConfig.MAX_TOKENS}")
    print(f"  Temperature: {LLMConfig.TEMPERATURE}")
    
    print("\n[Scoring Configuration]")
    print(f"  Must-Have Weight: {ScoringConfig.MUST_HAVE_WEIGHT}%")
    print(f"  Nice-to-Have Weight: {ScoringConfig.NICE_TO_HAVE_WEIGHT}%")
    print(f"  Scoring Exponent: {ScoringConfig.SCORING_EXPONENT}")
    print(f"  Semantic Threshold: {ScoringConfig.SEMANTIC_MATCH_THRESHOLD}")
    
    print("\n[Model Configuration]")
    print(f"  Semantic Model: {ModelConfig.SEMANTIC_TRANSFORMER_MODEL}")
    print(f"  spaCy Model: {ModelConfig.SPACY_MODEL}")
    
    print("\n[Server Configuration]")
    print(f"  Host: {ServerConfig.HOST}")
    print(f"  Port: {ServerConfig.PORT}")
    print(f"  Log Level: {ServerConfig.LOG_LEVEL}")
    
    print("=" * 60)


# Auto-validate on import (can be disabled with env var)
if not _parse_bool("SKIP_CONFIG_VALIDATION", False):
    validate_all_configs()

