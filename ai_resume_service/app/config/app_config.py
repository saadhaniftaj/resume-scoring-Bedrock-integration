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

**CRITICAL EXTRACTION RULES:**
- Extract the candidate's Name and Location ONLY from the top header of the resume. Do not guess. Leave blank if not found.
- For work_experience_years, analyze the entire work history and sum the durations of all relevant roles. Use the present year for 'Present' values.

**SUMMARY FORMAT (MANDATORY):**
- The summary must be a 2-3 sentence professional biography of the candidate, written as a standalone short-bio. It must NEVER reference the job, matching, score, or suitability. Example: 'Jane Doe is a Marketing Specialist with 8 years of experience in SEO and SEM, based in London, UK. She has managed global ad campaigns for major clients and holds advanced Google certifications.'

**SCORING RULES (as previously):**
1. LOCATION CAP (see above)
2. MUST-HAVES (zero tolerance)
3. Synonym Intelligence
4. Soft Skills Exclusion
5. Date Calculation
6. Nice-to-haves Bonus
7. Education (as previously)

**STRICT JSON OUTPUT:**
Return the following object ONLY:
{
  "match_percentage": <float>,
  "summary": "<string>",
  "candidate_info": {
    "name": "<string>",
    "location": "<string>",
    "work_experience_years": <int>
  },
  "analysis_breakdown": {
    "location_match": <bool>,
    "must_haves_total": <int>,
    "must_haves_met": <int>,
    "nice_to_haves_total": <int>,
    "nice_to_haves_met": <int>
  },
  "error": null
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

