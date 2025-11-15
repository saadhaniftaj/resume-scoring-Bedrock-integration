"""
Configuration Package

Provides centralized configuration management for the entire application.
All configuration is loaded from environment variables with sensible defaults.

Usage:
    from app.config import ScoringConfig, LLMConfig
    
    max_tokens = LLMConfig.MAX_TOKENS
    threshold = ScoringConfig.SEMANTIC_MATCH_THRESHOLD
"""

from .app_config import (
    LLMConfig,
    ScoringConfig,
    TextProcessingConfig,
    ModelConfig,
    ServerConfig,
    PatternConfig,
    validate_all_configs,
    print_config_summary,
    AI_SCORING_SYSTEM_PROMPT
)

# Validate configuration on package import
# This ensures any configuration errors are caught early
validate_all_configs()

__all__ = [
    "LLMConfig",
    "ScoringConfig",
    "TextProcessingConfig",
    "ModelConfig",
    "ServerConfig",
    "PatternConfig",
    "validate_all_configs",
    "print_config_summary",
    "AI_SCORING_SYSTEM_PROMPT"
]

