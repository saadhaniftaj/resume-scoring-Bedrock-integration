"""
Processors Package

Provides AI-powered resume analysis, matching, and semantic processing.

Key Components:
- llm_factory: Factory for LLM client instantiation
- model_factory: Factory for ML model instantiation
- ai_matcher: Core resume-to-job matching logic
- semantic_matcher: Semantic similarity matching
- local_parser: Basic resume parsing
- deepseek_wrapper: DeepSeek LLM interface

Usage:
    from app.processors import get_llm_client, get_semantic_model
    from app.processors.ai_matcher import analyze_resume_with_job
    
    llm = get_llm_client()
    result = analyze_resume_with_job(resume, job_desc)
"""

from .llm_factory import (
    get_llm_client,
    reset_llm_client,
    is_llm_initialized
)

from .model_factory import (
    get_semantic_model,
    reset_semantic_model,
    is_semantic_model_initialized,
    preload_models
)

__all__ = [
    # LLM factory
    "get_llm_client",
    "reset_llm_client",
    "is_llm_initialized",
    # Model factory
    "get_semantic_model",
    "reset_semantic_model",
    "is_semantic_model_initialized",
    "preload_models"
]

