"""
Model Factory Module

Provides factory functions for ML model instantiation with proper singleton pattern.
Handles sentence transformers and other ML models with lazy loading.
"""

from typing import Optional
from sentence_transformers import SentenceTransformer
from app.config.app_config import ModelConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global singleton instance for semantic model
_semantic_model_instance: Optional[SentenceTransformer] = None


def get_semantic_model() -> SentenceTransformer:
    """
    Get or create sentence transformer model singleton.
    
    Lazy loads the model on first call and reuses it for subsequent calls.
    This is important because model loading is expensive (~100-200ms).
    
    Returns:
        Initialized SentenceTransformer model
    
    Usage:
        from app.processors.model_factory import get_semantic_model
        
        model = get_semantic_model()
        embeddings = model.encode(["text1", "text2"])
    """
    global _semantic_model_instance
    
    if _semantic_model_instance is None:
        logger.info(f"Loading semantic model: {ModelConfig.SEMANTIC_TRANSFORMER_MODEL}")
        _semantic_model_instance = SentenceTransformer(ModelConfig.SEMANTIC_TRANSFORMER_MODEL)
        logger.info("Semantic model ready")
    
    return _semantic_model_instance


def reset_semantic_model() -> None:
    """
    Reset the semantic model singleton.
    
    Useful for testing or when you need to reinitialize with a different model.
    
    Warning:
        This will cause the next call to get_semantic_model() to reload the model.
    """
    global _semantic_model_instance
    _semantic_model_instance = None
    logger.info("Semantic model reset")


def is_semantic_model_initialized() -> bool:
    """
    Check if semantic model has been initialized.
    
    Returns:
        True if model instance exists, False otherwise
    """
    return _semantic_model_instance is not None


def preload_models() -> None:
    """
    Preload all models at application startup.
    
    Call this during application initialization to avoid
    lazy loading delays on first request.
    
    Usage:
        from app.processors.model_factory import preload_models
        
        # In main.py or startup event
        preload_models()
    """
    from app.processors.llm_factory import get_llm_client
    
    logger.info("Preloading models...")
    logger.info("  - Loading semantic transformer model...")
    get_semantic_model()
    
    logger.info("  - Loading LLM client...")
    get_llm_client()
    
    logger.info("All models loaded and ready!")

