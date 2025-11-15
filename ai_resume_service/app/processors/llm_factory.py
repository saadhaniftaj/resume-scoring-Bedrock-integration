"""
LLM Factory Module

Provides factory functions for LLM client instantiation with proper singleton pattern.
Enables easy swapping between different LLM implementations.

Environment-Aware LLM Selection:
- If AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set → Uses AWS Bedrock
- Otherwise → Falls back to local DeepSeek model
"""

import os
from typing import Optional, Union
from app.processors.deepseek_wrapper import DeepSeekLLM
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Global singleton instance (can be either DeepSeekLLM or BedrockDeepSeekWrapper)
_llm_instance: Optional[Union[DeepSeekLLM, 'BedrockDeepSeekWrapper']] = None


def _should_use_bedrock() -> bool:
    """
    Determine if AWS Bedrock should be used based on environment variables.
    
    Returns:
        True if AWS credentials are present, False otherwise
    """
    has_access_key = bool(os.getenv("AWS_ACCESS_KEY_ID"))
    has_secret_key = bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
    return has_access_key and has_secret_key


def get_llm_client() -> Union[DeepSeekLLM, 'BedrockDeepSeekWrapper']:
    """
    Get or create LLM client singleton (environment-aware).
    
    This factory function automatically selects between:
    - AWS Bedrock DeepSeek (if AWS credentials are present)
    - Local DeepSeek model (fallback if no AWS credentials)
    
    The choice is made only once (on first call) and cached as a singleton.
    
    Environment Variables (for Bedrock):
        AWS_ACCESS_KEY_ID: Required for Bedrock
        AWS_SECRET_ACCESS_KEY: Required for Bedrock
        AWS_REGION: Optional (default: us-east-1)
        BEDROCK_MODEL_ID: Optional (default: deepseek-r1-distill-llama-70b)
    
    Returns:
        Initialized LLM client instance (Bedrock or local)
    
    Usage:
        from app.processors.llm_factory import get_llm_client
        
        llm = get_llm_client()
        response = llm.ask("prompt here", max_tokens=512)
    """
    global _llm_instance
    
    if _llm_instance is None:
        logger.info("Initializing LLM client (first call)...")
        
        if _should_use_bedrock():
            # Use AWS Bedrock
            logger.info("✅ AWS credentials detected → Using AWS Bedrock DeepSeek")
            try:
                from app.processors.bedrock_wrapper import BedrockDeepSeekWrapper
                _llm_instance = BedrockDeepSeekWrapper()
                logger.info("✅ Bedrock LLM client ready")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Bedrock client: {str(e)}")
                logger.warning("⚠️  Falling back to local DeepSeek model...")
                _llm_instance = DeepSeekLLM()
                logger.info("✅ Local LLM client ready (fallback)")
        else:
            # Use local model
            logger.info("ℹ️  No AWS credentials detected → Using local DeepSeek model")
            _llm_instance = DeepSeekLLM()
            logger.info("✅ Local LLM client ready")
    
    return _llm_instance


def reset_llm_client() -> None:
    """
    Reset the LLM client singleton.
    
    Useful for testing or when you need to reinitialize with different settings.
    
    Warning:
        This will cause the next call to get_llm_client() to create a new instance.
    """
    global _llm_instance
    _llm_instance = None
    logger.info("LLM client reset")


def is_llm_initialized() -> bool:
    """
    Check if LLM client has been initialized.
    
    Returns:
        True if instance exists, False otherwise
    """
    return _llm_instance is not None

