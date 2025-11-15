"""
Logging Utility Module

Provides centralized logging configuration for the application.
Uses Python's built-in logging with structured output.
"""

import logging
import sys
from typing import Optional
from app.config.app_config import ServerConfig

# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logging() -> logging.Logger:
    """
    Setup and configure application logging.
    
    Configures log format, level, and handlers based on ServerConfig.
    
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # Create logger
    logger = logging.getLogger("tgs_ai")
    logger.setLevel(getattr(logging, ServerConfig.LOG_LEVEL.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, ServerConfig.LOG_LEVEL.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    _logger = logger
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get application logger.
    
    Args:
        name: Optional logger name (defaults to "tgs_ai")
    
    Returns:
        Logger instance
    
    Usage:
        from app.utils.logger import get_logger
        
        logger = get_logger(__name__)
        logger.info("Processing resume")
        logger.warning("Missing required field")
        logger.error("Failed to parse JSON", exc_info=True)
    """
    if _logger is None:
        setup_logging()
    
    if name and name != "tgs_ai":
        return logging.getLogger(f"tgs_ai.{name}")
    
    return _logger


# Convenience functions for common logging patterns

def log_info(message: str, **kwargs) -> None:
    """Log info message."""
    get_logger().info(message, **kwargs)


def log_warning(message: str, **kwargs) -> None:
    """Log warning message."""
    get_logger().warning(message, **kwargs)


def log_error(message: str, exc_info: bool = False, **kwargs) -> None:
    """
    Log error message.
    
    Args:
        message: Error message
        exc_info: Include exception traceback (default: False)
        **kwargs: Additional logging parameters
    """
    get_logger().error(message, exc_info=exc_info, **kwargs)


def log_debug(message: str, **kwargs) -> None:
    """Log debug message."""
    get_logger().debug(message, **kwargs)


def log_llm_call(prompt_length: int, response_length: int, duration_ms: float) -> None:
    """
    Log LLM API call with metrics.
    
    Args:
        prompt_length: Length of prompt in characters
        response_length: Length of response in characters
        duration_ms: Call duration in milliseconds
    """
    get_logger().info(
        f"LLM call completed | prompt: {prompt_length} chars | "
        f"response: {response_length} chars | duration: {duration_ms:.2f}ms"
    )


def log_analysis_start(resume_length: int, jd_length: int) -> None:
    """
    Log start of resume analysis.
    
    Args:
        resume_length: Resume text length
        jd_length: Job description length
    """
    get_logger().info(
        f"Starting resume analysis | resume: {resume_length} chars | "
        f"jd: {jd_length} chars"
    )


def log_analysis_complete(match_percentage: float, duration_ms: float) -> None:
    """
    Log completion of resume analysis.
    
    Args:
        match_percentage: Final match score
        duration_ms: Analysis duration in milliseconds
    """
    get_logger().info(
        f"Analysis complete | match: {match_percentage}% | "
        f"duration: {duration_ms:.2f}ms"
    )


# Initialize logging on module import
setup_logging()

