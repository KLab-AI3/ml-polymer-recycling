"""Centralized error handling utility for the backend API.
Provides consistent error logging without UI dependencies"""

import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handler for backend operations"""

    @staticmethod
    def log_error(error: Exception, context: str = "", include_traceback: bool = False) -> None:
        """Log error for backend operations"""
        error_msg = f"ERROR {context}: {str(error)}" if context else f"ERROR {str(error)}"
        
        if include_traceback:
            error_msg += f"\nTraceback: {traceback.format_exc()}"
        
        logger.error(error_msg)

    @staticmethod
    def log_warning(message: str, context: str = "") -> None:
        """Log warning for backend operations"""
        warning_msg = f"WARNING {context}: {message}" if context else f"WARNING {message}"
        logger.warning(warning_msg)

def safe_execute(func, *args, default_return=None, error_context="", **kwargs):
    """Safely execute a function and handle errors"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        ErrorHandler.log_error(e, error_context)
        return default_return
