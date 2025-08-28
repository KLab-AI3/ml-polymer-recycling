"""
Centralized error handling utility for the polymer classification app.
Provides consistent error logging and graceful UI error reporting.
"""

import streamlit as st
import traceback
from typing import Optional, Any
from pathlib import Path


class ErrorHandler:
    """Centralized error handler for the application"""
    
    @staticmethod
    def log_error(error: Exception, context: str = "", include_traceback: bool = False) -> None:
        """Log error to session state for display in UI"""
        if "log_messages" not in st.session_state:
            st.session_state["log_messages"] = []
        
        error_msg = f"[ERROR] {context}: {str(error)}" if context else f"[ERROR] {str(error)}"
        
        if include_traceback:
            error_msg += f"\nTraceback: {traceback.format_exc()}"
        
        st.session_state["log_messages"].append(error_msg)
    
    @staticmethod
    def log_warning(message: str, context: str = "") -> None:
        """Log warning to session state"""
        if "log_messages" not in st.session_state:
            st.session_state["log_messages"] = []
        
        warning_msg = f"[WARNING] {context}: {message}" if context else f"[WARNING] {message}"
        st.session_state["log_messages"].append(warning_msg)
    
    @staticmethod
    def log_info(message: str, context: str = "") -> None:
        """Log info message to session state"""
        if "log_messages" not in st.session_state:
            st.session_state["log_messages"] = []
        
        info_msg = f"[INFO] {context}: {message}" if context else f"[INFO] {message}"
        st.session_state["log_messages"].append(info_msg)
    
    @staticmethod
    def handle_file_error(filename: str, error: Exception) -> str:
        """Handle file processing errors and return user-friendly message"""
        ErrorHandler.log_error(error, f"File processing: {filename}")
        
        if isinstance(error, FileNotFoundError):
            return f"❌ File not found: {filename}"
        elif isinstance(error, PermissionError):
            return f"❌ Permission denied accessing: {filename}"
        elif isinstance(error, (ValueError, TypeError)):
            return f"❌ Invalid file format in: {filename}. Please ensure it contains wavenumber and intensity columns."
        else:
            return f"❌ Error processing file: {filename}. {str(error)}"
    
    @staticmethod
    def handle_inference_error(model_name: str, error: Exception) -> str:
        """Handle model inference errors and return user-friendly message"""
        ErrorHandler.log_error(error, f"Model inference: {model_name}")
        
        if "CUDA" in str(error) or "device" in str(error).lower():
            return f"❌ Device error with model {model_name}. Falling back to CPU."
        elif "shape" in str(error).lower() or "dimension" in str(error).lower():
            return f"❌ Input shape mismatch for model {model_name}. Please check spectrum data format."
        else:
            return f"❌ Inference failed for model {model_name}: {str(error)}"
    
    @staticmethod
    def handle_parsing_error(filename: str, error: Exception) -> str:
        """Handle spectrum parsing errors and return user-friendly message"""
        ErrorHandler.log_error(error, f"Spectrum parsing: {filename}")
        
        if "could not convert" in str(error).lower():
            return f"❌ Invalid data format in {filename}. Expected numeric wavenumber and intensity columns."
        elif "empty" in str(error).lower():
            return f"❌ File {filename} appears to be empty or contains no valid data."
        elif "columns" in str(error).lower():
            return f"❌ File {filename} must contain exactly 2 columns (wavenumber, intensity)."
        else:
            return f"❌ Failed to parse spectrum data from {filename}: {str(error)}"
    
    @staticmethod
    def clear_logs() -> None:
        """Clear all logged messages"""
        st.session_state["log_messages"] = []
    
    @staticmethod
    def get_logs() -> list[str]:
        """Get all logged messages"""
        return st.session_state.get("log_messages", [])
    
    @staticmethod
    def display_error_ui(error_message: str, show_details: bool = False) -> None:
        """Display error in Streamlit UI with optional details"""
        st.error(error_message)
        
        if show_details and st.session_state.get("log_messages"):
            with st.expander("Error Details", expanded=False):
                for msg in st.session_state["log_messages"][-5:]:  # Show last 5 log entries
                    st.text(msg)


def safe_execute(func, *args, error_context: str = "", show_error: bool = True, **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        error_context: Context description for error logging
        show_error: Whether to show error in UI
        **kwargs: Keyword arguments for the function
    
    Returns:
        Tuple of (result, success_flag)
    """
    try:
        result = func(*args, **kwargs)
        return result, True
    except Exception as e:
        ErrorHandler.log_error(e, error_context)
        if show_error:
            error_msg = f"Error in {error_context}: {str(e)}" if error_context else str(e)
            st.error(error_msg)
        return None, False