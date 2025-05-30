"""
Utility functions for Azure GPT-4o Chat Application
Provides logging, formatting, and helper functions
"""

import logging
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional
import os

def setup_logging():
    """Setup application logging"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create logger
    logger = logging.getLogger('azure_gpt_chat')
    return logger

def format_timestamp(timestamp: datetime) -> str:
    """
    Format timestamp for display
    
    Args:
        timestamp: Datetime object
    
    Returns:
        Formatted timestamp string
    """
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def format_message_for_display(message: Dict) -> str:
    """
    Format message for chat display
    
    Args:
        message: Message dictionary
    
    Returns:
        Formatted message string
    """
    
    content = message.get('content', '')
    role = message.get('role', 'unknown')
    
    # Truncate very long messages
    if len(content) > 2000:
        content = content[:2000] + "..."
    
    return content

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    
    # Rough estimation: 1 token ≈ 0.75 words
    word_count = len(text.split())
    return int(word_count * 1.3)

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Input text
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

def get_user_id() -> str:
    """
    Get user ID for session tracking
    
    Returns:
        User ID string
    """
    
    # For demo purposes, use session state or default
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "default_user"
    
    return st.session_state.user_id

def validate_pdf_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate uploaded PDF file
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file type
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Please upload a PDF file"
    
    # Check file size (limit to 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "File size must be less than 10MB"
    
    return True, ""

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: File size in bytes
    
    Returns:
        Formatted file size string
    """
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def clean_filename(filename: str) -> str:
    """
    Clean filename for safe storage
    
    Args:
        filename: Original filename
    
    Returns:
        Cleaned filename
    """
    
    # Remove or replace problematic characters
    import re
    
    # Remove path separators and other problematic characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(cleaned) > 100:
        name, ext = os.path.splitext(cleaned)
        cleaned = name[:95] + ext
    
    return cleaned

def format_analytics_data(analytics: Dict) -> Dict:
    """
    Format analytics data for display
    
    Args:
        analytics: Raw analytics dictionary
    
    Returns:
        Formatted analytics dictionary
    """
    
    formatted = analytics.copy()
    
    # Format large numbers
    for key in ['total_tokens', 'total_messages']:
        if key in formatted:
            value = formatted[key]
            if value >= 1000000:
                formatted[f"{key}_formatted"] = f"{value / 1000000:.1f}M"
            elif value >= 1000:
                formatted[f"{key}_formatted"] = f"{value / 1000:.1f}K"
            else:
                formatted[f"{key}_formatted"] = str(value)
    
    return formatted

def create_download_link(data: str, filename: str, mime_type: str = "text/plain") -> str:
    """
    Create download link for data
    
    Args:
        data: Data to download
        filename: Filename for download
        mime_type: MIME type
    
    Returns:
        HTML download link
    """
    
    import base64
    
    b64 = base64.b64encode(data.encode()).decode()
    
    return f'''
    <a href="data:{mime_type};base64,{b64}" download="{filename}">
        📥 Download {filename}
    </a>
    '''

def show_error_with_details(error_title: str, error_message: str, details: Optional[str] = None):
    """
    Show detailed error message
    
    Args:
        error_title: Error title
        error_message: Main error message
        details: Optional additional details
    """
    
    st.error(f"**{error_title}**")
    st.error(error_message)
    
    if details:
        with st.expander("🔍 Error Details"):
            st.code(details)

def show_success_message(title: str, message: str):
    """
    Show success message
    
    Args:
        title: Success title
        message: Success message
    """
    
    st.success(f"**{title}**")
    st.success(message)

def initialize_session_state():
    """Initialize session state variables"""
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # PDF processing state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    if 'pdf_filename' not in st.session_state:
        st.session_state.pdf_filename = ""
    
    # User settings
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 1000

def log_app_event(event_type: str, details: str):
    """
    Log application events
    
    Args:
        event_type: Type of event
        details: Event details
    """
    
    logger = logging.getLogger('azure_gpt_chat')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"[{timestamp}] {event_type}: {details}")

