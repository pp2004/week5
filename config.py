"""
Configuration management for Azure GPT-4o Chat Application
Handles environment variables and Azure OpenAI settings
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, Optional

def load_config() -> Dict[str, str]:
    """
    Load configuration from environment variables and .env file
    
    Returns:
        Dict containing configuration values
    """
    
    # Load .env file if it exists
    if os.path.exists('.env'):
        load_dotenv('.env')
    
    config = {
        'azure_endpoint': os.getenv('AZURE_OPENAI_ENDPOINT', ''),
        'azure_api_key': os.getenv('AZURE_OPENAI_API_KEY', ''),
        'deployment_name': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o-deployment'),
        'api_version': os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15'),
        'admin_password': os.getenv('ADMIN_PASSWORD', 'admin123'),
        'database_url': os.getenv('DATABASE_URL', 'sqlite:///./chat_logs.db')
    }
    
    return config

def validate_azure_config(endpoint: str, api_key: str, deployment_name: str, api_version: str) -> tuple[bool, str]:
    """
    Validate Azure OpenAI configuration
    
    Args:
        endpoint: Azure OpenAI endpoint URL
        api_key: API key
        deployment_name: Deployment name
        api_version: API version
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    if not endpoint:
        return False, "Azure OpenAI endpoint is required"
    
    if not endpoint.startswith('https://'):
        return False, "Endpoint must start with https://"
    
    if not endpoint.endswith('.openai.azure.com/') and not endpoint.endswith('.openai.azure.com'):
        return False, "Endpoint must be a valid Azure OpenAI endpoint"
    
    if not api_key:
        return False, "API key is required"
    
    if len(api_key) < 32:
        return False, "API key appears to be invalid (too short)"
    
    if not deployment_name:
        return False, "Deployment name is required"
    
    if not api_version:
        return False, "API version is required"
    
    return True, ""

def get_azure_config_from_session() -> Dict[str, str]:
    """
    Get Azure configuration from Streamlit session state
    
    Returns:
        Dict containing Azure configuration
    """
    
    return {
        'endpoint': st.session_state.get('azure_endpoint', ''),
        'api_key': st.session_state.get('azure_api_key', ''),
        'deployment_name': st.session_state.get('deployment_name', ''),
        'api_version': st.session_state.get('api_version', '2023-05-15')
    }

def save_azure_config_to_session(endpoint: str, api_key: str, deployment_name: str, api_version: str):
    """
    Save Azure configuration to Streamlit session state
    
    Args:
        endpoint: Azure OpenAI endpoint
        api_key: API key
        deployment_name: Deployment name
        api_version: API version
    """
    
    st.session_state.azure_endpoint = endpoint
    st.session_state.azure_api_key = api_key
    st.session_state.deployment_name = deployment_name
    st.session_state.api_version = api_version

def setup_sidebar_config():
    """
    Setup Azure OpenAI configuration in sidebar
    """
    
    config = load_config()
    
    st.sidebar.header("🔧 Azure OpenAI Configuration")
    
    # Initialize session state with config defaults if not set
    if 'azure_endpoint' not in st.session_state:
        st.session_state.azure_endpoint = config['azure_endpoint']
    if 'azure_api_key' not in st.session_state:
        st.session_state.azure_api_key = config['azure_api_key']
    if 'deployment_name' not in st.session_state:
        st.session_state.deployment_name = config['deployment_name']
    if 'api_version' not in st.session_state:
        st.session_state.api_version = config['api_version']
    
    # Configuration form
    with st.sidebar.form("azure_config"):
        endpoint = st.text_input(
            "Endpoint URL",
            value=st.session_state.azure_endpoint,
            placeholder="https://your-resource.openai.azure.com/",
            help="Your Azure OpenAI endpoint URL"
        )
        
        api_key = st.text_input(
            "API Key",
            value=st.session_state.azure_api_key,
            type="password",
            placeholder="Your Azure OpenAI API key",
            help="API key from Azure Portal"
        )
        
        deployment_name = st.text_input(
            "Deployment Name",
            value=st.session_state.deployment_name,
            placeholder="gpt-4o-deployment",
            help="Name of your GPT-4o deployment"
        )
        
        api_version = st.selectbox(
            "API Version",
            options=["2025-01-01-preview", "2023-05-15", "2023-06-01-preview", "2023-07-01-preview", "2023-12-01-preview"],
            index=0 if st.session_state.api_version == "2025-01-01-preview" else 0,
            help="Azure OpenAI API version"
        )
        
        submitted = st.form_submit_button("💾 Save Configuration")
        
        if submitted:
            is_valid, error_msg = validate_azure_config(endpoint, api_key, deployment_name, api_version)
            
            if is_valid:
                save_azure_config_to_session(endpoint, api_key, deployment_name, api_version)
                st.sidebar.success("✅ Configuration saved!")
                st.experimental_rerun()
            else:
                st.sidebar.error(f"❌ {error_msg}")
    
    # Show current status
    azure_config = get_azure_config_from_session()
    
    if all(azure_config.values()):
        st.sidebar.success("✅ Azure OpenAI Configured")
    else:
        st.sidebar.warning("⚠️ Please configure Azure OpenAI")
    
    return azure_config

