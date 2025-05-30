"""
Azure GPT-4o Chat Application with RAG
Production-style Streamlit web app for Azure OpenAI integration
"""

import streamlit as st
import os
from config import load_config
from db import init_db
from utils import setup_logging

# Page configuration
st.set_page_config(
    page_title="Azure GPT-4o Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = load_config()
    
    # Initialize database
    init_db()
    
    # Main page content
    st.title("🤖 Azure GPT-4o Chat Application")
    
    st.markdown("""
    ### Welcome to the Azure GPT-4o Chat Application
    
    This application provides:
    - **Secure Chat Interface**: Connect to Azure OpenAI GPT-4o
    - **PDF RAG Capabilities**: Upload and chat with your documents
    - **Analytics Dashboard**: Admin view for usage statistics
    
    **Getting Started:**
    1. Navigate to the **Chat** page to start chatting
    2. Configure your Azure OpenAI credentials in the sidebar
    3. Upload PDFs for document-based conversations
    4. Use the **Admin** page to view analytics (password required)
    """)
    
    # Quick setup guide
    with st.expander("🔧 Setup Instructions"):
        st.markdown("""
        ### VS Code Setup (Windows)
        
        1. **Install Python 3.8+**
           ```bash
           # Download from python.org or use Windows Store
           python --version
           ```
        
        2. **Create Virtual Environment**
           ```bash
           python -m venv venv
           venv\\Scripts\\activate  # Windows
           ```
        
        3. **Install Dependencies**
           ```bash
           pip install streamlit python-dotenv sqlalchemy chromadb sentence-transformers PyPDF2 azure-ai-openai azure-identity
           ```
        
        4. **Configure Environment**
           - Copy `.env.example` to `.env`
           - Fill in your Azure OpenAI credentials
        
        5. **Run Application**
           ```bash
           streamlit run app.py --server.port 5000
           ```
        
        ### Azure OpenAI Setup
        
        1. **Create Azure OpenAI Resource**
           - Go to Azure Portal
           - Create new OpenAI resource
           - Note your endpoint URL
        
        2. **Deploy GPT-4o Model**
           - Go to Azure OpenAI Studio
           - Deploy gpt-4o model
           - Note your deployment name
        
        3. **Get API Key**
           - In Azure Portal, go to your OpenAI resource
           - Copy API key from Keys and Endpoint section
        
        4. **Configure Application**
           - Enter credentials in the Chat page sidebar
           - Or set them in `.env` file
        """)
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Database", "✅ Connected" if os.path.exists("chat_logs.db") else "❌ Not Found")
    
    with col2:
        azure_configured = (
            st.session_state.get("azure_endpoint") and 
            st.session_state.get("azure_api_key") and 
            st.session_state.get("deployment_name")
        )
        st.metric("Azure OpenAI", "✅ Configured" if azure_configured else "⚙️ Setup Required")
    
    with col3:
        st.metric("Pages", "2 Available")

if __name__ == "__main__":
    main()

