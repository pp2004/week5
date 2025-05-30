"""
Admin page for Azure GPT-4o Chat Application
Analytics dashboard with password protection
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from config import load_config
from db import get_chat_analytics, get_recent_messages, get_user_activity, clear_chat_history
from utils import format_analytics_data, format_timestamp

# Page configuration
st.set_page_config(
    page_title="Admin - Azure GPT-4o",
    page_icon="👑",
    layout="wide"
)

def check_admin_password() -> bool:
    """
    Check admin password
    
    Returns:
        Boolean indicating if password is correct
    """
    
    config = load_config()
    admin_password = config.get('admin_password', 'admin123')
    
    # Check if already authenticated
    if st.session_state.get('admin_authenticated'):
        return True
    
    # Show password form
    st.title("🔒 Admin Access")
    st.write("Please enter the admin password to access the analytics dashboard.")
    
    with st.form("admin_login"):
        password = st.text_input("Admin Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if password == admin_password:
                st.session_state.admin_authenticated = True
                st.success("✅ Access granted!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid password")
    
    return False

def show_analytics_overview():
    """Show analytics overview section"""
    
    st.header("📊 Analytics Overview")
    
    # Get analytics data
    analytics = get_chat_analytics()
    formatted_analytics = format_analytics_data(analytics)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Messages",
            formatted_analytics.get('total_messages_formatted', '0'),
            help="Total number of messages in the system"
        )
    
    with col2:
        st.metric(
            "Total Tokens",
            formatted_analytics.get('total_tokens_formatted', '0'),
            help="Total tokens consumed across all conversations"
        )
    
    with col3:
        st.metric(
            "PDF Messages",
            analytics.get('pdf_messages', 0),
            help="Messages that included PDF context"
        )
    
    with col4:
        st.metric(
            "Recent Activity",
            analytics.get('recent_messages', 0),
            help="Messages in the last 24 hours"
        )
    
    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "User Messages",
            analytics.get('user_messages', 0),
            help="Total messages from users"
        )
    
    with col6:
        st.metric(
            "Assistant Messages",
            analytics.get('assistant_messages', 0),
            help="Total messages from AI assistant"
        )
    
    with col7:
        # Calculate average tokens per message
        total_messages = analytics.get('total_messages', 0)
        total_tokens = analytics.get('total_tokens', 0)
        avg_tokens = int(total_tokens / total_messages) if total_messages > 0 else 0
        
        st.metric(
            "Avg Tokens/Message",
            avg_tokens,
            help="Average tokens per message"
        )
    
    with col8:
        # Calculate PDF usage percentage
        pdf_messages = analytics.get('pdf_messages', 0)
        total_messages = analytics.get('total_messages', 0)
        pdf_percentage = int((pdf_messages / total_messages) * 100) if total_messages > 0 else 0
        
        st.metric(
            "PDF Usage",
            f"{pdf_percentage}%",
            help="Percentage of messages using PDF context"
        )

def show_recent_messages():
    """Show recent messages section"""
    
    st.header("💬 Recent Messages")
    
    # Controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        message_limit = st.selectbox(
            "Messages to show",
            options=[25, 50, 100, 200],
            index=1,
            help="Number of recent messages to display"
        )
    
    with col2:
        if st.button("🔄 Refresh", type="secondary"):
            st.experimental_rerun()
    
    # Get recent messages
    messages = get_recent_messages(limit=message_limit)
    
    if not messages:
        st.info("No messages found in the database.")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(messages)
    
    # Reorder columns for better display
    column_order = ['timestamp', 'user_id', 'message_type', 'content', 'tokens_used', 'has_pdf_context', 'model_name']
    df = df[column_order]
    
    # Rename columns for display
    df.columns = ['Timestamp', 'User ID', 'Type', 'Content', 'Tokens', 'PDF Context', 'Model']
    
    # Style the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        height=400,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                format="MM/DD/YY HH:mm:ss"
            ),
            "Type": st.column_config.TextColumn(
                "Type",
                width="small"
            ),
            "Content": st.column_config.TextColumn(
                "Content",
                width="large"
            ),
            "Tokens": st.column_config.NumberColumn(
                "Tokens",
                width="small"
            ),
            "PDF Context": st.column_config.CheckboxColumn(
                "PDF Context",
                width="small"
            )
        }
    )
    
    # Export option
    if st.button("📥 Export Messages CSV"):
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"chat_messages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_user_activity():
    """Show user activity section"""
    
    st.header("👥 User Activity")
    
    # Get user activity data
    user_activity = get_user_activity()
    
    if not user_activity:
        st.info("No user activity data available.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(user_activity)
    
    # Display summary
    total_users = len(df)
    total_user_messages = df['user_messages'].sum()
    total_assistant_messages = df['assistant_messages'].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Users", total_users)
    
    with col2:
        st.metric("User Messages", total_user_messages)
    
    with col3:
        st.metric("Assistant Messages", total_assistant_messages)
    
    # Display user activity table
    st.subheader("User Breakdown")
    
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "user_id": "User ID",
            "user_messages": st.column_config.NumberColumn(
                "User Messages",
                help="Messages sent by the user"
            ),
            "assistant_messages": st.column_config.NumberColumn(
                "Assistant Messages", 
                help="Messages from AI assistant"
            ),
            "total_messages": st.column_config.NumberColumn(
                "Total Messages",
                help="Total messages in conversation"
            )
        }
    )

def show_system_management():
    """Show system management section"""
    
    st.header("⚙️ System Management")
    
    # Database management
    st.subheader("🗄️ Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Clear Chat History**")
        st.write("This will permanently delete all chat messages from the database.")
        
        if st.button("🗑️ Clear All Messages", type="secondary"):
            st.warning("Are you sure you want to clear all messages? This action cannot be undone.")
            
            col_confirm, col_cancel = st.columns(2)
            
            with col_confirm:
                if st.button("✅ Confirm Clear", type="primary"):
                    success = clear_chat_history()
                    if success:
                        st.success("✅ All messages cleared successfully!")
                    else:
                        st.error("❌ Failed to clear messages")
            
            with col_cancel:
                if st.button("❌ Cancel"):
                    st.experimental_rerun()
    
    with col2:
        st.write("**Database Info**")
        import os
        
        if os.path.exists("chat_logs.db"):
            file_size = os.path.getsize("chat_logs.db")
            st.info(f"Database size: {file_size / 1024:.2f} KB")
        else:
            st.warning("Database file not found")
    
    # Session management
    st.subheader("🔐 Session Management")
    
    if st.button("🚪 Logout", type="secondary"):
        st.session_state.admin_authenticated = False
        st.experimental_rerun()

def main():
    """Main admin page function"""
    
    # Check admin password
    if not check_admin_password():
        return
    
    # Show admin dashboard
    st.title("👑 Admin Dashboard")
    st.write("Welcome to the Azure GPT-4o Chat Application admin panel.")
    
    # Sidebar navigation
    st.sidebar.title("📋 Admin Navigation")
    
    section = st.sidebar.selectbox(
        "Select Section",
        options=[
            "Analytics Overview",
            "Recent Messages", 
            "User Activity",
            "System Management"
        ],
        index=0
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)")
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.experimental_rerun()
    
    # Display selected section
    if section == "Analytics Overview":
        show_analytics_overview()
    elif section == "Recent Messages":
        show_recent_messages()
    elif section == "User Activity":
        show_user_activity()
    elif section == "System Management":
        show_system_management()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

