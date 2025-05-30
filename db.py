"""
Database models and operations for Azure GPT-4o Chat Application
Handles SQLite database operations for logging chat messages and analytics
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from config import load_config
import streamlit as st

# Database setup
Base = declarative_base()
config = load_config()

# Create engine
engine = create_engine(config['database_url'], echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ChatMessage(Base):
    """Model for storing chat messages"""
    
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, default="default_user")
    message_type = Column(String, index=True)  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    tokens_used = Column(Integer, default=0)
    model_name = Column(String, default="gpt-4o")
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=1000)
    has_pdf_context = Column(String, default="false")  # 'true' or 'false'

def init_db():
    """Initialize the database and create tables"""
    
    try:
        Base.metadata.create_all(bind=engine)
        return True
    except Exception as e:
        st.error(f"Failed to initialize database: {str(e)}")
        return False

def get_db() -> Session:
    """Get database session"""
    
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

def log_message(
    user_id: str,
    message_type: str,
    content: str,
    tokens_used: int = 0,
    model_name: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    has_pdf_context: bool = False
) -> bool:
    """
    Log a chat message to the database
    
    Args:
        user_id: User identifier
        message_type: 'user' or 'assistant'
        content: Message content
        tokens_used: Number of tokens used
        model_name: AI model name
        temperature: Temperature setting
        max_tokens: Max tokens setting
        has_pdf_context: Whether PDF context was used
    
    Returns:
        Boolean indicating success
    """
    
    try:
        db = get_db()
        
        message = ChatMessage(
            user_id=user_id,
            message_type=message_type,
            content=content,
            tokens_used=tokens_used,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            has_pdf_context="true" if has_pdf_context else "false"
        )
        
        db.add(message)
        db.commit()
        db.close()
        
        return True
        
    except Exception as e:
        st.error(f"Failed to log message: {str(e)}")
        return False

def get_chat_analytics() -> Dict:
    """
    Get chat analytics from the database
    
    Returns:
        Dictionary containing analytics data
    """
    
    try:
        db = get_db()
        
        # Total messages
        total_messages = db.query(ChatMessage).count()
        
        # Total tokens
        total_tokens = db.query(ChatMessage).with_entities(
            ChatMessage.tokens_used
        ).all()
        total_tokens = sum([t[0] for t in total_tokens if t[0]])
        
        # Messages by type
        user_messages = db.query(ChatMessage).filter(
            ChatMessage.message_type == "user"
        ).count()
        
        assistant_messages = db.query(ChatMessage).filter(
            ChatMessage.message_type == "assistant"
        ).count()
        
        # Messages with PDF context
        pdf_messages = db.query(ChatMessage).filter(
            ChatMessage.has_pdf_context == "true"
        ).count()
        
        # Recent activity (last 24 hours)
        from datetime import timedelta
        recent_cutoff = datetime.utcnow() - timedelta(days=1)
        recent_messages = db.query(ChatMessage).filter(
            ChatMessage.timestamp >= recent_cutoff
        ).count()
        
        db.close()
        
        return {
            'total_messages': total_messages,
            'total_tokens': total_tokens,
            'user_messages': user_messages,
            'assistant_messages': assistant_messages,
            'pdf_messages': pdf_messages,
            'recent_messages': recent_messages
        }
        
    except Exception as e:
        st.error(f"Failed to get analytics: {str(e)}")
        return {
            'total_messages': 0,
            'total_tokens': 0,
            'user_messages': 0,
            'assistant_messages': 0,
            'pdf_messages': 0,
            'recent_messages': 0
        }

def get_recent_messages(limit: int = 50) -> List[Dict]:
    """
    Get recent chat messages
    
    Args:
        limit: Maximum number of messages to retrieve
    
    Returns:
        List of message dictionaries
    """
    
    try:
        db = get_db()
        
        messages = db.query(ChatMessage).order_by(
            ChatMessage.timestamp.desc()
        ).limit(limit).all()
        
        result = []
        for msg in messages:
            result.append({
                'id': msg.id,
                'user_id': msg.user_id,
                'message_type': msg.message_type,
                'content': msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                'timestamp': msg.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'tokens_used': msg.tokens_used,
                'model_name': msg.model_name,
                'temperature': msg.temperature,
                'max_tokens': msg.max_tokens,
                'has_pdf_context': msg.has_pdf_context == "true"
            })
        
        db.close()
        return result
        
    except Exception as e:
        st.error(f"Failed to get recent messages: {str(e)}")
        return []

def get_user_activity() -> List[Dict]:
    """
    Get user activity statistics
    
    Returns:
        List of user activity dictionaries
    """
    
    try:
        db = get_db()
        
        # Query for user activity
        user_stats = db.query(
            ChatMessage.user_id,
            ChatMessage.message_type
        ).all()
        
        # Process the results
        user_activity = {}
        for user_id, message_type in user_stats:
            if user_id not in user_activity:
                user_activity[user_id] = {'user': 0, 'assistant': 0}
            user_activity[user_id][message_type] += 1
        
        # Convert to list format
        result = []
        for user_id, stats in user_activity.items():
            result.append({
                'user_id': user_id,
                'user_messages': stats['user'],
                'assistant_messages': stats['assistant'],
                'total_messages': stats['user'] + stats['assistant']
            })
        
        # Sort by total messages
        result.sort(key=lambda x: x['total_messages'], reverse=True)
        
        db.close()
        return result
        
    except Exception as e:
        st.error(f"Failed to get user activity: {str(e)}")
        return []

def clear_chat_history() -> bool:
    """
    Clear all chat history from the database
    
    Returns:
        Boolean indicating success
    """
    
    try:
        db = get_db()
        db.query(ChatMessage).delete()
        db.commit()
        db.close()
        return True
        
    except Exception as e:
        st.error(f"Failed to clear chat history: {str(e)}")
        return False

