"""
Chat engine for Azure GPT-4o Chat Application
Handles Azure OpenAI API calls, PDF processing, and RAG functionality
"""

import os
import time
import hashlib
import streamlit as st
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Azure OpenAI imports
try:
    from openai import AzureOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    st.error("Azure OpenAI libraries not available. Please install openai.")

# PDF and RAG imports - lazy loaded to avoid startup errors
def lazy_import_rag_dependencies():
    """Lazy import RAG dependencies to avoid startup errors"""
    
    global PyPDF2, chromadb, SentenceTransformer
    
    try:
        import PyPDF2
        import chromadb
        from sentence_transformers import SentenceTransformer
        return True
    except ImportError as e:
        st.error(f"RAG dependencies not available: {str(e)}")
        return False

class AzureGPTClient:
    """Azure OpenAI GPT client for chat completions"""
    
    def __init__(self, endpoint: str, api_key: str, deployment_name: str, api_version: str):
        """
        Initialize Azure OpenAI client
        
        Args:
            endpoint: Azure OpenAI endpoint
            api_key: API key
            deployment_name: Deployment name
            api_version: API version
        """
        
        if not AZURE_AVAILABLE:
            raise ImportError("Azure OpenAI libraries not available")
        
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        
        # Initialize client
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Azure OpenAI client: {str(e)}")
    
    def get_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Tuple[str, int]:
        """
        Get chat completion from Azure OpenAI
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature for response
            max_tokens: Maximum tokens for response
            stream: Whether to stream response
        
        Returns:
            Tuple of (response_content, tokens_used)
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                # Handle streaming response
                content = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                        yield chunk.choices[0].delta.content
                
                # Estimate token usage for streaming
                tokens_used = len(content.split()) * 1.3  # Rough estimation
                return content, int(tokens_used)
            else:
                # Handle non-streaming response
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0
                
                return content, tokens_used
                
        except Exception as e:
            raise Exception(f"Azure OpenAI API error: {str(e)}")

class PDFProcessor:
    """PDF processing and RAG functionality"""
    
    def __init__(self):
        """Initialize PDF processor"""
        
        self.chroma_client = None
        self.collection = None
        self.embedder = None
        self.rag_available = False
    
    def initialize_rag(self) -> bool:
        """
        Initialize RAG dependencies (lazy loading)
        
        Returns:
            Boolean indicating success
        """
        
        if self.rag_available:
            return True
        
        if not lazy_import_rag_dependencies():
            return False
        
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize sentence transformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.rag_available = True
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize RAG: {str(e)}")
            return False
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from uploaded PDF file
        
        Args:
            pdf_file: Streamlit uploaded file object
        
        Returns:
            Extracted text content
        """
        
        if not self.initialize_rag():
            raise Exception("RAG dependencies not available")
        
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Chunk text into smaller passages
        
        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Find last space before end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_pdf(self, pdf_file, filename: str) -> bool:
        """
        Process PDF file and store in vector database
        
        Args:
            pdf_file: Streamlit uploaded file object
            filename: Name of the PDF file
        
        Returns:
            Boolean indicating success
        """
        
        if not self.initialize_rag():
            return False
        
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                st.error("No text could be extracted from the PDF")
                return False
            
            # Chunk text
            chunks = self.chunk_text(text, chunk_size=1000, overlap=100)
            
            if not chunks:
                st.error("No chunks could be created from the PDF text")
                return False
            
            # Generate embeddings and store in ChromaDB
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self.embedder.encode(chunk).tolist()
                    
                    # Create unique ID for chunk
                    chunk_id = hashlib.md5(f"{filename}_{i}_{chunk[:50]}".encode()).hexdigest()
                    
                    # Store in collection
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        ids=[chunk_id],
                        metadatas=[{
                            "filename": filename,
                            "chunk_index": i,
                            "timestamp": datetime.now().isoformat()
                        }]
                    )
                    
                    progress_bar.progress((i + 1) / len(chunks))
                    
                except Exception as e:
                    st.warning(f"Failed to process chunk {i}: {str(e)}")
                    continue
            
            progress_bar.empty()
            st.success(f"Successfully processed {len(chunks)} chunks from {filename}")
            return True
            
        except Exception as e:
            st.error(f"Failed to process PDF: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, n_results: int = 3) -> List[str]:
        """
        Search for similar chunks in the vector database
        
        Args:
            query: Search query
            n_results: Number of results to return
        
        Returns:
            List of similar text chunks
        """
        
        if not self.rag_available or not self.collection:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Extract documents
            if results and results['documents']:
                return results['documents'][0]
            
            return []
            
        except Exception as e:
            st.error(f"Failed to search similar chunks: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector collection
        
        Returns:
            Dictionary with collection statistics
        """
        
        if not self.rag_available or not self.collection:
            return {"count": 0, "error": "RAG not initialized"}
        
        try:
            count = self.collection.count()
            return {"count": count}
            
        except Exception as e:
            return {"count": 0, "error": str(e)}

def ask_gpt(
    messages: List[Dict[str, str]],
    azure_config: Dict[str, str],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    pdf_context: Optional[List[str]] = None,
    stream: bool = False
) -> Tuple[str, int, bool]:
    """
    Ask GPT with optional PDF context
    
    Args:
        messages: Conversation messages
        azure_config: Azure OpenAI configuration
        temperature: Temperature setting
        max_tokens: Max tokens setting
        pdf_context: Optional PDF context chunks
        stream: Whether to stream response
    
    Returns:
        Tuple of (response, tokens_used, success)
    """
    
    try:
        # Initialize Azure client
        client = AzureGPTClient(
            endpoint=azure_config['endpoint'],
            api_key=azure_config['api_key'],
            deployment_name=azure_config['deployment_name'],
            api_version=azure_config['api_version']
        )
        
        # Prepare messages with PDF context
        enhanced_messages = messages.copy()
        
        if pdf_context:
            # Add PDF context as system message
            context_text = "\n\n".join(pdf_context)
            system_message = {
                "role": "system",
                "content": f"You are an AI assistant with access to relevant document context. Use the following context to help answer the user's question:\n\n{context_text}\n\nIf the context is relevant to the user's question, incorporate it into your response. If not, provide a helpful response based on your general knowledge."
            }
            
            # Insert system message at the beginning or update existing one
            if enhanced_messages and enhanced_messages[0]["role"] == "system":
                enhanced_messages[0] = system_message
            else:
                enhanced_messages.insert(0, system_message)
        
        # Get response
        if stream:
            return client.get_chat_completion(enhanced_messages, temperature, max_tokens, stream=True)
        else:
            response, tokens = client.get_chat_completion(enhanced_messages, temperature, max_tokens)
            return response, tokens, True
            
    except Exception as e:
        error_msg = f"Failed to get GPT response: {str(e)}"
        st.error(error_msg)
        return error_msg, 0, False

# Global PDF processor instance
pdf_processor = PDFProcessor()

def process_uploaded_pdf(uploaded_file) -> bool:
    """
    Process uploaded PDF file
    
    Args:
        uploaded_file: Streamlit uploaded file
    
    Returns:
        Boolean indicating success
    """
    
    return pdf_processor.process_pdf(uploaded_file, uploaded_file.name)

def search_pdf_context(query: str) -> List[str]:
    """
    Search for relevant PDF context
    
    Args:
        query: Search query
    
    Returns:
        List of relevant text chunks
    """
    
    return pdf_processor.search_similar_chunks(query, n_results=3)

def get_pdf_stats() -> Dict:
    """
    Get PDF collection statistics
    
    Returns:
        Dictionary with statistics
    """
    
    return pdf_processor.get_collection_stats()

