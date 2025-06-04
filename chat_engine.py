"""
chat_engine.py

Chat engine for Azure GPT-4o Chat Application
Handles Azure OpenAI API calls, PDF processing, and RAG functionality
"""

import os
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
    # We do NOT call st.error() at import time (this can break Streamlit's ordering);
    # we will raise or show errors only when the user actually tries to call Azure.


# PDF and RAG imports - lazy loaded to avoid startup errors
def lazy_import_rag_dependencies() -> bool:
    """Lazy import RAG dependencies to avoid startup errors."""
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

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str
    ):
        """
        Initialize Azure OpenAI client.

        Args:
            endpoint: Azure OpenAI endpoint (e.g. https://myresource.openai.azure.com/)
            api_key: Azure OpenAI API key
            deployment_name: Name of the deployed model (e.g. "gpt-4o-deployment")
            api_version: Azure OpenAI API version (e.g. "2023-05-15")
        """
        if not AZURE_AVAILABLE:
            raise ImportError("Azure OpenAI libraries not installed. Please `pip install openai`.")

        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version

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
        Get chat completion from Azure OpenAI.

        Args:
            messages: List of message dicts, e.g. [{"role":"system","content":"..."}, ...]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: If True, returns partial chunks; if False, returns full content

        Returns:
            (response_content: str, tokens_used: int)

        Raises:
            Exception if the API call fails.
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
                # STREAMING: collect all partial deltas into a single string
                full_content = ""
                for chunk in response:
                    # each chunk.choices[0].delta.content may be a partial string
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        full_content += delta.content
                # Rough token‐usage estimate
                tokens_used = int(len(full_content.split()) * 1.3)
                return full_content, tokens_used

            else:
                # NON‐STREAMING: just read the final choices[0].message.content
                full_message = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0
                return full_message, tokens_used

        except Exception as e:
            # Bubble up as a generic Exception
            raise Exception(f"Azure OpenAI API error: {str(e)}")


class PDFProcessor:
    """PDF processing and RAG functionality"""

    def __init__(self):
        """Initialize PDF processor (lazy RAG)."""
        self.chroma_client = None
        self.collection = None
        self.embedder = None
        self.rag_available = False

    def initialize_rag(self) -> bool:
        """
        Initialize RAG dependencies (PyPDF2, chromadb, SentenceTransformer).

        Returns True if successful, False otherwise.
        """
        if self.rag_available:
            return True

        if not lazy_import_rag_dependencies():
            return False

        try:
            # Create persistent ChromaDB folder “./chroma_db”
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.rag_available = True
            return True

        except Exception as e:
            st.error(f"Failed to initialize RAG: {str(e)}")
            return False

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract all text from a Streamlit-uploaded PDF.

        Args:
            pdf_file: Streamlit file object.

        Returns:
            The entire text (all pages concatenated).

        Raises:
            Exception if extraction fails.
        """
        if not self.initialize_rag():
            raise Exception("RAG dependencies not available; cannot extract text.")

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
        Break a long text into overlapping chunks (for embedding).

        Args:
            text: The full text string.
            chunk_size: Maximum characters per chunk.
            overlap: Characters to overlap between chunks.

        Returns:
            A list of chunk strings.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            fragment = text[start:end].strip()
            if fragment:
                chunks.append(fragment)

            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def process_pdf(self, pdf_file, filename: str) -> bool:
        """
        Process an uploaded PDF: extract text, chunk, embed, and store in ChromaDB.

        Args:
            pdf_file: Streamlit file object.
            filename: The original filename (used in metadata).

        Returns:
            True if processing succeeded, False otherwise.
        """
        if not self.initialize_rag():
            return False

        try:
            text = self.extract_text_from_pdf(pdf_file)
            if not text.strip():
                st.error("No text could be extracted from the PDF.")
                return False

            chunks = self.chunk_text(text, chunk_size=1000, overlap=100)
            if not chunks:
                st.error("No chunks could be created from the PDF text.")
                return False

            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.embedder.encode(chunk).tolist()
                    chunk_id = hashlib.md5(f"{filename}_{i}_{chunk[:50]}".encode()).hexdigest()
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

                except Exception as chunk_e:
                    st.warning(f"Failed to process chunk {i}: {str(chunk_e)}")
                    continue

            progress_bar.empty()
            st.success(f"Successfully processed {len(chunks)} chunks from “{filename}”.")
            return True

        except Exception as e:
            st.error(f"Failed to process PDF: {str(e)}")
            return False

    def search_similar_chunks(self, query: str, n_results: int = 3) -> List[str]:
        """
        Query the vector store for the top‐k chunks most similar to `query`.

        Args:
            query: The user’s question or context string.
            n_results: How many top results to return.

        Returns:
            A list of up to n_results text chunks. If no data or RAG not initialized, returns [].
        """
        if not self.rag_available or not self.collection:
            return []

        try:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            # `results['documents'][0]` is a list of up to n_results chunk strings
            if results and results.get('documents'):
                return results['documents'][0]
            return []

        except Exception as e:
            st.error(f"Failed to search similar chunks: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict:
        """
        Return some basic stats about the current ChromaDB collection.

        Returns:
            A dict: { "count": <int>, "error": <optional str> }
        """
        if not self.rag_available or not self.collection:
            return {"count": 0, "error": "RAG not initialized"}
        try:
            return {"count": self.collection.count()}
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
    Ask GPT (with optional PDF context) and ALWAYS return exactly three values:
      (response_content: str, tokens_used: int, success: bool).

    Args:
        messages: Existing conversation history.
        azure_config: {
          "endpoint": <str>,
          "api_key": <str>,
          "deployment_name": <str>,
          "api_version": <str>
        }
        temperature: Sampling temperature.
        max_tokens: Max tokens to allow.
        pdf_context: Optional list of relevant PDF chunks.
        stream: Whether to use streaming. (We will ignore partial yields and just collect the full string.)

    Returns:
        A 3‐tuple: (response str, tokens_used int, success bool).
        If anything fails, `success` is False and the first element is an error message.
    """
    try:
        # 1) Initialize Azure client
        client = AzureGPTClient(
            endpoint=azure_config["endpoint"],
            api_key=azure_config["api_key"],
            deployment_name=azure_config["deployment_name"],
            api_version=azure_config["api_version"]
        )

        # 2) Build the “enhanced_messages” list, inserting PDF context as a system message if present
        enhanced_messages = messages.copy()
        if pdf_context:
            context_text = "\n\n".join(pdf_context)
            system_message = {
                "role": "system",
                "content": (
                    "You are an AI assistant with access to relevant document context. "
                    "Use the following context to help answer the user's question:\n\n"
                    f"{context_text}\n\n"
                    "If the context is relevant, incorporate it; otherwise answer from your general knowledge."
                )
            }
            if enhanced_messages and enhanced_messages[0].get("role") == "system":
                enhanced_messages[0] = system_message
            else:
                enhanced_messages.insert(0, system_message)

        # 3) Call get_chat_completion and ALWAYS collect into a single string
        if stream:
            # Even if stream=True, we collect everything and return it at once
            full_content = ""
            tokens_used = 0
            # get_chat_completion yields partial chunks when stream=True
            for delta_chunk in client.get_chat_completion(
                enhanced_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            ):
                full_content += delta_chunk
            # Estimate token usage
            tokens_used = int(len(full_content.split()) * 1.3)
            return full_content, tokens_used, True

        else:
            # Non‐streaming path: get a 2‐tuple (content, tokens) back
            content, tokens_used = client.get_chat_completion(
                enhanced_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return content, tokens_used, True

    except Exception as e:
        error_msg = f"Failed to get GPT response: {str(e)}"
        st.error(error_msg)
        return error_msg, 0, False


# -----------------------------------------------------------------------------
# Instantiate a single global PDFProcessor so that we don’t reload models each time
pdf_processor = PDFProcessor()


def process_uploaded_pdf(uploaded_file) -> bool:
    """
    Wrapper around pdf_processor.process_pdf(...) for the Streamlit front‐end.

    Args:
        uploaded_file: Streamlit PDF file.

    Returns:
        True if successful, False otherwise.
    """
    return pdf_processor.process_pdf(uploaded_file, uploaded_file.name)


def search_pdf_context(query: str) -> List[str]:
    """
    Wrapper around pdf_processor.search_similar_chunks(...).

    Args:
        query: The user’s question or context string.

    Returns:
        A list of up to 3 PDF text chunks that appear most relevant.
    """
    return pdf_processor.search_similar_chunks(query, n_results=3)


def get_pdf_stats() -> Dict:
    """
    Wrapper around pdf_processor.get_collection_stats().

    Returns:
        A dict: {"count": <int>, "error": <optional str>}
    """
    return pdf_processor.get_collection_stats()