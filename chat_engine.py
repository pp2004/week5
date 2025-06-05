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
    # We will raise an error at runtime if someone actually tries to use AzureOpenAI.


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
            endpoint: Azure OpenAI endpoint (e.g. "https://<your-resource>.openai.azure.com/")
            api_key: Your Azure OpenAI API key
            deployment_name: The name of the deployed model (e.g. "o4-mini-4")
            api_version: The API version (e.g. "2023-05-15")
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
            messages: List of {"role":"user"/"assistant"/"system", "content": "…"}
            temperature: Sampling temperature
            max_tokens: Maximum tokens for the response (we will send as max_completion_tokens)
            stream: Whether to stream partial results

        Returns:
            Tuple of (response_content: str, tokens_used: int)
        """
        try:
            # ─── Azure REST call ──────────────────────────────────────────────────
            # IMPORTANT: use max_completion_tokens instead of max_tokens
            if stream:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,   # <- swapped here
                    stream=True
                )
                # Collect streamed chunks into full_content
                full_content = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        full_content += delta.content
                        # (If you want to yield partials, you could, but our function 
                        #  always returns a final string. So we accumulate.)
                tokens_used = int(len(full_content.split()) * 1.3)
                return full_content, tokens_used

            else:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,   # <- and here
                    stream=False
                )
                full_text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0
                return full_text, tokens_used

        except Exception as e:
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
        Initialize RAG dependencies: PyPDF2, chromadb, SentenceTransformer.
        Returns True on success; False otherwise.
        """
        if self.rag_available:
            return True

        if not lazy_import_rag_dependencies():
            return False

        try:
            # Create or open local ChromaDB at "./chroma_db"
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
        Raises if RAG is not available or extraction fails.
        """
        if not self.initialize_rag():
            raise Exception("RAG dependencies not available; cannot extract PDF text.")

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
        Break a long text into overlapping chunks (for vector embedding).
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
        Process the uploaded PDF: extract text, chunk it, embed each chunk,
        and store embeddings in ChromaDB.
        Returns True on success, False otherwise.
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
        Query the vector store to find up to n_results PDF chunks most similar to `query`.
        """
        if not self.rag_available or not self.collection:
            return []

        try:
            query_embedding = self.embedder.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            if results and results.get('documents'):
                return results['documents'][0]
            return []
        except Exception as e:
            st.error(f"Failed to search similar chunks: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict:
        """
        Return basic stats on the ChromaDB collection { "count": <int>, "error": <optional> }.
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
    Ask GPT with optional PDF context. ALWAYS return exactly three values:
       (response: str, tokens_used: int, success: bool).

    If an error occurs, the first element is the error message, tokens=0, success=False.
    """
    try:
        # 1) Initialize Azure client
        client = AzureGPTClient(
            endpoint=azure_config["endpoint"],
            api_key=azure_config["api_key"],
            deployment_name=azure_config["deployment_name"],
            api_version=azure_config["api_version"]
        )

        # 2) Prepend PDF context as a system message if provided
        enhanced_messages = messages.copy()
        if pdf_context:
            context_text = "\n\n".join(pdf_context)
            system_msg = {
                "role": "system",
                "content": (
                    "You are an AI assistant with access to relevant document context. "
                    "Use the following context to help answer the user's question:\n\n"
                    f"{context_text}\n\n"
                    "If the context is relevant, incorporate it; otherwise answer from your general knowledge."
                )
            }
            if enhanced_messages and enhanced_messages[0].get("role") == "system":
                enhanced_messages[0] = system_msg
            else:
                enhanced_messages.insert(0, system_msg)

        # 3) Call Azure with max_completion_tokens instead of max_tokens
        if stream:
            full_response = ""
            tokens_used = 0
            for delta in client.get_chat_completion(
                enhanced_messages,
                temperature=temperature,
                max_tokens=max_tokens,       # this argument will be ignored by get_chat_completion
                stream=True
            ):
                # Our get_chat_completion yields chunks, but we collect them fully here
                full_response += delta
            tokens_used = int(len(full_response.split()) * 1.3)
            return full_response, tokens_used, True

        else:
            content, tokens = client.get_chat_completion(
                enhanced_messages,
                temperature=temperature,
                max_tokens=max_tokens,       # passed into get_chat_completion → used as max_completion_tokens
                stream=False
            )
            return content, tokens, True

    except Exception as e:
        err_str = f"Failed to get GPT response: {str(e)}"
        st.error(err_str)
        return err_str, 0, False


# Create one global PDF processor instance so we don’t reload models each time
pdf_processor = PDFProcessor()


def process_uploaded_pdf(uploaded_file) -> bool:
    """
    Wrapper around pdf_processor.process_pdf(...).
    """
    return pdf_processor.process_pdf(uploaded_file, uploaded_file.name)


def search_pdf_context(query: str) -> List[str]:
    """
    Wrapper around pdf_processor.search_similar_chunks(...).
    """
    return pdf_processor.search_similar_chunks(query, n_results=3)


def get_pdf_stats() -> Dict:
    """
    Wrapper around pdf_processor.get_collection_stats().
    """
    return pdf_processor.get_collection_stats()

