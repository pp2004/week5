
## Project Overview

This repository provides a complete, drop-in Streamlit application to chat with Azure’s GPT-4o. It supports:

- **Standard chat**: Ask GPT-4o questions, get conversational responses.  
- **PDF RAG**: Upload a PDF, ChromaDB ingests and indexes text chunks. GPT-4o can reference document content during the conversation.  
- **Session logging**: All messages (user + assistant) are stored in a SQLite database (`chat_logs.db`) by default. You can switch to PostgreSQL if desired.  
- **Admin panel**: View or delete past chat logs, protected by a simple sidebar password.  

By following these instructions, you’ll have a working local version—no Poetry/pyproject.toml required; all dependencies are managed strictly via `requirements.txt`.

---

## Features

- **Azure OpenAI Integration**  
  - Fully configurable through sidebar: endpoint, API key, deployment name, API version.  
- **Retrieval-Augmented Generation (RAG)**  
  - Upload any PDF; text is chunked & vector-indexed with ChromaDB & HNSWlib.  
  - GPT-4o can retrieve relevant PDF snippets on demand, displaying context indicators.  
- **Real-time Chat**  
  - Streamlit’s `st.chat_message` bubbles (or fallback markdown) for user/assistant messages.  
  - Temperature & Max Tokens sliders in the sidebar to control generation.  
- **Session Persistence**  
  - SQLite by default (no additional setup).  
  - Optional PostgreSQL support if you prefer a server-based DB.  
- **Admin Panel**  
  - Protected by `ADMIN_PASSWORD` (set in `.env`).  
  - View, search, and delete chat logs.  
- **Robust Dependency Handling**  
  - Pre-pinned versions to avoid known conflicts (Protobuf, OpenTelemetry, ChromaDB).  
  - Instructions included to resolve common Windows build tool issues.  

---

## Prerequisites

- **Operating System**  
  - macOS / Linux / Windows 10+  
- **Python**  
  - 3.9 ≤ Python < 3.13 (we recommend 3.10 or 3.11).  
- **Git** (optional, for cloning).  
- **(Windows Only)**: Microsoft Visual C++ Build Tools (for building C extensions like hnswlib).  
- **Azure OpenAI Access**  
  - You need an Azure OpenAI resource with a GPT-4o deployment.  
  - Know your:  
    - Endpoint (e.g. `https://my-resource.openai.azure.com/`)  
    - API Key  
    - Deployment Name (e.g. `gpt-4o`)  
    - API Version (e.g. `2023-05-15`).

---

## Project Structure

```
AzureAiChatbot/
├── app.py                     # Main Streamlit entry point
├── chat_engine.py             # Core logic: Azure OpenAI calls, PDF processing, RAG
├── config.py                  # Sidebar config + Azure/OpenTelemetry setup
├── db.py                      # SQLAlchemy models & DB utility functions
├── utils.py                   # Helper functions (session state, validation, etc.)
├── requirements.txt           # All pinned dependencies
├── .env.example               # Example environment variables
└── pages/
    ├── __init__.py
    ├── 1_Chat.py              # Chat interface
    └── 2_Admin.py             # Admin panel
```

- **`app.py`**: Simply routes to the pages based on Streamlit’s multipage feature.  
- **`chat_engine.py`**:  
  - `ask_gpt(history, azure_config, ...)`  
  - `process_uploaded_pdf(file)`  
  - `search_pdf_context(query)`  
  - `get_pdf_stats()`  
- **`config.py`**:  
  - `setup_sidebar_config()` reads/writes `st.session_state` for Azure settings.  
  - Validates credentials, calls `st.experimental_rerun()` or `st.stop()` on invalid config.  
- **`db.py`**:  
  - SQLAlchemy setup (`engine`, `Base`, `SessionLocal`).  
  - `log_message(...)` to insert chat records.  
  - `get_all_logs()`, `delete_log(id)`, etc.  
- **`utils.py`**:  
  - Session initialization (`initialize_session_state()`).  
  - PDF validation (`validate_pdf_file()`, `format_file_size()`).  
  - Error popups (`show_error_with_details()`, `show_success_message()`).  

---

## Installation

Follow these steps exactly **in the same Python interpreter** you will use to run Streamlit. We strongly recommend using a **virtual environment** or **Conda environment** to isolate dependencies.

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/AzureAiChatbot.git
cd AzureAiChatbot
```

> Alternatively, download and unzip the `AzureAiChatbot_project.zip` provided.

### 2. Python Version & Interpreter

Ensure you have a supported Python:

```bash
python3 --version
# Should be 3.9.x, 3.10.x, or 3.11.x
```

If you have multiple Python installations, use the one you intend for Streamlit:

- **macOS/Linux**: `python3` or explicit `python3.10`  
- **Windows**: `py -3.10` (to pick Python 3.10), or your virtual environment’s `python`

### 3. Install Build Tools (Windows Only)

ChromaDB depends on `hnswlib`, which requires a C compiler on Windows. If you see errors about “Microsoft Visual C++ 14.0 or greater is required,” do the following:

1. Download and install [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/).  
2. Select the **“Desktop development with C++”** workload.  
3. Restart your PowerShell/CMD to pick up the new `cl.exe` compiler.  

If you cannot install the Build Tools, skip to **Option B** (Conda) or use a prebuilt hnswlib wheel (see [ChromaDB & hnswlib](#chromadb--hnswlib) below).

### 4. Install Dependencies via `requirements.txt`

#### 4.1. Upgrade Pip & Packaging

```bash
# macOS/Linux
python3 -m pip install --upgrade pip setuptools wheel

# Windows (PowerShell)
py -3.10 -m pip install --upgrade pip setuptools wheel
```

#### 4.2. Install Everything

```bash
# macOS/Linux
python3 -m pip install -r requirements.txt

# Windows (PowerShell)
py -3.10 -m pip install -r requirements.txt
```

Watch the console carefully. You should see:

- **`protobuf-5.26.0`** (or your chosen 5.x)  
- **All `opentelemetry-*-1.33.1`** packages  
- **`streamlit-1.30.0`**  
- **RAG dependencies** (`chromadb-0.4.0`, `sentence-transformers-2.2.2`, `PyPDF2-3.0.0`, `hnswlib-0.5.1`)  
- **Charting** (`altair-4.2.2`, `vega_datasets-0.9.0`)  
- **`openai>=0.27.0`**, `SQLAlchemy-1.4.54`, `pandas-1.5.3`, `python-dotenv-0.20.0`

If you encounter errors about missing compilers for `hnswlib`, either:

- Install Visual C++ Build Tools (see Step 3), or  
- Use Conda:  
  ```bash
  conda create -n chat-env python=3.10 -y
  conda activate chat-env
  conda install -c conda-forge chromadb -y
  pip install -r requirements.txt   # Remove chromadb from requirements.txt before
  ```

### 5. Configure Environment Variables

Copy the example `.env` and fill in your Azure/OpenAI credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2023-05-15

ADMIN_PASSWORD=YourAdminPassword

# Default: SQLite local DB
DATABASE_URL=sqlite:///./chat_logs.db

# Uncomment for PostgreSQL (if you installed psycopg2-binary):
# DATABASE_URL=postgresql+psycopg2://user:password@hostname/database_name
```

- **`AZURE_OPENAI_…`**: Your Azure OpenAI resource details.  
- **`ADMIN_PASSWORD`**: Protects the Admin panel.  
- **`DATABASE_URL`**:  
  - Default: SQLite file `./chat_logs.db`.  
  - PostgreSQL example: `postgresql+psycopg2://chatuser:chatpassword@localhost/chatdb`.

---

## Running the App

From the project root:

```bash
# macOS/Linux
python3 -m streamlit run app.py

# Windows PowerShell
py -3.10 -m streamlit run app.py
```

- By default, Streamlit serves on [http://localhost:8501](http://localhost:8501).  
- To choose another port, e.g. 5000:
  ```bash
  python3 -m streamlit run app.py --server.port 5000
  ```

Stop the server with **Ctrl +C**.

---

## Usage

### Chat Interface

1. Navigate to **Chat** (default landing page).  
2. If you have not configured Azure credentials, you will see a prompt in the main pane. On the sidebar:  
   - Enter **Endpoint**, **API Key**, **Deployment Name**, **API Version**.  
   - Click **Save**. The page will automatically rerun.  
3. Use the **Temperature** and **Max Tokens** sliders to adjust generation (defaults saved in session).  
4. Type your message into the **“Type your message here…”** box and click **Send**.  
   - The chat history appears in “chat bubbles” (or markdown fallback).  
   - If you have processed a PDF, GPT-4o will retrieve relevant chunks and display 📄 context indicators for assistant replies.  

### PDF Ingestion & RAG

1. In the sidebar under **“📄 PDF Upload & RAG”**, click **Upload PDF** and select a valid PDF (≤ 50 MB recommended).  
2. If the file is valid, you will see filename + size.  
3. Click **🔄 Process PDF**. A spinner appears while ChromaDB ingests & indexes chunks.  
4. On success, you’ll see **✅ PDF processed successfully!** and chunk count.  
5. Now, any new user prompt automatically triggers RAG:  
   - The app calls `search_pdf_context(prompt)` to get a list of relevant chunks.  
   - These chunks are passed as `pdf_context` to `ask_gpt(...)`.  
   - The assistant reply contains an indicator if PDF context was used.  
6. To replace or switch PDFs, simply upload a different file; old index is discarded and new file is processed.

### Admin Panel

1. Click **Admin** in the left sidebar (or visit `/?page=Admin`).  
2. Enter the `ADMIN_PASSWORD` (set in your `.env`).  
3. You can:
   - **View all logs**: ID, timestamp, user prompt, assistant response, tokens used, PDF context flag.  
   - **Delete** individual logs by clicking a “Delete” button next to each record.  
4. Use this to manage historical data or clear old sessions.
