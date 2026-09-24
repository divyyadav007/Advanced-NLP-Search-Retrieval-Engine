# Enterprise Hybrid RAG Frontend Dashboard

Streamlit-based user interface for query execution, knowledge synthesis, citation verification diagnostics, and document ingestion.

## Features

- **Query Interface**: Execute natural language queries with verified citation tracking.
- **Ingestion Control Panel**: Drag-and-drop ingestion of PDF, Markdown, TXT, and HTML files.
- **Dual Mode**:
  - **Microservice Mode (Default)**: Connects to the FastAPI backend API via HTTP. Minimal local dependencies.
  - **Standalone Mode**: Can run directly in-process when backend packages are installed.

## Quick Start

### 1. Set Up Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Backend Connection (Optional)
By default, the frontend connects to `http://localhost:8000`. You can configure this via `.env`:
```bash
cp .env.example .env
```

### 4. Run the Streamlit Dashboard
```bash
python run.py
# Or directly via streamlit:
streamlit run src/app.py
```
Open your browser at `http://localhost:8501`.
