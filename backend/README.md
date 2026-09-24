# Enterprise Hybrid RAG Backend Service

High-performance dual-index retrieval orchestration layers powered by FastAPI, ChromaDB, BM25, and SentenceTransformers.

## Features

- **FastAPI Endpoints**: `/v1/ask`, `/v1/ingest`, `/health`, `/`
- **Dual Retrieval**: BM25 keyword matching + ChromaDB vector embeddings
- **Reciprocal Rank Fusion & Cross-Encoder Re-Ranking**
- **Citation Verification Guardrail**

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

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and set your Groq API key:
```bash
cp .env.example .env
```

### 4. Run the Backend API Server
```bash
python run.py
# Or directly via uvicorn:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs: `http://localhost:8000/docs`

### 5. Run Tests
```bash
pytest tests/ -v
```
