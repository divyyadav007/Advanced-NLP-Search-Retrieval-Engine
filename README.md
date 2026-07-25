# Enterprise Hybrid RAG Engine

An enterprise-grade, asynchronous, and fully decoupled Retrieval-Augmented Generation (RAG) engine engineered to perform accurate, citation-verified question answering over complex institutional compliance specifications and corporate PDF/HTML/Markdown files.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Orchestration-Docker-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Short Project Summary
This project implements a complete RAG pipeline featuring double-engine retrieval (lexical BM25 and semantic vector search) combined using Reciprocal Rank Fusion (RRF) and re-ranked using a neural Cross-Encoder model. The system operates either as a standalone local desktop pipeline or as a decoupled production-ready microservice architecture. A deterministic citation verification guardrail matches LLM-generated assertions directly back to character spans in document chunks to prevent hallucination drift.

---

## Project Preview
![Streamlit Interface Mockup](https://github.com/user-attachments/assets/924d09f9-2e4f-46a9-9214-caf42d7f8794)
*Streamlit Dashboard featuring Query Interfaces, Citation Verifiers, and the Asset Ingestion panel.*

---

## Key Features
* **Dual-Index Search Ingress**: Merges exact keyword matching (**BM25Okapi**) with neural similarity search (**ChromaDB HNSW graph**).
* **Reciprocal Rank Fusion (RRF)**: Fuses sparse and dense candidate lists using rank-based reciprocal scaling parameters.
* **Neural Re-Ranking Pass**: Maximizes context density and filters background noise using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
* **Microservices Integration**: Decoupled design where the UI can offload indexing and query execution to the FastAPI backend service.
* **Citation Trace Auditing**: Deterministically audits bracketed LLM references (e.g. `[1]`) against source index spans to ensure truthfulness.
* **Unicode & Text Normalization**: Prevents database and Pydantic validation crashes by filtering null bytes, surrogates, and PDF bullet extraction gluing errors.
* **Semantic Deduplicator Node**: Employs cosine embeddings to remove redundant incoming chunks (>95% similarity) to minimize LLM token costs.

---

## Architecture Overview

When a document is uploaded, it is cleaned, split into chunks, deduplicated, and indexed into both a BM25 sparse index and a ChromaDB dense vector store.

When a query is submitted, both search indexes retrieve candidate chunks. Reciprocal Rank Fusion (RRF) merges the results, a Cross-Encoder re-ranks top candidates, the LLM generates an answer based on those context blocks, and a verifier checks the citations.

```mermaid
graph TD
    A[Source Documents] --> B[Sanitization & Text Cleaning]
    B --> C[Chunking Engine]
    C --> D[Semantic Deduplication]
    D --> E[Sparse BM25 Index]
    D --> F[Dense Vector Index: ChromaDB]

    G[User Query] --> E
    G --> F
    E --> H[Reciprocal Rank Fusion RRF]
    F --> H
    H --> I[Cross-Encoder Reranker]
    I --> J[Grounded LLM Generation: Groq]
    J --> K[Citation Verifier Guardrail]
    K --> L[Verified Response]
```

---

## Tech Stack

| Category | Technology | Usage Rationale |
| :--- | :--- | :--- |
| **Languages** | Python | Core programming language for AI/ML pipelines and API backend. |
| **Backend** | FastAPI / Uvicorn | High-performance asynchronous web framework for REST API endpoints. |
| **Frontend** | Streamlit | Python-native web interface framework for interactive dashboards. |
| **AI / ML** | SentenceTransformers | Local embedding model (`all-MiniLM-L6-v2`) and re-ranker (`ms-marco-MiniLM-L-6-v2`). |
| **AI / ML** | Rank-BM25 | BM25Okapi algorithm implementation for lexical keyword search. |
| **AI / ML** | Groq SDK | High-speed LLM inference API (`llama-3.1-8b-instant`). |
| **Database** | ChromaDB | Local vector database using HNSW graph indexing and persistent SQLite storage. |
| **Tools** | Docker / Compose | Containerization for reproducible microservice deployment. |
| **Tools** | Pytest | Test framework for unit and API route verification. |

---

## Folder Structure

```text
Advanced-NLP-Search-Retrieval-Engine/
├── data/                 # ChromaDB vector database and sparse index files
├── sample_data/          # Reference sample policy document
├── src/                  # Core source code modules
│   ├── evaluation/       # RAG metric evaluation scripts
│   ├── generation/       # LLM generation and citation verifier
│   ├── indexing/         # BM25, ChromaDB, and hybrid retrieval logic
│   ├── ingestion/        # Parsers, chunkers, and deduplication logic
│   ├── reranking/        # Cross-Encoder model re-ranking
│   └── ui/               # Streamlit application dashboard
├── tests/                # Automated unit test suite
├── Dockerfile            # Container definition
├── docker-compose.yaml   # Multi-container orchestration specification
├── Makefile              # Utility command shortcuts
├── pyproject.toml        # Package and tool configuration
├── requirements.txt      # Dependency manifest
├── run.py                # Streamlit UI direct entrypoint
└── README.md             # Project documentation
```

### Purpose of Key Folders
- `src/ingestion/`: Parses documents (PDF, HTML, MD, TXT), cleans text, chunks documents, and removes duplicate content.
- `src/indexing/`: Manages sparse BM25 indexing, ChromaDB vector database storage, and Reciprocal Rank Fusion.
- `src/reranking/`: Uses a Cross-Encoder model to select the top context blocks for the LLM.
- `src/generation/`: Builds prompts for Groq LLM generation and verifies cited references.
- `src/ui/`: Contains the Streamlit user dashboard code.
- `tests/`: Unit tests for API endpoints, ingestion, retrieval, and verification logic.

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- A Groq API key from [Groq Console](https://console.groq.com/)

### Step 1: Clone Repository
```bash
git clone https://github.com/divyyadav007/RAG-Pipeline-with-Hybrid-Search-Over-Internal-Docs.git
cd Advanced-NLP-Search-Retrieval-Engine
```

### Step 2: Create Virtual Environment
```bash
# On Linux / macOS
python3 -m venv myvenv
source myvenv/bin/activate

# On Windows (PowerShell)
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory:
```bash
GROQ_API_KEY="your_groq_api_key_here"
```

### Step 5: Run Backend API
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Documentation: `http://localhost:8000/docs`

### Step 6: Run Frontend UI
In a separate terminal window:
```bash
python run.py
```
- Web Application: `http://localhost:8501`

### Running via Docker Compose
To run both backend API and frontend UI in containers:
```bash
# Set environment variable and start containers
export GROQ_API_KEY="your_groq_api_key_here"   # On Linux/macOS
$env:GROQ_API_KEY="your_groq_api_key_here"      # On Windows PowerShell

docker compose up --build
```

---

## Environment Variables

| Variable | Description | Required |
| :--- | :--- | :---: |
| `GROQ_API_KEY` | API authentication key for Groq LLM inference services. | **Yes** |
| `BACKEND_API_URL` | Base URL of backend FastAPI service. Used by Streamlit UI in microservice mode. | Optional |
| `PYTHONPATH` | Root path for resolving local Python module imports. | Optional |

---

## Usage

1. Open the web interface at `http://localhost:8501`.
2. **Ingest Documents**: In the right-hand panel, upload a document (`.pdf`, `.md`, `.txt`, `.html`) and click **Trigger Asset Ingestion Pipeline**. The document will be parsed, chunked, deduplicated, and indexed.
3. **Ask Questions**: In the left-hand panel, enter a question (e.g., *"What is the minimum attendance requirement for university exams?"*) and click **Execute Intelligence Query**.
4. **View Output**: The dashboard displays the synthesized answer along with a citation verification badge confirming whether all cited source indices are valid.

---

## API Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Returns API status and metadata. |
| `POST` | `/v1/ingest` | Parses, chunks, deduplicates, and indexes a file path into search stores. |
| `POST` | `/v1/ask` | Performs hybrid search, re-ranking, LLM answer synthesis, and citation verification. |
| `GET` | `/health` | Liveness health check endpoint. |

---

## Screenshots

![Dashboard Interface](https://github.com/user-attachments/assets/924d09f9-2e4f-46a9-9214-caf42d7f8794)
*Figure 1: Streamlit User Dashboard showing Query Interface, Citation Diagnostics, and Ingestion Panel.*

---

## Future Improvements

- Add metadata pre-filtering by category, document type, or date.
- Integrate multi-turn conversational session memory.
- Add background task queue processing for large PDF files.
- Support fully offline LLM generation via Ollama or llama.cpp.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
