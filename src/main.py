import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import config
from src.ingestion.parsers import DocumentParserRouter
from src.ingestion.chunkers import ChunkingEngine
from src.ingestion.deduplicator import ChunkDeduplicator
from src.indexing.sparse import SparseBM25Index
from src.indexing.dense import DenseVectorIndex
from src.indexing.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder import DocumentReranker
from src.generation.generator import GroundedGenerator
from src.generation.verifier import CitationVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    logger.info("Initializing Hybrid RAG Search Engine API...")
    try:
        config.validate_environment()
        
        # Load and store component instances in FastAPI state for application lifecycle
        sparse_index = SparseBM25Index()
        sparse_index.load_index()
        dense_index = DenseVectorIndex()
        
        app.state.parser_router = DocumentParserRouter()
        app.state.deduplicator = ChunkDeduplicator()
        app.state.sparse_index = sparse_index
        app.state.dense_index = dense_index
        app.state.hybrid_retriever = HybridRetriever(sparse_index, dense_index)
        app.state.reranker = DocumentReranker()
        app.state.generator = GroundedGenerator()
        
        logger.info("All engine models and indexes loaded successfully.")
    except Exception as err:
        logger.critical(f"Failed to initialize RAG Engine: {err}")
    yield
    logger.info("Shutting down API server...")


# Instantiate the FastAPI application
app = FastAPI(
    title="Enterprise Hybrid RAG Engine API",
    description="High-performance dual-index retrieval orchestration layers.",
    version="1.0.0",
    lifespan=app_lifespan
)

# Initialize state placeholders for test mock compatibility
app.state.parser_router = None
app.state.deduplicator = None
app.state.sparse_index = None
app.state.dense_index = None
app.state.hybrid_retriever = None
app.state.reranker = None
app.state.generator = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    file_path: str = Field(..., min_length=1, max_length=500, description="Path to document file")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="User query string")


from fastapi.responses import HTMLResponse


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def root_index():
    """Return API server visual landing page and interactive documentation links."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Hybrid RAG Engine API</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;800&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
            color: #f8fafc; 
            margin: 0; 
            padding: 20px; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
        }
        .card { 
            background: rgba(30, 41, 59, 0.75); 
            backdrop-filter: blur(16px); 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            border-radius: 16px; 
            padding: 40px; 
            max-width: 650px; 
            width: 100%; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.5); 
        }
        .badge { 
            display: inline-flex; 
            align-items: center; 
            gap: 6px; 
            background: rgba(34, 197, 94, 0.15); 
            color: #4ade80; 
            border: 1px solid rgba(34, 197, 94, 0.3); 
            padding: 6px 14px; 
            border-radius: 20px; 
            font-weight: 600; 
            font-size: 0.85rem; 
            margin-bottom: 20px; 
        }
        .pulse {
            width: 8px;
            height: 8px;
            background-color: #4ade80;
            border-radius: 50%;
            box-shadow: 0 0 8px #4ade80;
        }
        h1 { 
            font-family: 'Outfit', sans-serif; 
            font-size: 2.3rem; 
            margin: 0 0 12px 0; 
            background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }
        p { 
            color: #94a3b8; 
            font-size: 1rem; 
            line-height: 1.6; 
            margin-bottom: 28px; 
        }
        .btn-group { 
            display: flex; 
            gap: 12px; 
            margin-bottom: 32px; 
            flex-wrap: wrap;
        }
        .btn { 
            display: inline-flex; 
            align-items: center; 
            justify-content: center; 
            padding: 12px 24px; 
            border-radius: 8px; 
            font-weight: 600; 
            text-decoration: none; 
            transition: all 0.2s ease; 
            font-size: 0.95rem; 
        }
        .btn-primary { 
            background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%); 
            color: white; 
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4); 
        }
        .btn-primary:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6); 
        }
        .btn-secondary { 
            background: rgba(255, 255, 255, 0.08); 
            color: #f8fafc; 
            border: 1px solid rgba(255, 255, 255, 0.15); 
        }
        .btn-secondary:hover { 
            background: rgba(255, 255, 255, 0.15); 
            transform: translateY(-2px);
        }
        .endpoint-list { 
            background: rgba(15, 23, 42, 0.6); 
            border-radius: 12px; 
            padding: 18px 24px; 
            border: 1px solid rgba(255, 255, 255, 0.06); 
        }
        .endpoint-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 12px;
            font-weight: 600;
        }
        .endpoint-item { 
            display: flex; 
            align-items: center; 
            justify-content: space-between; 
            padding: 10px 0; 
            border-bottom: 1px solid rgba(255, 255, 255, 0.06); 
            font-size: 0.9rem; 
        }
        .endpoint-item:last-child { border-bottom: none; }
        .method { 
            font-weight: 700; 
            font-size: 0.75rem; 
            padding: 3px 8px; 
            border-radius: 4px; 
            margin-right: 8px;
        }
        .post { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .get { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
        .path { font-family: monospace; color: #e2e8f0; }
        .desc { color: #64748b; font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="card">
        <div class="badge"><span class="pulse"></span> Engine Online & Operational</div>
        <h1>Enterprise Hybrid RAG API</h1>
        <p>Production-grade dual-index retrieval engine combining Lexical BM25, Dense Vector HNSW (ChromaDB), Cross-Encoder Neural Re-ranking, and Grounded Llama 3.1 LLM Generation.</p>
        
        <div class="btn-group">
            <a href="/docs" class="btn btn-primary">🚀 Open API Swagger Docs (/docs)</a>
            <a href="http://localhost:8501" target="_blank" class="btn btn-secondary">🖥️ Streamlit UI Dashboard</a>
        </div>

        <div class="endpoint-list">
            <div class="endpoint-title">Available API Routes</div>
            <div class="endpoint-item">
                <div><span class="method post">POST</span><span class="path">/v1/ingest</span></div>
                <span class="desc">Parse, chunk, deduplicate & index</span>
            </div>
            <div class="endpoint-item">
                <div><span class="method post">POST</span><span class="path">/v1/ask</span></div>
                <span class="desc">Hybrid search & LLM generation</span>
            </div>
            <div class="endpoint-item">
                <div><span class="method get">GET</span><span class="path">/health</span></div>
                <span class="desc">Liveness health check probe</span>
            </div>
        </div>
    </div>
</body>
</html>"""


@app.post("/v1/ingest")
async def ingest_document(payload: IngestRequest) -> Dict[str, Any]:
    """Ingest a document, chunk it, deduplicate chunks, and update vector/sparse indexes."""
    if not payload.file_path.strip():
        raise HTTPException(status_code=400, detail="File path string cannot be empty.")
        
    try:
        logger.info(f"Processing ingestion request for: {payload.file_path}")
        
        # 1. Extract raw text from file
        document = app.state.parser_router.process_file(payload.file_path)
        
        # 2. Select chunking strategy based on file type
        if document.metadata.file_type == "md":
            logger.info("Using structure-aware Markdown chunking.")
            raw_chunks = ChunkingEngine.structure_aware_markdown_chunk(document)
        else:
            logger.info(f"Using fixed-size sliding window chunking for .{document.metadata.file_type}")
            raw_chunks = ChunkingEngine.fixed_size_chunk(
                document, 
                chunk_size=config.CHUNK_SIZE, 
                chunk_overlap=config.CHUNK_OVERLAP
            )
        
        # 3. Deduplicate semantically redundant chunks using dense embeddings
        embedding_fn = lambda texts: app.state.dense_index.embedding_fn(texts)
        clean_chunks = app.state.deduplicator.deduplicate(raw_chunks, embedding_fn=embedding_fn)
        
        if not clean_chunks:
            logger.info("No unique chunks found after deduplication.")
            return {"status": "success", "message": "No new unique content chunks detected. Index skipped."}

        # 4. Update both sparse (BM25) and dense (ChromaDB) indexes
        app.state.sparse_index.index_chunks(clean_chunks)
        app.state.dense_index.index_chunks(clean_chunks)
        
        logger.info(f"Successfully indexed {len(clean_chunks)} chunks from {payload.file_path}")
        return {
            "status": "success", 
            "chunks_indexed": len(clean_chunks), 
            "source": payload.file_path
        }
    except FileNotFoundError as fnf_err:
        logger.error(f"Ingestion file target not found: {fnf_err}")
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as e:
        logger.error(f"Ingestion pipeline failure: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failure: {str(e)}")


@app.post("/v1/ask")
async def process_query(payload: QueryRequest) -> Dict[str, Any]:
    """Retrieve relevant context chunks via hybrid search, rerank, and generate grounded answer."""
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Query question string cannot be empty.")
        
    try:
        logger.info(f"Processing query: '{payload.question[:60]}...'")
        
        # 1. Hybrid retrieval (BM25 + Vector HNSW)
        hybrid_candidates = app.state.hybrid_retriever.retrieve(payload.question, top_k=config.RETRIEVAL_TOP_K)
        
        if not hybrid_candidates:
            logger.info("Hybrid search returned empty candidate set.")
            return {
                "answer": "Documentation index is currently completely empty. Please ingest tracking documents first.",
                "is_context_sufficient": False,
                "verification_matrix": {"is_valid": False, "flagged_issues": ["No context available"]}
            }

        # 2. Neural Cross-Encoder Reranking
        elite_chunks = app.state.reranker.rerank(payload.question, hybrid_candidates, top_n=config.RERANK_TOP_N)

        # 3. Grounded answer generation via Groq LLM
        llm_output = app.state.generator.generate_answer(payload.question, elite_chunks)

        # 4. Verify citations against source chunks
        verification_report = CitationVerifier.verify_citations(llm_output["answer"], elite_chunks)

        logger.info("Query processed and citations verified successfully.")
        return {
            "answer": llm_output["answer"],
            "is_context_sufficient": llm_output["is_context_sufficient"],
            "verification_matrix": verification_report
        }
    except Exception as e:
        logger.error(f"Query resolution pipeline failure: {e}")
        raise HTTPException(status_code=500, detail=f"Query resolution pipeline failure: {str(e)}")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for API status verification."""
    return {"status": "healthy", "engine": "enterprise_hybrid_rag"}