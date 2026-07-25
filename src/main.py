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


# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.get("/")
async def root_index():
    """Return API metadata and health status."""
    return {
        "title": app.title,
        "description": app.description,
        "version": app.version,
        "status": "healthy",
        "docs_url": "/docs"
    }


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