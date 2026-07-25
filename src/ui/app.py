import os
import gc
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Add project root directory to Python path to ensure module imports work reliably
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "").strip().rstrip("/")

st.set_page_config(
    page_title="Enterprise Hybrid-RAG Dashboard", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling (Dark Theme with glassmorphism cards and Inter/Outfit typography)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
        color: #f8fafc !important;
    }
    .main-title { 
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem !important; 
        font-weight: 800 !important; 
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-top: 0.5rem;
    }
    .sub-title { 
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem !important; 
        color: #94a3b8; 
        margin-bottom: 2rem; 
        font-weight: 400;
    }
    .section-header { 
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem !important; 
        font-weight: 700 !important; 
        color: #f1f5f9; 
        border-bottom: 2px solid #334155; 
        padding-bottom: 0.6rem; 
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #4338ca 100%) !important;
        transform: translateY(-1px);
    }
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Enterprise Hybrid RAG Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production-Grade Knowledge Synthesis & Dual-Index Retrieval Architecture</div>', unsafe_allow_html=True)

# Sidebar - Telemetry & Configuration Status
with st.sidebar:
    st.markdown("### 🖥️ Engine Status")
    if BACKEND_API_URL:
        st.success("🌐 **Microservice Mode**")
        st.markdown(f"- **API Node**: `{BACKEND_API_URL}`")
        st.markdown("- **Resource Footprint**: Minimal (API-Delegated)")
    else:
        st.info("🔌 **Standalone Mode**")
        st.markdown("- **Execution**: Local Process")
        st.markdown("- **Models**: Loaded in local memory")
    
    st.markdown("---")
    st.markdown("### ⚙️ Engine Configurations")
    st.markdown(f"- **Chunk Size**: `{config.CHUNK_SIZE} chars`")
    st.markdown(f"- **Chunk Overlap**: `{config.CHUNK_OVERLAP} chars`")
    st.markdown(f"- **Retrieval Candidates**: `{config.RETRIEVAL_TOP_K}`")
    st.markdown(f"- **Rerank Candidates**: `{config.RERANK_TOP_N}`")

# Pipeline Initialization
if "pipeline" not in st.session_state:
    if BACKEND_API_URL:
        st.session_state.pipeline_mode = "microservice"
        st.session_state.pipeline = True
    else:
        st.session_state.pipeline_mode = "standalone"
        with st.spinner("Initializing search indexes and loading models..."):
            config.validate_environment()
            
            sparse_idx = SparseBM25Index()
            sparse_idx.load_index()
            dense_idx = DenseVectorIndex()
            
            st.session_state.retriever = HybridRetriever(sparse_idx, dense_idx)
            st.session_state.reranker = DocumentReranker()
            st.session_state.generator = GroundedGenerator()
            st.session_state.parser = DocumentParserRouter()
            st.session_state.deduplicator = ChunkDeduplicator()
            
            # Pre-load embedding and reranker transformer models to ensure a warm start
            try:
                st.session_state.retriever.dense_index.embedding_fn(["warmup text sequence"])
                from src.ingestion.schemas import Chunk, ChunkMetadata
                warmup_meta = ChunkMetadata(source_path="warmup.txt", file_type="txt", chunk_index=0, parent_document_id="warmup")
                warmup_chunk = Chunk(id="warmup", page_content="warmup text sequence", metadata=warmup_meta)
                st.session_state.reranker.rerank("warmup", [{"chunk": warmup_chunk}], top_n=1)
                print("✅ All transformer models loaded and ready.")
            except Exception as warmup_err:
                print(f"⚠️ Pre-loading notice: {warmup_err}")
                
            st.session_state.pipeline = True

# Main Layout: 2 Columns (Query Interface & File Ingestion)
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">🔍 Query Interface</div>', unsafe_allow_html=True)
    st.write("")
    
    user_query = st.text_input(
        "Enter your inquiry:", 
        placeholder="e.g., What are the rules regarding campus Wi-Fi network utilization?",
        label_visibility="visible"
    )
    
    if st.button("Execute Intelligence Query", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Query error: Input cannot be empty.")
        else:
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    if st.session_state.pipeline_mode == "microservice":
                        response = requests.post(
                            f"{BACKEND_API_URL}/v1/ask",
                            json={"question": user_query},
                            timeout=60
                        )
                        if response.status_code == 200:
                            payload = response.json()
                            st.markdown("### 🤖 Synthesized Knowledge Output")
                            st.success(payload["answer"])
                            
                            st.markdown("### 🛡️ Citation Trace Integrity Diagnostics")
                            v_matrix = payload["verification_matrix"]
                            if v_matrix.get("is_valid", False):
                                st.info("✅ Verification Complete: All assertions map to document source chunks.")
                            else:
                                st.error("⚠️ Verification Warning: Claims failed index validation.")
                                if v_matrix.get("flagged_issues"):
                                    st.json(v_matrix["flagged_issues"])
                        else:
                            st.error(f"Backend API Error ({response.status_code}): {response.text}")
                    else:
                        hybrid_candidates = st.session_state.retriever.retrieve(user_query, top_k=config.RETRIEVAL_TOP_K)
                        
                        if not hybrid_candidates:
                            st.info("System Notice: Index is currently empty. Please upload documents first.")
                        else:
                            reranked = st.session_state.reranker.rerank(user_query, hybrid_candidates, top_n=config.RERANK_TOP_N)
                            payload = st.session_state.generator.generate_answer(user_query, reranked)
                            
                            st.markdown("### 🤖 Synthesized Knowledge Output")
                            st.success(payload["answer"])
                            
                            v_matrix = CitationVerifier.verify_citations(payload["answer"], reranked)
                            
                            st.markdown("### 🛡️ Citation Trace Integrity Diagnostics")
                            if v_matrix.get("is_valid", False):
                                st.info("✅ Verification Complete: All assertions map to document source chunks.")
                            else:
                                st.error("⚠️ Verification Warning: Claims failed index validation.")
                                if v_matrix.get("flagged_issues"):
                                    st.json(v_matrix["flagged_issues"])
                                    
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")

with col2:
    st.markdown('<div class="section-header">📂 Ingestion Control Panel</div>', unsafe_allow_html=True)
    st.write("")
    
    uploaded_file = st.file_uploader(
        "Ingest Knowledge Base Asset:", 
        type=["txt", "md", "pdf", "html", "htm"],
        help="Supported formats: PDF, Markdown, TXT, HTML"
    )
    
    if st.button("Trigger Asset Ingestion Pipeline", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please select a file first.")
        else:
            with st.status("Ingesting document...", expanded=True) as status_box:
                try:
                    temp_dir = Path(config.DATA_DIR) / "uploaded_files"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file_path = temp_dir / uploaded_file.name

                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if st.session_state.pipeline_mode == "microservice":
                        status_box.write("Uploading to remote microservice API...")
                        response = requests.post(
                            f"{BACKEND_API_URL}/v1/ingest",
                            json={"file_path": str(temp_file_path)},
                            timeout=120
                        )
                        if response.status_code == 200:
                            res_data = response.json()
                            status_box.update(label=f"✅ Asset Indexed: {res_data.get('chunks_indexed', 0)} chunks processed.", state="complete")
                            st.balloons()
                        else:
                            status_box.update(label=f"❌ Ingestion Failed: {response.text}", state="error")
                    else:
                        status_box.write("Parsing document text...")
                        document = st.session_state.parser.process_file(str(temp_file_path))
                        
                        if document.metadata.file_type == "md":
                            status_box.write("Splitting Markdown sections...")
                            raw_chunks = ChunkingEngine.structure_aware_markdown_chunk(document)
                        else:
                            status_box.write(f"Splitting .{document.metadata.file_type} via character window...")
                            raw_chunks = ChunkingEngine.fixed_size_chunk(
                                document, 
                                chunk_size=config.CHUNK_SIZE, 
                                chunk_overlap=config.CHUNK_OVERLAP
                            )
                        
                        def ui_embedding_fn(texts):
                            batch_size = 16
                            all_embeddings = []
                            for i in range(0, len(texts), batch_size):
                                batch_texts = texts[i:i + batch_size]
                                batch_res = st.session_state.retriever.dense_index.embedding_fn(batch_texts)
                                all_embeddings.extend(batch_res)
                            return all_embeddings
                            
                        status_box.write("Deduplicating redundant chunks...")
                        clean_chunks = st.session_state.deduplicator.deduplicate(raw_chunks, embedding_fn=ui_embedding_fn)
                        
                        if not clean_chunks:
                            status_box.update(label="ℹ️ Duplicate content skipped.", state="complete")
                        else:
                            status_box.write(f"Indexing {len(clean_chunks)} chunks into sparse and vector stores...")
                            st.session_state.retriever.sparse_index.index_chunks(clean_chunks)
                            
                            vector_batch_size = 25
                            for j in range(0, len(clean_chunks), vector_batch_size):
                                sub_batch = clean_chunks[j:j + vector_batch_size]
                                st.session_state.retriever.dense_index.index_chunks(sub_batch)
                            
                            status_box.update(label="✅ Ingestion Succeeded!", state="complete")
                            st.balloons()
                        
                except Exception as e:
                    status_box.update(label=f"❌ Ingestion Failed: {e}", state="error")
                finally:
                    if 'uploaded_file' in locals():
                        del uploaded_file
                    gc.collect()

st.markdown('<div class="footer-text">Enterprise Hybrid RAG Engine Node v1.0.0 • Architecture: Cosine HNSW (ChromaDB) + BM25 Lexical Inverted Index</div>', unsafe_allow_html=True)