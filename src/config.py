import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    """Centralized configuration for system paths, model names, and processing parameters."""
    
    # Base Directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_STORAGE_DIR = str(DATA_DIR / "chroma_db")
    SPARSE_INDEX_PATH = str(DATA_DIR / "sparse_index.pkl")
    
    # Models
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    GENERATION_MODEL_NAME = "llama-3.1-8b-instant"
    JUDGE_MODEL_NAME = "llama-3.3-70b-versatile"
    
    # Text Processing Hyperparameters
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    RETRIEVAL_TOP_K = 10
    RERANK_TOP_N = 5
    
    # Security Credentials
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    @classmethod
    def validate_environment(cls):
        """Validate required environment variables and ensure target data directories exist."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required but not set.")
        os.makedirs(cls.DATA_DIR, exist_ok=True)


config = AppConfig()