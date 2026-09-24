import os
import sys
from pathlib import Path
import uvicorn

# Ensure backend root is in Python module search path
BACKEND_DIR = str(Path(__file__).resolve().parent)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() in ("true", "1", "yes")
    print(f"Starting Hybrid RAG FastAPI server on http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
