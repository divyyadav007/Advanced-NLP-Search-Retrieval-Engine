import os
import sys
from pathlib import Path
import streamlit.web.cli as stcli

# Ensure frontend and project root are on sys.path
FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
for p in (str(FRONTEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    app_path = str(FRONTEND_DIR / "src" / "app.py")
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]
    sys.exit(stcli.main())
