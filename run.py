import os
import sys
from pathlib import Path
import streamlit.web.cli as stcli

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
BACKEND_DIR = ROOT_DIR / "backend"

for p in (str(ROOT_DIR), str(FRONTEND_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Programmatically invoke Streamlit to launch the frontend UI application
if __name__ == "__main__":
    app_target = str(FRONTEND_DIR / "src" / "app.py")
    sys.argv = ["streamlit", "run", app_target]
    sys.exit(stcli.main())