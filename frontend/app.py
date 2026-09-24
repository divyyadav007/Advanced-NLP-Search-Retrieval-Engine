"""Streamlit Frontend Application Entrypoint."""
import runpy
from pathlib import Path

# Run the primary application located in src/app.py
target_script = Path(__file__).resolve().parent / "src" / "app.py"
runpy.run_path(str(target_script), run_name="__main__")
