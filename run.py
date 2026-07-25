import os
import sys
import streamlit.web.cli as stcli

# Add project root directory to Python module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Programmatically invoke Streamlit to launch the UI application
if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "src/ui/app.py"]
    sys.exit(stcli.main())