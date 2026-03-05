# Run the NotebookPRO application
# This script launches the Streamlit application

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import os
    
    # Set the path to App.py
    app_path = project_root / "App.py"
    
    # Run Streamlit
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost"
    ]
    
    sys.exit(stcli.main())
