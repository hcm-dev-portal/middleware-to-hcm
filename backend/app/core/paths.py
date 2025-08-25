# backend/app/core/paths.py
# Alternative: Use root app/data directory
from pathlib import Path

# This file lives at backend/app/core/paths.py
# Go up to the project root and use app/data there

CORE_DIR = Path(__file__).resolve().parent           # .../backend/app/core
APP_DIR = CORE_DIR.parent                            # .../backend/app  
BACKEND_DIR = APP_DIR.parent                         # .../backend
PROJECT_ROOT = BACKEND_DIR.parent                    # .../project-root
DATA_DIR = PROJECT_ROOT / "backend" / "app" / "data"             # .../project-root/app/data

# Ensure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define the file paths
METADATA_PATH = DATA_DIR / "schema_metadata.json"
INDEX_PATH = DATA_DIR / "schema_index.json"

# Debug: Print paths when imported
import logging
logger = logging.getLogger(__name__)
logger.info(f"Paths configured - DATA_DIR: {DATA_DIR}, exists: {DATA_DIR.exists()}")
logger.info(f"METADATA_PATH: {METADATA_PATH}, exists: {METADATA_PATH.exists()}")
logger.info(f"INDEX_PATH: {INDEX_PATH}, exists: {INDEX_PATH.exists()}")