"""
Global Configuration and Path Management.

Centralizes paths, project constants, schema column names, and model hyperparameters
to maintain reproducibility across Antigravity, GitHub, and Colab workflows.
"""

from pathlib import Path
import os

# Project root directory (resolved dynamically relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_LABELLED_DIR = DATA_DIR / "labelled"

# Models and serialization paths
MODELS_DIR = PROJECT_ROOT / "models"

# Results and experiment tracking paths
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# Reproducibility seed
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# Default embedding model identifier
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Standard Data Schema Definitions
COL_ENTERPRISE = "enterprise_name"
COL_MATERIAL_CODE = "material_code"
COL_DESCRIPTION = "material_description"
COL_CLEAN_DESC = "cleaned_description"
COL_CATEGORY = "category"
COL_UOM = "uom"
COL_EXTRACTED_ATTRIBUTES = "extracted_attributes"

# Standard Labelled Pairs Schema Definitions
COL_RECORD_ID_1 = "record_id_1"
COL_RECORD_ID_2 = "record_id_2"
COL_LABEL = "label"  # 1 for match, 0 for non-match
COL_CLUSTER_ID = "cluster_id"

# Core required columns for any ingested enterprise catalogue
REQUIRED_CATALOGUE_COLUMNS = [
    COL_ENTERPRISE,
    COL_MATERIAL_CODE,
    COL_DESCRIPTION,
]
