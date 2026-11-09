"""
Data Ingestion and Schema Validation Module.

Provides utilities to load, validate, and standardize heterogeneous material
catalogues across public enterprises, as well as labelled evaluation pairs.
"""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

from src.config import (
    REQUIRED_CATALOGUE_COLUMNS,
    COL_ENTERPRISE,
    COL_MATERIAL_CODE,
    COL_DESCRIPTION,
    COL_RECORD_ID_1,
    COL_RECORD_ID_2,
    COL_LABEL,
)


def validate_catalogue_schema(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """
    Validate that an input catalogue DataFrame contains all mandatory columns.

    Args:
        df: Pandas DataFrame representing an enterprise material catalogue.
        required_columns: List of required column names. Defaults to REQUIRED_CATALOGUE_COLUMNS.

    Returns:
        True if all required columns are present.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    if required_columns is None:
        required_columns = REQUIRED_CATALOGUE_COLUMNS

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Catalogue DataFrame is missing mandatory columns: {missing_cols}. "
            f"Expected columns: {required_columns}"
        )
    return True


def load_material_catalogue(
    file_path: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
) -> pd.DataFrame:
    """
    Load an enterprise material catalogue from CSV, Excel, or Parquet format.

    Args:
        file_path: Path to the catalogue file.
        sheet_name: Sheet identifier if loading an Excel workbook.

    Returns:
        A validated pandas DataFrame containing material records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Material catalogue file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .csv, .xlsx, .parquet")

    # Clean whitespace in column names
    df.columns = [str(col).strip() for col in df.columns]

    validate_catalogue_schema(df)
    return df


def load_labelled_pairs(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load ground-truth labelled pairs for training or evaluating entity resolution.

    Args:
        file_path: Path to the labelled pairs CSV or Parquet file.

    Returns:
        DataFrame containing pair comparisons with ground truth labels (0 or 1).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Labelled pairs file not found: {path}")

    df = pd.read_csv(path)
    required_cols = [COL_RECORD_ID_1, COL_RECORD_ID_2, COL_LABEL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Labelled pairs file missing required columns: {missing}")

    return df
