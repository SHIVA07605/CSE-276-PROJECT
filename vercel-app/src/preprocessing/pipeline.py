"""
Preprocessing Pipeline Module.

Orchestrates end-to-end text normalisation, attribute extraction, and schema alignment
for raw enterprise material catalogues and candidate pools.
"""

from typing import Optional
import pandas as pd

from src.config import (
    COL_DESCRIPTION,
    COL_CLEAN_DESC,
    COL_EXTRACTED_ATTRIBUTES,
    COL_ENTERPRISE,
    COL_MATERIAL_CODE,
)
from src.preprocessing.text_cleaner import normalise_text
from src.preprocessing.attribute_extractor import extract_material_attributes


def preprocess_catalogue(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    remove_boilerplate: bool = True,
) -> pd.DataFrame:
    """
    Apply full text normalisation and attribute extraction to a material catalogue DataFrame.

    Handles:
    - Flexible text column detection ('material_description' or 'description').
    - Standard text cleaning, abbreviation expansion, and unit normalisation.
    - Regular expression technical attribute extraction.
    - Standard enterprise and material code alignment.
    - Raw description preserved in 'raw_description' column.

    Args:
        df: Input DataFrame.
        text_col: Name of column containing unstructured text. If None, auto-detected.
        remove_boilerplate: Whether to strip irrelevant procurement boilerplate.

    Returns:
        Augmented DataFrame with raw_description, cleaned_description, extracted_attributes,
        and individual attr_* columns.
    """
    df_out = df.copy()

    # Determine source text column
    if text_col is not None and text_col in df_out.columns:
        source_col = text_col
    elif COL_DESCRIPTION in df_out.columns:
        source_col = COL_DESCRIPTION
    elif "description" in df_out.columns:
        source_col = "description"
    else:
        raise ValueError(
            f"No valid description column found in DataFrame. Available columns: {list(df_out.columns)}"
        )

    # Ensure standardized column aliases for downstream modules if missing
    if COL_ENTERPRISE not in df_out.columns and "organization" in df_out.columns:
        df_out[COL_ENTERPRISE] = df_out["organization"]
    if COL_MATERIAL_CODE not in df_out.columns and "corpus_id" in df_out.columns:
        df_out[COL_MATERIAL_CODE] = df_out["corpus_id"]
    if COL_DESCRIPTION not in df_out.columns:
        df_out[COL_DESCRIPTION] = df_out[source_col]

    # Preserve original raw description
    df_out["raw_description"] = df_out[source_col].fillna("").astype(str)

    # Apply text normalisation
    df_out[COL_CLEAN_DESC] = df_out[source_col].apply(
        lambda t: normalise_text(t, remove_boilerplate=remove_boilerplate)
    )

    # Extract structured technical attributes
    extracted_series = df_out[COL_CLEAN_DESC].apply(extract_material_attributes)
    df_out[COL_EXTRACTED_ATTRIBUTES] = extracted_series

    # Flatten individual attribute columns
    df_out["attr_metric_thread"] = extracted_series.apply(lambda d: d.get("metric_thread"))
    df_out["attr_diameter"] = extracted_series.apply(lambda d: d.get("diameter"))
    df_out["attr_material_grade"] = extracted_series.apply(lambda d: d.get("material_grade"))
    df_out["attr_pressure_class"] = extracted_series.apply(lambda d: d.get("pressure_class"))
    df_out["attr_schedule"] = extracted_series.apply(lambda d: d.get("schedule"))
    df_out["attr_voltage"] = extracted_series.apply(lambda d: d.get("voltage"))
    df_out["attr_capacity"] = extracted_series.apply(lambda d: d.get("capacity"))
    df_out["attr_ambiguous_flags"] = extracted_series.apply(
        lambda d: d.get("ambiguous_flags") or []
    )

    return df_out
