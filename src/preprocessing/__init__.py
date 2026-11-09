"""
Preprocessing Package.

Exports core functions and modules for text cleaning, candidate partitioning,
attribute extraction, and catalogue preprocessing.
"""

from src.preprocessing.text_cleaner import (
    normalise_text,
    ABBREVIATION_MAP,
    UNIT_NORMALISATIONS,
    LEADING_BOILERPLATE_PATTERNS,
    ADMIN_FILLER_PATTERNS,
)
from src.preprocessing.attribute_extractor import extract_material_attributes
from src.preprocessing.partition import (
    partition_corpus,
    validate_partitions,
    TIER_1_KINDS,
    TIER_3_KINDS,
    EXPECTED_COUNTS,
)
from src.preprocessing.pipeline import preprocess_catalogue

__all__ = [
    "normalise_text",
    "extract_material_attributes",
    "partition_corpus",
    "validate_partitions",
    "preprocess_catalogue",
    "ABBREVIATION_MAP",
    "UNIT_NORMALISATIONS",
    "LEADING_BOILERPLATE_PATTERNS",
    "ADMIN_FILLER_PATTERNS",
    "TIER_1_KINDS",
    "TIER_3_KINDS",
    "EXPECTED_COUNTS",
]
