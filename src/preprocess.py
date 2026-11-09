"""
Text Normalisation, Unit Standardisation, and Attribute Extraction Module.

Maintains backward compatibility with src.preprocess while delegating to the
modular src.preprocessing package.
"""

from src.preprocessing import (
    normalise_text,
    extract_material_attributes,
    partition_corpus,
    validate_partitions,
    preprocess_catalogue,
    ABBREVIATION_MAP,
    UNIT_NORMALISATIONS,
    LEADING_BOILERPLATE_PATTERNS,
    ADMIN_FILLER_PATTERNS,
    TIER_1_KINDS,
    TIER_3_KINDS,
    EXPECTED_COUNTS,
)

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
