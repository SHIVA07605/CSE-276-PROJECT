"""
Tests for Data Preprocessing, Text Cleaning, Partitioning, and Attribute Extraction.

Coverage:
- Text normalisation (basic, units, boilerplate removal, null handling)
- Abbreviation expansion (SS, CS, MS, HEX, DIA, CI, GI)
- Procurement boilerplate stripping
- Engineering attribute extraction:
    * Diameter (MM, INCH, DN notation)
    * Metric thread (M10, M12)
    * Pressure class (150#, PN16, Class 150)
    * Pipe schedule (SCH 40, SCH 80)
    * Material grade (SS304, SS316, IS2062, ASTM A106)
    * Electrical voltage and capacity
- Edge cases: missing descriptions, ambiguous specs
- Preservation of technical terms (PIPE, FLANGE, SEAMLESS, SE)
- Full corpus partition validation (skipped if CSV absent)
- preprocess_catalogue pipeline
"""

import pytest
import pandas as pd
from pathlib import Path

from src.preprocessing import (
    normalise_text,
    extract_material_attributes,
    partition_corpus,
    validate_partitions,
    preprocess_catalogue,
    EXPECTED_COUNTS,
)
from src.config import DATA_RAW_DIR


# ==============================================================================
# 1. Text Normalisation — basic
# ==============================================================================

def test_normalise_text_basic():
    raw = "  Stainless Steel Bolt M10  "
    expected = "STAINLESS STEEL BOLT M10"
    assert normalise_text(raw) == expected


def test_normalise_text_required_specification_examples():
    """Verify exact examples from Notebook 02 specification."""
    assert normalise_text("10 mm") == "10MM"
    assert normalise_text("2 inch") == "2INCH"
    assert normalise_text("S.S. 304") == "SS304"
    assert normalise_text("Stainless Steel") == "STAINLESS STEEL"


def test_normalise_text_none_handling():
    assert normalise_text(None) == ""
    assert normalise_text("") == ""


def test_normalise_text_whitespace_collapsing():
    raw = "  CARBON   STEEL   PIPE   "
    result = normalise_text(raw)
    assert "  " not in result
    assert result == result.strip()


def test_normalise_text_unicode_quotes_and_dashes():
    """Smart quotes and unicode dashes are normalised."""
    raw = "\u201cstainless steel\u201d \u2013 bolt"
    result = normalise_text(raw)
    assert "\u201c" not in result
    assert "\u2013" not in result


# ==============================================================================
# 2. Abbreviation Expansion
# ==============================================================================

def test_normalise_text_acronym_expansion():
    raw = "SS Hex Bolt, 10mm"
    clean = normalise_text(raw)
    assert "STAINLESS STEEL" in clean
    assert "HEXAGONAL" in clean
    assert "10MM" in clean


def test_normalise_text_cs_expansion():
    clean = normalise_text("CS Flange 150# PN16")
    assert "CARBON STEEL" in clean


def test_normalise_text_ms_expansion():
    clean = normalise_text("MS Plate 10mm Thk")
    assert "MILD STEEL" in clean
    assert "THICKNESS" in clean


def test_normalise_text_ci_expansion():
    clean = normalise_text("CI Gate Valve 2 inch")
    assert "CAST IRON" in clean


def test_normalise_text_gi_expansion():
    clean = normalise_text("GI pipe 1 inch")
    assert "GALVANISED IRON" in clean


def test_normalise_text_dia_expansion():
    clean = normalise_text("Bolt DIA 10mm")
    assert "DIAMETER" in clean


def test_normalise_text_ss_grade_preserved():
    """SS 304 should become SS304 (not STAINLESS STEEL 304)."""
    clean = normalise_text("SS 304 Pipe")
    assert "SS304" in clean
    # 'STAINLESS STEEL304' or 'STAINLESS STEEL 304' would be incorrect
    assert "STAINLESS STEEL304" not in clean


def test_normalise_text_ss316_preserved():
    clean = normalise_text("SS 316L Flange")
    assert "SS316L" in clean or "SS316" in clean


# ==============================================================================
# 3. Procurement Boilerplate Removal
# ==============================================================================

def test_normalise_text_boilerplate_removal():
    raw_tender = "Consolidated Procurement of CI Gate Valves for Farakka & Darlipalli"
    clean = normalise_text(raw_tender)
    assert not clean.startswith("CONSOLIDATED PROCUREMENT OF")
    assert "CAST IRON" in clean
    assert "GATE VALVES" in clean


def test_normalise_text_supply_of_removed():
    raw_supply = "Supply of Seamless Steel Pipe 50mm NB"
    clean = normalise_text(raw_supply)
    assert not clean.startswith("SUPPLY OF")
    assert "SEAMLESS STEEL PIPE" in clean
    assert "50MM NB" in clean


def test_normalise_text_amc_removed():
    raw = "Annual Maintenance Contract for DG Sets at Duliajan"
    clean = normalise_text(raw)
    assert not clean.startswith("ANNUAL MAINTENANCE CONTRACT")


def test_normalise_text_tender_for_removed():
    raw = "Tender for supply of HDPE Pipes 110mm OD"
    clean = normalise_text(raw)
    assert not clean.upper().startswith("TENDER FOR")
    assert "HDPE" in clean.upper() or "PIPES" in clean.upper()


# ==============================================================================
# 4. Preservation of Technical Terms
# ==============================================================================

def test_normalise_text_preserve_pipe():
    result = normalise_text("SEAMLESS PIPE 2INCH")
    assert "PIPE" in result


def test_normalise_text_preserve_flange():
    result = normalise_text("CS FLANGE 150#")
    assert "FLANGE" in result


def test_normalise_text_preserve_se_identifier():
    """SE in Tata SE 1613/42 must NOT be removed (not a procurement boilerplate)."""
    result = normalise_text("TRUCK CHASSIS TATA SE 1613/42")
    assert "SE" in result


def test_normalise_text_preserve_valve():
    result = normalise_text("CS GLOBE VALVE 2 INCH PN16")
    assert "VALVE" in result


def test_normalise_text_preserve_astm():
    result = normalise_text("Seamless Pipe ASTM A106 Grade B 50mm")
    assert "ASTM" in result


def test_normalise_text_preserve_welded():
    result = normalise_text("WELDED PIPE 3INCH SCH 40")
    assert "WELDED" in result


# ==============================================================================
# 5. Diameter Extraction
# ==============================================================================

def test_extract_diameter_mm():
    attrs = extract_material_attributes("SEAMLESS PIPE 50MM NB SS316")
    assert attrs["diameter"] == "50MM NB"


def test_extract_diameter_inch():
    attrs = extract_material_attributes("CS ELBOW 2INCH 150#")
    assert attrs["diameter"] == "2INCH"


def test_extract_diameter_fractional():
    attrs = extract_material_attributes("SS VALVE 1/2INCH PN16")
    assert attrs["diameter"] == "1/2INCH"


def test_extract_diameter_dn_notation():
    attrs = extract_material_attributes("GATE VALVE DN100 PN16")
    assert attrs["diameter"] == "DN100"


def test_extract_diameter_dn_with_space():
    attrs = extract_material_attributes("BUTTERFLY VALVE DN 200 PN40")
    assert attrs["diameter"] == "DN200"


def test_extract_no_diameter_when_absent():
    attrs = extract_material_attributes("STAINLESS STEEL BOLT M10")
    assert attrs["diameter"] is None


# ==============================================================================
# 6. Metric Thread Extraction
# ==============================================================================

def test_extract_thread_m10():
    attrs = extract_material_attributes("HEXAGONAL BOLT M10 SS304")
    assert attrs["metric_thread"] == "M10"


def test_extract_thread_m12():
    attrs = extract_material_attributes("HEX NUT M12 GALVANISED IRON")
    assert attrs["metric_thread"] == "M12"


def test_extract_thread_not_confused_with_diameter():
    """M10 is a thread, not a 10MM diameter."""
    attrs = extract_material_attributes("BOLT M10")
    assert attrs["metric_thread"] == "M10"
    # diameter should be None (no MM/INCH/DN token)
    assert attrs["diameter"] is None


# ==============================================================================
# 7. Pressure Class Extraction
# ==============================================================================

def test_extract_pressure_150hash():
    attrs = extract_material_attributes("CS GATE VALVE 2INCH 150#")
    assert attrs["pressure_class"] == "150#"


def test_extract_pressure_300hash():
    attrs = extract_material_attributes("SS GLOBE VALVE 300#")
    assert attrs["pressure_class"] == "300#"


def test_extract_pressure_pn16():
    attrs = extract_material_attributes("BUTTERFLY VALVE DN100 PN16")
    assert attrs["pressure_class"] == "PN16"


def test_extract_pressure_pn40():
    attrs = extract_material_attributes("GATE VALVE 50MM PN40")
    assert attrs["pressure_class"] == "PN40"


def test_extract_pressure_class_150():
    attrs = extract_material_attributes("FLANGE CLASS 150 2INCH")
    assert attrs["pressure_class"] == "CLASS 150"


def test_extract_no_pressure_when_absent():
    attrs = extract_material_attributes("STAINLESS STEEL BOLT M10 50MM")
    assert attrs["pressure_class"] is None


# ==============================================================================
# 8. Pipe Schedule Extraction
# ==============================================================================

def test_extract_schedule_sch40():
    attrs = extract_material_attributes("SEAMLESS PIPE 2INCH SCH 40")
    assert attrs["schedule"] == "SCH 40"


def test_extract_schedule_sch80():
    attrs = extract_material_attributes("CS PIPE 3INCH SCH 80 ASTM A106")
    assert attrs["schedule"] == "SCH 80"


def test_extract_schedule_sch_xs():
    attrs = extract_material_attributes("SS PIPE 4INCH SCH XS")
    assert attrs["schedule"] == "SCH XS"


def test_schedule_not_confused_with_pressure():
    """SCH 40 is a wall schedule, NOT a pressure class."""
    attrs = extract_material_attributes("PIPE 2INCH SCH 40 SS316")
    assert attrs["schedule"] == "SCH 40"
    assert attrs["pressure_class"] is None


# ==============================================================================
# 9. Material Grade Extraction
# ==============================================================================

def test_extract_grade_ss304():
    attrs = extract_material_attributes("HEX BOLT SS304 M10")
    assert attrs["material_grade"] == "SS304"


def test_extract_grade_ss316():
    attrs = extract_material_attributes("SS316 SEAMLESS PIPE 50MM")
    assert attrs["material_grade"] == "SS316"


def test_extract_grade_ss304_not_ss316():
    """SS304 and SS316 must be distinguished."""
    attrs_304 = extract_material_attributes("PIPE SS304 2INCH")
    attrs_316 = extract_material_attributes("PIPE SS316 2INCH")
    assert attrs_304["material_grade"] == "SS304"
    assert attrs_316["material_grade"] == "SS316"
    assert attrs_304["material_grade"] != attrs_316["material_grade"]


def test_extract_grade_is2062():
    attrs = extract_material_attributes("MS PLATE IS2062 10MM THK")
    assert attrs["material_grade"] == "IS2062"


def test_extract_grade_astm_a106():
    attrs = extract_material_attributes("SEAMLESS PIPE ASTM A106 GRADE B 3INCH SCH 80")
    assert attrs["material_grade"] is not None
    assert "ASTM" in attrs["material_grade"] or "A106" in attrs["material_grade"]


def test_extract_grade_carbon_steel():
    attrs = extract_material_attributes("CARBON STEEL FLANGE 150# 2INCH")
    assert attrs["material_grade"] == "CARBONSTEEL"


def test_extract_no_grade_when_absent():
    attrs = extract_material_attributes("GATE VALVE 2INCH 150#")
    assert attrs["material_grade"] is None


# ==============================================================================
# 10. Missing / Empty Description Edge Cases
# ==============================================================================

def test_extract_material_attributes_empty():
    attrs = extract_material_attributes("")
    assert attrs["metric_thread"] is None
    assert attrs["diameter"] is None
    assert attrs["material_grade"] is None
    assert attrs["pressure_class"] is None
    assert attrs["schedule"] is None


def test_extract_material_attributes_none():
    attrs = extract_material_attributes(None)
    assert attrs["metric_thread"] is None
    assert attrs["diameter"] is None


def test_extract_material_attributes_whitespace_only():
    attrs = extract_material_attributes("   ")
    assert attrs["metric_thread"] is None


def test_normalise_text_numeric_only():
    """A description that is just a number should not raise."""
    result = normalise_text("12345")
    assert isinstance(result, str)


# ==============================================================================
# 11. Ambiguous Specification Flagging
# ==============================================================================

def test_ambiguous_flag_for_unrecognised_hash():
    """A hash-number not matching a known class should be flagged."""
    attrs = extract_material_attributes("FITTING 75# RATING UNKNOWN")
    # 75# is not a standard class, should appear in ambiguous_flags
    assert attrs["ambiguous_flags"] is not None


# ==============================================================================
# 12. Corpus Partition Validation (integration — skipped if CSV absent)
# ==============================================================================

def test_corpus_partition_reconstruction_and_validation():
    corpus_csv = DATA_RAW_DIR / "material_description_corpus.csv"
    if not corpus_csv.exists():
        pytest.skip(f"Corpus file not found: {corpus_csv}")

    df = pd.read_csv(corpus_csv)
    assert len(df) == 21513

    partitions = partition_corpus(df)
    validation = validate_partitions(df, partitions, strict=True)

    assert validation["valid"] is True
    assert validation["all_counts_match"] is True
    assert validation["has_unique_ids"] is True
    assert validation["is_mutually_exclusive"] is True
    assert validation["is_collectively_exhaustive"] is True

    assert len(partitions["tier1_discrete"]) == EXPECTED_COUNTS["tier1_discrete"]       # 923
    assert len(partitions["tier2_tagged"]) == EXPECTED_COUNTS["tier2_tagged"]           # 5711
    assert len(partitions["tier2_untagged"]) == EXPECTED_COUNTS["tier2_untagged"]       # 14542
    assert len(partitions["tier3_administrative"]) == EXPECTED_COUNTS["tier3_administrative"]  # 337
    assert len(partitions["candidate_pool"]) == EXPECTED_COUNTS["candidate_pool"]       # 6634
    assert len(partitions["remaining_pool"]) == EXPECTED_COUNTS["remaining_pool"]       # 14879


# ==============================================================================
# 13. preprocess_catalogue pipeline
# ==============================================================================

def test_preprocess_catalogue_pipeline():
    sample_df = pd.DataFrame([
        {
            "corpus_id": "TEST_001",
            "organization": "NTPC Limited",
            "description": "Supply of S.S. 304 Hex Bolt 10 mm",
        },
        {
            "corpus_id": "TEST_002",
            "organization": "Oil India Limited",
            "description": "Procurement of Seamless Pipe 2 inch Sch 40",
        },
    ])

    processed = preprocess_catalogue(sample_df)

    # Required column presence
    assert "raw_description" in processed.columns
    assert "cleaned_description" in processed.columns
    assert "extracted_attributes" in processed.columns
    assert "attr_material_grade" in processed.columns
    assert "attr_diameter" in processed.columns
    assert "attr_schedule" in processed.columns

    # Raw description preserved unchanged
    assert processed.iloc[0]["raw_description"] == "Supply of S.S. 304 Hex Bolt 10 mm"

    # Check first record
    row0 = processed.iloc[0]
    assert "SS304" in row0["cleaned_description"]
    assert "10MM" in row0["cleaned_description"]
    assert row0["attr_material_grade"] == "SS304"
    assert row0["attr_diameter"] == "10MM"

    # Check second record
    row1 = processed.iloc[1]
    assert "PIPE" in row1["cleaned_description"]
    assert "2INCH" in row1["cleaned_description"]
    assert row1["attr_diameter"] == "2INCH"
    assert row1["attr_schedule"] == "SCH 40"
