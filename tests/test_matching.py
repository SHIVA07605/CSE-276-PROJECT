"""
Tests for Candidate Generation, TF-IDF Retrieval, and Attribute Compatibility.
"""

import pytest
import pandas as pd
from src.candidate_generation import generate_tfidf_candidates
from src.feature_extraction import (
    compare_extracted_attributes,
    check_attribute_conflicts,
    compute_lexical_features,
)


def test_compute_lexical_features():
    rapidfuzz = pytest.importorskip("rapidfuzz", reason="rapidfuzz required for lexical matching tests")
    s1 = "STAINLESS STEEL BOLT M10"
    s2 = "SS BOLT M10"
    features = compute_lexical_features(s1, s2)

    assert "levenshtein_ratio" in features
    assert "token_sort_ratio" in features
    assert "jaccard_similarity" in features
    assert 0.0 <= features["token_sort_ratio"] <= 1.0


def test_compare_extracted_attributes_match():
    attr1 = {"metric_thread": "M10", "material_grade": "SS304"}
    attr2 = {"metric_thread": "M10", "material_grade": "SS304"}
    comparison = compare_extracted_attributes(attr1, attr2)

    assert comparison["attr_metric_thread_match"] == 1.0
    assert comparison["attr_material_grade_match"] == 1.0


def test_compare_extracted_attributes_conflict():
    attr1 = {"metric_thread": "M10"}
    attr2 = {"metric_thread": "M12"}
    comparison = compare_extracted_attributes(attr1, attr2)

    assert comparison["attr_metric_thread_match"] == -1.0


def test_attribute_conflicts_ss304_vs_ss316():
    """Verify SS304 vs SS316 is explicitly flagged as a conflict."""
    attr1 = {"material_grade": "SS304", "diameter": "50MM"}
    attr2 = {"material_grade": "SS316", "diameter": "50MM"}
    audit = check_attribute_conflicts(attr1, attr2)

    assert audit["has_conflict"] is True
    assert any("material_grade" in c for c in audit["conflicts"])
    assert any("diameter" in c for c in audit["confirmations"])


def test_attribute_conflicts_pressure_rating():
    """Verify 150# vs 300# is flagged as a conflict."""
    attr1 = {"pressure_class": "150#"}
    attr2 = {"pressure_class": "300#"}
    audit = check_attribute_conflicts(attr1, attr2)

    assert audit["has_conflict"] is True
    assert any("pressure_class" in c for c in audit["conflicts"])


def test_attribute_conflicts_schedule():
    """Verify SCH 40 vs SCH 80 is flagged as a conflict."""
    attr1 = {"schedule": "SCH 40"}
    attr2 = {"schedule": "SCH 80"}
    audit = check_attribute_conflicts(attr1, attr2)

    assert audit["has_conflict"] is True
    assert any("schedule" in c for c in audit["conflicts"])


def test_attribute_missing_is_not_conflict():
    """Missing attributes should be encoded as 0.0, NOT as conflict (-1.0)."""
    attr1 = {"material_grade": "SS304", "diameter": "50MM"}
    attr2 = {"material_grade": None, "diameter": "50MM"}
    audit = check_attribute_conflicts(attr1, attr2)

    assert audit["has_conflict"] is False
    assert audit["comparison_scores"]["attr_material_grade_match"] == 0.0


def test_generate_tfidf_candidates_structure():
    df = pd.DataFrame([
        {"corpus_id": "M01", "organization": "EntA", "cleaned_description": "STAINLESS STEEL BOLT M10"},
        {"corpus_id": "M02", "organization": "EntB", "cleaned_description": "STAINLESS STEEL FASTENER M10"},
        {"corpus_id": "M03", "organization": "EntC", "cleaned_description": "COPPER WIRE 2.5 SQMM"},
    ])
    candidates = generate_tfidf_candidates(df, min_similarity=0.1)

    assert isinstance(candidates, pd.DataFrame)
    assert "corpus_id_1" in candidates.columns
    assert "corpus_id_2" in candidates.columns
    assert "tfidf_similarity" in candidates.columns
    assert "is_cross_enterprise" in candidates.columns


def test_generate_tfidf_candidates_no_self_matches():
    df = pd.DataFrame([
        {"corpus_id": "M01", "organization": "EntA", "cleaned_description": "CENTRIFUGAL WATER PUMP"},
        {"corpus_id": "M02", "organization": "EntB", "cleaned_description": "CENTRIFUGAL WATER PUMP"},
    ])
    candidates = generate_tfidf_candidates(df, min_similarity=0.1)

    for _, row in candidates.iterrows():
        assert row["corpus_id_1"] != row["corpus_id_2"]


def test_generate_tfidf_candidates_no_duplicate_pair_directions():
    """Ensure if (A, B) is returned, (B, A) is not also returned."""
    df = pd.DataFrame([
        {"corpus_id": "M01", "organization": "EntA", "cleaned_description": "CENTRIFUGAL WATER PUMP"},
        {"corpus_id": "M02", "organization": "EntB", "cleaned_description": "CENTRIFUGAL WATER PUMP HIGH HEAD"},
    ])
    candidates = generate_tfidf_candidates(df, min_similarity=0.1)
    pair_sets = [tuple(sorted([r["corpus_id_1"], r["corpus_id_2"]])) for _, r in candidates.iterrows()]
    assert len(pair_sets) == len(set(pair_sets))


def test_generate_tfidf_candidates_cross_enterprise_filtering():
    df = pd.DataFrame([
        {"corpus_id": "M01", "organization": "IOCL", "cleaned_description": "AIR COMPRESSOR 100 PSI"},
        {"corpus_id": "M02", "organization": "IOCL", "cleaned_description": "AIR COMPRESSOR 150 PSI"},
        {"corpus_id": "M03", "organization": "OIL",  "cleaned_description": "AIR COMPRESSOR 200 PSI"},
    ])
    candidates_all = generate_tfidf_candidates(df, min_similarity=0.1, cross_enterprise_only=False)
    candidates_cross = generate_tfidf_candidates(df, min_similarity=0.1, cross_enterprise_only=True)

    assert len(candidates_all) >= len(candidates_cross)
    assert all(candidates_cross["is_cross_enterprise"])


def test_generate_tfidf_candidates_missing_commodity_hints_fallback():
    """Verify records with missing item_type_hint are preserved via fallback strategy."""
    df = pd.DataFrame([
        {"corpus_id": "M01", "organization": "IOCL", "cleaned_description": "GATE VALVE 2INCH", "item_type_hint": "valve"},
        {"corpus_id": "M02", "organization": "OIL",  "cleaned_description": "GATE VALVE 2INCH", "item_type_hint": None},
        {"corpus_id": "M03", "organization": "NTPC", "cleaned_description": "GATE VALVE 2INCH", "item_type_hint": "valve"},
    ])
    candidates = generate_tfidf_candidates(df, min_similarity=0.3, block_by_hint=True)

    # M02 (untagged) should pair with M01 and M03 through the fallback mechanism
    pairs_with_m02 = candidates[(candidates["corpus_id_1"] == "M02") | (candidates["corpus_id_2"] == "M02")]
    assert len(pairs_with_m02) > 0, "Untagged record was improperly discarded!"
