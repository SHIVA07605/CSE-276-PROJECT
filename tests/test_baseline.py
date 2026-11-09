"""
Tests for Exact Match Baseline.
"""

import pytest
import pandas as pd
from src.baseline import exact_match_predict, evaluate_baseline_on_pairs, find_exact_matches


def test_exact_match_identical():
    desc1 = "Stainless Steel Bolt M10"
    desc2 = "Stainless Steel Bolt M10"
    assert exact_match_predict(desc1, desc2) == 1


def test_exact_match_different_case_normalised():
    desc1 = "stainless steel bolt m10"
    desc2 = "STAINLESS STEEL BOLT M10"
    assert exact_match_predict(desc1, desc2, normalise=True) == 1


def test_exact_match_different_items():
    desc1 = "Stainless Steel Bolt M10"
    desc2 = "Carbon Steel Flange 2 inch"
    assert exact_match_predict(desc1, desc2) == 0


def test_evaluate_baseline_on_pairs_dataframe():
    df = pd.DataFrame([
        {"material_description_1": "Bolt M10", "material_description_2": "Bolt M10"},
        {"material_description_1": "Bolt M10", "material_description_2": "Nut M10"},
    ])
    preds = evaluate_baseline_on_pairs(df)
    assert list(preds) == [1, 0]


def test_find_exact_matches_basic():
    df = pd.DataFrame([
        {"corpus_id": "M1", "organization": "IOCL", "cleaned_description": "AIR COMPRESSOR"},
        {"corpus_id": "M2", "organization": "OIL", "cleaned_description": "AIR COMPRESSOR"},
        {"corpus_id": "M3", "organization": "OIL", "cleaned_description": "AIR COMPRESSOR"},
        {"corpus_id": "M4", "organization": "NTPC", "cleaned_description": "CENTRIFUGAL PUMP"},
    ])
    matches = find_exact_matches(df)

    # 3 records share 'AIR COMPRESSOR' -> 3 * 2 / 2 = 3 pairs
    assert len(matches) == 3
    assert set(matches["matching_description"]) == {"AIR COMPRESSOR"}

    # Exclude self matches: corpus_id_1 != corpus_id_2 for all
    for _, r in matches.iterrows():
        assert r["corpus_id_1"] != r["corpus_id_2"]


def test_find_exact_matches_cross_enterprise_flag():
    df = pd.DataFrame([
        {"corpus_id": "M1", "organization": "IOCL", "cleaned_description": "BEARINGS"},
        {"corpus_id": "M2", "organization": "OIL", "cleaned_description": "BEARINGS"},
        {"corpus_id": "M3", "organization": "OIL", "cleaned_description": "BEARINGS"},
    ])
    matches = find_exact_matches(df)

    # M1(IOCL) - M2(OIL): cross
    # M1(IOCL) - M3(OIL): cross
    # M2(OIL) - M3(OIL): intra
    cross = matches[matches["is_cross_enterprise"]]
    intra = matches[~matches["is_cross_enterprise"]]

    assert len(cross) == 2
    assert len(intra) == 1


def test_find_exact_matches_no_duplicates():
    df = pd.DataFrame([
        {"corpus_id": "M1", "organization": "IOCL", "cleaned_description": "VALVE"},
        {"corpus_id": "M2", "organization": "OIL", "cleaned_description": "PUMP"},
    ])
    matches = find_exact_matches(df)
    assert len(matches) == 0
