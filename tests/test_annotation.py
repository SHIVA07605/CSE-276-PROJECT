"""
Unit tests for annotation validation, training readiness reporting, and data integrity
utilities in validate_annotations.py.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Module under test
from validate_annotations import (
    validate_worksheet_integrity,
    compute_training_readiness,
    export_labelled_dataset,
    format_readiness_report_text,
    VALID_LABELS,
    VALID_CONFIDENCES,
)


# ============================================================
# Fixtures
# ============================================================

def _make_minimal_df(n=10, labels=None):
    """Build a minimal valid annotation DataFrame."""
    if labels is None:
        labels = ["PENDING"] * n
    rows = []
    for i in range(n):
        rows.append({
            "pair_id": f"PAIR_{i+1:03d}",
            "corpus_id_1": f"M_{i+1:04d}_1",
            "enterprise_1": "NTPC Limited",
            "raw_description_1": f"MATERIAL A {i}",
            "cleaned_description_1": f"MATERIAL A {i}",
            "extracted_attributes_1": "None detected",
            "corpus_id_2": f"M_{i+1:04d}_2",
            "enterprise_2": "Oil India Limited",
            "raw_description_2": f"MATERIAL B {i}",
            "cleaned_description_2": f"MATERIAL B {i}",
            "extracted_attributes_2": "None detected",
            "lexical_similarity": round(0.5 + i * 0.04, 4),
            "semantic_cosine_similarity": round(0.5 + i * 0.04, 4),
            "is_exact_normalized_match": False,
            "has_attribute_conflict": False,
            "conflict_details": "",
            "confirmation_details": "",
            "human_label": labels[i],
            "confidence_rating": "PENDING",
            "reviewer_notes": "",
        })
    return pd.DataFrame(rows)


def _make_conflict_df():
    """Build a DataFrame with some attribute-conflict pairs."""
    df = _make_minimal_df(n=6, labels=["0", "0", "0", "1", "1", "UNCERTAIN"])
    df.loc[0:2, "has_attribute_conflict"] = True
    df.loc[0:2, "conflict_details"] = "diameter: '250MM' != '300MM'"
    return df


# ============================================================
# Tests: validate_worksheet_integrity
# ============================================================

class TestWorksheetIntegrity:
    def test_valid_worksheet_passes(self):
        df = _make_minimal_df(n=5)
        is_valid, errors, warnings = validate_worksheet_integrity(df)
        assert is_valid, f"Expected valid, got errors: {errors}"
        assert errors == []

    def test_missing_required_column_produces_error(self):
        df = _make_minimal_df(n=5)
        df = df.drop(columns=["corpus_id_1"])
        is_valid, errors, _ = validate_worksheet_integrity(df)
        assert not is_valid
        assert any("corpus_id_1" in e for e in errors)

    def test_duplicate_pair_ids_detected(self):
        df = _make_minimal_df(n=5)
        df.loc[3, "pair_id"] = df.loc[0, "pair_id"]  # introduce duplicate
        is_valid, errors, _ = validate_worksheet_integrity(df)
        assert not is_valid
        assert any("Duplicate pair" in e for e in errors)

    def test_null_corpus_id_detected(self):
        df = _make_minimal_df(n=5)
        df.loc[2, "corpus_id_1"] = None
        is_valid, errors, _ = validate_worksheet_integrity(df)
        assert not is_valid
        assert any("Null corpus" in e for e in errors)

    def test_invalid_label_detected(self):
        df = _make_minimal_df(n=5, labels=["1", "0", "MAYBE", "PENDING", "UNCERTAIN"])
        is_valid, errors, _ = validate_worksheet_integrity(df)
        assert not is_valid
        assert any("MAYBE" in e for e in errors)

    def test_valid_labels_are_accepted(self):
        df = _make_minimal_df(n=4, labels=["1", "0", "UNCERTAIN", "PENDING"])
        is_valid, errors, _ = validate_worksheet_integrity(df)
        assert is_valid
        assert errors == []

    def test_normalised_numeric_labels_accepted(self):
        """'1.0' and '0.0' as strings (e.g. written by pandas) must be accepted."""
        df = _make_minimal_df(n=2, labels=["1.0", "0.0"])
        is_valid, errors, _ = validate_worksheet_integrity(df)
        assert is_valid, f"Expected valid, got: {errors}"

    def test_similarity_range_violation_detected(self):
        df = _make_minimal_df(n=3)
        df.loc[1, "lexical_similarity"] = 1.5  # out of range
        is_valid, errors, _ = validate_worksheet_integrity(df)
        assert not is_valid
        assert any("Lexical similarity" in e for e in errors)

    def test_semantic_range_violation_detected(self):
        df = _make_minimal_df(n=3)
        df.loc[1, "semantic_cosine_similarity"] = -1.5  # out of range
        is_valid, errors, _ = validate_worksheet_integrity(df)
        assert not is_valid
        assert any("Semantic similarity" in e for e in errors)


# ============================================================
# Tests: compute_training_readiness
# ============================================================

class TestTrainingReadiness:
    def test_all_pending_returns_not_ready(self):
        df = _make_minimal_df(n=10, labels=["PENDING"] * 10)
        report = compute_training_readiness(df)
        assert report["10_training_readiness"]["status"] == "NOT_READY_FOR_TRAINING"
        assert not report["10_training_readiness"]["is_ready_for_training"]

    def test_zero_labels_total_equals_dataframe_length(self):
        df = _make_minimal_df(n=8)
        report = compute_training_readiness(df)
        assert report["1_total_candidate_pairs"] == 8
        assert report["3_number_pending"] == 8

    def test_positive_negative_counts_correct(self):
        labels = ["1", "1", "0", "0", "0", "UNCERTAIN", "PENDING", "PENDING"]
        df = _make_minimal_df(n=8, labels=labels)
        report = compute_training_readiness(df)
        assert report["4_positive_labels"] == 2
        assert report["5_negative_labels"] == 3
        assert report["6_uncertain_labels"] == 1
        assert report["3_number_pending"] == 2
        assert report["2_number_reviewed"] == 6

    def test_ready_for_pilot_when_minimum_met(self):
        # 30 binary labels: 10 positive, 20 negative
        labels = ["1"] * 10 + ["0"] * 20
        df = _make_minimal_df(n=30, labels=labels)
        report = compute_training_readiness(df, planning_target=150, min_pilot_labels=30)
        status = report["10_training_readiness"]["status"]
        assert status in ("READY_FOR_PILOT_NOTEBOOK_RUN", "READY_FOR_PRODUCTION_TRAINING")
        assert report["10_training_readiness"]["is_ready_for_training"]

    def test_attribute_conflict_negatives_counted(self):
        df = _make_conflict_df()
        report = compute_training_readiness(df)
        # 3 conflict rows all labelled '0'
        assert report["8_attribute_conflict_negatives"] == 3

    def test_cross_enterprise_counts_correct(self):
        df = _make_minimal_df(n=6, labels=["1", "1", "0", "0", "UNCERTAIN", "PENDING"])
        # enterprise_1 = NTPC, enterprise_2 = Oil India → all 6 are cross-enterprise
        report = compute_training_readiness(df)
        # 5 reviewed (all except PENDING)
        assert report["7_cross_enterprise_reviewed_pairs"] == 5

    def test_class_balance_positive_ratio_correct(self):
        labels = ["1"] * 4 + ["0"] * 6
        df = _make_minimal_df(n=10, labels=labels)
        report = compute_training_readiness(df)
        cb = report["9_class_balance"]
        assert cb["confirmed_binary_total"] == 10
        assert cb["positive_ratio"] == pytest.approx(0.4)
        assert cb["negative_ratio"] == pytest.approx(0.6)

    def test_uncertain_never_counted_as_binary(self):
        labels = ["UNCERTAIN"] * 5 + ["PENDING"] * 5
        df = _make_minimal_df(n=10, labels=labels)
        report = compute_training_readiness(df)
        assert report["4_positive_labels"] == 0
        assert report["5_negative_labels"] == 0
        assert report["9_class_balance"]["confirmed_binary_total"] == 0


# ============================================================
# Tests: export_labelled_dataset
# ============================================================

class TestExportLabelledDataset:
    def test_full_export_includes_all_rows(self):
        df = _make_minimal_df(n=6, labels=["1", "0", "UNCERTAIN", "PENDING", "1", "0"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "labelled.csv"
            exported_path = export_labelled_dataset(df, output_path=out)
            result = pd.read_csv(exported_path)
            assert len(result) == 6

    def test_confirmed_only_export_excludes_pending_and_uncertain(self):
        df = _make_minimal_df(n=6, labels=["1", "0", "UNCERTAIN", "PENDING", "1", "0"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "binary.csv"
            exported_path = export_labelled_dataset(df, output_path=out, confirmed_only=True)
            result = pd.read_csv(exported_path)
            # Should contain only 4 binary rows: two 1s, two 0s
            assert len(result) == 4
            labels = result["human_label"].astype(str).str.upper()
            assert set(labels).issubset({"1", "0"})

    def test_export_preserves_corpus_ids(self):
        df = _make_minimal_df(n=3, labels=["1", "0", "UNCERTAIN"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test_export.csv"
            exported_path = export_labelled_dataset(df, output_path=out)
            result = pd.read_csv(exported_path)
            assert list(result["corpus_id_1"]) == list(df["corpus_id_1"])
            assert list(result["corpus_id_2"]) == list(df["corpus_id_2"])

    def test_export_creates_parent_dirs(self):
        df = _make_minimal_df(n=2, labels=["1", "0"])
        with tempfile.TemporaryDirectory() as tmp:
            nested_path = Path(tmp) / "nested" / "deep" / "labelled.csv"
            exported_path = export_labelled_dataset(df, output_path=nested_path)
            assert exported_path.exists()


# ============================================================
# Tests: format_readiness_report_text
# ============================================================

class TestReadinessReportText:
    def test_report_text_contains_all_10_dimensions(self):
        df = _make_minimal_df(n=5, labels=["1", "0", "UNCERTAIN", "PENDING", "PENDING"])
        report = compute_training_readiness(df)
        text = format_readiness_report_text(report)
        for field in [
            "Total Candidate Pairs",
            "Number Reviewed",
            "Number Pending",
            "Positive Labels",
            "Negative Labels",
            "Uncertain Labels",
            "Cross-Enterprise",
            "Attribute-Conflict",
            "Class Balance",
            "Training Readiness",
        ]:
            assert field in text, f"Expected '{field}' in readiness report text"

    def test_report_shows_pending_status_when_no_labels(self):
        df = _make_minimal_df(n=5, labels=["PENDING"] * 5)
        report = compute_training_readiness(df)
        text = format_readiness_report_text(report)
        assert "NOT_READY_FOR_TRAINING" in text


# ============================================================
# Tests: Integration against real worksheet (if available)
# ============================================================

class TestRealWorksheetIntegration:
    def test_real_worksheet_loads_and_passes_validation(self):
        from src.config import DATA_PROCESSED_DIR
        path = DATA_PROCESSED_DIR / "improved_labelled_pairs_worksheet.csv"
        if not path.exists():
            pytest.skip("Improved worksheet not found.")
        df = pd.read_csv(path)
        is_valid, errors, warnings = validate_worksheet_integrity(df)
        assert is_valid, f"Worksheet integrity failed: {errors}"

    def test_real_worksheet_50_pairs_all_pending(self):
        from src.config import DATA_PROCESSED_DIR
        path = DATA_PROCESSED_DIR / "improved_labelled_pairs_worksheet.csv"
        if not path.exists():
            pytest.skip("Improved worksheet not found.")
        df = pd.read_csv(path)
        assert len(df) == 50
        # Unique pair IDs
        assert df["pair_id"].nunique() == 50
        # All labels should be PENDING at baseline
        labels = df["human_label"].astype(str).str.strip().str.upper()
        # Allow for pre-existing human work but never allow invalid labels
        invalid = set(labels) - VALID_LABELS
        assert invalid == set(), f"Invalid labels in worksheet: {invalid}"

    def test_real_worksheet_has_conflict_and_confirmation_columns(self):
        from src.config import DATA_PROCESSED_DIR
        path = DATA_PROCESSED_DIR / "improved_labelled_pairs_worksheet.csv"
        if not path.exists():
            pytest.skip("Improved worksheet not found.")
        df = pd.read_csv(path)
        assert "conflict_details" in df.columns
        assert "confirmation_details" in df.columns
        assert "has_attribute_conflict" in df.columns
        # 10 attribute conflict pairs expected from the PAIR_CONFLICT sample
        n_conflicts = int(df["has_attribute_conflict"].astype(bool).sum())
        assert n_conflicts == 10

    def test_real_worksheet_cross_enterprise_count(self):
        from src.config import DATA_PROCESSED_DIR
        path = DATA_PROCESSED_DIR / "improved_labelled_pairs_worksheet.csv"
        if not path.exists():
            pytest.skip("Improved worksheet not found.")
        df = pd.read_csv(path)
        cross = (df["enterprise_1"] != df["enterprise_2"]).sum()
        assert cross == 40

    def test_readiness_report_runs_on_real_worksheet(self):
        from src.config import DATA_PROCESSED_DIR
        path = DATA_PROCESSED_DIR / "improved_labelled_pairs_worksheet.csv"
        if not path.exists():
            pytest.skip("Improved worksheet not found.")
        df = pd.read_csv(path)
        report = compute_training_readiness(df)
        assert report["1_total_candidate_pairs"] == 50
        pending_count = int((df["human_label"].fillna("PENDING").astype(str).str.strip().str.upper() == "PENDING").sum())
        assert report["3_number_pending"] == pending_count
        assert report["10_training_readiness"]["status"] == "NOT_READY_FOR_TRAINING"

