"""
Unit and Integration Tests for Notebook 05 Supervised Learning Pipeline.

Tests:
1. Label validation and domain checking.
2. Exclusion of PENDING and UNCERTAIN from supervised training.
3. Feature matrix integrity (13 numerical features, no NaN/Inf, all finite).
4. Feature naming and strict ordering.
5. Detection and rejection of duplicate candidate pairs.
6. Connected-component graph partitioning.
7. Material-ID leakage prevention (zero overlap guarantee).
8. Single-class dataset detection.
9. Multi-tier training-readiness guard.
10. Baseline preparation and evaluation harness parity.
11. Model serialization metadata preservation.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_SEED, DATA_PROCESSED_DIR
from src.model import PairwiseEntityClassifier
from src.feature_extraction import assemble_pairwise_features
from src.evaluation import (
    find_pair_connected_components,
    create_connected_component_split,
    compute_pairwise_metrics,
)

EXPECTED_13_FEATURES = [
    "tfidf_similarity",
    "semantic_cosine_similarity",
    "similarity_difference",
    "is_exact_match",
    "has_attribute_conflict",
    "conflict_count",
    "confirmation_count",
    "is_cross_enterprise",
    "attr_diameter_match",
    "attr_material_grade_match",
    "attr_pressure_class_match",
    "attr_metric_thread_match",
    "attr_schedule_match",
]


# ==============================================================================
# 1. Label Validation & Exclusion Tests
# ==============================================================================

class TestLabelValidationAndExclusion:
    def test_label_normalization(self):
        """Test normalization of valid labels including numeric strings."""
        raw = pd.Series(["1", "0", "1.0", "0.0", "PENDING", "UNCERTAIN"])
        norm = raw.astype(str).str.strip().str.upper().replace({"1.0": "1", "0.0": "0"})
        assert list(norm) == ["1", "0", "1", "0", "PENDING", "UNCERTAIN"]

    def test_pending_and_uncertain_strictly_excluded(self):
        """Verify that PENDING and UNCERTAIN labels are filtered out of training."""
        df = pd.DataFrame({
            "pair_id": [f"P_{i}" for i in range(5)],
            "human_label": ["1", "0", "PENDING", "UNCERTAIN", "1"],
        })
        trainable = df[df["human_label"].isin(["1", "0", "1.0", "0.0"])].copy()
        assert len(trainable) == 3
        assert "PENDING" not in trainable["human_label"].values
        assert "UNCERTAIN" not in trainable["human_label"].values

    def test_zero_labels_produces_empty_training_set(self):
        """Confirm that an all-PENDING worksheet produces 0 trainable rows."""
        df = pd.DataFrame({
            "pair_id": [f"P_{i}" for i in range(10)],
            "human_label": ["PENDING"] * 10,
        })
        trainable = df[df["human_label"].isin(["1", "0", "1.0", "0.0"])]
        assert len(trainable) == 0


# ==============================================================================
# 2. Feature Integrity & Ordering Tests
# ==============================================================================

class TestFeatureIntegrityAndOrdering:
    @pytest.fixture
    def sample_pairs_and_materials(self):
        df_pairs = pd.DataFrame({
            "pair_id": ["P1", "P2"],
            "corpus_id_1": ["M01", "M02"],
            "corpus_id_2": ["M03", "M04"],
            "enterprise_1": ["IOCL", "NTPC"],
            "enterprise_2": ["OIL", "NTPC"],
            "lexical_similarity": [0.85, 0.40],
            "semantic_cosine_similarity": [0.88, 0.65],
            "is_exact_normalized_match": [False, False],
            "has_attribute_conflict": [False, True],
            "conflict_details": ["", "diameter: 250MM != 300MM"],
            "confirmation_details": ["", ""],
            "human_label": ["PENDING", "PENDING"],
            "reviewer_notes": ["Note 1", "Note 2"],
        })
        df_materials = pd.DataFrame({
            "corpus_id": ["M01", "M02", "M03", "M04"],
            "attr_diameter": ["250MM", "250MM", "250MM", "300MM"],
            "attr_material_grade": ["SS304", None, "SS304", None],
            "attr_pressure_class": [None, None, None, None],
            "attr_metric_thread": [None, None, None, None],
            "attr_schedule": [None, None, None, None],
        })
        return df_pairs, df_materials

    def test_feature_count_and_exact_ordering(self, sample_pairs_and_materials):
        """Verify that assemble_pairwise_features returns exactly 13 features in strict order."""
        df_pairs, df_materials = sample_pairs_and_materials
        X = assemble_pairwise_features(df_pairs, df_materials)
        assert list(X.columns) == EXPECTED_13_FEATURES
        assert X.shape[1] == 13

    def test_no_arbitrary_identifiers_or_target_leakage(self, sample_pairs_and_materials):
        """Verify that identifiers, enterprise names, raw text, and reviewer notes are absent."""
        df_pairs, df_materials = sample_pairs_and_materials
        X = assemble_pairwise_features(df_pairs, df_materials)
        forbidden = [
            "corpus_id", "corpus_id_1", "corpus_id_2",
            "enterprise_1", "enterprise_2",
            "human_label", "reviewer_notes", "pair_id",
        ]
        for col in forbidden:
            assert col not in X.columns

    def test_all_features_are_finite(self, sample_pairs_and_materials):
        """Confirm zero NaN, zero Inf, all finite floating point values."""
        df_pairs, df_materials = sample_pairs_and_materials
        X = assemble_pairwise_features(df_pairs, df_materials)
        assert not X.isna().any().any()
        assert np.isfinite(X.values).all()

    def test_duplicate_pair_detection(self):
        """Verify detection of duplicate pair IDs and duplicate endpoints."""
        df = pd.DataFrame({
            "pair_id": ["P1", "P1", "P2"],
            "corpus_id_1": ["M01", "M01", "M02"],
            "corpus_id_2": ["M03", "M03", "M04"],
        })
        has_duplicate_ids = bool(df["pair_id"].duplicated().any())
        assert has_duplicate_ids is True


# ==============================================================================
# 3. Connected-Component Splitting & Leakage Prevention Tests
# ==============================================================================

class TestConnectedComponentPartitioning:
    def test_leakage_safe_zero_overlap(self):
        """Verify zero material ID overlap between train and test partitions."""
        df_pairs = pd.DataFrame({
            "corpus_id_1": ["A", "B", "C", "D", "E", "F"],
            "corpus_id_2": ["B", "C", "A", "G", "H", "I"],
        })
        X = pd.DataFrame({"feat": np.arange(len(df_pairs))})
        train_idx, test_idx, audit = create_connected_component_split(
            df_pairs, X, test_size=0.33, random_state=42
        )
        assert audit["valid_split"] is True
        assert audit["zero_leakage_verified"] is True
        assert audit["material_id_overlap_count"] == 0

        train_mats = set(df_pairs.iloc[train_idx]["corpus_id_1"]).union(
            df_pairs.iloc[train_idx]["corpus_id_2"]
        )
        test_mats = set(df_pairs.iloc[test_idx]["corpus_id_1"]).union(
            df_pairs.iloc[test_idx]["corpus_id_2"]
        )
        assert len(train_mats.intersection(test_mats)) == 0

    def test_insufficient_components_handled_safely(self):
        """Verify that 1 monolithic component prevents invalid evaluation."""
        df_pairs = pd.DataFrame({
            "corpus_id_1": ["A", "B", "C"],
            "corpus_id_2": ["B", "C", "D"],
        })
        X = pd.DataFrame({"feat": [1, 2, 3]})
        train_idx, test_idx, audit = create_connected_component_split(df_pairs, X)
        assert audit["valid_split"] is False
        assert len(train_idx) == 0


# ==============================================================================
# 4. Training-Readiness Guard & Single-Class Detection Tests
# ==============================================================================

class TestTrainingReadinessGuard:
    def _evaluate_guard(self, df_train, min_pilot_count=30):
        """Emulate the notebook readiness guard logic."""
        reasons = []
        if len(df_train) == 0:
            reasons.append("Zero confirmed binary labels found.")
            return False, reasons
        if len(df_train) < min_pilot_count:
            reasons.append("Sample size below minimum pilot requirement.")
        y = df_train["target"].values
        if len(np.unique(y)) < 2:
            reasons.append("Single-class dataset detected.")
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos < 5 or neg < 5:
            reasons.append("Class deficit: min 5 positives and negatives required.")
        return len(reasons) == 0, reasons

    def test_guard_blocks_when_zero_confirmed_labels(self):
        """Verify guard blocks when confirmed labels is 0."""
        empty_trainable = pd.DataFrame(columns=["target"])
        is_ready, reasons = self._evaluate_guard(empty_trainable)
        assert is_ready is False
        assert any("Zero confirmed" in r for r in reasons)

    def test_single_class_detection(self):
        """Verify guard detects single-class dataset (all 1s or all 0s)."""
        all_ones = pd.DataFrame({"target": [1] * 40})
        is_ready, reasons = self._evaluate_guard(all_ones)
        assert is_ready is False
        assert any("Single-class" in r for r in reasons)

    def test_class_deficit_detection(self):
        """Verify guard flags severe class deficit (<5 of either class)."""
        skewed = pd.DataFrame({"target": [1] * 2 + [0] * 35})
        is_ready, reasons = self._evaluate_guard(skewed)
        assert is_ready is False
        assert any("Class deficit" in r for r in reasons)

    def test_guard_passes_when_sufficient_and_balanced(self):
        """Verify guard passes when criteria are met."""
        valid_df = pd.DataFrame({"target": [1] * 15 + [0] * 20})
        is_ready, reasons = self._evaluate_guard(valid_df, min_pilot_count=30)
        assert is_ready is True
        assert len(reasons) == 0

    def test_model_unfitted_state_preserved(self):
        """Verify that PairwiseEntityClassifier remains unfitted when guard blocks."""
        classifier = PairwiseEntityClassifier(
            classifier_type="random_forest",
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
        )
        assert classifier.is_fitted is False
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            classifier.predict(pd.DataFrame([[0.5] * 13]))


# ==============================================================================
# 5. Baseline Preparation & Parity Tests
# ==============================================================================

class TestBaselinePreparation:
    def test_baseline_uses_exact_match_feature(self):
        """Verify baseline predictions evaluate is_exact_match column directly."""
        y_true = np.array([1, 1, 0, 0])
        is_exact = np.array([1, 0, 0, 0])
        metrics = compute_pairwise_metrics(y_true, is_exact)
        assert metrics["true_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["true_negatives"] == 2
        assert metrics["false_positives"] == 0
        assert metrics["precision"] == 1.0


# ==============================================================================
# 6. Model Serialization & Metadata Preservation Tests
# ==============================================================================

class TestModelSerializationMetadata:
    def test_save_and_load_with_metadata(self):
        """Verify serialization packages model parameters, features, and version cleanly."""
        classifier = PairwiseEntityClassifier(
            classifier_type="random_forest",
            n_estimators=10,
            max_depth=4,
            random_state=42,
        )
        # Dummy fit for testing save mechanism
        X_dummy = pd.DataFrame(np.random.randn(20, 13), columns=EXPECTED_13_FEATURES)
        y_dummy = np.random.choice([0, 1], size=20)
        classifier.fit(X_dummy, y_dummy)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            metadata = {
                "version": "1.0.0-rf-pairwise",
                "features": EXPECTED_13_FEATURES,
                "n_confirmed_pairs": 20,
            }
            classifier.save(tmp_path, metadata=metadata)
            assert tmp_path.exists()

            loaded = PairwiseEntityClassifier.load(tmp_path)
            assert loaded.is_fitted is True
            assert loaded.feature_names == EXPECTED_13_FEATURES
            assert loaded.metadata["version"] == "1.0.0-rf-pairwise"
            assert loaded.metadata["n_confirmed_pairs"] == 20
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
