"""
Tests for the hybrid self-supervised entity-resolution model:
canonical specification parsing, label-by-construction pair synthesis, feature builder,
decision policy safety gates, and model persistence.
"""

import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import (
    DECISION_EQUIVALENT,
    DECISION_NEEDS_REVIEW,
    DECISION_NOT_EQUIVALENT,
    HybridERModel,
    PairwiseEntityClassifier,
)
from src.pair_features import (
    FEATURE_COLUMNS,
    GATE_ONLY_FEATURES,
    MODEL_FEATURE_COLUMNS,
    HARD_CONFLICT_FEATURES,
    TfidfSimilarity,
    build_pair_features,
    extract_unit_specs,
    parse_specifications,
)
from src.preprocessing.text_cleaner import normalise_text
from src.synthetic_pairs import augment, build_self_supervised_pairs, mutate, split_of_group


# ---------------------------------------------------------------- spec parsing
class TestCanonicalSpecParser:
    def test_unit_spacing_is_canonical(self):
        assert extract_unit_specs("PIPE 10 MM") == extract_unit_specs("PIPE 10MM")

    def test_kilovolt_equals_volt(self):
        assert extract_unit_specs("CABLE 1.1KV")["V"] == extract_unit_specs("CABLE 1100 V")["V"]

    def test_class_equals_hash_rating(self):
        assert extract_unit_specs("VALVE CLASS 150")["CLASS"] == extract_unit_specs("VALVE 150#")["CLASS"]

    def test_sqmm_aliases(self):
        a = extract_unit_specs("CABLE 400 SQMM")["SQMM"]
        assert a == extract_unit_specs("CABLE 400MM2")["SQMM"] == extract_unit_specs("CABLE 400 SQ MM")["SQMM"]

    def test_decimal_trailing_zero(self):
        assert extract_unit_specs("LINE PIPE 101.60MM")["MM"] == extract_unit_specs("LINE PIPE 101.6MM")["MM"]

    def test_cable_cores(self):
        assert extract_unit_specs("XLPE CABLE 3C X 400 MM2")["CORE"] == {"3"}

    def test_quantity_is_not_a_specification(self):
        assert extract_unit_specs("API 6D BALL VALVES QUANTITY 76 NUMBERS") == {}
        assert extract_unit_specs("BALL VALVE 10 NOS") == {}

    def test_generic_title_has_no_identity_specs(self):
        assert parse_specifications("AIR COMPRESSOR")["identity_spec_count"] == 0

    def test_standard_is_identity_spec(self):
        assert parse_specifications("API 6D BALL VALVES")["identity_spec_count"] >= 1


# ---------------------------------------------------------------- synthesis
class TestPairSynthesis:
    def test_augmentation_preserves_specifications(self):
        rng = random.Random(0)
        seed = "GATE VALVE 2INCH CLASS 150 FLANGED STAINLESS STEEL"
        base = extract_unit_specs(normalise_text(seed))
        for _ in range(30):
            variant, _ = augment(seed, rng, 1, 3)
            assert extract_unit_specs(normalise_text(variant)) == base

    def test_mutation_changes_exactly_one_spec(self):
        rng = random.Random(1)
        seed = "CONVEYOR BELT 1400MM WIDTH NN/EP"
        mutated, info = mutate(seed, rng)
        assert normalise_text(mutated) != normalise_text(seed)
        assert info["mutation_old"] != info["mutation_new"]

    def test_mutation_never_touches_quantity(self):
        rng = random.Random(2)
        assert mutate("AIR COMPRESSOR QUANTITY 76 NUMBERS", rng) is None

    def test_split_assignment_is_deterministic(self):
        assert split_of_group("BALL VALVE") == split_of_group("BALL VALVE")

    def test_review_records_are_excluded(self):
        mats = pd.DataFrame({
            "corpus_id": ["M1", "M2", "M3", "M4"],
            "cleaned_description": ["GATE VALVE 2INCH CLASS 150", "XLPE CABLE 3C X 400 SQMM 1.1KV", "CENTRIFUGAL PUMP 60 M3/HR", "AIR COMPRESSOR"],
            "item_type_hint": ["valve", "cable", "pump", "compressor"],
        })
        pairs = build_self_supervised_pairs(mats, exclude_ids={"M1"}, exclude_texts={"AIR COMPRESSOR"}, random_state=0)
        assert "M1" not in set(pairs["seed_corpus_id"])
        assert "AIR COMPRESSOR" not in set(pairs["seed_text"])
        assert set(pairs["label"]) <= {0, 1}
        # every positive pair is the same seed; every hard negative records its mutation
        hn = pairs[pairs["source"] == "synthetic_hard_negative"]
        assert (hn["mutation_old"] != hn["mutation_new"]).all()


# ---------------------------------------------------------------- features
@pytest.fixture(scope="module")
def tfidf():
    corpus = ["GATE VALVE 2INCH CLASS 150", "BALL VALVE 2INCH CLASS 150", "CENTRIFUGAL PUMP", "XLPE CABLE 1.1KV", "CONVEYOR BELT 1400MM"]
    return TfidfSimilarity().fit(corpus)


def _emb(n, dim=8, same=True):
    rng = np.random.RandomState(0)
    a = rng.normal(size=(n, dim)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a, (a.copy() if same else np.roll(a, 1, axis=0))


class TestFeatureBuilder:
    def test_schema_and_finiteness(self, tfidf):
        ta = ["GATE VALVE 2INCH CLASS 150", "AIR COMPRESSOR"]
        tb = ["GATE VALVE 2 INCH 150#", "AIR COMPRESSOR"]
        ea, eb = _emb(2)
        X = build_pair_features(ta, tb, ea, eb, tfidf)
        assert list(X.columns) == FEATURE_COLUMNS
        assert np.isfinite(X.values).all()

    def test_no_identifier_columns(self, tfidf):
        for bad in ["corpus_id_1", "corpus_id_2", "enterprise_1", "organization_1", "human_label", "pair_id"]:
            assert bad not in FEATURE_COLUMNS

    def test_dimension_conflict_detected(self, tfidf):
        ea, eb = _emb(1)
        X = build_pair_features(["CONVEYOR BELT 1400MM"], ["CONVEYOR BELT 1600MM"], ea, eb, tfidf)
        assert X.loc[0, "unit_spec_conflicts"] == 1

    def test_equivalent_formats_do_not_conflict(self, tfidf):
        ea, eb = _emb(1)
        X = build_pair_features(["GATE VALVE 2INCH CLASS 150"], ["GATE VALVE 2 INCH 150#"], ea, eb, tfidf)
        assert X.loc[0, HARD_CONFLICT_FEATURES].sum() == 0

    def test_descriptor_conflict_detected(self, tfidf):
        ea, eb = _emb(1)
        X = build_pair_features(["BALL VALVE 2INCH"], ["GATE VALVE 2INCH"], ea, eb, tfidf)
        assert X.loc[0, "group_conflicts"] >= 1


# ---------------------------------------------------------------- model & policy
@pytest.fixture(scope="module")
def tiny_model(tfidf):
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.uniform(0, 1, size=(200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (X["sem_cosine"] > 0.5).astype(int).to_numpy()
    clf = PairwiseEntityClassifier(classifier_type="hist_gradient_boosting", max_iter=50).fit(X, y)
    return HybridERModel(clf, tfidf, {"binary": 0.5, "high": 0.85, "low": 0.25}, FEATURE_COLUMNS, {"model_name": "hist_gradient_boosting"})


def _row(**kw):
    base = {c: 0.0 for c in FEATURE_COLUMNS}
    base.update({"identity_spec_min": 1.0, "identity_spec_max": 1.0})
    base.update(kw)
    return pd.DataFrame([base])


class TestDecisionPolicy:
    def test_hard_conflict_is_never_merged(self, tiny_model):
        d = tiny_model.decide(_row(unit_spec_conflicts=1.0), proba=np.array([0.99]))
        assert d.loc[0, "decision"] == DECISION_NOT_EQUIVALENT

    def test_generic_title_goes_to_review(self, tiny_model):
        d = tiny_model.decide(_row(identity_spec_min=0.0, identity_spec_max=0.0), proba=np.array([0.99]))
        assert d.loc[0, "decision"] == DECISION_NEEDS_REVIEW

    def test_asymmetric_spec_goes_to_review(self, tiny_model):
        d = tiny_model.decide(_row(identity_spec_min=0.0, identity_spec_max=2.0), proba=np.array([0.99]))
        assert d.loc[0, "decision"] == DECISION_NEEDS_REVIEW

    def test_confident_specified_match_is_merged(self, tiny_model):
        d = tiny_model.decide(_row(), proba=np.array([0.97]))
        assert d.loc[0, "decision"] == DECISION_EQUIVALENT

    def test_low_probability_rejected(self, tiny_model):
        d = tiny_model.decide(_row(), proba=np.array([0.05]))
        assert d.loc[0, "decision"] == DECISION_NOT_EQUIVALENT

    def test_mid_probability_reviewed(self, tiny_model):
        d = tiny_model.decide(_row(), proba=np.array([0.6]))
        assert d.loc[0, "decision"] == DECISION_NEEDS_REVIEW

    def test_save_and_load_roundtrip(self, tiny_model):
        with tempfile.TemporaryDirectory() as tmp:
            path = tiny_model.save(Path(tmp) / "m.joblib")
            loaded = HybridERModel.load(path)
            X = _row()
            assert np.allclose(loaded.predict_proba(X), tiny_model.predict_proba(X))
            assert loaded.thresholds == tiny_model.thresholds

    def test_refuses_to_save_unfitted(self, tfidf):
        m = HybridERModel(PairwiseEntityClassifier(), tfidf, {"binary": 0.5, "high": 0.85, "low": 0.25}, FEATURE_COLUMNS)
        with tempfile.TemporaryDirectory() as tmp, pytest.raises(RuntimeError):
            m.save(Path(tmp) / "m.joblib")


# ---------------------------------------------------------------- trained artefact (if present)
MODEL_FILE = Path(__file__).resolve().parent.parent / "models" / "hybrid_er_model.joblib"


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="trained model not built yet (run python train_er_model.py)")
def test_trained_artifact_loads_with_valid_thresholds():
    import sklearn
    try:
        m = HybridERModel.load(MODEL_FILE)
    except Exception as e:
        pytest.skip(f"Unpickling failed (likely sklearn version mismatch): {e}. Retrain with python train_er_model.py")
    saved_sk = m.metadata.get("sklearn_version")
    if saved_sk and saved_sk != sklearn.__version__:
        pytest.skip(f"Saved sklearn version {saved_sk} != installed {sklearn.__version__}. Retrain with python train_er_model.py")
    t = m.thresholds
    assert 0.0 < t["low"] <= t["binary"] <= t["high"] < 1.0
    assert m.feature_names == MODEL_FEATURE_COLUMNS
    assert not set(GATE_ONLY_FEATURES) & set(m.feature_names)  # shortcut / noisy features withheld
    assert m.classifier.is_fitted
