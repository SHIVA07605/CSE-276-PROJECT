"""
Pairwise Supervised Entity Resolution Classifier Module.

Encapsulates machine learning classifiers (Random Forest, Logistic Regression)
trained to predict whether a candidate record pair is functionally equivalent.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import RANDOM_SEED, MODELS_DIR


class PairwiseEntityClassifier:
    """
    Supervised classifier for pairwise record matching.
    """

    def __init__(
        self,
        classifier_type: str = "random_forest",
        threshold: float = 0.5,
        random_state: int = RANDOM_SEED,
        **kwargs,
    ):
        """
        Initialise classifier.

        Args:
            classifier_type: 'random_forest' or 'logistic_regression'.
            threshold: Probability decision threshold for predicting a positive match.
            random_state: Seed for reproducibility.
            **kwargs: Additional hyperparameters passed to the scikit-learn estimator.
        """
        self.classifier_type = classifier_type
        self.threshold = threshold
        self.random_state = random_state

        if classifier_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                min_samples_leaf=kwargs.get("min_samples_leaf", 1),
                min_samples_split=kwargs.get("min_samples_split", 2),
                random_state=self.random_state,
                class_weight=kwargs.get("class_weight", "balanced"),
            )
        elif classifier_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=kwargs.get("max_iter", 1000),
                class_weight=kwargs.get("class_weight", "balanced"),
            )
        elif classifier_type == "scaled_logistic_regression":
            self.model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    random_state=self.random_state,
                    max_iter=kwargs.get("max_iter", 3000),
                    C=kwargs.get("C", 1.0),
                    class_weight=kwargs.get("class_weight", "balanced"),
                ),
            )
        elif classifier_type == "hist_gradient_boosting":
            self.model = HistGradientBoostingClassifier(
                max_iter=kwargs.get("max_iter", 300),
                learning_rate=kwargs.get("learning_rate", 0.08),
                max_leaf_nodes=kwargs.get("max_leaf_nodes", 31),
                min_samples_leaf=kwargs.get("min_samples_leaf", 20),
                l2_regularization=kwargs.get("l2_regularization", 1.0),
                class_weight=kwargs.get("class_weight", "balanced"),
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        self.is_fitted = False
        self.feature_names: Optional[list] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "PairwiseEntityClassifier":
        """
        Train the classifier on pairwise feature vectors.

        Args:
            X: Feature matrix of candidate pairs.
            y: Binary target labels (1 = match, 0 = non-match).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict pairwise match probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability of match class (1).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba can be called.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict binary labels applying the decision threshold.

        Args:
            X: Feature matrix.

        Returns:
            Binary predictions (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def save(self, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Serialize model and configuration to disk.

        Args:
            filepath: Path to save the serialized model (.joblib or .pkl).
            metadata: Optional dictionary recording model parameters, evaluation metrics,
                      training dataset info, and version.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "classifier_type": self.classifier_type,
            "threshold": self.threshold,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }
        if metadata:
            payload["metadata"] = metadata
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PairwiseEntityClassifier":
        """
        Load serialized model from disk.
        """
        data = joblib.load(filepath)
        instance = cls(
            classifier_type=data["classifier_type"],
            threshold=data["threshold"],
            random_state=data["random_state"],
        )
        instance.model = data["model"]
        instance.is_fitted = data["is_fitted"]
        instance.feature_names = data.get("feature_names")
        instance.metadata = data.get("metadata", {})
        return instance


# ==============================================================================
# Hybrid self-supervised entity-resolution model (feature builder + classifier +
# precision-first three-way decision policy)
# ==============================================================================
DECISION_EQUIVALENT = "EQUIVALENT"
DECISION_NOT_EQUIVALENT = "NOT_EQUIVALENT"
DECISION_NEEDS_REVIEW = "NEEDS_REVIEW"


class HybridERModel:
    """
    Deployable entity-resolution model.

    Pipeline: (desc_a, desc_b, embeddings) -> 38 hybrid features -> fitted classifier ->
    match probability -> three-way decision with specification safety gates:

      1. explicit canonical specification conflict          -> NOT_EQUIVALENT (HIGH)
      2. probability <= t_low                                -> NOT_EQUIVALENT
      3. probability >= t_high and both records carry identity specifications that do not
         appear on one side only                             -> EQUIVALENT
      4. otherwise (generic titles, asymmetric specs, mid scores) -> NEEDS_REVIEW
    """

    VERSION = "2.0.0-hybrid-self-supervised"

    def __init__(
        self,
        classifier: PairwiseEntityClassifier,
        tfidf: Any,
        thresholds: Dict[str, float],
        feature_names: Sequence[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.classifier = classifier
        self.tfidf = tfidf
        self.thresholds = dict(thresholds)
        self.feature_names = list(feature_names)
        self.metadata = dict(metadata or {})

    # ---------------------------------------------------------------- features
    def featurize(self, texts_a: Sequence[str], texts_b: Sequence[str], emb_a: np.ndarray, emb_b: np.ndarray, spec_cache: Optional[dict] = None) -> pd.DataFrame:
        from src.pair_features import build_pair_features

        # full feature frame: the classifier reads self.feature_names, the safety gates read the rest
        return build_pair_features(texts_a, texts_b, emb_a, emb_b, self.tfidf, spec_cache=spec_cache)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(X[self.feature_names])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.thresholds["binary"]).astype(int)

    # ---------------------------------------------------------------- decisions
    def decide(self, X: pd.DataFrame, proba: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Return decision, confidence and a human-readable reason for every pair."""
        from src.pair_features import HARD_CONFLICT_FEATURES

        p = self.predict_proba(X) if proba is None else np.asarray(proba)
        t_low, t_high = self.thresholds["low"], self.thresholds["high"]
        out: List[Dict[str, Any]] = []
        for i in range(len(X)):
            row = X.iloc[i]
            conflicts = [f for f in HARD_CONFLICT_FEATURES if row[f] > 0]
            if conflicts:
                out.append({"decision": DECISION_NOT_EQUIVALENT, "confidence": "HIGH", "reason": "Explicit specification conflict: " + ", ".join(conflicts)})
            elif p[i] <= t_low:
                out.append({"decision": DECISION_NOT_EQUIVALENT, "confidence": "HIGH" if p[i] <= t_low / 2 else "MEDIUM", "reason": f"Low match probability ({p[i]:.3f} <= {t_low:.3f})"})
            elif p[i] >= t_high:
                if row["identity_spec_max"] == 0:
                    out.append({"decision": DECISION_NEEDS_REVIEW, "confidence": "LOW", "reason": f"High similarity ({p[i]:.3f}) but generic title: no size/rating/standard/grade on either record"})
                elif row["identity_spec_min"] == 0 or row["unit_spec_one_side"] > 0 or row.get("group_one_side", 0) > 0:
                    out.append({"decision": DECISION_NEEDS_REVIEW, "confidence": "MEDIUM", "reason": f"High similarity ({p[i]:.3f}) but asymmetric specification (stated on one record only)"})
                else:
                    out.append({"decision": DECISION_EQUIVALENT, "confidence": "HIGH" if p[i] >= max(0.95, t_high) else "MEDIUM", "reason": f"High match probability ({p[i]:.3f}) with shared, non-conflicting specifications"})
            else:
                out.append({"decision": DECISION_NEEDS_REVIEW, "confidence": "LOW", "reason": f"Uncertain match probability ({p[i]:.3f})"})
        df = pd.DataFrame(out, index=X.index)
        df.insert(0, "match_probability", np.round(p, 4))
        return df

    # ---------------------------------------------------------------- persistence
    def save(self, filepath: Union[str, Path]) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.classifier.is_fitted:
            raise RuntimeError("Refusing to serialise an unfitted model.")
        import sklearn
        self.metadata["sklearn_version"] = sklearn.__version__
        joblib.dump(
            {
                "version": self.VERSION,
                "classifier": self.classifier,
                "tfidf": self.tfidf,
                "thresholds": self.thresholds,
                "feature_names": self.feature_names,
                "metadata": self.metadata,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HybridERModel":
        try:
            data = joblib.load(filepath)
        except Exception as e:
            print("Failed to unpickle model. Please retrain with: python train_er_model.py")
            raise e
        return cls(
            classifier=data["classifier"],
            tfidf=data["tfidf"],
            thresholds=data["thresholds"],
            feature_names=data["feature_names"],
            metadata=data.get("metadata", {}),
        )
