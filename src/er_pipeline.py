"""
End-to-End Training & Evaluation Pipeline for the Hybrid Entity-Resolution Model.

Used by `train_er_model.py` (CLI) and notebooks 05 / 07 so that every reported number is
produced by one code path.

Stages
------
1. Load the real project assets: candidate pool, cached MiniLM embeddings, candidate pairs,
   and the human review set (150 pairs).
2. Build the self-supervised training corpus (labels by construction, see synthetic_pairs.py),
   excluding every record / description that appears in the human review set.
3. Embed new texts with MiniLM, compute 38 hybrid features.
4. Train Logistic Regression, Random Forest and Histogram Gradient Boosting; select on the
   validation split; choose precision-first decision thresholds on validation only.
5. Evaluate on the untouched synthetic test split (group-disjoint) and on review-set labels (AI-assisted, not independent human ground truth),
   against the exact-match, TF-IDF, MiniLM and rule-engine baselines.
6. Score all 22,464 real candidate pairs, cluster with conflict-aware clustering and emit a
   harmonised master with proposed codes and evidence.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.clustering import conflict_aware_clustering
from src.config import (
    DATA_LABELLED_DIR,
    DATA_PROCESSED_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from src.evaluation import compute_cluster_purity, compute_pairwise_metrics
from src.feature_extraction import generate_embeddings, load_embeddings
from src.harmonisation import generate_harmonised_code
from src.matching_engine import PrecisionMatchingEngine
from src.model import (
    DECISION_EQUIVALENT,
    DECISION_NEEDS_REVIEW,
    DECISION_NOT_EQUIVALENT,
    HybridERModel,
    PairwiseEntityClassifier,
)
from src.pair_features import (
    FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TfidfSimilarity,
    build_pair_features,
    extract_unit_specs,
    hard_conflict_mask,
)
from src.preprocessing.text_cleaner import normalise_text
from src.synthetic_pairs import build_self_supervised_pairs

# ------------------------------------------------------------------ artefact paths
MODEL_PATH = MODELS_DIR / "hybrid_er_model.joblib"
TRAINING_PAIRS_PATH = DATA_PROCESSED_DIR / "self_supervised_training_pairs.csv"
METRICS_JSON_PATH = METRICS_DIR / "er_model_metrics.json"
COMPARISON_CSV_PATH = METRICS_DIR / "model_comparison_synthetic_test.csv"
GOLD_COMPARISON_CSV_PATH = METRICS_DIR / "model_comparison_gold_labels.csv"
MUTATION_CSV_PATH = METRICS_DIR / "hard_negative_detection_by_mutation.csv"
GOLD_PREDICTIONS_PATH = METRICS_DIR / "review_set_model_predictions.csv"
REVIEW_PRIORITY_PATH = METRICS_DIR / "batch2_review_priority.csv"
BATCH2_LABELS_PATH = DATA_LABELLED_DIR / "ai_assisted_batch2_labels.csv"
BATCH2_COMPARISON_CSV_PATH = METRICS_DIR / "model_comparison_ai_assisted_batch2.csv"
FEATURE_IMPORTANCE_PATH = METRICS_DIR / "feature_importance.csv"
SCORED_CANDIDATES_PATH = RESULTS_DIR / "matching" / "model_scored_candidate_pairs.csv"
HARMONISED_MASTER_PATH = RESULTS_DIR / "matching" / "harmonised_master_model.csv"

CANDIDATE_MODELS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {"classifier_type": "scaled_logistic_regression", "C": 1.0},
    "random_forest": {"classifier_type": "random_forest", "n_estimators": 300, "max_depth": 14, "min_samples_leaf": 2},
    "hist_gradient_boosting": {"classifier_type": "hist_gradient_boosting", "max_iter": 300, "learning_rate": 0.08},
}
BASELINE_FEATURES = {
    "exact_match_baseline": "exact_norm_match",
    "tfidf_threshold_baseline": "tfidf_word_cosine",
    "minilm_threshold_baseline": "sem_cosine",
    "rule_engine_baseline": "engine_equivalent",
}
TARGET_AUTO_MERGE_PRECISION = 0.99
TARGET_AUTO_REJECT_NPV = 0.99
# Synthetic validation data is cleaner than production data, so the abstention band is never
# narrower than these precision-first safety margins.
MIN_AUTO_MERGE_THRESHOLD = 0.85
MAX_AUTO_REJECT_THRESHOLD = 0.25


def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


# ================================================================== data loading
def load_project_data() -> Dict[str, Any]:
    mat_path = DATA_PROCESSED_DIR / "preprocessed_materials.csv"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Required file '{mat_path}' not found. Please obtain it from data/processed/ or run preprocessing."
        )
    materials = pd.read_csv(mat_path, low_memory=False)
    materials["corpus_id"] = materials["corpus_id"].astype(str)

    emb_path = MODELS_DIR / "candidate_embeddings.npy"
    map_path = MODELS_DIR / "embedding_id_mapping.csv"
    if emb_path.exists() and map_path.exists():
        emb, id_to_idx = load_embeddings(emb_path, map_path)
    else:
        emb, id_to_idx = np.empty((0, 384), dtype=np.float32), {}

    cand_path = RESULTS_DIR / "matching" / "semantic_candidate_pairs.csv"
    if cand_path.exists():
        candidates = pd.read_csv(cand_path, low_memory=False)
    else:
        from src.candidate_generation import generate_tfidf_candidates
        _log("results/matching/semantic_candidate_pairs.csv missing; regenerating via TF-IDF candidate generation...")
        candidates = generate_tfidf_candidates(materials, min_similarity=0.35, ngram_range=(1, 2), min_df=2)
        lookup_tmp = EmbeddingLookup(materials, emb, id_to_idx)
        c1 = candidates["cleaned_description_1"].fillna("").astype(str).tolist()
        c2 = candidates["cleaned_description_2"].fillna("").astype(str).tolist()
        v1 = lookup_tmp.matrix(c1)
        v2 = lookup_tmp.matrix(c2)
        candidates["semantic_cosine_similarity"] = (v1 * v2).sum(axis=1) if len(candidates) else np.array([])
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(cand_path, index=False)

    rev_path = DATA_PROCESSED_DIR / "combined_candidate_pairs_150.csv"
    if not rev_path.exists():
        raise FileNotFoundError(
            f"Required file '{rev_path}' not found. Please obtain it from data/processed/."
        )
    review = pd.read_csv(rev_path)
    return {"materials": materials, "embeddings": emb, "id_to_idx": id_to_idx, "candidates": candidates, "review": review}


def review_set_exclusions(review: pd.DataFrame, materials: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """IDs and normalised descriptions of every record in the human review set (never trained on)."""
    ids = set(review["corpus_id_1"].astype(str)) | set(review["corpus_id_2"].astype(str))
    texts = set(materials.loc[materials["corpus_id"].isin(ids), "cleaned_description"].fillna("").astype(str).str.strip().str.upper())
    texts |= set(review["cleaned_description_1"].fillna("").astype(str).str.strip().str.upper())
    texts |= set(review["cleaned_description_2"].fillna("").astype(str).str.strip().str.upper())
    texts.discard("")
    return ids, texts


class EmbeddingLookup:
    """text -> MiniLM vector, seeded with the cached corpus embeddings; encodes unseen text lazily."""

    def __init__(self, materials: pd.DataFrame, embeddings: Optional[np.ndarray] = None, id_to_idx: Optional[Dict[str, int]] = None):
        self.vectors: Dict[str, np.ndarray] = {}
        if embeddings is not None and id_to_idx is not None and len(embeddings) > 0:
            for cid, text in zip(materials["corpus_id"], materials["cleaned_description"].fillna("").astype(str)):
                idx = id_to_idx.get(str(cid))
                if idx is not None:
                    self.vectors.setdefault(text.strip().upper(), embeddings[idx])
            self.dim = embeddings.shape[1]
        else:
            self.dim = 384
        self.n_encoded = 0

    def ensure(self, texts: Sequence[str], batch_size: int = 128) -> None:
        missing = sorted({str(t).strip().upper() for t in texts} - set(self.vectors))
        if missing:
            vecs = generate_embeddings(missing, model_name=DEFAULT_EMBEDDING_MODEL, batch_size=batch_size, normalize_embeddings=True)
            for t, v in zip(missing, vecs):
                self.vectors[t] = v
            self.n_encoded += len(missing)

    def matrix(self, texts: Sequence[str]) -> np.ndarray:
        self.ensure(texts)
        return np.vstack([self.vectors[str(t).strip().upper()] for t in texts]) if len(texts) else np.empty((0, self.dim), dtype=np.float32)


def featurize(df: pd.DataFrame, col_a: str, col_b: str, lookup: EmbeddingLookup, tfidf: TfidfSimilarity, spec_cache: Dict, engine: PrecisionMatchingEngine) -> pd.DataFrame:
    ta = df[col_a].fillna("").astype(str).str.strip().str.upper().tolist()
    tb = df[col_b].fillna("").astype(str).str.strip().str.upper().tolist()
    X = build_pair_features(ta, tb, lookup.matrix(ta), lookup.matrix(tb), tfidf, spec_cache=spec_cache, engine=engine)
    X.index = df.index
    return X


# ================================================================== metrics helpers
def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binary_metrics(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    m = compute_pairwise_metrics(y, pred)
    m["accuracy"] = float((pred == y).mean()) if len(y) else float("nan")
    m["threshold"] = float(threshold)
    if len(np.unique(y)) == 2:
        m["roc_auc"] = float(roc_auc_score(y, s))
        m["pr_auc"] = float(average_precision_score(y, s))
    else:
        m["roc_auc"] = float("nan")
        m["pr_auc"] = float("nan")
    return m


def tune_f1_threshold(y: Sequence[int], s: Sequence[float]) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if len(np.unique(s)) <= 2 and set(np.unique(s)) <= {0.0, 1.0}:
        return 0.5  # already binary
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else 0.5


def select_policy_thresholds(y: Sequence[int], p: Sequence[float]) -> Dict[str, float]:
    """Precision-first thresholds chosen on validation data only."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    binary = tune_f1_threshold(y, p)
    grid = np.unique(np.round(p, 4))
    high = None
    for t in grid:  # smallest t reaching the precision target
        sel = p >= t
        if sel.sum() >= 20 and y[sel].mean() >= TARGET_AUTO_MERGE_PRECISION:
            high = float(t)
            break
    low = None
    for t in grid[::-1]:  # largest t keeping auto-rejects clean
        sel = p <= t
        if sel.sum() >= 20 and (1 - y[sel].mean()) >= TARGET_AUTO_REJECT_NPV:
            low = float(t)
            break
    high = max(binary, high if high is not None else binary, MIN_AUTO_MERGE_THRESHOLD)
    low = min(binary, low if low is not None else MAX_AUTO_REJECT_THRESHOLD, MAX_AUTO_REJECT_THRESHOLD)
    return {"binary": float(binary), "high": float(high), "low": float(low)}


def policy_metrics(y: Sequence[int], decisions: Sequence[str]) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    d = np.asarray(decisions)
    eq, ne, rv = d == DECISION_EQUIVALENT, d == DECISION_NOT_EQUIVALENT, d == DECISION_NEEDS_REVIEW
    out = {
        "n": int(len(y)),
        "auto_merge_count": int(eq.sum()),
        "auto_reject_count": int(ne.sum()),
        "review_count": int(rv.sum()),
        "review_rate": float(rv.mean()) if len(y) else float("nan"),
        "auto_merge_precision": float(y[eq].mean()) if eq.any() else float("nan"),
        "auto_reject_npv": float(1 - y[ne].mean()) if ne.any() else float("nan"),
        "auto_merge_recall": float((eq & (y == 1)).sum() / max(1, (y == 1).sum())),
    }
    return out


# ================================================================== training
def _proba(clf: PairwiseEntityClassifier, X: pd.DataFrame) -> np.ndarray:
    """Classifier probability on the model-feature subset (gate-only features are withheld)."""
    return clf.predict_proba(X[MODEL_FEATURE_COLUMNS])


def train_and_select(
    X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray, verbose: bool = True
) -> Tuple[Dict[str, PairwiseEntityClassifier], pd.DataFrame, str]:
    models: Dict[str, PairwiseEntityClassifier] = {}
    rows = []
    for name, cfg in CANDIDATE_MODELS.items():
        cfg = dict(cfg)
        ctype = cfg.pop("classifier_type")
        t0 = time.time()
        clf = PairwiseEntityClassifier(classifier_type=ctype, random_state=RANDOM_SEED, **cfg)
        if ctype == "random_forest":
            clf.model.set_params(n_jobs=-1)
        clf.fit(X_tr[MODEL_FEATURE_COLUMNS], y_tr)
        p_va = _proba(clf, X_va)
        thr = tune_f1_threshold(y_va, p_va)
        m = binary_metrics(y_va, p_va, thr)
        m.update({"model": name, "train_seconds": round(time.time() - t0, 2)})
        rows.append(m)
        models[name] = clf
        _log(f"   - {name:24s} val F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  ({m['train_seconds']}s)", verbose)
    val_table = pd.DataFrame(rows).set_index("model")
    best = str(val_table.sort_values(["pr_auc", "f1"], ascending=False).index[0])
    return models, val_table, best


def run_training(verbose: bool = True, save: bool = True) -> Dict[str, Any]:
    """Full pipeline. Returns a dict holding every intermediate object for notebooks."""
    t_start = time.time()
    _log("[1/7] Loading project data ...", verbose)
    data = load_project_data()
    materials, review, candidates = data["materials"], data["review"], data["candidates"]
    excl_ids, excl_texts = review_set_exclusions(review, materials)
    _log(f"      candidate pool={len(materials):,}  candidate pairs={len(candidates):,}  review pairs={len(review)}  excluded review IDs={len(excl_ids)}", verbose)

    _log("[2/7] Building self-supervised training pairs (labels by construction) ...", verbose)
    pairs = build_self_supervised_pairs(
        materials,
        candidate_pairs=candidates,
        exclude_ids=excl_ids,
        exclude_texts=excl_texts,
        random_state=RANDOM_SEED,
        unit_spec_parser=extract_unit_specs,
    )
    _log(f"      pairs={len(pairs):,}  by source={pairs['source'].value_counts().to_dict()}", verbose)
    _log(f"      by split={pairs['split'].value_counts().to_dict()}  positives={int(pairs['label'].sum()):,}", verbose)

    _log("[3/7] Embedding texts (MiniLM) and computing hybrid features ...", verbose)
    lookup = EmbeddingLookup(materials, data["embeddings"], data["id_to_idx"])
    tfidf = TfidfSimilarity().fit(materials["cleaned_description"].fillna("").astype(str).tolist())
    spec_cache: Dict[str, Any] = {}
    engine = PrecisionMatchingEngine()
    t0 = time.time()
    lookup.ensure(pd.concat([pairs["text_a"], pairs["text_b"]]).tolist())
    X_all = featurize(pairs, "text_a", "text_b", lookup, tfidf, spec_cache, engine)
    y_all = pairs["label"].to_numpy().astype(int)
    _log(f"      newly encoded texts={lookup.n_encoded:,}; feature matrix={X_all.shape}; {time.time() - t0:.1f}s", verbose)

    tr, va, te = (pairs["split"] == s for s in ("train", "val", "test"))
    leak = set(pairs.loc[tr, "group_id"]) & (set(pairs.loc[va, "group_id"]) | set(pairs.loc[te, "group_id"]))
    _log(f"      group leakage between train and val/test: {len(leak)} groups", verbose)

    _log("[4/7] Training candidate classifiers ...", verbose)
    models, val_table, best_name = train_and_select(X_all[tr], y_all[tr], X_all[va], y_all[va], verbose)
    best = models[best_name]
    p_va = _proba(best, X_all[va])
    thresholds = select_policy_thresholds(y_all[va], p_va)
    _log(f"      selected model: {best_name}; thresholds={thresholds}", verbose)

    hybrid = HybridERModel(
        classifier=best,
        tfidf=tfidf,
        thresholds=thresholds,
        feature_names=MODEL_FEATURE_COLUMNS,
        metadata={
            "model_name": best_name,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "training_paradigm": "self-supervised (labels by construction) - no human or AI labels used for fitting",
            "n_train_pairs": int(tr.sum()),
            "n_val_pairs": int(va.sum()),
            "n_test_pairs": int(te.sum()),
            "random_seed": RANDOM_SEED,
        },
    )

    _log("[5/7] Evaluating on held-out synthetic test split ...", verbose)
    X_te, y_te = X_all[te], y_all[te]
    test_rows = []
    test_scores: Dict[str, np.ndarray] = {}
    for name, clf in models.items():
        s = _proba(clf, X_te)
        thr = tune_f1_threshold(y_all[va], _proba(clf, X_all[va]))
        m = binary_metrics(y_te, s, thr)
        m["model"] = name
        test_rows.append(m)
        test_scores[name] = s
    for name, feat in BASELINE_FEATURES.items():
        thr = tune_f1_threshold(y_all[va], X_all.loc[va, feat])
        m = binary_metrics(y_te, X_te[feat], thr)
        m["model"] = name
        test_rows.append(m)
        test_scores[name] = X_te[feat].to_numpy()
    test_table = pd.DataFrame(test_rows).set_index("model")
    p_te = test_scores[best_name]
    dec_te = hybrid.decide(X_te, p_te)
    test_policy = policy_metrics(y_te, dec_te["decision"])
    _log(test_table[["precision", "recall", "f1", "roc_auc", "pr_auc"]].round(4).to_string(), verbose)
    _log(f"      deployed policy on test: {test_policy}", verbose)

    # hard-negative detection by mutation family
    te_pairs = pairs[te].copy()
    te_pairs["p_match"] = p_te
    te_pairs["pred"] = (p_te >= thresholds["binary"]).astype(int)
    neg = te_pairs[te_pairs["label"] == 0]
    mut_table = (
        neg.groupby(["mutation_kind", "mutation_group"]).agg(n=("pred", "size"), detected=("pred", lambda s: int((s == 0).sum())))
        .assign(detection_rate=lambda d: d["detected"] / d["n"])
        .reset_index()
        .sort_values("n", ascending=False)
    )
    unseen = neg[neg["mutation_unseen_by_features"].astype(bool)]
    pos_te = te_pairs[te_pairs["label"] == 1]
    op_rows = []
    for op in ["boilerplate_prefix", "quantity_suffix", "noun_first_reorder", "abbreviation", "unit_format", "typo", "plural", "filler_drop", "punctuation", "casing"]:
        sel = pos_te[pos_te["aug_ops_b"].str.contains(op, regex=False)]
        if len(sel):
            op_rows.append({"augmentation": op, "n": len(sel), "recall": float(sel["pred"].mean())})
    aug_table = pd.DataFrame(op_rows).sort_values("n", ascending=False)

    # cluster purity on the synthetic test split (entities known by construction)
    purity = synthetic_cluster_purity(te_pairs, p_te >= thresholds["binary"], X_te)

    _log("[6/7] Evaluating on review-set labels (AI-assisted, not independent human ground truth) ...", verbose)
    gold = evaluate_review_set(hybrid, models, review, lookup, spec_cache, engine, y_va=y_all[va], X_va=X_all[va])

    _log("[7/7] Scoring all real candidate pairs and building harmonised master ...", verbose)
    scored = score_candidate_pairs(hybrid, candidates, lookup, spec_cache, engine)
    master, cluster_audit = build_harmonised_master(scored, materials)
    _log(f"      decisions={scored['model_decision'].value_counts().to_dict()}  multi-record clusters={len(master)}", verbose)

    importance = permutation_importance_table(best, X_te[MODEL_FEATURE_COLUMNS], y_te)

    metrics = {
        "model_version": HybridERModel.VERSION,
        "selected_model": best_name,
        "thresholds": thresholds,
        "training_data": {
            "total_pairs": int(len(pairs)),
            "by_source": pairs["source"].value_counts().to_dict(),
            "by_split": pairs["split"].value_counts().to_dict(),
            "positives": int(pairs["label"].sum()),
            "negatives": int((pairs["label"] == 0).sum()),
            "excluded_review_ids": len(excl_ids),
            "group_leakage": len(leak),
            "newly_encoded_texts": lookup.n_encoded,
        },
        "validation": val_table.reset_index().to_dict(orient="records"),
        "synthetic_test": test_table.reset_index().to_dict(orient="records"),
        "synthetic_test_policy": test_policy,
        "synthetic_test_cluster_purity": purity,
        "unseen_mutation_detection_rate": float((unseen["pred"] == 0).mean()) if len(unseen) else float("nan"),
        "gold": gold["summary"],
        "ai_assisted_batch2": gold.get("ai_assisted_batch2"),
        "real_candidates": {
            "n_pairs": int(len(scored)),
            "decisions": scored["model_decision"].value_counts().to_dict(),
            "cross_enterprise_decisions": scored.loc[scored["is_cross_enterprise"].astype(bool), "model_decision"].value_counts().to_dict(),
            "multi_record_clusters": int(len(master)),
            "records_in_clusters": int(master["member_count"].sum()) if len(master) else 0,
            "blocked_merges": int(cluster_audit["blocked_merges_count"]),
            "transitive_conflicts": int(cluster_audit["transitive_conflicts_count"]),
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    if save:
        for p in [MODEL_PATH, TRAINING_PAIRS_PATH, METRICS_JSON_PATH, SCORED_CANDIDATES_PATH]:
            p.parent.mkdir(parents=True, exist_ok=True)
        hybrid.metadata["metrics_summary"] = {
            "synthetic_test_f1": float(test_table.loc[best_name, "f1"]),
            "synthetic_test_pr_auc": float(test_table.loc[best_name, "pr_auc"]),
            "gold_binary_pairs": gold["summary"]["n_binary"],
        }
        hybrid.save(MODEL_PATH)
        pairs.to_csv(TRAINING_PAIRS_PATH, index=False)
        test_table.to_csv(COMPARISON_CSV_PATH)
        gold["comparison"].to_csv(GOLD_COMPARISON_CSV_PATH)
        if "ai_assisted_batch2_comparison" in gold and gold["ai_assisted_batch2_comparison"] is not None:
            gold["ai_assisted_batch2_comparison"].to_csv(BATCH2_COMPARISON_CSV_PATH)
        mut_table.to_csv(MUTATION_CSV_PATH, index=False)
        gold["predictions"].to_csv(GOLD_PREDICTIONS_PATH, index=False)
        gold["review_priority"].to_csv(REVIEW_PRIORITY_PATH, index=False)
        importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        scored.to_csv(SCORED_CANDIDATES_PATH, index=False)
        master.to_csv(HARMONISED_MASTER_PATH, index=False)
        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=_json_default)
        _log(f"[OK] Saved model -> {MODEL_PATH}", verbose)

    _log(f"[DONE] Pipeline finished in {time.time() - t_start:.1f}s", verbose)
    return {
        "data": data,
        "pairs": pairs,
        "X_all": X_all,
        "y_all": y_all,
        "masks": {"train": tr, "val": va, "test": te},
        "models": models,
        "best_name": best_name,
        "hybrid": hybrid,
        "val_table": val_table,
        "test_table": test_table,
        "test_scores": test_scores,
        "test_decisions": dec_te,
        "test_policy": test_policy,
        "mutation_table": mut_table,
        "augmentation_table": aug_table,
        "purity": purity,
        "gold": gold,
        "scored": scored,
        "master": master,
        "cluster_audit": cluster_audit,
        "importance": importance,
        "metrics": metrics,
        "lookup": lookup,
    }


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


# ================================================================== evaluation pieces
def synthetic_cluster_purity(te_pairs: pd.DataFrame, pred_match: np.ndarray, X_te: pd.DataFrame) -> Dict[str, Any]:
    """
    Cluster purity on the synthetic test split where true entities are known by construction:
    {anchor, positive variant} form one entity; a mutated variant is a separate entity.
    """
    syn = te_pairs["source"].isin(["synthetic_positive", "synthetic_hard_negative"]).to_numpy()
    df = te_pairs.loc[syn, ["node_a", "node_b", "entity_a", "entity_b", "pair_uid"]].copy()
    truth: Dict[str, str] = {}
    for na, nb, ea, eb in zip(df["node_a"], df["node_b"], df["entity_a"], df["entity_b"]):
        truth[na] = ea
        truth[nb] = eb
    conflict = hard_conflict_mask(X_te)[syn]
    results = {}
    variants = {
        "exact_match_baseline": X_te["exact_norm_match"].to_numpy()[syn] >= 0.5,
        "minilm_threshold_0.90": X_te["sem_cosine"].to_numpy()[syn] >= 0.90,
        "hybrid_model": np.asarray(pred_match)[syn],
    }
    for name, pm in variants.items():
        d = df.copy()
        d["prediction"] = np.where(pm, "1", "0")
        if name == "hybrid_model":
            d["has_attribute_conflict"] = conflict
        clusters, audit = conflict_aware_clustering(d, id1_col="node_a", id2_col="node_b", prediction_col="prediction", isolate_ambiguous=False)
        purity = compute_cluster_purity(clusters, truth)
        inverse = compute_cluster_purity(truth, clusters)
        f = 2 * purity * inverse / (purity + inverse) if purity + inverse else 0.0
        results[name] = {
            "purity": round(purity, 4),
            "inverse_purity": round(inverse, 4),
            "f_measure": round(f, 4),
            "n_records": len(truth),
            "n_true_entities": len(set(truth.values())),
            "n_predicted_clusters": len(set(clusters.values())),
        }
    return results


def evaluate_review_set(
    hybrid: HybridERModel,
    models: Dict[str, PairwiseEntityClassifier],
    review: pd.DataFrame,
    lookup: EmbeddingLookup,
    spec_cache: Dict,
    engine: PrecisionMatchingEngine,
    y_va: np.ndarray,
    X_va: pd.DataFrame,
) -> Dict[str, Any]:
    """Score the 150 review pairs; metrics on confirmed binary review-set labels (AI-assisted, not independent human ground truth)."""
    rv = review.copy()
    X = featurize(rv, "cleaned_description_1", "cleaned_description_2", lookup, tfidf=hybrid.tfidf, spec_cache=spec_cache, engine=engine)
    dec = hybrid.decide(X)
    label = rv["human_label"].astype(str).str.strip().str.upper().str.replace(".0", "", regex=False)
    preds = pd.DataFrame({
        "pair_id": rv["pair_id"],
        "corpus_id_1": rv["corpus_id_1"],
        "corpus_id_2": rv["corpus_id_2"],
        "description_1": rv["cleaned_description_1"],
        "description_2": rv["cleaned_description_2"],
        "human_label": label,
        "model_match_probability": dec["match_probability"].values,
        "model_binary_prediction": (dec["match_probability"].values >= hybrid.thresholds["binary"]).astype(int),
        "model_decision": dec["decision"].values,
        "model_confidence": dec["confidence"].values,
        "model_reason": dec["reason"].values,
        "rule_engine_decision": np.where(X["engine_equivalent"] > 0, DECISION_EQUIVALENT, np.where(X["engine_not_equivalent"] > 0, DECISION_NOT_EQUIVALENT, DECISION_NEEDS_REVIEW)),
        "ai_suggested_hypothesis": rv.get("ai_suggested_hypothesis"),
    })

    binary = label.isin(["0", "1"]).to_numpy()
    y = label[binary].astype(int).to_numpy()
    Xb = X[binary]
    comp_rows = []
    for name, clf in models.items():
        s = _proba(clf, Xb)
        thr = hybrid.thresholds["binary"] if name == hybrid.metadata.get("model_name") else tune_f1_threshold(y_va, _proba(clf, X_va))
        m = binary_metrics(y, s, thr)
        m["model"] = name
        comp_rows.append(m)
    for name, feat in BASELINE_FEATURES.items():
        thr = tune_f1_threshold(y_va, X_va[feat])
        m = binary_metrics(y, Xb[feat], thr)
        m["model"] = name
        comp_rows.append(m)
    comparison = pd.DataFrame(comp_rows).set_index("model")
    for col in ["precision", "recall", "accuracy"]:
        lo_hi = []
        for name, r in comparison.iterrows():
            if col == "precision":
                k, n = r["true_positives"], r["true_positives"] + r["false_positives"]
            elif col == "recall":
                k, n = r["true_positives"], r["true_positives"] + r["false_negatives"]
            else:
                k, n = r["true_positives"] + r["true_negatives"], r["total_pairs"]
            lo, hi = wilson_interval(int(k), int(n))
            lo_hi.append(f"[{lo:.2f}, {hi:.2f}]" if n else "n/a")
        comparison[f"{col}_95ci"] = lo_hi

    pol_binary = policy_metrics(y, dec.loc[binary, "decision"])
    uncertain = (label == "UNCERTAIN").to_numpy()
    unc_dec = dec.loc[uncertain, "decision"].value_counts().to_dict()
    pending = (label == "PENDING").to_numpy()

    # active-learning priority for the pending Batch 2: closest to the decision boundary first
    pr = preds[pending].copy()
    pr["uncertainty"] = 1 - (2 * (pr["model_match_probability"] - 0.5)).abs()
    pr = pr.sort_values(["uncertainty"], ascending=False)
    pr.insert(0, "review_rank", range(1, len(pr) + 1))

    summary = {
        "n_review_pairs": int(len(rv)),
        "n_binary": int(binary.sum()),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_uncertain": int(uncertain.sum()),
        "n_pending": int(pending.sum()),
        "selected_model_binary": comparison.loc[hybrid.metadata.get("model_name")].to_dict(),
        "deployed_policy_on_binary": pol_binary,
        "uncertain_pairs_decisions": unc_dec,
        "uncertain_pairs_auto_merged": int(unc_dec.get(DECISION_EQUIVALENT, 0)),
        "pending_pairs_decisions": dec.loc[pending, "decision"].value_counts().to_dict(),
    }

    # AI-assisted Batch 2 evaluation tier (if data/labelled/ai_assisted_batch2_labels.csv exists)
    b2_summary = None
    b2_comparison = None
    b2_preds = None
    if BATCH2_LABELS_PATH.exists():
        b2_labels_df = pd.read_csv(BATCH2_LABELS_PATH)
        b2_map = b2_labels_df.set_index("pair_id")["ai_label"].to_dict()
        b2_label = rv["pair_id"].map(b2_map).fillna("").astype(str).str.strip().str.upper()
        b2_mask = b2_label.isin(["0", "1", "UNCERTAIN"]).to_numpy()
        b2_binary = b2_label.isin(["0", "1"]).to_numpy()
        b2_uncertain = (b2_label == "UNCERTAIN").to_numpy()

        if b2_binary.sum() > 0:
            y_b2 = b2_label[b2_binary].astype(int).to_numpy()
            X_b2 = X[b2_binary]
            b2_comp_rows = []
            for name, clf in models.items():
                s = _proba(clf, X_b2)
                thr = hybrid.thresholds["binary"] if name == hybrid.metadata.get("model_name") else tune_f1_threshold(y_va, _proba(clf, X_va))
                m = binary_metrics(y_b2, s, thr)
                m["model"] = name
                b2_comp_rows.append(m)
            for name, feat in BASELINE_FEATURES.items():
                thr = tune_f1_threshold(y_va, X_va[feat])
                m = binary_metrics(y_b2, X_b2[feat], thr)
                m["model"] = name
                b2_comp_rows.append(m)
            b2_comparison = pd.DataFrame(b2_comp_rows).set_index("model")
            for col in ["precision", "recall", "accuracy"]:
                lo_hi = []
                for name, r in b2_comparison.iterrows():
                    if col == "precision":
                        k, n = r["true_positives"], r["true_positives"] + r["false_positives"]
                    elif col == "recall":
                        k, n = r["true_positives"], r["true_positives"] + r["false_negatives"]
                    else:
                        k, n = r["true_positives"] + r["true_negatives"], r["total_pairs"]
                    lo, hi = wilson_interval(int(k), int(n))
                    lo_hi.append(f"[{lo:.2f}, {hi:.2f}]" if n else "n/a")
                b2_comparison[f"{col}_95ci"] = lo_hi

            b2_pol_binary = policy_metrics(y_b2, dec.loc[b2_binary, "decision"])
            b2_unc_dec = dec.loc[b2_uncertain, "decision"].value_counts().to_dict()
            b2_auto_merged = int(b2_unc_dec.get(DECISION_EQUIVALENT, 0))

            b2_summary = {
                "n_pairs": int(b2_mask.sum()),
                "n_binary": int(b2_binary.sum()),
                "n_positive": int((y_b2 == 1).sum()),
                "n_negative": int((y_b2 == 0).sum()),
                "n_uncertain": int(b2_uncertain.sum()),
                "selected_model_binary": b2_comparison.loc[hybrid.metadata.get("model_name")].to_dict(),
                "deployed_policy_on_binary": b2_pol_binary,
                "uncertain_pairs_decisions": b2_unc_dec,
                "uncertain_pairs_auto_merged": b2_auto_merged,
                "note": "Agreement with an AI annotator applying docs/annotation_guidelines.md blind to model output. This is not independent human ground truth; annotator shares the designer's rules, so routing agreement on UNCERTAIN pairs is expected by design.",
            }
            b2_preds = pd.DataFrame({
                "pair_id": rv.loc[b2_mask, "pair_id"],
                "description_1": rv.loc[b2_mask, "cleaned_description_1"],
                "description_2": rv.loc[b2_mask, "cleaned_description_2"],
                "ai_label": b2_label[b2_mask],
                "model_match_probability": dec.loc[b2_mask, "match_probability"].values,
                "model_binary_prediction": (dec.loc[b2_mask, "match_probability"].values >= hybrid.thresholds["binary"]).astype(int),
                "model_decision": dec.loc[b2_mask, "decision"].values,
                "model_confidence": dec.loc[b2_mask, "confidence"].values,
                "model_reason": dec.loc[b2_mask, "reason"].values,
            })

    out_res = {"predictions": preds, "comparison": comparison, "summary": summary, "review_priority": pr, "features": X}
    if b2_summary is not None:
        out_res["ai_assisted_batch2"] = b2_summary
        out_res["ai_assisted_batch2_comparison"] = b2_comparison
        out_res["ai_assisted_batch2_predictions"] = b2_preds
    return out_res


def score_candidate_pairs(hybrid: HybridERModel, candidates: pd.DataFrame, lookup: EmbeddingLookup, spec_cache: Dict, engine: PrecisionMatchingEngine) -> pd.DataFrame:
    X = featurize(candidates, "cleaned_description_1", "cleaned_description_2", lookup, hybrid.tfidf, spec_cache, engine)
    dec = hybrid.decide(X)
    keep = ["corpus_id_1", "organization_1", "cleaned_description_1", "corpus_id_2", "organization_2", "cleaned_description_2", "is_cross_enterprise", "tfidf_similarity", "semantic_cosine_similarity"]
    out = candidates[keep].copy()
    out["model_match_probability"] = dec["match_probability"].values
    out["model_decision"] = dec["decision"].values
    out["model_confidence"] = dec["confidence"].values
    out["model_reason"] = dec["reason"].values
    out["hard_spec_conflict"] = hard_conflict_mask(X)
    return out


def build_harmonised_master(scored: pd.DataFrame, materials: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Conflict-aware clustering of model decisions -> proposed harmonised codes with evidence."""
    df = scored.copy()
    df["prediction"] = df["model_decision"]
    df["has_attribute_conflict"] = df["hard_spec_conflict"] | (df["model_decision"] == DECISION_NOT_EQUIVALENT)
    clusters, audit = conflict_aware_clustering(
        df,
        id1_col="corpus_id_1",
        id2_col="corpus_id_2",
        prediction_col="prediction",
        similarity_col="model_match_probability",
        isolate_ambiguous=True,
    )
    members: Dict[str, List[str]] = {}
    for rid, cid in clusters.items():
        members.setdefault(cid, []).append(rid)
    info = materials.set_index("corpus_id")
    eq = df[df["model_decision"] == DECISION_EQUIVALENT]
    rows = []
    for cid, ids in members.items():
        if len(ids) < 2:
            continue
        ids = sorted(ids)
        sub = info.loc[[i for i in ids if i in info.index]]
        descs = sub["cleaned_description"].fillna("").astype(str)
        probs = eq[eq["corpus_id_1"].isin(ids) & eq["corpus_id_2"].isin(ids)]["model_match_probability"]
        rows.append({
            "harmonised_code": generate_harmonised_code("|".join(ids)),
            "member_count": len(ids),
            "enterprises": "; ".join(sorted(sub["organization"].astype(str).unique())),
            "is_cross_enterprise": sub["organization"].nunique() > 1,
            "member_corpus_ids": "; ".join(ids),
            "representative_description": max(descs, key=len) if len(descs) else "",
            "distinct_descriptions": " || ".join(sorted(set(descs))[:5]),
            "mean_match_probability": round(float(probs.mean()), 4) if len(probs) else None,
            "min_match_probability": round(float(probs.min()), 4) if len(probs) else None,
            "review_status": "PROPOSED",
        })
    master = pd.DataFrame(rows)
    if len(master):
        master = master.sort_values(["is_cross_enterprise", "member_count", "mean_match_probability"], ascending=[False, False, False]).reset_index(drop=True)
    return master, audit


def permutation_importance_table(clf: PairwiseEntityClassifier, X: pd.DataFrame, y: np.ndarray, n: int = 3000) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    r = permutation_importance(clf.model, X.iloc[idx], y[idx], scoring="average_precision", n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)
    return (
        pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ================================================================== inference helper
def load_model(path: Path = MODEL_PATH) -> HybridERModel:
    try:
        return HybridERModel.load(path)
    except Exception as e:
        print("Failed to unpickle model. Please retrain with: python train_er_model.py")
        raise e


def match_descriptions(desc_a: str, desc_b: str, model: Optional[HybridERModel] = None, lookup: Optional[EmbeddingLookup] = None) -> Dict[str, Any]:
    """Convenience API: score two raw descriptions end-to-end (used in notebooks / demo)."""
    model = model or load_model()
    a, b = normalise_text(desc_a), normalise_text(desc_b)
    if lookup is None:
        emb = generate_embeddings([a, b])
        ea, eb = emb[[0]], emb[[1]]
    else:
        ea, eb = lookup.matrix([a]), lookup.matrix([b])
    X = model.featurize([a], [b], ea, eb)
    d = model.decide(X).iloc[0].to_dict()
    d.update({"normalised_a": a, "normalised_b": b})
    return d
