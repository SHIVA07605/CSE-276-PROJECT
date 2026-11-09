"""
Train, evaluate and deploy the hybrid self-supervised entity-resolution model.

Usage:
    python train_er_model.py          # train + evaluate + score
    python train_er_model.py --check  # preflight environment check

Produces:
    models/hybrid_er_model.joblib                         fitted model + thresholds + metadata
    data/processed/self_supervised_training_pairs.csv     training corpus with full provenance
    results/metrics/er_model_metrics.json                 every reported number
    results/metrics/model_comparison_synthetic_test.csv   models vs baselines (held-out test)
    results/metrics/model_comparison_gold_labels.csv      models vs baselines (human labels)
    results/metrics/model_comparison_ai_assisted_batch2.csv models vs baselines (Batch 2 tier)
    results/metrics/review_set_model_predictions.csv      predictions for the 150 review pairs
    results/metrics/batch2_review_priority.csv            active-learning order for pending pairs
    results/matching/model_scored_candidate_pairs.csv     all 22,464 real candidate pairs scored
    results/matching/harmonised_master_model.csv          proposed harmonised codes + evidence
"""

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")


def run_preflight() -> bool:
    """Preflight checks for Python version, packages, required files, and MiniLM model loading."""
    ok = True
    print("Running preflight checks...")

    # 1. Python version >= 3.9
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python >= 3.9 required. Found Python {sys.version.split()[0]}")
        ok = False
    else:
        print(f"[OK] Python version: {sys.version.split()[0]}")

    # 2. Required packages
    packages = [
        ("sklearn", "scikit-learn>=1.3"),
        ("pandas", "pandas>=2.0"),
        ("sentence_transformers", "sentence-transformers>=2.2.2"),
        ("rapidfuzz", "rapidfuzz>=3.0"),
        ("nbformat", "nbformat"),
        ("nbclient", "nbclient"),
        ("ipykernel", "ipykernel"),
        ("openpyxl", "openpyxl"),
    ]
    missing_pkgs = []
    for mod, pkg_name in packages:
        try:
            __import__(mod)
        except ImportError:
            missing_pkgs.append(pkg_name)

    if missing_pkgs:
        print(f"[FAIL] Missing required packages: {', '.join(missing_pkgs)}")
        print(f"       Fix with: pip install {' '.join(missing_pkgs)}")
        ok = False
    else:
        print("[OK] All required packages installed.")

    # 3. Required files
    required_files = [
        PROJECT_ROOT / "data" / "processed" / "preprocessed_materials.csv",
        PROJECT_ROOT / "data" / "processed" / "combined_candidate_pairs_150.csv",
    ]
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        for f in missing_files:
            print(f"[FAIL] Required file missing: {f}")
        ok = False
    else:
        print("[OK] Core required data files present.")

    emb_file = PROJECT_ROOT / "models" / "candidate_embeddings.npy"
    if not emb_file.exists():
        print("[INFO] models/candidate_embeddings.npy missing (will lazily encode with MiniLM).")

    # 4. SentenceTransformer MiniLM loading
    try:
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer("all-MiniLM-L6-v2")
        print("[OK] SentenceTransformer ('all-MiniLM-L6-v2') loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load MiniLM model: {e}")
        ok = False

    if ok:
        print("[PREFLIGHT OK]")
    else:
        print("[PREFLIGHT FAILED]")
    return ok


if __name__ == "__main__":
    if "--check" in sys.argv:
        success = run_preflight()
        sys.exit(0 if success else 1)

    from src.er_pipeline import run_training  # noqa: E402

    result = run_training(verbose=True, save=True)
    m = result["metrics"]
    best = m["selected_model"]
    test = {r["model"]: r for r in m["synthetic_test"]}
    print("\n" + "=" * 78)
    print(f"SELECTED MODEL: {best}")
    print(f"Held-out synthetic test  : P={test[best]['precision']:.4f} R={test[best]['recall']:.4f} F1={test[best]['f1']:.4f} PR-AUC={test[best]['pr_auc']:.4f}")
    print(f"Exact-match baseline     : P={test['exact_match_baseline']['precision']:.4f} R={test['exact_match_baseline']['recall']:.4f} F1={test['exact_match_baseline']['f1']:.4f}")
    g = m["gold"]["selected_model_binary"]
    print(f"Review-set labels (AI-assisted, n={m['gold']['n_binary']}): TP={g['true_positives']} FP={g['false_positives']} TN={g['true_negatives']} FN={g['false_negatives']}")
    print(f"UNCERTAIN pairs auto-merged by model: {m['gold']['uncertain_pairs_auto_merged']} / {m['gold']['n_uncertain']}")
    if m.get("ai_assisted_batch2"):
        b2 = m["ai_assisted_batch2"]["selected_model_binary"]
        print(f"AI-assisted Batch-2 tier (n={m['ai_assisted_batch2']['n_binary']}): TP={b2['true_positives']} FP={b2['false_positives']} TN={b2['true_negatives']} FN={b2['false_negatives']}")
        print(f"Batch-2 UNCERTAIN auto-merged: {m['ai_assisted_batch2']['uncertain_pairs_auto_merged']} / {m['ai_assisted_batch2']['n_uncertain']}")
    print("=" * 78)
