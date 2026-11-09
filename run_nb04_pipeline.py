"""
Pipeline Script for Notebook 04: Semantic Matching Using Sentence Transformers.

Processes:
1. Ingests preprocessed candidate materials (6,634 records) and TF-IDF candidate pairs (22,464 pairs).
2. Generates 384-dimensional dense semantic embeddings using sentence-transformers/all-MiniLM-L6-v2.
3. Saves models/candidate_embeddings.npy and models/embedding_id_mapping.csv.
4. Computes semantic cosine similarity for all 22,464 candidate pairs.
5. Performs deep cross-enterprise similarity analysis and threshold retention evaluation.
6. Exports results/matching/semantic_candidate_pairs.csv, results/reports/semantic_matching_summary.json,
   and results/figures/04_semantic_vs_lexical_similarity.png.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
)
from src.feature_extraction import (
    generate_embeddings,
    compute_pair_semantic_similarity,
    save_embeddings,
    load_embeddings,
)


def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("CSE 276 — NOTEBOOK 04: SEMANTIC MATCHING PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Input Validation
    # -------------------------------------------------------------------------
    materials_path = DATA_PROCESSED_DIR / "preprocessed_materials.csv"
    candidates_path = RESULTS_DIR / "matching" / "tfidf_candidate_pairs.csv"

    if not materials_path.exists():
        raise FileNotFoundError(f"Missing input materials: {materials_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidate pairs: {candidates_path}")

    print(f"[INFO] Ingesting candidate materials from {materials_path}...")
    df_materials = pd.read_csv(materials_path)
    print(f"[OK] Loaded {len(df_materials):,} candidate records.")

    assert len(df_materials) == 6634, f"Expected 6,634 materials, found {len(df_materials)}"
    assert "corpus_id" in df_materials.columns, "corpus_id column missing"
    assert "cleaned_description" in df_materials.columns, "cleaned_description column missing"
    assert df_materials["corpus_id"].is_unique, "corpus_id values are not unique!"

    print(f"[INFO] Ingesting TF-IDF candidate pairs from {candidates_path}...")
    df_candidates = pd.read_csv(candidates_path)
    print(f"[OK] Loaded {len(df_candidates):,} candidate pairs.")

    assert len(df_candidates) == 22464, f"Expected 22,464 pairs, found {len(df_candidates)}"
    cross_pairs_count = int(df_candidates["is_cross_enterprise"].sum())
    print(f"[OK] Verified cross-enterprise candidate pairs: {cross_pairs_count:,}")
    assert cross_pairs_count == 930, f"Expected 930 cross-enterprise pairs, found {cross_pairs_count}"

    # -------------------------------------------------------------------------
    # 2. Embedding Generation / Loading
    # -------------------------------------------------------------------------
    embeddings_file = MODELS_DIR / "candidate_embeddings.npy"
    mapping_file = MODELS_DIR / "embedding_id_mapping.csv"

    descriptions = df_materials["cleaned_description"].fillna("").astype(str).tolist()
    corpus_ids = df_materials["corpus_id"].astype(str).tolist()

    force_recompute = "--force-recompute" in sys.argv
    if embeddings_file.exists() and mapping_file.exists() and not force_recompute:
        print(f"[INFO] Found existing precomputed embeddings at {embeddings_file}.")
        print("[INFO] Loading cached embeddings for rapid execution (pass --force-recompute to re-encode)...")
        embeddings, id_to_idx = load_embeddings(embeddings_file, mapping_file)
        embed_duration = 0.05
    else:
        print(f"[INFO] Encoding {len(descriptions):,} descriptions with {DEFAULT_EMBEDDING_MODEL}...")
        embed_start = time.time()
        embeddings = generate_embeddings(
            texts=descriptions,
            model_name=DEFAULT_EMBEDDING_MODEL,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embed_duration = time.time() - embed_start
        print(f"[OK] Generated embeddings in {embed_duration:.2f} seconds.")
        # Serialize embeddings & mapping
        save_embeddings(embeddings, corpus_ids, embeddings_file, mapping_file)
        print(f"[OK] Serialized embeddings to: {embeddings_file}")
        print(f"[OK] Serialized ID mapping to: {mapping_file}")

    # Validate embeddings
    assert embeddings.shape == (6634, 384), f"Unexpected embedding shape: {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
    assert not np.isnan(embeddings).any(), "Found NaN values in embeddings!"
    assert not np.isinf(embeddings).any(), "Found Inf values in embeddings!"

    # Validate L2 normalization: norms should be ~1.0
    norms = np.linalg.norm(embeddings, axis=1)
    valid_norms = norms[norms > 1e-6]
    assert np.allclose(valid_norms, 1.0, atol=1e-4), "Embeddings are not properly normalized!"
    print(f"[OK] Embeddings validated: shape={embeddings.shape}, dtype={embeddings.dtype}, normalized=True, NaNs=0, Infs=0")

    # Test reloading to verify serialization integrity
    reloaded_emb, id_to_idx = load_embeddings(embeddings_file, mapping_file)
    assert np.array_equal(embeddings, reloaded_emb), "Reloaded embeddings do not match original matrix!"
    assert len(id_to_idx) == 6634, f"Expected 6,634 mapped IDs, found {len(id_to_idx)}"
    assert id_to_idx[corpus_ids[0]] == 0, "ID mapping misalignment at index 0!"
    print("[OK] Reload integrity verification passed: 100% exact numerical match.")

    # -------------------------------------------------------------------------
    # 3. Compute Pairwise Semantic Similarity
    # -------------------------------------------------------------------------
    print(f"[INFO] Computing pairwise semantic cosine similarity for {len(df_candidates):,} candidate pairs...")
    sim_start = time.time()
    semantic_sims = compute_pair_semantic_similarity(
        candidate_df=df_candidates,
        embeddings=reloaded_emb,
        id_to_idx=id_to_idx,
        id_col_1="corpus_id_1",
        id_col_2="corpus_id_2",
    )
    sim_duration = time.time() - sim_start
    print(f"[OK] Computed {len(semantic_sims):,} pairwise similarities in {sim_duration:.4f}s.")

    # Validate similarity scores
    assert len(semantic_sims) == len(df_candidates), "Length mismatch in computed similarities!"
    assert not np.isnan(semantic_sims).any(), "Found NaNs in computed similarities!"
    assert np.all(semantic_sims >= -1.0) and np.all(semantic_sims <= 1.0), "Similarity score out of [-1.0, 1.0] range!"

    # -------------------------------------------------------------------------
    # 4. Construct Augmented Candidate DataFrame
    # -------------------------------------------------------------------------
    df_augmented = df_candidates.copy()
    df_augmented["semantic_cosine_similarity"] = semantic_sims

    # Add required aliases and columns
    df_augmented["lexical_similarity"] = df_augmented["tfidf_similarity"]
    df_augmented["enterprise_1"] = df_augmented["organization_1"]
    df_augmented["enterprise_2"] = df_augmented["organization_2"]
    df_augmented["attribute_conflict"] = df_augmented["has_attribute_conflict"]

    # Reorder columns for optimal readability while preserving all original columns
    primary_cols = [
        "corpus_id_1",
        "enterprise_1",
        "organization_1",
        "cleaned_description_1",
        "corpus_id_2",
        "enterprise_2",
        "organization_2",
        "cleaned_description_2",
        "semantic_cosine_similarity",
        "tfidf_similarity",
        "lexical_similarity",
        "is_cross_enterprise",
        "attribute_conflict",
        "has_attribute_conflict",
        "conflict_details",
        "confirmation_details",
        "record_id_1",
        "record_id_2",
    ]
    df_augmented = df_augmented[primary_cols]

    output_csv = RESULTS_DIR / "matching" / "semantic_candidate_pairs.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_augmented.to_csv(output_csv, index=False)
    print(f"[OK] Exported augmented candidate pairs to: {output_csv}")

    # -------------------------------------------------------------------------
    # 5. Cross-Enterprise & Statistical Analysis
    # -------------------------------------------------------------------------
    df_cross = df_augmented[df_augmented["is_cross_enterprise"]].copy()
    df_intra = df_augmented[~df_augmented["is_cross_enterprise"]].copy()

    all_sims = df_augmented["semantic_cosine_similarity"]
    cross_sims = df_cross["semantic_cosine_similarity"]
    intra_sims = df_intra["semantic_cosine_similarity"]

    def compute_stats(series):
        return {
            "count": int(len(series)),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "p25": round(float(series.quantile(0.25)), 4),
            "median": round(float(series.median()), 4),
            "p75": round(float(series.quantile(0.75)), 4),
            "p90": round(float(series.quantile(0.90)), 4),
            "p95": round(float(series.quantile(0.95)), 4),
            "p99": round(float(series.quantile(0.99)), 4),
            "max": round(float(series.max()), 4),
        }

    stats_all = compute_stats(all_sims)
    stats_cross = compute_stats(cross_sims)
    stats_intra = compute_stats(intra_sims)

    corr_all = round(float(df_augmented["tfidf_similarity"].corr(df_augmented["semantic_cosine_similarity"])), 4)
    corr_cross = round(float(df_cross["tfidf_similarity"].corr(df_cross["semantic_cosine_similarity"])), 4)

    print("\n--- Similarity Statistics ---")
    print(f"Overall (22,464 pairs): Mean={stats_all['mean']}, Median={stats_all['median']}, Min={stats_all['min']}, Max={stats_all['max']}")
    print(f"Cross-Enterprise (930 pairs): Mean={stats_cross['mean']}, Median={stats_cross['median']}, Min={stats_cross['min']}, Max={stats_cross['max']}")
    print(f"Correlation (Lexical vs Semantic) -> Overall: {corr_all}, Cross-Enterprise: {corr_cross}")

    # -------------------------------------------------------------------------
    # 6. Exploratory Threshold Analysis
    # -------------------------------------------------------------------------
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    threshold_results = []

    print("\n--- Exploratory Threshold Analysis (Candidate Retention) ---")
    print("Threshold | Total Retained | % Total | Cross Retained | % Cross")
    print("-" * 65)
    for t in thresholds:
        tot_count = int((df_augmented["semantic_cosine_similarity"] >= t).sum())
        tot_pct = round(tot_count / len(df_augmented) * 100, 2)
        crs_count = int((df_cross["semantic_cosine_similarity"] >= t).sum())
        crs_pct = round(crs_count / len(df_cross) * 100, 2)
        threshold_results.append({
            "threshold": t,
            "total_retained": tot_count,
            "total_retention_pct": tot_pct,
            "cross_enterprise_retained": crs_count,
            "cross_enterprise_retention_pct": crs_pct,
        })
        print(f"  {t:0.2f}    |     {tot_count:5d}      |  {tot_pct:5.2f}% |     {crs_count:5d}      | {crs_pct:5.2f}%")

    # -------------------------------------------------------------------------
    # 7. Exemplary Pairs Extraction
    # -------------------------------------------------------------------------
    # A. High Semantic (>0.75) but Lower Lexical (<0.50) in Cross-Enterprise (Synonym discovery)
    synonym_candidates = df_cross[
        (df_cross["semantic_cosine_similarity"] >= 0.75) &
        (df_cross["tfidf_similarity"] < 0.50) &
        (~df_cross["attribute_conflict"])
    ].sort_values(by="semantic_cosine_similarity", ascending=False).head(5)

    # B. High Semantic (>=0.80) Cross-Enterprise Matches
    high_sem_cross = df_cross[
        (df_cross["semantic_cosine_similarity"] >= 0.80) &
        (~df_cross["attribute_conflict"])
    ].sort_values(by="semantic_cosine_similarity", ascending=False).head(5)

    # C. Cross-Enterprise Pairs with Physical Attribute Conflicts
    conflict_cross = df_cross[
        df_cross["attribute_conflict"]
    ].sort_values(by="semantic_cosine_similarity", ascending=False)

    # D. Top Overall Pairs with High Semantic Similarity (>=0.90) BUT Critical Physical Conflicts
    conflict_overall = df_augmented[
        (df_augmented["semantic_cosine_similarity"] >= 0.90) &
        (df_augmented["attribute_conflict"])
    ].sort_values(by="semantic_cosine_similarity", ascending=False).head(5)

    def extract_pair_dict(df_sub):
        out = []
        for _, r in df_sub.iterrows():
            out.append({
                "corpus_id_1": str(r["corpus_id_1"]),
                "enterprise_1": str(r["enterprise_1"]),
                "description_1": str(r["cleaned_description_1"]),
                "corpus_id_2": str(r["corpus_id_2"]),
                "enterprise_2": str(r["enterprise_2"]),
                "description_2": str(r["cleaned_description_2"]),
                "semantic_similarity": float(r["semantic_cosine_similarity"]),
                "lexical_similarity": float(r["tfidf_similarity"]),
                "attribute_conflict": bool(r["attribute_conflict"]),
                "conflict_details": str(r["conflict_details"]) if pd.notna(r["conflict_details"]) else "",
            })
        return out

    examples_synonyms = extract_pair_dict(synonym_candidates)
    examples_high_sem = extract_pair_dict(high_sem_cross)
    examples_conflicts_cross = extract_pair_dict(conflict_cross)
    examples_conflicts_overall = extract_pair_dict(conflict_overall)

    print(f"\n[INFO] Identified {len(examples_synonyms)} synonym candidate pairs (High Sem / Lower Lexical).")
    print(f"[INFO] Identified {len(examples_conflicts_cross)} cross-enterprise attribute conflict pairs.")
    print(f"[INFO] Identified {len(examples_conflicts_overall)} high-semantic attribute conflict pairs overall.")

    # -------------------------------------------------------------------------
    # 8. Export Summary Report (JSON)
    # -------------------------------------------------------------------------
    summary_report = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_configuration": {
            "model_name": DEFAULT_EMBEDDING_MODEL,
            "embedding_dimension": 384,
            "normalized": True,
            "pooling": "mean",
            "batch_size": 64,
            "device": "cpu",
            "mode": "inference_only (no fine-tuning)",
        },
        "dataset_validation": {
            "candidate_materials_count": len(df_materials),
            "candidate_pairs_count": len(df_candidates),
            "cross_enterprise_pairs_count": cross_pairs_count,
            "intra_enterprise_pairs_count": len(df_candidates) - cross_pairs_count,
            "embedding_matrix_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "has_nan": bool(np.isnan(embeddings).any()),
            "has_inf": bool(np.isinf(embeddings).any()),
        },
        "similarity_statistics": {
            "all_pairs": stats_all,
            "cross_enterprise": stats_cross,
            "intra_enterprise": stats_intra,
            "correlation_lexical_semantic": {
                "overall": corr_all,
                "cross_enterprise": corr_cross,
            },
        },
        "exploratory_threshold_retention": threshold_results,
        "conflict_analysis": {
            "total_conflicts_in_candidates": int(df_augmented["attribute_conflict"].sum()),
            "cross_enterprise_conflicts": int(df_cross["attribute_conflict"].sum()),
            "cross_enterprise_conflict_rate_pct": round(float(df_cross["attribute_conflict"].sum() / len(df_cross) * 100), 2),
        },
        "exemplary_cross_enterprise_pairs": {
            "high_semantic_low_lexical_synonyms": examples_synonyms,
            "high_semantic_consistent_matches": examples_high_sem,
            "cross_enterprise_with_attribute_conflicts": examples_conflicts_cross,
            "top_overall_high_semantic_attribute_conflicts": examples_conflicts_overall,
        },
        "methodological_notes": [
            "Evaluated strictly on candidate pairs retrieved via TF-IDF blocking (tau >= 0.35).",
            "Exhaustive pairwise computation (6,634 x 6,634 / 2 = ~22M pairs) was deliberately avoided for O(N log N) scaling.",
            "Threshold retention statistics represent candidate retention rates, NOT precision or recall.",
            "No ground-truth labels or supervised classifiers were trained in this notebook.",
        ],
    }

    report_path = REPORTS_DIR / "semantic_matching_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[OK] Exported semantic matching summary to: {report_path}")

    # -------------------------------------------------------------------------
    # 9. Visualization
    # -------------------------------------------------------------------------
    figure_path = FIGURES_DIR / "04_semantic_vs_lexical_similarity.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # Subplot 1: Scatter plot Lexical vs Semantic (All Pairs)
    ax1 = axes[0, 0]
    ax1.scatter(
        df_intra["tfidf_similarity"],
        df_intra["semantic_cosine_similarity"],
        alpha=0.15,
        s=12,
        color="#4a7bb0",
        label=f"Intra-Enterprise (N={len(df_intra):,})",
    )
    ax1.scatter(
        df_cross["tfidf_similarity"],
        df_cross["semantic_cosine_similarity"],
        alpha=0.55,
        s=20,
        color="#e65c00",
        label=f"Cross-Enterprise (N={len(df_cross):,})",
    )
    ax1.set_xlabel("Lexical Similarity (TF-IDF Cosine)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Semantic Cosine Similarity", fontsize=11, fontweight="bold")
    ax1.set_title("A. Lexical vs Semantic Similarity (Candidate Pairs)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Cross-Enterprise Semantic vs Lexical with Attribute Conflicts
    ax2 = axes[0, 1]
    conflict_mask = df_cross["attribute_conflict"]
    ax2.scatter(
        df_cross.loc[~conflict_mask, "tfidf_similarity"],
        df_cross.loc[~conflict_mask, "semantic_cosine_similarity"],
        alpha=0.6,
        s=25,
        color="#2b8cbe",
        label=f"Compatible Spec (N={(~conflict_mask).sum():,})",
    )
    ax2.scatter(
        df_cross.loc[conflict_mask, "tfidf_similarity"],
        df_cross.loc[conflict_mask, "semantic_cosine_similarity"],
        alpha=0.8,
        s=35,
        color="#de2d26",
        marker="x",
        label=f"Attribute Conflict (N={conflict_mask.sum():,})",
    )
    # Highlight quadrant: High Semantic (>0.70), Lower Lexical (<0.50)
    ax2.axvline(x=0.50, color="gray", linestyle=":", alpha=0.7)
    ax2.axhline(y=0.70, color="gray", linestyle=":", alpha=0.7)
    ax2.text(0.36, 0.88, "Synonym\nDiscovery\nZone", fontsize=9, color="#6a3d9a", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0e6ff", edgecolor="#9970ab", alpha=0.8))
    ax2.set_xlabel("Lexical Similarity (TF-IDF Cosine)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Semantic Cosine Similarity", fontsize=11, fontweight="bold")
    ax2.set_title("B. Cross-Enterprise Pairs & Physical Compatibility", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Distribution Comparison (KDE / Hist)
    ax3 = axes[1, 0]
    sns.kdeplot(df_intra["semantic_cosine_similarity"], ax=ax3, color="#4a7bb0", fill=True, alpha=0.3, label="Intra-Enterprise")
    sns.kdeplot(df_cross["semantic_cosine_similarity"], ax=ax3, color="#e65c00", fill=True, alpha=0.3, label="Cross-Enterprise")
    ax3.axvline(x=stats_all["mean"], color="#4a7bb0", linestyle="--", label=f"Intra Mean ({stats_intra['mean']:.2f})")
    ax3.axvline(x=stats_cross["mean"], color="#e65c00", linestyle="--", label=f"Cross Mean ({stats_cross['mean']:.2f})")
    ax3.set_xlabel("Semantic Cosine Similarity", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax3.set_title("C. Semantic Similarity Distribution Density", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Candidate Retention Curve across Thresholds
    ax4 = axes[1, 1]
    thresh_x = [r["threshold"] for r in threshold_results]
    ret_tot = [r["total_retention_pct"] for r in threshold_results]
    ret_crs = [r["cross_enterprise_retention_pct"] for r in threshold_results]
    ax4.plot(thresh_x, ret_tot, marker="o", color="#2b8cbe", linewidth=2.2, label="Total Pairs Retention %")
    ax4.plot(thresh_x, ret_crs, marker="s", color="#e65c00", linewidth=2.2, label="Cross-Enterprise Retention %")
    for x, yt, yc in zip(thresh_x, ret_tot, ret_crs):
        ax4.annotate(f"{yt:.1f}%", (x, yt), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color="#2b8cbe")
        ax4.annotate(f"{yc:.1f}%", (x, yc), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8.5, color="#e65c00")
    ax4.set_xlabel("Semantic Cosine Similarity Cutoff (τ_sem)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Candidate Retention Rate (%)", fontsize=11, fontweight="bold")
    ax4.set_title("D. Exploratory Candidate Retention Rates", fontsize=12, fontweight="bold")
    ax4.set_ylim(-5, 105)
    ax4.legend(loc="upper right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("CSE 276 — Notebook 04: Semantic Similarity Analysis\nModel: sentence-transformers/all-MiniLM-L6-v2 (384-d)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Exported diagnostic figure to: {figure_path}")

    total_time = time.time() - start_time
    print("=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
