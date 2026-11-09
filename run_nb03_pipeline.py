"""
Notebook 03 Pipeline Execution Script.

Generates:
1. results/matching/exact_matches.csv
2. results/matching/tfidf_candidate_pairs.csv
3. data/processed/labelled_pairs_template.csv
4. results/reports/baseline_matching_summary.json
5. results/figures/03_matches_vs_candidates_comparison.png
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, DATA_LABELLED_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR
from src.baseline import find_exact_matches
from src.candidate_generation import generate_tfidf_candidates
from src.feature_extraction import check_attribute_conflicts

# Setup output directories
MATCHING_DIR = RESULTS_DIR / "matching"
MATCHING_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load and Validate Input Dataset
# ----------------------------------------------------------------------
print("=" * 70)
print("[Step 1] Loading preprocessed candidate pool...")
cand_csv = DATA_PROCESSED_DIR / "preprocessed_materials.csv"
df = pd.read_csv(cand_csv)
print(f"Loaded {len(df):,} records, {len(df.columns)} columns from {cand_csv}")

assert len(df) == 6634, f"Expected 6,634 records, found {len(df)}"
assert df["corpus_id"].nunique() == len(df), "Non-unique corpus IDs detected!"
assert "cleaned_description" in df.columns, "Missing cleaned_description column!"
assert "organization" in df.columns, "Missing organization column!"
assert "item_type_hint" in df.columns, "Missing item_type_hint column!"

# ----------------------------------------------------------------------
# 2. Exact String Matching Baseline
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[Step 2] Executing exact string matching baseline...")
t0 = time.time()
df_exact = find_exact_matches(df)
print(f"Exact matching completed in {time.time() - t0:.2f}s.")
print(f"Total exact match pairs: {len(df_exact):,}")
print("Breakdown by enterprise relationship:")
print(df_exact["is_cross_enterprise"].value_counts().to_string())

exact_csv = MATCHING_DIR / "exact_matches.csv"
df_exact.to_csv(exact_csv, index=False)
print(f"Saved exact matches to: {exact_csv}")

# ----------------------------------------------------------------------
# 3. TF-IDF Candidate Retrieval
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[Step 3] Executing TF-IDF candidate retrieval (ngram_range=(1,2), min_df=2, tau=0.35)...")
t0 = time.time()
df_tfidf = generate_tfidf_candidates(
    df,
    text_col="cleaned_description",
    min_similarity=0.35,
    ngram_range=(1, 2),
    min_df=2,
    batch_size=500,
    block_by_hint=False,
    cross_enterprise_only=False,
)
print(f"TF-IDF retrieval completed in {time.time() - t0:.2f}s.")
print(f"Total TF-IDF candidate pairs (tau >= 0.35): {len(df_tfidf):,}")
print("Breakdown by enterprise relationship:")
print(df_tfidf["is_cross_enterprise"].value_counts().to_string())

# ----------------------------------------------------------------------
# 4. Engineering Attribute Compatibility Checks
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[Step 4] Performing attribute compatibility and conflict checks...")

# Build attribute lookup dictionary
attr_lookup = {}
for _, row in df.iterrows():
    attr_lookup[row["corpus_id"]] = {
        "metric_thread": row["attr_metric_thread"] if pd.notnull(row["attr_metric_thread"]) else None,
        "diameter": row["attr_diameter"] if pd.notnull(row["attr_diameter"]) else None,
        "material_grade": row["attr_material_grade"] if pd.notnull(row["attr_material_grade"]) else None,
        "pressure_class": row["attr_pressure_class"] if pd.notnull(row["attr_pressure_class"]) else None,
        "schedule": row["attr_schedule"] if pd.notnull(row["attr_schedule"]) else None,
    }

has_conflicts = []
conflict_reasons = []
confirmation_reasons = []

for _, row in df_tfidf.iterrows():
    a1 = attr_lookup.get(row["corpus_id_1"], {})
    a2 = attr_lookup.get(row["corpus_id_2"], {})
    audit = check_attribute_conflicts(a1, a2)
    has_conflicts.append(audit["has_conflict"])
    conflict_reasons.append("; ".join(audit["conflicts"]) if audit["conflicts"] else "")
    confirmation_reasons.append("; ".join(audit["confirmations"]) if audit["confirmations"] else "")

df_tfidf["has_attribute_conflict"] = has_conflicts
df_tfidf["conflict_details"] = conflict_reasons
df_tfidf["confirmation_details"] = confirmation_reasons

conflict_count = sum(has_conflicts)
print(f"Candidate pairs with explicit attribute conflicts: {conflict_count:,} ({conflict_count / len(df_tfidf) * 100:.2f}%)")
print(f"Candidate pairs without attribute conflicts: {len(df_tfidf) - conflict_count:,}")

tfidf_csv = MATCHING_DIR / "tfidf_candidate_pairs.csv"
df_tfidf.to_csv(tfidf_csv, index=False)
print(f"Saved TF-IDF candidate pairs to: {tfidf_csv}")

# ----------------------------------------------------------------------
# 5. Labelled Pair Template Creation (Academic Integrity Check)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[Step 5] Checking ground-truth label availability and preparing audit template...")

# Verify data/labelled/
labelled_files = list(DATA_LABELLED_DIR.glob("*.csv"))
has_genuine_labels = len(labelled_files) > 0
print(f"Genuine labelled files in {DATA_LABELLED_DIR}: {len(labelled_files)}")
if not has_genuine_labels:
    print("STATUS: No pre-existing labelled pairs found. Creating structured human-reviewer template.")

# Sample representative pairs for the review template:
# 15 exact cross-enterprise, 15 high-sim TF-IDF, 10 conflict pairs, 10 boundary TF-IDF pairs
template_rows = []

# 1. Exact cross-enterprise matches
exact_cross = df_exact[df_exact["is_cross_enterprise"]].copy()
for _, r in exact_cross.iterrows():
    template_rows.append({
        "pair_id": f"PAIR_EXACT_{len(template_rows) + 1:03d}",
        "corpus_id_1": r["corpus_id_1"],
        "organization_1": r["organization_1"],
        "description_1": r["matching_description"],
        "corpus_id_2": r["corpus_id_2"],
        "organization_2": r["organization_2"],
        "description_2": r["matching_description"],
        "retrieval_method": "exact_match",
        "similarity_score": 1.0,
        "has_attribute_conflict": False,
        "conflict_details": "",
        "reviewer_decision": "PENDING_HUMAN_REVIEW",  # MUST NOT FABRICATE
        "confidence_rating": "HIGH",
        "evidence_notes": "Verbatim normalized description match across CPSEs.",
    })

# 2. High similarity TF-IDF cross-enterprise pairs
tfidf_cross_high = df_tfidf[df_tfidf["is_cross_enterprise"] & (df_tfidf["tfidf_similarity"] >= 0.70)].head(15)
for _, r in tfidf_cross_high.iterrows():
    template_rows.append({
        "pair_id": f"PAIR_TFIDF_HIGH_{len(template_rows) + 1:03d}",
        "corpus_id_1": r["corpus_id_1"],
        "organization_1": r["organization_1"],
        "description_1": r["cleaned_description_1"],
        "corpus_id_2": r["corpus_id_2"],
        "organization_2": r["organization_2"],
        "description_2": r["cleaned_description_2"],
        "retrieval_method": "tfidf_cosine",
        "similarity_score": r["tfidf_similarity"],
        "has_attribute_conflict": r["has_attribute_conflict"],
        "conflict_details": r["conflict_details"],
        "reviewer_decision": "PENDING_HUMAN_REVIEW",
        "confidence_rating": "MEDIUM_HIGH",
        "evidence_notes": f"TF-IDF cosine similarity {r['tfidf_similarity']}.",
    })

# 3. Conflict pairs (high lexical similarity but engineering conflict)
conflict_pairs = df_tfidf[df_tfidf["has_attribute_conflict"] & (df_tfidf["tfidf_similarity"] >= 0.50)].head(10)
for _, r in conflict_pairs.iterrows():
    template_rows.append({
        "pair_id": f"PAIR_CONFLICT_{len(template_rows) + 1:03d}",
        "corpus_id_1": r["corpus_id_1"],
        "organization_1": r["organization_1"],
        "description_1": r["cleaned_description_1"],
        "corpus_id_2": r["corpus_id_2"],
        "organization_2": r["organization_2"],
        "description_2": r["cleaned_description_2"],
        "retrieval_method": "tfidf_cosine",
        "similarity_score": r["tfidf_similarity"],
        "has_attribute_conflict": True,
        "conflict_details": r["conflict_details"],
        "reviewer_decision": "PENDING_HUMAN_REVIEW",
        "confidence_rating": "LOW_CONFLICT",
        "evidence_notes": f"High lexical overlap but engineering conflict: {r['conflict_details']}",
    })

# 4. Moderate similarity boundary pairs (tau 0.35 - 0.45)
boundary_pairs = df_tfidf[df_tfidf["is_cross_enterprise"] & (df_tfidf["tfidf_similarity"].between(0.35, 0.45))].head(10)
for _, r in boundary_pairs.iterrows():
    template_rows.append({
        "pair_id": f"PAIR_BOUNDARY_{len(template_rows) + 1:03d}",
        "corpus_id_1": r["corpus_id_1"],
        "organization_1": r["organization_1"],
        "description_1": r["cleaned_description_1"],
        "corpus_id_2": r["corpus_id_2"],
        "organization_2": r["organization_2"],
        "description_2": r["cleaned_description_2"],
        "retrieval_method": "tfidf_cosine",
        "similarity_score": r["tfidf_similarity"],
        "has_attribute_conflict": r["has_attribute_conflict"],
        "conflict_details": r["conflict_details"],
        "reviewer_decision": "PENDING_HUMAN_REVIEW",
        "confidence_rating": "UNCERTAIN_BOUNDARY",
        "evidence_notes": f"Retrieval cutoff boundary pair (score: {r['tfidf_similarity']}).",
    })

df_template = pd.DataFrame(template_rows)
template_csv = DATA_PROCESSED_DIR / "labelled_pairs_template.csv"
df_template.to_csv(template_csv, index=False)
print(f"Generated structured review template: {template_csv} ({len(df_template)} pairs)")

# ----------------------------------------------------------------------
# 6. Diagnostic Visualisation
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[Step 6] Generating comparison visualisation...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Exact matches vs TF-IDF candidates (total vs cross-enterprise)
labels = ["Exact Matching", "TF-IDF (tau >= 0.35)"]
total_counts = [len(df_exact), len(df_tfidf)]
cross_counts = [int(df_exact["is_cross_enterprise"].sum()), int(df_tfidf["is_cross_enterprise"].sum())]

x = np.arange(len(labels))
width = 0.35

rects1 = ax1.bar(x - width/2, total_counts, width, label="Total Pairs", color="#1f77b4")
rects2 = ax1.bar(x + width/2, cross_counts, width, label="Cross-Enterprise Pairs", color="#ff7f0e")

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f"{h:,}", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f"{h:,}", xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=9)

ax1.set_ylabel("Pair Count", fontsize=11)
ax1.set_title("Exact String Matches vs TF-IDF Candidate Pairs", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.legend(fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
ax1.set_ylim(0, max(total_counts) * 1.15)
ax1.spines[["top", "right"]].set_visible(False)

# Plot 2: Cosine Similarity Distribution for TF-IDF Candidates
sims = df_tfidf["tfidf_similarity"]
ax2.hist(sims, bins=30, color="#2ca02c", edgecolor="white", alpha=0.85)
ax2.axvline(0.35, color="red", linestyle="--", linewidth=1.5, label="Cutoff tau = 0.35")
ax2.axvline(float(sims.mean()), color="blue", linestyle=":", linewidth=1.5, label=f"Mean = {sims.mean():.2f}")
ax2.set_xlabel("TF-IDF Cosine Similarity", fontsize=11)
ax2.set_ylabel("Candidate Frequency", fontsize=11)
ax2.set_title("Candidate Pair Similarity Distribution", fontsize=12, fontweight="bold")
ax2.legend(fontsize=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig_path = FIGURES_DIR / "03_matches_vs_candidates_comparison.png"
plt.savefig(fig_path, dpi=180)
plt.close()
print(f"Saved diagnostic chart to: {fig_path}")

# ----------------------------------------------------------------------
# 7. Export Baseline Summary Report JSON
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("[Step 7] Exporting baseline matching summary report...")

summary_dict = {
    "notebook": "03_baseline_matching.ipynb",
    "dataset": "data/processed/preprocessed_materials.csv",
    "candidate_pool_records": int(len(df)),
    "exact_matching": {
        "total_exact_match_pairs": int(len(df_exact)),
        "intra_enterprise_pairs": int((~df_exact["is_cross_enterprise"]).sum()),
        "cross_enterprise_pairs": int(df_exact["is_cross_enterprise"].sum()),
        "unique_repeated_descriptions": int(df_exact["matching_description"].nunique()),
    },
    "tfidf_candidate_retrieval": {
        "similarity_threshold": 0.35,
        "ngram_range": [1, 2],
        "min_df": 2,
        "total_candidate_pairs": int(len(df_tfidf)),
        "intra_enterprise_pairs": int((~df_tfidf["is_cross_enterprise"]).sum()),
        "cross_enterprise_pairs": int(df_tfidf["is_cross_enterprise"].sum()),
        "pairs_with_attribute_conflicts": int(conflict_count),
        "attribute_conflict_rate_pct": round(float(conflict_count / len(df_tfidf) * 100), 2),
    },
    "ground_truth_labels": {
        "genuine_labels_present": bool(has_genuine_labels),
        "status": "PENDING_HUMAN_REVIEW",
        "template_path": str(template_csv),
        "template_sample_size": int(len(df_template)),
        "note": "Precision, recall, and F1 calculation requires genuine human reviewer labels. Academic integrity rule adhered to: no fabricated ground-truth.",
    },
    "output_files": {
        "exact_matches_csv": str(exact_csv),
        "tfidf_candidate_pairs_csv": str(tfidf_csv),
        "labelled_pairs_template_csv": str(template_csv),
        "comparison_figure": str(fig_path),
    }
}

summary_json_path = REPORTS_DIR / "baseline_matching_summary.json"
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(summary_dict, f, indent=2)
print(f"Exported summary JSON to: {summary_json_path}")

print("\n" + "=" * 70)
print("NOTEBOOK 03 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 70)

