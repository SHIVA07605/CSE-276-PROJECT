"""
Notebook 02 Pipeline Execution Script.

Runs the full preprocessing pipeline on the corpus and generates:
  - data/processed/preprocessed_materials.csv
  - data/processed/preprocessed_materials.parquet  (if pyarrow available)
  - data/processed/held_back_records.csv
  - results/reports/preprocessing_summary.json
  - results/figures/02_*.png  (diagnostic charts)

Usage:
    cd "c:/project ai"
    python run_nb02_pipeline.py
"""

import json
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR
from src.preprocessing import (
    partition_corpus,
    validate_partitions,
    preprocess_catalogue,
    EXPECTED_COUNTS,
)

# ---------------------------------------------------------------------------
# 0. Setup output directories
# ---------------------------------------------------------------------------
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load corpus
# ---------------------------------------------------------------------------
corpus_path = DATA_RAW_DIR / "material_description_corpus.csv"
print(f"Loading corpus from: {corpus_path}")
df_raw = pd.read_csv(corpus_path)
print(f"  Loaded {len(df_raw):,} records, {len(df_raw.columns)} columns")
print(f"  Columns: {list(df_raw.columns)}")

assert len(df_raw) == 21513, f"Expected 21,513 records, got {len(df_raw)}"

# ---------------------------------------------------------------------------
# 2. Partition corpus
# ---------------------------------------------------------------------------
print("\n[Step 2] Partitioning corpus...")
partitions = partition_corpus(df_raw)
validation = validate_partitions(df_raw, partitions, strict=True)

print(f"  Tier 1 discrete     : {len(partitions['tier1_discrete']):>6,}")
print(f"  Tier 2 tagged       : {len(partitions['tier2_tagged']):>6,}")
print(f"  Tier 2 untagged     : {len(partitions['tier2_untagged']):>6,}")
print(f"  Tier 3 admin        : {len(partitions['tier3_administrative']):>6,}")
print(f"  Candidate pool      : {len(partitions['candidate_pool']):>6,}")
print(f"  Remaining pool      : {len(partitions['remaining_pool']):>6,}")
print(f"  Validation passed   : {validation['valid']}")

# ---------------------------------------------------------------------------
# 3. Preprocess candidate pool
# ---------------------------------------------------------------------------
print("\n[Step 3] Preprocessing candidate pool (6,634 records)...")
df_candidate = partitions["candidate_pool"].copy()
df_remaining = partitions["remaining_pool"].copy()

# Add partition_tier column before preprocessing (it's on the sub-frames already)
for key in ["tier1_discrete", "tier2_tagged"]:
    partitions[key]["partition_tier"] = (
        "Tier 1: Discrete material lines"
        if key == "tier1_discrete"
        else "Tier 2: Tagged procurement descriptions"
    )

df_candidate = preprocess_catalogue(df_candidate, text_col="description")
print(f"  Preprocessed {len(df_candidate):,} candidate records.")

# ---------------------------------------------------------------------------
# 4. Before / after examples
# ---------------------------------------------------------------------------
print("\n[Step 4] Before / After preprocessing examples:")
examples = [
    "Supply of S.S. 304 Hex Bolt 10 mm",
    "Procurement of Seamless Pipe 2 inch Sch 40",
    "CS Gate Valve 2INCH 150# PN16",
    "Consolidated Procurement of CI Gate Valves for Farakka & Darlipalli",
    "MS Plate IS2062 Grade A 10mm Thk",
    "TRUCK CHASSIS TATA SE 1613/42",
]
from src.preprocessing import normalise_text, extract_material_attributes
print(f"\n{'Original':<55} {'Normalised'}")
print("-" * 110)
for ex in examples:
    norm = normalise_text(ex)
    print(f"  {ex:<53} {norm}")

# ---------------------------------------------------------------------------
# 5. Attribute extraction coverage stats
# ---------------------------------------------------------------------------
attr_cols = [
    "attr_metric_thread", "attr_diameter", "attr_material_grade",
    "attr_pressure_class", "attr_schedule", "attr_voltage", "attr_capacity"
]
attr_labels = {
    "attr_metric_thread": "Metric Thread",
    "attr_diameter": "Diameter",
    "attr_material_grade": "Material Grade",
    "attr_pressure_class": "Pressure Class",
    "attr_schedule": "Pipe Schedule",
    "attr_voltage": "Voltage",
    "attr_capacity": "Capacity",
}

print("\n[Step 5] Attribute extraction coverage on candidate pool (6,634 records):")
coverage = {}
for col in attr_cols:
    if col in df_candidate.columns:
        n_filled = df_candidate[col].notnull().sum()
        pct = n_filled / len(df_candidate) * 100
        coverage[col] = {"count": int(n_filled), "pct": round(float(pct), 2)}
        print(f"  {attr_labels[col]:<18}: {n_filled:>5,} ({pct:5.1f}%)")

# ---------------------------------------------------------------------------
# 6. Generate diagnostic plots
# ---------------------------------------------------------------------------
print("\n[Step 6] Generating diagnostic charts...")

# --- Plot 1: Candidate distribution by tier ---
tier_counts = {
    "Tier 1\nDiscrete": len(partitions["tier1_discrete"]),
    "Tier 2\nTagged": len(partitions["tier2_tagged"]),
    "Tier 2\nUntagged": len(partitions["tier2_untagged"]),
    "Tier 3\nAdmin": len(partitions["tier3_administrative"]),
}
fig1, ax1 = plt.subplots(figsize=(8, 4.5))
bars = ax1.bar(
    tier_counts.keys(), tier_counts.values(),
    color=["#1f77b4", "#2ca02c", "#aec7e8", "#ffbb78"], edgecolor="white", width=0.55
)
for bar, val in zip(bars, tier_counts.values()):
    ax1.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
        f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold"
    )
ax1.set_title("Corpus Partition Distribution (21,513 Records)", fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Record Count", fontsize=11)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.set_ylim(0, max(tier_counts.values()) * 1.15)
ax1.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig1_path = FIGURES_DIR / "02_corpus_partition_distribution.png"
plt.savefig(fig1_path, dpi=150)
plt.close()
print(f"  Saved: {fig1_path}")

# --- Plot 2: Attribute extraction coverage ---
attr_present = [coverage.get(col, {}).get("pct", 0.0) for col in attr_cols if col in df_candidate.columns]
attr_missing = [100 - p for p in attr_present]
attr_names_plot = [attr_labels[col] for col in attr_cols if col in df_candidate.columns]

fig2, ax2 = plt.subplots(figsize=(9, 4))
x = np.arange(len(attr_names_plot))
bar_w = 0.45
b1 = ax2.bar(x - bar_w / 2, attr_present, bar_w, label="Extracted (%)", color="#2ca02c", alpha=0.85)
b2 = ax2.bar(x + bar_w / 2, attr_missing, bar_w, label="Missing (%)", color="#d62728", alpha=0.65)
for bar, val in zip(b1, attr_present):
    if val > 2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
ax2.set_xticks(x)
ax2.set_xticklabels(attr_names_plot, fontsize=9)
ax2.set_ylabel("Percentage of 6,634 records", fontsize=10)
ax2.set_title("Engineering Attribute Extraction Coverage — Candidate Pool", fontsize=12, fontweight="bold", pad=10)
ax2.legend(fontsize=9)
ax2.set_ylim(0, 115)
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig2_path = FIGURES_DIR / "02_attribute_extraction_coverage.png"
plt.savefig(fig2_path, dpi=150)
plt.close()
print(f"  Saved: {fig2_path}")

# --- Plot 3: Raw vs cleaned text length ---
raw_lens = df_candidate["raw_description"].fillna("").apply(len)
clean_lens = df_candidate["cleaned_description"].fillna("").apply(len)
fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.hist(raw_lens.clip(0, 400), bins=60, alpha=0.55, label=f"Raw (mean={raw_lens.mean():.0f} chars)", color="#d62728", density=True)
ax3.hist(clean_lens.clip(0, 400), bins=60, alpha=0.55, label=f"Cleaned (mean={clean_lens.mean():.0f} chars)", color="#1f77b4", density=True)
ax3.set_title("Character Length Distribution: Raw vs Cleaned", fontsize=12, fontweight="bold", pad=10)
ax3.set_xlabel("Character Count", fontsize=10)
ax3.set_ylabel("Density", fontsize=10)
ax3.legend(fontsize=9)
ax3.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig3_path = FIGURES_DIR / "02_raw_vs_cleaned_length.png"
plt.savefig(fig3_path, dpi=150)
plt.close()
print(f"  Saved: {fig3_path}")

# ---------------------------------------------------------------------------
# 7. Export output files
# ---------------------------------------------------------------------------
print("\n[Step 7] Exporting output files...")

# 7a. Preprocessed candidate pool
candidate_csv = DATA_PROCESSED_DIR / "preprocessed_materials.csv"
df_candidate.to_csv(candidate_csv, index=False)
print(f"  OK: {candidate_csv} ({len(df_candidate):,} records)")

# 7b. Parquet (optional)
try:
    candidate_parquet = DATA_PROCESSED_DIR / "preprocessed_materials.parquet"
    df_candidate.to_parquet(candidate_parquet, index=False)
    print(f"  OK: {candidate_parquet} (parquet)")
except Exception as e:
    print(f"  SKIP Parquet: {e}")

# 7c. Held-back records
held_csv = DATA_PROCESSED_DIR / "held_back_records.csv"
df_remaining.to_csv(held_csv, index=False)
print(f"  OK: {held_csv} ({len(df_remaining):,} records)")

# 7d. Summary JSON
summary = {
    "notebook": "02_data_preprocessing.ipynb",
    "corpus_total_records": int(len(df_raw)),
    "corpus_columns": list(df_raw.columns),
    "partition_validation": {
        "valid": bool(validation["valid"]),
        "all_counts_match": bool(validation["all_counts_match"]),
        "has_unique_ids": bool(validation["has_unique_ids"]),
        "is_mutually_exclusive": bool(validation["is_mutually_exclusive"]),
        "is_collectively_exhaustive": bool(validation["is_collectively_exhaustive"]),
    },
    "partition_counts": {k: int(v) for k, v in validation["counts"].items()},
    "expected_counts": {k: int(v) for k, v in EXPECTED_COUNTS.items()},
    "candidate_pool_records": int(len(df_candidate)),
    "held_back_records": int(len(df_remaining)),
    "attribute_coverage": {
        attr_labels.get(col, col): coverage.get(col, {})
        for col in attr_cols
        if col in df_candidate.columns
    },
    "text_length_stats": {
        "raw_mean_chars": round(float(raw_lens.mean()), 2),
        "cleaned_mean_chars": round(float(clean_lens.mean()), 2),
        "mean_reduction_chars": round(float(raw_lens.mean() - clean_lens.mean()), 2),
    },
    "output_files": {
        "candidate_csv": str(candidate_csv),
        "held_back_csv": str(held_csv),
        "figures": [str(fig1_path), str(fig2_path), str(fig3_path)],
    },
}

summary_path = REPORTS_DIR / "preprocessing_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"  OK: {summary_path}")

print("\n" + "=" * 60)
print("NOTEBOOK 02 PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
