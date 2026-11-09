"""
Integration script to combine Batch 1 and Batch 2 into a single unified 150-pair dataset.
Preserves:
- All 50 Batch 1 rows with existing human annotations intact (2 pos, 17 neg, 31 unc).
- All 100 Batch 2 rows with human labels as PENDING and AI suggestions segregated.
- Adds source references and material domains to Batch 1.
- Validates 150 unique pairs, zero duplicate pair IDs, 150 total records.
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(r"c:\project ai\data\processed")
b1_path = DATA_DIR / "improved_labelled_pairs_worksheet.csv"
b2_path = DATA_DIR / "candidate_pairs_batch2_expansion_100.csv"
corpus_path = DATA_DIR / "preprocessed_materials.csv"
out_path = DATA_DIR / "combined_candidate_pairs_150.csv"

# Load
b1 = pd.read_csv(b1_path, dtype=str)
b2 = pd.read_csv(b2_path, dtype=str)
corpus = pd.read_csv(corpus_path, dtype=str).set_index("corpus_id")

print(f"Batch 1 rows: {len(b1)}")
print(f"Batch 2 rows: {len(b2)}")

# Build source references for Batch 1
def get_source_ref(cid):
    if cid in corpus.index:
        row = corpus.loc[cid]
        t = str(row.get("tender_reference", "")) or str(row.get("tender_id", ""))
        sec = str(row.get("source_section", ""))
        doc = str(row.get("document_url", "")) or str(row.get("source_url", ""))
        return f"Tender: {t if t != 'nan' else 'nan'} | Section: {sec if sec != 'nan' else 'nan'} | Doc: {doc if doc != 'nan' else 'nan'}"
    return "Tender: nan | Section: nan | Doc: nan"

def infer_domain(text):
    t = str(text).upper()
    if any(k in t for k in ["VALVE", "COCK", "ACTUATOR"]):
        return "Valves & Flow Control"
    if any(k in t for k in ["CABLE", "TRANSFORMER", "MOTOR", "ELECTRICAL", "SWITCH", "BREAKER"]):
        return "Electrical & Power Systems"
    if any(k in t for k in ["PUMP", "HYDRAULIC"]):
        return "Pumps & Hydraulic Machinery"
    if any(k in t for k in ["COMPRESSOR", "TURBINE", "ROTATING", "BLOWER"]):
        return "Compressors, Turbines & Rotating Equipment"
    if any(k in t for k in ["TRANSMITTER", "GAUGE", "METER", "CHROMATOGRAPH", "STIRRER", "ANALYZER", "INDICATOR"]):
        return "Instrumentation & Analytical Equipment"
    if any(k in t for k in ["PIPE", "TUBE", "FLANGE", "FITTING", "ELBOW", "TEE"]):
        return "Pipes, Flanges & Fittings"
    if any(k in t for k in ["BEARING", "SEAL", "GASKET", "O-RING", "COUPLING"]):
        return "Bearings, Seals & Mechanical Spares"
    if any(k in t for k in ["STEEL", "CONVEYOR", "BELT", "PLATE", "STRUCTURAL"]):
        return "Structural Steel & Heavy Equipment"
    return "General Spares & Consumables"

# Populate missing fields in Batch 1
if "source_reference_1" not in b1.columns:
    b1["source_reference_1"] = b1["corpus_id_1"].apply(get_source_ref)
if "source_reference_2" not in b1.columns:
    b1["source_reference_2"] = b1["corpus_id_2"].apply(get_source_ref)
if "material_domain" not in b1.columns:
    b1["material_domain"] = b1["cleaned_description_1"].apply(infer_domain)

if "ai_suggested_hypothesis" not in b1.columns:
    b1["ai_suggested_hypothesis"] = None
if "ai_suggested_confidence" not in b1.columns:
    b1["ai_suggested_confidence"] = None
if "ai_engineering_rationale" not in b1.columns:
    b1["ai_engineering_rationale"] = None

# Ensure unified columns in exact order
target_columns = [
    "pair_id",
    "corpus_id_1",
    "enterprise_1",
    "raw_description_1",
    "cleaned_description_1",
    "extracted_attributes_1",
    "source_reference_1",
    "corpus_id_2",
    "enterprise_2",
    "raw_description_2",
    "cleaned_description_2",
    "extracted_attributes_2",
    "source_reference_2",
    "material_domain",
    "lexical_similarity",
    "semantic_cosine_similarity",
    "is_exact_normalized_match",
    "has_attribute_conflict",
    "conflict_details",
    "confirmation_details",
    "human_label",
    "confidence_rating",
    "reviewer_notes",
    "ai_suggested_hypothesis",
    "ai_suggested_confidence",
    "ai_engineering_rationale",
    "review_guidance",
]

b1_aligned = b1[target_columns].copy()
b2_aligned = b2[target_columns].copy()

combined = pd.concat([b1_aligned, b2_aligned], ignore_index=True)

# Validation
assert len(combined) == 150, f"Expected 150 rows, got {len(combined)}"
assert combined["pair_id"].nunique() == 150, "Duplicate pair_id found"

# Check human labels
labels = combined["human_label"].fillna("PENDING").astype(str).str.strip().str.upper()
assert (labels.iloc[:50] != "PENDING").all(), "Batch 1 labels must not be PENDING"
assert (labels.iloc[50:] == "PENDING").all(), "Batch 2 labels must all be PENDING"

print("Labels in combined dataset:")
print(labels.value_counts())

# Save combined
combined.to_csv(out_path, index=False)
print(f"Successfully saved combined 150-pair dataset to: {out_path}")

