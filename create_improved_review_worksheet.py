"""
Script to create the improved human review worksheet for Task 1.
Preserves data/processed/labelled_pairs_template.csv exactly as-is.
Generates data/processed/improved_labelled_pairs_worksheet.csv enriched with:
- Raw original descriptions for both records
- Cleaned normalized descriptions for both records
- Both enterprise names
- Lexical similarity (TF-IDF cosine)
- Semantic cosine similarity (SentenceTransformer)
- Discrete engineering attributes for both records (diameter, grade, pressure, thread, schedule)
- Attribute conflict indicators and mismatch details
- Confirmation details
- Standard human label field ('PENDING', accepting 1, 0, UNCERTAIN)
- Reviewer notes field
- Protocol reference and review guidance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import load_embeddings, compute_pair_semantic_similarity
from src.config import MODELS_DIR

def build_improved_worksheet():
    template_path = PROJECT_ROOT / "data" / "processed" / "labelled_pairs_template.csv"
    materials_path = PROJECT_ROOT / "data" / "processed" / "preprocessed_materials.csv"
    semantic_pairs_path = PROJECT_ROOT / "results" / "matching" / "semantic_candidate_pairs.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "improved_labelled_pairs_worksheet.csv"

    print("=" * 80)
    print("CREATING IMPROVED HUMAN REVIEW WORKSHEET")
    print("=" * 80)

    assert template_path.exists(), f"Missing template: {template_path}"
    assert materials_path.exists(), f"Missing materials: {materials_path}"

    df_template = pd.read_csv(template_path)
    df_mat = pd.read_csv(materials_path).set_index("corpus_id")

    # Load precomputed embeddings for precise semantic similarity lookup
    emb_file = MODELS_DIR / "candidate_embeddings.npy"
    map_file = MODELS_DIR / "embedding_id_mapping.csv"
    embeddings, id_to_idx = load_embeddings(emb_file, map_file)

    # Compute semantic cosine similarity for the 50 template pairs
    semantic_sims = compute_pair_semantic_similarity(
        df_template, embeddings, id_to_idx, id_col_1="corpus_id_1", id_col_2="corpus_id_2"
    )

    enriched_rows = []
    attr_cols = ["attr_diameter", "attr_material_grade", "attr_pressure_class", "attr_metric_thread", "attr_schedule"]

    for idx, row in df_template.iterrows():
        c1 = str(row["corpus_id_1"])
        c2 = str(row["corpus_id_2"])

        mat1 = df_mat.loc[c1] if c1 in df_mat.index else {}
        mat2 = df_mat.loc[c2] if c2 in df_mat.index else {}

        # Format attribute summary strings
        def format_attrs(mat):
            parts = []
            for col in attr_cols:
                val = mat.get(col)
                if pd.notna(val) and str(val).strip():
                    short_name = col.replace("attr_", "")
                    parts.append(f"{short_name}={val}")
            return "; ".join(parts) if parts else "None detected"

        attrs_1_str = format_attrs(mat1)
        attrs_2_str = format_attrs(mat2)

        raw_desc_1 = mat1.get("raw_description", row.get("description_1", ""))
        raw_desc_2 = mat2.get("raw_description", row.get("description_2", ""))

        clean_desc_1 = mat1.get("cleaned_description", row.get("description_1", ""))
        clean_desc_2 = mat2.get("cleaned_description", row.get("description_2", ""))

        lexical_sim = float(row.get("similarity_score", 0.0))
        sem_sim = float(semantic_sims[idx])

        # Exact match indicator
        is_exact = str(clean_desc_1).strip().upper() == str(clean_desc_2).strip().upper()
        if is_exact and lexical_sim < 1.0:
            lexical_sim = 1.0

        # Check confirmed matching attributes
        confirm_parts = []
        for col in attr_cols:
            v1 = mat1.get(col)
            v2 = mat2.get(col)
            if pd.notna(v1) and pd.notna(v2) and str(v1).strip() and str(v2).strip():
                if str(v1).strip().upper() == str(v2).strip().upper():
                    short_name = col.replace("attr_", "")
                    confirm_parts.append(f"{short_name}: '{v1}'")
        confirms_str = "; ".join(confirm_parts) if confirm_parts else ""

        enriched_rows.append({
            "pair_id": row["pair_id"],
            "corpus_id_1": c1,
            "enterprise_1": row["organization_1"],
            "raw_description_1": raw_desc_1,
            "cleaned_description_1": clean_desc_1,
            "extracted_attributes_1": attrs_1_str,
            "corpus_id_2": c2,
            "enterprise_2": row["organization_2"],
            "raw_description_2": raw_desc_2,
            "cleaned_description_2": clean_desc_2,
            "extracted_attributes_2": attrs_2_str,
            "lexical_similarity": round(lexical_sim, 4),
            "semantic_cosine_similarity": round(sem_sim, 4),
            "is_exact_normalized_match": is_exact,
            "has_attribute_conflict": bool(row.get("has_attribute_conflict", False)),
            "conflict_details": row.get("conflict_details", "") if pd.notna(row.get("conflict_details")) else "",
            "confirmation_details": confirms_str,
            "human_label": "PENDING",  # Reviewer enters 1, 0, or UNCERTAIN
            "confidence_rating": "PENDING",  # Reviewer enters HIGH, MEDIUM, LOW
            "reviewer_notes": "",
            "review_guidance": row.get("evidence_notes", ""),
        })

    df_improved = pd.DataFrame(enriched_rows)
    df_improved.to_csv(output_path, index=False)

    print(f"[OK] Successfully created improved worksheet: {output_path}")
    print(f"[OK] Total review pairs: {len(df_improved)}")
    print(f"[OK] Total columns: {len(df_improved.columns)}")
    print("Columns included in improved review worksheet:")
    for c in df_improved.columns:
        print(f"  - {c}")

if __name__ == "__main__":
    build_improved_worksheet()

