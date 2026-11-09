"""
Candidate Generation and Blocking Module.

Prunes the quadratic O(N^2) search space of all possible record pairs down to a
tractable set of high-recall candidate pairs using indexing techniques:
1. TF-IDF token n-gram retrieval with Cosine Similarity thresholding.
2. Commodity hint / category blocking with fallback for untagged records.
"""

from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from src.config import COL_CLEAN_DESC, COL_MATERIAL_CODE, COL_ENTERPRISE


def generate_tfidf_candidates(
    df: pd.DataFrame,
    text_col: str = COL_CLEAN_DESC,
    id_col: str = "corpus_id",
    org_col: str = "organization",
    hint_col: str = "item_type_hint",
    min_similarity: float = 0.35,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    batch_size: int = 500,
    block_by_hint: bool = False,
    cross_enterprise_only: bool = False,
) -> pd.DataFrame:
    """
    Retrieve candidate record pairs using TF-IDF token vectors and sparse matrix operations.

    Features:
    - Avoids instantiating full N x N matrix by chunked sparse dot products.
    - Excludes self-matches (i < j strictly enforced).
    - Excludes duplicate pair directions.
    - Computes and preserves cross-enterprise indicators.
    - Configurable similarity threshold (default: 0.35).
    - Supports category blocking by item_type_hint with full fallback for untagged records.

    Args:
        df: Input DataFrame containing material records.
        text_col: Column containing normalized text descriptions.
        id_col: Column containing unique record IDs.
        org_col: Column containing enterprise / organization names.
        hint_col: Column containing commodity category hints.
        min_similarity: Cosine similarity cutoff (default 0.35).
        ngram_range: N-gram range for TfidfVectorizer (default (1, 2)).
        min_df: Minimum document frequency for vocabulary (default 2).
        batch_size: Number of query rows per sparse matrix multiplication.
        block_by_hint: Whether to restrict candidate comparisons within same hint blocks.
        cross_enterprise_only: Whether to retain only cross-enterprise candidate pairs.

    Returns:
        DataFrame containing candidate pairs with columns:
        ['corpus_id_1', 'organization_1', 'cleaned_description_1',
         'corpus_id_2', 'organization_2', 'cleaned_description_2',
         'tfidf_similarity', 'is_cross_enterprise']
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=[
            "corpus_id_1", "organization_1", "cleaned_description_1",
            "corpus_id_2", "organization_2", "cleaned_description_2",
            "tfidf_similarity", "is_cross_enterprise", "record_id_1", "record_id_2"
        ])

    # Resolve column names flexibly
    c_desc = text_col if text_col in df.columns else (
        "cleaned_description" if "cleaned_description" in df.columns else df.columns[0]
    )
    c_id = id_col if id_col in df.columns else (
        COL_MATERIAL_CODE if COL_MATERIAL_CODE in df.columns else df.columns[0]
    )
    c_org = org_col if org_col in df.columns else (
        COL_ENTERPRISE if COL_ENTERPRISE in df.columns else "organization"
    )

    descriptions = df[c_desc].fillna("").astype(str).tolist()
    ids = df[c_id].tolist() if c_id in df.columns else [f"REC_{i}" for i in range(len(df))]
    orgs = df[c_org].tolist() if c_org in df.columns else ["ENT" for _ in range(len(df))]

    # Adjust min_df if corpus is very small (e.g. in unit tests)
    actual_min_df = min(min_df, max(1, len(df) // 2))

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=actual_min_df,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(descriptions)

    # Check for empty vocabulary
    if X.shape[1] == 0:
        return pd.DataFrame(columns=[
            "corpus_id_1", "organization_1", "cleaned_description_1",
            "corpus_id_2", "organization_2", "cleaned_description_2",
            "tfidf_similarity", "is_cross_enterprise", "record_id_1", "record_id_2"
        ])

    candidate_records: List[Dict[str, Any]] = []
    n = X.shape[0]

    # Pre-extract hints if blocking
    hints = df[hint_col].fillna("UNCLASSIFIED").tolist() if (block_by_hint and hint_col in df.columns) else None

    # Chunked sparse multiplication
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_sim = X[start:end].dot(X.T)
        coo = batch_sim.tocoo()

        for r, c, val in zip(coo.row, coo.col, coo.data):
            i = start + r
            j = c
            if i >= j:
                continue  # enforce i < j (excludes self-matches and duplicate directions)

            if val < min_similarity:
                continue

            org1 = orgs[i]
            org2 = orgs[j]
            is_cross = bool(org1 != org2)

            if cross_enterprise_only and not is_cross:
                continue

            if hints is not None:
                # If either is UNCLASSIFIED, allow fallback pairing; otherwise require matching hints
                h1 = hints[i]
                h2 = hints[j]
                if h1 != "UNCLASSIFIED" and h2 != "UNCLASSIFIED" and h1 != h2:
                    continue

            id1 = str(ids[i])
            id2 = str(ids[j])
            candidate_records.append({
                "corpus_id_1": id1,
                "organization_1": org1,
                "cleaned_description_1": descriptions[i],
                "corpus_id_2": id2,
                "organization_2": org2,
                "cleaned_description_2": descriptions[j],
                "tfidf_similarity": round(float(val), 4),
                "is_cross_enterprise": is_cross,
                # Backward compatibility aliases
                "record_id_1": id1,
                "record_id_2": id2,
            })

    if not candidate_records:
        return pd.DataFrame(columns=[
            "corpus_id_1", "organization_1", "cleaned_description_1",
            "corpus_id_2", "organization_2", "cleaned_description_2",
            "tfidf_similarity", "is_cross_enterprise", "record_id_1", "record_id_2"
        ])

    return pd.DataFrame(candidate_records)
