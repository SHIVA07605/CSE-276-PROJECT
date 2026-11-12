"""
Exact String Matching Baseline Module.

Implements the official project baseline required by Objective 3:
"Report pairwise precision/recall/F1, cluster purity. Baseline: exact string match."

Evaluates pairs based on verbatim or normalised string equality to establish the
lower-bound benchmark that our machine learning system must outperform.
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd

from src.preprocess import normalise_text


def exact_match_predict(desc1: str, desc2: str, normalise: bool = True) -> int:
    """
    Predict match label between two material descriptions using exact string comparison.

    Args:
        desc1: First material description.
        desc2: Second material description.
        normalise: Whether to apply standard lexical normalisation before comparison.

    Returns:
        1 if strings match exactly, 0 otherwise.
    """
    s1 = normalise_text(desc1) if normalise else str(desc1).strip()
    s2 = normalise_text(desc2) if normalise else str(desc2).strip()

    return 1 if (s1 == s2 and len(s1) > 0) else 0


def evaluate_baseline_on_pairs(
    pairs_df: pd.DataFrame,
    desc1_col: str = "material_description_1",
    desc2_col: str = "material_description_2",
    normalise: bool = True,
) -> pd.Series:
    """
    Run the exact match baseline over a DataFrame of record pairs.

    Args:
        pairs_df: DataFrame containing pairwise comparisons.
        desc1_col: Name of column containing record 1's description.
        desc2_col: Name of column containing record 2's description.
        normalise: Whether to normalise strings before equality check.

    Returns:
        Pandas Series of binary predictions (0 or 1).
    """
    return pairs_df.apply(
        lambda row: exact_match_predict(row[desc1_col], row[desc2_col], normalise=normalise),
        axis=1,
    )


def find_exact_matches(
    df: pd.DataFrame,
    desc_col: str = "cleaned_description",
    id_col: str = "corpus_id",
    org_col: str = "organization",
) -> pd.DataFrame:
    """
    Identify all pairs of records sharing verbatim identical normalized descriptions.

    Generates strictly deduplicated pairs (i < j), excludes self-matches,
    and identifies cross-enterprise vs intra-enterprise matches.

    Args:
        df: Input DataFrame containing material records.
        desc_col: Column with normalized descriptions.
        id_col: Column with unique record identifiers.
        org_col: Column with enterprise/organization names.

    Returns:
        DataFrame with columns:
        ['corpus_id_1', 'organization_1', 'corpus_id_2', 'organization_2',
         'matching_description', 'is_cross_enterprise', 'exact_match']
    """
    if df.empty or desc_col not in df.columns:
        return pd.DataFrame(columns=[
            "corpus_id_1", "organization_1", "corpus_id_2", "organization_2",
            "matching_description", "is_cross_enterprise", "exact_match"
        ])

    records_by_desc: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in df.iterrows():
        desc = row[desc_col]
        if pd.isna(desc) or not str(desc).strip():
            continue
        desc_str = str(desc).strip()
        rec = {
            "id": row[id_col] if id_col in row else row.name,
            "org": row[org_col] if org_col in row else "UNKNOWN",
        }
        records_by_desc.setdefault(desc_str, []).append(rec)

    pairs = []
    for desc, group in records_by_desc.items():
        n = len(group)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                r1 = group[i]
                r2 = group[j]
                pairs.append({
                    "corpus_id_1": r1["id"],
                    "organization_1": r1["org"],
                    "corpus_id_2": r2["id"],
                    "organization_2": r2["org"],
                    "matching_description": desc,
                    "is_cross_enterprise": bool(r1["org"] != r2["org"]),
                    "exact_match": 1,
                })

    if not pairs:
        return pd.DataFrame(columns=[
            "corpus_id_1", "organization_1", "corpus_id_2", "organization_2",
            "matching_description", "is_cross_enterprise", "exact_match"
        ])

    return pd.DataFrame(pairs)
