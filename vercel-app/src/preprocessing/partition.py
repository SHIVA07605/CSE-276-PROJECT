"""
Candidate Partition and Corpus Hierarchy Module.

Reconstructs and validates the approved multi-tier candidate partition:
- Tier 1: Discrete material lines (923 records)
- Tier 2 Tagged: Tagged procurement descriptions (5,711 records)
- Tier 2 Untagged: Untagged procurement descriptions (14,542 records)
- Tier 3: Administrative / works descriptions (337 records)

Candidate pool for entity resolution: Tier 1 + Tier 2 Tagged (6,634 records).
Remaining records preserved separately: Tier 2 Untagged + Tier 3 (14,879 records).
"""

from typing import Dict, Any, List
import pandas as pd


# Explicit approved classification of description_kind values
TIER_1_KINDS: List[str] = [
    "procurement_plan_item",
    "doc_spec_sentence",
    "doc_spec_string",
]

TIER_3_KINDS: List[str] = [
    "tender_title",
    "work_description",
    "doc_brief_description",
    "corrigendum",
    "doc_tender_title",
]

EXPECTED_COUNTS: Dict[str, int] = {
    "tier1_discrete": 923,
    "tier2_tagged": 5711,
    "tier2_untagged": 14542,
    "tier3_administrative": 337,
    "total": 21513,
    "candidate_pool": 6634,
    "remaining_pool": 14879,
}


def partition_corpus(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Partition the material corpus into the approved candidate tiers.

    Partitions:
    - tier1_discrete: Line items and specification sentences/strings.
    - tier2_tagged: Procurement descriptions with industrial commodity tags (item_type_hint).
    - tier2_untagged: Procurement descriptions without commodity tags.
    - tier3_administrative: Tender package titles, scopes of work, and administrative notices.
    - candidate_pool: Combined Tier 1 + Tier 2 Tagged records (6,634 records).
    - remaining_pool: Combined Tier 2 Untagged + Tier 3 records (14,879 records).

    Args:
        df: Raw corpus DataFrame (expected 21,513 records).

    Returns:
        Dictionary of partitioned DataFrames.
    """
    t1_mask = df["description_kind"].isin(TIER_1_KINDS)
    t3_mask = df["description_kind"].isin(TIER_3_KINDS)
    t2_mask = ~t1_mask & ~t3_mask

    tier1 = df[t1_mask].copy()
    tier3 = df[t3_mask].copy()

    tier2 = df[t2_mask]
    tier2_tagged = tier2[tier2["item_type_hint"].notnull()].copy()
    tier2_untagged = tier2[tier2["item_type_hint"].isnull()].copy()

    candidate_pool = pd.concat([tier1, tier2_tagged], axis=0).copy()
    remaining_pool = pd.concat([tier2_untagged, tier3], axis=0).copy()

    # Assign partition category metadata
    tier1["partition_tier"] = "Tier 1: Discrete material lines"
    tier2_tagged["partition_tier"] = "Tier 2: Tagged procurement descriptions"
    tier2_untagged["partition_tier"] = "Tier 2: Untagged descriptions"
    tier3["partition_tier"] = "Tier 3: Administrative/works descriptions"

    return {
        "tier1_discrete": tier1,
        "tier2_tagged": tier2_tagged,
        "tier2_untagged": tier2_untagged,
        "tier3_administrative": tier3,
        "candidate_pool": candidate_pool,
        "remaining_pool": remaining_pool,
    }


def validate_partitions(
    df: pd.DataFrame,
    partitions: Dict[str, pd.DataFrame],
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Programmatically validate counts, uniqueness of corpus IDs, mutual exclusivity,
    and collective exhaustiveness against the approved project definitions.

    Args:
        df: Input raw corpus DataFrame.
        partitions: Dictionary output from partition_corpus.
        strict: If True, raises ValueError on any mismatch.

    Returns:
        Dictionary of validation metrics.
    """
    t1 = partitions["tier1_discrete"]
    t2_tag = partitions["tier2_tagged"]
    t2_untag = partitions["tier2_untagged"]
    t3 = partitions["tier3_administrative"]
    cand = partitions["candidate_pool"]
    rem = partitions["remaining_pool"]

    counts = {
        "tier1_discrete": len(t1),
        "tier2_tagged": len(t2_tag),
        "tier2_untagged": len(t2_untag),
        "tier3_administrative": len(t3),
        "total": len(df),
        "candidate_pool": len(cand),
        "remaining_pool": len(rem),
    }

    # Verify ID sets
    id_col = "corpus_id" if "corpus_id" in df.columns else df.columns[0]
    ids_t1 = set(t1[id_col])
    ids_t2_tag = set(t2_tag[id_col])
    ids_t2_untag = set(t2_untag[id_col])
    ids_t3 = set(t3[id_col])
    all_ids = set(df[id_col])

    unique_id_count = df[id_col].nunique()
    has_unique_ids = (unique_id_count == len(df))

    is_mutually_exclusive = (
        len(ids_t1 & ids_t2_tag) == 0
        and len(ids_t1 & ids_t2_untag) == 0
        and len(ids_t1 & ids_t3) == 0
        and len(ids_t2_tag & ids_t2_untag) == 0
        and len(ids_t2_tag & ids_t3) == 0
        and len(ids_t2_untag & ids_t3) == 0
    )

    is_collectively_exhaustive = (ids_t1 | ids_t2_tag | ids_t2_untag | ids_t3) == all_ids

    # Validate against expected counts
    count_matches = {k: counts[k] == EXPECTED_COUNTS[k] for k in EXPECTED_COUNTS}
    all_counts_match = all(count_matches.values())

    results = {
        "counts": counts,
        "expected_counts": EXPECTED_COUNTS,
        "count_matches": count_matches,
        "all_counts_match": all_counts_match,
        "unique_id_count": unique_id_count,
        "has_unique_ids": has_unique_ids,
        "is_mutually_exclusive": is_mutually_exclusive,
        "is_collectively_exhaustive": is_collectively_exhaustive,
        "valid": all_counts_match and has_unique_ids and is_mutually_exclusive and is_collectively_exhaustive,
    }

    if strict and not results["valid"]:
        raise ValueError(f"Partition validation failed: {results}")

    return results
