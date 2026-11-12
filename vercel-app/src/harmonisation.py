"""
Material Code Harmonisation and Evidence Generation Module.

Generates unified canonical material identifiers for resolved record clusters and
compiles comprehensive technical evidence to empower human procurement reviewers.
"""

from typing import Dict, List, Any, Optional
import hashlib
import pandas as pd


def generate_harmonised_code(cluster_id: str, prefix: str = "HMC-MAT") -> str:
    """
    Generate a standardized, canonical harmonised material code for an entity cluster.

    Args:
        cluster_id: Unique internal cluster identifier.
        prefix: Enterprise harmonisation prefix.

    Returns:
        Formatted canonical harmonised code (e.g. 'HMC-MAT-7A8B9C').
    """
    # Deterministic 6-character hash suffix
    hash_suffix = hashlib.sha256(str(cluster_id).encode("utf-8")).hexdigest()[:6].upper()
    return f"{prefix}-{hash_suffix}"


def compile_cluster_evidence(
    cluster_records: pd.DataFrame,
    cluster_id: str,
) -> Dict[str, Any]:
    """
    Synthesise audit evidence for a proposed harmonised cluster.

    Collects:
    - Contributing enterprises and original material codes
    - Representative canonical description
    - Agreed-upon physical attributes
    - Suggested harmonised code

    Args:
        cluster_records: DataFrame containing all member records belonging to this cluster.
        cluster_id: Unique cluster identifier.

    Returns:
        Dictionary containing structured audit evidence for human review.
    """
    harmonised_code = generate_harmonised_code(cluster_id)

    enterprises = list(cluster_records["enterprise_name"].unique()) if "enterprise_name" in cluster_records else []
    original_codes = list(cluster_records["material_code"].unique()) if "material_code" in cluster_records else []
    descriptions = list(cluster_records["material_description"].unique()) if "material_description" in cluster_records else []

    # Select representative description (e.g. the most complete or longest description)
    rep_desc = max(descriptions, key=len) if descriptions else "No description available"

    return {
        "cluster_id": cluster_id,
        "harmonised_code": harmonised_code,
        "member_count": len(cluster_records),
        "enterprises": enterprises,
        "original_codes": original_codes,
        "descriptions": descriptions,
        "representative_description": rep_desc,
        "review_status": "PROPOSED",  # PROPOSED, APPROVED, REJECTED, UNCERTAIN
    }
