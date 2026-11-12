"""
Entity Grouping and Clustering Module with Conflict-Aware Safeguards.

Aggregates pairwise match predictions into multi-record entity clusters using graph algorithms
and precision-first clustering safeguards:
1. Standard Connected Components (transitive closure of high-confidence match edges).
2. Conflict-Aware Clustering:
   - Validates pairwise compatibility between all members before merging clusters.
   - Detects and blocks merges on confirmed engineering specification conflicts.
   - Identifies transitive conflicts (A matches B, B matches C, but A conflicts with C).
   - Isolates ambiguous and unverified records for reviewer approval.
"""

from typing import Dict, List, Set, Any, Tuple, Optional, Callable, Union
try:
    import networkx as nx
except ImportError:
    nx = None
import pandas as pd


def cluster_connected_components(
    pairs_df: pd.DataFrame,
    id1_col: str = "record_id_1",
    id2_col: str = "record_id_2",
    prediction_col: str = "prediction",
    all_record_ids: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Partition records into clusters via connected components on positive match edges.

    Args:
        pairs_df: DataFrame containing evaluated candidate pairs.
        id1_col: Column with record 1 ID.
        id2_col: Column with record 2 ID.
        prediction_col: Binary prediction column (1 = match, 0 = non-match).
        all_record_ids: Optional complete list of all records (to ensure singletons get clusters).

    Returns:
        Mapping of record_id -> cluster_id (e.g. 'REC_101' -> 'CLUSTER_0001').
    """
    G = nx.Graph()

    # Resolve column names flexibly
    c1 = id1_col if id1_col in pairs_df.columns else "corpus_id_1"
    c2 = id2_col if id2_col in pairs_df.columns else "corpus_id_2"
    pred_col = prediction_col if prediction_col in pairs_df.columns else (
        "human_label" if "human_label" in pairs_df.columns else "decision"
    )

    # Add all records as nodes if supplied
    if all_record_ids:
        for rid in all_record_ids:
            G.add_node(str(rid))

    # Filter for asserted matches
    if pred_col in pairs_df.columns:
        # Match if prediction == 1 or prediction == '1' or 'EQUIVALENT'
        is_match = pairs_df[pred_col].astype(str).str.strip().str.upper().isin(["1", "1.0", "EQUIVALENT", "TRUE"])
        matches = pairs_df[is_match]
    else:
        matches = pairs_df

    for _, row in matches.iterrows():
        u = str(row[c1])
        v = str(row[c2])
        G.add_edge(u, v)

    # Ensure all nodes appearing in pairs_df exist in graph
    for _, row in pairs_df.iterrows():
        G.add_node(str(row[c1]))
        G.add_node(str(row[c2]))

    # Extract connected components
    clusters: Dict[str, str] = {}
    for cluster_idx, component in enumerate(nx.connected_components(G), start=1):
        cluster_id = f"CLUSTER_{cluster_idx:05d}"
        for node in component:
            clusters[node] = cluster_id

    return clusters


def conflict_aware_clustering(
    pairs_df: pd.DataFrame,
    id1_col: str = "record_id_1",
    id2_col: str = "record_id_2",
    prediction_col: str = "prediction",
    similarity_col: Optional[str] = None,
    conflict_pairs: Optional[Set[Tuple[str, str]]] = None,
    conflict_checker: Optional[Callable[[str, str], bool]] = None,
    all_record_ids: Optional[List[str]] = None,
    isolate_ambiguous: bool = True,
    ambiguous_values: Optional[Set[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Cluster records into harmonised entities while enforcing strict specification conflict safeguards.

    Guarantees:
    1. Compatibility check: Before merging two clusters, verifies compatibility between ALL cross-cluster member pairs.
    2. Conflict blocking: Blocks edge merge if ANY member of cluster A conflicts with ANY member of cluster B.
    3. Transitive conflict detection: Explicitly detects and logs situations where A matches B, B matches C, but A conflicts with C.
    4. Isolation of ambiguous records: Records with ambiguous match status (e.g. UNCERTAIN, NEEDS_REVIEW) are quarantined in singleton clusters.

    Args:
        pairs_df: DataFrame of candidate or evaluated pairs.
        id1_col: First ID column name.
        id2_col: Second ID column name.
        prediction_col: Decision or prediction column.
        similarity_col: Optional similarity column for ordering candidate edges (highest similarity first).
        conflict_pairs: Pre-identified set of incompatible pairs (min_id, max_id).
        conflict_checker: Optional function (id1, id2) -> bool returning True if specification conflict exists.
        all_record_ids: Optional exhaustive list of all corpus IDs.
        isolate_ambiguous: If True, ambiguous edges are not merged.
        ambiguous_values: Set of string values representing ambiguity (default: {'UNCERTAIN', 'NEEDS_REVIEW', 'PENDING'}).

    Returns:
        Tuple of:
        - clusters: Dict mapping record_id -> cluster_id
        - audit_report: Dict with blocked merges, transitive conflicts, and clustering statistics
    """
    if ambiguous_values is None:
        ambiguous_values = {"UNCERTAIN", "NEEDS_REVIEW", "PENDING"}

    # Flexible column resolution
    c1 = id1_col if id1_col in pairs_df.columns else "corpus_id_1"
    c2 = id2_col if id2_col in pairs_df.columns else "corpus_id_2"
    pred_col = prediction_col if prediction_col in pairs_df.columns else (
        "human_label" if "human_label" in pairs_df.columns else "decision"
    )

    # Collect all unique record IDs
    all_ids: Set[str] = set()
    if all_record_ids:
        all_ids.update(str(x) for x in all_record_ids)
    for _, row in pairs_df.iterrows():
        all_ids.add(str(row[c1]))
        all_ids.add(str(row[c2]))

    # Initialize disjoint set / cluster map
    node_to_cluster: Dict[str, str] = {rid: f"CLUSTER_{idx:05d}" for idx, rid in enumerate(sorted(all_ids), start=1)}
    cluster_members: Dict[str, Set[str]] = {cid: {rid} for rid, cid in node_to_cluster.items()}

    # Initialize conflict registry
    disallowed: Set[Tuple[str, str]] = set()
    if conflict_pairs:
        for u, v in conflict_pairs:
            pair = tuple(sorted([str(u), str(v)]))
            disallowed.add(pair)

    # Detect conflicts already present in pairs_df
    # Check for conflict flags or negative labels
    for _, row in pairs_df.iterrows():
        u = str(row[c1])
        v = str(row[c2])
        pair = tuple(sorted([u, v]))

        # Confirmed negative or explicit conflict
        val = str(row.get(pred_col, "")).strip().upper()
        has_conf = row.get("has_attribute_conflict", False)
        if str(has_conf).lower() in ["true", "1"] or val in ["0", "0.0", "NOT_EQUIVALENT", "FALSE"]:
            disallowed.add(pair)

    # Candidate positive edges
    candidate_edges: List[Dict[str, Any]] = []
    quarantined_pairs: List[Dict[str, Any]] = []

    for _, row in pairs_df.iterrows():
        u = str(row[c1])
        v = str(row[c2])
        val = str(row.get(pred_col, "")).strip().upper()

        if val in ambiguous_values:
            if isolate_ambiguous:
                quarantined_pairs.append({"u": u, "v": v, "status": val, "pair_id": row.get("pair_id", "")})
                continue

        if val in ["1", "1.0", "EQUIVALENT", "TRUE"]:
            sim = 0.0
            if similarity_col and similarity_col in row:
                try:
                    sim = float(row[similarity_col])
                except (ValueError, TypeError):
                    sim = 0.0
            candidate_edges.append({"u": u, "v": v, "sim": sim, "pair_id": row.get("pair_id", "")})

    # Sort edges descending by similarity (Kruskal-style greedy grouping)
    if similarity_col:
        candidate_edges.sort(key=lambda x: x["sim"], reverse=True)

    # Audit tracking
    blocked_merges: List[Dict[str, Any]] = []
    transitive_conflicts: List[Dict[str, Any]] = []
    successful_merges: List[Dict[str, Any]] = []

    # Process match edges
    for edge in candidate_edges:
        u = edge["u"]
        v = edge["v"]
        pid = edge["pair_id"]

        c_u = node_to_cluster[u]
        c_v = node_to_cluster[v]

        # Already in the same cluster
        if c_u == c_v:
            continue

        members_u = cluster_members[c_u]
        members_v = cluster_members[c_v]

        # Compatibility check: cross-product validation between all members of c_u and c_v
        merge_blocked = False
        conflict_reason = ""
        conflicting_members = None

        for m_u in members_u:
            for m_v in members_v:
                pair_check = tuple(sorted([m_u, m_v]))

                # 1. Known disallowed pair
                if pair_check in disallowed:
                    merge_blocked = True
                    conflicting_members = (m_u, m_v)
                    conflict_reason = f"Confirmed specification conflict between member '{m_u}' and member '{m_v}'"
                    break

                # 2. External conflict checker function
                if conflict_checker is not None:
                    if conflict_checker(m_u, m_v):
                        merge_blocked = True
                        disallowed.add(pair_check)
                        conflicting_members = (m_u, m_v)
                        conflict_reason = f"Conflict checker flagged incompatibility between member '{m_u}' and member '{m_v}'"
                        break

            if merge_blocked:
                break

        if merge_blocked:
            # Check if this is a transitive conflict
            # Direct conflict: u conflicts directly with v
            # Transitive conflict: u and v are asserted to match, but some existing member m_u != u or m_v != v has a conflict!
            is_transitive = (conflicting_members != tuple(sorted([u, v])))

            event = {
                "pair_id": pid,
                "attempted_edge": (u, v),
                "cluster_1": c_u,
                "cluster_2": c_v,
                "conflicting_pair": conflicting_members,
                "is_transitive_conflict": is_transitive,
                "reason": conflict_reason,
            }
            blocked_merges.append(event)
            if is_transitive:
                transitive_conflicts.append(event)
        else:
            # Merge cluster c_v into c_u
            for m in members_v:
                node_to_cluster[m] = c_u
                members_u.add(m)
            del cluster_members[c_v]
            successful_merges.append({"pair_id": pid, "merged_edge": (u, v), "target_cluster": c_u})

    # Renumber clusters compactly
    unique_clusters = sorted(list(cluster_members.keys()))
    cluster_remap = {old_cid: f"CLUSTER_{new_idx:05d}" for new_idx, old_cid in enumerate(unique_clusters, start=1)}
    final_clusters = {rid: cluster_remap[cid] for rid, cid in node_to_cluster.items()}

    # Compute audit statistics
    cluster_sizes = [len(members) for members in cluster_members.values()]
    audit_report = {
        "total_records": len(all_ids),
        "total_clusters": len(cluster_members),
        "singleton_clusters": sum(1 for sz in cluster_sizes if sz == 1),
        "multi_record_clusters": sum(1 for sz in cluster_sizes if sz > 1),
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "successful_merges_count": len(successful_merges),
        "blocked_merges_count": len(blocked_merges),
        "transitive_conflicts_count": len(transitive_conflicts),
        "quarantined_ambiguous_pairs_count": len(quarantined_pairs),
        "blocked_merges": blocked_merges,
        "transitive_conflicts": transitive_conflicts,
        "quarantined_pairs": quarantined_pairs,
    }

    return final_clusters, audit_report
