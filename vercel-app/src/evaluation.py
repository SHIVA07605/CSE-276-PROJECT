"""
Evaluation and Metrics Reporting Module.

Implements the official evaluation criteria required by Objective 2 & 3:
- Pairwise Precision, Recall, F1-Score
- Confusion Matrix (TP, FP, TN, FN)
- Cluster Purity against ground-truth material equivalence groups
- Comparative performance reporting against the Exact Match baseline
"""

from typing import Dict, Any, List, Union, Set, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


def compute_pairwise_metrics(
    y_true: Union[List[int], np.ndarray, pd.Series],
    y_pred: Union[List[int], np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Calculate standard pairwise entity resolution metrics.

    Args:
        y_true: Ground truth binary labels (1 = match, 0 = non-match).
        y_pred: Predicted binary labels.

    Returns:
        Dictionary containing Precision, Recall, F1-score, and Confusion Matrix breakdown.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_t, y_p, zero_division=0))
    recall = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_pairs": len(y_t),
    }


def compute_cluster_purity(
    predicted_clusters: Dict[str, str],
    ground_truth_clusters: Dict[str, str],
) -> float:
    """
    Compute cluster purity comparing predicted clusters with ground-truth entity classes.

    Purity = (1 / N) * sum_k [ max_j |omega_k cap c_j| ]
    where omega_k is predicted cluster k, and c_j is true cluster j.

    Args:
        predicted_clusters: Mapping of record_id -> predicted_cluster_id.
        ground_truth_clusters: Mapping of record_id -> true_cluster_id.

    Returns:
        Cluster purity score between 0.0 and 1.0.
    """
    common_keys = [k for k in predicted_clusters if k in ground_truth_clusters]
    n = len(common_keys)
    if n == 0:
        return 0.0

    # Build contingency table
    pred_to_true_counts: Dict[str, Dict[str, int]] = {}
    for k in common_keys:
        p_c = predicted_clusters[k]
        g_c = ground_truth_clusters[k]

        if p_c not in pred_to_true_counts:
            pred_to_true_counts[p_c] = {}
        pred_to_true_counts[p_c][g_c] = pred_to_true_counts[p_c].get(g_c, 0) + 1

    # Sum the majority class count for each predicted cluster
    majority_sum = sum(
        max(true_dist.values())
        for true_dist in pred_to_true_counts.values()
    )

    return float(majority_sum / n)


def generate_comparative_summary(
    baseline_metrics: Dict[str, float],
    model_metrics: Dict[str, float],
) -> pd.DataFrame:
    """
    Create a side-by-side comparison DataFrame between Baseline and proposed ML Model.

    Args:
        baseline_metrics: Metrics dictionary for exact match baseline.
        model_metrics: Metrics dictionary for candidate ML model.

    Returns:
        Pandas DataFrame formatted for report inclusion and Streamlit display.
    """
    rows = []
    keys = ["precision", "recall", "f1", "cluster_purity"]
    for k in keys:
        b_val = baseline_metrics.get(k, np.nan)
        m_val = model_metrics.get(k, np.nan)
        diff = m_val - b_val if not (np.isnan(b_val) or np.isnan(m_val)) else np.nan
        rows.append({
            "Metric": k.upper(),
            "Baseline (Exact Match)": round(b_val, 4) if not np.isnan(b_val) else "N/A",
            "ML Proposed Model": round(m_val, 4) if not np.isnan(m_val) else "N/A",
            "Absolute Improvement": f"{diff:+.4f}" if not np.isnan(diff) else "N/A",
        })

    return pd.DataFrame(rows)


def find_pair_connected_components(
    df_pairs: pd.DataFrame,
    id_col_1: str = "corpus_id_1",
    id_col_2: str = "corpus_id_2",
) -> Tuple[pd.Series, Dict[int, Set[str]]]:
    """
    Construct an undirected graph where each material ID is a node and each pair is an edge.
    Partition the pairs and material IDs into connected components.

    Args:
        df_pairs: DataFrame containing candidate / labelled pairs.
        id_col_1: Name of the first material ID column.
        id_col_2: Name of the second material ID column.

    Returns:
        Tuple of:
        - pair_component_series: pd.Series aligned with df_pairs.index mapping each pair to its component_id (int).
        - component_nodes: dict mapping component_id -> set of all material IDs in that component.
    """
    c1 = id_col_1 if id_col_1 in df_pairs.columns else "record_id_1"
    c2 = id_col_2 if id_col_2 in df_pairs.columns else "record_id_2"

    adj: Dict[str, Set[str]] = {}
    nodes: Set[str] = set()

    for _, row in df_pairs.iterrows():
        u = str(row[c1])
        v = str(row[c2])
        nodes.add(u)
        nodes.add(v)
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)

    visited: Set[str] = set()
    node_to_comp: Dict[str, int] = {}
    comp_to_nodes: Dict[int, Set[str]] = {}
    comp_id = 0

    for node in sorted(nodes):
        if node not in visited:
            comp_nodes: Set[str] = set()
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                comp_nodes.add(curr)
                node_to_comp[curr] = comp_id
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            comp_to_nodes[comp_id] = comp_nodes
            comp_id += 1

    pair_components = [node_to_comp[str(row[c1])] for _, row in df_pairs.iterrows()]
    pair_comp_series = pd.Series(pair_components, index=df_pairs.index, name="component_id")
    return pair_comp_series, comp_to_nodes


def create_connected_component_split(
    df_pairs: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    id_col_1: str = "corpus_id_1",
    id_col_2: str = "corpus_id_2",
    test_size: float = 0.25,
    random_state: int = 42,
    target_series: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Partition candidate/labelled pairs into train and test sets using connected-component grouping.
    Guarantees with 100% mathematical certainty that no material ID occurs in both train and test.

    Args:
        df_pairs: DataFrame of pairs.
        feature_matrix: Feature DataFrame or array aligned with df_pairs.
        id_col_1: Column for first material ID.
        id_col_2: Column for second material ID.
        test_size: Proportion of components to allocate to test partition.
        random_state: Seed for reproducibility.
        target_series: Optional binary target series to verify class representation in both splits.

    Returns:
        Tuple of (train_indices, test_indices, split_audit_metadata).
    """
    c1 = id_col_1 if id_col_1 in df_pairs.columns else "record_id_1"
    c2 = id_col_2 if id_col_2 in df_pairs.columns else "record_id_2"

    pair_comp_series, comp_to_nodes = find_pair_connected_components(df_pairs, c1, c2)
    n_components = len(comp_to_nodes)

    audit: Dict[str, Any] = {
        "total_pairs": len(df_pairs),
        "total_material_ids": len(set(df_pairs[c1]).union(df_pairs[c2])),
        "total_connected_components": n_components,
        "valid_split": False,
        "material_id_overlap_count": 0,
        "limitation_notes": [],
    }

    if n_components < 2:
        audit["limitation_notes"].append(
            f"Insufficient independent components ({n_components}): All pairs form a single connected component. "
            "A leakage-free train/test split cannot be formed without additional independent material entities."
        )
        return np.array([]), np.array([]), audit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(feature_matrix, groups=pair_comp_series))

    # Verify zero material-ID overlap
    train_ids = set(df_pairs.iloc[train_idx][c1]).union(df_pairs.iloc[train_idx][c2])
    test_ids = set(df_pairs.iloc[test_idx][c1]).union(df_pairs.iloc[test_idx][c2])
    overlap = train_ids.intersection(test_ids)

    audit["train_pairs_count"] = len(train_idx)
    audit["test_pairs_count"] = len(test_idx)
    audit["train_material_ids_count"] = len(train_ids)
    audit["test_material_ids_count"] = len(test_ids)
    audit["material_id_overlap_count"] = len(overlap)
    audit["zero_leakage_verified"] = bool(len(overlap) == 0)

    if len(overlap) > 0:
        audit["limitation_notes"].append(f"CRITICAL: Data leakage detected with {len(overlap)} overlapping material IDs!")
        audit["valid_split"] = False
        return train_idx, test_idx, audit

    # Check class distribution if targets provided
    if target_series is not None:
        y_arr = np.asarray(target_series)
        y_train = y_arr[train_idx]
        y_test = y_arr[test_idx]

        train_pos = int((y_train == 1).sum())
        train_neg = int((y_train == 0).sum())
        test_pos = int((y_test == 1).sum())
        test_neg = int((y_test == 0).sum())

        audit["class_distribution"] = {
            "train_positives": train_pos,
            "train_negatives": train_neg,
            "test_positives": test_pos,
            "test_negatives": test_neg,
        }

        if test_pos == 0 or test_neg == 0 or train_pos == 0 or train_neg == 0:
            audit["limitation_notes"].append(
                f"Class distribution imbalance across components (Train: +{train_pos}/-{train_neg}, Test: +{test_pos}/-{test_neg}). "
                "One or both partitions lack positive or negative examples. Metrics should not be claimed until balanced components are annotated."
            )
            audit["valid_split"] = False
        else:
            audit["valid_split"] = True
    else:
        audit["valid_split"] = True

    return train_idx, test_idx, audit
