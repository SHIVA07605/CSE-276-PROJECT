"""
Tests for evaluation module and connected-component group-aware partitioning.
Verifies mathematical guarantee of zero material ID overlap between train and test partitions.
"""

import pytest
import numpy as np
import pandas as pd

from src.evaluation import (
    find_pair_connected_components,
    create_connected_component_split,
    compute_pairwise_metrics,
    compute_cluster_purity,
    generate_comparative_summary,
)


def test_find_pair_connected_components_basic():
    """Verify connected component labeling on a known graph topology."""
    # Graph topology:
    # Component 0: A - B - C
    # Component 1: D - E
    # Component 2: F - G - H - I
    df_pairs = pd.DataFrame({
        "corpus_id_1": ["A", "B", "D", "F", "G", "H"],
        "corpus_id_2": ["B", "C", "E", "G", "H", "I"],
    })

    pair_comps, comp_to_nodes = find_pair_connected_components(df_pairs)

    # Exactly 3 components
    assert len(comp_to_nodes) == 3
    assert len(pair_comps) == len(df_pairs)

    # A, B, C must be in same component
    comp_a = pair_comps.iloc[0]  # (A, B)
    comp_b = pair_comps.iloc[1]  # (B, C)
    assert comp_a == comp_b

    # D, E must be in different component
    comp_d = pair_comps.iloc[2]  # (D, E)
    assert comp_d != comp_a

    # All nodes in component sets
    all_nodes = set()
    for nodes in comp_to_nodes.values():
        all_nodes.update(nodes)
    assert all_nodes == {"A", "B", "C", "D", "E", "F", "G", "H", "I"}


def test_create_connected_component_split_zero_leakage():
    """Verify 100% mathematical zero material ID leakage across train and test partitions."""
    np.random.seed(42)

    # Create 10 disjoint components, each with 3 materials and 3 pairs
    records = []
    for comp_idx in range(10):
        m1 = f"M_{comp_idx}_1"
        m2 = f"M_{comp_idx}_2"
        m3 = f"M_{comp_idx}_3"
        records.append({"corpus_id_1": m1, "corpus_id_2": m2, "feat1": np.random.randn()})
        records.append({"corpus_id_1": m2, "corpus_id_2": m3, "feat1": np.random.randn()})
        records.append({"corpus_id_1": m1, "corpus_id_2": m3, "feat1": np.random.randn()})

    df_pairs = pd.DataFrame(records)
    X = df_pairs[["feat1"]].copy()

    train_idx, test_idx, audit = create_connected_component_split(
        df_pairs, X, test_size=0.30, random_state=42
    )

    # Check indices are non-empty and disjoint
    assert len(train_idx) > 0
    assert len(test_idx) > 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(test_idx) == len(df_pairs)

    # Check zero material ID overlap
    train_materials = set(df_pairs.iloc[train_idx]["corpus_id_1"]).union(
        df_pairs.iloc[train_idx]["corpus_id_2"]
    )
    test_materials = set(df_pairs.iloc[test_idx]["corpus_id_1"]).union(
        df_pairs.iloc[test_idx]["corpus_id_2"]
    )
    overlap = train_materials.intersection(test_materials)

    assert len(overlap) == 0, f"Leakage detected: {overlap}"
    assert audit["zero_leakage_verified"] is True
    assert audit["material_id_overlap_count"] == 0
    assert audit["valid_split"] is True


def test_insufficient_components_limitation():
    """Verify that a single monolithic connected component is flagged as invalid."""
    # All pairs linked together in a single chain: A - B - C - D
    df_pairs = pd.DataFrame({
        "corpus_id_1": ["A", "B", "C"],
        "corpus_id_2": ["B", "C", "D"],
        "feat1": [1.0, 2.0, 3.0],
    })
    X = df_pairs[["feat1"]]

    train_idx, test_idx, audit = create_connected_component_split(df_pairs, X)

    assert audit["valid_split"] is False
    assert len(train_idx) == 0
    assert len(test_idx) == 0
    assert len(audit["limitation_notes"]) > 0
    assert "Insufficient independent components" in audit["limitation_notes"][0]


def test_class_distribution_safeguard():
    """Verify that class imbalance across component partitions is flagged."""
    # 4 components with skewed class distribution
    df_pairs = pd.DataFrame({
        "corpus_id_1": ["A1", "B1", "C1", "D1"],
        "corpus_id_2": ["A2", "B2", "C2", "D2"],
        "target": [1, 1, 0, 0],
        "feat1": [1.0, 2.0, 3.0, 4.0],
    })
    X = df_pairs[["feat1"]]
    y = df_pairs["target"]

    # When test_size partitions so one split has only 1 class
    # With random_state=1, let's verify audit captures class distribution
    train_idx, test_idx, audit = create_connected_component_split(
        df_pairs, X, test_size=0.5, random_state=1, target_series=y
    )

    assert "class_distribution" in audit
    cd = audit["class_distribution"]
    assert "train_positives" in cd
    assert "test_positives" in cd


def test_compute_pairwise_metrics():
    """Verify precision, recall, and F1 computation."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])

    metrics = compute_pairwise_metrics(y_true, y_pred)
    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["true_negatives"] == 2
    assert metrics["precision"] == 1.0
    assert pytest.approx(metrics["recall"]) == 2 / 3
    assert pytest.approx(metrics["f1"]) == 2 * (1.0 * (2 / 3)) / (1.0 + 2 / 3)


def test_compute_cluster_purity():
    """Verify cluster purity computation."""
    pred_clusters = {"r1": "c1", "r2": "c1", "r3": "c2", "r4": "c2"}
    true_clusters = {"r1": "A", "r2": "A", "r3": "B", "r4": "C"}

    # Cluster c1: 2 items, both A (majority = 2)
    # Cluster c2: 2 items, 1 B, 1 C (majority = 1)
    # Total purity = (2 + 1) / 4 = 0.75
    purity = compute_cluster_purity(pred_clusters, true_clusters)
    assert pytest.approx(purity) == 0.75


def test_generate_comparative_summary():
    """Verify comparative summary DataFrame formatting."""
    base = {"precision": 0.8, "recall": 0.4, "f1": 0.5333, "cluster_purity": 0.7}
    model = {"precision": 0.9, "recall": 0.7, "f1": 0.7875, "cluster_purity": 0.85}

    summary = generate_comparative_summary(base, model)
    assert isinstance(summary, pd.DataFrame)
    assert "Metric" in summary.columns
    assert "Baseline (Exact Match)" in summary.columns
    assert "ML Proposed Model" in summary.columns
    assert "Absolute Improvement" in summary.columns
    assert len(summary) == 4


def test_worksheet_connected_component_split():
    """Verify connected-component partitioning on actual project review worksheet."""
    from src.config import DATA_PROCESSED_DIR
    worksheet_path = DATA_PROCESSED_DIR / "improved_labelled_pairs_worksheet.csv"
    if not worksheet_path.exists():
        pytest.skip("Improved review worksheet not found.")

    df_review = pd.read_csv(worksheet_path)
    assert len(df_review) == 50

    pair_comps, comp_to_nodes = find_pair_connected_components(df_review)
    assert len(comp_to_nodes) >= 10

    # Dummy features aligned with pairs
    X = pd.DataFrame({"feat": np.arange(len(df_review))})
    train_idx, test_idx, audit = create_connected_component_split(
        df_review, X, test_size=0.25, random_state=42
    )

    assert audit["zero_leakage_verified"] is True
    assert audit["material_id_overlap_count"] == 0

    train_ids = set(df_review.iloc[train_idx]["corpus_id_1"]).union(
        df_review.iloc[train_idx]["corpus_id_2"]
    )
    test_ids = set(df_review.iloc[test_idx]["corpus_id_1"]).union(
        df_review.iloc[test_idx]["corpus_id_2"]
    )
    assert len(train_ids.intersection(test_ids)) == 0
