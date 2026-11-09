"""
Tests for Entity Clustering and Evaluation.
"""

import pytest
import pandas as pd
nx = pytest.importorskip("networkx", reason="networkx required for clustering tests")
from src.clustering import cluster_connected_components
from src.evaluation import compute_pairwise_metrics, compute_cluster_purity


def test_cluster_connected_components():
    pairs = pd.DataFrame([
        {"record_id_1": "REC_A", "record_id_2": "REC_B", "prediction": 1},
        {"record_id_1": "REC_B", "record_id_2": "REC_C", "prediction": 1},
        {"record_id_1": "REC_D", "record_id_2": "REC_E", "prediction": 1},
    ])
    clusters = cluster_connected_components(pairs)

    # REC_A, REC_B, REC_C should share the same cluster
    assert clusters["REC_A"] == clusters["REC_B"] == clusters["REC_C"]
    # REC_D and REC_E should be in a distinct cluster
    assert clusters["REC_D"] == clusters["REC_E"]
    assert clusters["REC_A"] != clusters["REC_D"]


def test_compute_pairwise_metrics():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]
    metrics = compute_pairwise_metrics(y_true, y_pred)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1


def test_compute_cluster_purity():
    # Ground truth: {A, B} in C1, {C, D} in C2
    ground_truth = {"A": "C1", "B": "C1", "C": "C2", "D": "C2"}
    # Perfect prediction
    pred_perfect = {"A": "K1", "B": "K1", "C": "K2", "D": "K2"}
    assert compute_cluster_purity(pred_perfect, ground_truth) == 1.0

    # Mixed prediction
    pred_mixed = {"A": "K1", "B": "K1", "C": "K1", "D": "K2"}
    # In K1: A, B belong to C1 (2), C belongs to C2 (1) -> max is 2
    # In K2: D belongs to C2 (1) -> max is 1
    # Purity = (2 + 1) / 4 = 0.75
    assert compute_cluster_purity(pred_mixed, ground_truth) == 0.75
