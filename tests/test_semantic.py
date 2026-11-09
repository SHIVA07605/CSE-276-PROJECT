"""
Unit and Integration Tests for Semantic Matching and Sentence Transformers.
Verifies:
1. Embedding dimensions and float32 dtype.
2. Handling of empty, null, and whitespace-only descriptions.
3. Serialization and reload reproducibility of embeddings and ID mapping.
4. Mathematical correctness of pairwise cosine similarity computation.
5. Graceful handling of missing record IDs in similarity calculation.
6. Integrity of production embedding artifacts and candidate pairs.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.feature_extraction import (
    generate_embeddings,
    compute_pair_semantic_similarity,
    save_embeddings,
    load_embeddings,
)
from src.config import MODELS_DIR, RESULTS_DIR, DATA_PROCESSED_DIR


def test_generate_embeddings_shape_and_dtype():
    """Verify generated embeddings have shape (N, 384) and float32 precision."""
    sample_texts = [
        "SEAMLESS CARBON STEEL PIPE 2 INCH SCH 40",
        "STAINLESS STEEL HEX BOLT M10 X 50MM SS304",
        "CENTRIFUGAL WATER PUMP 15 HP 3 PHASE",
    ]
    embeddings = generate_embeddings(sample_texts, show_progress_bar=False)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)
    assert embeddings.dtype == np.float32
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()

    # Verify L2 normalization: each row should have norm ~ 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_generate_embeddings_empty_and_missing_handling():
    """Verify empty lists, None values, and blank descriptions are handled without error."""
    # Empty list
    empty_emb = generate_embeddings([], show_progress_bar=False)
    assert empty_emb.shape == (0, 384)
    assert empty_emb.dtype == np.float32

    # Mixed with None, NaN, and whitespace
    dirty_texts = ["", None, float("nan"), "   ", "VALID TECHNICAL SPEC"]
    dirty_emb = generate_embeddings(dirty_texts, show_progress_bar=False)
    assert dirty_emb.shape == (5, 384)
    assert dirty_emb.dtype == np.float32
    assert not np.isnan(dirty_emb).any()
    assert not np.isinf(dirty_emb).any()


def test_embedding_serialization_and_reload():
    """Verify saved embeddings and ID mappings can be reloaded with 100% fidelity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        emb_file = tmp_path / "test_embeddings.npy"
        map_file = tmp_path / "test_mapping.csv"

        orig_emb = np.random.randn(10, 384).astype(np.float32)
        # Normalize
        orig_emb = orig_emb / np.linalg.norm(orig_emb, axis=1, keepdims=True)
        orig_ids = [f"MAT_{i:04d}" for i in range(10)]

        save_embeddings(orig_emb, orig_ids, emb_file, map_file)

        assert emb_file.exists()
        assert map_file.exists()

        reloaded_emb, id_to_idx = load_embeddings(emb_file, map_file)

        assert np.array_equal(orig_emb, reloaded_emb)
        assert len(id_to_idx) == 10
        for i, cid in enumerate(orig_ids):
            assert id_to_idx[cid] == i


def test_cosine_similarity_calculation_and_range():
    """Verify mathematical correctness of cosine dot product calculation."""
    # Create 3 synthetic normalized 4-dim vectors
    # u = [1, 0, 0, 0]
    # v = [0, 1, 0, 0] (orthogonal -> dot=0)
    # w = [1, 0, 0, 0] (identical -> dot=1)
    # z = [-1, 0, 0, 0] (opposite -> dot=-1)
    synthetic_emb = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    id_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}

    df_pairs = pd.DataFrame([
        {"corpus_id_1": "A", "corpus_id_2": "C"},  # identical -> 1.0
        {"corpus_id_1": "A", "corpus_id_2": "B"},  # orthogonal -> 0.0
        {"corpus_id_1": "A", "corpus_id_2": "D"},  # opposite -> -1.0
    ])

    sims = compute_pair_semantic_similarity(df_pairs, synthetic_emb, id_to_idx)

    assert len(sims) == 3
    assert np.isclose(sims[0], 1.0, atol=1e-4)
    assert np.isclose(sims[1], 0.0, atol=1e-4)
    assert np.isclose(sims[2], -1.0, atol=1e-4)
    assert all(-1.0 <= s <= 1.0 for s in sims)


def test_compute_pair_semantic_similarity_missing_id_safety():
    """Ensure candidate pairs referencing unmapped IDs default safely to 0.0 similarity."""
    synthetic_emb = np.eye(3, dtype=np.float32)
    id_to_idx = {"A": 0, "B": 1}

    df_pairs = pd.DataFrame([
        {"corpus_id_1": "A", "corpus_id_2": "B"},
        {"corpus_id_1": "A", "corpus_id_2": "UNKNOWN_ID"},
    ])

    sims = compute_pair_semantic_similarity(df_pairs, synthetic_emb, id_to_idx)
    assert len(sims) == 2
    assert sims[0] == 0.0  # orthogonal
    assert sims[1] == 0.0  # missing ID defaults safely to 0.0


def test_candidate_pairs_integrity_if_generated():
    """Verify output candidate pair CSV meets all project constraints when generated."""
    candidates_file = RESULTS_DIR / "matching" / "semantic_candidate_pairs.csv"
    if not candidates_file.exists():
        pytest.skip("semantic_candidate_pairs.csv not generated yet")

    df = pd.read_csv(candidates_file)
    assert len(df) == 22464, f"Expected 22,464 candidate pairs, found {len(df)}"
    assert "semantic_cosine_similarity" in df.columns
    assert "tfidf_similarity" in df.columns
    assert "is_cross_enterprise" in df.columns
    assert "attribute_conflict" in df.columns

    # Verify cross-enterprise count
    cross_count = int(df["is_cross_enterprise"].sum())
    assert cross_count == 930, f"Expected 930 cross-enterprise pairs, found {cross_count}"

    # Verify range
    assert not df["semantic_cosine_similarity"].isna().any()
    assert (df["semantic_cosine_similarity"] >= -1.0).all()
    assert (df["semantic_cosine_similarity"] <= 1.0).all()


def test_candidate_embeddings_integrity_if_generated():
    """Verify saved candidate embeddings matrix matches strict schema specifications."""
    emb_file = MODELS_DIR / "candidate_embeddings.npy"
    map_file = MODELS_DIR / "embedding_id_mapping.csv"

    if not emb_file.exists() or not map_file.exists():
        pytest.skip("Embeddings or ID mapping not generated yet")

    emb, id_to_idx = load_embeddings(emb_file, map_file)
    assert emb.shape == (6634, 384)
    assert emb.dtype == np.float32
    assert not np.isnan(emb).any()
    assert not np.isinf(emb).any()

    # Verify ID mapping
    assert len(id_to_idx) == 6634
    materials_file = DATA_PROCESSED_DIR / "preprocessed_materials.csv"
    if materials_file.exists():
        mat_df = pd.read_csv(materials_file)
        assert list(id_to_idx.keys()) == mat_df["corpus_id"].astype(str).tolist()


def test_reuse_embeddings_without_loading_transformer():
    """Verify precomputed embeddings can be loaded and scored without initializing SentenceTransformer."""
    import src.feature_extraction as fe
    # Save original cached model reference
    cached_model = fe._TRANSFORMER_MODEL
    try:
        # Force cached transformer to None
        fe._TRANSFORMER_MODEL = None

        emb_file = MODELS_DIR / "candidate_embeddings.npy"
        map_file = MODELS_DIR / "embedding_id_mapping.csv"
        if not emb_file.exists() or not map_file.exists():
            pytest.skip("Embeddings files not found")

        # Load embeddings via pure NumPy/Pandas
        emb, id_to_idx = load_embeddings(emb_file, map_file)
        assert emb.shape == (6634, 384)

        # Score a small test DataFrame
        df_test = pd.DataFrame([
            {"corpus_id_1": "M000001", "corpus_id_2": "M000002"},
        ])
        sims = compute_pair_semantic_similarity(df_test, emb, id_to_idx)
        assert len(sims) == 1
        assert -1.0 <= sims[0] <= 1.0

        # Assert SentenceTransformer was NEVER initialized
        assert fe._TRANSFORMER_MODEL is None, "SentenceTransformer was unnecessarily initialized!"
    finally:
        fe._TRANSFORMER_MODEL = cached_model


