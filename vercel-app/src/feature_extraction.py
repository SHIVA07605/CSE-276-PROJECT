"""
Multi-Modal Feature Extraction Module.

Constructs rich pairwise feature vectors combining:
1. Lexical features: RapidFuzz token sort ratio, Jaccard token overlap, character Levenshtein.
2. Semantic features: Dense embeddings from sentence-transformers/all-MiniLM-L6-v2 and cosine similarity.
3. Attribute features: Discrete comparison indicators (match, conflict, missing) for engineering parameters.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DEFAULT_EMBEDDING_MODEL

# Lazy loader for SentenceTransformer to optimize startup time
_TRANSFORMER_MODEL = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Load or retrieve cached SentenceTransformer instance.
    """
    global _TRANSFORMER_MODEL
    if _TRANSFORMER_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _TRANSFORMER_MODEL = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. Please install it using: "
                "pip install sentence-transformers"
            )
    return _TRANSFORMER_MODEL


def compute_lexical_features(text1: str, text2: str) -> Dict[str, float]:
    """
    Calculate lexical distance and string similarity metrics between two text descriptions.

    Args:
        text1: First description.
        text2: Second description.

    Returns:
        Dictionary of lexical similarity scores in [0.0, 1.0].
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        raise ImportError("rapidfuzz is required for compute_lexical_features")

    s1 = str(text1).strip().upper()
    s2 = str(text2).strip().upper()

    tokens1 = set(s1.split())
    tokens2 = set(s2.split())

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    jaccard = (intersection / union) if union > 0 else 0.0

    return {
        "levenshtein_ratio": fuzz.ratio(s1, s2) / 100.0,
        "token_sort_ratio": fuzz.token_sort_ratio(s1, s2) / 100.0,
        "token_set_ratio": fuzz.token_set_ratio(s1, s2) / 100.0,
        "jaccard_similarity": float(jaccard),
    }


def compare_extracted_attributes(attr1: Dict[str, Any], attr2: Dict[str, Any]) -> Dict[str, float]:
    """
    Compare extracted technical attributes to identify confirmations and critical conflicts.

    Encoding logic:
      +1.0 : Both records contain the attribute, and they match (confirmation).
      -1.0 : Both records contain the attribute, but values differ (conflict / incompatible spec).
       0.0 : Attribute is missing in one or both records.

    Target attributes:
      metric_thread, diameter, material_grade, pressure_class, schedule

    Args:
        attr1: Attribute dictionary for record 1.
        attr2: Attribute dictionary for record 2.

    Returns:
        Dictionary of attribute alignment scores.
    """
    keys = ["metric_thread", "diameter", "material_grade", "pressure_class", "schedule"]
    result: Dict[str, float] = {}

    for k in keys:
        v1 = attr1.get(k)
        v2 = attr2.get(k)

        if v1 is not None and v2 is not None and str(v1).strip() and str(v2).strip():
            s1 = str(v1).strip().upper()
            s2 = str(v2).strip().upper()
            result[f"attr_{k}_match"] = 1.0 if s1 == s2 else -1.0
        else:
            result[f"attr_{k}_match"] = 0.0

    return result


def check_attribute_conflicts(
    attr1: Dict[str, Any],
    attr2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Audit engineering compatibility between two attribute dictionaries.

    Distinguishes:
    - Conflicting attributes (both populated, incompatible values -> -1.0)
    - Confirmed attributes (both populated, identical values -> +1.0)
    - Missing attributes (at least one unpopulated -> 0.0)

    Args:
        attr1: Extracted attributes for record 1.
        attr2: Extracted attributes for record 2.

    Returns:
        Dictionary containing:
        - 'has_conflict': bool
        - 'conflicts': List[str] describing mismatched fields
        - 'confirmations': List[str] describing matching fields
        - 'comparison_scores': Dict[str, float]
    """
    scores = compare_extracted_attributes(attr1, attr2)
    conflicts: List[str] = []
    confirmations: List[str] = []

    for k, score in scores.items():
        attr_name = k.replace("attr_", "").replace("_match", "")
        if score == -1.0:
            val1 = attr1.get(attr_name)
            val2 = attr2.get(attr_name)
            conflicts.append(f"{attr_name}: '{val1}' != '{val2}'")
        elif score == 1.0:
            val = attr1.get(attr_name)
            confirmations.append(f"{attr_name}: '{val}'")

    return {
        "has_conflict": len(conflicts) > 0,
        "conflict_count": len(conflicts),
        "confirmation_count": len(confirmations),
        "conflicts": conflicts,
        "confirmations": confirmations,
        "comparison_scores": scores,
    }


def generate_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    device: Optional[str] = None,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Generate dense semantic embeddings using SentenceTransformers (inference-only).

    Requirements:
    - 384-dimensional dense representations for sentence-transformers/all-MiniLM-L6-v2.
    - Robust handling of missing, empty, or non-string values.
    - Automatic L2 normalization for efficient cosine similarity via dot products.
    - Zero NaNs / Infs with dimension and alignment validation.

    Args:
        texts: List of text descriptions to encode.
        model_name: Hugging Face model identifier (default: all-MiniLM-L6-v2).
        batch_size: Mini-batch size for inference (default: 64).
        normalize_embeddings: Whether to L2-normalize vectors for cosine similarity (default: True).
        device: Target execution device ('cuda', 'cpu', or None for auto-detection).
        show_progress_bar: Whether to display a progress bar during encoding.

    Returns:
        np.ndarray of shape (len(texts), 384) with dtype float32.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    # Sanitize inputs: convert nulls/nans/empties to safe string tokens
    safe_texts = [
        str(t).strip() if (t is not None and not (isinstance(t, float) and np.isnan(t)) and str(t).strip()) else ""
        for t in texts
    ]

    model = get_embedding_model(model_name)
    if device is not None:
        model = model.to(device)

    embeddings = model.encode(
        safe_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Strict validation: guard against numerical edge-cases
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

    return embeddings


def compute_pair_semantic_similarity(
    candidate_df: pd.DataFrame,
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int],
    id_col_1: str = "corpus_id_1",
    id_col_2: str = "corpus_id_2",
) -> np.ndarray:
    """
    Compute pairwise semantic cosine similarity for candidate pairs using precomputed normalized embeddings.

    Because embeddings are L2-normalized, cosine similarity is simply the inner (dot) product:
        sim(u, v) = u . v

    Does NOT compute full N x N matrix; strictly operates on candidate pairs.

    Args:
        candidate_df: DataFrame containing candidate pairs with identifier columns.
        embeddings: Normalized 2D numpy array of shape (N, D).
        id_to_idx: Dictionary mapping unique record ID (str) to embedding row index.
        id_col_1: Name of the first ID column (default 'corpus_id_1').
        id_col_2: Name of the second ID column (default 'corpus_id_2').

    Returns:
        1D numpy array of cosine similarity scores in range [-1.0, 1.0], rounded to 4 decimals.
    """
    if candidate_df.empty:
        return np.array([], dtype=np.float32)

    col1 = id_col_1 if id_col_1 in candidate_df.columns else "record_id_1"
    col2 = id_col_2 if id_col_2 in candidate_df.columns else "record_id_2"

    ids_1 = candidate_df[col1].astype(str).tolist()
    ids_2 = candidate_df[col2].astype(str).tolist()

    idx1 = []
    idx2 = []
    missing_indices = []

    for i, (i1, i2) in enumerate(zip(ids_1, ids_2)):
        if i1 in id_to_idx and i2 in id_to_idx:
            idx1.append(id_to_idx[i1])
            idx2.append(id_to_idx[i2])
        else:
            missing_indices.append(i)
            idx1.append(0)
            idx2.append(0)

    u = embeddings[idx1]
    v = embeddings[idx2]
    sims = np.sum(u * v, axis=1)

    if missing_indices:
        sims[missing_indices] = 0.0

    sims = np.clip(sims, -1.0, 1.0)
    return np.round(sims.astype(np.float32), 4)


def save_embeddings(
    embeddings: np.ndarray,
    corpus_ids: List[str],
    embeddings_path: Union[str, Path],
    mapping_path: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    Serialize embeddings matrix and explicit record-ID mapping.

    Args:
        embeddings: 2D numpy array of shape (N, 384).
        corpus_ids: Ordered list of record IDs matching embedding matrix rows.
        embeddings_path: Destination .npy file path.
        mapping_path: Destination .csv mapping file path.

    Returns:
        Tuple of (Path(embeddings_path), Path(mapping_path)).
    """
    p_emb = Path(embeddings_path)
    p_map = Path(mapping_path)

    p_emb.parent.mkdir(parents=True, exist_ok=True)
    p_map.parent.mkdir(parents=True, exist_ok=True)

    if len(corpus_ids) != len(embeddings):
        raise ValueError(
            f"Length mismatch: {len(corpus_ids)} corpus IDs vs {len(embeddings)} embedding rows"
        )

    # Save numpy embedding matrix
    np.save(p_emb, embeddings.astype(np.float32))

    # Save ID mapping CSV preserving exact matrix row indices
    mapping_df = pd.DataFrame({
        "index": list(range(len(corpus_ids))),
        "corpus_id": [str(cid) for cid in corpus_ids],
    })
    mapping_df.to_csv(p_map, index=False)

    return p_emb, p_map


def load_embeddings(
    embeddings_path: Union[str, Path],
    mapping_path: Union[str, Path],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load precomputed embeddings and construct record ID to row index lookup.

    Args:
        embeddings_path: Path to serialized .npy embeddings.
        mapping_path: Path to serialized .csv index-to-ID mapping.

    Returns:
        Tuple of (embeddings array, id_to_idx dictionary).
    """
    p_emb = Path(embeddings_path)
    p_map = Path(mapping_path)

    if not p_emb.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p_emb}")
    if not p_map.exists():
        raise FileNotFoundError(f"ID mapping file not found: {p_map}")

    embeddings = np.load(p_emb)
    mapping_df = pd.read_csv(p_map)

    id_to_idx = {str(row["corpus_id"]): int(row["index"]) for _, row in mapping_df.iterrows()}
    return embeddings, id_to_idx


def assemble_pairwise_features(
    df_pairs: pd.DataFrame,
    df_materials: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construct numeric feature matrix for supervised pairwise classification.

    Combines:
    1. Lexical similarity (TF-IDF cosine).
    2. Dense semantic similarity (SentenceTransformer cosine).
    3. Similarity discrepancy (semantic - lexical, capturing synonym effects).
    4. Exact normalized match indicator.
    5. Attribute conflict indicator and count.
    6. Attribute confirmation count.
    7. Cross-enterprise relationship indicator.
    8. Discrete attribute match scores (+1 match, 0 missing, -1 conflict)
       for diameter, grade, pressure, schedule, metric thread.

    Strictly excludes corpus IDs, tender IDs, source URLs, enterprise names,
    or arbitrary text identifiers to prevent target leakage.

    Args:
        df_pairs: DataFrame containing candidate/labelled pairs.
        df_materials: Optional DataFrame of material records for discrete attribute lookup.

    Returns:
        DataFrame containing strictly numerical features.
    """
    features = pd.DataFrame(index=df_pairs.index)

    # 1. Lexical similarity
    if "lexical_similarity" in df_pairs.columns:
        features["tfidf_similarity"] = df_pairs["lexical_similarity"].fillna(0.0).astype(float)
    elif "tfidf_similarity" in df_pairs.columns:
        features["tfidf_similarity"] = df_pairs["tfidf_similarity"].fillna(0.0).astype(float)
    elif "similarity_score" in df_pairs.columns:
        features["tfidf_similarity"] = df_pairs["similarity_score"].fillna(0.0).astype(float)
    else:
        features["tfidf_similarity"] = 0.0

    # 2. Semantic similarity
    if "semantic_cosine_similarity" in df_pairs.columns:
        features["semantic_cosine_similarity"] = df_pairs["semantic_cosine_similarity"].fillna(0.0).astype(float)
    else:
        features["semantic_cosine_similarity"] = features["tfidf_similarity"]

    # 3. Discrepancy feature (Semantic - Lexical): High positive indicates synonym gap
    features["similarity_difference"] = features["semantic_cosine_similarity"] - features["tfidf_similarity"]

    # 4. Exact normalized match indicator
    if "is_exact_normalized_match" in df_pairs.columns:
        features["is_exact_match"] = df_pairs["is_exact_normalized_match"].astype(int)
    elif "cleaned_description_1" in df_pairs.columns and "cleaned_description_2" in df_pairs.columns:
        d1 = df_pairs["cleaned_description_1"].fillna("").astype(str).str.strip().str.upper()
        d2 = df_pairs["cleaned_description_2"].fillna("").astype(str).str.strip().str.upper()
        features["is_exact_match"] = (d1 == d2).astype(int)
    else:
        features["is_exact_match"] = (features["tfidf_similarity"] >= 0.999).astype(int)

    # 5. Attribute conflict indicators
    if "has_attribute_conflict" in df_pairs.columns:
        features["has_attribute_conflict"] = df_pairs["has_attribute_conflict"].astype(int)
    elif "attribute_conflict" in df_pairs.columns:
        features["has_attribute_conflict"] = df_pairs["attribute_conflict"].astype(int)
    else:
        features["has_attribute_conflict"] = 0

    # 6. Conflict & Confirmation counts
    if "conflict_details" in df_pairs.columns:
        features["conflict_count"] = df_pairs["conflict_details"].fillna("").apply(
            lambda x: len(str(x).split(";")) if str(x).strip() else 0
        )
    else:
        features["conflict_count"] = features["has_attribute_conflict"]

    if "confirmation_details" in df_pairs.columns:
        features["confirmation_count"] = df_pairs["confirmation_details"].fillna("").apply(
            lambda x: len(str(x).split(";")) if str(x).strip() else 0
        )
    else:
        features["confirmation_count"] = 0

    # 7. Cross-enterprise indicator
    if "is_cross_enterprise" in df_pairs.columns:
        features["is_cross_enterprise"] = df_pairs["is_cross_enterprise"].astype(int)
    elif "enterprise_1" in df_pairs.columns and "enterprise_2" in df_pairs.columns:
        features["is_cross_enterprise"] = (df_pairs["enterprise_1"] != df_pairs["enterprise_2"]).astype(int)
    elif "organization_1" in df_pairs.columns and "organization_2" in df_pairs.columns:
        features["is_cross_enterprise"] = (df_pairs["organization_1"] != df_pairs["organization_2"]).astype(int)
    else:
        features["is_cross_enterprise"] = 0

    # 8. Discrete attribute match scores (+1.0 match, 0.0 missing, -1.0 conflict)
    attr_names = ["diameter", "material_grade", "pressure_class", "metric_thread", "schedule"]
    if df_materials is not None:
        mat_dict = df_materials.set_index("corpus_id").to_dict(orient="index") if "corpus_id" in df_materials.columns else df_materials.to_dict(orient="index")
        id_col1 = "corpus_id_1" if "corpus_id_1" in df_pairs.columns else "record_id_1"
        id_col2 = "corpus_id_2" if "corpus_id_2" in df_pairs.columns else "record_id_2"

        for attr in attr_names:
            col_name = f"attr_{attr}_match"
            field_name = f"attr_{attr}"
            matches = []
            for _, r in df_pairs.iterrows():
                id1 = str(r.get(id_col1, ""))
                id2 = str(r.get(id_col2, ""))
                v1 = mat_dict.get(id1, {}).get(field_name) if id1 in mat_dict else None
                v2 = mat_dict.get(id2, {}).get(field_name) if id2 in mat_dict else None
                if pd.notna(v1) and pd.notna(v2) and str(v1).strip() and str(v2).strip():
                    score = 1.0 if str(v1).strip().upper() == str(v2).strip().upper() else -1.0
                else:
                    score = 0.0
                matches.append(score)
            features[col_name] = matches
    else:
        # Fallback if attribute match scores already exist in df_pairs
        for attr in attr_names:
            col_name = f"attr_{attr}_match"
            if col_name in df_pairs.columns:
                features[col_name] = df_pairs[col_name].fillna(0.0).astype(float)
            else:
                features[col_name] = 0.0

    return features
