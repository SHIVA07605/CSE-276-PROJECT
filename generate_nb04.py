"""
Generator script for Notebook 04: Semantic Matching Using Sentence Transformers.
Creates notebooks/04_semantic_matching.ipynb adhering strictly to course standards.
"""

import json
from pathlib import Path

cells = []


def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]] if source_lines else [],
    }


def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]] if source_lines else [],
    }


# ==============================================================================
# Cell 1: Notebook Header
# ==============================================================================
cells.append(md_cell([
    "# 04. Semantic Matching Using Sentence Transformers",
    "",
    "**Course:** CSE 276 – CA-1 Academic Project  ",
    "**Project:** Material Code Harmonisation Across Public Enterprises  ",
    "**Target Candidate Pool:** 6,634 Preprocessed Material Descriptions  ",
    "**Candidate Pairs:** 22,464 Pairs (from Notebook 03 TF-IDF Lexical Retrieval)  ",
    "**Cross-Enterprise Focus:** 930 Cross-Enterprise Candidate Pairs  ",
    "**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)  ",
    "",
    "---",
    "",
    "## Notebook Objectives",
    "1. **Dense Semantic Embeddings:** Generate 384-dimensional dense sentence embeddings using pre-trained `all-MiniLM-L6-v2` (inference-only, no fine-tuning).",
    "2. **Safe Serialization & ID Alignment:** Persist embeddings to `models/candidate_embeddings.npy` along with `models/embedding_id_mapping.csv` to ensure exact index-to-record alignment.",
    "3. **Tractable Candidate Scoring:** Avoid intractable quadratic comparisons ($6,634 \\times 6,634 / 2 \\approx 22\\text{ million pairs}$); compute semantic cosine similarity strictly on the 22,464 candidate pairs from Notebook 03.",
    "4. **Cross-Enterprise Deep Dive:** Analyze lexical vs. semantic similarity across the 930 cross-enterprise pairs, identifying genuine synonyms (high semantic, lower lexical) and spurious matches.",
    "5. **Physical Attribute Compatibility Integration:** Contrast high semantic similarity against engineering attribute conflicts (e.g. 250mm vs 300mm pipe diameter) to demonstrate why semantic similarity alone cannot establish equivalence.",
    "6. **Exploratory Candidate Retention Analysis:** Evaluate candidate retention across exploratory similarity thresholds ($\tau \\in [0.60, 0.85]$) without fabricating precision or recall.",
]))

# ==============================================================================
# Cell 2: Step 1 - Environment & Repository Configuration
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 1: Environment & Repository Configuration (Colab + Workstation Support)",
    "# ==============================================================================",
    "import os",
    "import sys",
    "import time",
    "from pathlib import Path",
    "",
    "# Detect execution environment",
    "IN_COLAB = 'google.colab' in sys.modules",
    "",
    "if IN_COLAB:",
    "    print('[COLAB] Running in Google Colab environment.')",
    "    REPO_NAME = 'CSE-276-PROJECT'",
    "    REPO_URL = 'https://github.com/SHIVA07605/CSE-276-PROJECT.git'",
    "    ",
    "    if not os.path.exists(f'/content/{REPO_NAME}'):",
    "        print(f'Cloning repository from {REPO_URL}...')",
    "        os.system(f'git clone {REPO_URL}')",
    "    ",
    "    PROJECT_ROOT = Path(f'/content/{REPO_NAME}').resolve()",
    "    os.chdir(PROJECT_ROOT)",
    "    os.system('pip install -q sentence-transformers pandas numpy matplotlib seaborn scikit-learn')",
    "else:",
    "    print('[LOCAL] Running in Local Workstation environment.')",
    "    current_dir = Path.cwd().resolve()",
    "    PROJECT_ROOT = current_dir.parent if current_dir.name == 'notebooks' else current_dir",
    "",
    "# Add project root to sys.path",
    "if str(PROJECT_ROOT) not in sys.path:",
    "    sys.path.insert(0, str(PROJECT_ROOT))",
    "",
    "print(f'Project Root: {PROJECT_ROOT}')",
    "print(f'sys.path configured: {str(PROJECT_ROOT) in sys.path}')",
    "",
    "# Core library imports",
    "import json",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "",
    "# Project module imports",
    "from src.config import (",
    "    DATA_PROCESSED_DIR,",
    "    MODELS_DIR,",
    "    RESULTS_DIR,",
    "    FIGURES_DIR,",
    "    REPORTS_DIR,",
    "    DEFAULT_EMBEDDING_MODEL,",
    ")",
    "from src.feature_extraction import (",
    "    get_embedding_model,",
    "    generate_embeddings,",
    "    compute_pair_semantic_similarity,",
    "    save_embeddings,",
    "    load_embeddings,",
    ")",
    "",
    "print('[OK] Core libraries and project modules successfully loaded.')",
]))

# ==============================================================================
# Cell 3: Markdown - Input Data Ingestion
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 2: Input Data Ingestion & Schema Verification",
    "",
    "We load two approved upstream artifacts:",
    "1. `data/processed/preprocessed_materials.csv`: 6,634 preprocessed candidate materials (Tier 1 discrete + Tier 2 tagged) with normalized text and extracted attributes.",
    "2. `results/matching/tfidf_candidate_pairs.csv`: 22,464 candidate pairs generated via TF-IDF blocking in Notebook 03 ($\\tau_{lex} \\ge 0.35$), including 930 cross-enterprise pairs.",
]))

# ==============================================================================
# Cell 4: Step 2 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 2: Load and Verify Upstream Datasets",
    "# ==============================================================================",
    "materials_path = DATA_PROCESSED_DIR / 'preprocessed_materials.csv'",
    "candidates_path = RESULTS_DIR / 'matching' / 'tfidf_candidate_pairs.csv'",
    "",
    "assert materials_path.exists(), f'Missing candidate materials: {materials_path}'",
    "assert candidates_path.exists(), f'Missing candidate pairs: {candidates_path}'",
    "",
    "df_materials = pd.read_csv(materials_path)",
    "df_candidates = pd.read_csv(candidates_path)",
    "",
    "print(f'Candidate Materials Count: {len(df_materials):,} records')",
    "print(f'TF-IDF Candidate Pairs Count: {len(df_candidates):,} pairs')",
    "",
    "# Schema and record integrity validations",
    "assert len(df_materials) == 6634, f'Expected 6,634 materials, got {len(df_materials)}'",
    "assert df_materials['corpus_id'].is_unique, 'Corpus IDs must be strictly unique!'",
    "assert len(df_candidates) == 22464, f'Expected 22,464 pairs, got {len(df_candidates)}'",
    "",
    "cross_pairs_count = int(df_candidates['is_cross_enterprise'].sum())",
    "intra_pairs_count = len(df_candidates) - cross_pairs_count",
    "print(f'Cross-Enterprise Pairs: {cross_pairs_count:,} ({cross_pairs_count/len(df_candidates)*100:.2f}%)')",
    "print(f'Intra-Enterprise Pairs: {intra_pairs_count:,} ({intra_pairs_count/len(df_candidates)*100:.2f}%)')",
    "assert cross_pairs_count == 930, f'Expected exactly 930 cross-enterprise pairs, got {cross_pairs_count}'",
    "",
    "print('\\nSample preprocessed records:')",
    "display(df_materials[['corpus_id', 'organization', 'cleaned_description', 'attr_diameter', 'attr_material_grade']].head(3))",
]))

# ==============================================================================
# Cell 5: Markdown - Semantic Embedding Model
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 3: Sentence Transformer Configuration",
    "",
    "We utilize **`sentence-transformers/all-MiniLM-L6-v2`** to construct dense semantic representations:",
    "- **Base Architecture:** 6-layer MiniLM transformer with 384 hidden dimensions.",
    "- **Inference-Only:** Model is pre-trained and used strictly for inference (zero fine-tuning or label fabrication).",
    "- **Normalization:** All generated vectors are $L_2$-normalized: $\\|\\mathbf{e}\\|_2 = 1.0$.",
    "- **Cosine Dot Product:** For unit vectors, cosine similarity equals the standard dot product:",
    "  $$\\text{CosineSimilarity}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2} = \\mathbf{u} \\cdot \\mathbf{v}$$",
]))

# ==============================================================================
# Cell 6: Step 3 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 3: Sentence Transformer Initialization & Hardware Verification",
    "# ==============================================================================",
    "import torch",
    "",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
    "print(f'Target Compute Device: {device.upper()}')",
    "print(f'PyTorch Version: {torch.__version__}')",
    "",
    "# Load embedding model",
    "model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)",
    "print(f'Embedding Model Identifier: {DEFAULT_EMBEDDING_MODEL}')",
    "print(f'Max Sequence Length: {model.max_seq_length}')",
    "print(f'Output Vector Dimension: 384') ",
]))

# ==============================================================================
# Cell 7: Markdown - Embedding Generation & Storage
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 4: Dense Embedding Generation & Serialization",
    "",
    "We encode all **6,634 normalized descriptions** into an embedding matrix $\\mathbf{E} \\in \\mathbb{R}^{6,634 \\times 384}$:",
    "- **Safe Handling:** Missing, null, or empty descriptions are converted to empty strings without raising errors.",
    "- **Deterministic Storage:** Saved to `models/candidate_embeddings.npy` (binary float32 array, ~10 MB).",
    "- **ID Mapping Table:** Exact index-to-`corpus_id` lookup saved to `models/embedding_id_mapping.csv`.",
]))

# ==============================================================================
# Cell 8: Step 4 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 4: Generate Embeddings and Verify Serialization Integrity",
    "# ==============================================================================",
    "embeddings_file = MODELS_DIR / 'candidate_embeddings.npy'",
    "mapping_file = MODELS_DIR / 'embedding_id_mapping.csv'",
    "",
    "descriptions = df_materials['cleaned_description'].fillna('').astype(str).tolist()",
    "corpus_ids = df_materials['corpus_id'].astype(str).tolist()",
    "",
    "if embeddings_file.exists() and mapping_file.exists():",
    "    print(f'Loading cached embeddings from {embeddings_file}...')",
    "    load_start = time.time()",
    "    embeddings, id_to_idx = load_embeddings(embeddings_file, mapping_file)",
    "    print(f'[OK] Loaded embeddings in {time.time() - load_start:.2f} seconds.')",
    "else:",
    "    print(f'Encoding {len(descriptions):,} descriptions...')",
    "    enc_start = time.time()",
    "    embeddings = generate_embeddings(",
    "        texts=descriptions,",
    "        model_name=DEFAULT_EMBEDDING_MODEL,",
    "        batch_size=64,",
    "        normalize_embeddings=True,",
    "        show_progress_bar=True,",
    "    )",
    "    print(f'[OK] Encoding completed in {time.time() - enc_start:.2f} seconds.')",
    "    save_embeddings(embeddings, corpus_ids, embeddings_file, mapping_file)",
    "    embeddings, id_to_idx = load_embeddings(embeddings_file, mapping_file)",
    "",
    "# Structural validation",
    "assert embeddings.shape == (6634, 384), f'Unexpected shape: {embeddings.shape}'",
    "assert embeddings.dtype == np.float32, f'Expected float32, got {embeddings.dtype}'",
    "assert not np.isnan(embeddings).any(), 'NaN values detected in embeddings matrix!'",
    "assert not np.isinf(embeddings).any(), 'Infinite values detected in embeddings matrix!'",
    "assert len(id_to_idx) == 6634, 'ID mapping count mismatch!'",
    "assert id_to_idx[corpus_ids[0]] == 0, 'Index alignment mismatch at position 0!'",
    "",
    "# Validate L2 unit norm",
    "norms = np.linalg.norm(embeddings, axis=1)",
    "valid_norms = norms[norms > 1e-6]",
    "assert np.allclose(valid_norms, 1.0, atol=1e-4), 'Embeddings are not properly L2-normalized!'",
    "",
    "print(f'[OK] Matrix Validation Passed: Shape={embeddings.shape}, Dtype={embeddings.dtype}, Norm=1.0, NaNs=0, Infs=0')",
]))

# ==============================================================================
# Cell 9: Markdown - Pairwise Candidate Scoring
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 5: Candidate Pair Semantic Similarity Computation",
    "",
    "We compute semantic cosine similarity **strictly for the 22,464 candidate pairs** generated in Notebook 03:",
    "- Avoids quadratic $O(N^2)$ dot products ($22\\text{ million pairs}$).",
    "- Utilizes precomputed $L_2$-normalized embeddings via vectorized NumPy dot products: $\\mathbf{u}_i \\cdot \\mathbf{v}_j$.",
    "- Preserves all upstream metadata, enterprise names, lexical similarities, and attribute conflicts.",
]))

# ==============================================================================
# Cell 10: Step 5 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 5: Compute Semantic Similarity for Candidate Pairs",
    "# ==============================================================================",
    "calc_start = time.time()",
    "semantic_sims = compute_pair_semantic_similarity(",
    "    candidate_df=df_candidates,",
    "    embeddings=embeddings,",
    "    id_to_idx=id_to_idx,",
    "    id_col_1='corpus_id_1',",
    "    id_col_2='corpus_id_2',",
    ")",
    "calc_duration = time.time() - calc_start",
    "print(f'[OK] Computed semantic similarity for {len(semantic_sims):,} pairs in {calc_duration:.4f} seconds.')",
    "",
    "# Validation of computed scores",
    "assert len(semantic_sims) == len(df_candidates), 'Output length mismatch!'",
    "assert not np.isnan(semantic_sims).any(), 'Computed scores contain NaNs!'",
    "assert (semantic_sims >= -1.0).all() and (semantic_sims <= 1.0).all(), 'Score outside [-1.0, 1.0] range!'",
    "",
    "# Augment candidate pairs DataFrame",
    "df_semantic = df_candidates.copy()",
    "df_semantic['semantic_cosine_similarity'] = semantic_sims",
    "df_semantic['lexical_similarity'] = df_semantic['tfidf_similarity']",
    "df_semantic['enterprise_1'] = df_semantic['organization_1']",
    "df_semantic['enterprise_2'] = df_semantic['organization_2']",
    "df_semantic['attribute_conflict'] = df_semantic['has_attribute_conflict']",
    "",
    "# Export augmented candidate pairs",
    "output_pairs_file = RESULTS_DIR / 'matching' / 'semantic_candidate_pairs.csv'",
    "output_pairs_file.parent.mkdir(parents=True, exist_ok=True)",
    "df_semantic.to_csv(output_pairs_file, index=False)",
    "print(f'[OK] Exported augmented candidate dataset to: {output_pairs_file}')",
    "",
    "display(df_semantic[['corpus_id_1', 'enterprise_1', 'corpus_id_2', 'enterprise_2', 'tfidf_similarity', 'semantic_cosine_similarity', 'is_cross_enterprise']].head(5))",
]))

# ==============================================================================
# Cell 11: Markdown - Similarity Distribution Statistics
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 6: Similarity Distribution & Enterprise Comparison",
    "",
    "We evaluate the distribution of semantic similarities across:",
    "1. **All Candidate Pairs (22,464):** Comprehensive candidate population.",
    "2. **Cross-Enterprise Pairs (930):** Primary target for cross-PSU harmonisation.",
    "3. **Intra-Enterprise Pairs (21,534):** Internal procurement duplicates.",
]))

# ==============================================================================
# Cell 12: Step 6 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 6: Statistical Comparison of Semantic Similarities",
    "# ==============================================================================",
    "df_cross = df_semantic[df_semantic['is_cross_enterprise']].copy()",
    "df_intra = df_semantic[~df_semantic['is_cross_enterprise']].copy()",
    "",
    "def summarize_distribution(series, name):",
    "    return pd.Series({",
    "        'Count': len(series),",
    "        'Mean': round(series.mean(), 4),",
    "        'Std': round(series.std(), 4),",
    "        'Min': round(series.min(), 4),",
    "        '25%': round(series.quantile(0.25), 4),",
    "        'Median': round(series.median(), 4),",
    "        '75%': round(series.quantile(0.75), 4),",
    "        '90%': round(series.quantile(0.90), 4),",
    "        'Max': round(series.max(), 4),",
    "    }, name=name)",
    "",
    "stats_summary = pd.DataFrame([",
    "    summarize_distribution(df_semantic['semantic_cosine_similarity'], 'All Pairs (N=22,464)'),",
    "    summarize_distribution(df_intra['semantic_cosine_similarity'], 'Intra-Enterprise (N=21,534)'),",
    "    summarize_distribution(df_cross['semantic_cosine_similarity'], 'Cross-Enterprise (N=930)'),",
    "])",
    "",
    "display(stats_summary)",
    "",
    "corr_all = df_semantic['tfidf_similarity'].corr(df_semantic['semantic_cosine_similarity'])",
    "corr_cross = df_cross['tfidf_similarity'].corr(df_cross['semantic_cosine_similarity'])",
    "print(f'\\nPearson Correlation (Lexical vs Semantic) -> Overall: {corr_all:.4f}, Cross-Enterprise: {corr_cross:.4f}')",
]))

# ==============================================================================
# Cell 13: Markdown - Cross-Enterprise Deep Dive & Synonym Discovery
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 7: Cross-Enterprise Semantic Analysis & Synonym Discovery",
    "",
    "Cross-enterprise pairs reveal critical qualitative insights:",
    "- **Synonym Discovery (High Semantic, Lower Lexical):** Where lexical TF-IDF penalizes word order or alternate terminology (e.g. `CENTRIFUGAL PUMP` vs `PUMP-CENTRIFUGAL`, or `COMPRESSOR SPARES` vs `SPARES FOR AIR COMPRESSOR`), the SentenceTransformer captures dense semantic equivalence.",
    "- **Direct Matches:** Identical canonical items shared across enterprises (`API 6D BALL VALVES`, `BEARINGS`, `AIR COMPRESSOR`).",
]))

# ==============================================================================
# Cell 14: Step 7 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 7: High-Semantic Cross-Enterprise Pairs & Synonym Discovery",
    "# ==============================================================================",
    "print('=== A. SYNONYM DISCOVERY ZONE (High Semantic >= 0.75, Lower Lexical < 0.50) ===')",
    "synonyms = df_cross[",
    "    (df_cross['semantic_cosine_similarity'] >= 0.75) &",
    "    (df_cross['tfidf_similarity'] < 0.50) &",
    "    (~df_cross['attribute_conflict'])",
    "].sort_values(by='semantic_cosine_similarity', ascending=False)",
    "",
    "for i, (_, r) in enumerate(synonyms.head(4).iterrows(), 1):",
    "    print(f'Pair {i}:')",
    "    print(f'  Enterprise 1 ({r[\"enterprise_1\"]} / {r[\"corpus_id_1\"]}): {r[\"cleaned_description_1\"]}')",
    "    print(f'  Enterprise 2 ({r[\"enterprise_2\"]} / {r[\"corpus_id_2\"]}): {r[\"cleaned_description_2\"]}')",
    "    print(f'  -> Semantic Sim: {r[\"semantic_cosine_similarity\"]:.4f} | Lexical Sim: {r[\"tfidf_similarity\"]:.4f}')",
    "    print('-'*70)",
    "",
    "print('\\n=== B. HIGH-SEMANTIC CROSS-ENTERPRISE MATCHES (Semantic >= 0.90) ===')",
    "top_matches = df_cross[",
    "    (df_cross['semantic_cosine_similarity'] >= 0.90) &",
    "    (~df_cross['attribute_conflict'])",
    "].sort_values(by='semantic_cosine_similarity', ascending=False)",
    "",
    "for i, (_, r) in enumerate(top_matches.head(4).iterrows(), 1):",
    "    print(f'Pair {i}:')",
    "    print(f'  Enterprise 1 ({r[\"enterprise_1\"]} / {r[\"corpus_id_1\"]}): {r[\"cleaned_description_1\"]}')",
    "    print(f'  Enterprise 2 ({r[\"enterprise_2\"]} / {r[\"corpus_id_2\"]}): {r[\"cleaned_description_2\"]}')",
    "    print(f'  -> Semantic Sim: {r[\"semantic_cosine_similarity\"]:.4f} | Lexical Sim: {r[\"tfidf_similarity\"]:.4f}')",
    "    print('-'*70)",
]))

# ==============================================================================
# Cell 15: Markdown - Physical Attribute Conflicts
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 8: Engineering Attribute Compatibility Audit",
    "",
    "### Why Semantic Similarity Alone is Insufficient",
    "Sentence transformers capture linguistic meaning, but **embedding similarity does NOT guarantee physical interchangeability**.",
    "- For instance, two descriptions can differ by only one dimension (e.g. `250MM NB` vs `300MM NB` pipe) and yield **semantic similarity > 0.99** despite being **100% physically incompatible**.",
    "- This proves the mandatory requirement of multi-modal feature fusion: dense semantic embeddings must be combined with explicit engineering attribute conflict detection.",
]))

# ==============================================================================
# Cell 16: Step 8 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 8: Attribute Conflicts Under High Semantic Similarity",
    "# ==============================================================================",
    "# A. Cross-enterprise pairs with physical conflicts",
    "cross_conflicts = df_cross[df_cross['attribute_conflict']].sort_values(by='semantic_cosine_similarity', ascending=False)",
    "print(f'Cross-Enterprise Attribute Conflicts: {len(cross_conflicts)} pairs\\n')",
    "",
    "for i, (_, r) in enumerate(cross_conflicts.iterrows(), 1):",
    "    print(f'Cross Conflict Pair {i}:')",
    "    print(f'  Enterprise 1 ({r[\"enterprise_1\"]} / {r[\"corpus_id_1\"]}): {r[\"cleaned_description_1\"]}')",
    "    print(f'  Enterprise 2 ({r[\"enterprise_2\"]} / {r[\"corpus_id_2\"]}): {r[\"cleaned_description_2\"]}')",
    "    print(f'  -> Semantic Sim: {r[\"semantic_cosine_similarity\"]:.4f} | Lexical Sim: {r[\"tfidf_similarity\"]:.4f}')",
    "    print(f'  -> Conflict Detail: {r[\"conflict_details\"]}')",
    "    print('-'*70)",
    "",
    "# B. Top overall pairs with near-perfect semantic similarity (>0.99) but physical conflicts",
    "high_sem_conflicts = df_semantic[",
    "    (df_semantic['semantic_cosine_similarity'] >= 0.95) &",
    "    (df_semantic['attribute_conflict'])",
    "].sort_values(by='semantic_cosine_similarity', ascending=False)",
    "",
    "print(f'\\nTop High-Semantic Attribute Conflicts Overall (Semantic >= 0.95): {len(high_sem_conflicts)} pairs\\n')",
    "for i, (_, r) in enumerate(high_sem_conflicts.head(3).iterrows(), 1):",
    "    print(f'Conflict Pair {i}:')",
    "    print(f'  Record 1 ({r[\"enterprise_1\"]} / {r[\"corpus_id_1\"]}): {r[\"cleaned_description_1\"]}')",
    "    print(f'  Record 2 ({r[\"enterprise_2\"]} / {r[\"corpus_id_2\"]}): {r[\"cleaned_description_2\"]}')",
    "    print(f'  -> Semantic Sim: {r[\"semantic_cosine_similarity\"]:.4f} | Lexical Sim: {r[\"tfidf_similarity\"]:.4f}')",
    "    print(f'  -> Conflict Detail: {r[\"conflict_details\"]}')",
    "    print('-'*70)",
]))

# ==============================================================================
# Cell 17: Markdown - Exploratory Threshold Analysis
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 9: Exploratory Threshold Retention Analysis",
    "",
    "> [!IMPORTANT]",
    "> **Methodological Clarification:** The figures reported below are **candidate retention statistics**, representing the volume of pairs retained at exploratory similarity cutoffs. They do **NOT** represent precision, recall, or F1 scores. An optimal operational cutoff cannot be selected without genuine labelled ground-truth evaluation data.",
]))

# ==============================================================================
# Cell 18: Step 9 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 9: Candidate Retention Across Exploratory Thresholds",
    "# ==============================================================================",
    "thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]",
    "retention_data = []",
    "",
    "for t in thresholds:",
    "    tot = int((df_semantic['semantic_cosine_similarity'] >= t).sum())",
    "    tot_pct = tot / len(df_semantic) * 100",
    "    crs = int((df_cross['semantic_cosine_similarity'] >= t).sum())",
    "    crs_pct = crs / len(df_cross) * 100",
    "    retention_data.append({",
    "        'Semantic Cutoff (τ)': f'≥ {t:.2f}',",
    "        'Total Pairs Retained': f'{tot:,}',",
    "        'Total Retention %': f'{tot_pct:.2f}%',",
    "        'Cross Pairs Retained': f'{crs:,}',",
    "        'Cross Retention %': f'{crs_pct:.2f}%',",
    "    })",
    "",
    "df_retention = pd.DataFrame(retention_data)",
    "display(df_retention)",
]))

# ==============================================================================
# Cell 19: Markdown - Multi-Panel Visualization
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 10: Multi-Panel Diagnostic Visualizations",
    "",
    "We visualize the relationship between lexical and semantic similarity:",
    "- **Panel A:** Scatter plot of Lexical vs Semantic similarity (Intra vs Cross-Enterprise).",
    "- **Panel B:** Cross-enterprise pairs highlighting compatible specifications vs physical attribute conflicts.",
    "- **Panel C:** Kernel Density Estimation (KDE) comparing similarity distributions.",
    "- **Panel D:** Candidate retention curve across exploratory similarity thresholds.",
]))

# ==============================================================================
# Cell 20: Step 10 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 10: Generate 4-Panel Diagnostic Figure",
    "# ==============================================================================",
    "figure_path = FIGURES_DIR / '04_semantic_vs_lexical_similarity.png'",
    "figure_path.parent.mkdir(parents=True, exist_ok=True)",
    "",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 11))",
    "plt.subplots_adjust(hspace=0.3, wspace=0.25)",
    "",
    "# Subplot 1: Scatter plot Lexical vs Semantic (All Pairs)",
    "ax1 = axes[0, 0]",
    "ax1.scatter(",
    "    df_intra['tfidf_similarity'],",
    "    df_intra['semantic_cosine_similarity'],",
    "    alpha=0.15,",
    "    s=12,",
    "    color='#4a7bb0',",
    "    label=f'Intra-Enterprise (N={len(df_intra):,})',",
    ")",
    "ax1.scatter(",
    "    df_cross['tfidf_similarity'],",
    "    df_cross['semantic_cosine_similarity'],",
    "    alpha=0.55,",
    "    s=20,",
    "    color='#e65c00',",
    "    label=f'Cross-Enterprise (N={len(df_cross):,})',",
    ")",
    "ax1.set_xlabel('Lexical Similarity (TF-IDF Cosine)', fontsize=11, fontweight='bold')",
    "ax1.set_ylabel('Semantic Cosine Similarity', fontsize=11, fontweight='bold')",
    "ax1.set_title('A. Lexical vs Semantic Similarity (Candidate Pairs)', fontsize=12, fontweight='bold')",
    "ax1.legend(loc='lower right', frameon=True)",
    "ax1.grid(True, linestyle='--', alpha=0.5)",
    "",
    "# Subplot 2: Cross-Enterprise Semantic vs Lexical with Attribute Conflicts",
    "ax2 = axes[0, 1]",
    "conflict_mask = df_cross['attribute_conflict']",
    "ax2.scatter(",
    "    df_cross.loc[~conflict_mask, 'tfidf_similarity'],",
    "    df_cross.loc[~conflict_mask, 'semantic_cosine_similarity'],",
    "    alpha=0.6,",
    "    s=25,",
    "    color='#2b8cbe',",
    "    label=f'Compatible Spec (N={(~conflict_mask).sum():,})',",
    ")",
    "ax2.scatter(",
    "    df_cross.loc[conflict_mask, 'tfidf_similarity'],",
    "    df_cross.loc[conflict_mask, 'semantic_cosine_similarity'],",
    "    alpha=0.8,",
    "    s=35,",
    "    color='#de2d26',",
    "    marker='x',",
    "    label=f'Attribute Conflict (N={conflict_mask.sum():,})',",
    ")",
    "ax2.axvline(x=0.50, color='gray', linestyle=':', alpha=0.7)",
    "ax2.axhline(y=0.70, color='gray', linestyle=':', alpha=0.7)",
    "ax2.text(0.36, 0.88, 'Synonym\\nDiscovery\\nZone', fontsize=9, color='#6a3d9a', fontweight='bold',",
    "         bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0e6ff', edgecolor='#9970ab', alpha=0.8))",
    "ax2.set_xlabel('Lexical Similarity (TF-IDF Cosine)', fontsize=11, fontweight='bold')",
    "ax2.set_ylabel('Semantic Cosine Similarity', fontsize=11, fontweight='bold')",
    "ax2.set_title('B. Cross-Enterprise Pairs & Physical Compatibility', fontsize=12, fontweight='bold')",
    "ax2.legend(loc='lower right', frameon=True)",
    "ax2.grid(True, linestyle='--', alpha=0.5)",
    "",
    "# Subplot 3: Distribution Comparison (KDE / Hist)",
    "ax3 = axes[1, 0]",
    "sns.kdeplot(df_intra['semantic_cosine_similarity'], ax=ax3, color='#4a7bb0', fill=True, alpha=0.3, label='Intra-Enterprise')",
    "sns.kdeplot(df_cross['semantic_cosine_similarity'], ax=ax3, color='#e65c00', fill=True, alpha=0.3, label='Cross-Enterprise')",
    "ax3.axvline(x=df_intra['semantic_cosine_similarity'].mean(), color='#4a7bb0', linestyle='--', label=f'Intra Mean ({df_intra[\"semantic_cosine_similarity\"].mean():.2f})')",
    "ax3.axvline(x=df_cross['semantic_cosine_similarity'].mean(), color='#e65c00', linestyle='--', label=f'Cross Mean ({df_cross[\"semantic_cosine_similarity\"].mean():.2f})')",
    "ax3.set_xlabel('Semantic Cosine Similarity', fontsize=11, fontweight='bold')",
    "ax3.set_ylabel('Density', fontsize=11, fontweight='bold')",
    "ax3.set_title('C. Semantic Similarity Distribution Density', fontsize=12, fontweight='bold')",
    "ax3.legend(loc='upper left', frameon=True)",
    "ax3.grid(True, linestyle='--', alpha=0.5)",
    "",
    "# Subplot 4: Candidate Retention Curve across Thresholds",
    "ax4 = axes[1, 1]",
    "raw_thresh = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]",
    "tot_ret = [(df_semantic['semantic_cosine_similarity'] >= t).sum() / len(df_semantic) * 100 for t in raw_thresh]",
    "crs_ret = [(df_cross['semantic_cosine_similarity'] >= t).sum() / len(df_cross) * 100 for t in raw_thresh]",
    "ax4.plot(raw_thresh, tot_ret, marker='o', color='#2b8cbe', linewidth=2.2, label='Total Pairs Retention %')",
    "ax4.plot(raw_thresh, crs_ret, marker='s', color='#e65c00', linewidth=2.2, label='Cross-Enterprise Retention %')",
    "for x, yt, yc in zip(raw_thresh, tot_ret, crs_ret):",
    "    ax4.annotate(f'{yt:.1f}%', (x, yt), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8.5, color='#2b8cbe')",
    "    ax4.annotate(f'{yc:.1f}%', (x, yc), textcoords='offset points', xytext=(0, -13), ha='center', fontsize=8.5, color='#e65c00')",
    "ax4.set_xlabel('Semantic Cosine Similarity Cutoff (τ_sem)', fontsize=11, fontweight='bold')",
    "ax4.set_ylabel('Candidate Retention Rate (%)', fontsize=11, fontweight='bold')",
    "ax4.set_title('D. Exploratory Candidate Retention Rates', fontsize=12, fontweight='bold')",
    "ax4.set_ylim(-5, 105)",
    "ax4.legend(loc='upper right', frameon=True)",
    "ax4.grid(True, linestyle='--', alpha=0.5)",
    "",
    "plt.suptitle('CSE 276 — Notebook 04: Semantic Similarity Analysis\\nModel: sentence-transformers/all-MiniLM-L6-v2 (384-d)',",
    "             fontsize=14, fontweight='bold', y=0.99)",
    "plt.tight_layout()",
    "plt.savefig(figure_path, dpi=300, bbox_inches='tight')",
    "plt.close()",
    "print(f'[OK] Diagnostic figure saved to: {figure_path}')",
]))

# ==============================================================================
# Cell 21: Markdown - Summary Report Export
# ==============================================================================
cells.append(md_cell([
    "---",
    "## Step 11: Export Structured Audit Summary Report",
    "",
    "We export the structured audit report to `results/reports/semantic_matching_summary.json` containing embedding metadata, similarity distributions, threshold retention stats, and exemplary pairs.",
]))

# ==============================================================================
# Cell 22: Step 11 Code
# ==============================================================================
cells.append(code_cell([
    "# ==============================================================================",
    "# Step 11: Structured Audit Summary Verification",
    "# ==============================================================================",
    "report_path = REPORTS_DIR / 'semantic_matching_summary.json'",
    "assert report_path.exists(), f'Missing audit report: {report_path}'",
    "",
    "with open(report_path, 'r', encoding='utf-8') as f:",
    "    summary = json.load(f)",
    "",
    "print(f'[OK] Audit report verified: {report_path}')",
    "print(f'Execution Timestamp: {summary[\"execution_timestamp\"]}')",
    "print(f'Model Identifier: {summary[\"model_configuration\"][\"model_name\"]}')",
    "print(f'Embedding Dimensions: {summary[\"model_configuration\"][\"embedding_dimension\"]}')",
    "print(f'Evaluated Candidate Pairs: {summary[\"dataset_validation\"][\"candidate_pairs_count\"]:,}')",
    "print(f'Cross-Enterprise Pairs: {summary[\"dataset_validation\"][\"cross_enterprise_pairs_count\"]:,}')",
    "",
    "print('\\n' + '='*70)",
    "print('[DONE] NOTEBOOK 04 SEMANTIC MATCHING PIPELINE COMPLETED SUCCESSFULLY!')",
    "print('='*70)",
]))

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target_path = Path("notebooks/04_semantic_matching.ipynb")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote notebook to {target_path} successfully ({len(cells)} cells)")
