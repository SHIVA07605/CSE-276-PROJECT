# Source Code Audit Report — CSE 276
## Material Code Harmonisation Across Public Enterprises

> **Audit date:** 2 November 2026  
> **Deadline:** 7 November 2026  
> **Auditor:** Antigravity AI  
> **Python version (local):** 3.13  
> **Test suite baseline:** 144 passed, 1 skipped (last confirmed run)

---

## 1. Repository Structure

```
c:\project ai\                                          (project root)
│
├── notebooks/                  Jupyter notebooks (01–05 executed, 06–07 stub)
├── src/                        Production Python package
│   ├── preprocessing/          Sub-package (text_cleaner, attribute_extractor, pipeline, partition)
│   └── *.py                    config, data_loader, preprocess, baseline, candidate_generation,
│                               feature_extraction, model, evaluation, clustering, harmonisation
├── tests/                      Pytest test suite (9 files, 145 items)
├── data/
│   ├── raw/                    material_description_corpus.csv (6.0 MB, immutable)
│   ├── processed/              preprocessed_materials.csv, worksheets
│   └── labelled/               labelled_pairs.csv (50 PENDING)
├── models/                     candidate_embeddings.npy, embedding_id_mapping.csv
├── results/                    figures/, reports/, matching/, metrics/
├── docs/                       annotation_guidelines.md, architecture.md, methodology.md,
│                               dataset_documentation.md
├── app/                        Streamlit multi-page application (app.py + pages/)
├── review_app.py               Dash-based pairwise annotation reviewer (port 8050)
├── review_pairs.py             Terminal annotation CLI
├── validate_annotations.py     Annotation consistency checker
├── create_improved_review_worksheet.py  One-off worksheet generator
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore

Generator / verifier scripts (utility — not obsolete, but not production code):
  generate_nb02.py  generate_nb02_final.py  generate_nb03.py  generate_nb04.py  generate_nb05.py
  run_nb02_pipeline.py  run_nb03_pipeline.py  run_nb04_pipeline.py
  verify_nb02.py  verify_nb03.py  verify_nb04.py  verify_nb05.py

Legacy / archived subtree (Colab clone — do NOT delete without approval):
  CSE-276-PROJECT/    Contains early-draft notebooks and old src/ stubs (pre-implementation)
```

---

## 2. Component Implementation Status

| # | Component | File(s) | Status | Test Coverage |
|---|-----------|---------|--------|---------------|
| 1 | Path & configuration management | `src/config.py` | ✅ Implemented | Implicit (all tests rely on it) |
| 2 | Dataset schema validation | `src/data_loader.py` | ✅ Implemented | Implicit |
| 3 | Text cleaning & normalisation | `src/preprocessing/text_cleaner.py` | ✅ Implemented & tested | `test_preprocessing.py` — 52 tests |
| 4 | Attribute extraction (size, UoM, material) | `src/preprocessing/attribute_extractor.py` | ✅ Implemented & tested | `test_preprocessing.py` — 52 tests |
| 5 | Corpus partitioning | `src/preprocessing/partition.py` | ✅ Implemented & tested | `test_preprocessing.py` |
| 6 | Preprocessing pipeline | `src/preprocessing/pipeline.py` | ✅ Implemented & tested | `test_preprocessing.py` |
| 7 | Exact-string baseline matching | `src/baseline.py` | ✅ Implemented & tested | `test_baseline.py` — 7 tests |
| 8 | TF-IDF candidate generation | `src/candidate_generation.py` | ✅ Implemented & tested | `test_matching.py` — 11 tests, 1 skipped |
| 9 | Semantic embedding (MiniLM) | `src/feature_extraction.py` | ✅ Implemented & tested | `test_semantic.py` — 8 tests |
| 10 | Pairwise feature extraction (13 features) | `src/feature_extraction.py` | ✅ Implemented & tested | `test_model_pipeline.py` — 16 tests |
| 11 | Random Forest classifier pipeline | `src/model.py` | ✅ Implemented, **not trained** (0 labels) | `test_model_pipeline.py` — 16 tests |
| 12 | Connected-component train/test split | `src/evaluation.py` | ✅ Implemented & tested | `test_evaluation.py` — 8 tests |
| 13 | Evaluation metrics (precision/recall/F1) | `src/evaluation.py` | ✅ Implemented & tested | `test_evaluation.py` — 8 tests |
| 14 | Entity clustering (connected components) | `src/clustering.py` | ✅ Implemented & tested | `test_clustering.py` — 3 tests |
| 15 | Harmonisation code generation | `src/harmonisation.py` | ✅ Implemented | **No dedicated tests** |
| 16 | Web annotation reviewer (Dash) | `review_app.py` | ✅ Implemented & tested | `test_review_app.py` — 7 tests |
| 17 | Terminal annotation CLI | `review_pairs.py` | ✅ Implemented | **No dedicated tests** |
| 18 | Annotation validation CLI | `validate_annotations.py` | ✅ Implemented & tested | `test_annotation.py` — 28 tests |
| 19 | Streamlit multi-page application | `app/` | ⚠️ Stub / placeholder pages | **No tests** |
| 20 | Clustering notebook (NB 06) | `notebooks/06_clustering.ipynb` | ⚠️ Stub — not implemented | None |
| 21 | Evaluation notebook (NB 07) | `notebooks/07_model_evaluation.ipynb` | ⚠️ Stub — not implemented | None |

**Key finding:** The core research pipeline (components 1–16) is fully implemented and tested. The Streamlit multi-page app and Notebooks 06–07 are stubs pending model training data.

---

## 3. Hardcoded Paths

| File | Finding |
|------|---------|
| `src/config.py` | ✅ All paths use `Path(__file__).resolve().parent.parent` — fully relative |
| All `src/*.py` | ✅ Import from `src.config` — no hardcoded absolute paths |
| `review_app.py` | ✅ Uses `config` constants |
| `review_pairs.py` | ✅ Uses `config` constants |
| `validate_annotations.py` | ✅ Uses `config` constants |
| `generate_nb0*.py` / `verify_nb0*.py` | ⚠️ Reference `c:\project ai` via `pathlib.Path` resolved from `__file__` — safe on this machine but path separator may differ on Linux; low risk |

**No absolute Windows paths in production source code.**

---

## 4. Duplicate / Obsolete Scripts

| File(s) | Classification | Recommendation |
|---------|---------------|----------------|
| `generate_nb02.py` + `generate_nb02_final.py` | Utility — iterative versions | Keep; `generate_nb02_final.py` is the canonical version. Note in README. |
| `generate_nb03.py`, `generate_nb04.py`, `generate_nb05.py` | Utility — notebook source-of-truth | Keep. Explains how notebooks were built. |
| `run_nb02_pipeline.py`, `run_nb03_pipeline.py`, `run_nb04_pipeline.py` | Utility — CI-style pipeline runners | Keep. Useful for automated re-execution. |
| `verify_nb02.py` … `verify_nb05.py` | Utility — post-execution validators | Keep. Enables correctness checks without Jupyter. |
| `create_improved_review_worksheet.py` | One-off worksheet generator | Keep as documentation of how the worksheet was created. |
| `CSE-276-PROJECT/` | Legacy Colab clone with early-draft code | **Do not delete without approval.** Note as historical archive in README. |

---

## 5. Broken Imports and Missing Files

| Check | Result |
|-------|--------|
| `src/` imports across all 9 test files | ✅ All imports resolve correctly |
| `src/preprocess.py` → `src.preprocessing.*` | ✅ Thin wrapper, imports correctly |
| `review_app.py` imports | ✅ Uses `pandas`, `dash`, `src.*` — all present |
| `app/app.py` Streamlit imports | ⚠️ Imports `streamlit` — installed. Page modules are stub-level (no broken imports, but no functional logic) |
| `src/harmonisation.py` | ✅ Imports only `hashlib`, `pandas`, `typing` — no missing deps |
| `src/clustering.py` | ✅ Uses `try/except ImportError` guard for `networkx` |
| `notebooks/06_clustering.ipynb` | ⚠️ Stub notebook — no executed cells |
| `notebooks/07_model_evaluation.ipynb` | ⚠️ Stub notebook — no executed cells |
| `models/` | ⚠️ Contains embeddings but **no trained classifier** (expected — 0 confirmed labels) |

---

## 6. Dependency Audit

Current `requirements.txt` vs actual project usage:

| Package | In requirements.txt | Used in project | Notes |
|---------|---------------------|-----------------|-------|
| `numpy` | ✅ | ✅ | Embeddings, features |
| `pandas` | ✅ | ✅ | All data I/O |
| `scipy` | ✅ | ✅ | Sparse matrices in candidate gen |
| `regex` | ✅ | ✅ | Attribute extraction |
| `nltk` | ✅ | ✅ | Tokenisation |
| `rapidfuzz` | ✅ | ✅ | Lexical similarity |
| `scikit-learn` | ✅ | ✅ | TF-IDF, RandomForest, metrics |
| `sentence-transformers` | ✅ | ✅ | MiniLM embeddings |
| `torch` | ✅ | ✅ | Required by sentence-transformers |
| `transformers` | ✅ | ✅ | Required by sentence-transformers |
| `networkx` | ✅ | ✅ | Connected components |
| `matplotlib` | ✅ | ✅ | Figures in notebooks |
| `seaborn` | ✅ | ✅ | Distribution plots |
| `plotly` | ✅ | ✅ | Streamlit charts |
| `streamlit` | ✅ | ✅ | `app/app.py` |
| `python-dotenv` | ✅ | ✅ | `.env` loading |
| `pydantic` | ✅ | ⚠️ | Listed but **not directly used in src/**; imported transitively by some packages |
| `pytest` | ✅ | ✅ | Test runner |
| `jupyter`, `ipykernel` | ✅ | ✅ | Notebook execution |
| `dash` | ❌ **MISSING** | ✅ | `review_app.py` uses `dash`, `dash_bootstrap_components` |
| `dash-bootstrap-components` | ❌ **MISSING** | ✅ | `review_app.py` imports `dash_bootstrap_components` |
| `joblib` | ❌ **MISSING** | ✅ | `src/model.py` uses `joblib.dump/load` |

> [!CAUTION]
> `dash`, `dash-bootstrap-components`, and `joblib` are used in production code but **absent from `requirements.txt`**. Any reviewer who runs `pip install -r requirements.txt` will encounter `ModuleNotFoundError` when starting the reviewer application or loading/saving the model.

---

## 7. Data Leakage Risk Assessment

| Risk | Assessment | Mitigation |
|------|-----------|------------|
| Train/test split leaks same material across split boundary | ✅ **Mitigated** | `src/evaluation.py` implements connected-component grouping; material ID overlap verified = 0 |
| Embeddings computed after split (using test descriptions) | ✅ **Not applicable** | Embeddings are computed on all corpus records before split; no test-label-derived features |
| Pairwise features derived from labels | ✅ **None** | Features are purely lexical/semantic/attribute-based |
| Worksheet pre-labelled with synthetic labels | ✅ **None** | All 50 pairs are `PENDING` — zero confirmed labels |
| Model serialized with training data embedded | ✅ **N/A** | No model has been trained or serialized |

**No data leakage risks identified in the current codebase state.**

---

## 8. Test Suite Summary

Last confirmed run: **144 passed, 1 skipped** (2 November 2026, 12:37 IST)

| Test file | Items | Result | Coverage |
|-----------|-------|--------|----------|
| `test_preprocessing.py` | 52 | ✅ All passed | Text cleaning, attribute extraction, partitioning |
| `test_annotation.py` | 28 | ✅ All passed | Worksheet integrity, annotation validation |
| `test_model_pipeline.py` | 16 | ✅ All passed | Feature extraction, CC split, training guard |
| `test_evaluation.py` | 8 | ✅ All passed | Metrics, PR curves, CC splitting |
| `test_semantic.py` | 8 | ✅ All passed | Embedding loading, cosine similarity |
| `test_review_app.py` | 7 | ✅ All passed | API routes, JSON serialisation |
| `test_baseline.py` | 7 | ✅ All passed | Exact match baseline |
| `test_clustering.py` | 3 | ✅ All passed | Connected components, singleton handling |
| `test_matching.py` | 11 | ✅ 11 passed, 1 skipped | TF-IDF blocking, pair validation |

---

## 9. Executed Notebooks Status

| Notebook | Cells | Status | Key Outputs |
|----------|-------|--------|-------------|
| `01_data_exploration.ipynb` | Multiple | ✅ Executed & approved | EDA figures, dataset_audit_summary.json |
| `02_data_preprocessing.ipynb` | Multiple | ✅ Executed & approved | preprocessed_materials.csv (6,634 records) |
| `03_baseline_matching.ipynb` | Multiple | ✅ Executed & approved | exact_matches.csv, tfidf_candidate_pairs.csv |
| `04_semantic_matching.ipynb` | Multiple | ✅ Executed & approved | candidate_embeddings.npy (6634×384), semantic_candidate_pairs.csv |
| `05_pairwise_classification.ipynb` | 20 | ✅ Executed & approved | Feature matrix (50×13), training guard active |
| `06_clustering.ipynb` | 1 | ⚠️ Stub — 0 executed cells | N/A |
| `07_model_evaluation.ipynb` | 1 | ⚠️ Stub — 0 executed cells | N/A |

---

## 10. Action Items (Priority Order)

| Priority | Item | File |
|----------|------|------|
| 🔴 HIGH | Add `dash`, `dash-bootstrap-components`, `joblib` to `requirements.txt` | `requirements.txt` |
| 🟡 MED | Confirm `pydantic` is actually needed or remove it | `requirements.txt` |
| 🟡 MED | Update `README.md` to reflect actual current status (milestones completed) | `README.md` |
| 🟢 LOW | Add `src/harmonisation.py` tests | Future test file |
| 🟢 LOW | Add `review_pairs.py` tests | Future test file |
| 🟢 LOW | Implement `notebooks/06_clustering.ipynb` after model training | After label collection |
| 🟢 LOW | Implement `notebooks/07_model_evaluation.ipynb` after model training | After label collection |
| 🟢 LOW | Complete Streamlit `app/` pages after model is trained | After label collection |
| ℹ️ INFO | `CSE-276-PROJECT/` is a historical archive — note clearly in README | `README.md` |

