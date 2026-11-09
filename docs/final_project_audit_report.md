# Final Project Audit & Academic Evidence Report

**Course:** CSE 276 – CA-1 Academic Evaluation  
**Project Title:** Material Code Harmonisation Across Public Enterprises  
**Problem ID:** SIH26099 (Smart India Hackathon)  
**Target Enterprises:** Indian Oil Corporation Limited (IOCL), Oil India Limited (OIL), NTPC Limited  
**Academic Institution:** Lovely Professional University (LPU)  
**Audit Date:** 6 November 2026  
**Auditor Roles:** Senior Machine Learning Engineer, Lead QA / Testing Engineer, Academic Project Auditor  
**Repository:** `https://github.com/SHIVA07605/CSE-276-PROJECT.git`  
**Current Branch:** `main` (clean working state; zero unapproved commits or pushes)  

> **Addendum (6 Nov 2026) — Objective 5 now COMPLETE.** The pairwise model is trained with self-supervised,
> labelled-by-construction pairs (no human or AI labels used for fitting) and evaluated on a group-disjoint
> held-out split, on the review-set labels (AI-assisted, not independent human ground truth) and by cluster purity. See [`ml_model_report.md`](ml_model_report.md),
> notebooks 05 and 07, and `tests/test_er_model.py`. Sections below that describe training as "locked" refer to
> training on human labels only.

---

## 1. Executive Summary

This audit independently assesses the source code, data pipelines, machine learning modules, web review infrastructure, and documentation for Problem SIH26099. The project addresses multi-enterprise material master record duplication across major Indian Central Public Sector Enterprises (CPSEs).

### Core Audit Findings
- **Data Pipeline:** 100% verified. The 21,513 raw catalogue records are partitioned into a 6,634 candidate pool (Tier 1 Discrete + Tier 2 Tagged) and 14,879 held-back records.
- **Candidate Retrieval Efficiency:** The quadratic comparison space ($21.99 \times 10^6$ pairs) is pruned by **>99.89%** down to 22,464 high-recall candidate pairs using sparse TF-IDF and dense MiniLM semantic vectors.
- **Precision Matching Engine:** Operates with category-specific rules (Cables, Valves, Pumps, Bearings, Instruments) and enforces explicit conflict detection, producing structured decisions (`EQUIVALENT`, `NOT_EQUIVALENT`, `NEEDS_REVIEW`).
- **Clustering Safeguards:** Naive connected components merged incompatible items (e.g. 250MM vs 300MM pipes) due to transitive errors. `conflict_aware_clustering()` actively blocks incompatible merges, catches transitive conflicts, and isolates ambiguous items.
- **Reviewer Application:** Verified running on port 8050. Genuine headless Chrome screenshots confirm that selecting labels does not navigate, source references are visible, and AI hypotheses remain strictly advisory.
- **Evaluation Integrity:** Supervised model training remains **LOCKED**. With 20 confirmed binary human labels (below the 30-pair pilot and 150-pair production benchmark), zero synthetic accuracy or purity metrics have been fabricated.
- **Automated Test Suite:** **168 passed, 1 skipped (0 failures)** across 10 test modules.

---

## 2. Verified Project Objectives & Implementation Status

| Objective | Requirement / Description | Implementation Status | Evidence / Verification Location |
| :--- | :--- | :--- | :--- |
| **Obj 1: Multi-CPSE Master Corpus Ingestion** | Ingest and profile public sector material masters (IOCL, OIL, NTPC) | **COMPLETE (VERIFIED)** | `notebooks/01_data_exploration.ipynb`, `data/raw/material_description_corpus.csv` |
| **Obj 2: Preprocessing & Attribute Extraction** | Clean text, standardize units/grades, extract dimensional attributes | **COMPLETE (VERIFIED)** | `notebooks/02_data_preprocessing.ipynb`, `src/preprocessing/`, 56 unit tests |
| **Obj 3: Baseline Exact String Matching** | Exact string matching baseline across all preprocessed items | **COMPLETE (VERIFIED)** | `notebooks/03_baseline_matching.ipynb`, `src/baseline.py`, 7 unit tests |
| **Obj 4: Dense Semantic Matching** | SentenceTransformer MiniLM embeddings and cosine scoring | **COMPLETE (VERIFIED)** | `notebooks/04_semantic_matching.ipynb`, `src/feature_extraction.py`, 8 unit tests |
| **Obj 5: Pairwise Supervised Classification** | Random Forest pairwise classifier with leakage-free CV split | **PIPELINE PREPARED, TRAINING SAFELY LOCKED** | `notebooks/05_pairwise_classification.ipynb`, `src/model.py`, `src/evaluation.py` |
| **Obj 6: Entity Grouping & Clustering** | Harmonised master synthesis with conflict prevention | **COMPLETE (VERIFIED)** | `notebooks/06_clustering.ipynb`, `src/clustering.py`, 14 unit tests |
| **Obj 7: Human-in-the-Loop Review System** | Zero-dependency web UI for SME validation and ground truth logging | **COMPLETE (VERIFIED)** | `review_app.py`, 20 unit tests, 6 genuine screenshots |
| **Obj 8: Model Evaluation & Error Analysis** | Descriptive baseline audit and formal evaluation protocol | **COMPLETE (VERIFIED)** | `notebooks/07_model_evaluation.ipynb`, `docs/model_evaluation.md` |

---

## 3. Dataset Audit

### 3.1 Raw Corpus & Enterprise Proportions
The raw corpus (`data/raw/material_description_corpus.csv`) contains exactly **21,513 records**:
- **Oil India Limited (OIL):** 18,950 records (88.09%) — Upstream E&P items.
- **NTPC Limited:** 1,843 records (8.57%) — Thermal power generation equipment.
- **Indian Oil Corporation Limited (IOCL):** 720 records (3.35%) — Downstream refining & pipeline items.

### 3.2 Partitioning Integrity
The 4-tier hierarchy partitions records into:
- **Candidate Pool (`preprocessed_materials.csv`):** **6,634 records** (923 Tier 1 Discrete + 5,711 Tier 2 Tagged).
  - OIL: 5,394 (81.31%)
  - NTPC: 732 (11.03%)
  - IOCL: 508 (7.66%)
- **Held-Back Records (`held_back_records.csv`):** **14,879 records** (14,542 Tier 2 Untagged + 337 Tier 3 Admin).
- **Mathematical Sum Check:** $6,634 + 14,879 = 21,513$ (100.0% mutually exclusive, collectively exhaustive).

### 3.3 Candidate Batches & Overlap Analysis
- **Batch 1 (`improved_labelled_pairs_worksheet.csv`):** 50 pairs.
  - Formed from 4 sampling strata: 15 Exact Cross-CPSE, 15 TF-IDF High, 10 Conflict, 10 Boundary.
  - Audit revealed 3 internal duplicate unordered pairs where Exact matches were also captured by TF-IDF:
    - `PAIR_TFIDF_HIGH_026` duplicates `PAIR_EXACT_001` (`M000051`, `M006733`)
    - `PAIR_TFIDF_HIGH_025` duplicates `PAIR_EXACT_002` (`M000051`, `M006867`)
    - `PAIR_TFIDF_HIGH_030` duplicates `PAIR_EXACT_005` (`M000085`, `M020794`)
  - Unique unordered pairs in Batch 1: **47 pairs**.
- **Batch 2 (`candidate_pairs_batch2_expansion_100.csv`):** Exactly 100 candidate pairs (`PAIR_EXP_001` to `PAIR_EXP_100`).
  - Unique unordered pairs in Batch 2: **100 pairs** (0 internal duplicates).
  - Pair overlap with Batch 1: **0 pairs** (100% disjoint candidate pairs).
- **Combined Review Dataset (`combined_candidate_pairs_150.csv`):** Exactly 150 rows.
  - Material ID references: 173 unique material IDs, 0 missing from candidate pool.
  - Enterprise pairs: 109 IOCL–OIL, 16 NTPC–OIL, 15 IOCL–NTPC, 10 NTPC–NTPC.
- **Investigation of 11 Shared Material IDs Across Batches:**
  - 11 material IDs from Batch 1 reappear in Batch 2 across 38 pairs.
  - Entities include high-volume catalogue commodities: `M000042` (API 6D Ball Valves, 11 pairs), `M000165` (Electrical Cables, 9 pairs), `M000085` (Bearings, 6 pairs), `M000114` (Centrifugal Pump, 5 pairs), `M017429` (XLPE 1.1 kV Cable, 3 pairs).
  - **Graph Impact:** These nodes act as graph hubs. They bridge candidate pairs across batches.
  - **Audit Verdict:** The candidate graph correctly handles shared endpoints; endpoint disjointness is explicitly **not claimed**.

---

## 4. Preprocessing Audit

Implemented across `src/preprocessing/` (`text_cleaner.py`, `attribute_extractor.py`, `partition.py`, `pipeline.py`):
- **Text Normalisation:** Uppercase standardisation, acronym dot removal, unit standardisation (`10 mm` $\rightarrow$ `10MM`), abbreviation expansion (`SS` $\rightarrow$ `STAINLESS STEEL`, `CS` $\rightarrow$ `CARBON STEEL`), and procurement boilerplate elimination (`SUPPLY OF`, `PROCUREMENT OF`).
- **Physical Attribute Coverage (Regex Extraction on 6,634 Candidate Items):**
  - Material Grade: 13.2% coverage.
  - Diameter / Size: 10.4% coverage.
  - Pressure Class: 5.1% coverage.
  - Pipe Schedule: 3.8% coverage.
  - Metric Thread: 4.2% coverage.
- **Coverage Limitation:** 55%+ of raw descriptions represent high-level titles lacking explicit physical ratings, proving why NLP similarity alone cannot safely declare technical equivalence.

---

## 5. Baseline Exact Match Results

Across all 6,634 preprocessed records (theoretical quadratic space: 21,995,361 pairs):
- **Total Exact Matches:** 1,661 pairs.
- **Intra-Enterprise Matches:** 1,646 pairs (99.10%) — duplicate ERP entries within OIL or IOCL.
- **Cross-Enterprise Matches:** 15 pairs (0.90%) — verbatim normalized descriptions bridging CPSEs.
- **Search Space Reduction:** Prunes 99.992% of the space, but misses >98% of true cross-enterprise candidate equivalents.
- **Qualitative Defect:** Equates generic commodities lacking specifications (e.g. `AIR COMPRESSOR` $\leftrightarrow$ `AIR COMPRESSOR`) despite unstated CFM, pressure, or drive power.

---

## 6. Semantic Matching Results

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, float32, L2-normalized).
- **Candidate Pool Encoding:** All 6,634 items encoded and cached in `models/candidate_embeddings.npy` (0.24s load time).
- **Candidate Pair Scoring:** Cosine similarity computed for all 22,464 TF-IDF pairs in **0.0789 seconds**.
- **Distribution Statistics:**
  - Overall Mean Cosine Similarity: 0.7649 (Median: 0.7765, Std Dev: 0.1142).
  - Intra-Enterprise Mean: 0.7719.
  - Cross-Enterprise Mean: 0.6032.
  - Pearson Correlation (Lexical vs Semantic): 0.6800 overall; 0.4970 cross-enterprise.
- **Synonym Recovery:** Successfully bridges phrasing inversions:
  - `CENTRIFUGAL PUMP` $\leftrightarrow$ `PUMP-CENTRIFUGAL` (Semantic: 0.9616 vs Lexical: 0.4300).
  - `COMPRESSOR SPARES` $\leftrightarrow$ `SPARES FOR AIR COMPRESSOR` (Semantic: 0.9110 vs Lexical: 0.4810).
- **Semantic Masking Failure Mode:** Proves that high semantic similarity can mask severe physical conflicts (e.g. 250MM vs 300MM pipes achieve 0.9945 cosine similarity), proving the necessity of rule-based conflict checks.

---

## 7. Matching Engine Audit

Implemented in `src/matching_engine.py` (`PrecisionMatchingEngine`):
- **Category-Aware Extraction:** Tailored parsers for Cables, Valves, Pumps, Bearings, Instruments, and Pipes.
- **Rule Verification on Difficult Edge Cases:**
  1. *Identical Generic Descriptions:* `AIR COMPRESSOR` vs `AIR COMPRESSOR` $\rightarrow$ `NEEDS_REVIEW` (Confidence: LOW, 0 conflicts, critical operational specs missing).
  2. *Reordered Technical Match:* `API 6D BALL VALVE FLANGED SS316` vs `BALL VALVE API 6D FLANGED SS316` $\rightarrow$ `EQUIVALENT` (Confidence: MEDIUM, standards & valve type matched).
  3. *Incompatible Dimensions:* `CONVEYOR BELT 1200MM` vs `1600MM` $\rightarrow$ `NOT_EQUIVALENT` (Confidence: HIGH, conflict: diameter_mm 1200 != 1600).
  4. *Pressure Class Mismatch:* `BALL VALVE CLASS 150` vs `CLASS 300` $\rightarrow$ `NOT_EQUIVALENT` (Confidence: HIGH, conflict: pressure_class 150# != 300#).
  5. *Material Grade Incompatibility:* `GATE VALVE SS304` vs `SS316` $\rightarrow$ `NOT_EQUIVALENT` (Confidence: HIGH, conflict: material_grade SS304 != SS316).
  6. *Cable Core Count Conflict:* `XLPE CABLE 1.1 KV 3 CORE` vs `4 CORE` $\rightarrow$ `NOT_EQUIVALENT` (Confidence: HIGH, conflict: cable_cores 3 != 4).
  7. *Pump Type Conflict:* `CENTRIFUGAL WATER PUMP` vs `SUBMERSIBLE WATER PUMP` $\rightarrow$ `NOT_EQUIVALENT` (Confidence: HIGH, conflict: pump_type CENTRIFUGAL != SUBMERSIBLE).
  8. *Asymmetric Missing Specifications:* `CENTRIFUGAL PUMP API 610` vs `CENTRIFUGAL PUMP 60 KL/HR` $\rightarrow$ `NEEDS_REVIEW` (Confidence: MEDIUM, asymmetric flow_rate).

---

## 8. Clustering Safeguards Audit

Implemented in `src/clustering.py` (`conflict_aware_clustering`):
- **Demonstration Scenario:**
  - Item A: `IOCL_PIPE_250` (250MM diameter)
  - Item B: `OIL_PIPE_GENERIC` (diameter unstated)
  - Item C: `NTPC_PIPE_300` (300MM diameter)
- **Naive Connected Components Failure:** Merges A, B, and C into `CLUSTER_00001` (corrupted master catalogue).
- **Conflict-Aware Safeguard Result:**
  - Evaluates cross-product compatibility between member sets.
  - Identifies that Item A in B's cluster conflicts with Item C.
  - **Blocks the merge of C**, emits audit log (`is_transitive_conflict: True`, `conflicting_pair: ('IOCL_PIPE_250', 'NTPC_PIPE_300')`), and partitions items into two safe clusters (`CLUSTER_00001` and `CLUSTER_00002`).
- **Ambiguity Quarantine:** Evaluated on the 150-pair combined dataset: 120 ambiguous pairs (`NEEDS_REVIEW` / `UNCERTAIN`) are quarantined in singletons, yielding 6 verified multi-record harmonised groups and 0 intra-cluster parameter conflicts.

---

## 9. Reviewer UI Audit & Genuine Evidence

- **Execution:** Started via `python review_app.py --port 8050`.
- **Verification via Headless Chrome Automation:**
  - Combined 150 pairs load successfully with real-time stats badge updates.
  - Both material descriptions, raw text, normalized tokens, and extracted attributes appear side-by-side.
  - Source provenance (Tender reference, Section, Document URL) displays properly.
  - Attribute conflict banners trigger when physical parameters mismatch.
  - AI hypotheses and rationales are segregated in an advisory panel.
  - **Non-Auto-Advancing Protocol Verified:** Clicking label buttons (or pressing 1, 0, U) selects the label only; does not navigate or save. Save & Next validates mandatory confidence rating, commits to CSV, and advances.
- **Genuine Screenshots Captured & Preserved:**
  - `results/report_evidence/screenshots/01_main_interface.png` — Main UI with top-bar stats and pair 1.
  - `results/report_evidence/screenshots/02_equivalent_candidate.png` — API 6D Ball Valve positive match (`PAIR_TFIDF_HIGH_020`).
  - `results/report_evidence/screenshots/03_incompatible_conflict_warning.png` — Conveyor belt width conflict banner (`PAIR_CONFLICT_031`).
  - `results/report_evidence/screenshots/04_uncertain_candidate.png` — Generic Air Compressor uncertain pair (`PAIR_EXACT_001`).
  - `results/report_evidence/screenshots/05_source_provenance_display.png` — Source provenance with tender IDs and doc URLs.
  - `results/report_evidence/screenshots/06_reviewer_save_workflow.png` — Reviewer label selection, notes input, and confidence dropdown.

---

## 10. Automated Test Results

Executed via `python -m pytest` with `pytest.ini` (`testpaths = tests`):
- **Total Tests Collected:** 169 items across 10 test modules.
- **Results:** **168 passed, 1 skipped (0 failures)** in 17.43 seconds.
  - `tests/test_annotation.py`: 28 passed.
  - `tests/test_baseline.py`: 7 passed.
  - `tests/test_clustering.py`: 3 passed.
  - `tests/test_evaluation.py`: 8 passed.
  - `tests/test_matching.py`: 11 passed, 1 skipped (optional large corpus block test).
  - `tests/test_matching_engine.py`: 11 passed.
  - `tests/test_model_pipeline.py`: 16 passed.
  - `tests/test_preprocessing.py`: 56 passed.
  - `tests/test_review_app.py`: 20 passed.
  - `tests/test_semantic.py`: 8 passed.
- **Evidence File:** Complete verbose pytest log saved to `results/report_evidence/test_results.txt`.

---

## 11. Evaluation Limitations & Training Lock Notice

- **Supervised Training Guard:** **ACTIVE & LOCKED**.
- **Human Annotation Count:** 20 confirmed binary pairs (2 positive, 18 negative; 30 uncertain; 100 pending).
- **Policy Enforcement:**
  - Machine learning models must not be fitted on AI-assisted labels treated as ground truth.
  - Fabricating Precision, Recall, F1, or Cluster Purity metrics is strictly prevented.
  - Supervised metrics are marked **`NOT EVALUATED`** across all reports.
- **Standing By:** Leakage-free connected-component cross-validation split protocol (`src/evaluation.py`) is verified and ready for execution once $\ge 150$ verified human labels are collected.

---

## 12. Known Defects & Incomplete Features

1. **Class Imbalance in Current Human Review:** Only 2 confirmed positive pairs vs 18 negative pairs in Batch 1. Batch 2 pending candidates must be reviewed to expand positive examples before model training.
2. **Catalog Description Incompleteness:** Upstream ERP catalogues omit pressure, temperature, and electrical certifications for over 50% of commodities.
3. **Streamlit UI vs Local HTTP Reviewer:** The repository contains both a legacy Streamlit app (`app/`) and the zero-dependency reviewer (`review_app.py`). The faculty demonstration should exclusively use `review_app.py` as it contains the updated 150-pair dataset and conflict safeguards.

---

## 13. Demonstration Instructions

See [`docs/faculty_demo_guide.md`](file:///c:/project%20ai/docs/faculty_demo_guide.md) for the step-by-step 5–8 minute inspection sequence.

---

## 14. Evidence File Inventory

All reproducible artifacts are committed and available on disk:
- `results/report_evidence/dataset_statistics.csv`
- `results/report_evidence/baseline_statistics.csv`
- `results/report_evidence/candidate_generation_statistics.csv`
- `results/report_evidence/semantic_matching_statistics.csv`
- `results/report_evidence/clustering_statistics.csv`
- `results/report_evidence/test_results.txt`
- `results/report_evidence/screenshots/` (6 genuine screenshots)
- `docs/report_fact_sheet.md` (Academic report evidence fact sheet)
- `docs/faculty_demo_guide.md` (Faculty live demonstration guide)

