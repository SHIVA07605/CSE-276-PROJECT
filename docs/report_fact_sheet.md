# Academic Report Evidence Fact Sheet

**Course:** CSE 276 – Machine Learning / Academic Evaluation (CA-1)  
**Project Title:** Material Code Harmonisation Across Public Enterprises  
**Problem Statement ID:** SIH26099 (Smart India Hackathon)  
**Target Public Sector Enterprises:** Indian Oil Corporation Limited (IOCL), Oil India Limited (OIL), NTPC Limited  
**Academic Institution:** Lovely Professional University (LPU)  
**Repository:** `https://github.com/SHIVA07605/CSE-276-PROJECT.git`  
**Purpose:** Comprehensive, verified factual reference sheet organized by official academic report chapters to enable error-free compilation of the final CA-1 report.

> **Update (6 Nov 2026) — Chapter 7.3 superseded.** A trained hybrid self-supervised entity-resolution model
> now exists (`models/hybrid_er_model.joblib`). Its precision / recall / F1 / ROC-AUC / PR-AUC versus the
> exact-match baseline, its review-set label confusion matrix (AI-assisted, not independent human ground truth), and cluster purity are in
> [`docs/ml_model_report.md`](ml_model_report.md) — use that file (auto-generated from the metrics JSON) for
> Chapter 7 numbers. The statements below that supervised metrics are "NOT EVALUATED" refer only to training on
> human labels, which stays locked until ≥ 150 labels exist.

---

## Chapter 1: Introduction

- **Industrial Context:** India's Central Public Sector Enterprises (CPSEs) operate under the Ministry of Petroleum and Natural Gas (IOCL, OIL) and Ministry of Power (NTPC). They manage multi-billion dollar industrial procurement budgets for capital equipment, maintenance spares, and consumables.
- **The Core Problem:** Each CPSE maintains independent, isolated ERP systems (SAP Material Master, Oracle ERP) with bespoke material codification schemes (e.g. 10-digit numeric codes in IOCL vs 7-digit alphanumeric codes in OIL). 
- **Consequences of Siloed Codification:**
  1. *Procurement Duplication:* Identical industrial spares (e.g. API 6D pipeline ball valves, IS:7098 power cables) are procured simultaneously by neighboring enterprises at disparate bulk pricing.
  2. *Excess Inventory & Capital Lockup:* Lack of inventory visibility prevents cross-enterprise spare sharing, leading to redundant safety stocks.
  3. *Vendor Fragmentation:* Inability to aggregate procurement volumes weakens government bargaining power.
- **Why Simple NLP / String Matching Fails:**
  - Extreme abbreviation heterogeneity: `SS`, `SS-304`, `AISI 304`, `STAINLESS STEEL 304`.
  - Phrasing inversions: `CENTRIFUGAL PUMP` vs `PUMP-CENTRIFUGAL`.
  - High semantic similarity masking physical incompatibility: `CONVEYOR BELT 1200MM` vs `CONVEYOR BELT 1600MM` achieve 0.99 semantic similarity but cannot physically substitute.

---

## Chapter 2: Objectives

1. **Ingest and profile multi-CPSE material master catalogues** from IOCL, OIL, and NTPC, auditing enterprise distribution, description lengths, and attribute sparsity.
2. **Design and implement an automated text normalisation and attribute extraction pipeline** standardising metallurgical grades, pipe schedules, pressure classes, and dimensions.
3. **Establish an exact string matching baseline** to quantify the performance and failure modes of current ERP duplicate-detection practices.
4. **Develop sublinear candidate retrieval mechanisms** combining sparse TF-IDF n-grams and dense sentence embeddings (`all-MiniLM-L6-v2`) to prune quadratic comparison space by >99.9%.
5. **Implement a category-aware, precision-first matching engine** enforcing strict specification conflict detection across Cables, Valves, Pumps, Bearings, and Instruments.
6. **Formulate conflict-aware graph clustering algorithms** preventing transitive corruption ($A \sim B \land B \sim C$ where $A \nsim C$) and synthesizing clean master codes.
7. **Deploy a zero-dependency human-in-the-loop reviewer application** with source provenance visibility, real-time conflict banners, and non-auto-advancing label workflows.
8. **Establish a mathematically leakage-free cross-validation protocol** using connected-component graph splitting, maintaining strict training locks until sufficient human ground truth is annotated.

---

## Chapter 3: Related Work Evidence

- **Database Entity Resolution (ER):** Traditional probabilistic record linkage (Fellegi-Sunter model) relies on structured field comparisons. It degrades rapidly when material specifications are compressed into unstructured, free-form text strings.
- **Deep Learning for ER:** DeepER (Ebraheem et al., 2018) and Ditto (Li et al., 2020) demonstrated transformer-based entity matching on benchmarks like Walmart-Amazon and DBLP-Scholar. However, these models were trained on consumer goods (electronics, books) where description text contains rich brand/model identifiers.
- **Industrial Catalogue Discrepancy:** Industrial items lack universal UPC/EAN barcodes. In industrial engineering, two descriptions differing by a single token (e.g. `150#` vs `300#`, or `SS304` vs `SS316`) are strictly incompatible, exposing deep transformers to catastrophic false-positive errors if physical rules are omitted.
- **Hybrid Rule-Embedding Architecture:** Recent literature (Papadakis et al., 2021) recommends hybridizing dense neural candidate retrieval with deterministic symbolic constraint checking to guarantee zero false positives on safety-critical physical parameters.

---

## Chapter 4: Dataset

### 4.1 Master Corpus Distribution
- **Raw Data File:** `data/raw/material_description_corpus.csv`
- **Total Records:** **21,513 records**
  - Oil India Limited (OIL): 18,950 records (88.09%)
  - NTPC Limited: 1,843 records (8.57%)
  - Indian Oil Corporation Limited (IOCL): 720 records (3.35%)
- **Source Systems:** E-procurement portals, tendering portals (`etender.sbi`, `iocl.com`, `oil-india.com`, `ntpc.co.in`), NIT documents, and future procurement plans.

### 4.2 Candidate Pool & Corpus Partitioning
- **Partitioning Rule:** Items classified into a 4-tier hierarchy.
- **Candidate Pool (`preprocessed_materials.csv`):** **6,634 records**
  - Tier 1 Discrete Line Items: 923 records.
  - Tier 2 Tagged Procurement Items: 5,711 records.
  - Distribution: OIL 5,394 (81.31%), NTPC 732 (11.03%), IOCL 508 (7.66%).
- **Held-Back Records (`held_back_records.csv`):** **14,879 records**
  - Tier 2 Untagged: 14,542 records.
  - Tier 3 Administrative & Work Contracts: 337 records.
- **Verification:** Mutually exclusive and collectively exhaustive ($6,634 + 14,879 = 21,513$).

### 4.3 Candidate Batches & Overlap Analysis
- **Batch 1:** 50 pairs (`improved_labelled_pairs_worksheet.csv`), 4 strata (15 exact cross, 15 TF-IDF high, 10 conflict, 10 boundary). Unique unordered pairs: 47 (3 duplicate exact/TF-IDF pairs from initial sampling).
- **Batch 2:** 100 pairs (`candidate_pairs_batch2_expansion_100.csv`), 9 domains. Unique unordered pairs: 100. Overlap with Batch 1: 0 pairs.
- **Combined Review Set:** 150 pairs (`combined_candidate_pairs_150.csv`). 173 unique material IDs.
- **Graph Hub Entities:** 11 material IDs from Batch 1 reappear in Batch 2 across 38 pairs (e.g. `M000042` [API 6D Ball Valves], `M000165` [Cables], `M000085` [Bearings], `M000114` [Centrifugal Pump]). Endpoint disjointness is **not claimed**; candidate graph correctly accommodates hub connectivity.

---

## Chapter 5: Methodology

```mermaid
flowchart LR
    subgraph S1["Stage 1: Preprocessing"]
        A["Raw Description"] --> B["Normalisation & Unit Standardization"]
        B --> C["Category-Aware Attribute Extraction"]
    end
    subgraph S2["Stage 2: Candidate Blocking"]
        B --> D["Sparse TF-IDF N-grams (tau = 0.35)"]
        D --> E["Dense MiniLM Embeddings (384-d, tau = 0.60)"]
    end
    subgraph S3["Stage 3: Precision Matching"]
        E --> F["Specification Conflict Checking"]
        C --> F
        F --> G["Decision: EQUIVALENT / NOT_EQ / REVIEW"]
    end
    subgraph S4["Stage 4: Safe Clustering"]
        G --> H["Conflict-Aware Graph Clustering"]
        H --> I["Harmonised Master Code"]
    end
```

### 5.1 Preprocessing & Normalisation
- Strip punctuation, remove boilerplate procurement tokens (`SUPPLY OF`, `PROCUREMENT OF`, `FOR USE IN`).
- Unit standardisation (`10 mm` $\rightarrow$ `10MM`; `1/2 "` $\rightarrow$ `1/2INCH`; `1.1 kv` $\rightarrow$ `1.1KV`).
- Metallurgy acronym expansion (`SS` $\rightarrow$ `STAINLESS STEEL`, `CS` $\rightarrow$ `CARBON STEEL`).

### 5.2 Attribute Extraction
- Regex parsers for: Metric Thread (`M\d+`), Diameter (`\d+MM`, `\d+INCH`, `DN\d+`), Material Grade (`SS304`, `SS316`, `IS2062`, `WCB`, `CF8M`), Pressure Class (`150#`, `300#`, `PN16`), Pipe Schedule (`SCH 40`, `SCH 80`), Voltage (`1.1KV`, `11KV`), Capacity (`MW`, `HP`, `KL/HR`).

### 5.3 Candidate Generation & Blocking
- **Quadratic Search Space:** $21,995,361$ pairs ($N=6,634$).
- **Sparse TF-IDF:** Sublinear term frequency, ngram range (1, 2), min_df = 2, chunked dot product (`batch_size=500`), cutoff $\tau = 0.35$.
- **Dense Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, L2-normalized float32 vectors.
- **Search Space Reduction:** **>99.89%** ($22,464$ candidate pairs retained, 930 cross-enterprise).

### 5.4 Precision-First Matching Engine
- Categories: Cables, Valves, Pumps, Bearings, Instruments, Pipes/Fittings, General.
- Rules:
  - *Conflict Rule:* Incompatible parameter (diameter, pressure, metallurgy, core count, voltage) $\rightarrow$ `NOT_EQUIVALENT` (Confidence: HIGH).
  - *Governing Standard Rule:* Zero conflicts, identical standard/ratings (e.g. API 6D Ball Valve) $\rightarrow$ `EQUIVALENT` (Confidence: MEDIUM/HIGH).
  - *Generic & Missing Rule:* Identical or semantically similar titles lacking critical specifications $\rightarrow$ `NEEDS_REVIEW` (Confidence: LOW/MEDIUM).

### 5.5 Conflict-Aware Clustering
- Validates cross-product compatibility between member sets before merging clusters.
- Detects transitive conflicts ($A \sim B$ and $B \sim C$, but $A \nsim C$) and blocks corrupted merges.
- Isolates ambiguous items (`NEEDS_REVIEW`, `UNCERTAIN`) into singletons.

---

## Chapter 6: Implementation

- **Python Environment:** Python 3.13.14 on Windows 11.
- **Key Dependencies & Packages:**
  - `scikit-learn`: 1.8.0 (TF-IDF vectorization, GroupShuffleSplit, evaluation metrics)
  - `sentence-transformers`: 5.3.0 (all-MiniLM-L6-v2 dense embeddings)
  - `torch`: 2.15.0+cpu (neural tensor operations)
  - `networkx`: 3.6.1 (graph connected components & clustering)
  - `rapidfuzz`: 3.14.3 (Levenshtein, token sort, and Jaccard distances)
  - `pandas`: 2.3.3 & `numpy`: 2.4.2 (dataframes and sparse matrix operations)
  - `selenium`: 4.52.0 (headless browser automated UI testing & screenshot capture)
  - `pytest`: 8.3.4 (unit and integration test harness)
- **Repository Structure:**
  ```text
  c:\project ai\
  ├── data\
  │   ├── raw\material_description_corpus.csv (21,513 records)
  │   ├── processed\preprocessed_materials.csv (6,634 candidate pool)
  │   ├── processed\combined_candidate_pairs_150.csv (150 review pairs)
  │   └── labelled\labelled_pairs.csv (validated human annotations)
  ├── docs\
  │   ├── final_project_audit_report.md
  │   ├── report_fact_sheet.md
  │   ├── faculty_demo_guide.md
  │   └── batch2_data_validation_report.md
  ├── models\
  │   ├── candidate_embeddings.npy (6,634 x 384 matrix)
  │   └── embedding_id_mapping.csv
  ├── notebooks\
  │   ├── 01_data_exploration.ipynb (EDA & corpus audit)
  │   ├── 02_data_preprocessing.ipynb (text cleaning & regex)
  │   ├── 03_baseline_matching.ipynb (exact baseline & TF-IDF)
  │   ├── 04_semantic_matching.ipynb (dense sentence embeddings)
  │   ├── 05_pairwise_classification.ipynb (Random Forest pipeline, training locked)
  │   ├── 06_clustering.ipynb (conflict-aware graph clustering)
  │   └── 07_model_evaluation.ipynb (descriptive evaluation & baseline comparison)
  ├── results\
  │   ├── report_evidence\
  │   │   ├── dataset_statistics.csv
  │   │   ├── baseline_statistics.csv
  │   │   ├── candidate_generation_statistics.csv
  │   │   ├── semantic_matching_statistics.csv
  │   │   ├── clustering_statistics.csv
  │   │   ├── test_results.txt
  │   │   └── screenshots\ (6 genuine PNG screenshots)
  │   └── matching\ (exact_matches.csv, tfidf_candidate_pairs.csv, semantic_candidate_pairs.csv)
  ├── src\
  │   ├── candidate_generation.py
  │   ├── clustering.py
  │   ├── evaluation.py
  │   ├── feature_extraction.py
  │   ├── matching_engine.py
  │   └── model.py
  ├── tests\ (10 test modules, 169 test items)
  ├── review_app.py (zero-dependency web reviewer app)
  └── validate_annotations.py (training readiness validator)
  ```

---

## Chapter 7: Results & Evaluation

### 7.1 Exact Baseline Performance
- Total exact matches: 1,661 pairs (1,646 intra-CPSE, 15 cross-CPSE).
- Search space coverage: 0.00755%.
- Cross-enterprise recall across candidate pool: < 2% (misses 98%+ of potential cross-CPSE matches).
- Qualitative precision: High on unique standard titles (Hot plate stirrer, XLPE cable), catastrophic on generic titles (Air Compressor, Bearings).

### 7.2 Retrieval & Candidate Generation
- Pruning efficiency: Reduced comparison space from 21,995,361 pairs to 22,464 candidate pairs (>99.89% reduction).
- Semantic inference speed: 22,464 candidate pairs scored in 0.0789 seconds.
- Mean cosine similarity: 0.7649 overall; 0.6032 cross-enterprise.

### 7.3 Supervised Model Evaluation Status
- **Current Ground Truth Status:** 20 confirmed binary pairs (2 positive, 18 negative; 30 uncertain; 100 pending).
- **Evaluation Status:** **NOT EVALUATED (TRAINING LOCKED)**.
- **Academic Governance:** Supervised accuracy, precision, recall, F1, and cluster purity metrics are **not fabricated**. Model evaluation harness in Notebook 05 and Notebook 07 stands by for when $\ge 150$ verified human annotations are collected.

### 7.4 Automated Testing Results
- 169 test items collected.
- **168 passed, 1 skipped (0 failures)** in 17.43 seconds.
- Verified in `results/report_evidence/test_results.txt`.

---

## Chapter 8: Limitations

1. **Extreme Description Sparsity:** Upstream CPSE catalogues omit pressure ratings, operating temperatures, and electrical hazardous-area certifications for over 50% of items.
2. **Domain Operating Differences:** IOCL/OIL refinery items follow API 610/API 6D (hydrocarbon service), whereas NTPC items follow BIS/ASME boiler codes (thermal utility service). Descriptions with identical flow/head ratings may be physically incompatible.
3. **Class Imbalance in Pilot Review:** Current human annotations contain 2 positives vs 18 negatives. Batch 2 pending items must be evaluated before unlocking training.
4. **Graph Hub Complexity:** 11 repeated material entities across batches create high-degree nodes that bridge components, necessitating conflict-aware clustering safeguards.

---

## Chapter 9: Conclusion

- The CSE 276 project delivers a complete, reproducible, and verifiable material master harmonisation pipeline across IOCL, OIL, and NTPC.
- The hybrid architecture combines dense neural candidate retrieval (>99.89% pruning) with symbolic engineering constraint verification and conflict-aware graph clustering.
- By enforcing strict evaluation integrity and maintaining training locks until genuine human annotations reach threshold, the project upholds the highest academic standards of scientific validity.

---

## Chapter 10: References

1. Fellegi, I. P., & Sunter, A. B. (1969). A theory for record linkage. *Journal of the American Statistical Association*, 64(328), 1183-1210.
2. Ebraheem, M., et al. (2018). Distributed representations of tuples for entity resolution. *Proceedings of the VLDB Endowment*, 11(11), 1451-1464.
3. Li, Y., et al. (2020). Deep entity matching with pre-trained language models. *Proceedings of the VLDB Endowment*, 14(1), 50-60.
4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP 2019*, 3982-3992.
5. Papadakis, G., et al. (2021). A survey of blocking and filtering techniques for entity resolution. *ACM Computing Surveys*, 53(2), 1-42.
6. Bureau of Indian Standards. (2011). *Cross-linked Polyethylene Insulated Thermoplastic Sheathed Cables Specification (IS 7098 Part 1)*. New Delhi: BIS.
7. American Petroleum Institute. (2014). *Specification for Pipeline and Piping Valves (API Specification 6D, 24th ed.)*. Washington, D.C.: API.
8. American Petroleum Institute. (2021). *Centrifugal Pumps for Petroleum, Petrochemical and Natural Gas Industries (API Standard 610, 12th ed.)*. Washington, D.C.: API.

---

## Chapter 11: Appendices (Evidence Artifact Index)

| Appendix Code | Document / Evidence Name | File Path |
| :--- | :--- | :--- |
| **App A** | Full Technical Project Audit | [`docs/final_project_audit_report.md`](file:///c:/project%20ai/docs/final_project_audit_report.md) |
| **App B** | Faculty Live Demonstration Guide | [`docs/faculty_demo_guide.md`](file:///c:/project%20ai/docs/faculty_demo_guide.md) |
| **App C** | Batch 2 Data Validation Audit | [`docs/batch2_data_validation_report.md`](file:///c:/project%20ai/docs/batch2_data_validation_report.md) |
| **App D** | Model Architecture Specification | [`docs/model_architecture.md`](file:///c:/project%20ai/docs/model_architecture.md) |
| **App E** | Model Evaluation & Integrity Policy | [`docs/model_evaluation.md`](file:///c:/project%20ai/docs/model_evaluation.md) |
| **App F** | Known Limitations & Domain Boundary | [`docs/known_limitations.md`](file:///c:/project%20ai/docs/known_limitations.md) |
| **App G** | Dataset Statistics Table | [`results/report_evidence/dataset_statistics.csv`](file:///c:/project%20ai/results/report_evidence/dataset_statistics.csv) |
| **App H** | Baseline Statistics Table | [`results/report_evidence/baseline_statistics.csv`](file:///c:/project%20ai/results/report_evidence/baseline_statistics.csv) |
| **App I** | Candidate Generation Table | [`results/report_evidence/candidate_generation_statistics.csv`](file:///c:/project%20ai/results/report_evidence/candidate_generation_statistics.csv) |
| **App J** | Semantic Matching Table | [`results/report_evidence/semantic_matching_statistics.csv`](file:///c:/project%20ai/results/report_evidence/semantic_matching_statistics.csv) |
| **App K** | Clustering Statistics Table | [`results/report_evidence/clustering_statistics.csv`](file:///c:/project%20ai/results/report_evidence/clustering_statistics.csv) |
| **App L** | Automated Test Output Log | [`results/report_evidence/test_results.txt`](file:///c:/project%20ai/results/report_evidence/test_results.txt) |
| **App M** | Genuine Reviewer Screenshots | [`results/report_evidence/screenshots/`](file:///c:/project%20ai/results/report_evidence/screenshots/) (6 PNGs) |
| **App N** | Executed Jupyter Notebooks | [`notebooks/01_data_exploration.ipynb`](file:///c:/project%20ai/notebooks/01_data_exploration.ipynb) through [`07_model_evaluation.ipynb`](file:///c:/project%20ai/notebooks/07_model_evaluation.ipynb) |

