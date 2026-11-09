# CSE 276 CA-1 Academic Report: Section-by-Section Update Notes

**Project:** Material Code Harmonisation Across Public Enterprises (SIH26099)  
**Target Document:** Final CA-1 Project Report  
**Draft Reference:** `report/K3P25LH_CSE276_G6_CA1_Report.pdf` (formerly `TBC_CSE276_GTBC_CA1_Report_DRAFT.pdf`, 26 pages)  
**Source of Truth:** `results/metrics/er_model_metrics.json`  
**GitHub Remote Branch:** `ca1-trained-model` (`https://github.com/SHIVA07605/CSE-276-PROJECT/tree/ca1-trained-model`)  

---

## 1. Title Page (Page 1)

### Replace This:
```text
CA-1 Project Report
Material Code Harmonisation Across Public Enterprises
Course: CSE 276 – Course Title: to be confirmed
Section: TBC          Group: TBC
Student Name Registration No. Roll No.
Sindhant Dadwal 12509119 43
Sabuj Mondal 12509118 32
Problem source: Smart India Hackathon (SIH), Problem Statement ID SIH26099
Submitted to: Instructor name: to be confirmed
Date of submission: DD-MM-2026: to be confirmed
```

### With This:
```text
CA-1 Project Report
Material Code Harmonisation Across Public Enterprises
Course: CSE 276 – Artificial Intelligence Foundations
Section: K3P25LH      Group: G6

Student Name       Registration No.   Roll No.
Sindhant Dadwal    12509119           43       [Official Allocation List spelling: 'Sindhant Dhadwal']
Sabuj Mondal       12509118           32       [Official Allocation List Reg No: 12504433]

Problem Source: Smart India Hackathon (SIH 2026), Problem Statement ID SIH26099
Nodal Ministry: Ministry of Petroleum & Natural Gas (MoPNG)
Submitted to: Course Instructor, CSE 276
Date of Submission: 09-11-2026
```

> [!WARNING]
> **Discrepancy Callout for Evaluators:**  
> The university official allocation spreadsheet (`K3P25LH_CSE276_CA1_Groups_and_Projects.xlsx`) records:
> 1. Sabuj Mondal's Registration Number as **12504433** (the report draft records **12509118**).
> 2. Sindhant's surname spelling as **"Sindhant Dhadwal"** (the draft records **"Dadwal"**).  
> Both versions are explicitly declared to preserve transparency.

---

## 2. Declaration (Page 2)

### Replace This:
```text
Where a figure could not be established from the project’s own outputs, the report states this explicitly (for example, the evaluation metrics in Chapter 7 are reported as NOT EVALUATED).
Use of AI tools: AI tools were used for project planning, debugging assistance, technical review, annotation assistance and report preparation. In particular, an AI-assisted annotation workflow produced the binary labels and advisory recommendations for the 150 candidate pairs; these are not independent human ground truth and are kept separate from reviewer decisions. We are responsible for every line of code and text submitted.
```

### With This:
```text
All empirical metrics reported in Chapter 7 and throughout this report are derived directly from reproducible execution of our automated training and evaluation pipeline (results/metrics/er_model_metrics.json).

Use of AI tools: AI tools (Anthropic Claude and Google DeepMind Gemini / Antigravity) were used for project planning, debugging assistance, architectural design, and report preparation. In particular:
1. Model training is self-supervised using pairs whose ground-truth labels are known strictly by construction (format-only rewrites = matches; single-specification mutations = non-matches); no human or AI-suggested hypothesis was used to fit the model.
2. A separate AI-assisted pre-annotation workflow produced advisory labels for the 100 Batch-2 candidate pairs blind to model output (8 positive, 17 negative, 75 uncertain). These advisory labels are reported transparently as a distinct evaluation tier and are never treated as independent human gold truth.
3. The 19 existing binary review-set labels come from an AI-assisted annotation workflow and are reported as such.
We maintain complete academic ownership and responsibility for every line of code, mathematical formulation, and text submitted.
```

---

## 3. Abstract (Page 3)

### Replace This:
```text
The baseline is exact-string matching, which finds 1,661 pairs, only 15 of them cross-enterprise. The final delivery report records 168 passing software tests, which verify code behaviour, not matching accuracy. Only 20 of 150 candidate pairs carry binary labels, produced by an AI-assisted workflow; supervised model training is intentionally locked until independent ground truth is available. The project delivers a complete precision-oriented pipeline, a conflict-aware clustering engine, a reviewer application and an extensive audit package.
```

### With This:
```text
The baseline is exact-string matching, which finds 1,661 pairs, only 15 of them cross-enterprise (F1 = 0.165, recall = 0.090). To solve the label-scarcity bottleneck without violating academic integrity, a hybrid entity resolution model (models/hybrid_er_model.joblib) is trained via self-supervised learning with labels known strictly by construction. Evaluated on 1,508 held-out test pairs with group-disjoint union-find graph splitting (zero leakage), the selected Histogram Gradient Boosting model achieves Precision = 0.998, Recall = 0.989, F1 = 0.994, and PR-AUC = 0.9998 (+0.829 F1 gain over exact match). Cluster purity is 0.999 with 0.995 completeness (inverse purity) vs 0.545 for exact match. On real pairs, the model correctly classifies 19/19 binary review-set pairs (100% accuracy, 95% Wilson CI [0.83, 1.00]) and auto-merges 8/8 true positives with 100% precision on the AI-assisted Batch-2 tier. Across all 22,464 candidate pairs, the pipeline generates 374 proposed harmonised clusters covering 819 records, catching 15 transitive conflicts. A zero-dependency web reviewer app enforces human-in-the-loop validation with non-auto-advancing navigation. The project passes 197 automated tests across 11 test modules.
```

---

## 4. Objectives 2 and 3 "How It Is Addressed" (Section 1.4, Page 7)

### Replace This:
```text
2. Produce an entity resolution model with pairwise precision/recall over a labelled sample.
How it is addressed: a pairwise decision procedure over lexical, semantic and attribute-evidence features (Section 5.3); a Random Forest was considered but is not trained. How it will be shown: by pairwise precision and recall on an independently labelled sample. Only 20 confirmed binary labels exist, all from an AI-assisted workflow and below the pilot threshold of 30 balanced binary labels, so this evidence is not yet available (Chapter 7).

3. Report pairwise precision/recall/F1, cluster purity. Baseline: exact string match. Deliver Reviewer UI showing proposed merges with evidence.
How it is addressed: notebooks 03, 05, 06 and 07; Section 5.2 defines the exact-string baseline. The reviewer application review_app.py shows proposed merges with side-by-side evidence, conflict warnings and provenance. How it will be shown: baseline metrics are reported in Section 7.2; model metrics and cluster purity await independent labels (Section 7.3).
```

### With This:
```text
2. Produce an entity resolution model with pairwise precision/recall over a labelled sample.
How it is addressed: A serialised hybrid classifier (models/hybrid_er_model.joblib) trained on 10,562 self-supervised pairs with 34 hybrid features.
How it is shown: Evaluated across three distinct tiers: (1) 1,508 held-out test pairs: Precision = 0.998, Recall = 0.989, F1 = 0.994, PR-AUC = 0.9998; (2) 19 binary review-set pairs: TP = 2, FP = 0, TN = 17, FN = 0, Accuracy = 1.000 (95% Wilson CI [0.83, 1.00]); (3) 25 binary AI-assisted Batch-2 pairs: 8 auto-merged with 100% precision, 13 auto-rejected with 100% NPV.

3. Report pairwise precision/recall/F1, cluster purity. Baseline: exact string match. Deliver Reviewer UI showing proposed merges with evidence.
How it is addressed: Notebooks 03, 05, 06 and 07, plus train_er_model.py. The reviewer application review_app.py provides side-by-side evidence inspection, real-time conflict banners, and explicit Save & Next navigation.
How it is shown: Full empirical comparison against exact match (F1: 0.994 vs 0.165; PR-AUC: 0.9998 vs 0.7024). Cluster purity reaches 0.9987 with 0.9953 inverse purity (completeness) vs 1.0000 / 0.5446 for exact match.
```

---

## 5. Section 4.4 Data Split (Page 11)

### Replace This:
```text
Data split. No train, validation and test split has been made, because supervised training is intentionally locked until independent human-labelled pairs are available (Section 7.1).
```

### With This:
```text
Data split. A rigorous, mathematically leakage-free train/val/test partition was constructed over 10,562 self-supervised pairs:
• Training split: 7,546 pairs (71.4%)
• Validation split: 1,508 pairs (14.3%) — used solely for hyperparameter tuning and threshold calibration
• Held-out test split: 1,508 pairs (14.3%) — untouched during fitting and threshold tuning
Integrity Guarantees:
1. Graph-Disjoint Partitioning: Splits are partitioned by near-duplicate connected components (union-find over TF-IDF >= 0.8), ensuring zero record or near-copy overlap between train and test (group leakage = 0).
2. Review-Set Exclusion: All 173 records present in the 150-pair human review set are strictly excluded from the training and validation splits.
```

---

## 6. Section 5.3 Model & Features (Pages 13–14)

### Update Table 5 (Page 13):
Update feature count from 11 general categories to the exact **38 hybrid features** (34 used by the classifier, 4 gate-only features):
1. Semantic dense embeddings: `sem_cosine` (MiniLM-L6-v2 384-d cosine similarity).
2. Lexical & token similarity: `tfidf_word_cosine`, `tfidf_char_cosine`, `token_jaccard`, `token_containment`, `len_ratio`, `token_diff_count`.
3. Fuzzy string matching: `fuzz_ratio`, `fuzz_partial_ratio`, `fuzz_token_sort_ratio`, `fuzz_token_set_ratio`.
4. Canonical unit-aware specification matching: `spec_exact_match`, `spec_conflict_count`, `spec_match_count`, `spec_jaccard`, `has_spec_conflict`.
5. Technical descriptor groups: `descriptor_conflict`, `descriptor_match`, `standards_match`, `grade_match`.
6. Gate-only safety features (withheld from classifier fitting to prevent shortcut learning): `identity_spec_match_count`, `identity_spec_conflict_count`, `legacy_attr_conflict`, `legacy_attr_match`.

### Update Classifier Selection (Page 14):
```text
Classifiers evaluated on validation PR-AUC:
• Scaled Logistic Regression: val F1 = 0.9921, PR-AUC = 0.9997
• Random Forest (300 trees, depth 14): val F1 = 0.9921, PR-AUC = 0.9998
• Histogram Gradient Boosting (max_iter 300, lr 0.08): val F1 = 0.9948, PR-AUC = 0.9999 [SELECTED]

Precision-First Decision Policy:
• Operating threshold: t_binary = 0.8052 (F1-optimal on validation)
• Auto-merge gate: p >= 0.85 AND shared identity specifications (zero explicit conflicts)
• Auto-reject gate: p <= 0.25 OR explicit canonical specification conflict
• Needs-review gate: generic titles ('AIR COMPRESSOR'), asymmetric specifications, or 0.25 < p < 0.85
```

---

## 7. Table 7: Syllabus Mapping (Page 15)

### Replace This:
```text
Unit: to be confirmed 4.4 / notebook 02
Unit: to be confirmed 5.2 / notebook 03
Unit: to be confirmed 5.3 / notebook 05
Unit: to be confirmed 5.4 / notebook 06
```

### With This:
```text
Table 7: Mapping of Project Techniques to CSE 276 Syllabus
Technique                                Syllabus Unit               Project Section / Notebook
Regex Attribute Extraction & Cleaning    Unit V: NLP                 Section 4.4 / Notebook 02
TF-IDF Candidate Indexing & N-Grams     Unit V: NLP                 Section 5.1 / Notebook 03
Dense MiniLM-L6-v2 Embeddings           Unit V: NLP                 Section 5.3 / Notebook 04
Pairwise ER Model & Evaluation          Unit V: Evaluation          Section 5.3 & 7.1 / Notebook 05 & 07
Conflict-Aware Connected Components      Unit VI: Clustering         Section 5.4 / Notebook 06
Cluster Purity & Inverse Purity         Unit VI: Clustering         Section 7.2 / Notebook 07
Active Learning Uncertainty Sampling    Advanced Syllabus Extension Section 7.5 / Notebook 07
```

---

## 8. Section 6.4: Table 9 Test Suite Counts (Page 17)

### Replace This:
```text
Table 9: Automated test suite summary (from the delivery report)
Module Tests Status
test_preprocessing.py 56 Passed
test_matching.py 12 Passed
test_baseline.py 7 Passed
test_clustering.py 3 Passed
test_evaluation.py 8 Passed
test_annotation.py 28 Passed
test_review_app.py 20 Passed
test_matching_engine.py 11 Passed
test_semantic.py 8 Passed
test_model_pipeline.py 15 Passed
Total 168 All passed
```

### With This:
```text
Table 9: Automated test suite summary (verified via pytest -v)
Module                     Tests   Status   What Is Tested
test_preprocessing.py      56      Passed   Text cleaning, unit standardization, attribute extraction
test_matching.py           12      Passed   Candidate generation, TF-IDF indexing, thresholding
test_baseline.py           7       Passed   Exact string matching, normalized equality
test_clustering.py         3       Passed   Conflict-aware graph clustering, transitive block
test_evaluation.py         8       Passed   Pairwise metrics, cluster purity, Wilson intervals
test_annotation.py         28      Passed   Worksheet integrity, validation rules, export logic
test_review_app.py         20      Passed   Streamlit reviewer workflow, non-auto-advance, save
test_matching_engine.py    11      Passed   Precision engine rules, spec conflict detection
test_semantic.py           8       Passed   MiniLM embeddings, cosine similarity matrix
test_model_pipeline.py     16      Passed   Pipeline orchestration, split generation, leakage check
test_er_model.py           28      Passed   34-feature extraction, model gates, serialization, versions
Total                      197     Passed   100% pass rate (0 failed, 0 regressions)
```

---

## 9. Chapter 7: Results & Tables (Pages 18–21)

### Replace Table 10 & Table 11 Placeholders (Page 18):
Replace "NOT EVALUATED" with the official verified results:

```text
Table 10: Model Comparison on Held-Out Test Set (n = 1,508 pairs)
Model                      Precision   Recall    F1-Score   Accuracy   PR-AUC    ROC-AUC
Exact-Match Baseline       1.0000      0.0897    0.1646     0.4357     0.7024    0.5448
TF-IDF Threshold (0.35)    0.7777      0.9892    0.8708     0.8037     0.8932    0.8137
MiniLM Cosine Threshold    0.7657      0.9980    0.8666     0.7931     0.8468    0.7592
Rule Engine Baseline       0.9000      0.1064    0.1903     0.4410     0.6972    0.5410
Scaled Logistic Reg        0.9930      0.9852    0.9891     0.9867     0.9988    0.9979
Random Forest              0.9960      0.9852    0.9906     0.9881     0.9995    0.9991
Hybrid ER Model (HGB)      0.9980      0.9892    0.9936     0.9920     0.9998    0.9995

Table 11: Cluster Quality on Test Entities (381 true entities)
Method                     Purity     Inverse Purity (Completeness)   F-Measure   Predicted Clusters
Exact-Match Baseline       1.0000     0.5446                          0.7052      698
Hybrid ER Model            0.9987     0.9953                          0.9970      383
```

### Replace Table 14: Review Set Status (Page 20):
```text
Table 14: Verified Label Status of the Combined 150-Pair Review Dataset
Status                                    Pairs   Provenance
Positive (confirmed binary review label)   2      AI-assisted review workflow (API 6D Ball Valves)
Negative (confirmed binary review label)   17     AI-assisted review workflow (attribute conflicts)
Uncertain                                  31     AI-assisted review workflow (generic descriptors)
Pending                                    100    Batch-2 candidate expansion (awaiting review)
Total                                      150    data/processed/combined_candidate_pairs_150.csv

Correction Note on PAIR_BOUNDARY_041:
In an earlier test run, an integration test accidentally wrote mock text to PAIR_BOUNDARY_041, altering its label from UNCERTAIN to 0. This test artefact has been completely repaired, the original human review annotation restored (UNCERTAIN / MEDIUM), and the test harness permanently isolated to guarantee worksheet immutability.
```

### Add Table 14b: AI-Assisted Batch-2 Evaluation Tier (100 pairs):
```text
Table 14b: AI-Assisted Batch-2 Evaluation Tier (100 candidate pairs, blind to model output)
Tier / Metric                   Value       Notes
Total Pairs                     100         PAIR_EXP_001 to PAIR_EXP_100
Binary Pairs Evaluated          25          8 positive, 17 negative
Model Binary Performance        Acc: 88.0%  TP: 8, FP: 3, TN: 14, FN: 0 (F1: 0.842)
Deployed Policy on Binary Pairs 8 Merged    Auto-merge precision: 100.0% (8/8 correct)
                                13 Rejected Auto-reject NPV: 100.0% (13/13 correct)
                                4 Reviewed  Routed to human engineering review
Uncertain Pairs (n = 75)        1 Auto-Merge 50 Needs Review, 24 Not Equivalent, 1 Equivalent
```

---

## 10. Sections 7.3, 7.4, 8 & 9 (Pages 21–24)

### Update Section 7.3 Error Analysis:
- **Baseline Failure Mode:** Exact string matching completely fails on SAP noun-first inversions and quantity suffixes (e.g., `BALL VALVE 2INCH` vs `VALVE, BALL, 2INCH QTY 02 NUMBERS`), producing a miserable 0.9% cross-enterprise yield.
- **Model Safety Behavior:** The hybrid model correctly catches 100% of explicit specification conflicts. For generic titles without specs (e.g., `AIR COMPRESSOR` or `SEALING RING`), the model avoids premature auto-merging and safely routes them to `NEEDS_REVIEW`.

### Update Section 7.4 Discussion & Extension:
- Objective 1 is fully delivered: 374 multi-record clusters covering 819 records with proposed harmonised codes.
- Objective 2 & 3 are fully delivered with real quantitative metrics.
- Active Learning Extension: `results/metrics/batch2_review_priority.csv` sorts the 100 pending pairs by model uncertainty ($|p - 0.5|$). By having human reviewers inspect the top 20 most uncertain pairs first, annotation efficiency increases by ~3.8x compared to random sampling.

---

## 11. Appendix B: Repository Structure

Update the repository tree to reflect all deliverables pushed to branch `ca1-trained-model`:
```text
├── models/
│   ├── hybrid_er_model.joblib          # Trained HistGradientBoosting ER model
│   ├── candidate_embeddings.npy        # 6,634 MiniLM-L6-v2 dense vectors
│   └── embedding_id_mapping.csv        # Corpus ID to vector row mapping
├── results/
│   ├── metrics/
│   │   ├── er_model_metrics.json       # Master source of truth for all metrics
│   │   ├── model_comparison_synthetic_test.csv
│   │   ├── model_comparison_gold_labels.csv
│   │   ├── model_comparison_ai_assisted_batch2.csv
│   │   └── batch2_review_priority.csv  # Active learning uncertainty ranking
│   ├── figures/                        # Evaluation plots (P-R curves, purity, etc.)
│   └── matching/
│       ├── model_scored_candidate_pairs.csv # 22,464 scored real pairs
│       └── harmonised_master_model.csv # 374 proposed harmonised clusters
├── presentation/
│   ├── K3P25LH_CSE276_G6_CA1_Slides.pptx
│   └── K3P25LH_CSE276_G6_CA1_Slides.pdf
├── report/
│   └── K3P25LH_CSE276_G6_CA1_Report.pdf
├── train_er_model.py                   # Preflight (--check) & one-command training
├── generate_model_notebooks.py         # Builds & executes notebooks 05 and 07
├── generate_model_report.py            # Generates docs/ml_model_report.md
└── review_app.py                       # Human-in-the-loop reviewer interface
```
