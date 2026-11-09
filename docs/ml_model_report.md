# Hybrid Self-Supervised Entity-Resolution Model — Results Report

**Course:** CSE 276 – CA-1 Academic Project
**Project:** Material Code Harmonisation Across Public Enterprises (IOCL, OIL, NTPC)
**Model artefact:** `models/hybrid_er_model.joblib` (version `2.0.0-hybrid-self-supervised`)
**Selected classifier:** `hist_gradient_boosting` · **Reproduce:** `python train_er_model.py` (≈2 min on CPU)

> All numbers below are generated automatically from `results/metrics/er_model_metrics.json`
> by `generate_model_report.py`; nothing is typed by hand.

---

## 1. Why a self-supervised model

Only **19 human-confirmed binary labels** exist (2 positive, 17 negative) — too few to train or
fairly evaluate a supervised classifier, and the project policy forbids training on AI-suggested
hypotheses. The model is therefore trained on **pairs whose labels are known by construction**,
built from the real 6,634-record candidate pool:

| Source | Pairs | Label | Construction |
|---|---|---|---|
| synthetic_positive | 6,599 | 1 | real record vs. format-only rewrite of itself (boilerplate, quantity suffix, noun-first order, abbreviation, unit spacing, typo, plural) |
| synthetic_hard_negative | 1,812 | 0 | real record vs. rewrite with exactly one specification changed (size, rating, grade, type, standard) |
| random_cross_commodity_negative | 1,344 | 0 | two real records from different commodity classes |
| real_spec_conflict_negative | 807 | 0 | real candidate pair whose canonical number+unit specifications are disjoint |

* Total pairs: **10,562** (6,599 positive / 3,963 negative); split train/val/test = 7,546 / 1,508 / 1,508.
* Splits are formed by **near-duplicate group** (union-find over TF-IDF ≥ 0.8): group leakage = **0**.
* All **173 records of the human review set** (and any record with the same description) are excluded from training, so the review-set evaluation is untouched.

## 2. Model

* **Features (38):** MiniLM cosine; word & char TF-IDF cosine; Jaccard / containment; 4 RapidFuzz ratios; token-difference counts; canonical, unit-aware specification matches/conflicts (`10 MM`=`10MM`, `1.1 KV`=`1100V`, `CLASS 150`=`150#`); descriptor-group conflicts (BALL vs GATE valve, CENTRIFUGAL vs SUBMERSIBLE pump, XLPE vs PVC …); standards / grades / bearing series; `PrecisionMatchingEngine` decision & conflict counts; legacy regex attribute matches. The classifier uses **34** of them: specification counts (construction-shortcut risk) and the legacy string-equality attributes (false conflicts on `150#` vs `CLASS 150`) feed only the decision gates.
* **Classifiers compared:** Logistic Regression (scaled), Random Forest, Histogram Gradient Boosting — balanced class weights, selected on validation PR-AUC.
* **Decision policy (precision-first):** explicit spec conflict → `NOT_EQUIVALENT`; p ≤ 0.25 → `NOT_EQUIVALENT`; p ≥ 0.85 with shared identity specifications → `EQUIVALENT`; generic titles, asymmetric specifications and mid scores → `NEEDS_REVIEW`. Binary operating threshold (F1-optimal on validation) = 0.805.

## 3. Held-out test results (group-disjoint, n = 1,508)

| Model | Precision | Recall | F1 | Accuracy | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| logistic_regression | 0.9930 | 0.9852 | 0.9891 | 0.9854 | 0.9979 | 0.9988 |
| random_forest | 0.9960 | 0.9852 | 0.9906 | 0.9874 | 0.9991 | 0.9995 |
| **hist_gradient_boosting** | 0.9980 | 0.9892 | 0.9936 | 0.9914 | 0.9995 | 0.9998 |
| exact_match_baseline | 1.0000 | 0.0897 | 0.1646 | 0.3873 | 0.5448 | 0.7024 |
| tfidf_threshold_baseline | 0.7777 | 0.9892 | 0.8708 | 0.8024 | 0.8137 | 0.8932 |
| minilm_threshold_baseline | 0.7657 | 0.9980 | 0.8666 | 0.7931 | 0.7592 | 0.8468 |
| rule_engine_baseline | 0.9000 | 0.1064 | 0.1903 | 0.3906 | 0.5410 | 0.6972 |

* **F1 vs official exact-match baseline:** 0.165 → **0.994** (+0.829); recall 0.090 → **0.989**.
* **Deployed three-way policy:** auto-merge precision **99.5%**, auto-reject NPV **98.8%**, review rate 40.8% (generic titles and asymmetric specifications are deliberately sent to humans).
* **Hard-negative detection** for specification families *not* encoded as explicit features (orientation, drive, cooling, actuation, AC/DC, duty, mobility): **100.0%**.

### Cluster purity (entities known by construction)

| Clustering | Purity | Inverse purity | F-measure | Predicted clusters / true entities |
|---|---|---|---|---|
| exact_match_baseline | 1.0000 | 0.5448 | 0.7054 | 1939 / 1015 |
| minilm_threshold_0.90 | 0.9005 | 0.9025 | 0.9015 | 1011 / 1015 |
| hybrid_model | 0.9990 | 0.9946 | 0.9968 | 1024 / 1015 |

### Hard-negative detection by mutation family (test)

| Kind | Group | n | Detection rate |
|---|---|---|---|
| random | random | 182 | 1.000 |
| numeric | FACTOR | 62 | 1.000 |
| numeric | MM | 48 | 1.000 |
| numeric | KV | 24 | 1.000 |
| real_conflict | real_conflict | 22 | 1.000 |
| descriptor | actuation | 18 | 1.000 |
| descriptor | pipe_construction | 15 | 1.000 |
| numeric | INCH | 15 | 1.000 |
| descriptor | pump_type | 14 | 1.000 |
| descriptor | material_family | 13 | 1.000 |
| descriptor | drive | 13 | 1.000 |
| descriptor | current_type | 9 | 1.000 |
| standard | standard | 9 | 0.778 |
| descriptor | voltage_class | 8 | 1.000 |
| descriptor | valve_type | 6 | 1.000 |
| descriptor | orientation | 6 | 1.000 |
| numeric | KVA | 5 | 1.000 |
| descriptor | mobility | 5 | 1.000 |
| numeric | V | 4 | 1.000 |
| descriptor | cable_insulation | 3 | 1.000 |
| descriptor | end_connection | 3 | 1.000 |
| descriptor | copper_alloy | 2 | 1.000 |
| descriptor | conductor | 1 | 1.000 |
| descriptor | bearing_type | 1 | 1.000 |
| numeric | CORE | 1 | 1.000 |
| numeric | SQMM | 1 | 1.000 |
| numeric | SCH | 1 | 1.000 |
| numeric | PSI | 1 | 1.000 |
| numeric | MVA | 1 | 1.000 |

## 4. Review-Set Labels (AI-assisted, not independent human ground truth) — Real-Data Check

Scored on the **19 confirmed binary pairs** only (2 positive / 17 negative). With so few positives the
95 % Wilson intervals are wide; these results are evidence, not a precise estimate.

| Model | P | R | F1 | Acc | TP | FP | TN | FN | Acc 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2 | 0 | 17 | 0 | [0.83, 1.00] |
| random_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2 | 0 | 17 | 0 | [0.83, 1.00] |
| **hist_gradient_boosting** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2 | 0 | 17 | 0 | [0.83, 1.00] |
| exact_match_baseline | 0.0000 | 0.0000 | 0.0000 | 0.8947 | 0 | 0 | 17 | 2 | [0.69, 0.97] |
| tfidf_threshold_baseline | 0.2000 | 1.0000 | 0.3333 | 0.5789 | 2 | 8 | 9 | 0 | [0.36, 0.77] |
| minilm_threshold_baseline | 0.1250 | 1.0000 | 0.2222 | 0.2632 | 2 | 14 | 3 | 0 | [0.12, 0.49] |
| rule_engine_baseline | 1.0000 | 0.5000 | 0.6667 | 0.9474 | 1 | 0 | 17 | 1 | [0.75, 0.99] |

* **Deployed policy on review-set binary pairs:** 2 auto-merged (precision 100.0%), 17 auto-rejected (NPV 100.0%), 0 sent to review.
* **31 review-set pairs marked UNCERTAIN:** {'NEEDS_REVIEW': 28, 'NOT_EQUIVALENT': 2, 'EQUIVALENT': 1} — auto-merged: **1**.
* **Worksheet integrity note:** `PAIR_BOUNDARY_041` was restored to its genuine UNCERTAIN / MEDIUM annotation and test isolation verified.

## 4b. AI-Assisted Batch-2 Evaluation Tier (100 candidate pairs blind to model output)

> **Academic Integrity Statement:** Agreement with an AI annotator (Claude Opus, applying `docs/annotation_guidelines.md` blind to model output). This is **not independent human ground truth**; the annotator shares the model designer's rules, so routing agreement on UNCERTAIN pairs is expected by design.

* **Binary evaluation (n = 25 pairs: 8 positive / 17 negative):**

| Model | P | R | F1 | Acc | TP | FP | TN | FN | Acc 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.7273 | 1.0000 | 0.8421 | 0.8800 | 8 | 3 | 14 | 0 | [0.70, 0.96] |
| random_forest | 0.7273 | 1.0000 | 0.8421 | 0.8800 | 8 | 3 | 14 | 0 | [0.70, 0.96] |
| **hist_gradient_boosting** | 0.7273 | 1.0000 | 0.8421 | 0.8800 | 8 | 3 | 14 | 0 | [0.70, 0.96] |
| exact_match_baseline | 0.0000 | 0.0000 | 0.0000 | 0.6800 | 0 | 0 | 17 | 8 | [0.48, 0.83] |
| tfidf_threshold_baseline | 1.0000 | 0.8750 | 0.9333 | 0.9600 | 7 | 0 | 17 | 1 | [0.80, 0.99] |
| minilm_threshold_baseline | 0.3200 | 1.0000 | 0.4848 | 0.3200 | 8 | 17 | 0 | 0 | [0.17, 0.52] |
| rule_engine_baseline | 0.6000 | 0.3750 | 0.4615 | 0.7200 | 3 | 2 | 15 | 5 | [0.52, 0.86] |

* **Deployed policy on Batch-2 binary pairs:** 8 auto-merged (precision 100.0%), 13 auto-rejected (NPV 100.0%), 4 sent to review.
* **75 Batch-2 pairs marked UNCERTAIN:** {'NEEDS_REVIEW': 50, 'NOT_EQUIVALENT': 24, 'EQUIVALENT': 1} — auto-merged: **1**.

## 5. All 22,464 real candidate pairs (unlabelled, descriptive only)

* Decisions: {'NOT_EQUIVALENT': 16911, 'NEEDS_REVIEW': 4640, 'EQUIVALENT': 913}
* Cross-enterprise decisions: {'NOT_EQUIVALENT': 696, 'NEEDS_REVIEW': 222, 'EQUIVALENT': 12}
* Conflict-aware clustering: **374 proposed harmonised groups** covering 913 records; 50 merges blocked, 50 transitive conflicts caught.
* Outputs: `results/matching/model_scored_candidate_pairs.csv`, `results/matching/harmonised_master_model.csv`.
* Active learning: `results/metrics/batch2_review_priority.csv` ranks the 100 pending pairs by model uncertainty for the next review session.

## 6. Most important features (permutation importance on test, PR-AUC drop)

| Feature | Importance |
|---|---|
| `alpha_unmatched_fuzzy` | 0.0327 ± 0.0022 |
| `num_bare_jaccard` | 0.0024 ± 0.0003 |
| `fuzz_token_sort` | 0.0009 ± 0.0001 |
| `token_jaccard` | 0.0007 ± 0.0004 |
| `len_ratio` | 0.0006 ± 0.0002 |
| `unit_spec_conflicts` | 0.0003 ± 0.0001 |
| `fuzz_partial` | 0.0003 ± 0.0001 |
| `token_containment` | 0.0003 ± 0.0001 |
| `alpha_symdiff` | 0.0003 ± 0.0001 |
| `unit_spec_partial` | 0.0002 ± 0.0000 |

## 7. Limitations

1. Test pairs are constructed from real records; genuine cross-enterprise variation is messier, so held-out scores are an upper bound on real-world accuracy.
2. Review-set binary labels are few (19 binary, 2 positive). Label the pending pairs (start with `batch2_review_priority.csv`) and re-run `python train_er_model.py` — review metrics update automatically.
3. Specification parsing is regex-based; codes outside the parser are only caught through lexical evidence.
4. The model never sees enterprise identity; domain differences (API vs BIS service) that are not written in the text cannot be detected.

_Runtime of the last training run: 136.1 s._
