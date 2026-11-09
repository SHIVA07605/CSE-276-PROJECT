# Model Evaluation & Academic Integrity Report

**Course:** CSE 276 – CA-1 Academic Project  
**Project:** Material Code Harmonisation Across Public Enterprises (IOCL, OIL, NTPC)  
**Evaluation Target:** Baseline Exact Match, Candidate Generation, and Supervised Model Evaluation Protocol  
**Date:** 6 November 2026  
**Status:** ✅ Hybrid self-supervised ER model TRAINED and EVALUATED (see update below). Supervised training on *human* labels remains locked until ≥ 150 labels exist.

> **Update (6 Nov 2026) — trained model delivered.** A hybrid entity-resolution model
> (`models/hybrid_er_model.joblib`) is now trained on **self-supervised pairs whose labels are known by
> construction** (format-only rewrites of real records = match; single-specification changes = non-match).
> No human or AI-suggested label is used for fitting, so the integrity policy below is fully respected.
> It is evaluated (1) on a group-disjoint held-out split against the exact-match, TF-IDF, MiniLM and
> rule-engine baselines, (2) on review-set labels (AI-assisted, not independent human ground truth) with 95 % Wilson intervals, and (3) by cluster purity.
> **All results:** [`docs/ml_model_report.md`](ml_model_report.md) (auto-generated from
> `results/metrics/er_model_metrics.json`) · Notebooks: `05_pairwise_classification.ipynb`, `07_model_evaluation.ipynb`
> · Reproduce: `python train_er_model.py`.

---

## 1. Academic Integrity & Evaluation Policy

> [!IMPORTANT]
> **Strict Governance Mandate:**  
> Machine learning models must NEVER be fitted or evaluated on AI-generated hypotheses treated as independent ground truth. Validated supervised performance metrics (Precision, Recall, F1, Cluster Purity) can only be computed once sufficient genuine human-reviewed labels exist. Fabricating metrics, precision scores, or synthetic test splits is strictly prohibited by course guidelines.

### Status of Supervised Classifier Training
- **Total Candidate Pairs across Batches 1 & 2:** 150 pairs.
- **Reviewed Annotations (Batch 1):** 50 pairs.
  - Positive matches (`1`): 2 pairs (API 6D Ball Valves under Rule 6).
  - Negative matches (`0`): 17 pairs (10 conveyor belt width conflicts, 6 AMC vs physical instrument, 1 bearing standard).
  - Uncertain (`UNCERTAIN`): 31 pairs (generic category descriptors without technical specifications).
  - Confirmed binary training pairs: **19 pairs** (2 positive, 17 negative).
- **Pending Review (Batch 2):** 100 pairs (all `PENDING`).
- **Readiness Determination:** **`NOT_READY_FOR_TRAINING`**.
  - Minimum pilot requirement: 30 confirmed binary pairs with balanced classes ($\ge 5$ pos, $\ge 5$ neg).
  - Minimum production benchmark: 150 confirmed binary pairs ($\ge 25$ pos, $\ge 50$ neg).
  - **Action Taken:** Supervised model fitting (`src/model.py` and Notebook 05) is **strictly locked and blocked**. Unfitted models are prevented from serializing to disk.

---

## 2. Descriptive Evaluation of the Exact Match Baseline

The exact string matching baseline ($f(r_i, r_j) = \mathbb{I}(\text{norm}(D_i) = \text{norm}(D_j))$) represents the standard enterprise reconciliation practice currently employed by CPSE ERP systems.

### Baseline Results across 6,634 Preprocessed Records
- **Total Exact Match Pairs:** 1,661 pairs.
- **Intra-Enterprise Matches:** 1,646 pairs (99.1% — exact line-item duplicate procurements within OIL or IOCL).
- **Cross-Enterprise Matches:** 15 pairs (0.9% — identical normalized descriptions bridging different CPSEs).

### Qualitative Breakdown of Cross-Enterprise Exact Matches
1. **Generic Equipment Titles (Zero Verifiable Interchangeability):**
   - `AIR COMPRESSOR` (IOCL) $\leftrightarrow$ `AIR COMPRESSOR` (OIL) — 2 pairs.
   - `BEARINGS` (IOCL) $\leftrightarrow$ `BEARINGS` (OIL) — 2 pairs.
   - `BALL VALVE` (IOCL) $\leftrightarrow$ `BALL VALVE` (OIL) — 2 pairs.
   - *Failure Mode:* Identical strings mask completely different flow capacities, discharge pressures, bore sizes, and voltage ratings. Blindly harmonising these records introduces severe procurement risks.
2. **True Standard-Governed Commodity Equivalents:**
   - Standard laboratory equipment: `HOT PLATE MAGNETIC STIRRER` (IOCL) $\leftrightarrow$ `HOT PLATE MAGNETIC STIRRER` (OIL).
   - Standard low-voltage cable titles: `XLPE CABLE 1.1 KV` (NTPC) $\leftrightarrow$ `XLPE CABLE 1.1 KV` (OIL).
3. **Exact Baseline Limitations:**
   - **Severe Recall Deficit:** Fails to match inverted phrasing (e.g. `CENTRIFUGAL PUMP` vs `PUMP-CENTRIFUGAL`), abbreviation variants (`STAINLESS STEEL` vs `SS304`), or tender packaging variations (`API 6D BALL VALVES` vs `API 6D BALL VALVES QUANTITY 76 NUMBERS`).
   - **Zero Semantic Generalization:** Out of 22,464 candidate pairs identified by TF-IDF and dense embeddings, the exact baseline recovers less than 7.4%.

---

## 3. Candidate Generation & Blocking Statistics

Candidate generation prunes the quadratic space of all possible pairs down to high-recall candidate pairs. These retrieval metrics are reported separately from classification performance:

| Retrieval Stage | Search Space / Method | Total Pairs Retrieved | Cross-Enterprise Pairs | Search Space Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **Quadratic Full Comparison** | $N(N-1)/2$ for $N=6,634$ | 21,995,361 | $\approx 2,400,000$ | 0.0% (Intractable) |
| **Sparse TF-IDF Retrieval** | Character & word n-grams (1, 2), $\tau = 0.35$ | 22,464 | 930 | **>99.9% reduction** |
| **Dense Semantic Vector Ret.** | all-MiniLM-L6-v2, $\tau = 0.60$ | 22,464 | 930 | Precomputed in 0.079s |
| **Batch 1 Audit Sample** | 4-strata sampling (Exact, High TF-IDF, Conflict, Boundary) | 50 | 40 | Targeted SME verification |
| **Batch 2 Expansion Sample** | 9-domain stratified positive-weighted retrieval | 100 | 100 | Cross-enterprise expansion |
| **Total Review Candidates** | Integrated Review Dataset | **150** | **140 (93.3%)** | Active Review Pool |

---

## 4. Evaluation Protocol for Future Supervised Models

When sufficient human annotations are logged via `review_app.py`, the evaluation pipeline will execute the following rigorous protocol already implemented in `src/evaluation.py`:

### 1. Leakage-Free Connected-Component Cross-Validation
Standard k-fold cross-validation or random train/test splits suffer from severe data leakage in entity resolution: if material ID $M_{101}$ appears in the training set paired with $M_{102}$, and also in the test set paired with $M_{103}$, the model memorizes record-specific attributes rather than learning generalizable compatibility functions.

The project implements `create_connected_component_split()`:
- Forms an undirected graph $G = (V, E)$ where nodes $V$ are material IDs and edges $E$ are candidate pairs.
- Partitions pairs into disjoint connected components.
- Allocates entire components to Train (75%) and Test (25%).
- **Guarantees zero material ID overlap:** $V_{\text{train}} \cap V_{\text{test}} = \emptyset$.

### 2. Pairwise Classification Metrics
Computed via `compute_pairwise_metrics()` on the unseen test partition:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 3. Cluster Purity Metric
Evaluates global harmonisation quality by comparing predicted graph clusters $\Omega = \{\omega_1, \dots, \omega_K\}$ against ground-truth material equivalence groups $\mathbb{C} = \{c_1, \dots, c_J\}$:
$$\text{Purity}(\Omega, \mathbb{C}) = \frac{1}{N} \sum_{k=1}^K \max_{j} |\omega_k \cap c_j|$$
Computed via `compute_cluster_purity()` with contingency matrix aggregation.

### 4. Comparative Evaluation Table (Target Benchmark)
Once human annotations reach $\ge 150$ confirmed labels, `generate_comparative_summary()` produces the official CA-1 evaluation report:

| Metric | Baseline (Exact Match) | Precision-First Matching Engine | Random Forest (Supervised) | Absolute Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Pairwise Precision** | High (on specific specs), Poor (on generic titles) | High ($\ge 90\%$ targeted) | Standing By (Locked) | TBD upon annotation |
| **Pairwise Recall** | Extremely Low ($< 10\%$) | High ($\ge 75\%$ targeted) | Standing By (Locked) | TBD upon annotation |
| **Pairwise F1-Score** | Low | Balanced ($\ge 80\%$ targeted) | Standing By (Locked) | TBD upon annotation |
| **Cluster Purity** | Low (polluted by generic singletons) | High (safeguarded by conflict blocks) | Standing By (Locked) | TBD upon annotation |

