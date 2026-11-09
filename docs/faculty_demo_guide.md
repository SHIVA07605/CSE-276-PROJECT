# Faculty Live Demonstration Guide (5–8 Minute Sequence)

**Course:** CSE 276 – CA-1 Academic Evaluation  
**Project Title:** Material Code Harmonisation Across Public Enterprises (IOCL, OIL, NTPC)  
**Problem Statement ID:** SIH26099 (Smart India Hackathon)  
**Target Audience:** LPU Faculty Evaluators / External Examination Panel  
**Demonstration Time:** 5 to 8 Minutes  
**Repository:** `https://github.com/SHIVA07605/CSE-276-PROJECT.git`  

---

## Preparation (Before Demonstration)

Ensure your command line is open at `c:\project ai` and your virtual environment is active:
```powershell
cd "c:\project ai"
```

Verify that the test suite passes:
```powershell
python -m pytest
```
*Expected Output:* `168 passed, 1 skipped in ~17-21s`.

---

## 5–8 Minute Demonstration Script

### Step 1: Dataset & Preprocessing (1.0 Minute)
**Objective:** Show the faculty the real-world scale of the multi-CPSE corpus and text cleaning.

**Command / Notebook Action:**
Open and run cells in `notebooks/02_data_preprocessing.ipynb` or run:
```powershell
python -c "
import pandas as pd
raw = pd.read_csv('data/raw/material_description_corpus.csv', low_memory=False)
proc = pd.read_csv('data/processed/preprocessed_materials.csv', low_memory=False)
print('Total Raw Records:', len(raw))
print('Candidate Pool (Tier 1 + 2 Tagged):', len(proc))
print('Sample Raw -> Cleaned Text:')
for i in [1, 10, 42]:
    print(f'  RAW:   {proc.loc[i, \"raw_description\"]}')
    print(f'  CLEAN: {proc.loc[i, \"cleaned_description\"]}')
"
```

**Talking Point for Evaluator:**
> "Our raw dataset contains 21,513 records across IOCL, OIL, and NTPC. We partition this into an operational candidate pool of 6,634 items. Our text cleaner standardizes metallurgical abbreviations like SS to STAINLESS STEEL, standardizes units, and eliminates procurement boilerplate like 'SUPPLY OF' and 'PROCUREMENT OF'."

---

### Step 2: Sublinear Candidate Generation & Blocking (1.0 Minute)
**Objective:** Explain how quadratic complexity is reduced from 22 million pairs to 22,464 pairs.

**Command / Notebook Action:**
Demonstrate `notebooks/03_baseline_matching.ipynb` or execute:
```powershell
python -c "
import pandas as pd
exact = pd.read_csv('results/matching/exact_matches.csv')
print('Quadratic Space: 21,995,361 pairs')
print('Exact Matches Found:', len(exact))
print('  - Cross-Enterprise Exact Matches:', (exact['is_cross_enterprise'] == True).sum())
print('Sparse TF-IDF Candidates Retained: 22,464 pairs (>99.89% search space reduction)')
"
```

**Talking Point for Evaluator:**
> "Comparing all 6,634 items pairwise requires 22 million comparisons, which is computationally intractable for deep transformer models. The exact baseline finds only 15 cross-enterprise matches, missing over 98% of true equivalents. By using sparse TF-IDF n-gram blocking, we prune 99.89% of the search space down to 22,464 high-recall candidate pairs."

---

### Step 3: Dense Semantic Matching & Synonym Discovery (1.0 Minute)
**Objective:** Show dense vector representations and phrasing inversion recovery.

**Command / Notebook Action:**
Open `notebooks/04_semantic_matching.ipynb` or run:
```powershell
python -c "
from src.feature_extraction import compute_lexical_features, get_embedding_model
import numpy as np

m1 = 'CENTRIFUGAL PUMP'
m2 = 'PUMP-CENTRIFUGAL'
lex = compute_lexical_features(m1, m2)['token_sort_ratio']
model = get_embedding_model()
emb = model.encode([m1, m2], normalize_embeddings=True)
sem = float(np.dot(emb[0], emb[1]))
print(f'Comparing: \"{m1}\" <-> \"{m2}\"')
print(f'  Lexical Sim (RapidFuzz): {lex:.4f}')
print(f'  Semantic Sim (MiniLM):   {sem:.4f}')
print('  Discrepancy (Semantic - Lexical):', f'+{sem - lex:.4f}')
"
```

**Talking Point for Evaluator:**
> "Here is a concrete example of phrasing inversion across CPSEs: IOCL records 'CENTRIFUGAL PUMP' while OIL records 'PUMP-CENTRIFUGAL'. Lexical token sort scores only 0.43, but our 384-dimensional MiniLM embeddings score 0.9616. The high positive discrepancy (+0.53) allows us to discover cross-enterprise synonyms."

---

### Step 4: Attribute Conflict Detection (1.5 Minutes)
**Objective:** Prove that high semantic similarity alone is dangerous without physical attribute validation.

**Command / Notebook Action:**
Run the `PrecisionMatchingEngine` edge cases demonstration:
```powershell
python -c "
from src.matching_engine import PrecisionMatchingEngine
engine = PrecisionMatchingEngine()

# Incompatible conveyor belt widths
p1 = engine.evaluate_pair('M1', 'STEEL CORD CONVEYOR BELT 1200MM', 'M2', 'STEEL CORD CONVEYOR BELT 1600MM', lexical_sim=0.92, semantic_sim=0.98)
print('Example 1 (Conveyor Belt):')
print(f'  Decision: {p1.decision} (Confidence: {p1.confidence})')
print(f'  Detected Conflict: {p1.conflicting_attributes}')

# Generic Air Compressor
p2 = engine.evaluate_pair('M1', 'AIR COMPRESSOR', 'M2', 'AIR COMPRESSOR', lexical_sim=1.0, semantic_sim=1.0)
print('\nExample 2 (Generic Air Compressor):')
print(f'  Decision: {p2.decision} (Confidence: {p2.confidence})')
print(f'  Explanation: {p2.explanation[:85]}...')
"
```

**Talking Point for Evaluator:**
> "Notice this critical engineering distinction: two conveyor belts with 1200MM vs 1600MM width achieve 0.98 semantic similarity. Naive models would classify them as matches. Our matching engine detects the 400MM dimensional conflict and declares `NOT_EQUIVALENT` with HIGH confidence. Furthermore, for two identical descriptions like 'AIR COMPRESSOR', our engine refuses to declare equivalence because critical CFM and pressure ratings are unstated, correctly classifying it as `NEEDS_REVIEW`."

---

### Step 5: Conflict-Aware Clustering & Transitive Conflict Prevention (1.5 Minutes)
**Objective:** Demonstrate how graph clustering prevents transitive corruption.

**Command / Notebook Action:**
Run the transitive conflict demonstration from `notebooks/06_clustering.ipynb`:
```powershell
python -c "
import pandas as pd
from src.clustering import conflict_aware_clustering, cluster_connected_components

demo = pd.DataFrame([
    {'record_id_1': 'IOCL_PIPE_250', 'record_id_2': 'OIL_PIPE_GENERIC', 'prediction': '1'},
    {'record_id_1': 'OIL_PIPE_GENERIC', 'record_id_2': 'NTPC_PIPE_300', 'prediction': '1'},
    {'record_id_1': 'IOCL_PIPE_250', 'record_id_2': 'NTPC_PIPE_300', 'prediction': '0', 'has_attribute_conflict': True},
])

print('Naive Connected Components:')
print(cluster_connected_components(demo, prediction_col='prediction'))

print('\nConflict-Aware Clustering:')
clusters, audit = conflict_aware_clustering(demo, prediction_col='prediction')
print(clusters)
print('Transitive Conflicts Caught:', audit['transitive_conflicts_count'])
"
```

**Talking Point for Evaluator:**
> "In graph entity resolution, if Item A matches generic Item B, and Item B matches Item C, naive connected components merges all three into one cluster. But if Item A is a 250MM pipe and Item C is a 300MM pipe, the catalogue is corrupted. Our `conflict_aware_clustering` checks compatibility across all members, catches the transitive conflict, and partitions them into two separate safe clusters."

---

### Step 6: Reviewer Web Application Live Demo (1.5 Minutes)
**Objective:** Show the zero-dependency web interface and human-in-the-loop workflow.

**Command:**
```powershell
python review_app.py --port 8050
```
Open browser to `http://localhost:8050`.

**Demonstration Steps on Screen:**
1. **Top Bar & Stats:** Point out the dataset selector (`Combined Dataset 150 Pairs`), showing 150 candidate pairs loaded.
2. **Side-by-Side Cards:** Show Material A (IOCL) and Material B (OIL/NTPC) with highlighted preprocessed text and extracted physical attributes.
3. **Source Provenance:** Show the Tender reference, Catalogue section, and Document URL.
4. **AI Advisory Panel:** Highlight the purple card: explain that AI hypotheses are clearly segregated as advisory suggestions.
5. **Non-Auto-Advancing Labeling:** 
   - Click `[1] Functionally Equivalent` (or press hotkey `1`). Point out that the screen **does NOT advance or auto-save**.
   - Type reviewer engineering notes (e.g. `Verified API 6D standard alignment`).
   - Select confidence: `HIGH`.
   - Click `Save & Next ▶`. Show the success confirmation and advance to the next pending pair.

---

### Step 7: The Trained ML Model — Self-Supervised Entity Resolution (1.5 Minutes)
**Objective:** Show the trained model, how it was trained without fabricated labels, and its results.

**Live actions:**
1. Open `notebooks/05_pairwise_classification.ipynb` → Section B (training-pair examples) and Section E (precision–recall curve vs baselines).
2. Open `notebooks/07_model_evaluation.ipynb` → Section 2 (review-set labels) and Section 3 (cluster purity).
3. Score any two descriptions live:
   ```bash
   python -c "from src.er_pipeline import match_descriptions as m; print(m('CENTRIFUGAL PUMP API 610 60 M3/HR', 'PUMP-CENTRIFUGAL, API 610, 60 M3/HR'))"
   python -c "from src.er_pipeline import match_descriptions as m; print(m('GATE VALVE SS304 2INCH CLASS 150', 'GATE VALVE SS316 2INCH CLASS 150'))"
   ```

**Talking Point for Evaluator:**
> "We only have 19 confirmed binary review-set labels (AI-assisted, not independent human ground truth), so instead of training on AI guesses we generate training pairs whose labels are known *by construction* from the real catalogue: a record versus a format-only rewrite of itself is a match, and a record versus a copy with exactly one specification changed — 250 mm to 300 mm, SS304 to SS316, ball valve to gate valve — is a non-match. The 173 records in the review set are excluded from training, and splits are group-disjoint, so there is zero leakage. The selected Gradient-Boosting model reaches F1 ≈ 0.99 on held-out pairs versus ≈ 0.16 for the exact-match baseline, classifies the binary review-set pairs correctly, and its deployed policy never auto-merges a pair with an explicit specification conflict — generic titles like 'AIR COMPRESSOR' are sent to human review. All numbers are in `docs/ml_model_report.md`."

---

## Quick Reference Table for Evaluator Questions

| Evaluator Question | Verified Answer |
| :--- | :--- |
| **How many total records?** | 21,513 raw records; 6,634 candidate pool; 14,879 held-back records. |
| **How do you prevent data leakage?** | We use `create_connected_component_split()`, which allocates entire connected graph components to train or test, ensuring $V_{\text{train}} \cap V_{\text{test}} = \emptyset$ with 0 material ID overlap. |
| **What is the baseline?** | Exact normalized string matching ($f(r_i, r_j) = \mathbb{I}(\text{norm}(D_i) = \text{norm}(D_j))$), yielding 1,661 pairs (15 cross-enterprise). |
| **What is the test pass rate?** | 197 passed, 0 failed across 11 test modules (`python -m pytest`), including 28 tests for the ML model (`tests/test_er_model.py`). |
| **Is there a trained model?** | Yes — `models/hybrid_er_model.joblib`: Histogram Gradient Boosting over 34 hybrid features (MiniLM, TF-IDF, fuzzy, canonical specification conflicts, rule-engine signals) with a precision-first three-way decision policy. |
| **How was it trained with only 19 binary review labels?** | Self-supervised: labels are known by construction (format-only rewrite = match; one specification changed = non-match), built from the real candidate pool. No human or AI-suggested label is used for fitting. |
| **What are its results?** | See `docs/ml_model_report.md` (auto-generated): held-out F1 ≈ 0.99 vs ≈ 0.16 for exact match; binary review-set pairs correct; cluster purity / inverse purity ≈ 0.99. |
| **Why trust results on constructed pairs?** | Splits are group-disjoint (zero leakage), review-set records are excluded, specification families unseen by explicit features are reported separately, and the model is checked on real review-set labels; held-out scores are stated as an upper bound. |

