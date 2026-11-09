# Batch 2 Data Validation Report

**Course:** CSE 276 – CA-1 Academic Project  
**Project:** Material Code Harmonisation Across Public Enterprises (IOCL, OIL, NTPC)  
**Target:** Batch 2 Expansion Dataset (`data/processed/candidate_pairs_batch2_expansion_100.csv`)  
**Validation Date:** 6 November 2026  
**Auditor:** Antigravity Autonomous Pair Programming Agent  

---

## 1. Executive Summary

A comprehensive data validation audit was executed on the 100 new candidate pairs in Batch 2 (`data/processed/candidate_pairs_batch2_expansion_100.csv`). The objective is to verify dataset integrity, provenance, schema conformance, lack of duplicate pairs, and strict separation between automated AI suggestions and human ground-truth labels.

### High-Level Verdict: **PASSED WITH NOTED ARCHITECTURAL OBSERVATION**
- **Row count:** 100 candidate pairs (Exact match).
- **Identifier format:** 100 unique pair identifiers (`PAIR_EXP_001` through `PAIR_EXP_100`).
- **Referential integrity:** All 126 unique material IDs exist in `data/processed/preprocessed_materials.csv` (0 missing).
- **Pair uniqueness:** 0 duplicate unordered pairs in Batch 2.
- **Pair overlap with Batch 1:** 0 duplicate pairs between Batch 1 and Batch 2 (100% disjoint pairs).
- **Endpoint (node-level) material overlap:** 11 material IDs from Batch 1 re-occur across 38 pairs in Batch 2 (documented below as graph hub entities).
- **Enterprise distribution:** Matches report specifications exactly (73 IOCL–OIL, 15 NTPC–OIL, 12 IOCL–NTPC).
- **Source provenance:** 100% of candidate records retain source references (tender, document, section).
- **Human label isolation:** 100% of human label fields (`human_label`, `confidence_rating`) are `PENDING`, and all reviewer notes are blank.
- **AI suggestions:** Segregated into dedicated columns (`ai_suggested_hypothesis`, `ai_suggested_confidence`, `ai_engineering_rationale`).

---

## 2. Detailed Verification Checklist

| Dimension | Requirement | Actual Result | Status |
| :--- | :--- | :--- | :--- |
| **Row Count** | Exactly 100 candidate pairs | 100 rows | **PASS** |
| **Pair Identifier Uniqueness** | Exactly 100 unique pair IDs (`PAIR_EXP_001` - `PAIR_EXP_100`) | 100 unique IDs, 0 duplicates | **PASS** |
| **Material ID Validity** | No missing or null material IDs | 0 nulls across `corpus_id_1` and `corpus_id_2` | **PASS** |
| **Corpus Alignment** | All material IDs must exist in preprocessed corpus | 126/126 unique IDs verified in `preprocessed_materials.csv` | **PASS** |
| **Internal Pair Deduplication** | Zero duplicate unordered pairs $\{M_A, M_B\}$ within Batch 2 | 100 unique unordered pairs, 0 duplicates | **PASS** |
| **Cross-Batch Pair Deduplication** | Zero unordered pair overlap with Batch 1 (50 pairs) | 0 overlapping pairs between Batch 1 and Batch 2 | **PASS** |
| **Enterprise Pair Distribution** | Matches target: IOCL–OIL (73), NTPC–OIL (15), IOCL–NTPC (12) | IOCL–OIL: 73 (73%), NTPC–OIL: 15 (15%), IOCL–NTPC: 12 (12%) | **PASS** |
| **Source Reference Retention** | Source URLs / Tender references retained | 100/100 for `source_reference_1`, 100/100 for `source_reference_2` | **PASS** |
| **Human Label Schema** | `human_label` strictly `PENDING` | 100/100 `PENDING` | **PASS** |
| **Confidence Schema** | `confidence_rating` strictly `PENDING` | 100/100 `PENDING` | **PASS** |
| **Reviewer Notes** | Reviewer notes unpopulated | 100/100 empty string (`""`) | **PASS** |
| **AI Isolation** | AI recommendations strictly in separate fields | Present in `ai_suggested_*` columns only | **PASS** |

---

## 3. Critical Architectural Finding: Node-Level vs Pair-Level Overlap

The validation audit distinguished between two forms of overlap:

1. **Pair-Level Overlap (Zero Duplication):**
   - None of the 50 candidate pairs in Batch 1 appear in Batch 2.
   - For every $\{u, v\} \in \text{Batch 2}$, $\{u, v\} \notin \text{Batch 1}$.
   - **Result: 0 overlapping pairs.**

2. **Endpoint (Material ID) Overlap (Graph Hubs):**
   - Batch 1 contains 58 unique material IDs across 50 pairs.
   - Batch 2 contains 126 unique material IDs across 100 pairs.
   - Exactly **11 material IDs** from Batch 1 reappear in Batch 2 across 38 pairs.
   - Overlapping material entities include:
     - `M000042` (IOCL: API 6D BALL VALVES) — involved in 10 Batch 2 pairs
     - `M000165` (IOCL: ELECTRICAL CABLES) — involved in 9 Batch 2 pairs
     - `M000085` (IOCL: BEARINGS) — involved in 5 Batch 2 pairs
     - `M000114` (IOCL: CENTRIFUGAL PUMP) — involved in 5 Batch 2 pairs
     - `M017429` (OIL: XLPE CABLE 1.1 KV) — involved in 3 Batch 2 pairs
     - `M000285` (IOCL: MAGNETIC STIRRER) — involved in 2 Batch 2 pairs
     - `M000082` (IOCL: BALL VALVE) — involved in 1 Batch 2 pair
     - `M004585`, `M006811`, `M009963`, `M020224` — 1 pair each

### Impact on Machine Learning & Connected-Component Splitting
In entity resolution, high-volume catalogue commodities (e.g., standard bearings, generic cables, API valves) naturally form 1-to-many comparison stars. When Batch 1 and Batch 2 are combined, these 11 shared material nodes bridge candidate pairs from both batches into unified graph components. 

> [!IMPORTANT]
> **Data Leakage Safeguard:** If supervised learning were to be performed across combined batches, a standard random train/test split would cause material ID leakage. The project's `create_connected_component_split()` in `src/evaluation.py` explicitly tracks component connectivity to prevent material IDs from crossing partitions.

---

## 4. Enterprise-Pair Representation Breakdown

The cross-enterprise distribution of Batch 2 fulfills the project's requirement to connect public sector enterprises:

| Enterprise Pair | Count | Percentage | Primary Equipment Categories |
| :--- | :--- | :--- | :--- |
| **IOCL ↔ OIL** | 73 | 73.0% | API 6D Ball Valves, Centrifugal Pumps, Bearings, Cables, Turbines |
| **NTPC ↔ OIL** | 15 | 15.0% | XLPE Power Cables (1.1 kV), Submersible Pumps, Magnetic Stirrers |
| **IOCL ↔ NTPC** | 12 | 12.0% | Ball Valves, Steam Turbines, Conveyor Belting, Transformers |
| **Total** | **100** | **100.0%** | **9 Industrial Procurement Domains** |

---

## 5. Domain Category Distribution

Batch 2 candidates span 9 essential public enterprise procurement domains:

1. **Valves & Flow Control:** 18 pairs
2. **Electrical & Power Systems:** 15 pairs
3. **Pumps & Hydraulic Machinery:** 15 pairs
4. **Compressors, Turbines & Rotating Equipment:** 12 pairs
5. **Instrumentation & Analytical Equipment:** 12 pairs
6. **Pipes, Flanges & Fittings:** 12 pairs
7. **Bearings, Seals & Mechanical Spares:** 8 pairs
8. **Structural Steel & Heavy Equipment:** 5 pairs
9. **Chemicals & Consumables:** 3 pairs

---

## 6. AI Hypothesis Distribution vs Ground Truth

The AI suggestions embedded in Batch 2 are strictly advisory and provide baseline stratification for human reviewers:

- **AI Hypothesis `1` (Potential Positive Match):** 16 pairs (16.0%)
- **AI Hypothesis `0` (Probable Incompatible Match):** 5 pairs (5.0%)
- **AI Hypothesis `UNCERTAIN` (Requires SME Inspection):** 79 pairs (79.0%)

All 100 pairs maintain `human_label = PENDING`, preserving academic integrity and preventing premature training on unverified hypotheses.

