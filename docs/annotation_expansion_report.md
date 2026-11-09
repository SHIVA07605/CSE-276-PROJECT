# Annotation Quality Audit & Positive Candidate Expansion Report (Batch 2)

**Course:** CSE 276 – CA-1 Academic Project  
**Project:** Material Code Harmonisation Across Public Enterprises  
**Stage:** Post-Audit Candidate Pool Expansion  
**Date:** 4 November 2026  
**Status:** BATCH 2 CANDIDATE SET (100 PAIRS) GENERATED & VERIFIED — ZERO MODEL TRAINING  

---

## 1. Executive Summary & Audit of Priority Batch 1 Annotations

In the initial 50-pair candidate sample, AI-assisted engineering evaluation established:
* **Positive Matches (`1`):** 2 pairs (`PAIR_TFIDF_HIGH_020`, `021` — API 6D Ball Valves under Rule 6)
* **Negative Non-Matches (`0`):** 17 pairs (10 attribute conflict conveyor belt pairs under Rule 1, 6 service AMC vs physical instrument pairs under Rule 7, 1 bearing standards mismatch)
* **Uncertain (`UNCERTAIN`):** 31 pairs (generic category descriptors without technical specifications under Rule 5)

### Deep Engineering Audit of 5 Priority Pairs

Per user directive, the five most critical boundary and high-leverage pairs from Batch 1 were inspected in depth:

```
+----------------------------------------------------------------------------------------------------+
| PAIR_EXACT_011 & PAIR_EXACT_012: Hot plate magnetic stirrer                                       |
| M1: [M000229] (IOCL) "Hot plate magnetic stirrer"                                                  |
| M2: [M004585 / M019749] (OIL) "Hot Plate Magnetic Stirrer"                                        |
| Similarities: Lexical = 1.0000 | Semantic = 1.0000 | Exact Normalized = True                        |
| Current Status: UNCERTAIN (MEDIUM Confidence)                                                      |
+----------------------------------------------------------------------------------------------------+
* Technical Assessment: Both records describe standard benchtop laboratory equipment. In oil refinery 
  and field exploration quality control laboratories, standard digital hot plate magnetic stirrers 
  (typically ceramic top plate, stirring up to 20L, heating to 350°C, 1500 rpm) serve interchangeable 
  functional roles.
* Why Human Verification is High-Priority: If a procurement engineer inspects the equipment schedule 
  and confirms standard laboratory bench specifications, these two pairs can be promoted to Positive (1), 
  providing 2 high-integrity cross-enterprise positive anchors.
```

```
+----------------------------------------------------------------------------------------------------+
| PAIR_EXACT_015: Low-Voltage XLPE Cable                                                             |
| M1: [M001023] (NTPC) "XLPE Cable for Working Voltages up to and Including 1.1 KV"                  |
| M2: [M017429] (OIL) "XLPE Cable for Working Voltages up to and Including 1.1 KV"                   |
| Similarities: Lexical = 1.0000 | Semantic = 1.0000 | Exact Normalized = True                        |
| Current Status: UNCERTAIN (MEDIUM Confidence)                                                      |
+----------------------------------------------------------------------------------------------------+
* Technical Assessment: Both records verbatim describe Indian Standard IS:7098 (Part 1) cross-linked 
  polyethylene insulated low voltage electric cables. 
* Why Human Verification is High-Priority: This represents a cross-sector match between a thermal power 
  generation enterprise (NTPC) and an upstream oil producer (OIL). If tender schedules confirm matching 
  core configurations (e.g. standard 3.5C x 185 sq mm or 4C x 16 sq mm), this confirms genuine 
  cross-enterprise functional equivalence across distinct public sectors.
```

```
+----------------------------------------------------------------------------------------------------+
| PAIR_BOUNDARY_049: API 610 Centrifugal Pump vs Inverted Catalog Header                             |
| M1: [M000041] (IOCL) "API 610 Centrifugal Pump"                                                    |
| M2: [M020224] (OIL) "PUMP-CENTRIFUGAL"                                                             |
| Similarities: Lexical = 0.3924 | Semantic = 0.7874 | Exact Normalized = False                       |
| Current Status: UNCERTAIN (MEDIUM Confidence)                                                      |
+----------------------------------------------------------------------------------------------------+
* Technical Assessment: M2 uses inverted noun-adjective catalog syntax ("PUMP-CENTRIFUGAL"), while M1 
  specifies heavy-duty petroleum refinery compliance (API 610).
* Why Human Verification is High-Priority: Demonstrates the challenge of catalog formatting differences. 
  If OIL M020224's underlying SAP material group is designated for hydrocarbon service, it is a positive 
  match under Rule 6; if it denotes utility water service, it is an incompatible negative (0).
```

```
+----------------------------------------------------------------------------------------------------+
| PAIR_BOUNDARY_050: API 610 Centrifugal Pump vs Specific Flow Rate                                  |
| M1: [M000041] (IOCL) "API 610 Centrifugal Pump"                                                    |
| M2: [M017442] (OIL) "centrifugal pump of discharge flow 60 KL/Hr"                                  |
| Similarities: Lexical = 0.4148 | Semantic = 0.6535 | Exact Normalized = False                       |
| Current Status: UNCERTAIN (MEDIUM Confidence)                                                      |
+----------------------------------------------------------------------------------------------------+
* Technical Assessment: M1 provides the engineering standard (API 610) without capacity, while M2 
  provides the operational capacity (60,000 L/hr = 60 m3/hr) without stating the design standard.
* Why Human Verification is High-Priority: High-value example of asymmetric specification complementarity. 
  Human inspection of the pump datasheets will determine whether M1's rated duty matches 60 KL/Hr.
```

---

## 2. Complete Candidate Pool Exploration (6,634 Records)

The full candidate pool comprises **6,634 preprocessed material records** across three major Indian Central Public Sector Enterprises (CPSEs):
* **Oil India Limited (OIL):** 5,394 records (81.3%)
* **NTPC Limited (NTPC):** 732 records (11.0%)
* **Indian Oil Corporation Limited (IOCL):** 508 records (7.7%)

### Candidate Pair Retrieval Pool
The TF-IDF blocking and dense semantic vector retrieval (`all-MiniLM-L6-v2`) generated **22,464 candidate pairs** in Notebooks 03 and 04:
* **Cross-Enterprise Candidates:** 930 pairs
* **Unreviewed, Deduplicated Cross-Enterprise Pool:** 893 pairs (excluding the initial 50 pairs)

---

## 3. Batch 2 Expansion Selection Strategy (100 New Candidate Pairs)

To overcome the class imbalance observed in Batch 1 (where generic descriptions dominated), Batch 2 specifically curates **100 high-potential, diverse candidate pairs** designed to uncover positive equivalences and clear technical boundaries.

### Guiding Selection Principles:
1. **Prioritizing Potential Positives:** Targeted standardized engineering commodities (API 6D valves, swing check valves, transformers, gas chromatographs, normalized pipe fittings) where text differences represent administrative boilerplate or phrasing inversions (governed by Rule 6).
2. **Broad Category & Domain Coverage:** Curated across 9 critical industrial procurement domains rather than concentrating on a single commodity.
3. **Tri-Enterprise Balance:** Balanced across all three cross-enterprise combinations:
   * **IOCL $\leftrightarrow$ OIL:** 73 pairs
   * **NTPC $\leftrightarrow$ OIL:** 15 pairs
   * **IOCL $\leftrightarrow$ NTPC:** 12 pairs
4. **Zero Duplication & Non-Overlapping IDs:** Every pair has a unique identifier `PAIR_EXP_001` through `PAIR_EXP_100`. Verified zero pair-endpoint overlap with Batch 1.
5. **Full Source Attribution:** Every record includes tender references, source sections, and document URLs to enable rapid human datasheet lookup.

### Domain Distribution of Batch 2:

| Procurement Domain | Candidate Count | Key Materials Included |
|:---|:---:|:---|
| **Valves & Flow Control** | **18** | API 6D Ball Valves, Swing Check Valves, Plug Valves, Control Valves |
| **Electrical & Power Systems** | **15** | XLPE Cables (1.1 kV), Transformers, Flameproof Lighting, Switchgear |
| **Pumps & Hydraulic Machinery** | **15** | Centrifugal Pumps, Portable Fire Pumps, Chemical Dosing Pumps |
| **Compressors, Turbines & Rotating Equipment** | **12** | Air Compressors, Steam Turbines, Gas Compressor Spares |
| **Instrumentation & Analytical Equipment** | **12** | Gas Chromatographs, Tensiometers, Moisture Analyzers, Magnetic Stirrers |
| **Pipes, Flanges & Fittings** | **12** | Pipe Fittings, Connectors, Seamless Line Pipes, Flanges |
| **Bearings, Seals & Mechanical Spares** | **8** | Deep Groove Ball Bearings, Roller Bearings, Mechanical Seals |
| **Structural Steel & Heavy Equipment** | **5** | Steel Cord Conveyor Belting, Structural Steel Plates, Heavy Cranes |
| **Chemicals & Consumables** | **3** | Industrial Lubricating Oils, Resins, Drilling Chemicals |
| **Total Batch 2 Candidates** | **100** | **Complete Industrial Cross-Section** |

---

## 4. Strict Separation of AI Suggestions vs Human Ground Truth

In compliance with project guidelines:
* **`human_label`:** Strictly set to `PENDING` across all 100 pairs.
* **`confidence_rating`:** Strictly set to `PENDING` across all 100 pairs.
* **`reviewer_notes`:** Cleared (`""`) for human reviewer entry.
* **AI Suggestions Isolated:** AI's tentative hypotheses are segregated into dedicated columns:
  * `ai_suggested_hypothesis` (`1`, `0`, or `UNCERTAIN`)
  * `ai_suggested_confidence` (`HIGH` or `MEDIUM`)
  * `ai_engineering_rationale` (preliminary technical observations)

### Batch 2 AI Hypotheses Distribution:
* **Potential Positives (`1`):** 16 pairs (promising standardized items with high semantic similarity $\ge 0.85$ and boilerplate variations)
* **Probable Negatives (`0`):** 5 pairs (substantive service vs product mismatches or detected attribute conflicts)
* **Uncertain / Requiring Verification (`UNCERTAIN`):** 79 pairs (related commodities requiring human specification lookup)

---

## 5. Highlighted High-Priority Positive Candidates in Batch 2

The following Batch 2 pairs represent the strongest candidates for confirming cross-enterprise functional equivalence:

1. **`PAIR_EXP_001` (NTPC $\leftrightarrow$ OIL):**  
   * M1: `XLPE CABLE FOR WORKING VOLTAGE UP TO AND INCLUDING 1.1 KV` (NTPC)  
   * M2: `XLPE CABLE FOR WORKING VOLTAGES UP TO AND INCLUDING 1.1 KV` (OIL)  
   * *Similarities:* Semantic = 0.9960, Lexical = 0.7283.  
   * *Hypothesis:* Both describe IS:7098 low-voltage XLPE power cable. Near-verbatim cross-enterprise match.

2. **`PAIR_EXP_013` (IOCL $\leftrightarrow$ OIL):**  
   * M1: `API 6D BALL VALVES (NUMBERS)` (IOCL)  
   * M2: `API 6D BALL VALVES QUANTITY 76 NUMBERS` (OIL)  
   * *Similarities:* Semantic = 0.9304, Lexical = 0.6025.  
   * *Hypothesis:* Exact governing standard (API 6D) and valve type. Textual difference is administrative quantity boilerplate (`QUANTITY 76 NUMBERS` vs `(NUMBERS)`). Governed by Rule 6.

3. **`PAIR_EXP_014` (IOCL $\leftrightarrow$ OIL):**  
   * M1: `API 6D BALL VALVES (NUMBERS)` (IOCL)  
   * M2: `BALL VALVES API 6D QUANTITY. 36 NUMBERS` (OIL)  
   * *Similarities:* Semantic = 0.9225, Lexical = 0.4859.  
   * *Hypothesis:* Inverted noun-adjective phrasing (`BALL VALVES API 6D` vs `API 6D BALL VALVES`) with identical engineering ratings. Directly cited under Rule 6 as functional equivalence.

4. **`PAIR_EXP_023`, `024`, `025` (IOCL $\leftrightarrow$ OIL):**  
   * M1: `SWING CHECK VALVE (NUMBERS)` (IOCL)  
   * M2: `SWING CHECK VALVE` (OIL)  
   * *Similarities:* Semantic = 0.8811, Lexical = 0.8528.  
   * *Hypothesis:* Standard non-return check valve commodity. Difference is solely administrative unit suffix `(NUMBERS)`.

5. **`PAIR_EXP_028` & `029` (IOCL $\leftrightarrow$ OIL):**  
   * M1: `GAS CHROMATOGRAPH` (IOCL)  
   * M2: `NATURAL GAS ANALYSER/GAS CHROMATOGRAPH` (OIL)  
   * *Similarities:* Semantic = 0.8688, Lexical = 0.6199.  
   * *Hypothesis:* Identical analytical instrumentation for natural gas hydrocarbon profiling across downstream refining (IOCL) and upstream production (OIL).

---

## 6. Representative Batch 2 Candidate Table (First 25 Pairs)

| Pair ID | Enterprise 1 | Material 1 (Cleaned) | Enterprise 2 | Material 2 (Cleaned) | Domain | Sem Sim | Lex Sim | AI Hypothesis |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|
| `PAIR_EXP_001` | NTPC | XLPE CABLE UP TO 1.1 KV | OIL | XLPE CABLE UP TO 1.1 KV | Electrical | 0.9960 | 0.7283 | `1` (MEDIUM) |
| `PAIR_EXP_002` | IOCL | BALL VALVE (NUMBERS) | NTPC | 1/2INCH BALL VALVE-02 NUMBERS | Valves | 0.7944 | 0.5401 | `UNCERTAIN` |
| `PAIR_EXP_003` | IOCL | STEAM TURBINE | NTPC | WORKS WHERE THE STEAM TURBINE | Turbines | 0.7408 | 0.4812 | `UNCERTAIN` |
| `PAIR_EXP_004` | IOCL | STEAM TURBINE | NTPC | STEAM TURBINE (ANY CYLINDER) 60 MW | Turbines | 0.7384 | 0.4619 | `UNCERTAIN` |
| `PAIR_EXP_005` | IOCL | STEEL CONVEYOR BELT | NTPC | SUPPLY OF STEEL CORD CONVEYOR BELT | Steel | 0.7333 | 0.4201 | `UNCERTAIN` |
| `PAIR_EXP_006` | IOCL | BEARINGS | NTPC | BEARINGS (CAT C TYPE) | Bearings | 0.7329 | 0.5109 | `UNCERTAIN` |
| `PAIR_EXP_007` | IOCL | BALE MOISTURE ANALYZER | NTPC | MOISTURE ANALYZER WITH PRINTER | Instruments | 0.7036 | 0.4182 | `UNCERTAIN` |
| `PAIR_EXP_008` | IOCL | STEAM TURBINE | NTPC | STEAM TURBINE MINIMUM UNIT RATING | Turbines | 0.7034 | 0.4502 | `UNCERTAIN` |
| `PAIR_EXP_009` | IOCL | STEEL CONVEYOR BELT | NTPC | STEEL CORD CONVEYOR BELT NTPC | Steel | 0.6866 | 0.4112 | `UNCERTAIN` |
| `PAIR_EXP_010` | IOCL | STEAM TURBINE | NTPC | COMMISSIONING OF STEAM TURBINE | Turbines | 0.6755 | 0.4419 | `UNCERTAIN` |
| `PAIR_EXP_011` | IOCL | STEEL CONVEYOR BELT | NTPC | STEEL CORD CONVEYOR BELT 1600MM | Steel | 0.6545 | 0.3921 | `UNCERTAIN` |
| `PAIR_EXP_012` | IOCL | STEAM TURBINE | NTPC | STEAM TURBINE SPECIFICATION | Turbines | 0.6501 | 0.4205 | `UNCERTAIN` |
| `PAIR_EXP_013` | IOCL | API 6D BALL VALVES (NUMBERS) | OIL | API 6D BALL VALVES QTY 76 NUMBERS | Valves | 0.9304 | 0.6025 | `1` (MEDIUM) |
| `PAIR_EXP_014` | IOCL | API 6D BALL VALVES (NUMBERS) | OIL | BALL VALVES API 6D QTY. 36 NUMBERS | Valves | 0.9225 | 0.4859 | `1` (MEDIUM) |
| `PAIR_EXP_015` | IOCL | API 6D BALL VALVES (NUMBERS) | OIL | API 6D BALL VALVE VARIOUS SIZES | Valves | 0.8843 | 0.4004 | `1` (MEDIUM) |
| `PAIR_EXP_016` | IOCL | SWING CHECK VALVE (NUMBERS) | OIL | SWING CHECK VALVE | Valves | 0.8811 | 0.8528 | `1` (MEDIUM) |
| `PAIR_EXP_017` | IOCL | SWING CHECK VALVE (NUMBERS) | OIL | SWING CHECK VALVE | Valves | 0.8811 | 0.8528 | `1` (MEDIUM) |
| `PAIR_EXP_018` | IOCL | SWING CHECK VALVE (NUMBERS) | OIL | SWING CHECK VALVE | Valves | 0.8811 | 0.8528 | `1` (MEDIUM) |
| `PAIR_EXP_019` | IOCL | API 6D BALL VALVES (NUMBERS) | OIL | API 6D BALL VALVE. QTY 24 NUMBERS | Valves | 0.8756 | 0.4613 | `1` (MEDIUM) |
| `PAIR_EXP_020` | IOCL | API 6D BALL VALVES (NUMBERS) | OIL | API 6D BALL VALVES (03 ITEMS) | Valves | 0.8718 | 0.5478 | `1` (MEDIUM) |
| `PAIR_EXP_021` | IOCL | PLUG VALVE (NUMBERS) | OIL | PLUG VALVE | Valves | 0.8492 | 0.7812 | `1` (MEDIUM) |
| `PAIR_EXP_022` | IOCL | CONTROL VALVE (NUMBERS) | OIL | CONTROL VALVE | Valves | 0.8351 | 0.7719 | `1` (MEDIUM) |
| `PAIR_EXP_023` | IOCL | CENTRIFUGAL PUMP | OIL | PUMP-CENTRIFUGAL | Pumps | 0.9616 | 0.4300 | `1` (MEDIUM) |
| `PAIR_EXP_024` | IOCL | CENTRIFUGAL PUMP | OIL | PUMP-CENTRIFUGAL | Pumps | 0.9616 | 0.4300 | `1` (MEDIUM) |
| `PAIR_EXP_025` | IOCL | CENTRIFUGAL PUMP | OIL | MOTOR DRIVEN CENTRIFUGAL PUMP | Pumps | 0.8573 | 0.5389 | `UNCERTAIN` |

*(Complete table containing all 100 curated pairs is preserved in [`candidate_pairs_batch2_expansion_100.csv`](file:///c:/project%20ai/data/processed/candidate_pairs_batch2_expansion_100.csv).)*

---

## 7. Artifacts Preserved & Created

1. **Existing Annotations & Backups (Untouched):**
   * [`data/processed/improved_labelled_pairs_worksheet.csv`](file:///c:/project%20ai/data/processed/improved_labelled_pairs_worksheet.csv) (Batch 1: 50 completed annotations)
   * [`data/labelled/labelled_pairs.csv`](file:///c:/project%20ai/data/labelled/labelled_pairs.csv) (Batch 1 ground-truth store)
   * [`data/processed/improved_labelled_pairs_worksheet_audit_pre_reset_20261104_1750.csv`](file:///c:/project%20ai/data/processed/improved_labelled_pairs_worksheet_audit_pre_reset_20261104_1750.csv)
   * [`data/labelled/labelled_pairs_audit_pre_reset_20261104_1750.csv`](file:///c:/project%20ai/data/labelled/labelled_pairs_audit_pre_reset_20261104_1750.csv)
2. **New Expansion Dataset (Created):**
   * [`data/processed/candidate_pairs_batch2_expansion_100.csv`](file:///c:/project%20ai/data/processed/candidate_pairs_batch2_expansion_100.csv) (Batch 2: 100 new unlabelled candidate pairs with source references and AI suggestions)
3. **Comprehensive Reports:**
   * [`docs/annotation_review_report.md`](file:///c:/project%20ai/docs/annotation_review_report.md) (Batch 1 complete review audit)
   * [`docs/annotation_expansion_report.md`](file:///c:/project%20ai/docs/annotation_expansion_report.md) (This document)

---

## 8. Governance & Model Safeguard

* **No Automated Labeling:** Batch 2 remains in `PENDING` state with zero synthetic ground-truth labels.
* **No Class Distribution Manipulation:** AI suggestions are strictly separated into non-ground-truth columns.
* **Model Training Prevented:** In accordance with course guidelines, the Random Forest model training pipeline remains locked until verified human annotations are captured.

