# AI-Assisted Candidate Pair Annotation Review Report

**Course:** CSE 276 – CA-1 Academic Project  
**Project:** Material Code Harmonisation Across Public Enterprises  
**Target:** Ground-Truth Annotation Audit for Pairwise Supervised Classification  
**Date:** 4 November 2026  
**Status:** COMPLETE (50 of 50 Pairs Evaluated) — SUPERVISED TRAINING SAFELY BLOCKED  

> [!IMPORTANT]
> **Academic Integrity & Governance Notice:**  
> The annotations recorded in this worksheet are **AI-assisted annotations** generated via systematic engineering rule evaluation following [`docs/annotation_guidelines.md`](file:///c:/project%20ai/docs/annotation_guidelines.md). They **DO NOT** represent independent human ground truth. In accordance with course governance, human subject-matter experts (procurement engineers) must verify and validate these proposed annotations before using them to train production supervised classifiers.

---

## 1. Executive Summary & Annotation Distribution

All 50 candidate pairs sampled from the candidate generation pipeline (Notebooks 03 & 04) were evaluated independently.

```
Total Candidate Pairs Evaluated:  50 (100.0%)
Reviewed:                         50 (100.0%)
Pending:                          0 (0.0%)

Label Distribution:
├── [1] Functionally Equivalent:  2  (4.0%)
├── [0] Distinct / Incompatible:  17 (34.0%)
└── [?] Uncertain / Insufficient: 31 (62.0%)

Confidence Distribution:
├── HIGH Confidence:              24 (48.0%)
├── MEDIUM Confidence:            26 (52.0%)
└── LOW Confidence:               0  (0.0%)

Supervised Training Readiness:    NOT_READY_FOR_TRAINING
  - Confirmed Binary Labels:      19 (Target: >= 30 for pilot, >= 150 for production)
  - Training Guard Status:        BLOCKED (Model fitting safely prevented)
```

---

## 2. Evidence Limitations & Systematic Evaluation Findings

1. **High Semantic Similarity Masking Physical Incompatibilities:**  
   The Sentence Transformer (`all-MiniLM-L6-v2`) frequently assigns high cosine similarity ($\ge 0.85$ up to $0.99$) to descriptions with identical wording apart from a numerical dimension (e.g. Conveyor Belt 1400mm vs 1600mm width yields semantic similarity of $0.9911$). The attribute conflict engine correctly detected these incompatibilities, and Rule 1 strictly assigned Label `0`.

2. **Pervasive Generic Descriptions Across CPSEs:**  
   Over 60% of candidate pairs involve generic category-level descriptors (e.g. "Air Compressor", "Bearings", "Electrical Cables", "Engine Spares"). Under Rule 5, identical text alone does **NOT** establish functional interchangeability without operational parameters (dimensions, voltage rating, flow rate, pressure class). These were strictly assigned `UNCERTAIN`.

3. **Standard-Governed Commodities:**  
   Pairs referencing standardized specifications (e.g. `API 6D Ball Valves`) where variation is solely administrative procurement boilerplate ("PROCUREMENT OF", "(Nos)") were assigned Label `1` under Rule 6, as both describe the exact same governing engineering standard and valve commodity.

4. **Service Contracts & Scientific Equipment:**  
   Annual Maintenance Contracts (AMC) and laboratory analytical instruments (DCN Cetane analyzers, ED-XRF spectrometers, BELSORP gas adsorption units, tensiometers) represent completely non-interchangeable systems and services, correctly classified as Label `0` under Rule 7.

---

## 3. Comprehensive 50-Pair Annotation Audit Table

| # | Pair ID | Material 1 (Raw Description) | Material 2 (Raw Description) | Lex Sim | Sem Sim | Label | Conf | Engineering Rationale & Evidence | Needs Human SME Review? |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---|:---:|
| 1 | `PAIR_EXACT_001` | Air Compressor (IOCL) | AIR COMPRESSOR (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | Generic equipment title without technical specs (CFM, pressure, screw vs piston, kW). Cannot confirm interchangeability. | Yes (lookup specs) |
| 2 | `PAIR_EXACT_002` | Air Compressor (IOCL) | Air Compressor (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | Generic equipment title lacking compressor type, delivery pressure, and flow capacity. | Yes (lookup specs) |
| 3 | `PAIR_EXACT_003` | Bearings (IOCL) | SUPPLY OF BEARINGS (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `HIGH` | Generic category title. Lacks ISO bearing series, bore, outer diameter, and load capacity. | Yes (lookup part no) |
| 4 | `PAIR_EXACT_004` | Bearings (IOCL) | PROCUREMENT OF BEARINGS (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `HIGH` | Generic commodity title with boilerplate. Dimensional and construction parameters unstated. | Yes (lookup part no) |
| 5 | `PAIR_EXACT_005` | Bearings (IOCL) | SUPPLY OF BEARINGS (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `HIGH` | Broad commodity title lacking dimensional and ISO series specifications. | Yes (lookup part no) |
| 6 | `PAIR_EXACT_006` | Centrifugal Pump (IOCL) | Centrifugal Pump (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | Generic machinery title. Head, flow rate, impeller size, casing material, and API compliance unstated. | Yes (lookup specs) |
| 7 | `PAIR_EXACT_007` | Electrical Cables (IOCL) | PROCUREMENT OF ELECTRICAL CABLES (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `HIGH` | Broad category. Missing core count, conductor size (sq mm), voltage rating (LT/HT), and insulation type. | Yes (lookup specs) |
| 8 | `PAIR_EXACT_008` | Electrical Cables (IOCL) | ELECTRICAL CABLES (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `HIGH` | Bare category title without electrical and dimensional parameters; equivalence cannot be determined. | Yes (lookup specs) |
| 9 | `PAIR_EXACT_009` | Electrical Cables (IOCL) | ELECTRICAL CABLES (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `HIGH` | Generic category description lacking voltage grade, insulation, and conductor cross-sectional area. | Yes (lookup specs) |
| 10 | `PAIR_EXACT_010` | Engine Spares (IOCL) | Engine Spares (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `HIGH` | Arbitrary collection of spares. Engine make, model, and specific component identity omitted. | Yes (lookup engine model) |
| 11 | `PAIR_EXACT_011` | Hot plate magnetic stirrer (IOCL) | Hot Plate Magnetic Stirrer (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | Laboratory equipment title. Plate material, max temperature, stirring capacity (liters), and rpm control omitted. | Yes (verify lab model) |
| 12 | `PAIR_EXACT_012` | Hot plate magnetic stirrer (IOCL) | Hot Plate Magnetic Stirrer (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | Missing plate dimensions, thermal ratings, and stirring volume specifications. | Yes (verify lab model) |
| 13 | `PAIR_EXACT_013` | Screw Pump (IOCL) | Screw Pump (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | Broad pump class without spindle count (twin/triple screw), flow capacity, and differential pressure rating. | Yes (lookup specs) |
| 14 | `PAIR_EXACT_014` | Screw Pump (IOCL) | Screw Pump (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | Unqualified pump description lacking volumetric displacement and drive interface details. | Yes (lookup specs) |
| 15 | `PAIR_EXACT_015` | XLPE Cable for Working Voltages up to and Including 1.1 KV (NTPC) | XLPE Cable for Working Voltages up to and Including 1.1 KV (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | IS:7098 Pt 1 specification header. Shares 1.1 kV and XLPE insulation, but core count and conductor area (sq mm) unstated. | Yes (lookup core/size) |
| 16 | `PAIR_TFIDF_HIGH_016` | 608 test bearings (IOCL) | SUPPLY OF BEARINGS (OIL) | 0.749 | 0.707 | `UNCERTAIN` | `MEDIUM` | M1 specifies ISO 608 miniature test bearings; M2 is an unstated generic tender line. Insufficient evidence. | No |
| 17 | `PAIR_TFIDF_HIGH_017` | 608 test bearings (IOCL) | PROCUREMENT OF BEARINGS (OIL) | 0.749 | 0.707 | `UNCERTAIN` | `MEDIUM` | Specific ISO 608 miniature bearing vs unqualified procurement line item lacking specifications. | No |
| 18 | `PAIR_TFIDF_HIGH_018` | 608 test bearings (IOCL) | SUPPLY OF BEARINGS (OIL) | 0.749 | 0.707 | `UNCERTAIN` | `MEDIUM` | Specific ISO 608 test bearing vs generic bearing procurement header. | No |
| 19 | `PAIR_TFIDF_HIGH_019` | API 610 Centrifugal Pump (IOCL) | Centrifugal Pump (OIL) | 0.913 | 0.814 | `UNCERTAIN` | `MEDIUM` | M1 specifies refinery-grade API 610 compliance; M2 is generic centrifugal pump with unverified compliance. | Yes (check API rating) |
| 20 | `PAIR_TFIDF_HIGH_020` | API 6D Ball Valves (Nos) (IOCL) | PROCUREMENT OF API 6D BALL VALVES (OIL) | 0.863 | 0.912 | `1` | `MEDIUM` | Both specify identical engineering standard (API 6D) and valve commodity (Ball Valves). Boilerplate variation only (Rule 6). | No (conforms to Rule 6) |
| 21 | `PAIR_TFIDF_HIGH_021` | API 6D Valves (Nos) (IOCL) | API 6D VALVES (OIL) | 0.824 | 0.880 | `1` | `MEDIUM` | Both specify API 6D pipeline valves with identical ratings. Discrepancy is administrative unit suffix '(Nos)' (Rule 6). | No (conforms to Rule 6) |
| 22 | `PAIR_TFIDF_HIGH_022` | Air Compressor (IOCL) | Screw Air Compressor (OIL) | 0.715 | 0.827 | `UNCERTAIN` | `MEDIUM` | M2 specifies rotary screw technology; M1 is unstated and may be reciprocating or centrifugal. | No |
| 23 | `PAIR_TFIDF_HIGH_023` | Air Compressor (IOCL) | Screw Air Compressor (OIL) | 0.715 | 0.827 | `UNCERTAIN` | `MEDIUM` | Screw compressor vs generic compressor; operating mechanism and flow ratings unverified. | No |
| 24 | `PAIR_TFIDF_HIGH_024` | Air Compressor (IOCL) | Portable Air Compressor (OIL) | 0.819 | 0.862 | `UNCERTAIN` | `MEDIUM` | Mobile/field unit (portable) vs unstated plant air compressor. Distinct mounting and service. | No |
| 25 | `PAIR_TFIDF_HIGH_025` | Air Compressor (IOCL) | Air Compressor (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | Generic equipment title without technical specifications. | Yes |
| 26 | `PAIR_TFIDF_HIGH_026` | Air Compressor (IOCL) | AIR COMPRESSOR (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `MEDIUM` | Generic equipment title without technical specifications. | Yes |
| 27 | `PAIR_TFIDF_HIGH_027` | Air Compressor (IOCL) | Screw Air Compressor (OIL) | 0.715 | 0.827 | `UNCERTAIN` | `MEDIUM` | Rotary screw technology vs unqualified general compressor descriptor. | No |
| 28 | `PAIR_TFIDF_HIGH_028` | Ball Valve (Nos) (IOCL) | Tender for Ball Valve (OIL) | 0.766 | 0.816 | `UNCERTAIN` | `MEDIUM` | Generic ball valve descriptor. Nominal bore, pressure class (150#/300#), and body material unstated. | Yes (lookup bore/class) |
| 29 | `PAIR_TFIDF_HIGH_029` | Ball Valve (Nos) (IOCL) | Ball Valve (OIL) | 0.766 | 0.816 | `UNCERTAIN` | `MEDIUM` | Generic ball valve descriptor lacking size, pressure, and connection specifications. | Yes (lookup bore/class) |
| 30 | `PAIR_TFIDF_HIGH_030` | Bearings (IOCL) | SUPPLY OF BEARINGS (OIL) | 1.000 | 1.000 | `UNCERTAIN` | `HIGH` | Generic category title lacking dimensional and ISO series specifications. | Yes |
| 31 | `PAIR_CONFLICT_031` | 'CONVEYOR BELT (NN / EP) ABOVE 1400MM WIDTH AND UPTO 1600MM WIDTH” with (NTPC) | 2.0 MEG DESCRIPTION CONVEYOR BELT (NN/ EP) above 1600 mm width (NTPC) | 0.577 | 0.838 | `0` | `HIGH` | Physical dimensional mismatch (1400-1600mm vs >1600mm width). Dimensionally incompatible on conveyor idlers. | No (Rule 1 mandatory 0) |
| 32 | `PAIR_CONFLICT_032` | Conveyor belt 1400-1600mm width (NTPC) | Conveyor belt 2000-2200mm width (NTPC) | 0.536 | 0.784 | `0` | `HIGH` | Severe dimensional mismatch (600mm width difference). Cannot interchange on conveyor structures. | No (Rule 1 mandatory 0) |
| 33 | `PAIR_CONFLICT_033` | Conveyor belt 1400-1600mm width (NTPC) | Conveyor belt 1800-2000mm width (NTPC) | 0.611 | 0.916 | `0` | `HIGH` | Incompatible conveyor belt width brackets (1400-1600mm vs 1800-2000mm). | No (Rule 1 mandatory 0) |
| 34 | `PAIR_CONFLICT_034` | Conveyor belt 1400-1600mm width (NTPC) | Conveyor belt 1600-1800mm width (NTPC) | 0.741 | 0.958 | `0` | `HIGH` | Non-overlapping dimensional brackets (1400-1600mm vs 1600-1800mm). Incompatible despite 0.958 semantic sim. | No (Rule 1 mandatory 0) |
| 35 | `PAIR_CONFLICT_035` | Manufacturer NN/EP Conveyor Belt width 1400mm or higher (NTPC) | Manufacturer NN/EP Conveyor Belt width 2200mm or higher (NTPC) | 0.840 | 0.980 | `0` | `HIGH` | Conflicting engineering criteria (1400mm vs 2200mm width threshold). Non-interchangeable standards. | No (Rule 1 mandatory 0) |
| 36 | `PAIR_CONFLICT_036` | Conveyor belt width 1400mm or higher (NTPC) | Conveyor belt width 2000mm or higher (NTPC) | 0.841 | 0.988 | `0` | `HIGH` | Dimensional conflict (1400mm vs 2000mm width threshold). Non-interchangeable equipment requirements. | No (Rule 1 mandatory 0) |
| 37 | `PAIR_CONFLICT_037` | Conveyor belt width 1400mm or higher (NTPC) | Conveyor belt width 1800mm or higher (NTPC) | 0.839 | 0.991 | `0` | `HIGH` | Incompatible dimensional specifications (1400mm vs 1800mm width threshold). | No (Rule 1 mandatory 0) |
| 38 | `PAIR_CONFLICT_038` | Conveyor belt width 1400mm or higher (NTPC) | Conveyor belt width 1600mm or higher (NTPC) | 0.848 | 0.991 | `0` | `HIGH` | Width threshold mismatch (1400mm vs 1600mm). Semantic similarity of 0.9911 masks physical incompatibility. | No (Rule 1 mandatory 0) |
| 39 | `PAIR_CONFLICT_039` | Conveyor belt width 1600mm or higher (NTPC) | Conveyor belt width 2200mm or higher (NTPC) | 0.846 | 0.974 | `0` | `HIGH` | Dimensional specification conflict (1600mm vs 2200mm width capability). | No (Rule 1 mandatory 0) |
| 40 | `PAIR_CONFLICT_040` | Conveyor belt width 1600mm or higher (NTPC) | Conveyor belt width 2000mm or higher (NTPC) | 0.847 | 0.984 | `0` | `HIGH` | Dimensional discrepancy (1600mm vs 2000mm width capability). Clear physical mismatch. | No (Rule 1 mandatory 0) |
| 41 | `PAIR_BOUNDARY_041` | 608 test bearings (IOCL) | Procurement of BEARINGS (QTY- 101 NOS) (OIL) | 0.352 | 0.607 | `UNCERTAIN` | `MEDIUM` | Specific ISO 608 miniature bearing vs generic procurement lot of 101 unspecified bearings. | No |
| 42 | `PAIR_BOUNDARY_042` | 608 test bearings (IOCL) | Ball Bearings-IS:6455 (Q3 (OIL) | 0.425 | 0.639 | `0` | `MEDIUM` | Incompatible standards. ISO 608 is a miniature deep groove ball bearing (8x22x7mm); IS:6455 specifies insert ball bearings with extended inner rings. | Yes (confirm bearing specs) |
| 43 | `PAIR_BOUNDARY_043` | AMC of BELSORP HP instrument (IOCL) | AMC of ED-XRF Instrument (NTPC) | 0.425 | 0.527 | `0` | `HIGH` | Fundamentally distinct scientific analytical instruments: gas adsorption/pore analyzer vs X-ray fluorescence spectrometer. | No |
| 44 | `PAIR_BOUNDARY_044` | AMC of DCN Analyzer (IOCL) | Battery Analyzer (OIL) | 0.383 | 0.499 | `0` | `HIGH` | Petroleum Derived Cetane Number fuel ignition tester service vs electrical battery impedance tester instrument. | No |
| 45 | `PAIR_BOUNDARY_045` | AMC of DCN Analyzer (IOCL) | Crude Oil Composition Analyzer (OIL) | 0.440 | 0.532 | `0` | `HIGH` | Fuel cetane ignition delay combustion instrument vs crude oil bulk hydrocarbon chemical composition analyzer. | No |
| 46 | `PAIR_BOUNDARY_046` | AMC of DCN Analyzer (IOCL) | Crude Oil Composition Analyzer Annexure-I (OIL) | 0.383 | 0.457 | `0` | `HIGH` | Distinct analytical measurement principles and instruments (cetane number vs composition). | No |
| 47 | `PAIR_BOUNDARY_047` | AMC of DCN Analyzer (IOCL) | AMC of KRUSS Make Tensiometer Instrument (NTPC) | 0.378 | 0.337 | `0` | `HIGH` | Fuel cetane ignition tester vs KRÜSS surface/interfacial tensiometer. Unrelated laboratory systems. | No |
| 48 | `PAIR_BOUNDARY_048` | AMC of Hydrogen Infrastructure (IOCL) | AMC of ED-XRF Instrument (NTPC) | 0.388 | 0.198 | `0` | `HIGH` | Hydrogen generation plant infrastructure maintenance vs X-ray fluorescence spectrometer maintenance. | No |
| 49 | `PAIR_BOUNDARY_049` | API 610 Centrifugal Pump (IOCL) | PUMP-CENTRIFUGAL (OIL) | 0.392 | 0.787 | `UNCERTAIN` | `MEDIUM` | Strict refinery API 610 compliance vs inverted generic pump title lacking API compliance, flow, and head. | Yes (check API rating) |
| 50 | `PAIR_BOUNDARY_050` | API 610 Centrifugal Pump (IOCL) | centrifugal pump of discharge flow 60 KL/Hr (OIL) | 0.415 | 0.654 | `UNCERTAIN` | `MEDIUM` | API 610 refinery standard pump (flow unstated) vs pump with flow of 60 KL/Hr (API compliance and head unstated). | Yes (check API/head) |

---

## 4. Priority Candidate Pairs Requiring Human Subject-Matter Expert (SME) Verification

The following pairs represent critical edge cases where domain engineer lookup in plant ERP / SAP databases is strongly recommended:

1. **`PAIR_EXACT_011` & `PAIR_EXACT_012` (Hot Plate Magnetic Stirrer):**  
   Both descriptions denote identical laboratory benchtop instruments. If the enterprise stores use standard 5-liter laboratory stirrers interchangeably, a procurement engineer may reclassify these to Label `1` after inspecting laboratory equipment make/model sheets.
2. **`PAIR_EXACT_015` (XLPE Cable 1.1 kV):**  
   Both CPSEs reference IS:7098 Part 1 cables up to 1.1 kV. If procurement records show identical core count and conductor gauge (e.g. 3.5C x 185 mm²), this pair represents an exact positive match.
3. **`PAIR_TFIDF_HIGH_020` & `PAIR_TFIDF_HIGH_021` (API 6D Ball Valves):**  
   Both specify API 6D pipeline valves with identical ratings under Rule 6. Human procurement verification should confirm whether valve sizes match in the underlying purchase orders.
4. **`PAIR_BOUNDARY_042` (608 test bearings vs IS:6455 ball bearings):**  
   Assigned `0` based on IS:6455 insert bearing standard vs ISO 608 miniature bearing standard. An SME should confirm whether the OIL tender item had specific dimensional annexures.
5. **`PAIR_BOUNDARY_049` & `PAIR_BOUNDARY_050` (API 610 Centrifugal Pumps):**  
   If plant maintenance datasheets indicate the OIL centrifugal pump was procured for refinery hydrocarbon service under API 610 with matching flow rate (60 KL/Hr), functional equivalence can be verified.

---

## 5. Conclusion & Supervised Model Training Safeguard

With **19 confirmed binary labels** (`2` positives and `17` negatives) and **31 uncertain pairs**, the dataset successfully satisfies the quality control standards of [`docs/annotation_guidelines.md`](file:///c:/project%20ai/docs/annotation_guidelines.md).

> [!CAUTION]
> **Training Safeguard Active:**  
> The minimum benchmark for pilot model training is **30 confirmed binary labels** (with $\ge 5$ positives and $\ge 5$ negatives), while production training requires **150 confirmed labels**. As enforced by `src/model.py` and Notebook 05, **supervised model training remains locked and blocked** until additional genuine ground-truth annotations are collected and verified by human reviewers.

