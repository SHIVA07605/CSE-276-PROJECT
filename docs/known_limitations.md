# Known Limitations & Domain Constraints

**Course:** CSE 276 – CA-1 Academic Project  
**Project:** Material Code Harmonisation Across Public Enterprises (IOCL, OIL, NTPC)  
**Document:** System Limitations, Engineering Risks, and Domain Boundaries  
**Date:** 6 November 2026  

---

## 1. Overview

Material code harmonisation across public sector enterprises involves safety-critical industrial equipment deployed in high-pressure hydrocarbon refineries (IOCL), upstream exploration oilfields (OIL), and thermal power generating stations (NTPC). Erroneously asserting equivalence between incompatible engineering items can lead to equipment failure, procurement delays, or catastrophic industrial hazards.

This document records the empirical limitations of NLP and machine learning techniques on CPSE material master catalogues.

---

## 2. Primary Technical & Domain Limitations

### Limitation 1: Severe Catalog Text Ambiguity & Generic Titles
- **Observation:** In the candidate pool of 6,634 records, over 55% of descriptions represent high-level procurement summary titles (e.g. `AIR COMPRESSOR`, `BEARINGS`, `BALL VALVE`, `STEAM ENGINE SPARES`) rather than complete engineering line items.
- **NLP Failure Mode:** String similarity and dense semantic vectors produce near-perfect similarity scores ($\ge 0.95$) for these generic pairs. However, two items labeled `AIR COMPRESSOR` may differ completely in technology (rotary screw vs reciprocating), flow capacity ($50\text{ CFM}$ vs $1200\text{ CFM}$), and discharge pressure ($7\text{ bar}$ vs $350\text{ bar}$).
- **Mitigation:** The system strictly prohibits classifying records as `EQUIVALENT` based solely on high lexical or semantic similarity. Generic pairs lacking critical operational specifications are mandatorily routed to `NEEDS_REVIEW` or labeled `UNCERTAIN`.

### Limitation 2: Missing & Asymmetric Engineering Specifications
- **Observation:** Regex-based attribute extraction achieves variable coverage across the candidate pool:
  - Material Grade: $\approx 13.2\%$ coverage.
  - Diameter / Size: $\approx 10.4\%$ coverage.
  - Pressure Class: $\approx 5.1\%$ coverage.
  - Pipe Schedule: $\approx 3.8\%$ coverage.
  - Metric Thread: $\approx 4.2\%$ coverage.
- **Asymmetric Complementarity:** Frequently, Enterprise A specifies standard and material grade (e.g. `API 6D BALL VALVE SS316`) but omits nominal diameter, while Enterprise B specifies diameter and pressure class (e.g. `2 INCH BALL VALVE CLASS 150`) but omits the governing standard.
- **Mitigation:** The system never assumes missing attributes match. Asymmetric pairs cannot be auto-merged and are flagged with explicit missing-specification warnings in the reviewer UI.

### Limitation 3: Domain & Operating Environment Incompatibilities
- **Hydrocarbon vs Power Utility Standards:** IOCL and OIL require American Petroleum Institute standards (API 610 for centrifugal pumps, API 6D for pipeline valves, API 600 for steel gate valves) designed for volatile, toxic, flammable hydrocarbon service. NTPC catalogues prioritize Bureau of Indian Standards (BIS) and ASME boiler and pressure vessel codes for superheated steam and cooling water service.
- **Risk:** An NTPC water circulation pump and an IOCL crude distillation pump may share identical flow rates ($100\text{ m}^3/\text{hr}$) and head ($50\text{ m}$), but differ in mechanical seal plans, casing metallurgy, and ATEX/hazardous area electrical certifications.
- **Mitigation:** Category-aware rules flag cross-domain pairings and mandate SME review when operating service cannot be deduced from catalogue text.

### Limitation 4: Natural Graph Hubs & Component Bridging
- **Observation:** As discovered during the Batch 2 validation audit, 11 high-volume material entities (e.g. standard bearings, generic power cables, API 6D valves) overlap between Batch 1 and Batch 2 across 38 pairs.
- **Graph Impact:** These high-degree nodes act as graph hubs, bridging previously disjoint components into large connected subgraphs.
- **Mitigation:** Without conflict-aware clustering, standard connected components would merge dozens of disparate records into a single gigantic cluster. The project enforces transitive conflict detection, cross-product member validation, and leakage-free graph partitioning.

### Limitation 5: Impossibility of Fully Autonomous Entity Resolution
- **Conclusion:** Catalogue descriptions alone do not contain full engineering datasheets, dimensional drawings, or vendor part numbers. Autonomous AI cannot safely declare final legal equivalence for high-consequence public procurement.
- **Role of the System:** The pipeline serves as an intelligent, high-precision candidate filtering and audit assistant that eliminates 99.9% of non-matching pairs, detects obvious parameter conflicts, and presents structured evidence to human procurement engineers.


### Limitation 6: Self-Supervised Training Distribution Shift (added 6 Nov 2026)
- **Observation:** The trained hybrid model (`models/hybrid_er_model.joblib`) learns from pairs whose labels are known *by construction* (format-only rewrites = match; single-specification changes = non-match), generated from real catalogue records. Genuine cross-enterprise variation (different authors, partial specifications, vendor-specific naming) is messier than the constructed variation.
- **Consequence:** Held-out test scores are an **upper bound** on real-world accuracy. The 19 binary review-set labels (AI-assisted, not independent human ground truth) confirm the behaviour on real pairs but are too few (2 positives) for a precise estimate.
- **Mitigation:** (1) review-set records are excluded from training and splits are group-disjoint; (2) the deployed policy auto-merges only high-probability pairs that share identity specifications, and routes generic or asymmetric descriptions to human review; (3) `results/metrics/batch2_review_priority.csv` orders the pending pairs for labelling, and re-running `python train_er_model.py` re-evaluates on every new human label. See [`ml_model_report.md`](ml_model_report.md).
