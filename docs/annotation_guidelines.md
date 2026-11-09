# Human Review & Pairwise Annotation Guidelines

**Course:** CSE 276 – CA-1 Academic Project  
**Project:** Material Code Harmonisation Across Public Enterprises  
**Target:** Ground-Truth Annotation Protocol for Pairwise Supervised Classification  

---

## 1. Overview & Annotation Purpose

The objective of human annotation is to establish high-integrity ground-truth labels for candidate material pairs across public sector enterprises (Oil India Limited, Indian Oil Corporation Limited, NTPC Limited). 

Supervised entity resolution requires genuine binary labels predicting whether two descriptions represent **functionally equivalent materials** that could be substituted or harmonised under a unified material master code without operational, safety, or engineering risk.

> [!WARNING]
> **Prohibition of Automated / Threshold-Based Label Generation:**  
> Labels must **NEVER** be assigned automatically based on lexical similarity (TF-IDF) or semantic similarity (Sentence Transformers). High semantic similarity does not guarantee physical interchangeability (e.g. 250mm vs 300mm pipe diameter yields semantic similarity > 0.99, yet represents completely incompatible components). Every training label must stem from human engineering review.

---

## 2. Standard Label Taxonomy

Reviewers must assign one of the following three mutually exclusive labels in the `human_label` column:

| Label Code | Category | Operational Definition | Inclusion in Supervised Training |
|:---:|---|---|:---:|
| **`1`** | **Functionally Equivalent** | Both records describe the same physical engineering component or material that could be interchanged in industrial procurement without compromising safety, engineering function, or operational performance. | **Included (Positive Class)** |
| **`0`** | **Not Functionally Equivalent** | The records represent distinct, incompatible, or non-interchangeable items (e.g., conflicting dimensions, conflicting pressure ratings, incompatible material grades, or distinct equipment). | **Included (Negative Class)** |
| **`UNCERTAIN`** | **Insufficient Evidence** | The text descriptions do not provide sufficient technical specifications or evidence to verify functional interchangeability. | **Excluded from Training / Held for Audit** |

---

## 3. Decision Rules by Technical Scenario

### Rule 1: Confirmed Incompatible Physical Specifications (Mandatory Label `0`)
When both descriptions clearly define identity-defining physical specifications that differ incompatibly, the pair **MUST be assigned Label `0`**, regardless of how high lexical or semantic similarity scores are:
- **Diameter Mismatches:** e.g., `250MM NB` vs `300MM NB`, `15MM` vs `3.15MM`, `1400MM` vs `1600MM` conveyor belt width $\rightarrow$ **Label `0`**.
- **Material Grade Incompatibilities:** e.g., `SS304` (Austenitic) vs `SS316` (Molybdenum-bearing marine/acid grade) $\rightarrow$ **Label `0`**.
- **Pressure Class Discrepancies:** e.g., `150#` ($150\text{ psi}$ class) vs `300#` ($300\text{ psi}$ class) $\rightarrow$ **Label `0`**.
- **Pipe Schedule Variations:** e.g., `SCH 40` (standard wall) vs `SCH 80` (heavy wall) $\rightarrow$ **Label `0`**.
- **Thread Specification Differences:** e.g., `M10` vs `M12` metric thread $\rightarrow$ **Label `0`**.

### Rule 2: Confirmed Compatible Specifications (Supporting Evidence, NOT Automatic Proof)
- Identical extracted attributes (e.g. both have `diameter=2INCH` and `grade=SS304`) provide **strong supporting evidence**, but **do NOT automatically prove functional equivalence**.
- The reviewer must confirm that the underlying components belong to the same functional type. For example, a `2INCH SS304 GATE VALVE` and a `2INCH SS304 BALL VALVE` share identical diameter and grade, but are fundamentally distinct valve mechanisms $\rightarrow$ **Label `0`**.
- A positive match (`1`) requires **both** attribute compatibility AND functional commodity equivalence.

### Rule 3: Inspection Requirement for Extracted Conflicts (No Blind Regex Rejection)
> [!CRITICAL]
> **A regex-extracted conflict must NOT automatically become a negative label without verifying the underlying descriptions:**
> - Automated regex parsers may occasionally extract numbers from part numbers, drawing numbers, or order codes as diameters (e.g. `608 TEST BEARINGS` where `608` is an ISO bearing series number, not a 608mm diameter; or `16MEG-23` where `23` is a tender section).
> - Reviewers must inspect the `raw_description_1` and `raw_description_2` columns in the worksheet to verify whether an extracted attribute represents a true physical specification or a parsing artefact.
> - If an extracted conflict is determined to be a spurious extraction artefact, the reviewer must override the conflict and evaluate the true underlying text.

### Rule 4: Handling Missing Attributes (Do NOT Automatically Label `0`)
When key physical attributes are omitted in one or both records:
- **Do NOT automatically assign Label `0`:** Omission of an attribute does not imply incompatibility.
- If one record is a generic descriptor and the other is a specific item (e.g., `CENTRIFUGAL PUMP` vs `API 610 CENTRIFUGAL PUMP`), the generic record may represent the same procurement category, but specific interchangeability cannot be guaranteed.
- If the unstated attributes could reasonably differ across installations and cannot be confirmed from text alone, mark as **`UNCERTAIN`**.
- Only assign Label `0` if the missing-attribute item is clearly in a different commodity category (e.g. `IMPELLER FOR PUMP` vs `PUMP ASSEMBLY`).

### Rule 5: Ambiguous Descriptions & Insufficient Evidence (Mandatory `UNCERTAIN`)
- Generic descriptions lacking model numbers, dimensional ratings, or manufacturer codes (e.g., `BEARINGS` vs `BEARINGS`, `ELECTRICAL CABLES` vs `ELECTRICAL CABLES`, `ENGINE SPARES` vs `ENGINE SPARES`) cannot be verified as functionally equivalent from procurement text alone $\rightarrow$ **`UNCERTAIN`**.
- Tender line items referencing external documents without technical specifications (e.g., `AS PER ANNEXURE-I`, `AS PER TENDER SPECIFICATION`) lack sufficient evidence $\rightarrow$ **`UNCERTAIN`**.

### Rule 6: Procurement Boilerplate & Phrasing Inversions (Label `1`)
- Phrasing differences stemming from administrative procurement terms (e.g., `SUPPLY OF`, `PROCUREMENT OF`, `FOR USE IN`, `QUANTITY 36 NUMBERS`, `NOS`) that do not alter the physical item should be disregarded.
- Inverted noun-adjective syntax (e.g., `CENTRIFUGAL PUMP` vs `PUMP-CENTRIFUGAL`, or `BALL VALVES API 6D` vs `API 6D BALL VALVES`) with identical engineering ratings represents functional equivalence $\rightarrow$ **Label `1`**.

### Rule 7: Administrative & Work Contracts (Label `0`)
- Tender line items describing service contracts, Annual Maintenance Contracts (AMC), or labor works (e.g., `AMC OF DCN ANALYZER` vs `BATTERY ANALYZER`) describe services or instruments, not interchangeable material spares $\rightarrow$ **Label `0`**.

---

## 4. Reviewer Confidence Ratings

Reviewers should record their subjective confidence in the `confidence_rating` column:
- **`HIGH`:** Clear technical evidence present in both descriptions; unambiguously interchangeable or unambiguously incompatible.
- **`MEDIUM`:** Likely equivalent or non-equivalent based on domain convention, but some minor secondary specifications (e.g. finish, coating) are unstated.
- **`LOW`:** Significant ambiguity; recommendation leans toward a label, but field verification or engineering datasheet lookup is recommended.

---

## 5. Review Workflow & Audit Trail

1. **Worksheet Location:** Annotations must be recorded in [`data/processed/improved_labelled_pairs_worksheet.csv`](file:///c:/project%20ai/data/processed/improved_labelled_pairs_worksheet.csv).
2. **Preservation:** The baseline file `data/processed/labelled_pairs_template.csv` remains preserved in its original unedited state.
3. **Audit Evidence:** Reviewers must supply brief justification in `reviewer_notes` whenever assigning `UNCERTAIN` or when overriding high similarity scores with a `0`.
4. **Adjudication:** For production datasets, any pair where two independent reviewers disagree must be adjudicated by a third senior domain engineer.

