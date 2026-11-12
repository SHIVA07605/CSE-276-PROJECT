---
title: Material Match
emoji: 🔩
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Entity resolution for IOCL, OIL and NTPC material records
---

# Material Match

Public demo of the **hybrid entity-resolution model** from the CSE 276 (Artificial Intelligence Foundations) CA-1
project *Material Code Harmonisation Across Public Enterprises* (IOCL, OIL, NTPC).

- **Compare two materials**: paste two descriptions and get `EQUIVALENT`, `NOT_EQUIVALENT` or `NEEDS_REVIEW`, with
  the extracted specifications and the evidence behind the decision.
- **Search catalogue**: find the closest of 6,634 public procurement records for a new description.
- **Harmonised clusters**: browse the proposed harmonised material codes.
- **Model card**: all metrics, read from `results/metrics/er_model_metrics.json`.

How the model was trained: a gradient-boosting classifier over MiniLM, TF-IDF, fuzzy and unit-aware specification
features, trained on self-supervised pairs whose labels are known by construction (a format-only rewrite of a real
record is a match; a single-specification change is a non-match). Held-out scores are an upper bound on real-world
accuracy, and the small review set is AI-assisted, not independent human ground truth.

The app is read-only: inputs are scored in memory and are not stored.
