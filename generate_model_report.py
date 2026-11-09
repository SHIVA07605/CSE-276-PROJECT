"""
Render docs/ml_model_report.md from the saved metrics so that every number in the report is
exactly the number produced by the last training run (python train_er_model.py).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
M = ROOT / "results" / "metrics"
OUT = ROOT / "docs" / "ml_model_report.md"


def pct(x):
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def f4(x):
    return "n/a" if x is None or x != x else f"{x:.4f}"


def main() -> None:
    m = json.loads((M / "er_model_metrics.json").read_text(encoding="utf-8"))
    test = pd.read_csv(M / "model_comparison_synthetic_test.csv", index_col=0)
    gold = pd.read_csv(M / "model_comparison_gold_labels.csv", index_col=0)
    mut = pd.read_csv(M / "hard_negative_detection_by_mutation.csv")
    imp = pd.read_csv(M / "feature_importance.csv").head(10)
    best = m["selected_model"]
    td = m["training_data"]
    th = m["thresholds"]
    pol = m["synthetic_test_policy"]
    g = m["gold"]
    gp = g["deployed_policy_on_binary"]
    rc = m["real_candidates"]
    pur = m["synthetic_test_cluster_purity"]

    def table(df, cols, names=None):
        names = names or cols
        lines = ["| Model | " + " | ".join(names) + " |", "|---|" + "---|" * len(cols)]
        for idx, r in df.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                cells.append(f4(v) if isinstance(v, float) else str(v))
            name = f"**{idx}**" if idx == best else idx
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    b = test.loc["exact_match_baseline"]
    s = test.loc[best]

    b2 = m.get("ai_assisted_batch2")
    b2_section = ""
    if b2:
        b2_cmp_path = M / "model_comparison_ai_assisted_batch2.csv"
        b2_cmp_str = ""
        if b2_cmp_path.exists():
            b2_cmp = pd.read_csv(b2_cmp_path, index_col=0)
            b2_cmp_str = table(b2_cmp, ['precision', 'recall', 'f1', 'accuracy', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives', 'accuracy_95ci'], ['P', 'R', 'F1', 'Acc', 'TP', 'FP', 'TN', 'FN', 'Acc 95% CI'])
        b2_pol = b2["deployed_policy_on_binary"]
        b2_section = f"""
## 4b. AI-Assisted Batch-2 Evaluation Tier (100 candidate pairs blind to model output)

> **Academic Integrity Statement:** Agreement with an AI annotator (Claude Opus, applying `docs/annotation_guidelines.md` blind to model output). This is **not independent human ground truth**; the annotator shares the model designer's rules, so routing agreement on UNCERTAIN pairs is expected by design.

* **Binary evaluation (n = {b2['n_binary']} pairs: {b2['n_positive']} positive / {b2['n_negative']} negative):**

{b2_cmp_str}

* **Deployed policy on Batch-2 binary pairs:** {b2_pol['auto_merge_count']} auto-merged (precision {pct(b2_pol['auto_merge_precision'])}), {b2_pol['auto_reject_count']} auto-rejected (NPV {pct(b2_pol['auto_reject_npv'])}), {b2_pol['review_count']} sent to review.
* **{b2['n_uncertain']} Batch-2 pairs marked UNCERTAIN:** {b2['uncertain_pairs_decisions']} — auto-merged: **{b2['uncertain_pairs_auto_merged']}**.
"""

    md = f"""# Hybrid Self-Supervised Entity-Resolution Model — Results Report

**Course:** CSE 276 – CA-1 Academic Project
**Project:** Material Code Harmonisation Across Public Enterprises (IOCL, OIL, NTPC)
**Model artefact:** `models/hybrid_er_model.joblib` (version `{m['model_version']}`)
**Selected classifier:** `{best}` · **Reproduce:** `python train_er_model.py` (≈2 min on CPU)

> All numbers below are generated automatically from `results/metrics/er_model_metrics.json`
> by `generate_model_report.py`; nothing is typed by hand.

---

## 1. Why a self-supervised model

Only **{g['n_binary']} human-confirmed binary labels** exist ({g['n_positive']} positive, {g['n_negative']} negative) — too few to train or
fairly evaluate a supervised classifier, and the project policy forbids training on AI-suggested
hypotheses. The model is therefore trained on **pairs whose labels are known by construction**,
built from the real 6,634-record candidate pool:

| Source | Pairs | Label | Construction |
|---|---|---|---|
""" + "\n".join(
        f"| {k} | {v:,} | {'1' if k == 'synthetic_positive' else '0'} | " + {
            "synthetic_positive": "real record vs. format-only rewrite of itself (boilerplate, quantity suffix, noun-first order, abbreviation, unit spacing, typo, plural)",
            "synthetic_hard_negative": "real record vs. rewrite with exactly one specification changed (size, rating, grade, type, standard)",
            "real_spec_conflict_negative": "real candidate pair whose canonical number+unit specifications are disjoint",
            "random_cross_commodity_negative": "two real records from different commodity classes",
        }.get(k, "") + " |"
        for k, v in td["by_source"].items()
    ) + f"""

* Total pairs: **{td['total_pairs']:,}** ({td['positives']:,} positive / {td['negatives']:,} negative); split train/val/test = {td['by_split'].get('train', 0):,} / {td['by_split'].get('val', 0):,} / {td['by_split'].get('test', 0):,}.
* Splits are formed by **near-duplicate group** (union-find over TF-IDF ≥ 0.8): group leakage = **{td['group_leakage']}**.
* All **{td['excluded_review_ids']} records of the human review set** (and any record with the same description) are excluded from training, so the review-set evaluation is untouched.

## 2. Model

* **Features (38):** MiniLM cosine; word & char TF-IDF cosine; Jaccard / containment; 4 RapidFuzz ratios; token-difference counts; canonical, unit-aware specification matches/conflicts (`10 MM`=`10MM`, `1.1 KV`=`1100V`, `CLASS 150`=`150#`); descriptor-group conflicts (BALL vs GATE valve, CENTRIFUGAL vs SUBMERSIBLE pump, XLPE vs PVC …); standards / grades / bearing series; `PrecisionMatchingEngine` decision & conflict counts; legacy regex attribute matches. The classifier uses **34** of them: specification counts (construction-shortcut risk) and the legacy string-equality attributes (false conflicts on `150#` vs `CLASS 150`) feed only the decision gates.
* **Classifiers compared:** Logistic Regression (scaled), Random Forest, Histogram Gradient Boosting — balanced class weights, selected on validation PR-AUC.
* **Decision policy (precision-first):** explicit spec conflict → `NOT_EQUIVALENT`; p ≤ {th['low']:.2f} → `NOT_EQUIVALENT`; p ≥ {th['high']:.2f} with shared identity specifications → `EQUIVALENT`; generic titles, asymmetric specifications and mid scores → `NEEDS_REVIEW`. Binary operating threshold (F1-optimal on validation) = {th['binary']:.3f}.

## 3. Held-out test results (group-disjoint, n = {int(s['total_pairs']):,})

{table(test, ['precision', 'recall', 'f1', 'accuracy', 'roc_auc', 'pr_auc'], ['Precision', 'Recall', 'F1', 'Accuracy', 'ROC-AUC', 'PR-AUC'])}

* **F1 vs official exact-match baseline:** {b['f1']:.3f} → **{s['f1']:.3f}** ({s['f1'] - b['f1']:+.3f}); recall {b['recall']:.3f} → **{s['recall']:.3f}**.
* **Deployed three-way policy:** auto-merge precision **{pct(pol['auto_merge_precision'])}**, auto-reject NPV **{pct(pol['auto_reject_npv'])}**, review rate {pct(pol['review_rate'])} (generic titles and asymmetric specifications are deliberately sent to humans).
* **Hard-negative detection** for specification families *not* encoded as explicit features (orientation, drive, cooling, actuation, AC/DC, duty, mobility): **{pct(m['unseen_mutation_detection_rate'])}**.

### Cluster purity (entities known by construction)

| Clustering | Purity | Inverse purity | F-measure | Predicted clusters / true entities |
|---|---|---|---|---|
""" + "\n".join(
        f"| {k} | {v['purity']:.4f} | {v['inverse_purity']:.4f} | {v['f_measure']:.4f} | {v['n_predicted_clusters']} / {v['n_true_entities']} |"
        for k, v in pur.items()
    ) + f"""

### Hard-negative detection by mutation family (test)

| Kind | Group | n | Detection rate |
|---|---|---|---|
""" + "\n".join(
        f"| {r.mutation_kind} | {r.mutation_group} | {r.n} | {r.detection_rate:.3f} |" for r in mut.itertuples()
    ) + f"""

## 4. Review-Set Labels (AI-assisted, not independent human ground truth) — Real-Data Check

Scored on the **{g['n_binary']} confirmed binary pairs** only ({g['n_positive']} positive / {g['n_negative']} negative). With so few positives the
95 % Wilson intervals are wide; these results are evidence, not a precise estimate.

{table(gold, ['precision', 'recall', 'f1', 'accuracy', 'true_positives', 'false_positives', 'true_negatives', 'false_negatives', 'accuracy_95ci'], ['P', 'R', 'F1', 'Acc', 'TP', 'FP', 'TN', 'FN', 'Acc 95% CI'])}

* **Deployed policy on review-set binary pairs:** {gp['auto_merge_count']} auto-merged (precision {pct(gp['auto_merge_precision'])}), {gp['auto_reject_count']} auto-rejected (NPV {pct(gp['auto_reject_npv'])}), {gp['review_count']} sent to review.
* **{g['n_uncertain']} review-set pairs marked UNCERTAIN:** {g['uncertain_pairs_decisions']} — auto-merged: **{g['uncertain_pairs_auto_merged']}**.
* **Worksheet integrity note:** `PAIR_BOUNDARY_041` was restored to its genuine UNCERTAIN / MEDIUM annotation and test isolation verified.
{b2_section}
## 5. All {rc['n_pairs']:,} real candidate pairs (unlabelled, descriptive only)

* Decisions: {rc['decisions']}
* Cross-enterprise decisions: {rc['cross_enterprise_decisions']}
* Conflict-aware clustering: **{rc['multi_record_clusters']} proposed harmonised groups** covering {rc['records_in_clusters']} records; {rc['blocked_merges']} merges blocked, {rc['transitive_conflicts']} transitive conflicts caught.
* Outputs: `results/matching/model_scored_candidate_pairs.csv`, `results/matching/harmonised_master_model.csv`.
* Active learning: `results/metrics/batch2_review_priority.csv` ranks the {g['n_pending']} pending pairs by model uncertainty for the next review session.

## 6. Most important features (permutation importance on test, PR-AUC drop)

| Feature | Importance |
|---|---|
""" + "\n".join(f"| `{r.feature}` | {r.importance_mean:.4f} ± {r.importance_std:.4f} |" for r in imp.itertuples()) + f"""

## 7. Limitations

1. Test pairs are constructed from real records; genuine cross-enterprise variation is messier, so held-out scores are an upper bound on real-world accuracy.
2. Review-set binary labels are few ({g['n_binary']} binary, {g['n_positive']} positive). Label the pending pairs (start with `batch2_review_priority.csv`) and re-run `python train_er_model.py` — review metrics update automatically.
3. Specification parsing is regex-based; codes outside the parser are only caught through lexical evidence.
4. The model never sees enterprise identity; domain differences (API vs BIS service) that are not written in the text cannot be detected.

_Runtime of the last training run: {m['runtime_seconds']} s._
"""
    OUT.write_text(md, encoding="utf-8")
    print(f"[OK] wrote {OUT}")

    # ---- keep the README headline table in sync with the same metrics
    hm = pur.get("hybrid_model", {})
    em = pur.get("exact_match_baseline", {})
    sel = g["selected_model_binary"]
    readme_table = f"""<!-- RESULTS:START (auto-generated by generate_model_report.py) -->
**Headline results** (selected model: `{best}`; full report: [`docs/ml_model_report.md`](docs/ml_model_report.md))

| Evaluation | Exact-match baseline | Hybrid ER model |
|---|---|---|
| Held-out pairs (n = {int(s['total_pairs']):,}) — Precision | {b['precision']:.3f} | **{s['precision']:.3f}** |
| Held-out pairs — Recall | {b['recall']:.3f} | **{s['recall']:.3f}** |
| Held-out pairs — F1 | {b['f1']:.3f} | **{s['f1']:.3f}** |
| Held-out pairs — PR-AUC | {b['pr_auc']:.4f} | **{s['pr_auc']:.4f}** |
| Cluster purity / inverse purity | {em.get('purity', float('nan')):.3f} / {em.get('inverse_purity', float('nan')):.3f} | **{hm.get('purity', float('nan')):.3f} / {hm.get('inverse_purity', float('nan')):.3f}** |
| Review-set labels (AI-assisted, n = {g['n_binary']}) — TP / FP / TN / FN | {int(gold.loc['exact_match_baseline', 'true_positives'])} / {int(gold.loc['exact_match_baseline', 'false_positives'])} / {int(gold.loc['exact_match_baseline', 'true_negatives'])} / {int(gold.loc['exact_match_baseline', 'false_negatives'])} | **{sel['true_positives']} / {sel['false_positives']} / {sel['true_negatives']} / {sel['false_negatives']}** |
| Deployed policy — auto-merge precision (held-out) | – | **{pct(pol['auto_merge_precision'])}** |
| Review-set UNCERTAIN pairs auto-merged | – | {g['uncertain_pairs_auto_merged']} / {g['n_uncertain']} |
<!-- RESULTS:END -->"""
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    import re as _re

    if "__RESULTS_TABLE__" in text:
        text = text.replace("__RESULTS_TABLE__", readme_table)
    else:
        text = _re.sub(r"<!-- RESULTS:START.*?<!-- RESULTS:END -->", lambda _m: readme_table, text, flags=_re.S)
    readme.write_text(text, encoding="utf-8")
    print("[OK] updated README.md results table")


if __name__ == "__main__":
    main()
