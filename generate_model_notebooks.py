"""
Build and EXECUTE notebooks 05 (model training) and 07 (model evaluation) for the hybrid
self-supervised entity-resolution model. Every output in the notebooks is produced by real
execution through nbclient (no hand-written outputs).

Usage:
    python generate_model_notebooks.py            # build + execute both
    python generate_model_notebooks.py --no-exec  # build only
"""

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parent
NB_DIR = PROJECT_ROOT / "notebooks"

SETUP = r'''
import sys, json, os, subprocess, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# Colab environment setup
if "google.colab" in sys.modules:
    colab_repo = Path("/content/CSE-276-PROJECT")
    if not colab_repo.is_dir():
        subprocess.run(["git", "clone", "https://github.com/SHIVA07605/CSE-276-PROJECT.git", str(colab_repo)], check=True)
    os.chdir(str(colab_repo))
    req_file = colab_repo / "requirements.txt"
    if req_file.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz", "sentence-transformers"], check=True)

# Locate project root whether the kernel starts in notebooks/ or in the repository root
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "src").is_dir() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 200)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("Project root:", PROJECT_ROOT)
'''


def nb05() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(r"""# 05. Pairwise Entity-Resolution Model — Hybrid Self-Supervised Classifier

**Course:** CSE 276 – CA-1 Academic Project
**Project:** Material Code Harmonisation Across Public Enterprises (IOCL, OIL, NTPC)
**Status:** ✅ Model trained, evaluated and serialised (`models/hybrid_er_model.joblib`)

---

### The problem this notebook solves
Earlier versions of this notebook kept supervised training *locked*: only **20 human-confirmed binary labels** exist (2 positives, 18 negatives) — far too few to fit, let alone evaluate, a classifier. Training on the AI-suggested hypotheses would violate the project's integrity policy.

### The solution: self-supervised labels *by construction*
Following self-supervised entity-matching practice (data augmentation in *Ditto*, Li et al. 2020; contrastive pre-training in *Sudowoodo*, Wang et al. 2023), we generate training pairs **from the real 6,634-record candidate pool** whose labels are known with certainty because of *how they were made* — not because a human or an AI judged them:

| Pair type | How it is built | Label | Why the label is certain |
|---|---|---|---|
| **Positive** | real record vs. a *format-only rewrite* of itself (boilerplate, quantity suffix, SAP noun-first order, abbreviations, unit spacing, typo, plural) | 1 | the physical item is unchanged |
| **Hard negative** | real record vs. a rewrite with **exactly one specification changed** (250MM→300MM, SS304→SS316, CLASS 150→300, BALL→GATE valve, API 6D→6A …) | 0 | different specification ⇒ not functionally equivalent (project definition) |
| **Real spec-conflict negative** | real candidate pair whose canonical number+unit specs are disjoint (e.g. 1400MM vs 1800MM belts) | 0 | explicit, parsed specification conflict |
| **Random cross-commodity negative** | two real records from different commodity classes (pump vs cable) | 0 | different commodity |

**Integrity guarantees:** (1) no human or AI label is used for fitting; (2) all 173 records of the human review set — and any record sharing their description text — are excluded from training, so the review-set evaluation is untouched; (3) train/val/test are split by *near-duplicate group* (union-find over TF-IDF ≥ 0.8), so no record or near-copy appears on both sides.
"""),
        new_code_cell(SETUP),
        new_markdown_cell(r"""---
## Section A — Run the end-to-end training pipeline
`src/er_pipeline.run_training()` executes every stage (data → pairs → MiniLM embeddings → 38 hybrid features → 3 classifiers → threshold selection → evaluation → scoring of all 22,464 real candidate pairs → conflict-aware clustering) and saves every artefact. The same function is used by `python train_er_model.py`, so the numbers in this notebook and on disk are identical."""),
        new_code_cell(r'''from src.er_pipeline import run_training
R = run_training(verbose=True, save=True)'''),
        new_markdown_cell(r"""---
## Section B — The self-supervised training corpus"""),
        new_code_cell(r'''pairs = R["pairs"]
summary = (pairs.groupby(["source", "label"]).size().unstack(fill_value=0)
           .rename(columns={0: "negatives (0)", 1: "positives (1)"}))
display(summary)
display(pairs.groupby("split")["label"].agg(pairs="size", positives="sum").assign(negatives=lambda d: d.pairs - d.positives))
print("Group leakage between train and val/test:", R["metrics"]["training_data"]["group_leakage"], "groups")
print("Excluded human-review record IDs       :", R["metrics"]["training_data"]["excluded_review_ids"])'''),
        new_code_cell(r'''# Concrete examples of each pair type (raw strings as generated, before normalisation)
cols = ["source", "label", "raw_a", "raw_b", "aug_ops_b", "mutation_group", "mutation_old", "mutation_new"]
for src in ["synthetic_positive", "synthetic_hard_negative", "real_spec_conflict_negative", "random_cross_commodity_negative"]:
    print(f"\n### {src}")
    display(pairs.loc[pairs.source == src, cols].sample(4, random_state=7))'''),
        new_markdown_cell(r"""---
## Section C — Hybrid feature vector (38 features)

| Family | Features | Purpose |
|---|---|---|
| Semantic | `sem_cosine`, `sem_minus_lex` | MiniLM meaning; synonym / phrasing-inversion gap |
| Lexical | word & char TF-IDF cosine, Jaccard, containment, 4 fuzzy ratios, lengths, exact match | surface overlap, robust to typos & reordering |
| Token differences | `alpha_symdiff`, `alpha_unmatched_fuzzy`, bare-number Jaccard / symdiff | detects a changed descriptor or number |
| Canonical specifications | unit-spec matches / **conflicts** / partial / one-side; descriptor-group matches / **conflicts** / one-side; standards, grades, bearing series; identity-spec counts | unit-aware: `10 MM`=`10MM`, `1.1 KV`=`1100V`, `CLASS 150`=`150#` |
| Project rule signals | `PrecisionMatchingEngine` decision & conflict counts; legacy regex attribute matches / conflicts | reuses Notebook 04–06 domain rules |

**The classifier is trained on 34 of the 38 features.** Four are computed but used *only* by the decision-policy safety gates:
* `identity_spec_min/max` — only specification-rich records can be mutated into hard negatives, so raw spec counts would let the classifier learn a *construction shortcut* ("many specs ⇒ non-match"). Spec-rich seeds also receive a second positive for the same reason.
* `legacy_attr_matches/conflicts` — the original string-equality extractor reports false conflicts on equivalent formats (`150#` vs `CLASS 150`, `101.60MM` vs `101.6MM`); the canonical unit-aware features supersede it.

No IDs, enterprise names, tender numbers or labels are used (leakage guard)."""),
        new_code_cell(r'''X_all, y_all = R["X_all"], R["y_all"]
key = ["sem_cosine", "tfidf_word_cosine", "fuzz_token_set", "alpha_unmatched_fuzzy", "unit_spec_conflicts",
       "group_conflicts", "standard_conflicts", "identity_spec_max", "engine_not_equivalent"]
display(X_all[key].groupby(y_all).mean().rename(index={0: "mean | label 0", 1: "mean | label 1"}).T.round(3))
print("Feature matrix:", X_all.shape, "| all finite:", bool(np.isfinite(X_all.values).all()))'''),
        new_code_cell(r'''# The semantic-masking problem from Notebook 04, visible in the training data:
# hard negatives are almost as similar as positives in MiniLM space -> embeddings alone cannot decide.
src = R["pairs"]["source"].values
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for s, c in [("synthetic_positive", "#2a9d8f"), ("synthetic_hard_negative", "#e76f51"), ("random_cross_commodity_negative", "#8d99ae")]:
    ax[0].hist(X_all.loc[src == s, "sem_cosine"], bins=40, alpha=0.6, label=s.replace("_", " "), color=c)
ax[0].set_title("MiniLM cosine by pair type"); ax[0].set_xlabel("cosine similarity"); ax[0].legend(fontsize=7)
hn = src == "synthetic_hard_negative"
ax[1].scatter(X_all.loc[hn, "sem_cosine"], X_all.loc[hn, "unit_spec_conflicts"] + X_all.loc[hn, "group_conflicts"] + np.random.RandomState(0).uniform(-.1, .1, hn.sum()), s=6, alpha=.4, color="#e76f51")
ax[1].set_title("Hard negatives: high cosine, but explicit conflicts"); ax[1].set_xlabel("cosine similarity"); ax[1].set_ylabel("unit + descriptor conflicts")
plt.tight_layout(); plt.savefig(FIG_DIR / "05_semantic_masking_training_pairs.png", bbox_inches="tight"); plt.show()'''),
        new_markdown_cell(r"""---
## Section D — Model training & selection (validation split)
Three classifiers are trained on the **train** split with balanced class weights and compared on the **validation** split; the model with the highest validation PR-AUC is selected. The test split is not touched here."""),
        new_code_cell(r'''display(R["val_table"][["precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "train_seconds"]].round(4))
print("Selected model:", R["best_name"])
print("Decision thresholds (chosen on validation only):", {k: round(v, 4) for k, v in R["hybrid"].thresholds.items()})'''),
        new_markdown_cell(r"""---
## Section E — Held-out test split: models vs baselines
The **exact string match** baseline (the official project baseline) and three further baselines — TF-IDF threshold, MiniLM threshold and the rule-based `PrecisionMatchingEngine` — are evaluated on the *same* group-disjoint test pairs. Baseline thresholds are tuned on validation, exactly like the models."""),
        new_code_cell(r'''tt = R["test_table"][["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc", "true_positives", "false_positives", "true_negatives", "false_negatives"]]
display(tt.round(4))

from sklearn.metrics import precision_recall_curve
te = R["masks"]["test"]; y_te = y_all[te]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for name, s in R["test_scores"].items():
    p, r, _ = precision_recall_curve(y_te, s)
    ax.plot(r, p, label=f"{name} (AP={tt.loc[name, 'pr_auc']:.3f})", lw=2 if name == R["best_name"] else 1.2)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall on held-out test pairs"); ax.legend(fontsize=7, loc="lower left")
plt.tight_layout(); plt.savefig(FIG_DIR / "05_precision_recall_test.png", bbox_inches="tight"); plt.show()'''),
        new_markdown_cell(r"""---
## Section F — Deployed decision policy (precision-first, three-way)
The classifier probability is wrapped in safety gates that encode the project's annotation policy:
1. explicit canonical specification conflict → **NOT_EQUIVALENT**
2. p ≤ `t_low` → **NOT_EQUIVALENT**
3. p ≥ `t_high` **and** both records carry identity specifications (size / rating / standard / grade) that are not stated on one side only → **EQUIVALENT**
4. everything else (generic titles such as `AIR COMPRESSOR`, asymmetric specs, mid-range scores) → **NEEDS_REVIEW**"""),
        new_code_cell(r'''from sklearn.metrics import ConfusionMatrixDisplay
print("Deployed-policy outcome on held-out test pairs:")
display(pd.Series(R["test_policy"]).to_frame("value"))
pred = (R["test_scores"][R["best_name"]] >= R["hybrid"].thresholds["binary"]).astype(int)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["non-match", "match"], ax=ax[0], colorbar=False, cmap="Blues")
ax[0].set_title(f"{R['best_name']} (binary threshold)")
ct = pd.crosstab(pd.Series(y_te, name="true label"), R["test_decisions"]["decision"].values)
ct.plot.bar(ax=ax[1], color={"EQUIVALENT": "#2a9d8f", "NEEDS_REVIEW": "#e9c46a", "NOT_EQUIVALENT": "#e76f51"})
ax[1].set_title("Three-way decisions by true label"); ax[1].set_xticklabels(["non-match (0)", "match (1)"], rotation=0)
plt.tight_layout(); plt.savefig(FIG_DIR / "05_confusion_and_policy_test.png", bbox_inches="tight"); plt.show()'''),
        new_markdown_cell(r"""---
## Section G — Error analysis
**Which specification changes does the model catch?** Mutation groups marked *unseen* (orientation, drive, cooling, mobility, actuation, AC/DC, duty) are deliberately **not** encoded as explicit features — detecting them measures generalisation from lexical evidence alone."""),
        new_code_cell(r'''display(R["mutation_table"].round(3))
print(f"Detection rate on mutation groups unseen by explicit features: {R['metrics']['unseen_mutation_detection_rate']:.3f}")
print("\nPositive recall by augmentation type (robustness to formatting noise):")
display(R["augmentation_table"].round(3))'''),
        new_code_cell(r'''# Misclassified test pairs (binary threshold) — the honest failure cases
tp = R["pairs"][te].copy(); tp["p_match"] = R["test_scores"][R["best_name"]]
tp["pred"] = (tp.p_match >= R["hybrid"].thresholds["binary"]).astype(int)
err = tp[tp.pred != tp.label]
print(f"Errors: {len(err)} of {len(tp)} test pairs")
display(err[["source", "label", "p_match", "text_a", "text_b", "aug_ops_b", "mutation_group"]].sort_values("p_match").head(15))'''),
        new_markdown_cell(r"""---
## Section H — What the model relies on (permutation importance, PR-AUC drop on test)"""),
        new_code_cell(r'''imp = R["importance"].head(15)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(imp.feature[::-1], imp.importance_mean[::-1], xerr=imp.importance_std[::-1], color="#457b9d")
ax.set_xlabel("mean decrease in average precision"); ax.set_title(f"Permutation importance — {R['best_name']}")
plt.tight_layout(); plt.savefig(FIG_DIR / "05_feature_importance.png", bbox_inches="tight"); plt.show()'''),
        new_markdown_cell(r"""---
## Section I — Load the saved model and score new description pairs"""),
        new_code_cell(r'''from src.er_pipeline import load_model, match_descriptions, MODEL_PATH
model = load_model()
print("Loaded:", MODEL_PATH.name, "| version:", model.metadata.get("model_name"), model.VERSION)
demo = [
    ("CENTRIFUGAL PUMP API 610 60 M3/HR", "PUMP-CENTRIFUGAL, API 610, 60 M3/HR"),
    ("CONVEYOR BELT 1200MM WIDTH NN/EP", "CONVEYOR BELT 1600MM WIDTH NN/EP"),
    ("GATE VALVE SS304 2INCH CLASS 150 FLANGED", "GATE VALVE SS316 2INCH CLASS 150 FLANGED"),
    ("BALL VALVE 2 INCH 150# API 6D", "Supply of API 6D Ball Valves 2\" Class 150 (10 Nos)"),
    ("XLPE CABLE 3C X 400 SQMM 1.1KV ARMOURED", "XLPE CABLE 4C X 400 SQMM 1.1KV ARMOURED"),
    ("AIR COMPRESSOR", "AIR COMPRESSOR"),
    ("DIESEL DRIVEN FIRE WATER PUMP", "ELECTRIC DRIVEN FIRE WATER PUMP"),
]
rows = []
for a, b in demo:
    d = match_descriptions(a, b, model=model, lookup=R["lookup"])
    rows.append({"description A": a, "description B": b, "P(match)": d["match_probability"], "decision": d["decision"], "reason": d["reason"]})
display(pd.DataFrame(rows))'''),
        new_markdown_cell(r"""---
## Summary
* A real, trained entity-resolution model now exists: `models/hybrid_er_model.joblib` (classifier + TF-IDF models + thresholds + metadata).
* It was trained **without** any human or AI label — labels come from how the pairs were constructed — and evaluated on a group-disjoint held-out split.
* On held-out pairs it far exceeds the exact-match baseline and every single-signal baseline (see Section E); the deployed three-way policy trades recall for precision by routing generic or asymmetric descriptions to human review.
* Notebook 07 evaluates the model against the **review-set labels (AI-assisted, not independent human ground truth)**, reports cluster purity, and analyses the 22,464 real candidate pairs."""),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def nb07() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(r"""# 07. Model Evaluation — Baseline Comparison, Review-Set Labels, Cluster Purity & Error Analysis

**Course:** CSE 276 – CA-1 Academic Project
**Project:** Material Code Harmonisation Across Public Enterprises (IOCL, OIL, NTPC)
**Model under evaluation:** `models/hybrid_er_model.joblib` (trained in Notebook 05)

> **Evaluation integrity.** Three evidence levels are reported separately and never mixed:
> **(1) Held-out synthetic test** — labels known by construction, group-disjoint from training (large *n*, measures the core skill).
> **(2) Review-set labels (AI-assisted, not independent human ground truth)** — the 19 confirmed binary labels from the review app (real data, small *n*; 95 % Wilson intervals reported).
> **(3) Real candidate pairs** — 22,464 unlabelled pairs: descriptive statistics only, no accuracy claims.
> This notebook only reads artefacts written by Notebook 05 / `train_er_model.py`; it re-trains nothing."""),
        new_code_cell(SETUP + r'''
from src.er_pipeline import (METRICS_JSON_PATH, COMPARISON_CSV_PATH, GOLD_COMPARISON_CSV_PATH, GOLD_PREDICTIONS_PATH,
                             REVIEW_PRIORITY_PATH, SCORED_CANDIDATES_PATH, HARMONISED_MASTER_PATH, MUTATION_CSV_PATH)
metrics = json.loads(METRICS_JSON_PATH.read_text(encoding="utf-8"))
test_cmp = pd.read_csv(COMPARISON_CSV_PATH, index_col=0)
gold_cmp = pd.read_csv(GOLD_COMPARISON_CSV_PATH, index_col=0)
gold_pred = pd.read_csv(GOLD_PREDICTIONS_PATH, dtype={"human_label": str})
scored = pd.read_csv(SCORED_CANDIDATES_PATH)
master = pd.read_csv(HARMONISED_MASTER_PATH)
BEST = metrics["selected_model"]
print("Selected model:", BEST, "| thresholds:", {k: round(v, 3) for k, v in metrics["thresholds"].items()})'''),
        new_markdown_cell(r"""---
## 1. Pairwise precision / recall / F1 vs the exact-match baseline (held-out test)"""),
        new_code_cell(r'''cols = ["precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"]
display(test_cmp[cols].round(4))
b, m = test_cmp.loc["exact_match_baseline"], test_cmp.loc[BEST]
print(f"F1 improvement over exact-match baseline: {m.f1 - b.f1:+.4f}  ({b.f1:.3f} -> {m.f1:.3f})")
print(f"Recall improvement                      : {m.recall - b.recall:+.4f}  ({b.recall:.3f} -> {m.recall:.3f})")
ax = test_cmp[["precision", "recall", "f1"]].plot.bar(figsize=(9, 3.8), color=["#457b9d", "#e9c46a", "#2a9d8f"], width=0.8)
ax.set_ylim(0, 1.05); ax.set_title("Held-out test: models vs baselines"); ax.set_xticklabels(test_cmp.index, rotation=25, ha="right")
plt.tight_layout(); plt.savefig(FIG_DIR / "07_model_vs_baselines_test.png", bbox_inches="tight"); plt.show()'''),
        new_markdown_cell(r"""---
## 2. Review-set labels (AI-assisted, not independent human ground truth) — real-data validation
Only pairs whose `human_label` is `0` or `1` are scored; `UNCERTAIN` and `PENDING` pairs are analysed separately. With *n* = 19 (2 positives) the intervals are wide — this is reported, not hidden. As more pairs are labelled in `review_app.py`, re-running `python train_er_model.py` updates every number automatically."""),
        new_code_cell(r'''g = metrics["gold"]
print(f"Review set: {g['n_review_pairs']} pairs | binary review: {g['n_binary']} (+{g['n_positive']} / -{g['n_negative']}) | uncertain: {g['n_uncertain']} | pending: {g['n_pending']}")
display(gold_cmp[["precision", "recall", "f1", "accuracy", "true_positives", "false_positives", "true_negatives", "false_negatives", "precision_95ci", "recall_95ci", "accuracy_95ci"]].round(3))
print("\nDeployed three-way policy on the binary review pairs:")
display(pd.Series(g["deployed_policy_on_binary"]).to_frame("value"))'''),
        new_code_cell(r'''# Every review decision side by side (binary + uncertain)
show = gold_pred[gold_pred.human_label.isin(["0", "1", "UNCERTAIN"])]
display(pd.crosstab(show.human_label, show.model_decision, margins=True))
print("\nBinary review pairs:")
display(show[show.human_label.isin(["0", "1"])][["pair_id", "human_label", "model_match_probability", "model_decision", "description_1", "description_2"]])'''),
        new_code_cell(r'''# How the model treats the 31 pairs marked UNCERTAIN (generic titles without specifications)
unc = gold_pred[gold_pred.human_label == "UNCERTAIN"]
print("Decisions on UNCERTAIN pairs:", unc.model_decision.value_counts().to_dict())
print("UNCERTAIN pairs auto-merged (the dangerous error):", (unc.model_decision == "EQUIVALENT").sum())
display(unc[unc.model_decision != "NEEDS_REVIEW"][["pair_id", "model_match_probability", "model_decision", "model_reason", "description_1", "description_2"]])'''),
        new_markdown_cell(r"""---
## 2b. AI-assisted Batch-2 evaluation tier (100 pairs blind to model output)
> **Academic Note:** This tier measures agreement with an AI annotator (Claude Opus, applying `docs/annotation_guidelines.md` blind to model output). It is **not independent human ground truth**. The annotator shares the model designer's rules, so routing agreement on UNCERTAIN pairs is expected by design.

Out of 100 candidate pairs:
- **25 binary pairs** (8 positive, 17 negative)
- **75 UNCERTAIN pairs** (generic descriptors, asymmetric specs, lots of mixed items)"""),
        new_code_cell(r'''b2 = metrics.get("ai_assisted_batch2")
if b2:
    print(f"Batch-2 tier: {b2['n_pairs']} pairs | binary: {b2['n_binary']} (+{b2['n_positive']} / -{b2['n_negative']}) | uncertain: {b2['n_uncertain']}")
    b2_cmp_path = METRICS_JSON_PATH.parent / "model_comparison_ai_assisted_batch2.csv"
    if b2_cmp_path.exists():
        b2_cmp = pd.read_csv(b2_cmp_path, index_col=0)
        display(b2_cmp[["precision", "recall", "f1", "accuracy", "true_positives", "false_positives", "true_negatives", "false_negatives", "precision_95ci", "recall_95ci", "accuracy_95ci"]].round(3))
    print("\nDeployed three-way policy on the Batch-2 binary pairs:")
    display(pd.Series(b2["deployed_policy_on_binary"]).to_frame("value"))
    print("\nDecisions on the 75 UNCERTAIN pairs:", b2["uncertain_pairs_decisions"])
    print("UNCERTAIN pairs auto-merged (safety check):", b2["uncertain_pairs_auto_merged"])
else:
    print("AI-assisted Batch 2 tier not present in metrics.")'''),
        new_markdown_cell(r"""---
## 3. Cluster purity (entities known by construction)
Purity alone rewards putting every record in its own cluster, so **inverse purity** (completeness) and their harmonic mean (F-measure) are reported too. Clusters are formed with the project's `conflict_aware_clustering()`."""),
        new_code_cell(r'''pur = pd.DataFrame(metrics["synthetic_test_cluster_purity"]).T
display(pur)
ax = pur[["purity", "inverse_purity", "f_measure"]].plot.bar(figsize=(7, 3.4), color=["#457b9d", "#e9c46a", "#2a9d8f"])
ax.set_ylim(0, 1.05); ax.set_title("Cluster quality on held-out test entities"); ax.set_xticklabels(pur.index, rotation=0)
plt.tight_layout(); plt.savefig(FIG_DIR / "07_cluster_purity.png", bbox_inches="tight"); plt.show()'''),
        new_markdown_cell(r"""---
## 4. All 22,464 real candidate pairs (unlabelled — descriptive only)"""),
        new_code_cell(r'''rc = metrics["real_candidates"]
print("Decisions (all pairs)      :", rc["decisions"])
print("Decisions (cross-enterprise):", rc["cross_enterprise_decisions"])
print(f"Harmonised clusters: {rc['multi_record_clusters']} multi-record groups covering {rc['records_in_clusters']} records | blocked merges: {rc['blocked_merges']} | transitive conflicts caught: {rc['transitive_conflicts']}")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(scored.model_match_probability, bins=50, color="#457b9d")
for k, c in [("low", "#e76f51"), ("high", "#2a9d8f")]:
    ax[0].axvline(metrics["thresholds"][k], color=c, ls="--", label=f"t_{k}={metrics['thresholds'][k]:.2f}")
ax[0].set_yscale("log"); ax[0].set_title("Match probability on real candidate pairs"); ax[0].legend()
scored.model_decision.value_counts().plot.bar(ax=ax[1], color=["#e76f51", "#e9c46a", "#2a9d8f"])
ax[1].set_title("Model decisions on real candidate pairs"); ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(FIG_DIR / "07_real_candidate_decisions.png", bbox_inches="tight"); plt.show()'''),
        new_code_cell(r'''print("Cross-enterprise pairs proposed as EQUIVALENT:")
x = scored[(scored.model_decision == "EQUIVALENT") & scored.is_cross_enterprise.astype(bool)]
display(x[["model_match_probability", "organization_1", "cleaned_description_1", "organization_2", "cleaned_description_2"]].sort_values("model_match_probability", ascending=False).head(15))
print("\nHarmonised master (top proposed groups):")
display(master.head(12)[["harmonised_code", "member_count", "enterprises", "representative_description", "mean_match_probability"]])'''),
        new_markdown_cell(r"""---
## 5. Active learning — which pending pairs should humans label next?
The 100 `PENDING` Batch-2 pairs are ranked by model uncertainty (probability closest to 0.5 first). Labelling these first gives the most information per reviewer minute (uncertainty sampling)."""),
        new_code_cell(r'''prio = pd.read_csv(REVIEW_PRIORITY_PATH)
print("Pending-pair decisions:", metrics["gold"]["pending_pairs_decisions"])
display(prio.head(12)[["review_rank", "pair_id", "model_match_probability", "model_decision", "description_1", "description_2"]])'''),
        new_markdown_cell(r"""---
## 6. Conclusions & limitations
* **Objective 2 (ER model with pairwise P/R):** delivered — a trained, serialised hybrid classifier with precision, recall, F1, ROC-AUC and PR-AUC on held-out pairs, plus confusion matrices on review-set labels.
* **Objective 3 (metrics vs exact-match baseline, cluster purity):** delivered — the model is compared with exact match and three further baselines on identical pairs; purity / inverse purity are reported.
* **Safety:** explicit specification conflicts are never merged, generic titles and asymmetric specifications are routed to review, and conflict-aware clustering blocks transitive merges.

**Limitations (stated honestly):**
1. Held-out test pairs are *constructed* from real records; real cross-enterprise variation can be messier, so test scores are an upper bound on real-world accuracy.
2. Review-set binary labels are few (19 binary, 2 positive): the review-set confusion matrix is evidence, not a precise estimate — see the Wilson intervals.
3. The genuine annotation for `PAIR_BOUNDARY_041` (UNCERTAIN / MEDIUM) has been restored and the test suite isolated to protect worksheet integrity.
4. Specification parsing is regex-based; unusual units or codes outside the parser are only caught through lexical evidence (see the unseen-mutation detection rate in Notebook 05)."""),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def execute(nb: nbformat.NotebookNode, path: Path) -> None:
    from nbclient import NotebookClient

    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
    client.execute()
    nbformat.write(nb, str(path))


if __name__ == "__main__":
    run = "--no-exec" not in sys.argv
    for builder, name in [(nb05, "05_pairwise_classification.ipynb"), (nb07, "07_model_evaluation.ipynb")]:
        path = NB_DIR / name
        nb = builder()
        if run:
            print(f"Executing {name} ...", flush=True)
            execute(nb, path)
        else:
            nbformat.write(nb, str(path))
        print(f"[OK] wrote {path}", flush=True)
