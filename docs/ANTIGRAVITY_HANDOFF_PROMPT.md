You are continuing work on a CSE 276 (Artificial Intelligence Foundations) CA-1 project in `C:\project ai` (Windows, Git Bash/PowerShell, Python 3.13 from the Microsoft Store). Read this whole prompt before acting. Work in the order of the task list. After each task, run the verification it names, and do not move on while it fails.

## Project facts (verified; use these exact values)
- Title: Material Code Harmonisation Across Public Enterprises. Inspired by SIH26099 (Ministry of Petroleum & Natural Gas), https://sih.gov.in/sih2026PS
- Official allocation (from `C:\Users\LENOVO\OneDrive\Pictures\CSE 276 Project\CA1 Guidelines\AI Foundation (CSE276)\K3P25LH_CSE276_CA1_Groups_and_Projects.xlsx`): Section **K3P25LH**, Group **G6**, course **CSE276 – Artificial Intelligence Foundations**. Syllabus topics: Unit V NLP; Unit VI clustering; Unit V evaluation of matching. Optional advanced extension: "Add active learning so a reviewer resolves only the ambiguous pairs."
- Objectives (copy unchanged): (1) Build a matching system that groups functionally equivalent material records and proposes one harmonised code with evidence. (2) Produce an entity resolution model with pairwise precision/recall over a labelled sample. (3) Report pairwise precision/recall/F1, cluster purity. Baseline: exact string match. Deliver Reviewer UI showing proposed merges with evidence.
- **Discrepancies to flag to the user. Do NOT silently change them:** the official list gives Sabuj Mondal's registration number as **12504433**, but the report draft says 12509118. The list spells **"Sindhant Dhadwal"**, but the report says "Dadwal".
- Dataset: Hugging Face `Prasenjeet25/sih26099-cpse-material-codes` (CC BY 4.0). 21,513 raw records; 6,634 in the candidate pool; 22,464 TF-IDF candidate pairs.
- GitHub repo: https://github.com/SHIVA07605/CSE-276-PROJECT. Its local clone is the nested folder `C:\project ai\CSE-276-PROJECT`, with only 2 commits (README + notebooks) and some uncommitted copies. The outer repo `C:\project ai` has NO remote. Do not add one, and do not push the outer history.

## What is already done (do not redo)
- **Trained model:** `models/hybrid_er_model.joblib`, a Histogram Gradient Boosting classifier over 34 of 38 hybrid features, wrapped in `HybridERModel` with a precision-first 3-way policy (EQUIVALENT / NOT_EQUIVALENT / NEEDS_REVIEW).
  - Training is **self-supervised with labels by construction**, in `src/synthetic_pairs.py`. A real record vs a format-only rewrite is a match. A real record vs a copy with exactly one specification changed is a non-match. There are also real spec-conflict negatives and random cross-commodity negatives.
  - The 173 review-set records are excluded from training. Splits are near-duplicate-group-disjoint, with 0 leakage.
- **Code:**
  - `src/pair_features.py`: canonical unit-aware spec parser and features. `MODEL_FEATURE_COLUMNS` withholds `identity_spec_*` (a construction-shortcut risk) and `legacy_attr_*` (false conflicts such as 150# vs CLASS 150).
  - `src/er_pipeline.py`: the full pipeline.
  - `src/model.py`: `HybridERModel`, plus the `hist_gradient_boosting` and `scaled_logistic_regression` options.
  - `train_er_model.py`: the CLI.
  - `generate_model_notebooks.py`: builds and executes notebooks 05 and 07 with nbclient.
  - `generate_model_report.py`: writes `docs/ml_model_report.md` and the README results table between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->`.
  - `tests/test_er_model.py`: 28 tests.
- **Results (last run, 6 Nov 23:49):** held-out F1 0.994 vs exact-match 0.165; auto-merge precision 99.5%; cluster purity / inverse purity 0.999 / 0.995 vs 0.999+ / 0.545 for exact match. **These numbers are STALE. See Task 1.**
- **Bug fixed:** `tests/test_review_app.py` wrote the test note "Different motor power and flange dimensions…" into the REAL `data/processed/combined_candidate_pairs_150.csv` on every pytest run. The test is now isolated and asserts the real files are byte-identical. `PAIR_BOUNDARY_041` has been restored to the genuine Nov-4 annotation (UNCERTAIN / MEDIUM / original note), and a backup is in `backups/review_file_before_label_restore_20261106/`. Current label counts: 1=2, 0=17, UNCERTAIN=31, PENDING=100.
- `pip` packages already installed: rapidfuzz, nbformat, nbclient, ipykernel, openpyxl.

## Hard rules (academic integrity, and the user's own project policy)
1. NEVER write to the `human_label`, `confidence_rating` or `reviewer_notes` columns of any worksheet. Only a human writes there, through `review_app.py`.
2. The 19 existing binary labels come from an **AI-assisted annotation workflow**. The team's report declares this. Call them "review-set labels (AI-assisted, not independent human ground truth)", NEVER "human gold labels".
3. Never treat `ai_suggested_hypothesis` as ground truth. Never fit any model on review-set labels.
4. Every number in the docs, README, report notes and slides must come from `results/metrics/er_model_metrics.json` after Task 1. Never type metrics by hand, and never copy numbers from this prompt into deliverables.
5. Held-out scores are computed on constructed pairs. Always state that they are an upper bound on real-world accuracy.
6. Do not modify the user's notebooks 01–04 and 06, `review_app.py` behaviour, or the legacy `generate_nb0*.py` scripts. The legacy scripts hardcode `c:\project ai`; leave them alone.
7. Do not force-push, do not rewrite remote history, and do not push `backups/` or `data/raw/`.

## Task list

### Task 1: Relabel the evaluation tier, add the AI-assisted Batch-2 tier, and re-run everything
**a) Replace "human gold label" wording.** In these files, replace "human gold label(s)" / "human-confirmed" / "Human gold labels" / "Human-UNCERTAIN" with the review-set wording from rule 2:
- `src/er_pipeline.py` (docstrings and comments only; keep the JSON key `gold` so nothing breaks)
- `generate_model_notebooks.py` (markdown text)
- `generate_model_report.py` (report text and the README table row labels)
- `docs/faculty_demo_guide.md` (Step 7 talking point and Q&A table)
- `docs/model_evaluation.md`, `docs/report_fact_sheet.md`, `docs/final_project_audit_report.md`, `docs/known_limitations.md`
- `README.md` text outside the auto-generated block

**b) Create `data/labelled/ai_assisted_batch2_labels.csv`.** Columns: `pair_id, ai_label, ai_confidence, ai_rationale, label_source, annotator, blind_to_model_output, review_date`.
- Set `label_source="AI_ASSISTED_PRE_ANNOTATION (not human ground truth)"`, `annotator="Claude (Anthropic), applying docs/annotation_guidelines.md"`, `blind_to_model_output=True`, `review_date=2026-11-06`.
- The labels below were decided blind to the model's outputs:

| ai_label | Pairs | Rationale |
|---|---|---|
| `1` (MEDIUM) | PAIR_EXP_028, 029, 034, 036, 037, 038, 039, 040 | API 6D ball valves differing only in quantity, plural or word order (Rule 6; same precedent as PAIR_TFIDF_HIGH_020) |
| `0` | PAIR_EXP_014, 015 | 2000 kVA 11/0.433 kV vs 500 kVA 11-3.3/0.415 kV transformer (Rule 1, HIGH) |
| `0` | PAIR_EXP_011 | Steam turbine vs EOI design-study service (Rule 7, HIGH) |
| `0` | PAIR_EXP_024, 025, 026 | Hiring of pumps (service) vs sealless pumps (Rule 7, HIGH) |
| `0` | PAIR_EXP_086, 087 | AMC service vs gas chromatograph instrument (Rule 7, HIGH) |
| `0` | PAIR_EXP_088, 089 | Pig launcher/receiver (pressure trap) vs electronic pig transmitter/receiver (HIGH) |
| `0` | PAIR_EXP_006 | Bale moisture analyzer vs lab moisture analyzer with inbuilt weighing (different principle, MEDIUM) |
| `0` | PAIR_EXP_012 | Steel conveyor belt vs fabric belt (different construction, MEDIUM) |
| `0` | PAIR_EXP_018, 019 | Slurry vs sewage submersible pumps (different service, MEDIUM) |
| `0` | PAIR_EXP_098, 099, 100 | Vaseline petroleum jelly vs Balmerol lubricating grease (MEDIUM) |
| `UNCERTAIN` | All other 75 PAIR_EXP_ pairs | Generic titles, generic-vs-specific asymmetry, tender clause fragments, bundles, and API 6D lots of "various sizes" / "03 items" (030, 035). Rules 4 and 5. |

**c) Extend `evaluate_review_set()` in `src/er_pipeline.py`.** If the CSV exists, score the deployed model and all baselines on its binary pairs as a SEPARATE tier called `ai_assisted_batch2`. Report:
- the confusion matrix with Wilson 95% CIs,
- the decision distribution on its UNCERTAIN pairs,
- the auto-merge count on UNCERTAIN pairs.

Add the tier to the metrics JSON, to notebook 07 (a new section in `generate_model_notebooks.py`) and to `generate_model_report.py`. State in every place that it is agreement with an AI annotator, that it is not independent ground truth, and that the annotator shares the model designer's rules, so routing agreement on UNCERTAIN pairs is expected by design.

**d) Re-run in this order:**
```
python train_er_model.py
python generate_model_notebooks.py
python generate_model_report.py
python -m pytest
```
- Verify that the review set now has **19 binary pairs (2 positive, 17 negative), 31 uncertain, 100 pending**.
- Verify notebooks 05 and 07 have 0 error outputs.
- Verify all tests pass.
- Then regenerate `results/report_evidence/test_results.txt` with the same header format as before.

**Notes:**
- `KeyError` lines from `joblib/loky resource_tracker` are harmless.
- Use `PYTHONIOENCODING=utf-8` when printing notebook text to the console.

### Task 2: Make training work out of the box for any agent (Gemini/Colab) from a fresh clone
**a) Add fallbacks to `load_project_data()`:**
- If `models/candidate_embeddings.npy` is missing, start with an empty lookup; `EmbeddingLookup.ensure()` already encodes missing texts.
- If `results/matching/semantic_candidate_pairs.csv` is missing, regenerate it with `src.candidate_generation.generate_tfidf_candidates()` (threshold 0.35, ngram (1,2), min_df 2), add `semantic_cosine_similarity` from embeddings, and save it.
- If `preprocessed_materials.csv` or `combined_candidate_pairs_150.csv` is missing, raise a clear error that names the file and how to obtain it.

**b) Add a preflight to `train_er_model.py`,** also runnable as `python train_er_model.py --check`. It checks the Python version, the required packages (with the pip command to fix), the required files, and that the MiniLM model can load. It prints `[PREFLIGHT OK]` or exact fixes.

**c) Make the notebook SETUP cell in `generate_model_notebooks.py` Colab-aware.** When `"google.colab" in sys.modules`:
- clone the GitHub repo to `/content/CSE-276-PROJECT` if absent,
- `os.chdir` into it,
- `pip install -q -r requirements.txt` (or at least rapidfuzz and sentence-transformers).

Keep the existing walk-up root finder for local runs.

**d) Guard against scikit-learn version mismatch.** Store `sklearn.__version__` in the model metadata. Make `load_model()` print "retrain with python train_er_model.py" if unpickling fails. Make `test_trained_artifact_loads_with_valid_thresholds` skip when the installed sklearn version differs from the stored one.

**e) Pin `requirements.txt`** with minimum versions that work: scikit-learn>=1.3 (HistGradientBoosting `class_weight` needs 1.2+), pandas>=2.0, sentence-transformers>=2.2.2, rapidfuzz>=3.0, nbformat, nbclient, ipykernel, openpyxl.

**f) Create `GEMINI.md` and `AGENTS.md` at the repo root.** Keep them short: the hard rules above; the one-command train (`python train_er_model.py`); the verify commands; expected artefacts; "do not use legacy generate_nb0*.py"; known harmless warnings.

**Verification:** copy the to-be-committed files (Task 3 list) to a scratch folder in `%TEMP%`, then run `python train_er_model.py --check`, `python train_er_model.py` and `python -m pytest -q` there.

### Task 3: Push to GitHub (only after Tasks 1–2 pass)
**a) Update the `.gitignore` used for the push.** Keep `data/raw/*` ignored (the README gives the HF download link). Un-ignore exactly the files needed to train and to reproduce the report:
- `data/processed/preprocessed_materials.csv`
- `data/processed/combined_candidate_pairs_150.csv`
- `data/processed/candidate_pairs_batch2_expansion_100.csv`
- `data/processed/improved_labelled_pairs_worksheet.csv`
- `data/labelled/*.csv`
- `models/hybrid_er_model.joblib`
- `models/candidate_embeddings.npy` and `models/embedding_id_mapping.csv`
- `results/metrics/*`
- `results/figures/*`
- `results/matching/*`

Also add `backups/` to `.gitignore`. Keep every file under 50 MB.

**b) Prepare the nested clone.** In `C:\project ai\CSE-276-PROJECT`, run `git fetch origin`, then `git checkout -b ca1-trained-model origin/main`. Copy into it every file that the outer repo would track: from the outer repo, use `git ls-files -co --exclude-standard`, excluding `CSE-276-PROJECT/` and `backups/`. Before committing, check for secrets with `grep -ri "token\|password\|api_key\|secret"`; `.env` must not be present.

**c) Commit and push.** Commit with a clear message ending with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (plus your own attribution if your policy requires one). Push without hanging on a login prompt: `GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=Never git push -u origin ca1-trained-model`. If authentication fails, STOP and give the user the exact command to run themselves.

**d) Hand off the merge.** Tell the user to open "Compare & pull request" on GitHub and merge into `main`, because faculty and Colab clone `main`. Do NOT push to main yourself unless the user explicitly says so.

### Task 4: Presentation, `presentation/K3P25LH_CSE276_G6_CA1_Slides.pptx`
Follow the guidelines: **8–10 slides, about 8 minutes**, short text, figures. Use this order:
1. Title (project, Section K3P25LH, Group G6, members with registration and roll numbers; use the report's values but flag the discrepancies above)
2. Problem and who benefits
3. Source: SIH26099, Ministry of PNG, link
4. The three objectives, verbatim
5. Dataset: 21,513 → 6,634 records, per-enterprise counts, CC BY 4.0
6. Method and baseline: pipeline diagram, self-supervised labels-by-construction, exact-match baseline
7. Implementation / system diagram: `src` modules, reviewer app, conflict-aware clustering
8. Results vs baseline: the README results table from metrics, plus `results/figures/07_model_vs_baselines_test.png` and `07_cluster_purity.png`
9. Demo: `results/report_evidence/screenshots/*.png` and the live `match_descriptions` example
10. Limitations and next steps: constructed test set is an upper bound; few AI-assisted review labels; label pending pairs using `results/metrics/batch2_review_priority.csv` (this is the optional active-learning extension)

Also export a PDF copy if possible. Every number must come from the metrics JSON. Speaker notes should say which member presents each slide (leave as "TBC" if unknown).

### Task 5: Report update notes, `docs/REPORT_UPDATE_NOTES.md` (do not edit the user's report file)
The draft is `C:\Users\LENOVO\Downloads\TBC_CSE276_GTBC_CA1_Report_DRAFT.pdf`; its source .docx was not found. For each section, give "replace this → with this" text using numbers from the metrics JSON:
- title page (K3P25LH, G6, course title, the reg-no/spelling discrepancies)
- Abstract
- Objectives 2 and 3 "How it is addressed"
- §4.4 data split (now a group-disjoint train/val/test split of constructed pairs)
- §5.3 (classifier IS trained; self-supervised; Table 5 features updated)
- Table 7 syllabus units (Unit V NLP, Unit VI clustering, Unit V evaluation)
- §6.4 Table 9 (new test counts)
- §7 Tables 10, 11 (real numbers, constructed held-out set; review-set tier; AI-assisted Batch-2 tier) and 14 (2/17/31/100 after the restore, explaining the test-artefact fix)
- §7.3, §7.4, §8, §9
- Appendix B tree (after the push)
- the Declaration's AI-use paragraph (mention Claude and Gemini/Antigravity, the self-supervised training, and the AI-assisted Batch-2 pre-annotation)

Also: rename the file per the guidelines (`K3P25LH_CSE276_G6_CA1_Report.pdf`), and copy the final PDF into `report/` before submission.

## Finish
Give the user a short summary with:
- what changed,
- the final metrics (from the JSON),
- the test count,
- the GitHub branch link and the merge instruction,
- the slide file path,
- the discrepancies they must confirm (registration number, name spelling, slide presenters, contribution table).
