# Agent Instructions: Material Code Harmonisation (CSE 276 CA-1)

## Hard Rules (Academic Integrity)
1. **NEVER** write to `human_label`, `confidence_rating`, or `reviewer_notes` columns of any worksheet (`data/labelled/labelled_pairs.csv`, `data/processed/candidate_pairs_batch2_expansion_100.csv`, `data/processed/combined_candidate_pairs_150.csv`). Only human reviewers write there through `review_app.py`.
2. Existing 19 binary review labels are **review-set labels (AI-assisted, not independent human ground truth)**, NEVER "human gold labels".
3. **NEVER** treat `ai_suggested_hypothesis` as ground truth. Never fit any model on review-set labels.
4. Every metric in docs, README, report notes, and slides MUST come directly from `results/metrics/er_model_metrics.json`.
5. Held-out test pairs are constructed from real records; test scores are an upper bound on real-world accuracy.
6. Do **NOT** modify notebooks 01–04 and 06, `review_app.py` behavior, or the legacy `generate_nb0*.py` scripts (which hardcode local paths).
7. Do **NOT** push to the `main` branch directly. Push only to branch `ca1-trained-model`. Never commit or push `backups/`, `data/raw/`, or `.env`.

## Core Commands
- **Preflight environment check:** `python train_er_model.py --check`
- **One-command train & evaluation:** `python train_er_model.py`
- **Rebuild and execute notebooks 05 & 07:** `python generate_model_notebooks.py`
- **Regenerate documentation & README table:** `python generate_model_report.py`
- **Run test suite:** `python -m pytest`
- **Model website (read-only, http://localhost:8060):** `python web_app.py` (`web/index.html` is the page; tests in `tests/test_web_app.py`)
- **Vercel deployment:** `python deploy/vercel/build_vercel_app.py` regenerates `vercel-app/` (Vercel Root Directory). MiniLM runs in pure numpy there (no PyTorch, 500 MB cap); re-run after retraining.
- **Alternative host (Hugging Face Space, Docker):** `python deploy/build_hf_space.py [--push USER/SPACE]`.

## Key Artefacts Produced
- `models/hybrid_er_model.joblib`: Trained model (classifier + TF-IDF + decision thresholds + metadata).
- `results/metrics/er_model_metrics.json`: Source of truth for all project metrics.
- `results/metrics/model_comparison_synthetic_test.csv`: Held-out comparison table.
- `results/metrics/model_comparison_gold_labels.csv`: Review-set comparison table.
- `results/metrics/model_comparison_ai_assisted_batch2.csv`: Batch 2 AI-assisted evaluation table.
- `results/matching/model_scored_candidate_pairs.csv`: All 22,464 scored real pairs.
- `results/matching/harmonised_master_model.csv`: Proposed harmonised clusters and codes.

## Known Harmless Warnings
- `KeyError` lines from `joblib/loky resource_tracker` on Windows during multiprocessing exit.
- `HF_TOKEN` warning when downloading `all-MiniLM-L6-v2` without authentication.

