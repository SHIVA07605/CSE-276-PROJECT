"""
Assemble vercel-app/: a self-contained Vercel project for the Material Match website.

    python deploy/vercel/build_vercel_app.py

Vercel project setting: Root Directory = vercel-app. Re-run this after retraining so the deployed model
matches models/hybrid_er_model.joblib. Only the app, src/, the trained model, cached embeddings, the public
catalogue, metrics/cluster outputs and the numpy MiniLM encoder are copied: no labels, worksheets, raw data or .env.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "vercel-app"

COPY = {
    "web_app.py": "web_app.py",
    "src": "src",
    "web/index.html": "index.html",              # served statically at /
    "models/hybrid_er_model.joblib": "models/hybrid_er_model.joblib",
    "models/candidate_embeddings.npy": "models/candidate_embeddings.npy",
    "models/embedding_id_mapping.csv": "models/embedding_id_mapping.csv",
    "data/processed/preprocessed_materials.csv": "data/processed/preprocessed_materials.csv",
    "results/metrics/er_model_metrics.json": "results/metrics/er_model_metrics.json",
    "results/matching/harmonised_master_model.csv": "results/matching/harmonised_master_model.csv",
    "deploy/vercel/minilm_numpy.py": "minilm_numpy.py",
    "deploy/vercel/minilm": "minilm",
    "deploy/vercel/api_index.py": "api/index.py",
    "deploy/vercel/requirements.txt": "requirements.txt",
    "deploy/vercel/vercel.json": "vercel.json",
}
README = """# Material Match on Vercel (generated)

Built by `python deploy/vercel/build_vercel_app.py`; do not edit by hand. Vercel project Root Directory: `vercel-app`.
`index.html` is the page; `api/index.py` runs the trained hybrid entity-resolution model with MiniLM in pure numpy.
"""


def main() -> None:
    missing = [s for s in COPY if not (ROOT / s).exists()]
    if missing:
        sys.exit(f"[ERROR] Missing: {missing}. Train the model (python train_er_model.py) and export the encoder "
                 f"(python deploy/vercel/export_minilm_numpy.py) first.")
    if OUT.exists():
        shutil.rmtree(OUT)
    for src_rel, dst_rel in COPY.items():
        src, dst = ROOT / src_rel, OUT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.onnx", "tokenizer.json"))
        else:
            shutil.copy2(src, dst)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    files = [f for f in OUT.rglob("*") if f.is_file()]
    print(f"Built {OUT} ({len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
