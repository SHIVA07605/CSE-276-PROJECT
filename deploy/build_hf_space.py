"""
Assemble (and optionally publish) the Hugging Face Space for the Material Match web app.

    python deploy/build_hf_space.py                                  # build dist/hf-space only
    python deploy/build_hf_space.py --push YOUR_HF_USERNAME/material-match   # build + upload (public)
    python deploy/build_hf_space.py --push YOUR_HF_USERNAME/material-match --private

Publishing needs a one-time login first:  hf auth login   (paste a token with "write" access from
https://huggingface.co/settings/tokens). Only the files listed in INCLUDE are uploaded: no labels,
worksheets, raw data, backups or .env.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hf-space"
SPACE_FILES = ROOT / "deploy" / "huggingface"

INCLUDE = [
    "web_app.py",
    "web/index.html",
    "src",
    "models/hybrid_er_model.joblib",
    "models/candidate_embeddings.npy",
    "models/embedding_id_mapping.csv",
    "data/processed/preprocessed_materials.csv",
    "results/metrics/er_model_metrics.json",
    "results/matching/harmonised_master_model.csv",
]
GITATTRIBUTES = "*.npy filter=lfs diff=lfs merge=lfs -text\n*.joblib filter=lfs diff=lfs merge=lfs -text\n"


def build() -> Path:
    missing = [p for p in INCLUDE if not (ROOT / p).exists()]
    if missing:
        sys.exit(f"[ERROR] Missing required files (run python train_er_model.py first): {missing}")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for rel in INCLUDE:
        src, dst = ROOT / rel, OUT / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for name in ("Dockerfile", "README.md", "requirements-web.txt"):
        shutil.copy2(SPACE_FILES / name, OUT / name)
    (OUT / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    n = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"Built {OUT}  ({n} files, {total / 1e6:.1f} MB)")
    return OUT


def push(repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("[ERROR] Not logged in to Hugging Face. Run:  hf auth login   then re-run this command.")
    if "/" not in repo_id:
        repo_id = f"{user}/{repo_id}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(OUT),
                      commit_message="Deploy Material Match web app")
    print(f"Uploaded. The Space builds in ~5-10 minutes: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", metavar="USER/SPACE", help="upload to this Hugging Face Space")
    ap.add_argument("--private", action="store_true", help="create the Space as private (only you can open it)")
    args = ap.parse_args()
    build()
    if args.push:
        push(args.push, args.private)
