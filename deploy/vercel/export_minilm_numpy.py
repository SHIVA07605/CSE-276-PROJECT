"""
Export all-MiniLM-L6-v2 for the PyTorch-free Vercel deployment and verify it against sentence-transformers.

    python deploy/vercel/export_minilm_numpy.py          # export + verify
    python deploy/vercel/export_minilm_numpy.py --verify-only

Writes deploy/vercel/minilm/{weights.npz, vocab.txt}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "minilm"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from minilm_numpy import NumpyMiniLM, export_weights  # noqa: E402


def export() -> None:
    from sentence_transformers import SentenceTransformer

    from src.config import DEFAULT_EMBEDDING_MODEL

    st = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, device="cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    weights, _ = export_weights(st[0].auto_model.eval(), st.max_seq_length)
    np.savez_compressed(OUT / "weights.npz", **weights)
    vocab = st.tokenizer.get_vocab()
    (OUT / "vocab.txt").write_text("\n".join(t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])), encoding="utf-8")
    print(f"Exported {OUT / 'weights.npz'} ({(OUT / 'weights.npz').stat().st_size / 1e6:.1f} MB)")


def verify() -> None:
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    from src.config import DEFAULT_EMBEDDING_MODEL
    from src.feature_extraction import generate_embeddings

    mats = pd.read_csv(ROOT / "data/processed/preprocessed_materials.csv", low_memory=False)
    texts = sorted(set(mats["cleaned_description"].dropna().astype(str).str.strip().str.upper()) - {""})
    texts += ['BALL VALVE, 2", 150#, API-6D', "X", "Café crème ½\" Ø50 — 中文 test\tTAB", "A" * 300, " ".join(["VALVE"] * 400)]

    hf_tok = SentenceTransformer(DEFAULT_EMBEDDING_MODEL).tokenizer
    enc = NumpyMiniLM(OUT)
    bad = [t for t in texts if enc.tok.encode(t) != hf_tok(t, truncation=True, max_length=enc.tok.max_len)["input_ids"]]
    print(f"Tokenizer: {len(texts) - len(bad)}/{len(texts)} identical token sequences")
    if bad:
        sys.exit(f"[FAIL] tokenizer differs, e.g. {bad[:3]}")

    ref = generate_embeddings(texts, batch_size=128)
    got = enc.encode(texts)
    cos = (ref * got).sum(axis=1)
    print(f"Embeddings: min cosine {cos.min():.7f}, mean {cos.mean():.8f}, max abs diff {np.abs(ref - got).max():.2e}")
    if cos.min() < 0.9999:
        sys.exit("[FAIL] numpy MiniLM differs from sentence-transformers")
    print("[OK] numpy MiniLM matches sentence-transformers")


if __name__ == "__main__":
    if "--verify-only" not in sys.argv:
        export()
    verify()
