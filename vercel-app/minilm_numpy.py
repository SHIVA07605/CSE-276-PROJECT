"""
all-MiniLM-L6-v2 in pure numpy: BERT WordPiece tokenizer + 6-layer transformer + mean pooling + L2 norm.

Used on Vercel, where PyTorch / ONNX Runtime do not fit the 500 MB function limit. Output is verified against
sentence-transformers by export_minilm_numpy.py (identical token ids; cosine >= 0.99999 on the full catalogue).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.special import erf


# ----------------------------------------------------------------------------- tokenizer
def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPieceTokenizer:
    """Same behaviour as the HF BertNormalizer(lowercase) + BertPreTokenizer + WordPiece used by MiniLM."""

    def __init__(self, vocab_file: Path, max_len: int = 256) -> None:
        tokens = Path(vocab_file).read_text(encoding="utf-8").split("\n")
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(tokens) if t != "" or i == 0}
        self.unk, self.cls, self.sep = self.vocab["[UNK]"], self.vocab["[CLS]"], self.vocab["[SEP]"]
        self.max_len = max_len

    def _normalise(self, text: str) -> str:
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD:
                continue
            cat = unicodedata.category(ch)
            if ch in "\t\n\r" or cat == "Zs":
                out.append(" ")
            elif cat.startswith("C"):
                continue
            elif _is_cjk(cp):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        text = "".join(out).lower()
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def _pre_tokenise(self, text: str) -> List[str]:
        words: List[str] = []
        for chunk in text.split():
            cur = ""
            for ch in chunk:
                if _is_punct(ch):
                    if cur:
                        words.append(cur)
                        cur = ""
                    words.append(ch)
                else:
                    cur += ch
            if cur:
                words.append(cur)
        return words

    def _wordpiece(self, word: str) -> List[int]:
        if len(word) > 100:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else "##" + word[start:end]
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk]
            ids.append(found)
            start = end
        return ids

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for w in self._pre_tokenise(self._normalise(text)):
            ids.extend(self._wordpiece(w))
        return [self.cls] + ids[: self.max_len - 2] + [self.sep]


# ----------------------------------------------------------------------------- model
def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


class NumpyMiniLM:
    def __init__(self, folder: Path, batch_size: int = 32) -> None:
        folder = Path(folder)
        w = np.load(folder / "weights.npz")
        self.w = {k: w[k].astype(np.float32) for k in w.files}
        self.n_layers = int(self.w.pop("n_layers")[0])
        self.n_heads = int(self.w.pop("n_heads")[0])
        self.tok = WordPieceTokenizer(folder / "vocab.txt", max_len=int(self.w.pop("max_len")[0]))
        self.batch_size = batch_size

    def _forward(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        w, (B, L) = self.w, ids.shape
        x = w["word"][ids] + w["pos"][:L][None] + w["type"][0][None, None]
        x = _layer_norm(x, w["emb_ln_g"], w["emb_ln_b"])
        H, D = self.n_heads, x.shape[-1] // self.n_heads
        neg = (1.0 - mask[:, None, None, :].astype(np.float32)) * np.finfo(np.float32).min
        for i in range(self.n_layers):
            p = f"l{i}_"
            q = (x @ w[p + "q_w"].T + w[p + "q_b"]).reshape(B, L, H, D).transpose(0, 2, 1, 3)
            k = (x @ w[p + "k_w"].T + w[p + "k_b"]).reshape(B, L, H, D).transpose(0, 2, 1, 3)
            v = (x @ w[p + "v_w"].T + w[p + "v_b"]).reshape(B, L, H, D).transpose(0, 2, 1, 3)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D) + neg
            s = np.exp(s - s.max(axis=-1, keepdims=True))
            s /= s.sum(axis=-1, keepdims=True)
            ctx = (s @ v).transpose(0, 2, 1, 3).reshape(B, L, H * D)
            x = _layer_norm(x + ctx @ w[p + "ao_w"].T + w[p + "ao_b"], w[p + "ao_ln_g"], w[p + "ao_ln_b"])
            h = x @ w[p + "i_w"].T + w[p + "i_b"]
            h = 0.5 * h * (1.0 + erf(h / np.sqrt(2.0)))
            x = _layer_norm(x + h @ w[p + "o_w"].T + w[p + "o_b"], w[p + "o_ln_g"], w[p + "o_ln_b"])
        return x

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out: List[np.ndarray] = []
        texts = [str(t) for t in texts]
        for i in range(0, len(texts), self.batch_size):
            seqs = [self.tok.encode(t) for t in texts[i:i + self.batch_size]]
            L = max(len(s) for s in seqs)
            ids = np.zeros((len(seqs), L), dtype=np.int64)
            mask = np.zeros((len(seqs), L), dtype=np.int64)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = s
                mask[j, :len(s)] = 1
            hidden = self._forward(ids, mask)
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.append(pooled.astype(np.float32))
        return np.vstack(out) if out else np.empty((0, 384), dtype=np.float32)


def export_weights(hf_model, max_len: int) -> Tuple[Dict[str, np.ndarray], int]:
    """Flatten a HF BertModel state dict into the names used above (word embeddings stored as float16)."""
    sd = {k: v.detach().cpu().numpy() for k, v in hf_model.state_dict().items()}
    out = {
        "word": sd["embeddings.word_embeddings.weight"].astype(np.float16),
        "pos": sd["embeddings.position_embeddings.weight"],
        "type": sd["embeddings.token_type_embeddings.weight"],
        "emb_ln_g": sd["embeddings.LayerNorm.weight"], "emb_ln_b": sd["embeddings.LayerNorm.bias"],
    }
    n = hf_model.config.num_hidden_layers
    names = {"q": "attention.self.query", "k": "attention.self.key", "v": "attention.self.value",
             "ao": "attention.output.dense", "i": "intermediate.dense", "o": "output.dense"}
    for i in range(n):
        base = f"encoder.layer.{i}."
        for short, long in names.items():
            out[f"l{i}_{short}_w"] = sd[base + long + ".weight"]
            out[f"l{i}_{short}_b"] = sd[base + long + ".bias"]
        out[f"l{i}_ao_ln_g"] = sd[base + "attention.output.LayerNorm.weight"]
        out[f"l{i}_ao_ln_b"] = sd[base + "attention.output.LayerNorm.bias"]
        out[f"l{i}_o_ln_g"] = sd[base + "output.LayerNorm.weight"]
        out[f"l{i}_o_ln_b"] = sd[base + "output.LayerNorm.bias"]
    out["n_layers"] = np.array([n])
    out["n_heads"] = np.array([hf_model.config.num_attention_heads])
    out["max_len"] = np.array([max_len])
    return out, n
