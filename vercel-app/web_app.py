"""
Material Match: a browser interface for the trained hybrid entity-resolution model.

Run:
    python web_app.py                  # starts http://localhost:8060 and opens a browser tab
    python web_app.py --port 9000      # use another port
    python web_app.py --no-browser     # do not open a browser tab

What it does (all read-only, nothing on disk is modified):
    Compare      score two material descriptions -> EQUIVALENT / NOT_EQUIVALENT / NEEDS_REVIEW
    Search       find the closest records for a new description in the 6,634-record catalogue
    Clusters     browse the proposed harmonised codes (results/matching/harmonised_master_model.csv)
    Model card   every number comes from results/metrics/er_model_metrics.json

Requires the trained model at models/hybrid_er_model.joblib (python train_er_model.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, MODELS_DIR  # noqa: E402
from src.er_pipeline import HARMONISED_MASTER_PATH, METRICS_JSON_PATH, MODEL_PATH, load_model  # noqa: E402
from src.feature_extraction import generate_embeddings as _st_embeddings  # noqa: E402
from src.pair_features import HARD_CONFLICT_FEATURES, parse_specifications  # noqa: E402
from src.preprocessing.text_cleaner import normalise_text  # noqa: E402

DEFAULT_PORT = 8060
INDEX_HTML = PROJECT_ROOT / "web" / "index.html"
if not INDEX_HTML.exists():
    INDEX_HTML = PROJECT_ROOT / "index.html"
MAX_BODY_BYTES = 16_384
MAX_TEXT_CHARS = 1_000
SHORTLIST_PER_SIGNAL = 50
DECISION_TIER = {"EQUIVALENT": 0, "NEEDS_REVIEW": 1, "NOT_EQUIVALENT": 2}
ORG_SHORT = {
    "Indian Oil Corporation Limited": "IOCL",
    "Oil India Limited": "OIL",
    "NTPC Limited": "NTPC",
}
EVIDENCE_FEATURES = [
    ("sem_cosine", "Semantic similarity (MiniLM)"),
    ("tfidf_word_cosine", "Word TF-IDF cosine"),
    ("tfidf_char_cosine", "Character TF-IDF cosine"),
    ("fuzz_token_set", "Fuzzy token-set ratio"),
    ("token_jaccard", "Token Jaccard overlap"),
]
SPEC_COUNT_FEATURES = [
    ("unit_spec_matches", "Shared unit specifications"),
    ("unit_spec_conflicts", "Conflicting unit specifications"),
    ("unit_spec_one_side", "Unit specs stated on one side only"),
    ("group_matches", "Shared descriptor groups"),
    ("group_conflicts", "Conflicting descriptor groups"),
    ("group_one_side", "Descriptor groups on one side only"),
    ("standard_matches", "Shared standards"),
    ("standard_conflicts", "Conflicting standards"),
    ("grade_conflicts", "Conflicting material grades"),
    ("bearing_conflicts", "Conflicting bearing series"),
    ("engine_category_conflict", "Different equipment category (rule engine)"),
]


_LIGHT_ENCODER = None


def generate_embeddings(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """MiniLM sentence embeddings. Uses sentence-transformers (PyTorch) by default; on hosts without PyTorch
    (Vercel) set MATERIAL_MATCH_ENCODER=numpy to use the verified pure-numpy MiniLM (deploy/vercel/minilm_numpy.py)."""
    global _LIGHT_ENCODER
    if os.environ.get("MATERIAL_MATCH_ENCODER", "").lower() == "numpy":
        if _LIGHT_ENCODER is None:
            local_dir = PROJECT_ROOT / "deploy" / "vercel"
            if local_dir.exists() and str(local_dir) not in sys.path:
                sys.path.append(str(local_dir))
            from minilm_numpy import NumpyMiniLM

            folder = Path(os.environ.get("MATERIAL_MATCH_MINILM_DIR", local_dir / "minilm"))
            _LIGHT_ENCODER = NumpyMiniLM(folder)
        return _LIGHT_ENCODER.encode(texts)
    return _st_embeddings(texts, batch_size=batch_size)


class RequestError(ValueError):
    """Invalid user input; reported to the browser as HTTP 400."""


def _plain(o: Any) -> Any:
    """Make sets / numpy values JSON-serialisable."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return sorted(_plain(v) for v in o)
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _clean_input(text: Any, field: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise RequestError(f"'{field}' must be a non-empty material description.")
    if len(text) > MAX_TEXT_CHARS:
        raise RequestError(f"'{field}' is longer than {MAX_TEXT_CHARS} characters.")
    norm = normalise_text(text).strip().upper()
    if not norm:
        raise RequestError(f"'{field}' has no usable text after normalisation.")
    return norm


class MaterialMatchService:
    """Loads the model and catalogue once; answers compare / search / cluster / model-card queries."""

    def __init__(self, model_path: Path = MODEL_PATH, verbose: bool = True) -> None:
        t0 = time.time()
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run: python train_er_model.py")
        self.model = load_model(model_path)
        self.lock = threading.Lock()
        self.spec_cache: Dict[str, Any] = {}
        self._load_catalogue()
        self._load_clusters()
        generate_embeddings(["WARM UP"])  # load MiniLM now, not on the first request
        self.startup_seconds = round(time.time() - t0, 1)
        if verbose:
            print(f"Model + catalogue ready in {self.startup_seconds}s "
                  f"({self.n_records:,} records, {len(self.cat_desc):,} distinct descriptions, {len(self.clusters):,} clusters)")

    # ------------------------------------------------------------------ loading
    def _load_catalogue(self) -> None:
        mats = pd.read_csv(DATA_PROCESSED_DIR / "preprocessed_materials.csv", low_memory=False)
        mats["corpus_id"] = mats["corpus_id"].astype(str)
        mats["_key"] = mats["cleaned_description"].fillna("").astype(str).str.strip().str.upper()
        mats = mats[mats["_key"] != ""].copy()
        self.n_records = len(mats)

        emb_path, map_path = MODELS_DIR / "candidate_embeddings.npy", MODELS_DIR / "embedding_id_mapping.csv"
        row_of: Dict[str, int] = {}
        emb = None
        if emb_path.exists() and map_path.exists():
            emb = np.load(emb_path).astype(np.float32)
            mapping = pd.read_csv(map_path)
            row_of = dict(zip(mapping["corpus_id"].astype(str), mapping["index"].astype(int)))

        groups = mats.groupby("_key", sort=False)
        self.cat_desc: List[str] = list(groups.groups.keys())
        self.cat_records: List[List[Dict[str, str]]] = []
        self.cat_orgs: List[List[str]] = []
        vectors: List[Optional[np.ndarray]] = []
        for key in self.cat_desc:
            g = groups.get_group(key)
            recs = [
                {"corpus_id": cid, "organization": ORG_SHORT.get(org, str(org)), "material_code": str(code) if pd.notna(code) else ""}
                for cid, org, code in zip(g["corpus_id"], g["organization"], g["material_code"])
            ]
            self.cat_records.append(recs)
            self.cat_orgs.append(sorted({r["organization"] for r in recs}))
            idx = next((row_of[c] for c in g["corpus_id"] if c in row_of), None)
            vectors.append(emb[idx] if emb is not None and idx is not None else None)

        missing = [i for i, v in enumerate(vectors) if v is None]
        if missing:  # cached embeddings absent or incomplete -> encode with MiniLM
            enc = generate_embeddings([self.cat_desc[i] for i in missing], batch_size=128)
            for i, v in zip(missing, enc):
                vectors[i] = v
        self.cat_emb = np.vstack(vectors).astype(np.float32)
        self.cat_word = self.model.tfidf.word.transform(self.cat_desc)
        self.cat_char = self.model.tfidf.char.transform(self.cat_desc)

    def _load_clusters(self) -> None:
        if HARMONISED_MASTER_PATH.exists():
            self.clusters = pd.read_csv(HARMONISED_MASTER_PATH)
        else:
            self.clusters = pd.DataFrame()

    # ------------------------------------------------------------------ scoring core
    def _score(self, texts_a: List[str], texts_b: List[str], emb_a: np.ndarray, emb_b: np.ndarray):
        with self.lock:
            X = self.model.featurize(texts_a, texts_b, emb_a, emb_b, spec_cache=self.spec_cache)
            dec = self.model.decide(X)
        return X, dec

    # ------------------------------------------------------------------ public API
    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_version": self.model.VERSION,
            "classifier": str(self.model.metadata.get("model_name", "")),
            "embedding_model": str(self.model.metadata.get("embedding_model", "")),
            "thresholds": self.model.thresholds,
            "catalogue_records": self.n_records,
            "distinct_descriptions": len(self.cat_desc),
            "clusters": int(len(self.clusters)),
            "startup_seconds": self.startup_seconds,
            "hosted": os.environ.get("MATERIAL_MATCH_HOSTED", "") == "1",
        }

    def compare(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        a = _clean_input(payload.get("a"), "a")
        b = _clean_input(payload.get("b"), "b")
        emb = generate_embeddings([a, b])
        X, dec = self._score([a], [b], emb[[0]], emb[[1]])
        row, d = X.iloc[0], dec.iloc[0]
        return _plain({
            "normalised_a": a,
            "normalised_b": b,
            "decision": d["decision"],
            "confidence": d["confidence"],
            "match_probability": float(d["match_probability"]),
            "reason": d["reason"],
            "thresholds": self.model.thresholds,
            "similarity": [{"key": k, "label": lbl, "value": float(row[k])} for k, lbl in EVIDENCE_FEATURES],
            "spec_counts": [{"key": k, "label": lbl, "value": float(row[k]), "hard_conflict": k in HARD_CONFLICT_FEATURES}
                            for k, lbl in SPEC_COUNT_FEATURES],
            "hard_conflicts": [f for f in HARD_CONFLICT_FEATURES if row[f] > 0],
            "specs_a": parse_specifications(a),
            "specs_b": parse_specifications(b),
        })

    def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        q = _clean_input(payload.get("query"), "query")
        try:
            top_k = max(1, min(int(payload.get("top_k", 10)), 50))
        except (TypeError, ValueError):
            raise RequestError("'top_k' must be a number between 1 and 50.")
        org = str(payload.get("organization") or "").strip().upper()
        if org and org not in ORG_SHORT.values():
            raise RequestError(f"'organization' must be one of {sorted(ORG_SHORT.values())}.")

        t0 = time.time()
        qv = generate_embeddings([q])[0]
        sem = self.cat_emb @ qv
        word = np.asarray((self.cat_word @ self.model.tfidf.word.transform([q]).T).todense()).ravel()
        char = np.asarray((self.cat_char @ self.model.tfidf.char.transform([q]).T).todense()).ravel()
        allowed = np.ones(len(self.cat_desc), dtype=bool)
        if org:
            allowed = np.array([org in orgs for orgs in self.cat_orgs])
        shortlist: set = set()
        for signal in (sem, word, char):
            s = np.where(allowed, signal, -np.inf)
            k = min(SHORTLIST_PER_SIGNAL, int(allowed.sum()))
            if k:
                shortlist.update(int(i) for i in np.argpartition(-s, k - 1)[:k] if np.isfinite(s[i]))
        idx = sorted(shortlist)
        if not idx:
            return {"query": q, "results": [], "shortlist_size": 0, "decision_counts": {}, "seconds": 0.0}

        _, dec = self._score([q] * len(idx), [self.cat_desc[i] for i in idx], np.repeat(qv[None, :], len(idx), axis=0), self.cat_emb[idx])
        dec = dec.reset_index(drop=True)
        dec["cat_idx"] = idx
        dec["sem"] = sem[idx]
        # Matches first, then pairs needing review (both by probability), then the closest
        # non-equivalent records by semantic similarity ("similar but different").
        dec["tier"] = dec["decision"].map(DECISION_TIER).fillna(3)
        dec["rank_score"] = np.where(dec["decision"] == "NOT_EQUIVALENT", dec["sem"], dec["match_probability"])
        dec = dec.sort_values(["tier", "rank_score", "sem"], ascending=[True, False, False])
        results = []
        for _, r in dec.head(top_k).iterrows():
            i = int(r["cat_idx"])
            recs = self.cat_records[i]
            results.append({
                "description": self.cat_desc[i],
                "decision": r["decision"],
                "confidence": r["confidence"],
                "match_probability": float(r["match_probability"]),
                "reason": r["reason"],
                "semantic_similarity": round(float(r["sem"]), 4),
                "organizations": self.cat_orgs[i],
                "record_count": len(recs),
                "records": recs[:8],
            })
        return _plain({
            "query": q,
            "organization": org or "ALL",
            "shortlist_size": len(idx),
            "decision_counts": dec["decision"].value_counts().to_dict(),
            "results": results,
            "seconds": round(time.time() - t0, 2),
        })

    def clusters_query(self, params: Dict[str, str]) -> Dict[str, Any]:
        df = self.clusters
        if df.empty:
            return {"total": 0, "matched": 0, "rows": [], "available": False}
        q = params.get("q", "").strip().upper()
        cross = params.get("cross", "") in ("1", "true", "yes")
        try:
            limit = max(1, min(int(params.get("limit", 100)), 500))
        except ValueError:
            limit = 100
        m = pd.Series(True, index=df.index)
        if q:
            hay = (df["representative_description"].fillna("") + " " + df["distinct_descriptions"].fillna("") + " " + df["harmonised_code"].fillna("")).str.upper()
            m &= hay.str.contains(q, regex=False)
        if cross:
            m &= df["is_cross_enterprise"].astype(bool)
        sub = df[m]
        rows = sub.head(limit).where(pd.notna(sub.head(limit)), None).to_dict(orient="records")
        return _plain({
            "available": True,
            "total": int(len(df)),
            "cross_enterprise_total": int(df["is_cross_enterprise"].astype(bool).sum()),
            "matched": int(len(sub)),
            "rows": rows,
        })

    def model_card(self) -> Dict[str, Any]:
        if not METRICS_JSON_PATH.exists():
            return {"available": False}
        m = json.loads(METRICS_JSON_PATH.read_text(encoding="utf-8"))
        keep = ["model_version", "selected_model", "thresholds", "training_data", "synthetic_test",
                "synthetic_test_policy", "synthetic_test_cluster_purity", "unseen_mutation_detection_rate",
                "gold", "ai_assisted_batch2", "real_candidates", "runtime_seconds"]
        out = {k: m[k] for k in keep if k in m}
        out["available"] = True
        out["source"] = "results/metrics/er_model_metrics.json"
        return _plain(out)


# ====================================================================== HTTP layer
class Handler(BaseHTTPRequestHandler):
    service: MaterialMatchService  # set by make_server
    server_version = "MaterialMatch/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # concise access log
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.command} {self.path.split('?')[0]} -> {args[1] if len(args) > 1 else ''}\n")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj, allow_nan=False, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        params = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/health":
                self._json(200, self.service.health())
            elif url.path == "/api/model":
                self._json(200, self.service.model_card())
            elif url.path == "/api/clusters":
                self._json(200, self.service.clusters_query(params))
            else:
                self._json(404, {"error": "Not found"})
        except Exception as e:  # pragma: no cover - surfaced to the browser
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self) -> None:  # noqa: N802
        url = urllib.parse.urlparse(self.path)
        routes = {"/api/compare": self.service.compare, "/api/search": self.service.search}
        if url.path not in routes:
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise RequestError("Request body too large.")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise RequestError("Request body must be a JSON object.")
            self._json(200, routes[url.path](payload))
        except (RequestError, json.JSONDecodeError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # pragma: no cover - surfaced to the browser
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def make_server(service: MaterialMatchService, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    ap = argparse.ArgumentParser(description="Material Match web app for the hybrid ER model.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 (this computer only, default) or 0.0.0.0 (your local network)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()

    print("Loading model, MiniLM encoder and catalogue (first start takes ~20 s)...")
    try:
        service = MaterialMatchService()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    try:
        httpd = make_server(service, args.host, args.port)
    except OSError:
        print(f"\n[ERROR] Port {args.port} is already in use. Material Match may already be running: "
              f"open http://localhost:{args.port}, or start on another port with: python web_app.py --port {args.port + 1}")
        sys.exit(1)
    url = f"http://localhost:{args.port}"
    print("=" * 72)
    print(f"MATERIAL MATCH is running at: {url}")
    if args.host == "0.0.0.0":
        print("Listening on all network interfaces: anyone on your network can open it.")
    print("Press Ctrl+C to stop.")
    print("=" * 72)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Material Match.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
