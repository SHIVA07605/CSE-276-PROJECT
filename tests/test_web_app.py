"""Tests for the Material Match web app (web_app.py): service API, HTTP layer and read-only guarantee."""

import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "hybrid_er_model.joblib"
CATALOGUE = ROOT / "data" / "processed" / "preprocessed_materials.csv"

pytestmark = pytest.mark.skipif(
    not (MODEL_FILE.exists() and CATALOGUE.exists()),
    reason="trained model or catalogue missing (run python train_er_model.py)",
)

PROTECTED_FILES = [
    ROOT / "data" / "labelled" / "labelled_pairs.csv",
    ROOT / "data" / "processed" / "combined_candidate_pairs_150.csv",
    ROOT / "data" / "processed" / "candidate_pairs_batch2_expansion_100.csv",
    ROOT / "results" / "metrics" / "er_model_metrics.json",
]


def _digests():
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED_FILES if p.exists()}


@pytest.fixture(scope="module")
def service():
    from web_app import MaterialMatchService

    before = _digests()
    svc = MaterialMatchService(verbose=False)
    yield svc
    assert _digests() == before, "web app must never modify label, worksheet or metrics files"


def test_compare_flags_pressure_class_conflict(service):
    r = service.compare({"a": "GATE VALVE 4 INCH CLASS 150 API 600", "b": "GATE VALVE 4 INCH CLASS 300 API 600"})
    assert r["decision"] == "NOT_EQUIVALENT"
    assert "unit_spec_conflicts" in r["hard_conflicts"]
    assert r["specs_a"]["units"]["CLASS"] == ["150"]


def test_compare_accepts_format_variant(service):
    r = service.compare({"a": "BALL VALVE 2 INCH CLASS 150 API 6D", "b": 'BALL VALVE, 2", 150#, API-6D'})
    assert r["decision"] == "EQUIVALENT"
    assert r["match_probability"] >= service.model.thresholds["high"]


def test_compare_routes_generic_title_to_review(service):
    r = service.compare({"a": "AIR COMPRESSOR", "b": "AIR COMPRESSOR"})
    assert r["decision"] == "NEEDS_REVIEW"


@pytest.mark.parametrize("payload", [{"a": "", "b": "VALVE"}, {"a": "VALVE"}, {"a": "X" * 2000, "b": "VALVE"}])
def test_compare_rejects_bad_input(service, payload):
    from web_app import RequestError

    with pytest.raises(RequestError):
        service.compare(payload)


def test_search_finds_api6d_valves_and_orders_tiers(service):
    r = service.search({"query": "API 6D ball valve", "top_k": 20})
    assert r["results"][0]["decision"] == "EQUIVALENT"
    assert "API 6D BALL" in r["results"][0]["description"]
    tiers = [{"EQUIVALENT": 0, "NEEDS_REVIEW": 1, "NOT_EQUIVALENT": 2}[x["decision"]] for x in r["results"]]
    assert tiers == sorted(tiers)


def test_search_org_filter_and_validation(service):
    from web_app import RequestError

    r = service.search({"query": "ball valve", "organization": "NTPC", "top_k": 5})
    assert r["results"] and all("NTPC" in x["organizations"] for x in r["results"])
    with pytest.raises(RequestError):
        service.search({"query": "ball valve", "organization": "BHEL"})


def test_model_card_reads_metrics_json(service):
    card = service.model_card()
    metrics = json.loads((ROOT / "results" / "metrics" / "er_model_metrics.json").read_text(encoding="utf-8"))
    assert card["selected_model"] == metrics["selected_model"]
    assert card["gold"]["n_binary"] == metrics["gold"]["n_binary"]
    assert card["thresholds"] == metrics["thresholds"]


def test_clusters_filter(service):
    r = service.clusters_query({"q": "valve", "cross": "1"})
    if not r["available"]:
        pytest.skip("harmonised master not generated")
    assert all(row["is_cross_enterprise"] for row in r["rows"])
    assert all("VALVE" in (row["representative_description"] + row["distinct_descriptions"]).upper() for row in r["rows"])


def test_http_roundtrip(service):
    from web_app import make_server

    httpd = make_server(service, "127.0.0.1", 0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(base + "/") as resp:
            assert resp.status == 200 and b"Material Match" in resp.read()
        with urllib.request.urlopen(base + "/api/health") as resp:
            assert json.load(resp)["status"] == "ok"
        req = urllib.request.Request(base + "/api/compare", data=json.dumps({"a": "AIR COMPRESSOR", "b": "AIR COMPRESSOR"}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            assert json.load(resp)["decision"] == "NEEDS_REVIEW"
        bad = urllib.request.Request(base + "/api/compare", data=b"not json", headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad)
        assert exc.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_vercel_numpy_minilm_matches_sentence_transformers():
    """The PyTorch-free encoder used on Vercel must reproduce the training-time MiniLM embeddings."""
    import sys

    folder = ROOT / "vercel-app" / "minilm"
    if not (folder / "weights.npz").exists():
        pytest.skip("vercel-app not built (python deploy/vercel/build_vercel_app.py)")
    sys.path.insert(0, str(ROOT / "deploy" / "vercel"))
    from minilm_numpy import NumpyMiniLM
    from src.feature_extraction import generate_embeddings

    texts = ["BALL VALVE 2 INCH CLASS 150 API 6D", 'BALL VALVE, 2", 150#, API-6D', "AMC OF ED-XRF INSTRUMENT", "X"]
    ref = generate_embeddings(texts)
    got = NumpyMiniLM(folder).encode(texts)
    assert ((ref * got).sum(axis=1) > 0.9999).all()
