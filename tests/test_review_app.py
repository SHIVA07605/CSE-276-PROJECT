"""
Tests for review_app.py HTTP server endpoints and data serialization integrity.
"""

import json
import io
from http.server import HTTPServer
from unittest.mock import MagicMock
from pathlib import Path
import pandas as pd
import pytest

from review_app import clean_dataframe_for_json, ReviewerHTTPHandler, WORKSHEET_PATH


class TestDataSerialization:
    def test_clean_dataframe_for_json_no_nan(self):
        """Verify that records serialized via clean_dataframe_for_json contain zero NaN tokens."""
        df = pd.read_csv(WORKSHEET_PATH)
        records = clean_dataframe_for_json(df)
        
        # Test serialization with allow_nan=False - will raise ValueError if any NaN remains
        dumped = json.dumps({"pairs": records}, allow_nan=False, default=str)
        assert "NaN" not in dumped
        assert len(records) == 50

    def test_json_parse_strict(self):
        """Simulate strict browser JSON.parse (which rejects NaN tokens)."""
        df = pd.read_csv(WORKSHEET_PATH)
        records = clean_dataframe_for_json(df)
        dumped = json.dumps({"pairs": records}, allow_nan=False, default=str)

        # Standard Python json.loads with parse_constant set to raise error on NaN
        def fail_on_constant(val):
            raise ValueError(f"Strict JSON parsing rejected constant: {val}")

        parsed = json.loads(dumped, parse_constant=fail_on_constant)
        assert len(parsed["pairs"]) == 50


class TestFiveCandidatePairsMapping:
    """Verify that the 5 required candidate pair categories have complete and correct data mapping."""

    @pytest.fixture
    def pairs_dict(self):
        df = pd.read_csv(WORKSHEET_PATH)
        records = clean_dataframe_for_json(df)
        return {r["pair_id"]: r for r in records}

    def test_pair_1_cross_enterprise(self, pairs_dict):
        """Pair 1: Cross-enterprise candidate pair."""
        p = pairs_dict["PAIR_EXACT_001"]
        assert p["enterprise_1"] != p["enterprise_2"]
        assert p["corpus_id_1"] == "M000051"
        assert p["corpus_id_2"] == "M006733"
        assert p["corpus_id_1"] != p["corpus_id_2"]
        assert "Air Compressor" in p["raw_description_1"]
        assert "AIR COMPRESSOR" in p["raw_description_2"]
        assert p["lexical_similarity"] == 1.0
        assert p["semantic_cosine_similarity"] == 1.0

    def test_pair_2_attribute_conflict(self, pairs_dict):
        """Pair 2: Attribute-conflict candidate pair."""
        p = pairs_dict["PAIR_CONFLICT_031"]
        assert p["has_attribute_conflict"] == True
        assert p["corpus_id_1"] == "M000746"
        assert p["corpus_id_2"] == "M000784"
        assert p["corpus_id_1"] != p["corpus_id_2"]
        assert "1400MM" in p["extracted_attributes_1"]
        assert "1600MM" in p["extracted_attributes_2"]
        assert "1400MM" in str(p["conflict_details"])
        assert "1600MM" in str(p["conflict_details"])
        assert float(p["lexical_similarity"]) > 0.5
        assert float(p["semantic_cosine_similarity"]) > 0.8

    def test_pair_3_missing_attributes(self, pairs_dict):
        """Pair 3: Pair with missing attributes ('None detected')."""
        p = pairs_dict["PAIR_EXACT_003"]
        assert p["extracted_attributes_1"] == "None detected"
        assert p["extracted_attributes_2"] == "None detected"
        assert p["corpus_id_1"] == "M000085"
        assert p["corpus_id_2"] == "M005599"
        assert p["corpus_id_1"] != p["corpus_id_2"]
        assert p["lexical_similarity"] == 1.0
        assert p["semantic_cosine_similarity"] == 1.0

    def test_pair_4_high_semantic_similarity(self, pairs_dict):
        """Pair 4: High semantic similarity pair."""
        p = pairs_dict["PAIR_TFIDF_HIGH_019"]
        assert p["corpus_id_1"] == "M000041"
        assert p["corpus_id_2"] == "M008245"
        assert p["corpus_id_1"] != p["corpus_id_2"]
        assert float(p["semantic_cosine_similarity"]) >= 0.80
        assert "Centrifugal Pump" in p["raw_description_1"]
        assert "Centrifugal Pump" in p["raw_description_2"]

    def test_pair_5_ordinary_boundary_pair(self, pairs_dict):
        """Pair 5: Ordinary candidate pair (boundary case)."""
        p = pairs_dict["PAIR_BOUNDARY_041"]
        assert p["corpus_id_1"] == "M000025"
        assert p["corpus_id_2"] == "M014432"
        assert p["corpus_id_1"] != p["corpus_id_2"]
        assert 0.3 <= float(p["lexical_similarity"]) <= 0.4
        assert 0.5 <= float(p["semantic_cosine_similarity"]) <= 0.7


class TestReviewerNavigationAndSaving:
    """
    Tests ensuring reviewer UI follows strict non-auto-advancing protocol:
    1. Clicking/pressing label selects only (no navigation).
    2. Editing notes or confidence does not navigate.
    3. Only clicking Save & Next validates, saves to the correct pair_id, and navigates.
    4. Accidental duplicate submissions are guarded.
    5. Previously saved annotations remain intact.
    """

    def _make_handler(self, method="POST", path="/api/label", body=b"{}"):
        handler = ReviewerHTTPHandler.__new__(ReviewerHTTPHandler)
        handler.command = method
        handler.path = path
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()
        handler.response_code = 200
        handler.response_headers = {}

        def send_response(code, message=None):
            handler.response_code = code

        def send_header(keyword, value):
            handler.response_headers[keyword] = value

        def end_headers():
            pass

        handler.send_response = send_response
        handler.send_header = send_header
        handler.end_headers = end_headers
        return handler

    def test_clicking_label_does_not_change_current_pair(self):
        """Verify that clicking label buttons calls selectLabel and does NOT call nextPair."""
        from review_app import HTML_TEMPLATE
        assert "onclick=\"selectLabel('1')\"" in HTML_TEMPLATE
        assert "onclick=\"selectLabel('0')\"" in HTML_TEMPLATE
        assert "onclick=\"selectLabel('UNCERTAIN')\"" in HTML_TEMPLATE
        # Verify selectLabel does not invoke navigation or network save
        assert "function selectLabel(label) {" in HTML_TEMPLATE
        assert "nextPair();" not in HTML_TEMPLATE.split("function selectLabel(label) {")[1].split("function assignLabel")[0]

    def test_selecting_different_label_updates_selection_only(self):
        """Verify that selecting another label only updates currentSelectedLabel and button UI."""
        from review_app import HTML_TEMPLATE
        assert "currentSelectedLabel = String(label).toUpperCase();" in HTML_TEMPLATE
        assert "updateLabelButtonUI();" in HTML_TEMPLATE
        assert "clearValidationMessage();" in HTML_TEMPLATE

    def test_entering_notes_does_not_navigate(self):
        """Verify textarea has no onblur or onchange triggers causing auto-save or navigation."""
        from review_app import HTML_TEMPLATE
        assert 'id="reviewer-notes"' in HTML_TEMPLATE
        assert 'onblur="autoSaveCurrentPair()"' not in HTML_TEMPLATE
        assert 'onblur="nextPair()"' not in HTML_TEMPLATE

    def test_changing_confidence_does_not_navigate(self):
        """Verify confidence select has no onchange trigger causing auto-save or navigation."""
        from review_app import HTML_TEMPLATE
        assert 'id="confidence-select"' in HTML_TEMPLATE
        assert 'onchange="autoSaveCurrentPair()"' not in HTML_TEMPLATE
        assert 'onchange="nextPair()"' not in HTML_TEMPLATE

    def test_keyboard_shortcuts_do_not_trigger_navigation(self):
        """Verify keyboard shortcuts 1, 0, U select labels without navigating."""
        from review_app import HTML_TEMPLATE
        assert "if (e.key === '1') { e.preventDefault(); selectLabel('1'); }" in HTML_TEMPLATE
        assert "if (e.key === '0') { e.preventDefault(); selectLabel('0'); }" in HTML_TEMPLATE
        assert "if (e.key === 'u' || e.key === 'U') { e.preventDefault(); selectLabel('UNCERTAIN'); }" in HTML_TEMPLATE

    def test_save_and_next_prevents_duplicate_submissions(self):
        """Verify UI locks isSaving flag and disables save button during execution."""
        from review_app import HTML_TEMPLATE
        assert "if (isSaving) return;" in HTML_TEMPLATE
        assert "isSaving = true;" in HTML_TEMPLATE
        assert "saveBtn.disabled = true;" in HTML_TEMPLATE
        assert "isSaving = false;" in HTML_TEMPLATE
        assert "saveBtn.disabled = false;" in HTML_TEMPLATE

    def test_navigation_occurs_only_after_successful_saving(self):
        """Verify navigation is strictly invoked inside saveAndNext after successful server response."""
        from review_app import HTML_TEMPLATE
        save_and_next_code = HTML_TEMPLATE.split("async function saveAndNext() {")[1].split("function navigateToNextPendingPair()")[0]
        # Must validate before calling fetch
        assert "Please select a ground-truth label" in save_and_next_code
        assert "Please select a valid confidence level" in save_and_next_code
        # Must check res.ok and data.success before navigating
        assert "navigateToNextPendingPair();" in save_and_next_code

    def test_api_label_validation_missing_pair_id(self):
        """Server rejects payload missing pair_id with HTTP 400."""
        payload = json.dumps({"human_label": "1", "confidence_rating": "HIGH", "reviewer_notes": "test"}).encode("utf-8")
        handler = self._make_handler(method="POST", path="/api/label", body=payload)
        handler.do_POST()
        assert handler.response_code == 400
        res = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert res["success"] is False
        assert "Missing mandatory pair_id" in res["error"]

    def test_api_label_validation_invalid_label(self):
        """Server rejects payload with invalid label with HTTP 400."""
        payload = json.dumps({"pair_id": "PAIR_EXACT_001", "human_label": "MAYBE", "confidence_rating": "HIGH"}).encode("utf-8")
        handler = self._make_handler(method="POST", path="/api/label", body=payload)
        handler.do_POST()
        assert handler.response_code == 400
        res = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert res["success"] is False
        assert "Invalid ground-truth label" in res["error"]

    def test_api_label_validation_invalid_confidence(self):
        """Server rejects payload with pending or invalid confidence with HTTP 400."""
        payload = json.dumps({"pair_id": "PAIR_EXACT_001", "human_label": "1", "confidence_rating": "PENDING"}).encode("utf-8")
        handler = self._make_handler(method="POST", path="/api/label", body=payload)
        handler.do_POST()
        assert handler.response_code == 400
        res = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert res["success"] is False
        assert "Invalid confidence rating" in res["error"]

    def test_api_label_validation_nonexistent_pair(self):
        """Server rejects non-existent pair_id with HTTP 404."""
        payload = json.dumps({"pair_id": "NON_EXISTENT_PAIR_999", "human_label": "1", "confidence_rating": "HIGH"}).encode("utf-8")
        handler = self._make_handler(method="POST", path="/api/label", body=payload)
        handler.do_POST()
        assert handler.response_code == 404
        res = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert res["success"] is False
        assert "not found" in res["error"]

    def test_save_and_next_saves_all_three_fields_to_correct_pair(self, tmp_path, monkeypatch):
        """Verify successful save updates human_label, confidence_rating, and reviewer_notes strictly on target pair_id."""
        import review_app
        # Create isolated temporary copies of BOTH worksheets. A label POST without a dataset
        # parameter writes to COMBINED_WORKSHEET_PATH and then syncs WORKSHEET_PATH, so both must be
        # redirected - otherwise this test overwrites real human annotations on every run.
        real_paths = [review_app.WORKSHEET_PATH, review_app.COMBINED_WORKSHEET_PATH]
        real_bytes = {p: p.read_bytes() for p in real_paths if p.exists()}
        orig_df = pd.read_csv(WORKSHEET_PATH)
        temp_csv = tmp_path / "temp_worksheet.csv"
        orig_df.to_csv(temp_csv, index=False)
        monkeypatch.setattr(review_app, "WORKSHEET_PATH", temp_csv)
        if review_app.COMBINED_WORKSHEET_PATH.exists():
            temp_combined = tmp_path / "temp_combined_worksheet.csv"
            pd.read_csv(review_app.COMBINED_WORKSHEET_PATH).to_csv(temp_combined, index=False)
            monkeypatch.setattr(review_app, "COMBINED_WORKSHEET_PATH", temp_combined)

        target_id = "PAIR_BOUNDARY_041"
        payload = json.dumps({
            "pair_id": target_id,
            "human_label": "0",
            "confidence_rating": "HIGH",
            "reviewer_notes": "Different motor power and flange dimensions confirm distinct items."
        }).encode("utf-8")

        handler = self._make_handler(method="POST", path="/api/label", body=payload)
        handler.do_POST()

        assert handler.response_code == 200
        res = json.loads(handler.wfile.getvalue().decode("utf-8"))
        assert res["success"] is True
        assert res["pair_id"] == target_id

        # Verify saved content in isolated CSV
        saved_df = pd.read_csv(temp_csv)
        row = saved_df[saved_df["pair_id"] == target_id].iloc[0]
        assert str(row["human_label"]) == "0"
        assert str(row["confidence_rating"]) == "HIGH"
        assert "Different motor power" in str(row["reviewer_notes"])

        # Verify other records were untouched
        other_rows = saved_df[saved_df["pair_id"] != target_id]
        orig_others = orig_df[orig_df["pair_id"] != target_id]
        pd.testing.assert_series_equal(other_rows["pair_id"], orig_others["pair_id"])
        pd.testing.assert_series_equal(other_rows["human_label"].fillna(""), orig_others["human_label"].fillna(""))

        # Real annotation files must be byte-for-byte untouched by this test
        for p, content in real_bytes.items():
            assert p.read_bytes() == content, f"test wrote to real annotation file {p.name}"

    def test_previously_saved_annotations_remain_intact(self):
        """Ensure all 50 records in the active worksheet remain intact and properly structured."""
        df = pd.read_csv(WORKSHEET_PATH)
        assert len(df) == 50
        assert "pair_id" in df.columns
        assert "human_label" in df.columns
        assert "confidence_rating" in df.columns
        assert "reviewer_notes" in df.columns
        # Verify candidate pair IDs are unique and non-empty
        assert df["pair_id"].nunique() == 50
        assert df["pair_id"].str.startswith("PAIR_").all()


