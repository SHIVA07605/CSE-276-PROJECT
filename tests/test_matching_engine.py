"""
Unit tests for PrecisionMatchingEngine and conflict-aware clustering safeguards.
"""

import pytest
import pandas as pd
from src.matching_engine import CategoryAttributeExtractor, PrecisionMatchingEngine
from src.clustering import conflict_aware_clustering, cluster_connected_components


class TestCategoryAttributeExtractor:
    def test_cable_attribute_extraction(self):
        text = "XLPE INSULATED ARMOURED POWER CABLE 1.1 KV 3.5 CORE 120 SQMM COPPER CONDUCTOR IS:7098"
        attrs = CategoryAttributeExtractor.extract_attributes(text)
        assert attrs["category"] == "CABLES"
        assert attrs["cable_voltage"] == "1.1KV"
        assert attrs["cable_cores"] == "3.5"
        assert attrs["conductor_size_sqmm"] == 120.0
        assert attrs["conductor_material"] == "COPPER"
        assert attrs["cable_insulation"] == "XLPE"
        assert attrs["cable_armouring"] == "ARMOURED"
        assert "IS:7098" in attrs["standards"]

    def test_valve_attribute_extraction(self):
        text = "API 6D BALL VALVE 2 INCH CLASS 150 FLANGED SS316 WCB BODY"
        attrs = CategoryAttributeExtractor.extract_attributes(text)
        assert attrs["category"] == "VALVES"
        assert attrs["valve_type"] == "BALL_VALVE"
        assert attrs["end_connection"] == "FLANGED"
        assert attrs["dimensions"]["diameter_inch"] == "2"
        assert attrs["dimensions"]["pressure_class"] == "150#"
        assert "API 6D" in attrs["standards"]
        assert attrs["material_grade"] in ["SS316", "WCB"]

    def test_pump_attribute_extraction(self):
        text = "CENTRIFUGAL WATER PUMP 60 M3/HR HEAD 45 METER MOTOR 15 KW API 610"
        attrs = CategoryAttributeExtractor.extract_attributes(text)
        assert attrs["category"] == "PUMPS"
        assert attrs["pump_type"] == "CENTRIFUGAL"
        assert attrs["flow_rate"] == "60M3/HR"
        assert "API 610" in attrs["standards"]

    def test_bearing_attribute_extraction(self):
        text = "DEEP GROOVE BALL BEARING 6205 C3 CLEARANCE"
        attrs = CategoryAttributeExtractor.extract_attributes(text)
        assert attrs["category"] == "BEARINGS"
        assert attrs["bearing_type"] == "DEEP_GROOVE_BALL"
        assert attrs["bearing_series"] == "6205"
        assert attrs["bearing_clearance"] == "C3"


class TestPrecisionMatchingEngine:
    @pytest.fixture
    def engine(self):
        return PrecisionMatchingEngine()

    def test_generic_identical_descriptions_not_equivalent(self, engine):
        """Rule 1: Never classify records as EQUIVALENT merely because descriptions are identical."""
        res = engine.evaluate_pair(
            "REC_1", "AIR COMPRESSOR",
            "REC_2", "AIR COMPRESSOR",
            pair_id="P_EXACT",
            lexical_sim=1.0,
            semantic_sim=1.0,
        )
        assert res.decision == "NEEDS_REVIEW"
        assert res.confidence in ["LOW", "MEDIUM"]
        assert len(res.conflicting_attributes) == 0

    def test_conveyor_belt_dimension_conflict(self, engine):
        """Rule: Mismatched physical dimension yields NOT_EQUIVALENT with HIGH confidence."""
        res = engine.evaluate_pair(
            "M1", "STEEL CORD CONVEYOR BELT 1200MM WIDTH",
            "M2", "STEEL CORD CONVEYOR BELT 1600MM WIDTH",
            pair_id="P_CONF",
            lexical_sim=0.92,
            semantic_sim=0.98,
        )
        assert res.decision == "NOT_EQUIVALENT"
        assert res.confidence == "HIGH"
        assert "diameter_mm" in res.conflicting_attributes
        assert res.conflicting_attributes["diameter_mm"] == (1200.0, 1600.0)

    def test_cable_specification_conflict(self, engine):
        """Rule: Mismatched cable core count or voltage yields NOT_EQUIVALENT."""
        res = engine.evaluate_pair(
            "M1", "XLPE CABLE 1.1 KV 3 CORE 120 SQMM",
            "M2", "XLPE CABLE 1.1 KV 4 CORE 120 SQMM",
            pair_id="P_CABLE_CONF",
            lexical_sim=0.95,
            semantic_sim=0.97,
        )
        assert res.decision == "NOT_EQUIVALENT"
        assert "cable_cores" in res.conflicting_attributes

    def test_confirmed_standard_equivalent_match(self, engine):
        """Rule: Confirmed governing standards + valve type + zero conflicts yields EQUIVALENT."""
        res = engine.evaluate_pair(
            "M1", "API 6D BALL VALVE FLANGED SS316",
            "M2", "BALL VALVE API 6D FLANGED SS316",
            pair_id="P_VALVE_MATCH",
            lexical_sim=0.88,
            semantic_sim=0.96,
        )
        assert res.decision == "EQUIVALENT"
        assert len(res.conflicting_attributes) == 0
        assert "standards" in res.matched_attributes or "valve_type" in res.matched_attributes


class TestClusteringSafeguards:
    def test_conflict_aware_blocks_incompatible_merge(self):
        """Verify that a confirmed conflict blocks edge merge."""
        pairs = pd.DataFrame([
            {"record_id_1": "REC_A", "record_id_2": "REC_B", "prediction": "1", "pair_id": "P1"},
            {"record_id_1": "REC_B", "record_id_2": "REC_C", "prediction": "0", "has_attribute_conflict": True, "pair_id": "P2"},
        ])
        clusters, audit = conflict_aware_clustering(pairs, prediction_col="prediction")
        assert clusters["REC_A"] == clusters["REC_B"]
        assert clusters["REC_B"] != clusters["REC_C"]

    def test_transitive_conflict_detection(self):
        """
        Verify detection and blocking of transitive conflicts:
        A matches B, B matches C, but A conflicts with C.
        """
        pairs = pd.DataFrame([
            {"record_id_1": "REC_A", "record_id_2": "REC_B", "prediction": "1", "pair_id": "P1", "has_attribute_conflict": False},
            {"record_id_1": "REC_B", "record_id_2": "REC_C", "prediction": "1", "pair_id": "P2", "has_attribute_conflict": False},
            {"record_id_1": "REC_A", "record_id_2": "REC_C", "prediction": "0", "pair_id": "P3", "has_attribute_conflict": True},
        ])
        clusters, audit = conflict_aware_clustering(pairs, prediction_col="prediction")

        # REC_A and REC_B are clustered
        assert clusters["REC_A"] == clusters["REC_B"]
        # REC_C must NOT be clustered with REC_A/REC_B due to transitive conflict with REC_A!
        assert clusters["REC_C"] != clusters["REC_A"]
        assert audit["transitive_conflicts_count"] == 1
        assert audit["blocked_merges_count"] == 1
        assert audit["transitive_conflicts"][0]["conflicting_pair"] == ("REC_A", "REC_C")

    def test_isolation_of_ambiguous_records(self):
        """Ambiguous pairs (UNCERTAIN / NEEDS_REVIEW) remain quarantined in singleton clusters."""
        pairs = pd.DataFrame([
            {"record_id_1": "REC_X", "record_id_2": "REC_Y", "prediction": "UNCERTAIN", "pair_id": "P_UNC"},
            {"record_id_1": "REC_W", "record_id_2": "REC_Z", "prediction": "NEEDS_REVIEW", "pair_id": "P_NR"},
        ])
        clusters, audit = conflict_aware_clustering(pairs, prediction_col="prediction", isolate_ambiguous=True)
        # All 4 records must remain in distinct singleton clusters
        assert clusters["REC_X"] != clusters["REC_Y"]
        assert clusters["REC_W"] != clusters["REC_Z"]
        assert audit["quarantined_ambiguous_pairs_count"] == 2

