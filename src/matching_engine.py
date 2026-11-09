"""
Precision-First Rule-Based & Hybrid Matching Engine.

Implements rigorous, category-aware attribute extraction, specification conflict detection,
and evidence-based decision rules for material master code harmonisation.

Core Principles:
1. NEVER classify two records as EQUIVALENT merely because descriptions are identical or semantically similar.
2. NEVER assume missing attributes match.
3. Apply category-specific critical specification requirements (Cables, Valves, Pumps, Bearings, Instruments).
4. Treat extracted attributes as evidence subject to validation, not unquestionable ground truth.
5. Produce full auditable evidence: Decision, Evidence, Conflicting attributes, Missing attributes, Similarity scores, Explanation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set
import re
import pandas as pd
import numpy as np


@dataclass
class MatchAssessment:
    """Structured decision output for a candidate pair."""
    pair_id: str
    record_id_1: str
    record_id_2: str
    decision: str  # 'EQUIVALENT', 'NOT_EQUIVALENT', 'NEEDS_REVIEW'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    evidence: List[str] = field(default_factory=list)
    conflicting_attributes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    missing_critical_attributes: List[str] = field(default_factory=list)
    matched_attributes: Dict[str, Any] = field(default_factory=dict)
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    category: str = "GENERAL"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuple values in conflicting_attributes to string representation for JSON/CSV serialization
        d["conflicting_attributes"] = {k: f"{v[0]} != {v[1]}" for k, v in self.conflicting_attributes.items()}
        return d


class CategoryAttributeExtractor:
    """Extracts domain-specific engineering attributes using high-precision regex rules."""

    @staticmethod
    def detect_category(text: str) -> str:
        """Detect equipment category from description text."""
        t = str(text).upper()
        if re.search(r"\b(CABLE|CABLES|XLPE|PVC|FRLS|CONDUCTOR|WIRE|WIRING)\b", t):
            return "CABLES"
        if re.search(r"\b(VALVE|VALVES|COCK|BALL VALVE|GATE VALVE|GLOBE VALVE|CHECK VALVE|BUTTERFLY VALVE)\b", t):
            return "VALVES"
        if re.search(r"\b(PUMP|PUMPS|PUMPING|CENTRIFUGAL PUMP|SUBMERSIBLE PUMP)\b", t):
            return "PUMPS"
        if re.search(r"\b(BEARING|BEARINGS|BALL BEARING|ROLLER BEARING)\b", t):
            return "BEARINGS"
        if re.search(r"\b(TRANSMITTER|GAUGE|CHROMATOGRAPH|STIRRER|FLOWMETER|ANALYZER|THERMOCOUPLE|INDICATOR)\b", t):
            return "INSTRUMENTS"
        if re.search(r"\b(PIPE|PIPES|TUBING|FLANGE|FLANGES|ELBOW|TEE|REDUCER|FITTING|FITTINGS)\b", t):
            return "PIPES_FITTINGS"
        if re.search(r"\b(TURBINE|COMPRESSOR|BLOWER|FAN)\b", t):
            return "COMPRESSORS_TURBINES"
        return "GENERAL"

    @classmethod
    def extract_attributes(cls, text: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Extract category-specific and general attributes from raw or cleaned text."""
        t = str(text).upper()
        cat = category or cls.detect_category(t)
        attrs: Dict[str, Any] = {
            "category": cat,
            "standards": cls._extract_standards(t),
            "material_grade": cls._extract_material_grade(t),
            "dimensions": cls._extract_dimensions(t),
        }

        if cat == "CABLES":
            attrs.update(cls._extract_cable_attributes(t))
        elif cat == "VALVES":
            attrs.update(cls._extract_valve_attributes(t))
        elif cat == "PUMPS":
            attrs.update(cls._extract_pump_attributes(t))
        elif cat == "BEARINGS":
            attrs.update(cls._extract_bearing_attributes(t))
        elif cat == "INSTRUMENTS":
            attrs.update(cls._extract_instrument_attributes(t))
        elif cat == "PIPES_FITTINGS":
            attrs.update(cls._extract_pipe_attributes(t))

        return attrs

    @staticmethod
    def _extract_standards(t: str) -> List[str]:
        standards = []
        for pat in [
            r"\b(API\s*(?:6D|610|600|598|520|526))\b",
            r"\b(IS\s*[:\s\-]?\s*(?:7098|1554|2062|1239|3589|6455))\b",
            r"\b(ASME\s*B(?:16\.34|16\.5|16\.10|31\.3))\b",
            r"\b(ASTM\s*[A-Z]?\s*\d+)\b",
            r"\b(ISO\s*\d+)\b",
            r"\b(BS\s*\d+)\b",
        ]:
            matches = re.findall(pat, t)
            standards.extend([re.sub(r"\s+", " ", m.strip()) for m in matches])
        return sorted(list(set(standards)))

    @staticmethod
    def _extract_material_grade(t: str) -> Optional[str]:
        # Stainless steel, carbon steel, duplex, alloys
        m = re.search(r"\b(SS\s*316L?|SS\s*304L?|SS\s*321|SS\s*410|AISI\s*316|AISI\s*304|A105|A216\s*WCB|A351\s*CF8M?|WCB|CF8M|IS\s*2062(?:\s*GR\s*[AB])?|MONEL|HASTELLOY|INCONEL|BRONZE|BRASS|CAST\s*IRON|DUCTILE\s*IRON)\b", t)
        if m:
            val = re.sub(r"\s+", "", m.group(1))
            return val
        if "STAINLESS STEEL" in t or "STAINLESS" in t:
            return "STAINLESS_STEEL"
        if "CARBON STEEL" in t:
            return "CARBON_STEEL"
        return None

    @staticmethod
    def _extract_dimensions(t: str) -> Dict[str, Any]:
        dims: Dict[str, Any] = {}
        # Diameter / Size in MM, INCH, DN
        dm = re.search(r"\b(?:DN\s*|SIZE\s*)?(\d+(?:\.\d+)?)\s*(?:MM|MILLIMETER)\b", t)
        if dm:
            dims["diameter_mm"] = float(dm.group(1))

        d_inch = re.search(r"\b(\d+(?:\s*[\/\-]\s*\d+)?|\d+\.\d+)\s*(?:INCH|IN|\"|NB)\b", t)
        if d_inch:
            dims["diameter_inch"] = d_inch.group(1).replace(" ", "")

        # Pressure Class
        pc = re.search(r"\b(?:CLASS\s*|#\s*)?(150#?|300#?|600#?|900#?|1500#?|2500#?|PN\s*10|PN\s*16|PN\s*25|PN\s*40|PN\s*64)\b", t)
        if pc:
            val = pc.group(1).replace(" ", "")
            if val.isdigit():
                val = f"{val}#"
            dims["pressure_class"] = val

        # Pipe Schedule
        sch = re.search(r"\b(?:SCH|SCHEDULE)\s*[:\-]?\s*(10|20|30|40|60|80|120|160|XXS|XS|STD)\b", t)
        if sch:
            dims["schedule"] = f"SCH{sch.group(1)}"

        return dims

    @classmethod
    def _extract_cable_attributes(cls, t: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        # Voltage: 1.1KV, 6.6KV, 11KV, 33KV, 415V
        vm = re.search(r"\b(1\.1\s*KV|6\.6\s*KV|11\s*KV|33\s*KV|415\s*V|650\s*V|1100\s*V)\b", t)
        if vm:
            res["cable_voltage"] = re.sub(r"\s+", "", vm.group(1))

        # Core count: 1C, 2C, 3C, 3.5C, 4C, 1 CORE, 2 CORE, etc.
        cm = re.search(r"\b(1|2|3|3\.5|4|5|7|10|12|14|19|24)\s*(?:C\b|CORE\b)", t)
        if cm:
            res["cable_cores"] = cm.group(1)

        # Conductor size: e.g. 1.5 SQMM, 2.5 SQMM, 4 SQMM, 16 SQMM, 70 SQMM
        sm = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:SQ\.?\s*MM|SQMM|MM2)\b", t)
        if sm:
            res["conductor_size_sqmm"] = float(sm.group(1))

        # Conductor material: COPPER vs ALUMINIUM
        if re.search(r"\b(COPPER|CU)\b", t):
            res["conductor_material"] = "COPPER"
        elif re.search(r"\b(ALUMINIUM|ALUMINUM|AL)\b", t):
            res["conductor_material"] = "ALUMINIUM"

        # Insulation: XLPE vs PVC
        if "XLPE" in t:
            res["cable_insulation"] = "XLPE"
        elif "PVC" in t:
            res["cable_insulation"] = "PVC"

        # Armoured vs Unarmoured
        if "UNARMOURED" in t or "UN-ARMOURED" in t:
            res["cable_armouring"] = "UNARMOURED"
        elif "ARMOURED" in t or "ARMORED" in t:
            res["cable_armouring"] = "ARMOURED"

        return res

    @classmethod
    def _extract_valve_attributes(cls, t: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        # Valve specific type
        if "BALL VALVE" in t or "BALL VALVES" in t:
            res["valve_type"] = "BALL_VALVE"
        elif "GATE VALVE" in t or "GATE VALVES" in t:
            res["valve_type"] = "GATE_VALVE"
        elif "GLOBE VALVE" in t or "GLOBE VALVES" in t:
            res["valve_type"] = "GLOBE_VALVE"
        elif "CHECK VALVE" in t or "NON RETURN VALVE" in t or "NRV" in t or "SWING CHECK" in t:
            res["valve_type"] = "CHECK_VALVE"
        elif "BUTTERFLY VALVE" in t:
            res["valve_type"] = "BUTTERFLY_VALVE"
        elif "NEEDLE VALVE" in t:
            res["valve_type"] = "NEEDLE_VALVE"
        elif "PLUG VALVE" in t:
            res["valve_type"] = "PLUG_VALVE"
        elif "CONTROL VALVE" in t:
            res["valve_type"] = "CONTROL_VALVE"

        # End connection
        if re.search(r"\b(FLANGED|FLG|FLANGE END)\b", t):
            res["end_connection"] = "FLANGED"
        elif re.search(r"\b(SCREWED|THREADED|NPT|BSP)\b", t):
            res["end_connection"] = "THREADED"
        elif re.search(r"\b(SOCKET\s*WELD|SW)\b", t):
            res["end_connection"] = "SOCKET_WELD"
        elif re.search(r"\b(BUTT\s*WELD|BW)\b", t):
            res["end_connection"] = "BUTT_WELD"

        # Operation
        if re.search(r"\b(GEAR\s*OPERATED|LEVER\s*OPERATED|ACTUATED|PNEUMATIC|MOTORIZED)\b", t):
            res["operation"] = re.search(r"\b(GEAR\s*OPERATED|LEVER\s*OPERATED|ACTUATED|PNEUMATIC|MOTORIZED)\b", t).group(1)

        return res

    @classmethod
    def _extract_pump_attributes(cls, t: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        if "CENTRIFUGAL" in t:
            res["pump_type"] = "CENTRIFUGAL"
        elif "SUBMERSIBLE" in t:
            res["pump_type"] = "SUBMERSIBLE"
        elif "RECIPROCATING" in t:
            res["pump_type"] = "RECIPROCATING"
        elif "GEAR PUMP" in t:
            res["pump_type"] = "GEAR"
        elif "DIAPHRAGM" in t:
            res["pump_type"] = "DIAPHRAGM"

        # Flow / Capacity
        flow_m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:M3\/HR|M3\/H|KL\/HR|KLPH|LPM|GPM)\b", t)
        if flow_m:
            res["flow_rate"] = flow_m.group(0).replace(" ", "")

        # Head / Pressure
        head_m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:M(?:ETER)?\s*HEAD|MWC|BAR)\b", t)
        if head_m:
            res["pump_head"] = head_m.group(0).replace(" ", "")

        # Motor Rating / Power
        power_m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:HP|KW|MW)\b", t)
        if power_m:
            res["power_rating"] = power_m.group(0).replace(" ", "")

        return res

    @classmethod
    def _extract_bearing_attributes(cls, t: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        # Bearing series: ISO 4-digit code (e.g. 6205, 6309, 7208) or 3-digit (608)
        series_m = re.search(r"\b([1-8][0-4]\d{2}|608|609|NU\s*\d{3,4}|NJ\s*\d{3,4}|NUP\s*\d{3,4})\b", t)
        if series_m:
            res["bearing_series"] = series_m.group(1).replace(" ", "")

        if "DEEP GROOVE" in t:
            res["bearing_type"] = "DEEP_GROOVE_BALL"
        elif "SPHERICAL ROLLER" in t:
            res["bearing_type"] = "SPHERICAL_ROLLER"
        elif "CYLINDRICAL ROLLER" in t:
            res["bearing_type"] = "CYLINDRICAL_ROLLER"
        elif "TAPER ROLLER" in t:
            res["bearing_type"] = "TAPER_ROLLER"
        elif "THRUST" in t:
            res["bearing_type"] = "THRUST_BEARING"

        # Clearance
        clr_m = re.search(r"\b(C2|C3|C4|C5)\b", t)
        if clr_m:
            res["bearing_clearance"] = clr_m.group(1)

        return res

    @classmethod
    def _extract_instrument_attributes(cls, t: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        if "MAGNETIC STIRRER" in t:
            res["instrument_type"] = "MAGNETIC_STIRRER"
            if "HOT PLATE" in t or "HOTPLATE" in t:
                res["has_hot_plate"] = True
        elif "CHROMATOGRAPH" in t:
            res["instrument_type"] = "GAS_CHROMATOGRAPH"
        elif "PRESSURE TRANSMITTER" in t or "DIFF PRESSURE TRANSMITTER" in t or "DPT" in t:
            res["instrument_type"] = "PRESSURE_TRANSMITTER"
        elif "TEMPERATURE TRANSMITTER" in t or "RTD" in t or "THERMOCOUPLE" in t:
            res["instrument_type"] = "TEMPERATURE_SENSOR"
        elif "FLOW METER" in t or "FLOWMETER" in t:
            res["instrument_type"] = "FLOW_METER"
        elif "LEVEL TRANSMITTER" in t or "LEVEL GAUGE" in t:
            res["instrument_type"] = "LEVEL_INSTRUMENT"

        return res

    @classmethod
    def _extract_pipe_attributes(cls, t: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        if "SEAMLESS" in t:
            res["construction"] = "SEAMLESS"
        elif "ERW" in t or "WELDED" in t or "SAW" in t:
            res["construction"] = "WELDED"
        return res


class PrecisionMatchingEngine:
    """
    Evaluates candidate material pairs with domain-aware, precision-first rules.
    Outputs EQUIVALENT, NOT_EQUIVALENT, or NEEDS_REVIEW with full evidence trails.
    """

    CRITICAL_SPECS_PER_CATEGORY = {
        "CABLES": ["cable_voltage", "cable_cores", "conductor_size_sqmm", "conductor_material", "cable_insulation"],
        "VALVES": ["valve_type", "pressure_class", "diameter", "material_grade"],
        "PUMPS": ["pump_type", "flow_rate", "material_grade", "power_rating"],
        "BEARINGS": ["bearing_series", "bearing_type"],
        "INSTRUMENTS": ["instrument_type"],
        "PIPES_FITTINGS": ["diameter", "schedule", "material_grade"],
        "GENERAL": ["material_grade", "dimensions"],
    }

    def __init__(self, high_similarity_threshold: float = 0.75):
        self.extractor = CategoryAttributeExtractor()
        self.high_similarity_threshold = high_similarity_threshold

    def evaluate_pair(
        self,
        record_id_1: str,
        desc_1: str,
        record_id_2: str,
        desc_2: str,
        pair_id: str = "",
        lexical_sim: Optional[float] = None,
        semantic_sim: Optional[float] = None,
        raw_desc_1: str = "",
        raw_desc_2: str = "",
        enterprise_1: str = "",
        enterprise_2: str = "",
    ) -> MatchAssessment:
        """
        Perform complete multi-modal, evidence-based compatibility evaluation.
        """
        d1 = str(desc_1).strip().upper()
        d2 = str(desc_2).strip().upper()
        full_text_1 = f"{raw_desc_1} {desc_1}".upper()
        full_text_2 = f"{raw_desc_2} {desc_2}".upper()

        # 1. Category Detection
        cat1 = self.extractor.detect_category(full_text_1)
        cat2 = self.extractor.detect_category(full_text_2)

        # 2. Extract detailed technical attributes
        attrs1 = self.extractor.extract_attributes(full_text_1, cat1)
        attrs2 = self.extractor.extract_attributes(full_text_2, cat2)

        # Merge dimensional attributes
        dims1 = attrs1.get("dimensions", {})
        dims2 = attrs2.get("dimensions", {})

        # 3. Detect Conflicting Attributes
        conflicts: Dict[str, Tuple[Any, Any]] = {}
        matched: Dict[str, Any] = {}
        evidence: List[str] = []

        # Check Category Conflict
        if cat1 != "GENERAL" and cat2 != "GENERAL" and cat1 != cat2:
            conflicts["equipment_category"] = (cat1, cat2)
            evidence.append(f"Fundamental equipment category mismatch: '{cat1}' vs '{cat2}'.")

        # Check Dimensions
        if "diameter_mm" in dims1 and "diameter_mm" in dims2:
            if abs(dims1["diameter_mm"] - dims2["diameter_mm"]) > 1.0:
                conflicts["diameter_mm"] = (dims1["diameter_mm"], dims2["diameter_mm"])
                evidence.append(f"Incompatible diameter: {dims1['diameter_mm']}MM != {dims2['diameter_mm']}MM.")
            else:
                matched["diameter_mm"] = dims1["diameter_mm"]
                evidence.append(f"Confirmed matching diameter: {dims1['diameter_mm']}MM.")

        if "diameter_inch" in dims1 and "diameter_inch" in dims2:
            if dims1["diameter_inch"] != dims2["diameter_inch"]:
                conflicts["diameter_inch"] = (dims1["diameter_inch"], dims2["diameter_inch"])
                evidence.append(f"Incompatible diameter: {dims1['diameter_inch']} INCH != {dims2['diameter_inch']} INCH.")
            else:
                matched["diameter_inch"] = dims1["diameter_inch"]
                evidence.append(f"Confirmed matching diameter: {dims1['diameter_inch']} INCH.")

        if "pressure_class" in dims1 and "pressure_class" in dims2:
            if dims1["pressure_class"] != dims2["pressure_class"]:
                conflicts["pressure_class"] = (dims1["pressure_class"], dims2["pressure_class"])
                evidence.append(f"Incompatible pressure class: {dims1['pressure_class']} != {dims2['pressure_class']}.")
            else:
                matched["pressure_class"] = dims1["pressure_class"]
                evidence.append(f"Confirmed matching pressure class: {dims1['pressure_class']}.")

        if "schedule" in dims1 and "schedule" in dims2:
            if dims1["schedule"] != dims2["schedule"]:
                conflicts["schedule"] = (dims1["schedule"], dims2["schedule"])
                evidence.append(f"Incompatible pipe schedule: {dims1['schedule']} != {dims2['schedule']}.")
            else:
                matched["schedule"] = dims1["schedule"]
                evidence.append(f"Confirmed matching schedule: {dims1['schedule']}.")

        # Check Material Grade
        g1 = attrs1.get("material_grade")
        g2 = attrs2.get("material_grade")
        if g1 and g2:
            if g1 != g2:
                # Distinguish incompatible grades (e.g. SS304 vs SS316, or CS vs SS)
                conflicts["material_grade"] = (g1, g2)
                evidence.append(f"Incompatible material metallurgy: '{g1}' vs '{g2}'.")
            else:
                matched["material_grade"] = g1
                evidence.append(f"Confirmed matching material grade: '{g1}'.")

        # Category-Specific Comparisons
        active_cat = cat1 if cat1 == cat2 else ("GENERAL" if cat1 == "GENERAL" or cat2 == "GENERAL" else cat1)

        # CABLES
        if active_cat == "CABLES":
            for field in ["cable_voltage", "cable_cores", "conductor_size_sqmm", "conductor_material", "cable_insulation", "cable_armouring"]:
                v1, v2 = attrs1.get(field), attrs2.get(field)
                if v1 is not None and v2 is not None:
                    if v1 != v2:
                        conflicts[field] = (v1, v2)
                        evidence.append(f"Cable specification mismatch: {field} '{v1}' != '{v2}'.")
                    else:
                        matched[field] = v1
                        evidence.append(f"Confirmed matching {field}: '{v1}'.")

        # VALVES
        elif active_cat == "VALVES":
            for field in ["valve_type", "end_connection", "operation"]:
                v1, v2 = attrs1.get(field), attrs2.get(field)
                if v1 is not None and v2 is not None:
                    if v1 != v2:
                        conflicts[field] = (v1, v2)
                        evidence.append(f"Valve specification mismatch: {field} '{v1}' != '{v2}'.")
                    else:
                        matched[field] = v1
                        evidence.append(f"Confirmed matching {field}: '{v1}'.")

        # PUMPS
        elif active_cat == "PUMPS":
            for field in ["pump_type", "flow_rate", "pump_head", "power_rating"]:
                v1, v2 = attrs1.get(field), attrs2.get(field)
                if v1 is not None and v2 is not None:
                    if v1 != v2:
                        conflicts[field] = (v1, v2)
                        evidence.append(f"Pump specification mismatch: {field} '{v1}' != '{v2}'.")
                    else:
                        matched[field] = v1
                        evidence.append(f"Confirmed matching {field}: '{v1}'.")

        # BEARINGS
        elif active_cat == "BEARINGS":
            for field in ["bearing_series", "bearing_type", "bearing_clearance"]:
                v1, v2 = attrs1.get(field), attrs2.get(field)
                if v1 is not None and v2 is not None:
                    if v1 != v2:
                        conflicts[field] = (v1, v2)
                        evidence.append(f"Bearing specification mismatch: {field} '{v1}' != '{v2}'.")
                    else:
                        matched[field] = v1
                        evidence.append(f"Confirmed matching {field}: '{v1}'.")

        # INSTRUMENTS
        elif active_cat == "INSTRUMENTS":
            for field in ["instrument_type", "has_hot_plate"]:
                v1, v2 = attrs1.get(field), attrs2.get(field)
                if v1 is not None and v2 is not None:
                    if v1 != v2:
                        conflicts[field] = (v1, v2)
                        evidence.append(f"Instrument specification mismatch: {field} '{v1}' != '{v2}'.")
                    else:
                        matched[field] = v1
                        evidence.append(f"Confirmed matching {field}: '{v1}'.")

        # Standards Comparison
        std1 = set(attrs1.get("standards", []))
        std2 = set(attrs2.get("standards", []))
        common_stds = std1 & std2
        diff_stds = (std1 ^ std2)
        if common_stds:
            matched["standards"] = list(common_stds)
            evidence.append(f"Confirmed matching governing standard(s): {', '.join(common_stds)}.")

        # Missing Critical Attributes Analysis
        crit_keys = self.CRITICAL_SPECS_PER_CATEGORY.get(active_cat, ["material_grade", "dimensions"])
        missing_critical: List[str] = []
        for k in crit_keys:
            has_1 = (k in attrs1 or k in dims1 or (k == "dimensions" and bool(dims1)) or (k == "diameter" and ("diameter_mm" in dims1 or "diameter_inch" in dims1)))
            has_2 = (k in attrs2 or k in dims2 or (k == "dimensions" and bool(dims2)) or (k == "diameter" and ("diameter_mm" in dims2 or "diameter_inch" in dims2)))
            if not has_1 and not has_2:
                missing_critical.append(f"Both records lack '{k}'")
            elif not has_1:
                missing_critical.append(f"Record 1 lacks '{k}' (asymmetric specification)")
            elif not has_2:
                missing_critical.append(f"Record 2 lacks '{k}' (asymmetric specification)")

        # Similarity scores tracking
        sim_scores = {}
        if lexical_sim is not None:
            sim_scores["lexical_similarity"] = round(float(lexical_sim), 4)
        if semantic_sim is not None:
            sim_scores["semantic_cosine_similarity"] = round(float(semantic_sim), 4)

        # 4. DECISION LOGIC
        # RULE 1: Confirmed attribute conflicts -> NOT_EQUIVALENT
        if len(conflicts) > 0:
            decision = "NOT_EQUIVALENT"
            confidence = "HIGH"
            explanation = (
                f"Definite specification conflict detected across {len(conflicts)} parameter(s): "
                f"{', '.join(conflicts.keys())}. Items are physically or operationally incompatible."
            )

        # RULE 2: Different specific categories -> NOT_EQUIVALENT
        elif cat1 != "GENERAL" and cat2 != "GENERAL" and cat1 != cat2:
            decision = "NOT_EQUIVALENT"
            confidence = "HIGH"
            explanation = f"Cross-domain equipment mismatch: '{cat1}' cannot substitute for '{cat2}'."

        # RULE 3: Exact description match but missing critical specifications -> NEEDS_REVIEW
        # NEVER classify two records as equivalent merely because descriptions are identical or semantically similar.
        elif len(missing_critical) > 0:
            # Check if there is enough evidence to support equivalence despite missing some secondary specs
            # e.g., both explicitly specify "API 6D BALL VALVE" and have identical valve_type and standard
            has_strong_anchor = (
                len(matched) >= 2
                and (
                    "standards" in matched
                    or "bearing_series" in matched
                    or ("valve_type" in matched and "material_grade" in matched)
                    or ("cable_voltage" in matched and "cable_insulation" in matched)
                )
            )

            if has_strong_anchor and (semantic_sim is None or semantic_sim >= 0.85):
                decision = "EQUIVALENT"
                confidence = "MEDIUM"
                explanation = (
                    f"Strong governing specification alignment confirmed ({', '.join(matched.keys())}). "
                    f"Some operational ratings unstated ({len(missing_critical)} missing: {missing_critical[0]}), "
                    "but core engineering standard warrants equivalence."
                )
            else:
                decision = "NEEDS_REVIEW"
                confidence = "MEDIUM" if (semantic_sim and semantic_sim >= 0.70) else "LOW"
                explanation = (
                    f"Descriptions are semantically related but lack critical engineering specifications "
                    f"({len(missing_critical)} missing parameters). Functional substitution cannot be proven "
                    "without human inspection of full equipment datasheets."
                )

        # RULE 4: All critical attributes present, match confirmed, 0 conflicts -> EQUIVALENT
        elif len(matched) >= 1 and len(conflicts) == 0:
            decision = "EQUIVALENT"
            confidence = "HIGH"
            explanation = (
                f"Full engineering compatibility established: confirmed matching specifications ({', '.join(matched.keys())}) "
                "with zero parameter conflicts."
            )

        # RULE 5: High similarity generic descriptions lacking specs -> NEEDS_REVIEW
        else:
            decision = "NEEDS_REVIEW"
            confidence = "LOW"
            explanation = (
                "Insufficient technical detail in catalog text. Descriptions represent generic commodity titles "
                "without verifiable part numbers, ratings, or dimensions."
            )

        return MatchAssessment(
            pair_id=pair_id,
            record_id_1=record_id_1,
            record_id_2=record_id_2,
            decision=decision,
            confidence=confidence,
            evidence=evidence,
            conflicting_attributes=conflicts,
            missing_critical_attributes=missing_critical,
            matched_attributes=matched,
            similarity_scores=sim_scores,
            explanation=explanation,
            category=active_cat,
        )

    def evaluate_dataset(self, df_pairs: pd.DataFrame) -> List[MatchAssessment]:
        """Evaluate a DataFrame of candidate pairs and return assessments."""
        assessments: List[MatchAssessment] = []
        for _, row in df_pairs.iterrows():
            pid = str(row.get("pair_id", ""))
            id1 = str(row.get("corpus_id_1", row.get("record_id_1", "")))
            id2 = str(row.get("corpus_id_2", row.get("record_id_2", "")))
            d1 = str(row.get("cleaned_description_1", row.get("raw_description_1", "")))
            d2 = str(row.get("cleaned_description_2", row.get("raw_description_2", "")))
            raw1 = str(row.get("raw_description_1", ""))
            raw2 = str(row.get("raw_description_2", ""))
            e1 = str(row.get("enterprise_1", ""))
            e2 = str(row.get("enterprise_2", ""))
            lex = row.get("lexical_similarity")
            sem = row.get("semantic_cosine_similarity")

            assessment = self.evaluate_pair(
                record_id_1=id1,
                desc_1=d1,
                record_id_2=id2,
                desc_2=d2,
                pair_id=pid,
                lexical_sim=lex,
                semantic_sim=sem,
                raw_desc_1=raw1,
                raw_desc_2=raw2,
                enterprise_1=e1,
                enterprise_2=e2,
            )
            assessments.append(assessment)
        return assessments
