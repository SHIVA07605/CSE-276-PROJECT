"""
Engineering Attribute Extraction Module.

Extracts structured technical attributes from cleaned material descriptions:
- Metric thread (e.g., M10, M12, M16)
- Nominal diameter / dimensions (e.g., 50MM, 2INCH, DN50, DN100)
- Material grade / specification (e.g., SS304, SS316, IS2062, GRADE 8.8, ASTM A106)
- Pressure class / rating (e.g., 150#, 300#, PN16, PN40, Class 150, 10000PSI)
- Pipe schedule (e.g., SCH 40, SCH 80)
- Voltage / electrical capacity (e.g., 415V, 11KV, 100KVA)

Disambiguation rules:
- Metric thread (M10) is NOT the same as diameter (10MM NB).
- Pressure class (150#) is NOT the same as pipe schedule (SCH 40).
- PN16 is NOT the same as CLASS 150 (though they are approximate equivalents).
- SS304 is NOT SS316 — grade codes are captured verbatim.
- If a specification is ambiguous, it is flagged via 'ambiguous_flags'.
"""

from typing import Dict, Any, Optional, List
import re


def extract_material_attributes(text: Optional[str]) -> Dict[str, Any]:
    """
    Extract structured engineering attributes from cleaned text using regular expressions.

    Target attributes:
    - metric_thread   : Metric thread designation (e.g., M10, M12, M24).
    - diameter        : Nominal bore or diameter (e.g., 50MM NB, 2INCH, DN50).
    - material_grade  : Material composition or standard (e.g., SS304, IS2062, ASTM A106).
    - pressure_class  : Pressure class or rating (e.g., 150#, PN16, Class 150).
    - schedule        : Pipe wall thickness schedule (e.g., SCH 40, SCH 80).
    - voltage         : Electrical voltage rating (e.g., 415V, 11KV).
    - capacity        : Power or capacity rating (e.g., 100KVA, 50KW, 5HP).
    - ambiguous_flags : List of tokens that could not be unambiguously classified.

    Args:
        text: Cleaned, normalized description text.

    Returns:
        Dictionary mapping attribute names to string values (or None / list).
    """
    attributes: Dict[str, Any] = {
        "metric_thread": None,
        "diameter": None,
        "material_grade": None,
        "pressure_class": None,
        "schedule": None,
        "voltage": None,
        "capacity": None,
        "ambiguous_flags": [],
    }

    if not text or not isinstance(text, str):
        return attributes

    s = text.upper().strip()

    # ------------------------------------------------------------------
    # 1. Pipe schedule — must be extracted BEFORE diameter to prevent
    #    schedule numbers being misidentified as diameters or pressure class.
    # ------------------------------------------------------------------
    sch_match = re.search(r"\b(SCH(?:EDULE)?\s*(?:10[S]?|20|40[S]?|80[S]?|120|160|STD|XS|XXS))\b", s)
    if sch_match:
        attributes["schedule"] = re.sub(r"\s+", " ", sch_match.group(1)).strip()

    # ------------------------------------------------------------------
    # 2. Metric thread — M followed by 1–3 digits (M10, M12, M24).
    #    Disambiguation: metric thread is distinct from diameter.
    # ------------------------------------------------------------------
    thread_match = re.search(r"\bM(\d{1,3})\b", s)
    if thread_match:
        attributes["metric_thread"] = f"M{thread_match.group(1)}"

    # ------------------------------------------------------------------
    # 3. Nominal Diameter — includes MM/INCH/NB/OD/DIA and DN notation.
    #    Examples: 50MM NB, 2INCH, DN50, DN100, 1/2INCH.
    # ------------------------------------------------------------------
    dn_match = re.search(r"\bDN\s*(\d{2,4})\b", s)
    if dn_match:
        attributes["diameter"] = f"DN{dn_match.group(1)}"
    else:
        dia_match = re.search(
            r"\b(\d+(?:\.\d+)?|\d+/\d+)\s*(MM|INCH)(?:\s*(NB|OD|DIA|DIAMETER))?\b", s
        )
        if dia_match:
            num = dia_match.group(1)
            unit = dia_match.group(2)
            qualifier = dia_match.group(3) or ""
            attributes["diameter"] = (
                f"{num}{unit} {qualifier}".strip() if qualifier else f"{num}{unit}"
            )

    # ------------------------------------------------------------------
    # 4. Material grade / standard — priority: more-specific first.
    # ------------------------------------------------------------------
    GRADE_PATTERNS: List[str] = [
        r"\bSS\s*316[LN]?\b",
        r"\bSS\s*304[LN]?\b",
        r"\bSS\s*904L\b",
        r"\b316[LN]?\b",
        r"\b304[LN]?\b",
        r"\bIS\s*2062(?:\s*GR(?:ADE)?\s*[A-E])?\b",
        r"\bIS\s*1239(?:-1)?\b",
        r"\bIS\s*3589\b",
        r"\bIS\s*\d{4}\b",
        r"\bASTM\s*A\s*\d+[A-Z]?\s*(?:GR(?:ADE)?\s*[A-Z])?\b",
        r"\bAPI\s*5[LC]\b",
        r"\bGRADE\s*8\.8\b",
        r"\bGRADE\s*10\.9\b",
        r"\bGRADE\s*[AB]\b",
        r"\bAISI\s*\d+\b",
        r"\bCARBON STEEL\b",
        r"\bMILD STEEL\b",
    ]
    for pat in GRADE_PATTERNS:
        gm = re.search(pat, s)
        if gm:
            attributes["material_grade"] = re.sub(r"\s+", "", gm.group(0)).strip()
            break

    # ------------------------------------------------------------------
    # 5. Pressure class / rating.
    # ------------------------------------------------------------------
    pressure_match = re.search(
        r"\b(150#|300#|600#|900#|1500#|PN10|PN16|PN25|PN40|PN64"
        r"|CLASS\s*150|CLASS\s*300|CLASS\s*600|\d+PSI|\d+BAR)(?:\b|$|\s)",
        s,
    )
    if pressure_match:
        attributes["pressure_class"] = re.sub(r"\s+", " ", pressure_match.group(1)).strip()

    # ------------------------------------------------------------------
    # 6. Electrical voltage (e.g. 415V, 11KV, 33KV, 230V)
    # ------------------------------------------------------------------
    volt_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(KV|V)\b", s)
    if volt_match:
        attributes["voltage"] = f"{volt_match.group(1)}{volt_match.group(2)}"

    # ------------------------------------------------------------------
    # 7. Electrical / mechanical capacity (e.g. 100KVA, 50KW, 5HP, 7.5KW)
    # ------------------------------------------------------------------
    cap_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(KVA|MVA|KW|MW|HP)\b", s)
    if cap_match:
        attributes["capacity"] = f"{cap_match.group(1)}{cap_match.group(2)}"

    # ------------------------------------------------------------------
    # 8. Ambiguity flagging — stray hash-numbers not captured as a class.
    # ------------------------------------------------------------------
    ambiguous: List[str] = []
    stray_hash = re.findall(r"\b(\d+)#", s)
    for token in stray_hash:
        if f"{token}#" not in (attributes.get("pressure_class") or ""):
            ambiguous.append(f"UNCLASSIFIED_RATING:{token}#")
    if ambiguous:
        attributes["ambiguous_flags"] = ambiguous

    return attributes
