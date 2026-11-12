"""
Text Cleaning, Normalisation, and Standardisation Module.

Implements reusable text preprocessing functions for public sector engineering
and procurement material descriptions:
- Casing standardization (uppercase)
- Whitespace stripping and collapsing
- Punctuation and separator normalisation
- Acronym and engineering abbreviation expansion
- Engineering unit standardization (mm, inch, etc.)
- Procurement boilerplate stripping
- Technical token and specification preservation (PIPE, FLANGE, SE, ASTM, etc.)
"""

from typing import Optional, List, Tuple
import re
import pandas as pd


# Common engineering abbreviations mapped to canonical expansions
# Note: SS is handled dynamically so grades like SS304 remain SS304 while standalone SS -> STAINLESS STEEL
ABBREVIATION_MAP: List[Tuple[str, str]] = [
    (r"\bCS\b", "CARBON STEEL"),
    (r"\bMS\b", "MILD STEEL"),
    (r"\bGI\b", "GALVANISED IRON"),
    (r"\bCI\b", "CAST IRON"),
    (r"\bDI\b", "DUCTILE IRON"),
    (r"\bHEX\b", "HEXAGONAL"),
    (r"\bDIA\b", "DIAMETER"),
    (r"\bTHK\b", "THICKNESS"),
    (r"\bLG\b", "LONG"),
    (r"\bQTY\b", "QUANTITY"),
    (r"\bNO\b", "NUMBER"),
    (r"\bNOS\b", "NUMBERS"),
    (r"\bPCS\b", "PIECES"),
    (r"\bMTR\b", "METER"),
]

# Engineering unit normalisations (e.g., 10 mm -> 10MM, 2 inch -> 2INCH, 2" -> 2INCH)
UNIT_NORMALISATIONS: List[Tuple[str, str]] = [
    (r"(\d+(?:\.\d+)?)\s*(?:MM|M\.M\.)\b", r"\1MM"),
    (r"(\d+(?:\.\d+)?)\s*(?:CM|C\.M\.)\b", r"\1CM"),
    (r"(\d+(?:\.\d+)?)\s*(?:MTR|METERS?|M)\b", r"\1M"),
    (r"(\d+(?:\.\d+)?)\s*(?:INCH|INCHES|IN\.?)\b", r"\1INCH"),
    (r"(\d+(?:\.\d+)?)\s*\"", r"\1INCH"),
    (r"(\d+/\d+)\s*(?:INCH|INCHES|IN\.?|\")", r"\1INCH"),
    (r"(\d+(?:\.\d+)?)\s*(?:KG|KGS|KILOGRAMS?)\b", r"\1KG"),
    (r"(\d+)\s*#\b", r"\1#"),
    (r"(\d+)\s*(PSI|BAR)\b", r"\1\2"),
]

# Leading procurement boilerplate regex patterns (stripped only from description headers)
LEADING_BOILERPLATE_PATTERNS: List[str] = [
    r"^\s*(?:CONSOLIDATED\s+)?PROCUREMENT\s+OF\s+",
    r"^\s*SUPPLY\s+OF\s+",
    r"^\s*HIRING\s+(?:THE\s+)?SERVICES\s+OF\s+",
    r"^\s*HIRING\s+SERVICES\s+FOR\s+",
    r"^\s*HIRING\s+OF\s+",
    r"^\s*ANNUAL\s+MAINTENANCE\s+CONTRACT\s+(?:\(AMC\)\s+)?(?:FOR\s+)?",
    r"^\s*AMC\s+FOR\s+",
    r"^\s*TENDER\s+FOR\s+(?:THE\s+)?(?:SUPPLY\s+OF\s+)?",
    r"^\s*RATE\s+CONTRACT\s+FOR\s+",
    r"^\s*INVITATION\s+FOR\s+BIDS?\s+(?:\(IFB\)\s+)?FOR\s+",
    r"^\s*NOTICE\s+INVITING\s+TENDER\s+(?:\(NIT\)\s+)?(?:FOR\s+)?",
    r"^\s*NIT\s+FOR\s+",
]

# Administrative filler boilerplate phrases that do not describe the physical material
ADMIN_FILLER_PATTERNS: List[str] = [
    r"\bFOR\s+A\s+PERIOD\s+OF\s+\d+\s*(?:\([A-Z\s]+\)\s*)?(?:YEARS?|MONTHS?|DAYS?)\b",
    r"\bEXTENDABLE\s+BY\s+\d+\s*(?:\([A-Z\s]+\)\s*)?(?:YEARS?|MONTHS?|DAYS?)\b",
    r"\bWITH\s+(?:ALL\s+)?STANDARD\s+FITTINGS\s+(?:AND|&)\s+ACCESSORIES\b",
    r"\bREADILY\s+AVAILABLE\b",
    r"\bBRAND\s+NEW\b",
    r"\b1\s*\(\s*ONE\s*\)\s*NO\.?\b",
    r"\b\d+\s*\([A-Z\s]+\)\s*NOS?\.?\b",
    r"\bLATEST\s+EDITION\b",
    r"\bAS\s+PER\s+(?:ATTACHED\s+|DETAILED\s+)?SPECIFICATIONS?\b",
]


def normalise_text(text: Optional[str], remove_boilerplate: bool = True) -> str:
    """
    Standardize raw procurement and material descriptions according to engineering rules.

    Operations applied:
    1. Null check and typecast to string.
    2. Convert to uppercase for case-insensitive matching.
    3. Normalize unicode separators, quotes, dashes, and duplicate punctuation.
    4. Normalize acronym dots in domain tokens (e.g. S.S. -> SS, C.S. -> CS).
    5. Standardize material grades (e.g. S.S. 304, SS 304 -> SS304).
    6. Standardize units and dimensions (e.g. 10 mm -> 10MM, 2 inch -> 2INCH).
    7. Expand domain abbreviations (standalone SS -> STAINLESS STEEL, HEX -> HEXAGONAL).
    8. Optionally remove irrelevant administrative procurement boilerplate.
    9. Preserve critical technical nouns, codes, and attributes (e.g. PIPE, FLANGE, SE).
    10. Collapse redundant whitespace and trim.

    Args:
        text: Raw material description string.
        remove_boilerplate: Whether to strip administrative procurement boilerplate.

    Returns:
        Cleaned, normalized description string.
    """
    if text is None or pd.isna(text):
        return ""

    s = str(text).upper().strip()
    if not s:
        return ""

    # 1. Normalize unicode quotes, dashes, and bullets
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    s = re.sub(r"[\u2013\u2014\u2212]", "-", s)
    s = re.sub(r"[\uf0b7\u2022\u25cf\u2023\u2043\ufeff]", " ", s)

    # 2. Normalize acronym dots (e.g. S.S. -> SS, C.S. -> CS, M.M. -> MM)
    s = re.sub(r"\bS\.\s*S\.", "SS", s)
    s = re.sub(r"\bC\.\s*S\.", "CS", s)
    s = re.sub(r"\bM\.\s*S\.", "MS", s)
    s = re.sub(r"\bG\.\s*I\.", "GI", s)
    s = re.sub(r"\bC\.\s*I\.", "CI", s)
    s = re.sub(r"\bD\.\s*I\.", "DI", s)
    s = re.sub(r"\bM\.\s*M\.", "MM", s)
    s = re.sub(r"\bC\.\s*M\.", "CM", s)
    s = re.sub(r"\bN\.\s*B\.", "NB", s)
    s = re.sub(r"\bO\.\s*D\.", "OD", s)
    s = re.sub(r"\bI\.\s*D\.", "ID", s)

    # 3. Standardize Stainless Steel grades before expanding standalone SS:
    # E.g. SS 304, SS-304, SS 316, SS 316L -> SS304, SS316, SS316L
    s = re.sub(r"\bSS[\s\-_]*(3\d\d[A-Z]?|4\d\d[A-Z]?|904L)\b", r"SS\1", s)

    # 4. Standardize engineering units and dimensions (10 mm -> 10MM, 2 inch -> 2INCH)
    for pattern, replacement in UNIT_NORMALISATIONS:
        s = re.sub(pattern, replacement, s)

    # 5. Expand standalone abbreviations (SS -> STAINLESS STEEL when not part of SS304, etc.)
    s = re.sub(r"\bSS\b", "STAINLESS STEEL", s)
    for pattern, replacement in ABBREVIATION_MAP:
        s = re.sub(pattern, replacement, s)

    # 6. Remove administrative procurement boilerplate if requested
    if remove_boilerplate:
        for pat in LEADING_BOILERPLATE_PATTERNS:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)
        for pat in ADMIN_FILLER_PATTERNS:
            s = re.sub(pat, " ", s, flags=re.IGNORECASE)

    # 7. Normalize punctuation & separators (clean multiple commas, semicolons, extra spaces)
    # Note: Keep hyphens (-), slashes (/), hashes (#), dots (.) for specs and decimals
    s = re.sub(r"[,;:!?]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s
