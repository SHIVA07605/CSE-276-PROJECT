"""
Hybrid Pairwise Feature Builder for the Entity-Resolution Model.

Turns a pair of material descriptions into a fixed-length numeric feature vector that
combines four families of evidence:

1. Semantic   : MiniLM sentence-embedding cosine similarity.
2. Lexical    : word / character TF-IDF cosine, token Jaccard & containment, fuzzy ratios.
3. Specification evidence (canonical, unit-aware):
      - number+unit specs (10MM == 10 MM, 1.1KV == 1100V, CLASS 150 == 150#, 2.5 SQMM == 2.5MM2)
      - mutually exclusive descriptors (BALL vs GATE valve, CENTRIFUGAL vs SUBMERSIBLE pump,
        XLPE vs PVC insulation, stainless vs carbon steel, ...)
      - governing standards (API 6D vs API 6A), specific grades (SS304 vs SS316), bearing series
4. Rule signals reused from the project: PrecisionMatchingEngine decision / conflict counts and
   the legacy regex attribute extractor.

Leakage guard: no record IDs, enterprise names, tender numbers or labels are ever used.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:  # rapidfuzz is declared in requirements.txt; difflib fallback keeps tests runnable without it
    from rapidfuzz import fuzz as _fuzz

    def _ratio(a: str, b: str) -> float:
        return _fuzz.ratio(a, b) / 100.0

    def _token_sort(a: str, b: str) -> float:
        return _fuzz.token_sort_ratio(a, b) / 100.0

    def _token_set(a: str, b: str) -> float:
        return _fuzz.token_set_ratio(a, b) / 100.0

    def _partial(a: str, b: str) -> float:
        return _fuzz.partial_ratio(a, b) / 100.0

except ImportError:  # pragma: no cover - exercised only when rapidfuzz is absent
    import difflib

    def _ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _token_sort(a: str, b: str) -> float:
        return _ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))

    def _token_set(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        inter = " ".join(sorted(sa & sb))
        ra = (inter + " " + " ".join(sorted(sa - sb))).strip()
        rb = (inter + " " + " ".join(sorted(sb - sa))).strip()
        return max(_ratio(inter, ra), _ratio(inter, rb), _ratio(ra, rb)) if inter else _ratio(ra, rb)

    def _partial(a: str, b: str) -> float:
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        if not short:
            return 0.0
        m = difflib.SequenceMatcher(None, short, long_).find_longest_match(0, len(short), 0, len(long_))
        return m.size / len(short)


from src.matching_engine import PrecisionMatchingEngine
from src.preprocessing.attribute_extractor import extract_material_attributes

# ---------------------------------------------------------------------------
# Ordered feature schema (the model is trained on exactly this column order)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: List[str] = [
    # semantic & lexical similarity
    "sem_cosine",
    "tfidf_word_cosine",
    "tfidf_char_cosine",
    "sem_minus_lex",
    "token_jaccard",
    "token_containment",
    "fuzz_token_sort",
    "fuzz_token_set",
    "fuzz_ratio",
    "fuzz_partial",
    "len_min_tokens",
    "len_max_tokens",
    "len_ratio",
    "exact_norm_match",
    # token-level differences
    "alpha_symdiff",
    "alpha_unmatched_fuzzy",
    "num_bare_jaccard",
    "num_bare_symdiff",
    # canonical specification evidence
    "unit_spec_matches",
    "unit_spec_conflicts",
    "unit_spec_partial",
    "unit_spec_one_side",
    "group_matches",
    "group_conflicts",
    "group_one_side",
    "standard_matches",
    "standard_conflicts",
    "grade_conflicts",
    "bearing_conflicts",
    "identity_spec_min",
    "identity_spec_max",
    # project rule-engine signals
    "engine_equivalent",
    "engine_not_equivalent",
    "engine_conflicts",
    "engine_matches",
    "engine_category_conflict",
    "legacy_attr_matches",
    "legacy_attr_conflicts",
]

# Features the CLASSIFIER is trained on. Two groups are computed but deliberately withheld:
#  * identity_spec_min / identity_spec_max - only spec-rich records can be mutated into hard negatives,
#    so raw specification counts would let the classifier learn a construction shortcut
#    ("many specs => non-match"). They are used only by the decision-policy safety gates.
#  * legacy_attr_* - the original string-equality extractor reports false conflicts on equivalent
#    formats (e.g. '150#' vs 'CLASS 150', '101.60MM' vs '101.6MM'); the canonical unit-aware
#    features supersede them.
GATE_ONLY_FEATURES: List[str] = ["identity_spec_min", "identity_spec_max", "legacy_attr_matches", "legacy_attr_conflicts"]
MODEL_FEATURE_COLUMNS: List[str] = [c for c in FEATURE_COLUMNS if c not in GATE_ONLY_FEATURES]

# Hard-conflict features: any positive value is treated as decisive evidence of non-equivalence
HARD_CONFLICT_FEATURES: List[str] = [
    "unit_spec_conflicts",
    "group_conflicts",
    "standard_conflicts",
    "grade_conflicts",
    "bearing_conflicts",
    "engine_category_conflict",
]

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
_STOPWORDS: Set[str] = {
    "OF", "FOR", "THE", "AND", "WITH", "TO", "IN", "ON", "AT", "BY", "A", "AN", "OR", "AS",
    "FROM", "INTO", "UNDER", "PER", "ITS", "IS", "BE", "ALL", "ANY", "TYPE",
}
_QTY_WORDS: Set[str] = {
    "NUMBERS", "NUMBER", "NOS", "NO", "QUANTITY", "QTY", "SET", "SETS", "LOT", "LOTS", "PCS",
    "PIECES", "PIECE", "UNITS", "UNIT", "EACH", "EA", "Q1", "Q2", "Q3", "Q4",
}
_BOILER_WORDS: Set[str] = {
    "SUPPLY", "PROCUREMENT", "TENDER", "PURCHASE", "RATE", "CONTRACT", "DELIVERY", "BIDS", "BID",
}
_TOKEN_RE = re.compile(r"[A-Z0-9]+(?:[./][A-Z0-9]+)*")

_QTY_CONTEXT_PATTERNS = [
    re.compile(r"\bQUANTITY\s*[-:]?\s*\d[\d\s.,]*(?:NUMBERS?|NOS?\.?|SETS?|LOTS?|PCS|PIECES|UNITS?|EA|EACH|KM|MTRS?|M)?\b"),
    re.compile(r"\bQTY\s*[-:.]?\s*\d+(?:\.\d+)?"),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:NUMBERS?|NOS?\.?|SETS?|LOTS?|PCS|PIECES|UNITS?|EA|EACH)\b"),
    re.compile(r"\(\s*Q\d\s*\)?"),
    re.compile(r"\bFY\s*\d{2,4}(?:\s*-\s*\d{2,4})?"),
]


def _stem(tok: str) -> str:
    if tok.isalpha() and len(tok) > 3 and tok.endswith("S") and not tok.endswith("SS"):
        return tok[:-1]
    return tok


def content_tokens(text: str) -> List[str]:
    """Upper-case content tokens with stopwords, quantity words and boilerplate removed."""
    toks = _TOKEN_RE.findall(str(text).upper())
    out = []
    for t in toks:
        if t in _STOPWORDS or t in _QTY_WORDS or t in _BOILER_WORDS:
            continue
        out.append(_stem(t))
    return out


def strip_quantity_context(text: str) -> str:
    s = str(text).upper()
    for pat in _QTY_CONTEXT_PATTERNS:
        s = pat.sub(" ", s)
    return s


def bare_numbers(text: str) -> Set[str]:
    """All numeric literals outside quantity / fiscal-year contexts, canonicalised (10.0 -> 10)."""
    s = strip_quantity_context(text)
    out: Set[str] = set()
    for m in re.finditer(r"\d+(?:\.\d+)?", s):
        try:
            v = float(m.group(0))
        except ValueError:
            continue
        out.add(f"{v:g}")
    return out


# ---------------------------------------------------------------------------
# Canonical number+unit specification parser
# ---------------------------------------------------------------------------
_NUM = r"(\d+\s\d+/\d+|\d+-\d+/\d+|\d+\.\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
# (unit regex, canonical unit, scale) - ordered so that specific units win over prefixes
_UNIT_ALIASES: List[Tuple[str, str, float]] = [
    (r"MM2|SQ\.?\s*MM|SQMM", "SQMM", 1.0),
    (r"KL/HR|KLPH|KL/H", "KLHR", 1.0),
    (r"M3/HR|M3/H|CUM/HR|M3PH", "M3HR", 1.0),
    (r"LPM", "LPM", 1.0),
    (r"LPH", "LPH", 1.0),
    (r"GPM", "GPM", 1.0),
    (r"CFM", "CFM", 1.0),
    (r"SCMD", "SCMD", 1.0),
    (r"TPH", "TPH", 1.0),
    (r"KG/SQ\.?\s*CM|KG/CM2|KSC", "KSC", 1.0),
    (r"MWHR|MWH", "WH", 1e6),
    (r"KWH", "WH", 1e3),
    (r"MVA", "VA", 1e6),
    (r"KVA", "VA", 1e3),
    (r"MW", "W", 1e6),
    (r"KW", "W", 1e3),
    (r"KV", "V", 1e3),
    (r"VOLTS?|V", "V", 1.0),
    (r"BHP|HP", "HP", 1.0),
    (r"PSI", "PSI", 1.0),
    (r"BAR", "BAR", 1.0),
    (r"MILLIMET(?:ER|RE)S?|MM", "MM", 1.0),
    (r"CM", "CM", 1.0),
    (r"KM", "KM", 1.0),
    (r"INCH(?:ES)?|\"", "INCH", 1.0),
    (r"KGS?", "KG", 1.0),
    (r"MT|TONS?|TONNES?", "TON", 1.0),
    (r"KL", "L", 1e3),
    (r"LTRS?|LITRES?|LITERS?", "L", 1.0),
    (r"MTRS?|METERS?|METRES?|M", "M", 1.0),
    (r"RPM", "RPM", 1.0),
    (r"HZ", "HZ", 1.0),
    (r"CORES?", "CORE", 1.0),
]
_UNIT_RES = [
    (re.compile(r"(?<![A-Z0-9./])" + _NUM + r"\s*(?:" + alias + r")(?![A-WYZ0-9])"), canon, scale)
    for alias, canon, scale in _UNIT_ALIASES
]
_CORE_RE = re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)C(?![A-WYZ0-9])")  # 3C, 3.5C (attached only)
_SEP = r"\s*[-.:]?\s*"  # tolerate 'CLASS-150', 'CLASS. 150', 'SCH: 40'
_CLASS_RE = re.compile(r"\b(?:CLASS|CL)" + _SEP + r"(\d{2,4})\b|(?<![A-Z0-9./])(\d{2,4})\s*#")
_PN_RE = re.compile(r"\bPN" + _SEP + r"(\d{1,3})\b")
_DN_RE = re.compile(r"\bDN" + _SEP + r"(\d{2,4})\b")
_SCH_RE = re.compile(r"\bSCH(?:EDULE)?" + _SEP + r"(\d{1,3}S?|STD|XXS|XS)\b")


def _parse_number(raw: str) -> Optional[float]:
    raw = raw.strip()
    try:
        if re.fullmatch(r"\d+\s\d+/\d+", raw):  # 1 1/2
            whole, frac = raw.split()
            n, d = frac.split("/")
            return float(whole) + float(n) / float(d)
        if re.fullmatch(r"\d+-\d+/\d+", raw):  # 1-1/2
            whole, frac = raw.split("-")
            n, d = frac.split("/")
            return float(whole) + float(n) / float(d)
        if re.fullmatch(r"\d+\.\d+/\d+", raw):  # 5.1/2 (catalogue shorthand for 5 1/2)
            whole, frac = raw.split(".", 1)
            n, d = frac.split("/")
            return float(whole) + float(n) / float(d)
        if "/" in raw:
            n, d = raw.split("/")
            return float(n) / float(d) if float(d) else None
        return float(raw)
    except (ValueError, ZeroDivisionError):
        return None


def _canon_value(v: float) -> str:
    return f"{float(v):.6g}"


def extract_unit_specs(text: str) -> Dict[str, Set[str]]:
    """
    Parse number+unit specifications into {canonical_unit: {canonical_values}}.

    Examples:
        '10 MM'      -> {'MM': {'10'}}
        '1.1 KV'     -> {'V': {'1100'}}      (same as '1100V')
        'CLASS 150'  -> {'CLASS': {'150'}}   (same as '150#')
        '3C X 400 MM2' -> {'CORE': {'3'}, 'SQMM': {'400'}}
    Quantity phrases (QUANTITY 76 NUMBERS, 10 NOS, (Q3)) are ignored.
    """
    s = strip_quantity_context(text)
    specs: Dict[str, Set[str]] = {}

    def _add(unit: str, value: Optional[float]) -> None:
        if value is None:
            return
        specs.setdefault(unit, set()).add(_canon_value(value))

    # Codes first (they would otherwise be eaten by generic unit patterns)
    for m in _CLASS_RE.finditer(s):
        _add("CLASS", _parse_number(m.group(1) or m.group(2)))
    s = _CLASS_RE.sub(" ", s)
    for m in _PN_RE.finditer(s):
        _add("PN", _parse_number(m.group(1)))
    for m in _DN_RE.finditer(s):
        _add("DN", _parse_number(m.group(1)))
    for m in _SCH_RE.finditer(s):
        specs.setdefault("SCH", set()).add(m.group(1).rstrip("S") if m.group(1)[:-1].isdigit() else m.group(1))
    s = _SCH_RE.sub(" ", _DN_RE.sub(" ", _PN_RE.sub(" ", s)))

    for m in _CORE_RE.finditer(s):
        v = _parse_number(m.group(1))
        if v is not None and v <= 61:
            _add("CORE", v)
    s = _CORE_RE.sub(" ", s)

    for rx, canon, scale in _UNIT_RES:
        found = list(rx.finditer(s))
        for m in found:
            v = _parse_number(m.group(1))
            if v is not None:
                _add(canon, v * scale)
        if found:
            s = rx.sub(" ", s)
    return specs


# ---------------------------------------------------------------------------
# Mutually exclusive descriptor groups, standards, grades, bearing series
# ---------------------------------------------------------------------------
FEATURE_GROUPS: Dict[str, Dict[str, str]] = {
    "valve_type": {
        "BALL": r"\bBALL\s+VALVES?\b",
        "GATE": r"\bGATE\s+VALVES?\b",
        "GLOBE": r"\bGLOBE\s+VALVES?\b",
        "CHECK": r"\bCHECK\s+VALVES?\b|\bNON[\s-]*RETURN\s+VALVES?\b|\bNRV\b",
        "BUTTERFLY": r"\bBUTTERFLY\s+VALVES?\b",
        "PLUG": r"\bPLUG\s+VALVES?\b",
        "NEEDLE": r"\bNEEDLE\s+VALVES?\b",
    },
    "pump_type": {
        "CENTRIFUGAL": r"\bCENTRIFUGAL\b",
        "SUBMERSIBLE": r"\bSUBMERSIBLE\b",
        "RECIPROCATING": r"\bRECIPROCATING\b",
        "SCREW": r"\bSCREW\s+PUMPS?\b",
        "GEAR": r"\bGEAR\s+PUMPS?\b",
        "DIAPHRAGM": r"\bDIAPHRAGM\b",
    },
    "bearing_type": {
        "DEEP_GROOVE": r"\bDEEP\s+GROOVE\b",
        "TAPER_ROLLER": r"\bTAPER(?:ED)?\s+ROLLER\b",
        "SPHERICAL_ROLLER": r"\bSPHERICAL\s+ROLLER\b",
        "CYLINDRICAL_ROLLER": r"\bCYLINDRICAL\s+ROLLER\b",
        "NEEDLE_ROLLER": r"\bNEEDLE\s+(?:ROLLER\s+)?BEARINGS?\b",
        "ANGULAR_CONTACT": r"\bANGULAR\s+CONTACT\b",
    },
    "cable_insulation": {"XLPE": r"\bXLPE\b", "PVC": r"\bPVC\b(?=.*\bCABLES?\b)|\bCABLES?\b.*\bPVC\b"},
    "armour": {"UNARMOURED": r"\bUN-?ARMOU?RED\b", "ARMOURED": r"(?<!UN)(?<!UN-)\bARMOU?RED\b"},
    "pipe_construction": {"SEAMLESS": r"\bSEAMLESS\b|\bSMLS\b", "WELDED": r"\bERW\b|\bSAW\b|\bWELDED\b"},
    "end_connection": {
        "FLANGED": r"\bFLANGED\b|\bFLG\b",
        "THREADED": r"\bSCREWED\b|\bTHREADED\b|\bNPT\b|\bBSP\b",
        "SOCKET_WELD": r"\bSOCKET\s*WELD(?:ED)?\b",
        "BUTT_WELD": r"\bBUTT\s*WELD(?:ED)?\b",
    },
    "material_family": {
        "STAINLESS": r"\bSTAINLESS\b|\bSS\s*\d{3}L?\b|\bAISI\s*3\d\d\b",
        "CARBON": r"\bCARBON\s+STEEL\b|\bMILD\s+STEEL\b|\bA105\b|\bWCB\b",
        "ALLOY_STEEL": r"\bALLOY\s+STEEL\b",
        "CAST_IRON": r"\bCAST\s+IRON\b",
        "DUCTILE_IRON": r"\bDUCTILE\s+IRON\b",
        "GALVANISED_IRON": r"\bGALVANI[SZ]ED\s+IRON\b",
        "COPPER": r"\bCOPPER\b",
        "ALUMINIUM": r"\bALUMINI?UM\b",
        "BRASS": r"\bBRASS\b",
        "BRONZE": r"\bBRONZE\b",
        "HDPE": r"\bHDPE\b",
    },
    "phase": {"SINGLE": r"\bSINGLE\s+PHASE\b|\b1\s*PHASE\b", "THREE": r"\bTHREE\s+PHASE\b|\b3\s*PHASE\b"},
    "voltage_class": {"HT": r"\bHT\b", "LT": r"\bLT\b"},
}
_FEATURE_GROUP_RES = {
    g: {k: re.compile(v) for k, v in members.items()} for g, members in FEATURE_GROUPS.items()
}

_STANDARD_RES = [
    (re.compile(r"\bAPI\s*[-:]?\s*(\d{1,3}[A-Z]{0,2})\b"), "API"),
    (re.compile(r"\bIS\s*[:\-]?\s*(\d{3,5})\b"), "IS"),
    (re.compile(r"\bASME\s*(B\s*\d+\.\d+)\b"), "ASME"),
    (re.compile(r"\bASTM\s*([A-Z]\s*\d+)\b"), "ASTM"),
    (re.compile(r"\bBS\s*(\d{2,5})\b"), "BS"),
    (re.compile(r"\bIEC\s*(\d{2,5})\b"), "IEC"),
    (re.compile(r"\bDIN\s*(\d{2,5})\b"), "DIN"),
]
_GRADE_RE = re.compile(r"\bSS\s*(304L?|316L?|321|310|410|420|904L)\b|\b(WCB|LCB|CF8M|CF8|A105|LF2|F316L?|F304L?)\b")
_BEARING_SERIES_RE = re.compile(r"\b(6[0-4]\d{2}|608|609|22[2-3]\d{2}|NU\s*\d{3,4}|NJ\s*\d{3,4}|NUP\s*\d{3,4})\b")


def extract_group_values(text: str) -> Dict[str, Set[str]]:
    s = str(text).upper()
    out: Dict[str, Set[str]] = {}
    for g, members in _FEATURE_GROUP_RES.items():
        vals = {k for k, rx in members.items() if rx.search(s)}
        if vals:
            out[g] = vals
    return out


def extract_standards(text: str) -> Dict[str, Set[str]]:
    s = str(text).upper()
    out: Dict[str, Set[str]] = {}
    for rx, body in _STANDARD_RES:
        for m in rx.finditer(s):
            out.setdefault(body, set()).add(re.sub(r"\s+", "", m.group(1)))
    return out


def extract_grades(text: str) -> Set[str]:
    s = str(text).upper()
    out: Set[str] = set()
    for m in _GRADE_RE.finditer(s):
        out.add(("SS" + m.group(1)) if m.group(1) else m.group(2))
    return out


def extract_bearing_series(text: str) -> Set[str]:
    s = str(text).upper()
    if "BEARING" not in s:
        return set()
    return {re.sub(r"\s+", "", m.group(1)) for m in _BEARING_SERIES_RE.finditer(s)}


def parse_specifications(text: str) -> Dict[str, object]:
    """Full canonical specification profile of one description (cached by callers)."""
    units = extract_unit_specs(text)
    groups = extract_group_values(text)
    stds = extract_standards(text)
    grades = extract_grades(text)
    bearings = extract_bearing_series(text)
    identity = len(units) + sum(len(v) for v in stds.values()) + len(grades) + len(bearings)
    return {
        "units": units,
        "groups": groups,
        "standards": stds,
        "grades": grades,
        "bearings": bearings,
        "identity_spec_count": identity,
    }


def _compare_keyed_sets(a: Dict[str, Set[str]], b: Dict[str, Set[str]]) -> Tuple[int, int, int, int]:
    """Return (matches, conflicts, partial, one_side) for dict-of-sets evidence."""
    matches = conflicts = partial = 0
    for k in set(a) & set(b):
        if a[k] == b[k]:
            matches += 1
        elif not (a[k] & b[k]):
            conflicts += 1
        else:
            partial += 1
    one_side = len(set(a) ^ set(b))
    return matches, conflicts, partial, one_side


def _set_conflict(a: Set[str], b: Set[str]) -> int:
    return int(bool(a) and bool(b) and not (a & b))


# ---------------------------------------------------------------------------
# TF-IDF helper
# ---------------------------------------------------------------------------
class TfidfSimilarity:
    """Word and character n-gram TF-IDF models fitted once on the real catalogue."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, token_pattern=r"[A-Z0-9][A-Z0-9./#]*", lowercase=False)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=False)
        self.fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfSimilarity":
        texts = [str(t).upper() for t in texts]
        self.word.fit(texts)
        self.char.fit(texts)
        self.fitted = True
        return self

    def pair_cosines(self, texts_a: Sequence[str], texts_b: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        uniq = sorted(set(map(str, texts_a)) | set(map(str, texts_b)))
        idx = {t: i for i, t in enumerate(uniq)}
        ia = np.array([idx[str(t)] for t in texts_a])
        ib = np.array([idx[str(t)] for t in texts_b])
        out = []
        for vec in (self.word, self.char):
            M = vec.transform([u.upper() for u in uniq])
            sims = np.asarray(M[ia].multiply(M[ib]).sum(axis=1)).ravel()
            out.append(np.clip(sims, 0.0, 1.0))
        return out[0], out[1]


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def _alpha_diffs(ta: List[str], tb: List[str]) -> Tuple[int, int]:
    sa = {t for t in ta if t.isalpha()}
    sb = {t for t in tb if t.isalpha()}
    diff_a, diff_b = sa - sb, sb - sa
    symdiff = len(diff_a) + len(diff_b)

    def _matched(tok: str, pool: Set[str]) -> bool:
        for other in pool:
            if len(tok) >= 3 and len(other) >= 3 and (other.startswith(tok) or tok.startswith(other)):
                return True
            if len(tok) >= 4 and _ratio(tok, other) >= 0.8:
                return True
        return False

    unmatched = sum(1 for t in diff_a if not _matched(t, sb)) + sum(1 for t in diff_b if not _matched(t, sa))
    return symdiff, unmatched


def build_pair_features(
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    tfidf: TfidfSimilarity,
    spec_cache: Optional[Dict[str, Dict[str, object]]] = None,
    engine: Optional[PrecisionMatchingEngine] = None,
) -> pd.DataFrame:
    """
    Compute the hybrid feature matrix for aligned lists of (normalised) descriptions.

    Args:
        texts_a, texts_b: Normalised (cleaned) descriptions of the two records of each pair.
        emb_a, emb_b: L2-normalised sentence embeddings aligned with texts_a / texts_b.
        tfidf: Fitted TfidfSimilarity instance.
        spec_cache: Optional dict reused across calls to avoid re-parsing identical texts.
        engine: Optional PrecisionMatchingEngine instance.

    Returns:
        DataFrame with columns == FEATURE_COLUMNS (all finite floats).
    """
    n = len(texts_a)
    if n != len(texts_b) or n != len(emb_a) or n != len(emb_b):
        raise ValueError("texts and embeddings must be aligned")
    spec_cache = spec_cache if spec_cache is not None else {}
    engine = engine or PrecisionMatchingEngine()
    legacy_cache: Dict[str, Dict[str, object]] = {}

    def _spec(t: str) -> Dict[str, object]:
        if t not in spec_cache:
            spec_cache[t] = parse_specifications(t)
        return spec_cache[t]

    def _legacy(t: str) -> Dict[str, object]:
        if t not in legacy_cache:
            legacy_cache[t] = extract_material_attributes(t)
        return legacy_cache[t]

    sem = np.clip(np.sum(np.asarray(emb_a, dtype=np.float32) * np.asarray(emb_b, dtype=np.float32), axis=1), -1.0, 1.0)
    tf_word, tf_char = tfidf.pair_cosines(texts_a, texts_b) if n else (np.array([]), np.array([]))

    rows = []
    legacy_keys = ["metric_thread", "diameter", "material_grade", "pressure_class", "schedule", "voltage", "capacity"]
    for i in range(n):
        a = str(texts_a[i]).upper().strip()
        b = str(texts_b[i]).upper().strip()
        ta, tb = content_tokens(a), content_tokens(b)
        sa, sb = set(ta), set(tb)
        inter = len(sa & sb)
        union = len(sa | sb)
        jacc = inter / union if union else 0.0
        contain = inter / min(len(sa), len(sb)) if sa and sb else 0.0
        la, lb = len(ta), len(tb)
        symdiff, unmatched = _alpha_diffs(ta, tb)
        na, nb = bare_numbers(a), bare_numbers(b)
        nu = na | nb
        num_jacc = len(na & nb) / len(nu) if nu else 1.0

        pa, pb = _spec(a), _spec(b)
        u_m, u_c, u_p, u_o = _compare_keyed_sets(pa["units"], pb["units"])
        g_m, g_c, _, g_o = _compare_keyed_sets(pa["groups"], pb["groups"])
        s_m, s_c, _, _ = _compare_keyed_sets(pa["standards"], pb["standards"])
        # a shared standard anywhere counts as a match even if bodies differ in coverage
        s_m = max(s_m, int(any(pa["standards"].get(k, set()) & pb["standards"].get(k, set()) for k in pa["standards"])))

        assessment = engine.evaluate_pair("A", a, "B", b, semantic_sim=float(sem[i]))
        la_attr, lb_attr = _legacy(a), _legacy(b)
        leg_m = leg_c = 0
        for k in legacy_keys:
            v1, v2 = la_attr.get(k), lb_attr.get(k)
            if v1 and v2:
                if str(v1).strip().upper() == str(v2).strip().upper():
                    leg_m += 1
                else:
                    leg_c += 1

        rows.append((
            float(sem[i]),
            float(tf_word[i]),
            float(tf_char[i]),
            float(sem[i] - tf_word[i]),
            jacc,
            contain,
            _token_sort(a, b),
            _token_set(a, b),
            _ratio(a, b),
            _partial(a, b),
            float(min(la, lb)),
            float(max(la, lb)),
            (min(la, lb) / max(la, lb)) if max(la, lb) else 0.0,
            float(a == b and len(a) > 0),
            float(symdiff),
            float(unmatched),
            num_jacc,
            float(len(na ^ nb)),
            float(u_m),
            float(u_c),
            float(u_p),
            float(u_o),
            float(g_m),
            float(g_c),
            float(g_o),
            float(s_m),
            float(s_c),
            float(_set_conflict(pa["grades"], pb["grades"])),
            float(_set_conflict(pa["bearings"], pb["bearings"])),
            float(min(pa["identity_spec_count"], pb["identity_spec_count"])),
            float(max(pa["identity_spec_count"], pb["identity_spec_count"])),
            float(assessment.decision == "EQUIVALENT"),
            float(assessment.decision == "NOT_EQUIVALENT"),
            float(len(assessment.conflicting_attributes)),
            float(len(assessment.matched_attributes)),
            float("equipment_category" in assessment.conflicting_attributes),
            float(leg_m),
            float(leg_c),
        ))

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return X.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def hard_conflict_mask(X: pd.DataFrame) -> np.ndarray:
    """True where explicit, canonical specification evidence contradicts equivalence."""
    return (X[HARD_CONFLICT_FEATURES].sum(axis=1) > 0).to_numpy()
