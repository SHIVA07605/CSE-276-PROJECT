"""
Self-Supervised Training-Pair Synthesis (labels known by construction).

Why this module exists
----------------------
Only 20 human-confirmed binary labels exist (2 positive), far too few to train a classifier.
Following self-supervised entity-matching practice (data augmentation for EM, e.g. Ditto /
Sudowoodo), we derive training pairs from the REAL 6,634-record candidate pool whose labels are
known by construction rather than by anyone's judgement:

* POSITIVE (label 1)  : a record vs. a *format-only rewrite* of the same record - boilerplate,
  quantity suffixes, noun-first SAP ordering, abbreviations, unit spacing, typos, plurals.
  The physical item is unchanged, so the pair is equivalent by definition.
* HARD NEGATIVE (0)   : a record vs. a rewrite in which exactly ONE specification was changed
  (250MM -> 300MM, SS304 -> SS316, CLASS 150 -> 300, BALL -> GATE valve, API 6D -> API 6A ...).
  By the project's own definition, different specifications are not functionally equivalent.
* REAL NEGATIVES (0)  : (a) real candidate pairs whose canonical number+unit specs are disjoint
  (e.g. 1400MM vs 1800MM conveyor belts), and (b) random pairs from different commodity classes.

Nothing here uses human labels, AI hypotheses or enterprise identities. Records appearing in the
human review set are excluded entirely so that the gold evaluation stays untouched.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from src.preprocessing.text_cleaner import normalise_text

# ---------------------------------------------------------------------------
# Format-preserving (label-preserving) transformations
# ---------------------------------------------------------------------------
BOILERPLATE_PREFIXES = [
    "SUPPLY OF ", "PROCUREMENT OF ", "TENDER FOR ", "RATE CONTRACT FOR ", "PURCHASE OF ",
    "Supply of ", "procurement of ",
]
QUANTITY_SUFFIXES = [
    " (NUMBERS)", " (NOS)", " (NOS.)", " - {n} NOS", " QUANTITY {n} NUMBERS", " QTY: {n}",
    " ({n} NOS)", " (Q{q})", " - {n} NUMBERS",
]
ABBREVIATIONS = [
    ("STAINLESS STEEL", "SS"), ("CARBON STEEL", "CS"), ("MILD STEEL", "MS"),
    ("GALVANISED IRON", "GI"), ("CAST IRON", "CI"), ("DIAMETER", "DIA"),
    ("HEXAGONAL", "HEX"), ("THICKNESS", "THK"), ("QUANTITY", "QTY"), ("NUMBERS", "NOS"),
    ("ASSEMBLY", "ASSY"), ("EQUIPMENT", "EQPT"), ("ELECTRICAL", "ELEC"),
    ("MECHANICAL", "MECH"), ("INSTRUMENT", "INSTR"), ("NON RETURN VALVE", "NRV"),
]
HEAD_NOUNS = [
    "VALVE", "VALVES", "PUMP", "PUMPS", "CABLE", "CABLES", "BEARING", "BEARINGS", "PIPE",
    "PIPES", "FLANGE", "FLANGES", "GASKET", "GASKETS", "MOTOR", "MOTORS", "COMPRESSOR",
    "TRANSFORMER", "FILTER", "FILTERS", "BELT", "TANK", "BOILER", "TURBINE", "COUPLING",
    "SPARES", "ANALYZER", "STIRRER", "HOSE", "TUBE", "SEAL", "SWITCHGEAR", "CRANE",
]
_FILLERS = ["OF", "FOR", "WITH", "AND", "THE", "TYPE"]


def _aug_prefix(t: str, rng: random.Random) -> str:
    return rng.choice(BOILERPLATE_PREFIXES) + t


def _aug_quantity(t: str, rng: random.Random) -> str:
    return t + rng.choice(QUANTITY_SUFFIXES).format(n=rng.choice([1, 2, 4, 6, 10, 12, 20, 25, 50, 76, 100]), q=rng.randint(1, 4))


def _aug_noun_first(t: str, rng: random.Random) -> str:
    toks = t.split()
    for i, tok in enumerate(toks):
        if i > 0 and tok in HEAD_NOUNS:
            rest = toks[:i] + toks[i + 1:]
            sep = rng.choice([", ", "-", " "])
            return tok + sep + " ".join(rest)
    if 2 <= len(toks) <= 5:
        return toks[-1] + rng.choice([", ", "-"]) + " ".join(toks[:-1])
    return t


def _aug_abbrev(t: str, rng: random.Random) -> str:
    options = [(l, s) for l, s in ABBREVIATIONS if re.search(r"\b" + re.escape(l) + r"\b", t)]
    if not options:
        return t
    long_, short = rng.choice(options)
    return re.sub(r"\b" + re.escape(long_) + r"\b", short, t, count=1)


_UNIT_FORMATS = [
    (r"(\d+(?:\.\d+)?)MM\b", ["\\1 MM", "\\1 mm", "\\1MM.", "\\1 M.M."]),
    (r"(\d+(?:\.\d+)?|\d+/\d+)INCH\b", ["\\1\"", "\\1 INCH", "\\1 INCHES", "\\1 IN."]),
    (r"(\d+(?:\.\d+)?)\s*KV\b", ["\\1 KV", "\\1KV", "\\1 kV"]),
    (r"(\d+(?:\.\d+)?)\s*(?:SQ\.?\s*MM|SQMM|MM2)\b", ["\\1 SQMM", "\\1 SQ MM", "\\1 SQ.MM", "\\1MM2"]),
    (r"\bCLASS\s*(150|300|600|900|1500|2500)\b", ["\\1#", "\\1 #", "CLASS-\\1"]),
    (r"\b(150|300|600|900|1500|2500)#", ["CLASS \\1", "CL \\1#"]),
    (r"\b(\d+(?:\.\d+)?)C\b", ["\\1 CORE", "\\1 CORES", "\\1C"]),
    (r"\b(\d+(?:\.\d+)?)\s*CORES?\b", ["\\1C", "\\1 CORE"]),
    (r"\bSCH\s*(\d+)\b", ["SCH \\1", "SCHEDULE \\1", "SCH-\\1"]),
    (r"(\d+(?:\.\d+)?)KG\b", ["\\1 KG", "\\1 KGS", "\\1 kg"]),
]


def _aug_units(t: str, rng: random.Random) -> str:
    options = [(p, reps) for p, reps in _UNIT_FORMATS if re.search(p, t)]
    if not options:
        return t
    pat, reps = rng.choice(options)
    return re.sub(pat, rng.choice(reps), t, count=1)


def _aug_typo(t: str, rng: random.Random) -> str:
    toks = t.split()
    cands = [i for i, w in enumerate(toks) if w.isalpha() and len(w) >= 6]
    if not cands:
        return t
    i = rng.choice(cands)
    w = toks[i]
    j = rng.randint(1, len(w) - 2)
    op = rng.choice(["drop", "swap", "dup"])
    if op == "drop":
        w = w[:j] + w[j + 1:]
    elif op == "swap":
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    else:
        w = w[:j] + w[j] + w[j:]
    toks[i] = w
    return " ".join(toks)


def _aug_plural(t: str, rng: random.Random) -> str:
    toks = t.split()
    cands = [i for i, w in enumerate(toks) if w in HEAD_NOUNS]
    if not cands:
        return t
    i = rng.choice(cands)
    w = toks[i]
    toks[i] = w[:-1] if w.endswith("S") else w + "S"
    return " ".join(toks)


def _aug_filler(t: str, rng: random.Random) -> str:
    toks = t.split()
    cands = [i for i, w in enumerate(toks) if w in _FILLERS]
    if not cands or len(toks) <= 3:
        return t
    del toks[rng.choice(cands)]
    return " ".join(toks)


def _aug_punct(t: str, rng: random.Random) -> str:
    toks = t.split()
    if len(toks) < 3:
        return t
    i = rng.randint(1, len(toks) - 1)
    toks[i - 1] = toks[i - 1] + rng.choice([",", " -", ";", "."])
    return " ".join(toks)


def _aug_case(t: str, rng: random.Random) -> str:
    return rng.choice([t.lower(), t.title(), t])


# transformations that survive normalisation get extra weight so positives are non-trivial
FORMAT_TRANSFORMS = [
    ("boilerplate_prefix", _aug_prefix, 1.0),
    ("quantity_suffix", _aug_quantity, 1.5),
    ("noun_first_reorder", _aug_noun_first, 1.5),
    ("abbreviation", _aug_abbrev, 1.5),
    ("unit_format", _aug_units, 2.0),
    ("typo", _aug_typo, 1.2),
    ("plural", _aug_plural, 1.0),
    ("filler_drop", _aug_filler, 0.8),
    ("punctuation", _aug_punct, 0.8),
    ("casing", _aug_case, 0.5),
]


def augment(text: str, rng: random.Random, min_ops: int = 1, max_ops: int = 3) -> Tuple[str, List[str]]:
    """Apply 1-3 random label-preserving transformations. Returns (variant, applied_ops)."""
    names = [n for n, _, _ in FORMAT_TRANSFORMS]
    funcs = {n: f for n, f, _ in FORMAT_TRANSFORMS}
    weights = [w for _, _, w in FORMAT_TRANSFORMS]
    k = rng.randint(min_ops, max_ops)
    chosen: List[str] = []
    out = text
    for _ in range(k * 3):
        if len(chosen) >= k:
            break
        name = rng.choices(names, weights=weights, k=1)[0]
        if name in chosen:
            continue
        new = funcs[name](out, rng)
        if new != out:
            out = new
            chosen.append(name)
    return out, chosen


# ---------------------------------------------------------------------------
# Specification mutations (each produces a guaranteed non-equivalent item)
# ---------------------------------------------------------------------------
_SERIES = {
    "MM": [6, 8, 10, 12, 15, 20, 25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200],
    "KV": [0.415, 0.433, 1.1, 3.3, 6.6, 11, 22, 33, 66, 132, 220],
    "V": [12, 24, 48, 110, 230, 240, 415, 440, 690],
    "PSI": [500, 1000, 2000, 3000, 5000, 10000, 15000],
    "CLASS": [150, 300, 600, 900, 1500, 2500],
    "KVA": [25, 63, 100, 160, 250, 315, 500, 630, 1000, 1600, 2000, 2500],
    "MVA": [5, 10, 16, 20, 25, 31.5, 40, 50, 100],
    "CORE": [1, 2, 3, 3.5, 4],
    "SQMM": [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240, 300, 400],
}
_INCH_SERIES = ["1/4", "3/8", "1/2", "3/4", "1", "1-1/2", "2", "3", "4", "6", "8", "10", "12"]
_SCH_SERIES = ["10", "20", "40", "80", "120", "160", "XS", "XXS"]

_NUMERIC_MUTATION_RES = [
    ("CLASS", re.compile(r"\bCLASS\s*(150|300|600|900|1500|2500)\b")),
    ("CLASS", re.compile(r"(?<![A-Z0-9./])(150|300|600|900|1500|2500)\s*#")),
    ("SCH", re.compile(r"\bSCH(?:EDULE)?\s*(10|20|40|80|120|160|XXS|XS)\b")),
    ("SQMM", re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)\s*(?=(?:SQ\.?\s*MM|SQMM|MM2)\b)")),
    ("INCH", re.compile(r"(?<![A-Z0-9./])(\d+-\d+/\d+|\d+/\d+|\d+)\s*(?=(?:INCH|INCHES)\b)")),
    ("MM", re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)\s*(?=MM\b)")),
    ("KVA", re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)\s*(?=KVA\b)")),
    ("MVA", re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)\s*(?=MVA\b)")),
    ("KV", re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)\s*(?=KV\b)")),
    ("V", re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)\s*(?=V\b)")),
    ("PSI", re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)\s*(?=PSI\b)")),
    ("CORE", re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)(?=C\b|\s*CORES?\b)")),
    ("FACTOR", re.compile(r"(?<![A-Z0-9./])(\d+(?:\.\d+)?)\s*(?=(?:KW|MW|HP|KG|KL|MT|TON|TONS|BAR|M3/HR|KL/HR|LPH|LPM|TPH|CFM|SCMD|RPM|M)\b)")),
]

# Descriptor swap groups. Groups marked *unseen* are deliberately NOT encoded as explicit
# features in src/pair_features.py, so they test generalisation from lexical evidence alone.
MUTATION_GROUPS: Dict[str, Tuple[List[str], bool]] = {
    "valve_type": (["BALL VALVE", "GATE VALVE", "GLOBE VALVE", "CHECK VALVE", "BUTTERFLY VALVE", "PLUG VALVE", "NEEDLE VALVE"], False),
    "pump_type": (["CENTRIFUGAL", "SUBMERSIBLE", "RECIPROCATING"], False),
    "bearing_type": (["DEEP GROOVE", "TAPER ROLLER", "SPHERICAL ROLLER", "CYLINDRICAL ROLLER"], False),
    "cable_insulation": (["XLPE", "PVC"], False),
    "armour": (["ARMOURED", "UNARMOURED"], False),
    "pipe_construction": (["SEAMLESS", "ERW"], False),
    "end_connection": (["FLANGED", "SCREWED", "SOCKET WELD", "BUTT WELD"], False),
    "material_family": (["STAINLESS STEEL", "CARBON STEEL", "ALLOY STEEL", "CAST IRON", "DUCTILE IRON", "GALVANISED IRON"], False),
    "conductor": (["COPPER", "ALUMINIUM"], False),
    "copper_alloy": (["BRASS", "BRONZE"], False),
    "phase": (["SINGLE PHASE", "THREE PHASE"], False),
    "voltage_class": (["HT", "LT"], False),
    "specific_grade": (["SS304", "SS316", "SS316L", "SS321", "SS410"], False),
    # unseen-by-features groups
    "orientation": (["HORIZONTAL", "VERTICAL"], True),
    "drive": (["DIESEL", "PETROL", "ELECTRIC"], True),
    "cooling": (["AIR COOLED", "WATER COOLED", "OIL COOLED"], True),
    "mobility": (["PORTABLE", "STATIONARY"], True),
    "actuation": (["MANUAL", "PNEUMATIC", "HYDRAULIC", "MOTORISED"], True),
    "current_type": (["AC", "DC"], True),
    "duty": (["HEAVY DUTY", "MEDIUM DUTY", "LIGHT DUTY"], True),
}
STANDARD_SWAPS = [
    ("API 6D", "API 6A"), ("API 6A", "API 6D"), ("API 610", "API 674"), ("API 674", "API 610"),
    ("API 600", "API 602"), ("API 5L", "API 5CT"), ("API 5CT", "API 5L"),
    ("IS 7098", "IS 1554"), ("IS 1554", "IS 7098"), ("IS 2062", "IS 1079"), ("ASME B16.5", "ASME B16.47"),
]


def _fmt_number(v: float, like: str) -> str:
    if "." not in like and float(v).is_integer():
        return str(int(v))
    return f"{v:g}"


def _neighbour(series: List, old, rng: random.Random):
    try:
        idx = series.index(old)
    except ValueError:
        # nearest element
        idx = min(range(len(series)), key=lambda i: abs(float(series[i]) - float(old)) if not isinstance(series[i], str) else 0)
    offsets = [o for o in (-2, -1, 1, 2) if 0 <= idx + o < len(series)]
    new = series[idx + rng.choice(offsets)] if offsets else None
    return new if new != old else None


def _mutate_numeric(unit: str, raw: str, rng: random.Random) -> Optional[str]:
    if unit == "INCH":
        old = raw
        if old not in _INCH_SERIES:
            return None
        new = _neighbour(_INCH_SERIES, old, rng)
        return new
    if unit == "SCH":
        new = _neighbour(_SCH_SERIES, raw, rng)
        return new
    try:
        v = float(raw)
    except ValueError:
        return None
    if unit in _SERIES:
        series = _SERIES[unit]
        if v in series:
            new = _neighbour(series, v, rng)
        else:
            new = rng.choice([x for x in series if x != v])
        return _fmt_number(new, raw) if new is not None else None
    factor = rng.choice([0.5, 0.75, 1.5, 2.0, 2.5])
    new_v = v * factor
    if "." not in raw:
        new_v = round(new_v)
        if new_v == v or new_v <= 0:
            new_v = v + max(1, round(v * 0.5))
    else:
        new_v = round(new_v, len(raw.split(".")[1]))
        if new_v == v:
            new_v = v * 2
    return _fmt_number(new_v, raw)


def find_mutations(text: str) -> List[Dict[str, object]]:
    """Enumerate every single-specification change that can be applied to `text`."""
    t = str(text).upper()
    cands: List[Dict[str, object]] = []
    # quantity contexts must never be mutated
    protected: List[Tuple[int, int]] = [
        (m.start(), m.end()) for m in re.finditer(r"\bQUANTITY\s*[-:]?\s*\d[\d\s.,]*\w*|\b\d+\s*(?:NUMBERS?|NOS?\.?|SETS?|LOTS?|PCS|PIECES|UNITS?)\b|\(\s*Q\d\s*\)", t)
    ]

    def _is_protected(a: int, b: int) -> bool:
        return any(a < pe and b > ps for ps, pe in protected)

    seen_spans: Set[Tuple[int, int]] = set()
    for unit, rx in _NUMERIC_MUTATION_RES:
        for m in rx.finditer(t):
            span = m.span(1)
            if span in seen_spans or _is_protected(*span):
                continue
            seen_spans.add(span)
            cands.append({"kind": "numeric", "group": unit, "span": span, "old": m.group(1)})

    for group, (values, unseen) in MUTATION_GROUPS.items():
        for val in values:
            for m in re.finditer(r"(?<![A-Z-])" + re.escape(val) + r"(?![A-Z])", t):
                if group == "armour" and val == "ARMOURED" and t[max(0, m.start() - 2):m.start()] == "UN":
                    continue
                cands.append({"kind": "descriptor", "group": group, "span": m.span(), "old": val, "unseen": unseen})
    for old, new in STANDARD_SWAPS:
        for m in re.finditer(r"\b" + re.escape(old).replace(r"\ ", r"\s*") + r"\b", t):
            cands.append({"kind": "standard", "group": "standard", "span": m.span(), "old": old, "new": new})
    if "BEARING" in t:
        for m in re.finditer(r"\b(6[0-4]\d{2}|608|609)\b", t):
            cands.append({"kind": "bearing_series", "group": "bearing_series", "span": m.span(), "old": m.group(1)})
    return cands


def mutate(text: str, rng: random.Random) -> Optional[Tuple[str, Dict[str, object]]]:
    """Change exactly one specification. Returns (mutated_text, info) or None if impossible."""
    t = str(text).upper()
    cands = find_mutations(t)
    rng.shuffle(cands)
    for c in cands:
        a, b = c["span"]
        old = str(c["old"])
        new: Optional[str] = None
        if c["kind"] == "numeric":
            new = _mutate_numeric(str(c["group"]), old, rng)
        elif c["kind"] == "descriptor":
            values, _ = MUTATION_GROUPS[str(c["group"])]
            choices = [v for v in values if v != old]
            new = rng.choice(choices) if choices else None
        elif c["kind"] == "standard":
            new = str(c["new"])
        elif c["kind"] == "bearing_series":
            digits = list(old)
            pos = 1 if len(digits) == 4 else len(digits) - 1
            digits[pos] = rng.choice([d for d in "0123" if d != digits[pos]])
            new = "".join(digits)
        if new is None or new == old:
            continue
        mutated = t[:a] + new + t[b:]
        if normalise_text(mutated) == normalise_text(t):
            continue
        info = {
            "mutation_kind": c["kind"],
            "mutation_group": c["group"],
            "mutation_old": old,
            "mutation_new": new,
            "mutation_unseen_by_features": bool(c.get("unseen", False)),
        }
        return mutated, info
    return None


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _is_usable_seed(text: str) -> bool:
    toks = str(text).split()
    if not toks or len(toks) > 45:
        return False
    return any(len(re.sub(r"[^A-Z]", "", tok)) >= 3 for tok in toks)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_seed_groups(
    seed_texts: Sequence[str],
    candidate_pairs: Optional[pd.DataFrame],
    text_of_id: Dict[str, str],
    near_dup_threshold: float = 0.8,
) -> Dict[str, str]:
    """
    Group near-duplicate seed descriptions (TF-IDF >= threshold) with union-find so that
    near-identical records can never sit on both sides of the train/test boundary.
    """
    uf = _UnionFind()
    seeds = set(seed_texts)
    for t in seeds:
        uf.find(t)
    if candidate_pairs is not None and len(candidate_pairs):
        hi = candidate_pairs[candidate_pairs["tfidf_similarity"] >= near_dup_threshold]
        for id1, id2 in zip(hi["corpus_id_1"].astype(str), hi["corpus_id_2"].astype(str)):
            t1, t2 = text_of_id.get(id1), text_of_id.get(id2)
            if t1 in seeds and t2 in seeds:
                uf.union(t1, t2)
    return {t: uf.find(t) for t in seeds}


def split_of_group(group_key: str, val_frac: float = 0.15, test_frac: float = 0.15) -> str:
    h = (_stable_hash(group_key) % 10_000) / 10_000.0
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def build_self_supervised_pairs(
    materials: pd.DataFrame,
    candidate_pairs: Optional[pd.DataFrame] = None,
    exclude_ids: Optional[Set[str]] = None,
    exclude_texts: Optional[Set[str]] = None,
    random_state: int = 42,
    real_conflict_cap: int = 1500,
    random_negative_ratio: float = 0.35,
    unit_spec_parser=None,
) -> pd.DataFrame:
    """
    Build the labelled-by-construction training corpus from real catalogue records.

    Args:
        materials: preprocessed_materials.csv (needs corpus_id, cleaned_description, item_type_hint).
        candidate_pairs: TF-IDF/semantic candidate pairs (for near-duplicate grouping & real negatives).
        exclude_ids: corpus IDs to exclude entirely (the human review set).
        exclude_texts: normalised descriptions to exclude entirely (texts of the review set).
        random_state: seed for full reproducibility.
        real_conflict_cap: maximum number of real spec-conflict negatives.
        random_negative_ratio: random cross-commodity negatives per seed.
        unit_spec_parser: callable text -> {unit: set(values)} used to confirm real conflicts.

    Returns:
        DataFrame, one row per pair, with full provenance (no labels from humans or AI).
    """
    rng = random.Random(random_state)
    exclude_ids = {str(x) for x in (exclude_ids or set())}
    exclude_texts = {str(x).strip().upper() for x in (exclude_texts or set())}

    m = materials.copy()
    m["corpus_id"] = m["corpus_id"].astype(str)
    m["seed_text"] = m["cleaned_description"].fillna("").astype(str).str.strip().str.upper()
    m = m[~m["corpus_id"].isin(exclude_ids)]
    m = m[~m["seed_text"].isin(exclude_texts)]
    m = m[m["seed_text"].map(_is_usable_seed)]
    text_of_id = dict(zip(m["corpus_id"], m["seed_text"]))
    # one seed per unique description (exact duplicates would otherwise straddle splits)
    seeds = m.drop_duplicates("seed_text").reset_index(drop=True)
    hint_of_text = dict(zip(seeds["seed_text"], seeds["item_type_hint"].fillna("UNTAGGED").astype(str)))

    groups = assign_seed_groups(seeds["seed_text"].tolist(), candidate_pairs, text_of_id)
    split_of_text = {t: split_of_group(g) for t, g in groups.items()}

    rows: List[Dict[str, object]] = []

    def _row(kind: str, label: int, seed: str, seed_id: str, raw_a: str, raw_b: str, ops_a, ops_b, extra=None, node_suffix: Optional[str] = None) -> None:
        r = {
            "source": kind,
            "label": int(label),
            "seed_corpus_id": seed_id,
            "seed_text": seed,
            "group_id": groups[seed],
            "split": split_of_text[seed],
            "raw_a": raw_a,
            "raw_b": raw_b,
            "text_a": normalise_text(raw_a),
            "text_b": normalise_text(raw_b),
            "aug_ops_a": "|".join(ops_a),
            "aug_ops_b": "|".join(ops_b),
            "mutation_kind": "",
            "mutation_group": "",
            "mutation_old": "",
            "mutation_new": "",
            "mutation_unseen_by_features": False,
            "node_a": f"{seed_id}::a",
            "node_b": f"{seed_id}::{node_suffix or ('pos' if label == 1 else 'neg')}",
            "entity_a": seed_id,
            "entity_b": seed_id if label == 1 else f"{seed_id}#mut",
        }
        if extra:
            r.update(extra)
        rows.append(r)

    for _, s in seeds.iterrows():
        seed, sid = s["seed_text"], s["corpus_id"]
        # anchor: the record as catalogued, lightly perturbed half of the time
        if rng.random() < 0.5:
            anchor, ops_anchor = augment(seed, rng, 1, 1)
        else:
            anchor, ops_anchor = seed, []
        pos, ops_pos = augment(seed, rng, 1, 3)
        _row("synthetic_positive", 1, seed, sid, anchor, pos, ops_anchor, ops_pos)

        mut = mutate(seed, rng)
        if mut is not None:
            mutated, info = mut
            neg, ops_neg = augment(mutated, rng, 0, 2) if rng.random() < 0.7 else (mutated, [])
            _row("synthetic_hard_negative", 0, seed, sid, anchor, neg, ops_anchor, ops_neg, info)
            # Second positive for specification-rich seeds, so that the class prior does not depend on
            # how many specifications a description carries (prevents a "many specs => non-match" shortcut).
            pos2, ops_pos2 = augment(seed, rng, 2, 3)
            _row("synthetic_positive", 1, seed, sid, anchor, pos2, ops_anchor, ops_pos2, node_suffix="pos2")

    # -------- real negatives with confirmed canonical spec conflicts --------
    if candidate_pairs is not None and unit_spec_parser is not None and real_conflict_cap > 0:
        cp = candidate_pairs.copy()
        cp["corpus_id_1"] = cp["corpus_id_1"].astype(str)
        cp["corpus_id_2"] = cp["corpus_id_2"].astype(str)
        cp = cp[cp["corpus_id_1"].isin(text_of_id) & cp["corpus_id_2"].isin(text_of_id)]
        cp = cp.sample(frac=1.0, random_state=random_state)
        used: Set[Tuple[str, str]] = set()
        n_added = 0
        for id1, id2 in zip(cp["corpus_id_1"], cp["corpus_id_2"]):
            if n_added >= real_conflict_cap:
                break
            t1, t2 = text_of_id[id1], text_of_id[id2]
            if t1 == t2 or (t1, t2) in used or split_of_text.get(t1) != split_of_text.get(t2):
                continue
            u1, u2 = unit_spec_parser(t1), unit_spec_parser(t2)
            shared = set(u1) & set(u2)
            if not any(not (u1[k] & u2[k]) for k in shared):
                continue
            used.add((t1, t2))
            n_added += 1
            rows.append({
                "source": "real_spec_conflict_negative",
                "label": 0,
                "seed_corpus_id": id1,
                "seed_text": t1,
                "group_id": groups[t1],
                "split": split_of_text[t1],
                "raw_a": t1,
                "raw_b": t2,
                "text_a": normalise_text(t1),
                "text_b": normalise_text(t2),
                "aug_ops_a": "",
                "aug_ops_b": "",
                "mutation_kind": "real_conflict",
                "mutation_group": "real_conflict",
                "mutation_old": "",
                "mutation_new": "",
                "mutation_unseen_by_features": False,
                "node_a": f"{id1}::orig",
                "node_b": f"{id2}::orig",
                "entity_a": id1,
                "entity_b": id2,
            })

    # -------- random negatives across different commodity classes --------
    by_split: Dict[str, List[Tuple[str, str]]] = {}
    for t, sid in zip(seeds["seed_text"], seeds["corpus_id"]):
        by_split.setdefault(split_of_text[t], []).append((t, sid))
    n_random = int(len(seeds) * random_negative_ratio)
    for _ in range(n_random):
        split = rng.choices(["train", "val", "test"], weights=[0.7, 0.15, 0.15])[0]
        pool = by_split.get(split, [])
        if len(pool) < 2:
            continue
        (t1, id1), (t2, id2) = rng.sample(pool, 2)
        h1, h2 = hint_of_text.get(t1, "UNTAGGED"), hint_of_text.get(t2, "UNTAGGED")
        if h1 == h2 or "UNTAGGED" in (h1, h2) or h1 == "nan" or h2 == "nan":
            continue
        rows.append({
            "source": "random_cross_commodity_negative",
            "label": 0,
            "seed_corpus_id": id1,
            "seed_text": t1,
            "group_id": groups[t1],
            "split": split,
            "raw_a": t1,
            "raw_b": t2,
            "text_a": normalise_text(t1),
            "text_b": normalise_text(t2),
            "aug_ops_a": "",
            "aug_ops_b": "",
            "mutation_kind": "random",
            "mutation_group": "random",
            "mutation_old": "",
            "mutation_new": "",
            "mutation_unseen_by_features": False,
            "node_a": f"{id1}::orig",
            "node_b": f"{id2}::orig",
            "entity_a": id1,
            "entity_b": id2,
        })

    df = pd.DataFrame(rows)
    df = df[(df["text_a"].str.len() > 0) & (df["text_b"].str.len() > 0)].reset_index(drop=True)
    df.insert(0, "pair_uid", [f"SSP_{i:06d}" for i in range(len(df))])
    return df
