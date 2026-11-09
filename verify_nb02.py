"""
Comprehensive Verification Script for Notebook 02:
1. Executes all code cells in notebooks/02_data_preprocessing.ipynb.
2. Checks dataset integrity across raw, preprocessed, and held-back datasets.
3. Finds real representative records for the attribute extraction audit.
"""

import sys
import os
import json
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(".").resolve()

print("=" * 70)
print("PART 1: NOTEBOOK 02 CELL-BY-CELL EXECUTION VERIFICATION")
print("=" * 70)

nb_path = PROJECT_ROOT / "notebooks" / "02_data_preprocessing.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total cells in notebook: {len(cells)}")
code_cells = [c for c in cells if c.get("cell_type") == "code"]
print(f"Code cells to execute: {len(code_cells)}")

global_ns = {"__file__": str(nb_path)}
executed_count = 0

for i, cell in enumerate(cells):
    if cell.get("cell_type") == "code":
        executed_count += 1
        source = "".join(cell.get("source", []))
        first_line = source.strip().split("\n")[0] if source.strip() else "(empty)"
        print(f"Executing Code Cell {executed_count}/{len(code_cells)} [Cell #{i}]: {first_line[:60]}...")
        try:
            exec(source, global_ns)
            print(f"  -> SUCCESS")
        except Exception as e:
            print(f"  -> ERROR in Cell {executed_count} [Cell #{i}]: {e}")
            traceback.print_exc()
            sys.exit(1)

print(f"\nAll {executed_count} code cells executed with 0 errors!")

print("\n" + "=" * 70)
print("PART 2: DATASET INTEGRITY VERIFICATION")
print("=" * 70)

raw_path = PROJECT_ROOT / "data" / "raw" / "material_description_corpus.csv"
proc_path = PROJECT_ROOT / "data" / "processed" / "preprocessed_materials.csv"
held_path = PROJECT_ROOT / "data" / "processed" / "held_back_records.csv"

df_raw = pd.read_csv(raw_path)
df_proc = pd.read_csv(proc_path)
df_held = pd.read_csv(held_path)

print(f"Original dataset records   : {len(df_raw):,} (expected 21,513)")
print(f"Processed candidate records: {len(df_proc):,} (expected 6,634)")
print(f"Held-back records          : {len(df_held):,} (expected 14,879)")
print(f"Processed + Held-back sum  : {len(df_proc) + len(df_held):,} (expected 21,513)")

assert len(df_raw) == 21513, f"df_raw count mismatch: {len(df_raw)}"
assert len(df_proc) == 6634, f"df_proc count mismatch: {len(df_proc)}"
assert len(df_held) == 14879, f"df_held count mismatch: {len(df_held)}"

# Check duplicate corpus_ids
raw_dup_ids = df_raw["corpus_id"].duplicated().sum()
proc_dup_ids = df_proc["corpus_id"].duplicated().sum()
held_dup_ids = df_held["corpus_id"].duplicated().sum()

print(f"Duplicate corpus_ids in raw        : {raw_dup_ids}")
print(f"Duplicate corpus_ids in processed  : {proc_dup_ids}")
print(f"Duplicate corpus_ids in held-back  : {held_dup_ids}")

# Missing IDs check
raw_null_ids = df_raw["corpus_id"].isnull().sum()
proc_null_ids = df_proc["corpus_id"].isnull().sum()
held_null_ids = df_held["corpus_id"].isnull().sum()

print(f"Missing (null) corpus_ids in raw       : {raw_null_ids}")
print(f"Missing (null) corpus_ids in processed : {proc_null_ids}")
print(f"Missing (null) corpus_ids in held-back : {held_null_ids}")

# Check overlap
set_proc = set(df_proc["corpus_id"])
set_held = set(df_held["corpus_id"])
set_raw = set(df_raw["corpus_id"])

overlap = set_proc.intersection(set_held)
print(f"Overlap between processed & held-back  : {len(overlap)} (expected 0)")
assert len(overlap) == 0, f"Found overlap: {len(overlap)}"

# Check union matches raw
union_set = set_proc.union(set_held)
is_exact_match = (union_set == set_raw)
print(f"Union equals raw set exactly           : {is_exact_match}")
assert is_exact_match, "Union of processed and held-back does not match raw corpus IDs"

# Confirm raw_description matches original description exactly
# Build mapping from corpus_id to original description
raw_desc_map = dict(zip(df_raw["corpus_id"], df_raw["description"].fillna("").astype(str)))
proc_desc_match = True
mismatch_count = 0
for _, row in df_proc.iterrows():
    cid = row["corpus_id"]
    orig = raw_desc_map[cid]
    curr_raw = str(row["raw_description"]) if pd.notnull(row["raw_description"]) else ""
    if orig != curr_raw:
        mismatch_count += 1
        proc_desc_match = False

print(f"raw_description mismatches against original description: {mismatch_count}")
assert proc_desc_match, f"Found {mismatch_count} mismatches in raw_description"

print("Integrity verification: PASSED (Every record accounted for exactly once, zero modifications/discards).")

print("\n" + "=" * 70)
print("PART 3: FINDING REPRESENTATIVE REAL RECORDS FOR AUDIT")
print("=" * 70)

# We want 20 representative real records covering:
# 1. Diameter (MM, INCH)
# 2. Metric Thread (M10, M12, etc.)
# 3. DN notation (DN50, DN100, etc.)
# 4. Material grade (SS304, SS316, IS2062, ASTM)
# 5. Pressure rating (150#, PN16, Class 150)
# 6. Pipe schedule (SCH 40, SCH 80)
# 7. Voltage (11KV, 415V, etc.)
# 8. Capacity (100KVA, 50KW, etc.)
# 9. Ambiguous specifications or limitations

queries = [
    ("Diameter MM", df_proc[df_proc["attr_diameter"].str.contains("MM", na=False)]),
    ("Diameter INCH", df_proc[df_proc["attr_diameter"].str.contains("INCH", na=False)]),
    ("Metric Thread", df_proc[df_proc["attr_metric_thread"].notnull()]),
    ("DN notation", df_proc[df_proc["attr_diameter"].str.startswith("DN", na=False)]),
    ("Material Grade SS304/SS316", df_proc[df_proc["attr_material_grade"].str.contains("SS", na=False)]),
    ("Material Grade IS2062", df_proc[df_proc["attr_material_grade"].str.contains("IS2062", na=False)]),
    ("Material Grade ASTM", df_proc[df_proc["attr_material_grade"].str.contains("ASTM", na=False)]),
    ("Pressure 150#/300#", df_proc[df_proc["attr_pressure_class"].str.contains("#", na=False)]),
    ("Pressure PN16/PN40", df_proc[df_proc["attr_pressure_class"].str.contains("PN", na=False)]),
    ("Pressure CLASS", df_proc[df_proc["attr_pressure_class"].str.contains("CLASS", na=False)]),
    ("Pipe Schedule", df_proc[df_proc["attr_schedule"].notnull()]),
    ("Voltage", df_proc[df_proc["attr_voltage"].notnull()]),
    ("Capacity", df_proc[df_proc["attr_capacity"].notnull()]),
    ("Ambiguous / Multiple Attributes", df_proc[df_proc["attr_diameter"].notnull() & df_proc["attr_pressure_class"].notnull()]),
]

print("Available counts per query:")
for name, sub in queries:
    print(f"  {name:<35}: {len(sub):,} records")

# Save a json of 25 diverse real records for inspection
sample_records = []
seen_ids = set()

def add_records(sub_df, count=2):
    for _, r in sub_df.iterrows():
        if r["corpus_id"] not in seen_ids:
            seen_ids.add(r["corpus_id"])
            sample_records.append({
                "corpus_id": str(r["corpus_id"]),
                "organization": str(r["organization"]),
                "raw_description": str(r["raw_description"]),
                "cleaned_description": str(r["cleaned_description"]),
                "attr_metric_thread": r["attr_metric_thread"] if pd.notnull(r["attr_metric_thread"]) else None,
                "attr_diameter": r["attr_diameter"] if pd.notnull(r["attr_diameter"]) else None,
                "attr_material_grade": r["attr_material_grade"] if pd.notnull(r["attr_material_grade"]) else None,
                "attr_pressure_class": r["attr_pressure_class"] if pd.notnull(r["attr_pressure_class"]) else None,
                "attr_schedule": r["attr_schedule"] if pd.notnull(r["attr_schedule"]) else None,
                "attr_voltage": r["attr_voltage"] if pd.notnull(r["attr_voltage"]) else None,
                "attr_capacity": r["attr_capacity"] if pd.notnull(r["attr_capacity"]) else None,
                "attr_ambiguous_flags": eval(r["attr_ambiguous_flags"]) if isinstance(r["attr_ambiguous_flags"], str) and r["attr_ambiguous_flags"].startswith("[") else []
            })
            if len(seen_ids) >= 25:
                break

for name, sub in queries:
    add_records(sub, count=2)

with open(PROJECT_ROOT / "results" / "audit_sample_records.json", "w", encoding="utf-8") as f:
    json.dump(sample_records, f, indent=2)

print(f"\nSaved {len(sample_records)} real records to results/audit_sample_records.json for manual audit.")

