"""
Execution verification script for Notebook 03
"""

import json
import sys
import traceback
from pathlib import Path

nb_path = Path("notebooks/03_baseline_matching.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total cells in Notebook 03: {len(cells)}")
code_cells = [c for c in cells if c.get("cell_type") == "code"]
print(f"Code cells to execute: {len(code_cells)}")

global_ns = {"__file__": str(nb_path.resolve())}
executed = 0

for i, cell in enumerate(cells):
    if cell.get("cell_type") == "code":
        executed += 1
        source = "".join(cell.get("source", []))
        first_line = source.strip().split("\n")[0] if source.strip() else "(empty)"
        print(f"Executing Code Cell {executed}/{len(code_cells)} [Cell #{i}]: {first_line[:50]}...")
        try:
            exec(source, global_ns)
            print(f"  -> OK")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            traceback.print_exc()
            sys.exit(1)

print(f"\nSUCCESS: All {executed} code cells in Notebook 03 executed without error!")

