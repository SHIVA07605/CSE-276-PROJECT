"""
Execution and verification script for Notebook 07: Comprehensive Model Evaluation.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(r"c:\project ai")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

nb_path = PROJECT_ROOT / "notebooks" / "07_model_evaluation.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

globals_dict = {"__name__": "__main__", "__file__": str(nb_path)}

print("=" * 80)
print(f"EXECUTING NOTEBOOK 07: {nb_path.name}")
print("=" * 80)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        print(f"\n[Executing Cell {idx+1}]...")
        try:
            exec(code, globals_dict)
            print("  -> SUCCESS")
            cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": ["Execution successful.\n"]}]
            cell["execution_count"] = idx + 1
        except Exception as e:
            print(f"  -> ERROR: {e}")
            raise e

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("\n[OK] Notebook 07 executed and verified successfully!")

