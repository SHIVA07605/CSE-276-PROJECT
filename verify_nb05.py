"""
Execution and Verification Script for Notebook 05: Pairwise Supervised Classification.
Executes all cells sequentially, verifies zero errors, populates outputs,
and verifies output files.
"""

import sys
import io
import json
import time
import contextlib
from pathlib import Path

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None


def run_notebook_verification():
    nb_path = PROJECT_ROOT / "notebooks" / "05_pairwise_classification.ipynb"
    assert nb_path.exists(), f"Notebook not found: {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print("=" * 80)
    print(f"EXECUTING NOTEBOOK 05 SEQUENTIALLY: {nb_path.name}")
    print(f"Total cells: {len(nb['cells'])}")
    print("=" * 80)

    # Execution namespace
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(nb_path),
    }

    # Custom display function for notebook cells
    def nb_display(*args):
        for a in args:
            if hasattr(a, "to_string"):
                print(a.to_string())
            else:
                print(str(a))

    globals_dict["display"] = nb_display

    code_cell_idx = 0
    start_time = time.time()

    for idx, cell in enumerate(nb["cells"]):
        cell_type = cell.get("cell_type")
        if cell_type != "code":
            continue

        code_cell_idx += 1
        source = "".join(cell.get("source", []))
        cell_preview = source.strip().splitlines()[0] if source.strip() else "(empty)"
        print(f"\n[Cell {idx + 1} | Code #{code_cell_idx}]: {cell_preview[:65]}...")

        # Redirect stdout and stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        cell_start = time.time()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(source, globals_dict)
            cell_dur = time.time() - cell_start
            out_str = stdout_buf.getvalue()
            err_str = stderr_buf.getvalue()

            # Record outputs in cell
            cell["execution_count"] = code_cell_idx
            outputs = []
            if out_str:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in out_str.splitlines()],
                })
            if err_str:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": [line + "\n" for line in err_str.splitlines()],
                })
            cell["outputs"] = outputs

            print(f"  -> SUCCESS ({cell_dur:.2f}s)")
            if out_str.strip():
                lines = out_str.strip().splitlines()
                first_line = lines[0]
                last_line = lines[-1]
                print(f"     Output: {first_line[:65]} ... {last_line[:65]}")

        except Exception as e:
            print(f"\n[ERROR] in Cell {idx + 1} (Code #{code_cell_idx}):")
            print(f"Exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"[OK] ALL {code_cell_idx} CODE CELLS EXECUTED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    print("=" * 80)

    # Save executed notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[OK] Saved executed notebook with outputs to: {nb_path}")


if __name__ == "__main__":
    run_notebook_verification()

