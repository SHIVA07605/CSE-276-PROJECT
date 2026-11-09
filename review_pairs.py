"""
Interactive Terminal Reviewer Tool for Pairwise Material Entity Resolution.

Allows human reviewers to:
- Review candidate material pairs side-by-side with full technical details.
- View raw descriptions, preprocessed descriptions, enterprise names, similarities, and extracted attributes.
- Inspect attribute conflicts and confirmations.
- Assign labels strictly from {'1', '0', 'UNCERTAIN', 'PENDING'}.
- Add engineering reviewer notes and confidence ratings.
- Jump to specific pairs, revisit and edit past annotations, filter views.
- Save progress automatically and export validated labelled CSVs.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, DATA_LABELLED_DIR
from validate_annotations import (
    validate_worksheet_integrity,
    compute_training_readiness,
    export_labelled_dataset,
    format_readiness_report_text,
)


def clear_screen():
    # Only clear if terminal is interactive
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def display_pair_card(df: pd.DataFrame, idx: int, total_count: int):
    row = df.iloc[idx]
    pair_id = row.get("pair_id", f"PAIR_{idx+1:03d}")
    label = str(row.get("human_label", "PENDING")).strip().upper()
    conf = str(row.get("confidence_rating", "PENDING")).strip().upper()
    notes = str(row.get("reviewer_notes", "")).strip()
    if notes == "nan":
        notes = ""

    # Stats
    all_labels = df["human_label"].fillna("PENDING").astype(str).str.strip().str.upper()
    n_pos = int((all_labels == "1").sum())
    n_neg = int((all_labels == "0").sum())
    n_unc = int((all_labels == "UNCERTAIN").sum())
    n_pen = int((all_labels == "PENDING").sum())

    print("=" * 80)
    print(f"PAIR {idx + 1} OF {total_count}  |  ID: {pair_id}")
    print(f"PROGRESS: [1]={n_pos} | [0]={n_neg} | [UNCERTAIN]={n_unc} | [PENDING]={n_pen}")
    print("=" * 80)

    # Material 1
    print("\n[MATERIAL A]")
    print(f"  Enterprise:      {row.get('enterprise_1', 'N/A')}")
    print(f"  Corpus ID:       {row.get('corpus_id_1', 'N/A')}")
    print(f"  Raw Text:        {row.get('raw_description_1', 'N/A')}")
    print(f"  Preprocessed:    {row.get('cleaned_description_1', 'N/A')}")
    print(f"  Extracted Attrs: {row.get('extracted_attributes_1', 'None detected')}")

    # Material 2
    print("\n[MATERIAL B]")
    print(f"  Enterprise:      {row.get('enterprise_2', 'N/A')}")
    print(f"  Corpus ID:       {row.get('corpus_id_2', 'N/A')}")
    print(f"  Raw Text:        {row.get('raw_description_2', 'N/A')}")
    print(f"  Preprocessed:    {row.get('cleaned_description_2', 'N/A')}")
    print(f"  Extracted Attrs: {row.get('extracted_attributes_2', 'None detected')}")

    # Metrics & Engineering checks
    lex_sim = float(row.get("lexical_similarity", 0.0))
    sem_sim = float(row.get("semantic_cosine_similarity", 0.0))
    is_exact = bool(row.get("is_exact_normalized_match", False))
    has_conflict = bool(row.get("has_attribute_conflict", False))
    conflict_det = str(row.get("conflict_details", "")).strip()
    confirm_det = str(row.get("confirmation_details", "")).strip()
    guidance = str(row.get("review_guidance", "")).strip()

    print("\n[SIMILARITY & ENGINEERING CHECKS]")
    print(f"  Lexical Sim (TF-IDF):        {lex_sim:.4f}")
    print(f"  Semantic Sim (Transformer):   {sem_sim:.4f}")
    print(f"  Similarity Discrepancy:       {sem_sim - lex_sim:+.4f}")
    print(f"  Exact Match Verbatim:         {'YES' if is_exact else 'NO'}")

    if has_conflict:
        print(f"  Attribute Conflict:           [!] WARNING: CONFLICT DETECTED")
        print(f"  Conflict Details:             {conflict_det if conflict_det and conflict_det != 'nan' else 'Physical attribute mismatch'}")
    else:
        print(f"  Attribute Conflict:           No conflicts detected")

    if confirm_det and confirm_det != "nan":
        print(f"  Confirmed Matching Attrs:     {confirm_det}")

    if guidance and guidance != "nan":
        print(f"  Sampling Context:             {guidance}")

    # Current Annotation State
    print("\n[CURRENT ANNOTATION]")
    status_symbol = {
        "1": "[1] FUNCTIONALLY EQUIVALENT",
        "0": "[0] NOT EQUIVALENT",
        "UNCERTAIN": "[?] UNCERTAIN / INSUFFICIENT EVIDENCE",
        "PENDING": "[...] PENDING REVIEW",
    }.get(label, f"[{label}]")

    print(f"  Label:             {status_symbol}")
    print(f"  Confidence:        {conf}")
    print(f"  Reviewer Notes:    {notes if notes else '(none)'}")
    print("-" * 80)


def run_interactive_reviewer(worksheet_path: Optional[Path] = None):
    """Run interactive terminal session."""
    if worksheet_path is None:
        worksheet_path = DATA_PROCESSED_DIR / "improved_labelled_pairs_worksheet.csv"

    if not worksheet_path.exists():
        print(f"[ERROR] Worksheet file not found: {worksheet_path}")
        sys.exit(1)

    df = pd.read_csv(worksheet_path)
    for col in ["reviewer_notes", "confirmation_details", "conflict_details", "human_label", "confidence_rating"]:
        if col in df.columns:
            df[col] = df[col].astype(object)
    total_pairs = len(df)

    # Validate before starting
    is_valid, errors, warnings = validate_worksheet_integrity(df)
    if not is_valid:
        print(f"[ERROR] Worksheet integrity validation failed: {errors}")
        sys.exit(1)

    current_idx = 0
    # Find first pending item
    pending_indices = df.index[df["human_label"].astype(str).str.strip().str.upper() == "PENDING"].tolist()
    if pending_indices:
        current_idx = pending_indices[0]

    active_indices = list(range(total_pairs))
    filter_mode = "ALL"

    print("\n" + "=" * 80)
    print("WELCOME TO THE CSE 276 MATERIAL CODE HARMONISATION REVIEWER")
    print("=" * 80)
    print("Use this tool to assign ground-truth labels for pairwise entity resolution.")
    print("Labels: [1] Equivalent, [0] Distinct, [U] Uncertain, [P] Pending")
    print("Progress is saved automatically to the worksheet.")
    print("=" * 80)

    while True:
        if not active_indices:
            print("\n[INFO] No pairs match the active filter.")
            active_indices = list(range(total_pairs))
            filter_mode = "ALL"
            current_idx = 0

        # Ensure current_idx is valid
        if current_idx >= len(active_indices):
            current_idx = len(active_indices) - 1
        if current_idx < 0:
            current_idx = 0

        actual_df_idx = active_indices[current_idx]
        display_pair_card(df, actual_df_idx, total_pairs)

        print("\nOPTIONS:")
        print("  [1] Mark EQUIVALENT      [0] Mark DISTINCT       [U] Mark UNCERTAIN")
        print("  [N] Add/Edit Notes       [C] Set Confidence      [P] Reset to PENDING")
        print("  [ENTER/S] Next Pair      [B] Previous Pair       [J] Jump to Pair #")
        print("  [F] Filter View          [V] Readiness Report    [E] Export Labelled CSV")
        print("  [Q] Save & Quit")

        try:
            choice = input("\nSelect action: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting reviewer. Progress saved.")
            break

        if choice == "q":
            df.to_csv(worksheet_path, index=False)
            print(f"\n[OK] Progress saved to {worksheet_path.name}. Goodbye!")
            break

        elif choice == "1":
            df.at[actual_df_idx, "human_label"] = "1"
            if str(df.at[actual_df_idx, "confidence_rating"]).strip().upper() == "PENDING":
                df.at[actual_df_idx, "confidence_rating"] = "HIGH"
            df.to_csv(worksheet_path, index=False)
            print("[OK] Labelled as 1 (Equivalent). Saved.")
            if current_idx < len(active_indices) - 1:
                current_idx += 1

        elif choice == "0":
            df.at[actual_df_idx, "human_label"] = "0"
            if str(df.at[actual_df_idx, "confidence_rating"]).strip().upper() == "PENDING":
                df.at[actual_df_idx, "confidence_rating"] = "HIGH"
            df.to_csv(worksheet_path, index=False)
            print("[OK] Labelled as 0 (Not Equivalent). Saved.")
            if current_idx < len(active_indices) - 1:
                current_idx += 1

        elif choice == "u":
            df.at[actual_df_idx, "human_label"] = "UNCERTAIN"
            if str(df.at[actual_df_idx, "confidence_rating"]).strip().upper() == "PENDING":
                df.at[actual_df_idx, "confidence_rating"] = "LOW"
            # Prompt for note if blank
            cur_note = str(df.at[actual_df_idx, "reviewer_notes"]).strip()
            if not cur_note or cur_note == "nan":
                reason = input("Enter reason for UNCERTAIN (e.g., generic description, missing spec): ").strip()
                if reason:
                    df.at[actual_df_idx, "reviewer_notes"] = reason
            df.to_csv(worksheet_path, index=False)
            print("[OK] Labelled as UNCERTAIN. Saved.")
            if current_idx < len(active_indices) - 1:
                current_idx += 1

        elif choice == "p":
            df.at[actual_df_idx, "human_label"] = "PENDING"
            df.at[actual_df_idx, "confidence_rating"] = "PENDING"
            df.to_csv(worksheet_path, index=False)
            print("[OK] Reset to PENDING. Saved.")

        elif choice == "n":
            current_note = str(df.at[actual_df_idx, "reviewer_notes"])
            if current_note == "nan":
                current_note = ""
            print(f"Current Note: {current_note if current_note else '(none)'}")
            new_note = input("Enter new reviewer notes (leave empty to cancel): ").strip()
            if new_note:
                df.at[actual_df_idx, "reviewer_notes"] = new_note
                df.to_csv(worksheet_path, index=False)
                print("[OK] Notes updated.")

        elif choice == "c":
            print("Confidence options: [H] High  [M] Medium  [L] Low")
            conf_choice = input("Select confidence (H/M/L): ").strip().upper()
            mapping = {"H": "HIGH", "M": "MEDIUM", "L": "LOW"}
            if conf_choice in mapping:
                df.at[actual_df_idx, "confidence_rating"] = mapping[conf_choice]
                df.to_csv(worksheet_path, index=False)
                print(f"[OK] Confidence set to {mapping[conf_choice]}.")
            else:
                print("[WARN] Invalid confidence option.")

        elif choice in ["", "s"]:
            if current_idx < len(active_indices) - 1:
                current_idx += 1
            else:
                print("[INFO] Reached the end of the current list.")

        elif choice == "b":
            if current_idx > 0:
                current_idx -= 1
            else:
                print("[INFO] Already at the first pair.")

        elif choice == "j":
            target = input(f"Enter pair number (1-{total_pairs}) or pair_id: ").strip()
            found = False
            if target.isdigit():
                tgt_num = int(target)
                if 1 <= tgt_num <= total_pairs:
                    target_df_idx = tgt_num - 1
                    if target_df_idx in active_indices:
                        current_idx = active_indices.index(target_df_idx)
                    else:
                        print(f"[INFO] Pair {tgt_num} is hidden by filter {filter_mode}. Resetting filter to ALL.")
                        active_indices = list(range(total_pairs))
                        filter_mode = "ALL"
                        current_idx = target_df_idx
                    found = True
            else:
                # Match by pair_id
                matched = df.index[df["pair_id"].astype(str).str.upper() == target.upper()].tolist()
                if matched:
                    target_df_idx = matched[0]
                    if target_df_idx in active_indices:
                        current_idx = active_indices.index(target_df_idx)
                    else:
                        active_indices = list(range(total_pairs))
                        filter_mode = "ALL"
                        current_idx = target_df_idx
                    found = True

            if not found:
                print(f"[WARN] Pair '{target}' not found.")

        elif choice == "f":
            print("\nFilter options:")
            print("  [1] ALL (show all pairs)")
            print("  [2] PENDING only")
            print("  [3] REVIEWED only (1, 0, or UNCERTAIN)")
            print("  [4] ATTRIBUTE CONFLICTS only")
            print("  [5] POSITIVE MATCHES only (1)")
            print("  [6] NEGATIVE MATCHES only (0)")
            print("  [7] UNCERTAIN only")
            f_choice = input("Select filter (1-7): ").strip()

            labels_series = df["human_label"].fillna("PENDING").astype(str).str.strip().str.upper()
            if f_choice == "1":
                active_indices = list(range(total_pairs))
                filter_mode = "ALL"
            elif f_choice == "2":
                active_indices = df.index[labels_series == "PENDING"].tolist()
                filter_mode = "PENDING"
            elif f_choice == "3":
                active_indices = df.index[labels_series.isin(["1", "0", "UNCERTAIN"])].tolist()
                filter_mode = "REVIEWED"
            elif f_choice == "4":
                active_indices = df.index[df["has_attribute_conflict"].astype(bool)].tolist()
                filter_mode = "CONFLICTS"
            elif f_choice == "5":
                active_indices = df.index[labels_series == "1"].tolist()
                filter_mode = "POSITIVE (1)"
            elif f_choice == "6":
                active_indices = df.index[labels_series == "0"].tolist()
                filter_mode = "NEGATIVE (0)"
            elif f_choice == "7":
                active_indices = df.index[labels_series == "UNCERTAIN"].tolist()
                filter_mode = "UNCERTAIN"
            else:
                print("[WARN] Invalid filter option.")

            print(f"[OK] Filter applied: {filter_mode} ({len(active_indices)} pairs matched).")
            current_idx = 0

        elif choice == "v":
            report = compute_training_readiness(df)
            print("\n" + format_readiness_report_text(report))
            input("\nPress ENTER to return to review...")

        elif choice == "e":
            export_path = export_labelled_dataset(df)
            print(f"\n[OK] Successfully exported full labelled dataset to: {export_path}")
            # Also offer confirmed-only export
            n_binary = int((df["human_label"].astype(str).str.strip().str.upper().isin(["1", "0"])).sum())
            if n_binary > 0:
                bin_path = DATA_LABELLED_DIR / "labelled_pairs_binary_only.csv"
                export_labelled_dataset(df, bin_path, confirmed_only=True)
                print(f"[OK] Exported {n_binary} confirmed binary pairs to: {bin_path}")
            input("\nPress ENTER to continue...")

        else:
            print("[WARN] Unrecognized option. Please choose from the menu.")


if __name__ == "__main__":
    run_interactive_reviewer()

