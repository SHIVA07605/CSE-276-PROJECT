"""
Annotation Validation and Training Readiness Reporting Module.

Verifies:
1. Data integrity (preservation of corpus IDs, unique pair IDs, feature schemas).
2. Permitted label validation (strictly '1', '0', 'UNCERTAIN', 'PENDING').
3. Preservation of reviewer notes and confidence ratings.
4. Training readiness assessment across all 10 required project dimensions.
5. Export utility for verified human-labelled pairs.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, DATA_LABELLED_DIR

VALID_LABELS = {"1", "0", "UNCERTAIN", "PENDING"}
VALID_CONFIDENCES = {"HIGH", "MEDIUM", "LOW", "PENDING"}


def validate_worksheet_integrity(
    df: pd.DataFrame,
    original_template_path: Optional[Path] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate data integrity and label schemas of the annotation worksheet.

    Checks:
    - Required columns present.
    - Every pair is uniquely identifiable (no duplicate pair_id).
    - Valid label domain ('1', '0', 'UNCERTAIN', 'PENDING').
    - Non-null corpus IDs and descriptions.
    - Verification against original template features if provided.

    Returns:
        Tuple of (is_valid: bool, errors: List[str], warnings: List[str]).
    """
    errors: List[str] = []
    warnings: List[str] = []

    required_cols = [
        "pair_id",
        "corpus_id_1",
        "enterprise_1",
        "raw_description_1",
        "cleaned_description_1",
        "extracted_attributes_1",
        "corpus_id_2",
        "enterprise_2",
        "raw_description_2",
        "cleaned_description_2",
        "extracted_attributes_2",
        "lexical_similarity",
        "semantic_cosine_similarity",
        "is_exact_normalized_match",
        "has_attribute_conflict",
        "conflict_details",
        "confirmation_details",
        "human_label",
        "confidence_rating",
        "reviewer_notes",
    ]

    # 1. Column presence check
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required schema column: '{col}'")

    if errors:
        return False, errors, warnings

    # 2. Uniqueness of pair identifiers
    if df["pair_id"].duplicated().any():
        dups = df[df["pair_id"].duplicated()]["pair_id"].tolist()
        errors.append(f"Duplicate pair identifiers detected: {dups}")

    # 3. Non-null IDs
    if df["corpus_id_1"].isna().any() or df["corpus_id_2"].isna().any():
        errors.append("Null corpus IDs detected in candidate pairs.")

    # 4. Label validity check
    cleaned_labels = df["human_label"].fillna("PENDING").astype(str).str.strip().str.upper()
    # Normalize '1.0' and '0.0' to '1' and '0'
    cleaned_labels = cleaned_labels.replace({"1.0": "1", "0.0": "0"})
    invalid_labels = set(cleaned_labels) - VALID_LABELS
    if invalid_labels:
        errors.append(
            f"Invalid label values detected: {invalid_labels}. Permitted set: {VALID_LABELS}"
        )

    # 5. Confidence rating validity check
    if "confidence_rating" in df.columns:
        cleaned_conf = (
            df["confidence_rating"].fillna("PENDING").astype(str).str.strip().str.upper()
        )
        invalid_conf = set(cleaned_conf) - VALID_CONFIDENCES
        if invalid_conf:
            warnings.append(
                f"Unstandardized confidence values: {invalid_conf}. Expected: {VALID_CONFIDENCES}"
            )

    # 6. Similarity range sanity check
    if "lexical_similarity" in df.columns:
        lex = df["lexical_similarity"].astype(float)
        if (lex < 0.0).any() or (lex > 1.0001).any():
            errors.append("Lexical similarity scores outside valid range [0.0, 1.0].")

    if "semantic_cosine_similarity" in df.columns:
        sem = df["semantic_cosine_similarity"].astype(float)
        if (sem < -1.0001).any() or (sem > 1.0001).any():
            errors.append("Semantic similarity scores outside valid range [-1.0, 1.0].")

    # 7. Check template preservation if original exists
    if original_template_path is not None and original_template_path.exists():
        df_orig = pd.read_csv(original_template_path)
        if len(df) != len(df_orig):
            warnings.append(
                f"Row count mismatch with original template: {len(df)} vs {len(df_orig)}"
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def compute_training_readiness(
    df: pd.DataFrame,
    planning_target: int = 150,
    min_pilot_labels: int = 30,
) -> Dict[str, Any]:
    """
    Compute structured training readiness metrics across all 10 project dimensions.

    Args:
        df: Annotation DataFrame.
        planning_target: Recommended planning target for robust training (default 150-200).
        min_pilot_labels: Minimum binary labels for pilot notebook execution (default 30).

    Returns:
        Dictionary containing all 10 required metrics and a readiness determination.
    """
    labels = df["human_label"].fillna("PENDING").astype(str).str.strip().str.upper()
    labels = labels.replace({"1.0": "1", "0.0": "0"})

    total_pairs = len(df)
    n_positive = int((labels == "1").sum())
    n_negative = int((labels == "0").sum())
    n_uncertain = int((labels == "UNCERTAIN").sum())
    n_pending = int((labels == "PENDING").sum())

    n_reviewed = n_positive + n_negative + n_uncertain
    n_confirmed_binary = n_positive + n_negative

    # Cross-enterprise reviewed count
    c_org1 = "enterprise_1" if "enterprise_1" in df.columns else "organization_1"
    c_org2 = "enterprise_2" if "enterprise_2" in df.columns else "organization_2"
    is_cross = df[c_org1] != df[c_org2]
    cross_reviewed = int((is_cross & labels.isin(["1", "0", "UNCERTAIN"])).sum())
    cross_confirmed = int((is_cross & labels.isin(["1", "0"])).sum())

    # Attribute-conflict negative examples
    has_conflict = df.get("has_attribute_conflict", pd.Series(False, index=df.index)).astype(bool)
    conflict_negatives = int((has_conflict & (labels == "0")).sum())

    # Class balance calculation
    if n_confirmed_binary > 0:
        pos_ratio = round(n_positive / n_confirmed_binary, 4)
        neg_ratio = round(n_negative / n_confirmed_binary, 4)
        class_balance_str = f"{n_positive} Positives ({pos_ratio*100:.1f}%) / {n_negative} Negatives ({neg_ratio*100:.1f}%)"
    else:
        pos_ratio = 0.0
        neg_ratio = 0.0
        class_balance_str = "No confirmed binary labels yet"

    # Readiness determination
    ready_for_production = (
        (n_confirmed_binary >= planning_target)
        and (n_positive >= 25)
        and (n_negative >= 50)
        and (cross_confirmed >= 20)
    )

    ready_for_pilot = (
        (n_confirmed_binary >= min_pilot_labels)
        and (n_positive >= 5)
        and (n_negative >= 5)
    )

    if ready_for_production:
        readiness_status = "READY_FOR_PRODUCTION_TRAINING"
        readiness_reason = (
            f"Met planning target ({n_confirmed_binary}/{planning_target} confirmed binary labels) "
            "with balanced classes and cross-enterprise representation."
        )
    elif ready_for_pilot:
        readiness_status = "READY_FOR_PILOT_NOTEBOOK_RUN"
        readiness_reason = (
            f"Met minimum pilot requirement ({n_confirmed_binary}/{min_pilot_labels} binary labels). "
            f"Sub-optimal sample size for final deployment; additional labels recommended toward {planning_target}."
        )
    else:
        readiness_status = "NOT_READY_FOR_TRAINING"
        if n_confirmed_binary == 0:
            readiness_reason = (
                f"0 confirmed binary labels ({n_pending} pending, {n_uncertain} uncertain). "
                f"Model training strictly suspended pending human annotation."
            )
        else:
            readiness_reason = (
                f"Only {n_confirmed_binary} confirmed binary labels (need at least {min_pilot_labels} for pilot, "
                f"{planning_target} for production)."
            )

    return {
        "1_total_candidate_pairs": total_pairs,
        "2_number_reviewed": n_reviewed,
        "3_number_pending": n_pending,
        "4_positive_labels": n_positive,
        "5_negative_labels": n_negative,
        "6_uncertain_labels": n_uncertain,
        "7_cross_enterprise_reviewed_pairs": cross_reviewed,
        "7b_cross_enterprise_confirmed_pairs": cross_confirmed,
        "8_attribute_conflict_negatives": conflict_negatives,
        "9_class_balance": {
            "confirmed_binary_total": n_confirmed_binary,
            "positive_count": n_positive,
            "negative_count": n_negative,
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "summary_text": class_balance_str,
        },
        "10_training_readiness": {
            "status": readiness_status,
            "reason": readiness_reason,
            "is_ready_for_training": bool(ready_for_pilot or ready_for_production),
            "planning_target": planning_target,
            "progress_to_target_pct": round(n_confirmed_binary / planning_target * 100, 1),
        },
    }


def export_labelled_dataset(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    confirmed_only: bool = False,
) -> Path:
    """
    Export validated annotations to a standardized destination file.

    Args:
        df: Validated annotation DataFrame.
        output_path: Destination path (default: data/labelled/labelled_pairs.csv).
        confirmed_only: If True, exports only pairs with binary label in {'1', '0'}.

    Returns:
        Path to the exported CSV file.
    """
    if output_path is None:
        DATA_LABELLED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_LABELLED_DIR / "labelled_pairs.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()

    if confirmed_only:
        labels = export_df["human_label"].astype(str).str.strip().str.upper()
        export_df = export_df[labels.isin(["1", "0", "1.0", "0.0"])].copy()

    export_df.to_csv(output_path, index=False)
    return output_path


def format_readiness_report_text(report: Dict[str, Any]) -> str:
    """Format readiness metrics as a clean terminal / markdown report."""
    lines = []
    lines.append("=" * 75)
    lines.append("CSE 276 -- HUMAN ANNOTATION & TRAINING READINESS REPORT")
    lines.append("=" * 75)
    lines.append(f"1. Total Candidate Pairs:             {report['1_total_candidate_pairs']}")
    lines.append(f"2. Number Reviewed:                   {report['2_number_reviewed']} ({report['2_number_reviewed']/report['1_total_candidate_pairs']*100:.1f}%)")
    lines.append(f"3. Number Pending:                    {report['3_number_pending']} ({report['3_number_pending']/report['1_total_candidate_pairs']*100:.1f}%)")
    lines.append(f"4. Positive Labels (1 - Equivalent):   {report['4_positive_labels']}")
    lines.append(f"5. Negative Labels (0 - Distinct):     {report['5_negative_labels']}")
    lines.append(f"6. Uncertain Labels (Ambiguous):       {report['6_uncertain_labels']}")
    lines.append(f"7. Cross-Enterprise Reviewed Pairs:   {report['7_cross_enterprise_reviewed_pairs']}")
    lines.append(f"   - Cross-Enterprise Confirmed:      {report['7b_cross_enterprise_confirmed_pairs']}")
    lines.append(f"8. Attribute-Conflict Negatives:      {report['8_attribute_conflict_negatives']}")
    lines.append(f"9. Class Balance:                     {report['9_class_balance']['summary_text']}")
    lines.append("-" * 75)
    readiness = report["10_training_readiness"]
    lines.append(f"10. Supervised Training Readiness:    [{readiness['status']}]")
    lines.append(f"    - Target Benchmark:               {readiness['planning_target']} confirmed labels ({readiness['progress_to_target_pct']}% reached)")
    lines.append(f"    - Determination Rationale:        {readiness['reason']}")
    lines.append(f"    - Model Training Permitted:       {readiness['is_ready_for_training']}")
    lines.append("=" * 75)
    return "\n".join(lines)


def main():
    worksheet_path = DATA_PROCESSED_DIR / "improved_labelled_pairs_worksheet.csv"
    template_path = DATA_PROCESSED_DIR / "labelled_pairs_template.csv"

    if not worksheet_path.exists():
        print(f"[ERROR] Worksheet not found: {worksheet_path}")
        sys.exit(1)

    df_ws = pd.read_csv(worksheet_path)
    is_valid, errors, warnings = validate_worksheet_integrity(df_ws, template_path)

    print(f"Loading Worksheet: {worksheet_path.name}")
    print(f"Data Validation Status: {'PASSED' if is_valid else 'FAILED'}")

    if errors:
        print("\nValidation Errors:")
        for e in errors:
            print(f"  [ERROR] {e}")

    if warnings:
        print("\nValidation Warnings:")
        for w in warnings:
            print(f"  [WARN] {w}")

    report = compute_training_readiness(df_ws)
    print("\n" + format_readiness_report_text(report))

    # Optional export if --export flag is passed
    if "--export" in sys.argv:
        exported_path = export_labelled_dataset(df_ws)
        print(f"\n[OK] Exported labelled dataset to: {exported_path}")


if __name__ == "__main__":
    main()
