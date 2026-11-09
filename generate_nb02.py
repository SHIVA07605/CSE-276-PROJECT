import json
from pathlib import Path

cells = []

def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]] if source_lines else []
    }

def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]] if source_lines else []
    }

# 1. Header
cells.append(md_cell([
    "# 02. Data Preprocessing & Engineering Attribute Extraction",
    "",
    "**Course:** CSE 276 – CA-1 Academic Project  ",
    "**Project:** Material Code Harmonisation Across Public Enterprises  ",
    "**Corpus Target:** `data/raw/material_description_corpus.csv` (21,513 records across 3 CPSEs)  ",
    "**Target Candidate Pool:** 6,634 records (Tier 1 Discrete Materials + Tier 2 Tagged Procurement Descriptions)  ",
    "",
    "---",
    "",
    "## Notebook Objectives",
    "1. **Environment Setup:** Support both Google Colab and Local Workstation environments dynamically.",
    "2. **Approved Corpus Partitioning:** Reconstruct the approved 4-tier hierarchy programmatically and strictly validate counts (Tier 1: 923, Tier 2 Tagged: 5,711, Tier 2 Untagged: 14,542, Tier 3: 337).",
    "3. **Candidate Pool Isolation:** Isolate the 6,634 candidate records for matching while preserving the remaining 14,879 records separately.",
    "4. **Text Preprocessing & Normalization:** Apply uppercase conversion, whitespace stripping, separator normalization, engineering abbreviation expansion, and procurement boilerplate removal.",
    "5. **Domain Integrity Verification:** Validate that critical engineering specifications, dimensions, standards, and terms (e.g. `PIPE`, `FLANGE`, `SE`) are strictly preserved.",
    "6. **Engineering Attribute Extraction:** Extract structured parameters (metric thread, diameter/dimension, material grade, pressure rating, schedule).",
    "7. **Diagnostic Visualisation & Export:** Generate diagnostic charts and save preprocessed candidate pool and audit reports.",
    "",
    "> **Academic Integrity Rule:** No fabricated metrics, artificial tokens, or premature ML training. All operations reflect verifiable engineering specifications."
]))

# 2. Environment Setup
cells.append(md_cell([
    "---",
    "## 1. Environment & Workspace Setup",
    "",
    "Detects Google Colab vs Local Workstation, anchors `PROJECT_ROOT`, configures `sys.path`, and imports core libraries and project modules."
]))

cells.append(code_cell([
    "# ==============================================================================",
    "# Step 1: Environment & Repository Configuration",
    "# ==============================================================================",
    "import os",
    "import sys",
    "from pathlib import Path",
    "",
    "# Detect environment",
    "IN_COLAB = 'google.colab' in sys.modules",
    "",
    "if IN_COLAB:",
    "    print('🚀 Running in Google Colab environment.')",
    "    REPO_NAME = 'CSE-276-PROJECT'",
    "    REPO_URL = 'https://github.com/SHIVA07605/CSE-276-PROJECT.git'",
    "    ",
    "    if not os.path.exists(f'/content/{REPO_NAME}'):",
    "        print(f'Cloning repository from {REPO_URL}...')",
    "        os.system(f'git clone {REPO_URL}')",
    "    ",
    "    PROJECT_ROOT = Path(f'/content/{REPO_NAME}').resolve()",
    "    os.chdir(PROJECT_ROOT)",
    "    os.system('pip install -q pandas numpy matplotlib seaborn rapidfuzz')",
    "else:",
    "    print('💻 Running in Local Workstation environment.')",
    "    current_dir = Path.cwd().resolve()",
    "    PROJECT_ROOT = current_dir.parent if current_dir.name == 'notebooks' else current_dir",
    "",
    "# Add project root to sys.path",
    "if str(PROJECT_ROOT) not in sys.path:",
    "    sys.path.insert(0, str(PROJECT_ROOT))",
    "",
    "print(f'Project Root: {PROJECT_ROOT}')",
    "print(f'sys.path configured: {str(PROJECT_ROOT) in sys.path}')",
    "",
    "# Core library imports",
    "import re",
    "import json",
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "",
    "# Project module imports",
    "from src.config import (",
    "    DATA_RAW_DIR,",
    "    DATA_PROCESSED_DIR,",
    "    FIGURES_DIR,",
    "    REPORTS_DIR,",
    "    COL_ENTERPRISE,",
    "    COL_MATERIAL_CODE,",
    "    COL_DESCRIPTION,",
    "    COL_CLEAN_DESC,",
    "    COL_EXTRACTED_ATTRIBUTES,",
    ")",
    "from src.preprocessing import (",
    "    normalise_text,",
    "    extract_material_attributes,",
    "    partition_corpus,",
    "    validate_partitions,",
    "    preprocess_catalogue,",
    "    EXPECTED_COUNTS,",
    ")",
    "",
    "print('✓ All project modules and dependencies loaded successfully.')"
]))

# 3. Load Raw Corpus
cells.append(md_cell([
    "---",
    "## 2. Load Material Description Corpus",
    "",
    "Loads the raw master dataset `data/raw/material_description_corpus.csv` consisting of 21,513 records across 3 central public enterprises (Oil India Limited, NTPC Limited, Indian Oil Corporation Limited)."
]))

cells.append(code_cell([
    "# ==============================================================================",
    "# Step 2: Ingest Master Corpus and Inspect Core Schema",
    "# ==============================================================================",
    "corpus_path = DATA_RAW_DIR / 'material_description_corpus.csv'",
    "if not corpus_path.exists():",
    "    raise FileNotFoundError(f'Corpus file not found at: {corpus_path}')",
    "",
    "df_raw = pd.read_csv(corpus_path)",
    "print(f'Raw Corpus Loaded: {df_raw.shape[0]:,} records x {df_raw.shape[1]} columns')",
    "",
    "# Verify CPSE representation",
    "org_summary = df_raw['organization'].value_counts().reset_index()",
    "org_summary.columns = ['Public Sector Enterprise (CPSE)', 'Record Count']",
    "org_summary['Percentage (%)'] = (org_summary['Record Count'] / len(df_raw) * 100).round(2)",
    "display(org_summary)",
    "",
    "# Display sample records",
    "display(df_raw[['corpus_id', 'organization', 'description_kind', 'item_type_hint', 'description']].head(5))"
]))

# 4. Programmatic Partition Reconstruction
cells.append(md_cell([
    "---",
    "## 3. Programmatic Candidate Partition Reconstruction & Strict Validation",
    "",
    "We reconstruct the approved candidate partition hierarchy:",
    "- **Tier 1 (Discrete material lines):** Individual item sentences and specification strings (`procurement_plan_item`, `doc_spec_sentence`, `doc_spec_string`). Target: **923 records**.",
    "- **Tier 2 (Tagged procurement descriptions):** Tender and procurement items annotated with domain commodity hints (`pipe`, `valve`, `pump`, `cable`, etc.). Target: **5,711 records**.",
    "- **Tier 2 (Untagged procurement descriptions):** Procurement items lacking commodity tags. Target: **14,542 records**.",
    "- **Tier 3 (Administrative / works descriptions):** High-level tender package headers, corrigenda, and work contracts. Target: **337 records**.",
    "",
    "### Initial Candidate Pool for Entity Resolution:",
    "$$\\text{Candidate Pool} = \\text{Tier 1} + \\text{Tier 2 Tagged} = 923 + 5,711 = 6,634\\text{ records}$$",
    "The remaining **14,879 records** (Untagged Tier 2 + Administrative Tier 3) are preserved separately for future stages."
]))

cells.append(code_cell([
    "# ==============================================================================",
    "# Step 3: Reconstruct Partitions Programmatically and Validate Integrity",
    "# ==============================================================================",
    "partitions = partition_corpus(df_raw)",
    "",
    "# Execute strict validation",
    "validation = validate_partitions(df_raw, partitions, strict=True)",
    "",
    "print('✓ Partition Validation Status: PASS')",
    "print(f\"  • Total Records in Corpus: {validation['counts']['total']:,}\")",
    "print(f\"  • Unique Corpus IDs: {validation['unique_id_count']:,} (100% Unique: {validation['has_unique_ids']})\")",
    "print(f\"  • Mutually Exclusive Partitions: {validation['is_mutually_exclusive']}\")",
    "print(f\"  • Collectively Exhaustive: {validation['is_collectively_exhaustive']}\")",
    "print(f\"  • Initial Candidate Pool: {validation['counts']['candidate_pool']:,} records\")",
    "print(f\"  • Preserved Remaining Pool: {validation['counts']['remaining_pool']:,} records\")",
    "",
    "# Formatted Table of Approved Partitions",
    "partition_table = pd.DataFrame([",
    "    {'Tier': 'Tier 1', 'Category': 'Discrete material lines', 'Target Records': 923, 'Actual Records': len(partitions['tier1_discrete']), 'Status': 'MATCH'},",
    "    {'Tier': 'Tier 2 (Tagged)', 'Category': 'Tagged procurement descriptions', 'Target Records': 5711, 'Actual Records': len(partitions['tier2_tagged']), 'Status': 'MATCH'},",
    "    {'Tier': 'Tier 2 (Untagged)', 'Category': 'Untagged descriptions', 'Target Records': 14542, 'Actual Records': len(partitions['tier2_untagged']), 'Status': 'MATCH'},",
    "    {'Tier': 'Tier 3', 'Category': 'Administrative/works descriptions', 'Target Records': 337, 'Actual Records': len(partitions['tier3_administrative']), 'Status': 'MATCH'},",
    "    {'Tier': 'Total', 'Category': 'Full Master Corpus', 'Target Records': 21513, 'Actual Records': len(df_raw), 'Status': 'MATCH'},",
    "    {'Tier': 'Candidate Pool', 'Category': 'Tier 1 + Tagged Tier 2', 'Target Records': 6634, 'Actual Records': len(partitions['candidate_pool']), 'Status': 'MATCH'},",
    "])",
    "display(partition_table)"
]))

# 5. Text Normalisation Demonstrations
cells.append(md_cell([
    "---",
    "## 4. Text Normalisation & Domain Specification Standardisation",
    "",
    "We verify our reusable normalisation engine implemented in `src/preprocessing/text_cleaner.py`:",
    "1. **Casing & Whitespace:** Convert to uppercase, strip leading/trailing spaces, and collapse redundant spacing.",
    "2. **Separator & Punctuation Normalisation:** Normalize unicode quotes, hyphens, and dots in acronyms (`S.S.` $\\to$ `SS`).",
    "3. **Unit Standardisation:** Standardize units (`10 mm` $\\to$ `10MM`, `2 inch` $\\to$ `2INCH`, `2\"` $\\to$ `2INCH`).",
    "4. **Engineering Abbreviations:** Expand acronyms (`SS` $\\to$ `STAINLESS STEEL` when standalone, `HEX` $\\to$ `HEXAGONAL`, `DIA` $\\to$ `DIAMETER`).",
    "5. **Material Grade Formatting:** Unify grade notations (`S.S. 304`, `SS 304`, `SS-304` $\\to$ `SS304`).",
    "6. **Boilerplate Stripping:** Remove irrelevant tender prefixes (`SUPPLY OF...`, `CONSOLIDATED PROCUREMENT OF...`, `HIRING SERVICES FOR...`).",
    "7. **Domain Term Preservation:** Ensure words such as `PIPE`, `FLANGE`, and `SE` (e.g. screwed end or vehicle models) are strictly preserved."
]))

cells.append(code_cell([
    "# ==============================================================================",
    "# Step 4A: Validate User-Mandated Specification Examples",
    "# ==============================================================================",
    "mandated_specs = [",
    "    ('10 mm', '10MM'),",
    "    ('2 inch', '2INCH'),",
    "    ('S.S. 304', 'SS304'),",
    "    ('Stainless Steel', 'STAINLESS STEEL'),",
    "]",
    "",
    "spec_results = []",
    "for orig, target in mandated_specs:",
    "    cleaned = normalise_text(orig)",
    "    passed = (cleaned == target)",
    "    spec_results.append({'Original Text': orig, 'Expected Normalized': target, 'Actual Output': cleaned, 'Passed': passed})",
    "    assert passed, f'Failed: {orig} -> {cleaned} (Expected: {target})'",
    "",
    "print('✓ All 4 User-Mandated Normalization Test Cases PASSED:')",
    "display(pd.DataFrame(spec_results))",
    "",
    "# Verify Technical Terms Preservation",
    "preservation_checks = [",
    "    ('SEAMLESS PIPE 2INCH', 'PIPE'),",
    "    ('CS FLANGE 150#', 'FLANGE'),",
    "    ('TRUCK CHASSIS TATA SE 1613/42', 'SE'),",
    "]",
    "for text, token in preservation_checks:",
    "    res = normalise_text(text)",
    "    assert token in res, f'Term {token} was improperly removed from {text}!'",
    "    print(f\"✓ Preserved '{token}' in: '{res}'\")"
]))

cells.append(code_cell([
    "# ==============================================================================",
    "# Step 4B: Before & After Normalisation on Real Corpus Samples",
    "# ==============================================================================",
    "sample_descriptions = [",
    "    'Consolidated Procurement of CI Gate Valves for Farakka & Darlipalli',",
    "    'Supply of Seamless Steel Pipe 50mm NB Schedule 40',",
    "    'SS Hex Bolt, 10mm x 50mm lg with S.S. 304 nut',",
    "    'Hiring the services of 1(One) No. Brand New BoleroSLX, (4x2) Drive (2WD)',",
    "    'GALVANISED LINE PIPE Line Pipe, 25mm(1\") conforming to BIS:1239',",
    "    '1.4.4 Pressure meter test set up capable to apply pressure up to 80 kg/sq.cm - One (1)',",
    "]",
    "",
    "sample_comparison = []",
    "for raw in sample_descriptions:",
    "    clean = normalise_text(raw)",
    "    sample_comparison.append({'Raw Description': raw, 'Cleaned & Standardised': clean})",
    "",
    "display(pd.DataFrame(sample_comparison))"
]))

# 6. Attribute Extraction Demonstration
cells.append(md_cell([
    "---",
    "## 5. Engineering Attribute Extraction",
    "",
    "Extracts key physical and metallurgical parameters from cleaned descriptions:",
    "- `metric_thread`: Thread designation (e.g. `M10`, `M12`, `M16`)",
    "- `diameter`: Nominal bore or outer diameter (e.g. `50MM`, `100MM NB`, `2INCH`)",
    "- `material_grade`: Standard alloy/metallurgy grade (e.g. `SS304`, `SS316`, `IS2062`, `ASTM A106`, `GRADE 8.8`)",
    "- `pressure_class`: Rated working pressure (e.g. `150#`, `300#`, `PN16`, `PN40`, `10000PSI`)",
    "- `schedule`: Pipe wall thickness class (e.g. `SCH 40`, `SCH 80`)"
]))

cells.append(code_cell([
    "# ==============================================================================",
    "# Step 5: Attribute Extraction Test & Demonstration",
    "# ==============================================================================",
    "attr_samples = [",
    "    'STAINLESS STEEL HEXAGONAL BOLT M10 50MM SS304 PN16',",
    "    'SEAMLESS CARBON STEEL PIPE 100MM NB SCH 40 ASTM A106 GRADE B',",
    "    'CI GATE VALVES 2INCH 150# FLANGED',",
    "    'HIGH TENSILE FASTENER M16 75MM GRADE 8.8',",
    "]",
    "",
    "extracted_cards = []",
    "for s in attr_samples:",
    "    attrs = extract_material_attributes(s)",
    "    extracted_cards.append({",
    "        'Cleaned Description': s,",
    "        'Thread': attrs['metric_thread'],",
    "        'Diameter': attrs['diameter'],",
    "        'Grade': attrs['material_grade'],",
    "        'Pressure': attrs['pressure_class'],",
    "        'Schedule': attrs['schedule'],",
    "    })",
    "",
    "display(pd.DataFrame(extracted_cards))"
]))

# 7. Preprocessing Execution on Candidate Pool
cells.append(md_cell([
    "---",
    "## 6. Preprocessing Candidate Pool (6,634 Records)",
    "",
    "We process the initial candidate pool (Tier 1 + Tagged Tier 2 = 6,634 records) using `preprocess_catalogue`.  ",
    "The remaining 14,879 records are preserved separately in raw form for downstream catalog enrichment."
]))

cells.append(code_cell([
    "# ==============================================================================",
    "# Step 6: Execute Preprocessing Pipeline on Candidate Pool",
    "# ==============================================================================",
    "import time",
    "",
    "t_start = time.time()",
    "df_candidates = partitions['candidate_pool'].copy()",
    "df_remaining = partitions['remaining_pool'].copy()",
    "",
    "print(f'Starting preprocessing on Candidate Pool ({len(df_candidates):,} records)...')",
    "df_candidate_preprocessed = preprocess_catalogue(df_candidates)",
    "t_elapsed = time.time() - t_start",
    "",
    "print(f'✓ Successfully preprocessed {len(df_candidate_preprocessed):,} candidate records in {t_elapsed:.2f} seconds.')",
    "print(f'✓ Preserved {len(df_remaining):,} remaining records separately.')",
    "",
    "# Attribute Coverage Statistics",
    "attr_fields = ['attr_diameter', 'attr_pressure_class', 'attr_material_grade', 'attr_metric_thread', 'attr_schedule']",
    "coverage_rows = []",
    "for f in attr_fields:",
    "    count = df_candidate_preprocessed[f].notnull().sum()",
    "    pct = (count / len(df_candidate_preprocessed)) * 100",
    "    coverage_rows.append({'Attribute Field': f, 'Detected Count': count, 'Coverage (% of Candidates)': round(pct, 2)})",
    "",
    "display(pd.DataFrame(coverage_rows))",
    "",
    "# Display 5 sample preprocessed candidate records",
    "display(df_candidate_preprocessed[['corpus_id', 'organization', 'partition_tier', 'description', 'cleaned_description', 'attr_diameter', 'attr_material_grade']].head(5))"
]))

# 8. Diagnostic Visualisations
cells.append(md_cell([
    "---",
    "## 7. Diagnostic Visualisations & Profiling",
    "",
    "Generates diagnostic figures for the CA-1 report:",
    "1. **Partition Breakdown Chart:** Visualizes Tier 1, Tier 2 Tagged, Tier 2 Untagged, and Tier 3.",
    "2. **Extracted Attribute Coverage:** Quantifies detected engineering dimensions across the candidate pool.",
    "3. **Character & Word Length Distributions:** Profiles the impact of normalisation and boilerplate stripping."
]))

cells.append(code_cell([
    "# ==============================================================================",
    "# Step 7: Diagnostic Visualisations",
    "# ==============================================================================",
    "figures_dir = FIGURES_DIR",
    "figures_dir.mkdir(parents=True, exist_ok=True)",
    "sns.set_theme(style='whitegrid')",
    "",
    "# Figure 1: Approved Partition Distribution",
    "plt.figure(figsize=(9, 4.5))",
    "part_counts = pd.Series({",
    "    'Tier 1: Discrete Material Lines': 923,",
    "    'Tier 2: Tagged Procurement': 5711,",
    "    'Tier 2: Untagged Procurement': 14542,",
    "    'Tier 3: Administrative / Works': 337,",
    "})",
    "colors = ['#2ca02c', '#1f77b4', '#aec7e8', '#ff7f0e']",
    "ax1 = part_counts.plot(kind='barh', color=colors, edgecolor='black', alpha=0.85)",
    "plt.title('Corpus Partition Hierarchy & Candidate Isolation (N = 21,513)', fontsize=13, fontweight='bold', pad=12)",
    "plt.xlabel('Record Count', fontsize=11)",
    "for p in ax1.patches:",
    "    w = p.get_width()",
    "    ax1.annotate(f'{int(w):,} ({w/21513*100:.1f}%)', (w + 150, p.get_y() + p.get_height()/2), ha='left', va='center', fontsize=9.5)",
    "plt.xlim(0, max(part_counts) * 1.25)",
    "plt.tight_layout()",
    "fig1_path = figures_dir / '02_partition_breakdown.png'",
    "plt.savefig(fig1_path, dpi=200)",
    "plt.show()",
    "print(f'Saved Figure 1 to: {fig1_path}')",
    "",
    "# Figure 2: Attribute Extraction Coverage",
    "plt.figure(figsize=(8, 4))",
    "cov_df = pd.DataFrame(coverage_rows)",
    "ax2 = sns.barplot(x='Coverage (% of Candidates)', y='Attribute Field', data=cov_df, palette='viridis', hue='Attribute Field', legend=False)",
    "plt.title('Extracted Attribute Fill Rates in Candidate Pool (N = 6,634)', fontsize=13, fontweight='bold', pad=12)",
    "plt.xlabel('Fill Rate (% of 6,634 Candidates)', fontsize=11)",
    "for p in ax2.patches:",
    "    w = p.get_width()",
    "    ax2.annotate(f'{w:.2f}%', (w + 0.3, p.get_y() + p.get_height()/2), ha='left', va='center', fontsize=9.5)",
    "plt.xlim(0, max(cov_df['Coverage (% of Candidates)']) * 1.25)",
    "plt.tight_layout()",
    "fig2_path = figures_dir / '02_attribute_coverage.png'",
    "plt.savefig(fig2_path, dpi=200)",
    "plt.show()",
    "print(f'Saved Figure 2 to: {fig2_path}')",
    "",
    "# Figure 3: Length Reduction from Cleaning",
    "raw_lens = df_candidate_preprocessed['description'].fillna('').apply(len)",
    "clean_lens = df_candidate_preprocessed['cleaned_description'].fillna('').apply(len)",
    "",
    "plt.figure(figsize=(10, 4.5))",
    "sns.kdeplot(raw_lens, label=f'Raw Text (Mean: {raw_lens.mean():.1f})', color='#d62728', fill=True, alpha=0.3)",
    "sns.kdeplot(clean_lens, label=f'Cleaned Text (Mean: {clean_lens.mean():.1f})', color='#1f77b4', fill=True, alpha=0.3)",
    "plt.title('Character Length Distribution: Raw vs Cleaned Text', fontsize=13, fontweight='bold', pad=12)",
    "plt.xlabel('Character Count', fontsize=11)",
    "plt.ylabel('Density', fontsize=11)",
    "plt.xlim(0, 300)",
    "plt.legend(fontsize=10.5)",
    "plt.tight_layout()",
    "fig3_path = figures_dir / '02_cleaning_length_impact.png'",
    "plt.savefig(fig3_path, dpi=200)",
    "plt.show()",
    "print(f'Saved Figure 3 to: {fig3_path}')"
]))

# 9. Artifact Export
cells.append(md_cell([
    "---",
    "## 8. Export Processed Artifacts & Structured Summary",
    "",
    "Saves:",
    "1. `data/processed/candidate_pool_preprocessed.csv`: The cleaned candidate pool (6,634 records) ready for candidate generation and entity resolution.",
    "2. `data/processed/remaining_corpus_preserved.csv`: The remaining 14,879 records preserved separately.",
    "3. `results/reports/preprocessing_audit_summary.json`: Formal JSON report documenting partition metrics and attribute detection rates."
]))

cells.append(code_cell([
    "# ==============================================================================",
    "# Step 8: Export Processed Datasets and Structured Summary",
    "# ==============================================================================",
    "DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)",
    "REPORTS_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "# 1. Save Preprocessed Candidate Pool",
    "candidate_out_csv = DATA_PROCESSED_DIR / 'candidate_pool_preprocessed.csv'",
    "df_candidate_preprocessed.to_csv(candidate_out_csv, index=False)",
    "print(f'✓ Exported candidate pool: {candidate_out_csv} ({len(df_candidate_preprocessed):,} records)')",
    "",
    "# 2. Save Preserved Remaining Records Separately",
    "remaining_out_csv = DATA_PROCESSED_DIR / 'remaining_corpus_preserved.csv'",
    "df_remaining.to_csv(remaining_out_csv, index=False)",
    "print(f'✓ Exported preserved remaining corpus: {remaining_out_csv} ({len(df_remaining):,} records)')",
    "",
    "# 3. Save Structured Audit Report JSON",
    "audit_summary = {",
    "    'notebook': '02_data_preprocessing.ipynb',",
    "    'total_corpus_records': int(len(df_raw)),",
    "    'candidate_pool_records': int(len(df_candidate_preprocessed)),",
    "    'remaining_corpus_records': int(len(df_remaining)),",
    "    'partitions': {",
    "        'tier1_discrete': int(len(partitions['tier1_discrete'])),",
    "        'tier2_tagged': int(len(partitions['tier2_tagged'])),",
    "        'tier2_untagged': int(len(partitions['tier2_untagged'])),",
    "        'tier3_administrative': int(len(partitions['tier3_administrative'])),",
    "    },",
    "    'partition_validation': {",
    "        'all_counts_match': bool(validation['all_counts_match']),",
    "        'has_unique_ids': bool(validation['has_unique_ids']),",
    "        'is_mutually_exclusive': bool(validation['is_mutually_exclusive']),",
    "        'is_collectively_exhaustive': bool(validation['is_collectively_exhaustive']),",
    "    },",
    "    'attribute_detection_counts': {",
    "        f: int(df_candidate_preprocessed[f].notnull().sum()) for f in attr_fields",
    "    },",
    "    'attribute_detection_rates_pct': {",
    "        f: round(float(df_candidate_preprocessed[f].notnull().sum() / len(df_candidate_preprocessed) * 100), 2) for f in attr_fields",
    "    },",
    "    'raw_vs_clean_char_length': {",
    "        'raw_mean': round(float(raw_lens.mean()), 2),",
    "        'clean_mean': round(float(clean_lens.mean()), 2),",
    "        'mean_reduction_chars': round(float(raw_lens.mean() - clean_lens.mean()), 2),",
    "    }",
    "}",
    "",
    "audit_report_path = REPORTS_DIR / 'preprocessing_audit_summary.json'",
    "with open(audit_report_path, 'w', encoding='utf-8') as f:",
    "    json.dump(audit_summary, f, indent=2)",
    "print(f'✓ Exported structured audit report: {audit_report_path}')",
    "",
    "print('\\n' + '='*70)",
    "print('🎉 NOTEBOOK 02 PREPROCESSING PIPELINE SUCCESSFULLY COMPLETED!')",
    "print('='*70)"
]))

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_path = Path("c:/project ai/notebooks/02_data_preprocessing.ipynb")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote notebook to {target_path} successfully ({len(cells)} cells)")
