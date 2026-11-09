"""
Lightweight Zero-Dependency Web Reviewer UI for Pairwise Material Entity Resolution.

Runs a local HTTP server using Python's built-in http.server (no external web frameworks required).
Provides an interactive, modern browser interface to review candidate material pairs side-by-side,
assign ground truth labels, view engineering attribute conflicts, inspect AI suggestions, and export data.

Supports:
- Integrated 150-pair dataset (Batch 1 + Batch 2) with dataset switching.
- Source references and tender provenance display for both materials.
- Real-time specification conflict and missing attribute warnings.
- Explicit AI recommendation & rationale segregation.
- Strict non-auto-advancing label selection (keyboard 1, 0, U or click).
- Mandatory confidence rating and explicit Save & Next submission.

Usage:
    python review_app.py [--port 8050] [--dataset combined|batch1]
"""

import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

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

WORKSHEET_PATH = DATA_PROCESSED_DIR / "improved_labelled_pairs_worksheet.csv"
COMBINED_WORKSHEET_PATH = DATA_PROCESSED_DIR / "combined_candidate_pairs_150.csv"

# Active dataset path (default to combined if it exists)
DEFAULT_ACTIVE_PATH = COMBINED_WORKSHEET_PATH if COMBINED_WORKSHEET_PATH.exists() else WORKSHEET_PATH

WORKSHEET_DTYPES = {
    "pair_id": str,
    "corpus_id_1": str,
    "enterprise_1": str,
    "raw_description_1": str,
    "cleaned_description_1": str,
    "extracted_attributes_1": str,
    "source_reference_1": str,
    "corpus_id_2": str,
    "enterprise_2": str,
    "raw_description_2": str,
    "cleaned_description_2": str,
    "extracted_attributes_2": str,
    "source_reference_2": str,
    "material_domain": str,
    "conflict_details": str,
    "confirmation_details": str,
    "human_label": str,
    "confidence_rating": str,
    "reviewer_notes": str,
    "review_guidance": str,
    "ai_suggested_hypothesis": str,
    "ai_suggested_confidence": str,
    "ai_engineering_rationale": str,
}


def load_worksheet_df(path: Optional[Path] = None) -> pd.DataFrame:
    """Load worksheet CSV ensuring all text and annotation columns are object/str dtype."""
    target_path = path or WORKSHEET_PATH
    df = pd.read_csv(target_path, dtype=WORKSHEET_DTYPES)
    for col in [
        "reviewer_notes", "confirmation_details", "conflict_details",
        "human_label", "confidence_rating", "source_reference_1", "source_reference_2",
        "ai_suggested_hypothesis", "ai_suggested_confidence", "ai_engineering_rationale",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df


def clean_dataframe_for_json(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame records to JSON-compliant dicts, replacing all NaN/NaT with None."""
    records = []
    for r in df.to_dict(orient="records"):
        clean_record = {}
        for k, v in r.items():
            if pd.isna(v):
                clean_record[k] = None
            else:
                clean_record[k] = v
        records.append(clean_record)
    return records


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CSE 276 — Material Code Harmonisation Reviewer</title>
  <style>
    :root {
      --primary: #1e3a8a;
      --primary-light: #2563eb;
      --primary-subtle: #eff6ff;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --text-muted: #64748b;
      --success: #16a34a;
      --danger: #dc2626;
      --warning: #d97706;
      --info: #0284c7;
      --ai-purple: #7c3aed;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding-bottom: 60px;
    }
    header {
      background: var(--primary);
      color: white;
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    header h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; }
    header .subtitle { font-size: 0.8rem; opacity: 0.85; }
    .container { max-width: 1240px; margin: 20px auto; padding: 0 16px; }
    
    /* Top Bar & Stats */
    .top-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 20px;
    }
    .stats-group { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .stat-badge {
      font-size: 0.82rem;
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 600;
    }
    .badge-pos { background: #dcfce7; color: var(--success); }
    .badge-neg { background: #fee2e2; color: var(--danger); }
    .badge-unc { background: #fef3c7; color: var(--warning); }
    .badge-pen { background: #f1f5f9; color: var(--text-muted); }

    .nav-controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button {
      cursor: pointer;
      font-family: inherit;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 8px 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: white;
      transition: all 0.15s ease;
    }
    button:hover { background: #f1f5f9; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .btn-primary { background: var(--primary-light); color: white; border: none; }
    .btn-primary:hover:not(:disabled) { background: var(--primary); }
    .btn-export { background: #0f172a; color: white; border: none; }
    .btn-export:hover { background: #334155; }

    /* Pair Inspection Cards */
    .pair-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }
    @media (max-width: 768px) { .pair-grid { grid-template-columns: 1fr; } }
    .mat-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-title {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--primary-light);
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
    }
    .prop-row { margin-bottom: 10px; }
    .prop-label { font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
    .prop-value { font-size: 0.95rem; font-weight: 500; word-break: break-word; }
    .raw-text { font-family: monospace; background: #f1f5f9; padding: 8px 12px; border-radius: 4px; font-size: 0.88rem; margin-top: 4px; word-break: break-word; white-space: pre-wrap; }
    .attrs-tag { display: inline-block; background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 4px; font-size: 0.82rem; font-weight: 600; margin-top: 4px; word-break: break-word; }
    .attrs-none { display: inline-block; background: #f1f5f9; color: var(--text-muted); padding: 4px 10px; border-radius: 4px; font-size: 0.82rem; margin-top: 4px; }
    .src-ref-box { font-size: 0.78rem; background: #fafafa; border: 1px solid #e5e7eb; padding: 6px 10px; border-radius: 4px; color: #475569; word-break: break-word; margin-top: 4px; }

    .warning-missing {
      color: #b91c1c;
      background: #fef2f2;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.82rem;
      border: 1px dashed #f87171;
      display: inline-block;
    }

    /* Comparison Section */
    .comparison-section {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 20px;
    }
    .metrics-row {
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      align-items: center;
    }
    .metric-item { display: flex; flex-direction: column; }
    .metric-item .val { font-size: 1.1rem; font-weight: 700; color: var(--primary); }
    .conflict-banner {
      background: #fef2f2;
      border-left: 4px solid var(--danger);
      padding: 10px 14px;
      border-radius: 4px;
      color: #991b1b;
      font-size: 0.9rem;
      font-weight: 600;
      margin-top: 10px;
    }
    .confirm-banner {
      background: #f0fdf4;
      border-left: 4px solid var(--success);
      padding: 10px 14px;
      border-radius: 4px;
      color: #166534;
      font-size: 0.9rem;
      font-weight: 600;
      margin-top: 10px;
    }

    /* AI Suggestions Panel */
    .ai-panel {
      background: #faf5ff;
      border: 1.5px solid #d8b4fe;
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 20px;
    }
    .ai-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .ai-tag {
      background: #7c3aed;
      color: white;
      font-size: 0.72rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .ai-hyp { font-weight: 700; font-size: 0.95rem; }
    .ai-hyp-pos { color: #15803d; }
    .ai-hyp-neg { color: #b91c1c; }
    .ai-hyp-unc { color: #b45309; }
    .ai-rationale { font-size: 0.88rem; color: #4c1d95; line-height: 1.45; }

    /* Decision Panel */
    .decision-panel {
      background: var(--card-bg);
      border: 2px solid var(--primary-light);
      border-radius: 8px;
      padding: 18px 20px;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .decision-buttons {
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }
    .btn-label {
      flex: 1;
      min-width: 160px;
      padding: 12px 16px;
      font-size: 0.95rem;
      font-weight: 700;
      border-radius: 6px;
      border: 2px solid transparent;
      color: white;
      text-align: center;
      transition: opacity 0.15s ease, transform 0.1s ease;
    }
    .btn-label:hover { opacity: 0.9; }
    .btn-eq { background: var(--success); }
    .btn-eq.active { background: #15803d; border-color: #052e16; box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.4); }
    .btn-dist { background: var(--danger); }
    .btn-dist.active { background: #b91c1c; border-color: #450a0a; box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.4); }
    .btn-unc { background: var(--warning); }
    .btn-unc.active { background: #b45309; border-color: #451a03; box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.4); }

    .notes-group {
      display: flex;
      gap: 16px;
      align-items: flex-end;
      flex-wrap: wrap;
    }
    .notes-group textarea {
      flex: 1;
      min-width: 250px;
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-family: inherit;
      font-size: 0.88rem;
      resize: vertical;
      min-height: 48px;
    }
    .notes-group select {
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-family: inherit;
      font-size: 0.88rem;
      font-weight: 600;
      background: white;
    }

    .status-msg {
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 0.88rem;
      font-weight: 600;
      margin-top: 12px;
    }
    .status-error { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .status-success { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .status-info { background: #f0f9ff; color: #0369a1; border: 1px solid #bae6fd; }

    /* Modals */
    .modal-backdrop {
      display: none;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.5);
      z-index: 1000;
      justify-content: center;
      align-items: center;
    }
    .modal-content {
      background: white;
      padding: 24px;
      border-radius: 8px;
      max-width: 700px;
      width: 90%;
      max-height: 85vh;
      overflow-y: auto;
    }
    pre { background: #f1f5f9; padding: 12px; border-radius: 6px; font-size: 0.85rem; overflow-x: auto; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>CSE 276 — Material Code Harmonisation</h1>
      <div class="subtitle">Human Annotation & Precision Entity Resolution Reviewer (Batch 1 & 2 Integrated)</div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
      <select id="dataset-selector" onchange="switchDataset()" style="padding: 6px 10px; border-radius: 6px; font-weight: 600; font-size: 0.82rem; border: none; background: white; color: var(--primary);">
        <option value="combined" selected>Combined Dataset (150 Pairs)</option>
        <option value="batch1">Batch 1 Only (50 Pairs)</option>
      </select>
      <button class="btn-export" onclick="showReadinessModal()">Training Readiness</button>
      <button class="btn-export" onclick="exportCSV()">Export CSV</button>
    </div>
  </header>

  <div class="container">
    <div class="top-bar">
      <div class="stats-group">
        <span><strong>Pair:</strong> <span id="pair-index">Loading...</span></span>
        <span class="stat-badge badge-pos" id="badge-pos">1: 0</span>
        <span class="stat-badge badge-neg" id="badge-neg">0: 0</span>
        <span class="stat-badge badge-unc" id="badge-unc">Uncertain: 0</span>
        <span class="stat-badge badge-pen" id="badge-pen">Pending: 100</span>
      </div>

      <div class="nav-controls">
        <select id="filter-select" onchange="applyFilter()">
          <option value="ALL">Show All Pairs</option>
          <option value="PENDING" selected>Pending Only</option>
          <option value="REVIEWED">Reviewed Only</option>
          <option value="CONFLICTS">Conflicts Only</option>
          <option value="1">Positives (1) Only</option>
          <option value="0">Negatives (0) Only</option>
          <option value="UNCERTAIN">Uncertain Only</option>
          <option value="BATCH1">Batch 1 Pairs</option>
          <option value="BATCH2">Batch 2 Pairs</option>
        </select>
        <button onclick="prevPair()">◀ Prev</button>
        <button onclick="nextPair()">Next ▶</button>
      </div>
    </div>

    <div class="pair-grid">
      <!-- Material 1 -->
      <div class="mat-card">
        <div class="card-title">
          <span>Material A</span>
          <span id="mat1-org">Connecting to dataset...</span>
        </div>
        <div class="prop-row">
          <div class="prop-label">Corpus ID</div>
          <div class="prop-value" id="mat1-id">Connecting...</div>
        </div>
        <div class="prop-row">
          <div class="prop-label">Raw Material Description</div>
          <div class="raw-text" id="mat1-raw">Connecting to dataset...</div>
        </div>
        <div class="prop-row">
          <div class="prop-label">Preprocessed Text</div>
          <div class="prop-value" id="mat1-clean">Connecting to dataset...</div>
        </div>
        <div class="prop-row">
          <div class="prop-label">Extracted Physical Attributes</div>
          <div id="mat1-attrs-container">
            <span class="attrs-none" id="mat1-attrs">Connecting...</span>
          </div>
        </div>
        <div class="prop-row">
          <div class="prop-label">Source Provenance & Tender Reference</div>
          <div class="src-ref-box" id="mat1-src">Loading reference...</div>
        </div>
      </div>

      <!-- Material 2 -->
      <div class="mat-card">
        <div class="card-title">
          <span>Material B</span>
          <span id="mat2-org">Connecting to dataset...</span>
        </div>
        <div class="prop-row">
          <div class="prop-label">Corpus ID</div>
          <div class="prop-value" id="mat2-id">Connecting...</div>
        </div>
        <div class="prop-row">
          <div class="prop-label">Raw Material Description</div>
          <div class="raw-text" id="mat2-raw">Connecting to dataset...</div>
        </div>
        <div class="prop-row">
          <div class="prop-label">Preprocessed Text</div>
          <div class="prop-value" id="mat2-clean">Connecting to dataset...</div>
        </div>
        <div class="prop-row">
          <div class="prop-label">Extracted Physical Attributes</div>
          <div id="mat2-attrs-container">
            <span class="attrs-none" id="mat2-attrs">Connecting...</span>
          </div>
        </div>
        <div class="prop-row">
          <div class="prop-label">Source Provenance & Tender Reference</div>
          <div class="src-ref-box" id="mat2-src">Loading reference...</div>
        </div>
      </div>
    </div>

    <!-- Comparison Section -->
    <div class="comparison-section">
      <div class="metrics-row">
        <div class="metric-item">
          <span class="prop-label">Domain</span>
          <span class="val" id="metric-domain" style="font-size: 0.95rem;">--</span>
        </div>
        <div class="metric-item">
          <span class="prop-label">Lexical Sim (TF-IDF)</span>
          <span class="val" id="metric-lex">--</span>
        </div>
        <div class="metric-item">
          <span class="prop-label">Semantic Sim (Transformer)</span>
          <span class="val" id="metric-sem">--</span>
        </div>
        <div class="metric-item">
          <span class="prop-label">Discrepancy</span>
          <span class="val" id="metric-disc">--</span>
        </div>
        <div class="metric-item">
          <span class="prop-label">Exact Match</span>
          <span class="val" id="metric-exact">--</span>
        </div>
      </div>

      <div id="conflict-alert" class="conflict-banner" style="display: none;">
        ⚠️ <strong>Specification Conflict Warning:</strong> <span id="conflict-text"></span>
      </div>
      <div id="confirm-alert" class="confirm-banner" style="display: none;">
        ✓ <strong>Confirmed Specifications:</strong> <span id="confirm-text"></span>
      </div>
      <div class="prop-row" style="margin-top: 10px;">
        <span class="prop-label">Review Guidance:</span>
        <span id="guidance-text" style="font-size: 0.85rem; color: var(--text-muted);">None</span>
      </div>
    </div>

    <!-- AI Suggestion Panel (Advisory Only) -->
    <div class="ai-panel" id="ai-panel" style="display: none;">
      <div class="ai-header">
        <span class="ai-tag">AI Hypothesis (Advisory Only)</span>
        <span class="ai-hyp" id="ai-hyp-text">UNCERTAIN</span>
      </div>
      <div class="ai-rationale" id="ai-rationale-text">No advisory rationale available.</div>
    </div>

    <!-- Decision Panel -->
    <div class="decision-panel">
      <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;">
        Assign Ground-Truth Label (Keyboard: 1, 0, U) — Selects Label Only (Does Not Auto-Advance)
      </div>
      <div class="decision-buttons">
        <button class="btn-label btn-eq" id="btn-1" type="button" onclick="selectLabel('1')">
          [1] Functionally Equivalent
        </button>
        <button class="btn-label btn-dist" id="btn-0" type="button" onclick="selectLabel('0')">
          [0] Not Equivalent
        </button>
        <button class="btn-label btn-unc" id="btn-u" type="button" onclick="selectLabel('UNCERTAIN')">
          [?] Uncertain (Insufficient Spec)
        </button>
      </div>

      <div class="notes-group">
        <div style="flex: 1; min-width: 250px;">
          <div class="prop-label" style="margin-bottom: 4px;">Reviewer Engineering Notes / Rationale</div>
          <textarea id="reviewer-notes" placeholder="Enter engineering rationale (e.g. Incompatible 250mm vs 300mm pipe; or identical SS304 spec)..."></textarea>
        </div>
        <div>
          <div class="prop-label" style="margin-bottom: 4px;">Confidence <span style="color: var(--danger);">*</span></div>
          <select id="confidence-select">
            <option value="" disabled selected>-- Select Confidence --</option>
            <option value="HIGH">HIGH Confidence</option>
            <option value="MEDIUM">MEDIUM Confidence</option>
            <option value="LOW">LOW Confidence</option>
          </select>
        </div>
        <button class="btn-primary" id="btn-save-next" style="height: 48px;" type="button" onclick="saveAndNext()">Save & Next ▶</button>
      </div>

      <div id="save-status" class="status-msg" style="display: none;"></div>
    </div>
  </div>

  <!-- Modal for Readiness Report -->
  <div class="modal-backdrop" id="readiness-modal">
    <div class="modal-content">
      <h2 style="margin-bottom: 12px;">Training Readiness Audit</h2>
      <pre id="readiness-content">Loading report...</pre>
      <div style="text-align: right; margin-top: 16px;">
        <button onclick="closeReadinessModal()">Close</button>
      </div>
    </div>
  </div>

  <script>
    let allPairs = [];
    let filteredIndices = [];
    let currentFilteredIdx = 0;
    let currentSelectedLabel = null;
    let isSaving = false;

    async function switchDataset() {
      await loadData();
    }

    async function loadData() {
      const selector = document.getElementById('dataset-selector');
      const ds = selector ? selector.value : 'combined';
      try {
        const res = await fetch(`/api/pairs?dataset=${ds}`);
        if (!res.ok) throw new Error('Server returned HTTP ' + res.status + ': ' + res.statusText);
        const data = await res.json();
        allPairs = data.pairs || [];
        updateStats(data.stats);
        applyFilter();
      } catch (err) {
        console.error('Error loading pairs from API:', err);
        alert('CRITICAL ERROR: Failed to load candidate pairs from backend.\\n' + err.message);
        document.getElementById('pair-index').innerText = 'Data load failed!';
      }
    }

    function updateStats(stats) {
      if (!stats) return;
      document.getElementById('badge-pos').innerText = `1: ${stats.pos}`;
      document.getElementById('badge-neg').innerText = `0: ${stats.neg}`;
      document.getElementById('badge-unc').innerText = `Uncertain: ${stats.unc}`;
      document.getElementById('badge-pen').innerText = `Pending: ${stats.pen}`;
    }

    function applyFilter() {
      const filter = document.getElementById('filter-select').value;
      filteredIndices = [];
      for (let i = 0; i < allPairs.length; i++) {
        const p = allPairs[i];
        const lbl = (p.human_label || 'PENDING').toUpperCase();
        const isB2 = (p.pair_id || '').startsWith('PAIR_EXP_');

        if (filter === 'ALL') {
          filteredIndices.push(i);
        } else if (filter === 'PENDING' && lbl === 'PENDING') {
          filteredIndices.push(i);
        } else if (filter === 'REVIEWED' && lbl !== 'PENDING') {
          filteredIndices.push(i);
        } else if (filter === 'CONFLICTS' && (p.has_attribute_conflict === true || String(p.has_attribute_conflict).toLowerCase() === 'true')) {
          filteredIndices.push(i);
        } else if (filter === 'BATCH1' && !isB2) {
          filteredIndices.push(i);
        } else if (filter === 'BATCH2' && isB2) {
          filteredIndices.push(i);
        } else if (filter === lbl) {
          filteredIndices.push(i);
        }
      }
      if (filteredIndices.length === 0) {
        alert('No pairs match this filter. Resetting to All.');
        document.getElementById('filter-select').value = 'ALL';
        filteredIndices = allPairs.map((_, i) => i);
      }
      currentFilteredIdx = 0;
      renderCurrentPair();
    }

    function formatField(val, fieldName) {
      if (val === null || val === undefined || String(val).trim() === '' || String(val).trim() === 'nan') {
        return `<span class="warning-missing">[MISSING ${fieldName.toUpperCase()}]</span>`;
      }
      return val;
    }

    function formatAttributes(val) {
      if (val === null || val === undefined || String(val).trim() === '' || String(val).trim() === 'nan') {
        return `<span class="warning-missing">[MISSING ATTRIBUTES]</span>`;
      }
      const s = String(val).trim();
      if (s === 'None detected') return `<span class="attrs-none">None detected</span>`;
      return `<span class="attrs-tag">${s}</span>`;
    }

    function renderCurrentPair() {
      if (filteredIndices.length === 0 || allPairs.length === 0) return;
      const actualIdx = filteredIndices[currentFilteredIdx];
      const p = allPairs[actualIdx];

      document.getElementById('pair-index').innerText = `${currentFilteredIdx + 1} of ${filteredIndices.length} (Total: ${allPairs.length}) | ${p.pair_id}`;

      // Material A
      document.getElementById('mat1-org').innerHTML = formatField(p.enterprise_1, 'Enterprise');
      document.getElementById('mat1-id').innerHTML = formatField(p.corpus_id_1, 'Corpus ID');
      document.getElementById('mat1-raw').innerHTML = formatField(p.raw_description_1, 'Raw Description');
      document.getElementById('mat1-clean').innerHTML = formatField(p.cleaned_description_1, 'Preprocessed Text');
      document.getElementById('mat1-attrs-container').innerHTML = formatAttributes(p.extracted_attributes_1);
      document.getElementById('mat1-src').innerText = p.source_reference_1 || 'Reference not recorded';

      // Material B
      document.getElementById('mat2-org').innerHTML = formatField(p.enterprise_2, 'Enterprise');
      document.getElementById('mat2-id').innerHTML = formatField(p.corpus_id_2, 'Corpus ID');
      document.getElementById('mat2-raw').innerHTML = formatField(p.raw_description_2, 'Raw Description');
      document.getElementById('mat2-clean').innerHTML = formatField(p.cleaned_description_2, 'Preprocessed Text');
      document.getElementById('mat2-attrs-container').innerHTML = formatAttributes(p.extracted_attributes_2);
      document.getElementById('mat2-src').innerText = p.source_reference_2 || 'Reference not recorded';

      // Domain & Metrics
      document.getElementById('metric-domain').innerText = p.material_domain || 'General Commodity';

      const lexRaw = p.lexical_similarity;
      const semRaw = p.semantic_cosine_similarity;
      const isLexValid = (lexRaw !== null && lexRaw !== undefined && !isNaN(parseFloat(lexRaw)));
      const isSemValid = (semRaw !== null && semRaw !== undefined && !isNaN(parseFloat(semRaw)));

      document.getElementById('metric-lex').innerText = isLexValid ? parseFloat(lexRaw).toFixed(4) : '--';
      document.getElementById('metric-sem').innerText = isSemValid ? parseFloat(semRaw).toFixed(4) : '--';

      if (isLexValid && isSemValid) {
        const diff = parseFloat(semRaw) - parseFloat(lexRaw);
        document.getElementById('metric-disc').innerText = (diff >= 0 ? '+' : '') + diff.toFixed(4);
      } else {
        document.getElementById('metric-disc').innerText = 'N/A';
      }

      if (p.is_exact_normalized_match === true || String(p.is_exact_normalized_match).toLowerCase() === 'true') {
        document.getElementById('metric-exact').innerText = 'YES (Exact Match)';
      } else {
        document.getElementById('metric-exact').innerText = 'NO';
      }

      // Conflict Banner
      const confAlert = document.getElementById('conflict-alert');
      if (p.has_attribute_conflict === true || String(p.has_attribute_conflict).toLowerCase() === 'true') {
        confAlert.style.display = 'block';
        document.getElementById('conflict-text').innerText = p.conflict_details || 'Specification mismatch detected (attribute conflict)';
      } else {
        confAlert.style.display = 'none';
      }

      // Confirmation Banner
      const confmAlert = document.getElementById('confirm-alert');
      if (p.confirmation_details && String(p.confirmation_details).trim() && String(p.confirmation_details).trim() !== 'nan') {
        confmAlert.style.display = 'block';
        document.getElementById('confirm-text').innerText = p.confirmation_details;
      } else {
        confmAlert.style.display = 'none';
      }

      // Guidance
      document.getElementById('guidance-text').innerText = (p.review_guidance && String(p.review_guidance).trim() !== 'nan') ? p.review_guidance : 'None';

      // AI Panel
      const aiPanel = document.getElementById('ai-panel');
      if (p.ai_suggested_hypothesis && String(p.ai_suggested_hypothesis).trim() !== '' && String(p.ai_suggested_hypothesis).trim() !== 'nan') {
        aiPanel.style.display = 'block';
        const hyp = String(p.ai_suggested_hypothesis).trim();
        const conf = p.ai_suggested_confidence || 'MEDIUM';
        const rat = p.ai_engineering_rationale || 'No rationale recorded.';
        
        const hypEl = document.getElementById('ai-hyp-text');
        hypEl.innerText = `${hyp} (${conf} Confidence)`;
        hypEl.className = 'ai-hyp ' + (hyp === '1' ? 'ai-hyp-pos' : (hyp === '0' ? 'ai-hyp-neg' : 'ai-hyp-unc'));
        document.getElementById('ai-rationale-text').innerText = rat;
      } else {
        aiPanel.style.display = 'none';
      }

      // Decision State
      const curLabel = (p.human_label || 'PENDING').toUpperCase();
      currentSelectedLabel = (['1', '0', 'UNCERTAIN'].includes(curLabel)) ? curLabel : null;
      updateLabelButtonUI();

      document.getElementById('reviewer-notes').value = (p.reviewer_notes && String(p.reviewer_notes).trim() !== 'nan') ? p.reviewer_notes : '';
      
      const curConf = (p.confidence_rating && String(p.confidence_rating).trim() !== 'nan') ? p.confidence_rating.toUpperCase() : '';
      const confSelect = document.getElementById('confidence-select');
      if (['HIGH', 'MEDIUM', 'LOW'].includes(curConf)) {
        confSelect.value = curConf;
      } else {
        confSelect.value = '';
      }

      clearValidationMessage();
    }

    function showValidationMessage(msg, type = 'error') {
      const el = document.getElementById('save-status');
      if (!el) return;
      el.className = 'status-msg status-' + type;
      el.innerText = msg;
      el.style.display = 'block';
    }

    function clearValidationMessage() {
      const el = document.getElementById('save-status');
      if (!el) return;
      el.innerText = '';
      el.style.display = 'none';
    }

    function updateLabelButtonUI() {
      const btn1 = document.getElementById('btn-1');
      const btn0 = document.getElementById('btn-0');
      const btnU = document.getElementById('btn-u');
      if (btn1) btn1.classList.toggle('active', currentSelectedLabel === '1');
      if (btn0) btn0.classList.toggle('active', currentSelectedLabel === '0');
      if (btnU) btnU.classList.toggle('active', currentSelectedLabel === 'UNCERTAIN');
    }

    function selectLabel(label) {
      // Rule 1: Clicking or hotkey selects label ONLY. No navigation, no auto-save.
      currentSelectedLabel = String(label).toUpperCase();
      updateLabelButtonUI();
      clearValidationMessage();
    }

    function assignLabel(label) {
      selectLabel(label);
    }

    async function saveAndNext() {
      if (isSaving) return;

      if (filteredIndices.length === 0 || allPairs.length === 0) return;
      const actualIdx = filteredIndices[currentFilteredIdx];
      const p = allPairs[actualIdx];

      if (!currentSelectedLabel || !['1', '0', 'UNCERTAIN'].includes(currentSelectedLabel)) {
        showValidationMessage('⚠️ Please select a ground-truth label: [1] Equivalent, [0] Not Equivalent, or [?] Uncertain.', 'error');
        return;
      }

      const confSelect = document.getElementById('confidence-select');
      const conf = confSelect ? confSelect.value.trim().toUpperCase() : '';
      if (!conf || !['HIGH', 'MEDIUM', 'LOW'].includes(conf)) {
        showValidationMessage('⚠️ Please select a valid confidence level (HIGH, MEDIUM, or LOW).', 'error');
        if (confSelect) confSelect.focus();
        return;
      }

      const notesInput = document.getElementById('reviewer-notes');
      const notes = notesInput ? notesInput.value.trim() : '';

      isSaving = true;
      const saveBtn = document.getElementById('btn-save-next');
      if (saveBtn) {
        saveBtn.disabled = true;
        saveBtn.innerText = 'Saving...';
      }
      showValidationMessage(`⏳ Saving annotation for ${p.pair_id}...`, 'info');

      try {
        const selector = document.getElementById('dataset-selector');
        const ds = selector ? selector.value : 'combined';

        const res = await fetch('/api/label', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            pair_id: p.pair_id,
            human_label: currentSelectedLabel,
            confidence_rating: conf,
            reviewer_notes: notes,
            dataset: ds,
          })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.error || `HTTP ${res.status}: ${res.statusText}`);
        }

        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Server indicated save failed.');

        p.human_label = currentSelectedLabel;
        p.confidence_rating = conf;
        p.reviewer_notes = notes;
        updateStats(data.stats);

        showValidationMessage(`✓ Saved ${p.pair_id} successfully! Navigating to next pending pair...`, 'success');
        navigateToNextPendingPair();

      } catch (err) {
        console.error('Failed to save label:', err);
        showValidationMessage(`❌ Error saving annotation: ${err.message}`, 'error');
        alert(`Error saving annotation for ${p.pair_id}:\\n${err.message}`);
      } finally {
        isSaving = false;
        if (saveBtn) {
          saveBtn.disabled = false;
          saveBtn.innerText = 'Save & Next ▶';
        }
      }
    }

    function navigateToNextPendingPair() {
      const currentActualIdx = filteredIndices[currentFilteredIdx];
      let targetActualIdx = -1;

      for (let i = currentActualIdx + 1; i < allPairs.length; i++) {
        const lbl = (allPairs[i].human_label || 'PENDING').toUpperCase();
        if (lbl === 'PENDING') { targetActualIdx = i; break; }
      }
      if (targetActualIdx === -1) {
        for (let i = 0; i < currentActualIdx; i++) {
          const lbl = (allPairs[i].human_label || 'PENDING').toUpperCase();
          if (lbl === 'PENDING') { targetActualIdx = i; break; }
        }
      }

      const filterVal = document.getElementById('filter-select').value;
      if (targetActualIdx !== -1) {
        if (filterVal === 'PENDING') {
          filteredIndices = [];
          for (let i = 0; i < allPairs.length; i++) {
            if ((allPairs[i].human_label || 'PENDING').toUpperCase() === 'PENDING') {
              filteredIndices.push(i);
            }
          }
          currentFilteredIdx = filteredIndices.indexOf(targetActualIdx);
          if (currentFilteredIdx === -1) currentFilteredIdx = 0;
          renderCurrentPair();
        } else {
          const pos = filteredIndices.indexOf(targetActualIdx);
          if (pos !== -1) {
            currentFilteredIdx = pos;
            renderCurrentPair();
          } else {
            if (currentFilteredIdx < filteredIndices.length - 1) currentFilteredIdx++;
            renderCurrentPair();
          }
        }
      } else {
        alert('All candidate pairs in this dataset have been reviewed!');
        if (filterVal === 'PENDING') {
          document.getElementById('filter-select').value = 'ALL';
          applyFilter();
        } else {
          if (currentFilteredIdx < filteredIndices.length - 1) currentFilteredIdx++;
          renderCurrentPair();
        }
      }
    }

    function prevPair() {
      if (currentFilteredIdx > 0) {
        currentFilteredIdx--;
        renderCurrentPair();
      }
    }

    function nextPair() {
      if (currentFilteredIdx < filteredIndices.length - 1) {
        currentFilteredIdx++;
        renderCurrentPair();
      } else {
        alert('You have reached the end of the filtered list.');
      }
    }

    async function showReadinessModal() {
      const modal = document.getElementById('readiness-modal');
      const content = document.getElementById('readiness-content');
      modal.style.display = 'flex';
      content.innerText = 'Calculating metrics...';
      try {
        const selector = document.getElementById('dataset-selector');
        const ds = selector ? selector.value : 'combined';
        const res = await fetch(`/api/readiness?dataset=${ds}`);
        const data = await res.json();
        content.innerText = data.text_report;
      } catch (err) {
        content.innerText = 'Error generating readiness report: ' + err.message;
      }
    }

    function closeReadinessModal() {
      document.getElementById('readiness-modal').style.display = 'none';
    }

    async function exportCSV() {
      try {
        const selector = document.getElementById('dataset-selector');
        const ds = selector ? selector.value : 'combined';
        const res = await fetch('/api/export', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ dataset: ds })
        });
        const data = await res.json();
        alert(`Export Successful!\\nSaved to: ${data.export_path}\\nConfirmed pairs: ${data.confirmed_count}\\nTraining Status: ${data.training_ready ? 'READY' : 'NOT READY'}`);
      } catch (err) {
        alert('Error exporting CSV: ' + err.message);
      }
    }

    // Keyboard Shortcuts: 1, 0, U select only
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === '1') { e.preventDefault(); selectLabel('1'); }
      if (e.key === '0') { e.preventDefault(); selectLabel('0'); }
      if (e.key === 'u' || e.key === 'U') { e.preventDefault(); selectLabel('UNCERTAIN'); }
      if (e.key === 'ArrowRight') nextPair();
      if (e.key === 'ArrowLeft') prevPair();
    });

    loadData();
  </script>
</body>
</html>
"""


class ReviewerHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _resolve_worksheet_path(self, dataset_param: Optional[str] = None) -> Path:
        """Resolve which worksheet CSV to load based on dataset parameter."""
        if dataset_param == "batch1":
            return WORKSHEET_PATH
        if COMBINED_WORKSHEET_PATH.exists():
            return COMBINED_WORKSHEET_PATH
        return WORKSHEET_PATH

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        ds_param = query.get("dataset", [None])[0]

        if url.path == "/" or url.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif url.path == "/api/pairs":
            target_path = self._resolve_worksheet_path(ds_param)
            df = load_worksheet_df(target_path)
            labels = df["human_label"].fillna("PENDING").astype(str).str.strip().str.upper()
            labels = labels.replace({"1.0": "1", "0.0": "0"})
            stats = {
                "total": len(df),
                "pos": int((labels == "1").sum()),
                "neg": int((labels == "0").sum()),
                "unc": int((labels == "UNCERTAIN").sum()),
                "pen": int((labels == "PENDING").sum()),
            }
            clean_records = clean_dataframe_for_json(df)
            response = {"stats": stats, "pairs": clean_records, "dataset_file": target_path.name}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response, allow_nan=False, default=str).encode("utf-8"))

        elif url.path == "/api/readiness":
            target_path = self._resolve_worksheet_path(ds_param)
            df = load_worksheet_df(target_path)
            report = compute_training_readiness(df)
            text_rep = format_readiness_report_text(report)
            response = {"metrics": report, "text_report": text_rep}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response, allow_nan=False, default=str).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if url.path == "/api/label":
            pair_id = payload.get("pair_id")
            label = str(payload.get("human_label", "")).strip().upper()
            conf = str(payload.get("confidence_rating", "")).strip().upper()
            notes = str(payload.get("reviewer_notes", "")).strip()
            ds_param = payload.get("dataset")

            if not pair_id:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Missing mandatory pair_id."}).encode("utf-8"))
                return

            if label not in ["1", "0", "UNCERTAIN"]:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": f"Invalid ground-truth label: '{label}'. Must be '1', '0', or 'UNCERTAIN'."
                }).encode("utf-8"))
                return

            if conf not in ["HIGH", "MEDIUM", "LOW"]:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": f"Invalid confidence rating: '{conf}'. Must be 'HIGH', 'MEDIUM', or 'LOW'."
                }).encode("utf-8"))
                return

            target_path = self._resolve_worksheet_path(ds_param)
            df = load_worksheet_df(target_path)
            matched = df.index[df["pair_id"] == pair_id].tolist()

            # If not in target, try fallback to WORKSHEET_PATH
            if not matched and target_path != WORKSHEET_PATH and WORKSHEET_PATH.exists():
                df = load_worksheet_df(WORKSHEET_PATH)
                matched = df.index[df["pair_id"] == pair_id].tolist()
                target_path = WORKSHEET_PATH

            if not matched:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": False,
                    "error": f"Candidate pair '{pair_id}' was not found in annotation worksheet."
                }).encode("utf-8"))
                return

            idx = matched[0]
            df.at[idx, "human_label"] = label
            df.at[idx, "confidence_rating"] = conf
            df.at[idx, "reviewer_notes"] = notes
            df.to_csv(target_path, index=False)

            # If saving in combined and pair_id is in batch 1, keep batch 1 worksheet in sync
            if target_path == COMBINED_WORKSHEET_PATH and WORKSHEET_PATH.exists():
                b1_df = load_worksheet_df(WORKSHEET_PATH)
                b1_match = b1_df.index[b1_df["pair_id"] == pair_id].tolist()
                if b1_match:
                    b1_df.at[b1_match[0], "human_label"] = label
                    b1_df.at[b1_match[0], "confidence_rating"] = conf
                    b1_df.at[b1_match[0], "reviewer_notes"] = notes
                    b1_df.to_csv(WORKSHEET_PATH, index=False)

            labels = df["human_label"].fillna("PENDING").astype(str).str.strip().str.upper()
            labels = labels.replace({"1.0": "1", "0.0": "0"})
            stats = {
                "total": len(df),
                "pos": int((labels == "1").sum()),
                "neg": int((labels == "0").sum()),
                "unc": int((labels == "UNCERTAIN").sum()),
                "pen": int((labels == "PENDING").sum()),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True, "pair_id": pair_id, "stats": stats}, allow_nan=False, default=str).encode("utf-8"))

        elif url.path == "/api/export":
            ds_param = payload.get("dataset")
            target_path = self._resolve_worksheet_path(ds_param)
            df = load_worksheet_df(target_path)
            out_file = export_labelled_dataset(df)
            labels = df["human_label"].fillna("PENDING").astype(str).str.strip().str.upper()
            labels = labels.replace({"1.0": "1", "0.0": "0"})
            confirmed = int((labels.isin(["1", "0"])).sum())
            readiness = compute_training_readiness(df)
            is_ready = readiness["10_training_readiness"]["is_ready_for_training"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({
                    "success": True,
                    "export_path": str(out_file),
                    "confirmed_count": confirmed,
                    "training_ready": is_ready,
                    "readiness_status": readiness["10_training_readiness"]["status"],
                }, allow_nan=False, default=str).encode("utf-8")
            )

        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 8050):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, ReviewerHTTPHandler)
    print("=" * 80)
    print(f"CSE 276 MATERIAL REVIEWER WEB APP RUNNING AT: http://localhost:{port}")
    print("=" * 80)
    print("Serving integrated candidate review pairs (Batch 1 + Batch 2).")
    print("Features:")
    print("  - Side-by-side material card comparison with physical attributes and source references")
    print("  - Real-time conflict alerts (diameter, grade, pressure, thread mismatches)")
    print("  - AI advisory suggestions and engineering rationale (strictly separated from human labels)")
    print("  - Keyboard shortcuts: [1] Equivalent, [0] Not Equivalent, [U] Uncertain (selects only)")
    print("  - Explicit 'Save & Next' submission with mandatory confidence validation")
    print("  - One-click Training Readiness Audit and CSV Export")
    print("Press Ctrl+C to stop the server.")
    print("=" * 80)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server.")
        httpd.server_close()


if __name__ == "__main__":
    port = 8050
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port)
