"""
Page: Candidate Generation and Material Matching Execution.
"""

import streamlit as st

st.set_page_config(page_title="Material Matching | Material Harmonisation", layout="wide")
st.title("🔍 Candidate Retrieval & Match Execution")

st.markdown("""
Execute TF-IDF blocking, semantic sentence embeddings (`all-MiniLM-L6-v2`), and pairwise classification.
""")

st.subheader("Matching Controls")
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Top-K Candidates per Record", min_value=1, max_value=50, value=15)
with col2:
    threshold = st.slider("Classification Match Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.button("Run Entity Resolution Pipeline", disabled=True, help="Disabled until model weights are trained and persisted.")
st.caption("Status: Awaiting trained classifier model in `models/`.")
