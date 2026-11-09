"""
Page: Multi-Enterprise Catalogue Ingestion.
"""

import streamlit as st

st.set_page_config(page_title="Upload Materials | Material Harmonisation", layout="wide")
st.title("📥 Upload & Ingest Enterprise Material Catalogues")

st.markdown("""
Upload CSV or Excel files containing material records from public enterprises.
**Mandatory Columns:** `enterprise_name`, `material_code`, `material_description`.
""")

uploaded_file = st.file_uploader(
    "Select catalogue file (.csv, .xlsx)",
    type=["csv", "xlsx"],
    help="Upload enterprise material catalogue export."
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}. Schema validation pending in Milestone 1.")
