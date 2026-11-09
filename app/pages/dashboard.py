"""
Page: Procurement Dashboard & Inventory KPIs.
"""

import streamlit as st

st.set_page_config(page_title="Dashboard | Material Harmonisation", layout="wide")
st.title("📊 Procurement Inventory & Harmonisation Dashboard")

st.info("System Initialisation: Awaiting dataset ingestion and pipeline activation.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Catalogues", "0")
with col2:
    st.metric("Total Material Records", "0")
with col3:
    st.metric("Proposed Clusters", "0")
with col4:
    st.metric("Approved Master Items", "0")

st.markdown("### Cross-Enterprise Duplication Overview")
st.write("Visualisations showing enterprise catalog distribution will appear here once catalogues are loaded.")
