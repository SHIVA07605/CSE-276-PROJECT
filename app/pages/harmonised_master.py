"""
Page: Harmonised Master Catalogue.

Displays the definitive registry of human-approved harmonised material records,
their assigned canonical codes, and linked legacy enterprise stock codes.
"""

import streamlit as st

st.set_page_config(page_title="Harmonised Master | Material Harmonisation", layout="wide")
st.title("🏷️ Harmonised Material Master Catalogue")

st.markdown("""
Definitive registry of approved harmonised codes.
Enables cross-enterprise inventory searching and price benchmarking.
""")

st.info("The Harmonised Master Catalogue currently contains 0 approved entries.")
