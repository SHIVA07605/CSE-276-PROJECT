"""
Streamlit Main Application Entry Point.

Material Code Harmonisation Across Public Enterprises (CSE 276 CA-1 Project).
Provides a multi-page procurement navigation interface.
"""

import streamlit as st

st.set_page_config(
    page_title="Material Code Harmonisation",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚙️ Material Code Harmonisation System")
st.subheader("CSE 276 – CA-1 Academic Project | AI & ML Specialisation")

st.markdown("""
Welcome to the **Material Code Harmonisation Platform**.

### Official Project Objectives:
1. **Objective 1:** Build a matching system that groups functionally equivalent material records and proposes one harmonised code with evidence.
2. **Objective 2:** Produce an entity resolution model with pairwise precision/recall over a labelled sample.
3. **Objective 3:** Report pairwise precision/recall/F1, cluster purity. Baseline: exact string match. Deliver Reviewer UI showing proposed merges with evidence.

---

### Platform Navigation:
Use the sidebar on the left to navigate through the modules:
- 📊 **Dashboard:** High-level inventory statistics & harmonisation KPIs.
- 📥 **Upload Materials:** Multi-enterprise catalogue ingestion.
- 🔍 **Material Matching:** Candidate retrieval & match scoring.
- 🧑‍⚖️ **Reviewer Workspace:** Human-in-the-loop review with technical evidence.
- 🏷️ **Harmonised Master:** Approved canonical materials registry.
- 📈 **Model Evaluation:** Pairwise P/R/F1 curves, confusion matrices & cluster purity.

> *Note: This application is in development phase. Machine Learning modules will be activated incrementally as experiments conclude.*
""")
