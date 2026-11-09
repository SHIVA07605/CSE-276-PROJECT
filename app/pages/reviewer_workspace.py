"""
Page: Human-in-the-Loop Reviewer Workspace.

Mandatory Project Component:
Presents proposed merges with evidence, matching/conflicting attributes,
and approval/rejection/uncertain buttons to preserve human procurement oversight.
"""

import streamlit as st

st.set_page_config(page_title="Reviewer Workspace | Material Harmonisation", layout="wide")
st.title("🧑‍⚖️ Human-in-the-Loop Reviewer Workspace")

st.markdown("""
Review algorithmically proposed material merges with technical audit evidence.
No records are permanently merged into the Master Catalogue without human authorization.
""")

st.info("No active candidate matches awaiting review in the initial setup.")

# Prototype layout preview
with st.expander("Preview: Sample Reviewer Decision Interface", expanded=True):
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Record A (Source Enterprise)")
        st.write("**Enterprise:** Enterprise A (NTPC Sample)")
        st.write("**Material Code:** `MAT-101`")
        st.write("**Description:** Stainless Steel Bolt M10")
        st.write("**Extracted Thread:** `M10`")
        st.write("**Material Grade:** `STAINLESS STEEL`")

    with col_b:
        st.markdown("#### Record B (Candidate Match)")
        st.write("**Enterprise:** Enterprise B (BHEL Sample)")
        st.write("**Material Code:** `BOLT-450`")
        st.write("**Description:** SS Hex Bolt, 10mm")
        st.write("**Extracted Thread:** `M10` (from 10mm)")
        st.write("**Material Grade:** `STAINLESS STEEL` (SS expanded)")

    st.markdown("---")
    st.markdown("#### Model Recommendation & Supporting Evidence")
    ev1, ev2, ev3 = st.columns(3)
    ev1.metric("Predicted Similarity", "94.2%")
    ev2.metric("Matching Attributes", "Thread (M10), Material (SS)")
    ev3.metric("Conflicting Attributes", "None Detected")

    st.markdown("**Proposed Harmonised Master Code:** `HMC-FAST-M10-SS`")

    st.markdown("#### Reviewer Decision")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
        st.button("✅ Approve Harmonisation", key="btn_app_demo", disabled=True)
    with btn_col2:
        st.button("❌ Reject Match", key="btn_rej_demo", disabled=True)
    with btn_col3:
        st.button("⚠️ Mark Uncertain / Request Audit", key="btn_unc_demo", disabled=True)
