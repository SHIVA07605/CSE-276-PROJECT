"""
Page: Model Evaluation and Performance Metrics.

Required by Objective 3:
Displays Pairwise Precision, Recall, F1, and Cluster Purity compared against
the Exact String Match baseline.
"""

import streamlit as st

st.set_page_config(page_title="Model Evaluation | Material Harmonisation", layout="wide")
st.title("📈 Model Evaluation & Baseline Comparison")

st.markdown("""
### Official Evaluation Benchmark (Objective 3):
- **Pairwise Metrics:** Precision, Recall, F1-Score.
- **Clustering Quality:** Cluster Purity against ground-truth material equivalence groups.
- **Baseline:** Exact String Matching.
""")

st.info("Performance charts and confusion matrices will display evaluated metrics once model training completes.")
