"""
Home.py  — V8 Research Suite landing page
"""

import streamlit as st

st.set_page_config(
    page_title="V8 — Liver CT Research Suite",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 3rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 8px;
}
.hero-sub {
    color: #64748b; font-size: 1.05rem; max-width: 620px; line-height: 1.6;
}
.card {
    background: #1e293b; border: 1px solid #334155; border-radius: 16px;
    padding: 28px 28px 24px 28px; margin-bottom: 4px;
    transition: border-color 0.2s;
}
.card:hover { border-color: #60a5fa; }
.card-icon { font-size: 2.4rem; margin-bottom: 10px; }
.card-title { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin-bottom: 6px; }
.card-desc { color: #64748b; font-size: 0.88rem; line-height: 1.5; }
.card-tag {
    display: inline-block; background: rgba(96,165,250,0.12); color: #60a5fa;
    border-radius: 99px; padding: 2px 10px; font-size: 0.72rem;
    font-weight: 600; margin-top: 14px; margin-right: 4px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.disclaimer {
    background: rgba(99,102,241,0.07); border: 1px solid #6366f1;
    border-radius: 10px; padding: 14px 20px; color: #a5b4fc;
    font-size: 0.80rem; line-height: 1.6; margin-top: 40px;
}
.step {
    background: #1e293b; border-left: 3px solid #60a5fa;
    border-radius: 0 8px 8px 0; padding: 10px 16px;
    font-size: 0.85rem; color: #cbd5e1; margin-bottom: 8px;
}
.step b { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">V8 Liver CT<br>Research Suite</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">A research toolkit for multiphasic liver CT analysis — '
    '3D depth visualization, intensity anomaly detection, and multiphase enhancement scoring '
    'on the PLC-CECT dataset.</p>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── TOOL CARDS ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🌋</div>
        <div class="card-title">Depth Surface Analyzer</div>
        <div class="card-desc">
            Upload any CT slice or flat image and explore it as an interactive 3D
            surface mesh. Bright = elevated. Drag to rotate, scroll to zoom.
            Accepts PNG, JPEG, and NIfTI (.nii.gz) volumes.
        </div>
        <span class="card-tag">3D Visualization</span>
        <span class="card-tag">NIfTI</span>
        <span class="card-tag">Plotly</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Depth_Analyzer.py", label="→ Open Depth Analyzer", icon="🌋")

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🎯</div>
        <div class="card-title">Candidate Viewer</div>
        <div class="card-desc">
            3D intensity anomaly detection in multiphasic liver CT.
            Loads all four PLC-CECT phases, validates geometry, proposes 3D candidates
            using raw HU analysis, and scores them with cross-phase washout metrics.
        </div>
        <span class="card-tag">3D Detection</span>
        <span class="card-tag">Multiphase</span>
        <span class="card-tag">Evaluation</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Candidate_Viewer.py", label="→ Open Candidate Viewer", icon="🎯")

# ── QUICK START ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Quick Start")

st.markdown("""
<div class="step"><b>Step 1 — Depth Analyzer:</b> Upload any PNG/JPEG or a NIfTI CT slice → see it rendered as a rotatable 3D surface immediately.</div>
<div class="step"><b>Step 2 — Candidate Viewer (single phase):</b> Upload your arterial .nii.gz + liver mask → go to "3D Candidates" tab → adjust Z-score slider → see anomaly candidates overlaid on the scan.</div>
<div class="step"><b>Step 3 — Candidate Viewer (multiphase):</b> Also upload venous .nii.gz → geometry is checked automatically → candidates gain "Washout ΔHU" and enhancement pattern labels.</div>
<div class="step"><b>Step 4 — Evaluation:</b> Upload the GT lesion mask → "Evaluation" tab shows Dice, lesion-level sensitivity, and FP/scan for the current settings.</div>
<div class="step"><b>Step 5 — Save results:</b> Download the run config JSON from the Candidates tab to record exactly what settings produced your results.</div>
""", unsafe_allow_html=True)

# ── DATASET INFO ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️ About the PLC-CECT Dataset"):
    st.markdown("""
    **Primary Liver Cancer CECT Imaging Dataset** — publicly available on [PhysioNet](https://physionet.org).

    | Property | Value |
    |---|---|
    | Patients | 278 with liver cancer + 83 controls |
    | Cancer subtypes | HCC, ICC, cHCC-CCA |
    | CT phases per patient | Plain, Arterial, Venous, Delayed |
    | Labels | Expert voxel segmentation masks (liver + lesion) |
    | Format | NIfTI (.nii) |

    Files are indexed in `patient_data.csv`:
    - Column `CT File` → path to the CT phase volume
    - Column `Liver Mask File` → path to liver segmentation
    - Column `Mask File` → path to lesion segmentation (may be empty for controls)
    - Column `Stage` → C0=Plain, C1=Arterial, C2=Venous, C3=Delayed
    """)

# ── DISCLAIMER ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Research prototype only.</strong>
This tool is not validated for clinical use and must not be used for diagnosis,
treatment decisions, or patient management. All output is labeled with research
terminology ("intensity anomaly candidate", "research score") and does not
constitute a medical opinion. The term "HCC" appears only in offline evaluation
mode when comparing against ground-truth research labels.
</div>
""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown("**Stack:** Python · Streamlit · Plotly · NiBabel · SciPy")
c2.markdown("**Data:** PLC-CECT (PhysioNet)")
c3.markdown("**Status:** Research prototype v0.1")
