"""
candidate_viewer.py
────────────────────
V8 — Multiphase Liver CT Candidate Viewer (Stage 1–3)

Research prototype only. Not validated for clinical use.
Not for diagnosis, treatment, or patient management.

Workflow:
  Stage 1 — Geometry validation + synchronized 2D viewer
  Stage 2 — 3D single-phase intensity anomaly proposals
  Stage 3 — Multiphase enhancement scoring (cross-phase HU delta)
"""

import io as _io
import json
import tempfile
import os
import math
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.io import load_nifti
from utils.nifti_checks import (
    check_geometry_consistency,
    check_mask_compatibility,
    check_hu_plausibility,
    phases_are_registered,
)
from utils.visualization import get_2d_slice
from utils.anomaly_3d import propose_3d_candidates
from utils.enhancement import compute_multiphase_enhancement
from utils.evaluation import (
    compute_voxelwise_metrics,
    compute_lesion_level_metrics,
    make_patient_split,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Creator")
st.sidebar.markdown("**AKSHAT TIWARI**")
st.sidebar.markdown("---")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-title {
    font-size: 1.9rem; font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.5px;
}
.sub-title { color: #94a3b8; font-size: 0.85rem; margin-top: -6px; margin-bottom: 16px; }
.metric-box {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 12px 16px; text-align: center;
}
.metric-label { color: #64748b; font-size: 0.70rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { color: #e2e8f0; font-size: 1.1rem; font-weight: 600; margin-top: 2px; }
.warn-box {
    background: rgba(245,158,11,0.08); border-left: 3px solid #f59e0b;
    border-radius: 4px; padding: 8px 14px; font-size: 0.80rem; color: #fbbf24; margin-bottom: 8px;
}
.err-box {
    background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444;
    border-radius: 4px; padding: 8px 14px; font-size: 0.80rem; color: #fca5a5; margin-bottom: 8px;
}
.ok-box {
    background: rgba(52,211,153,0.08); border-left: 3px solid #34d399;
    border-radius: 4px; padding: 8px 14px; font-size: 0.80rem; color: #6ee7b7; margin-bottom: 8px;
}
.disclaimer {
    background: rgba(99,102,241,0.07); border: 1px solid #6366f1;
    border-radius: 8px; padding: 10px 16px; font-size: 0.78rem; color: #a5b4fc;
    margin-bottom: 16px; line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">V8 — Multiphase Liver CT Candidate Viewer</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">3D intensity anomaly proposals · Multiphase enhancement scoring · '
    'Offline evaluation against GT masks</p>', unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Research prototype only.</strong> Not validated for clinical use.
Not for diagnosis, treatment, or patient management.
All output uses research terminology: "intensity anomaly candidate", "enhancement-pattern candidate", "research score".
The term "HCC" never appears in detection output.
</div>
""", unsafe_allow_html=True)


# ── HELPERS ────────────────────────────────────────────────────────────────────

def nifti_from_upload(uploaded_file) -> dict | None:
    """Load a NIfTI from a Streamlit UploadedFile object."""
    if uploaded_file is None:
        return None
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        return load_nifti(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def metric_card(col, label: str, value: str):
    col.markdown(
        f'<div class="metric-box"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def show_alert(msg: str, kind: str = "warn"):
    css_cls = {"warn": "warn-box", "err": "err-box", "ok": "ok-box"}.get(kind, "warn-box")
    icon = {"warn": "⚠️", "err": "🚫", "ok": "✅"}.get(kind, "⚠️")
    st.markdown(f'<div class="{css_cls}">{icon} {msg}</div>', unsafe_allow_html=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Phase Uploads")
    art_file = st.file_uploader("Arterial phase (.nii/.nii.gz) *required*",
                                 type=["nii", "gz"], key="art")
    plain_file = st.file_uploader("Plain / non-contrast (optional)",
                                   type=["nii", "gz"], key="plain")
    venous_file = st.file_uploader("Venous / portal-venous (optional)",
                                    type=["nii", "gz"], key="venous")
    delayed_file = st.file_uploader("Delayed (optional)", type=["nii", "gz"], key="delayed")

    st.markdown("---")
    st.markdown("### 🫀 Masks")
    liver_file = st.file_uploader("Liver mask (.nii/.nii.gz) — required for detection",
                                   type=["nii", "gz"], key="liver")
    lesion_file = st.file_uploader("GT lesion mask (.nii/.nii.gz) — optional, evaluation only",
                                    type=["nii", "gz"], key="lesion")

    st.markdown("---")
    st.markdown("### 🔬 Detection Settings")
    run_detection = art_file is not None and liver_file is not None
    z_thresh = st.slider("Anomaly Z-score threshold", 1.5, 4.0, 2.5, 0.25,
                         help="SD above liver mean to flag as anomaly candidate.")
    direction = st.selectbox("Detection direction",
                              ["hyper (bright)", "hypo (dark)", "both"],
                              index=0)
    direction_map = {"hyper (bright)": "hyper", "hypo (dark)": "hypo", "both": "both"}
    direction_key = direction_map[direction]

    smooth_mm = st.slider("Gaussian smoothing (mm)", 0.0, 3.0, 1.0, 0.5,
                          help="Applied before anomaly map. Does not affect HU feature values.")
    open_mm = st.slider("Morphological opening radius (mm)", 0.5, 4.0, 1.5, 0.5)
    min_vol = st.slider("Flag small below (mm³)", 50, 1000, 100, 50)
    max_vol = st.slider("Flag large above (mm³)", 10000, 500000, 200000, 10000)
    suppress_vessels = st.checkbox("Flag possible vessels (Frangi heuristic)", value=True)

    st.markdown("---")
    st.markdown("### 📐 Slice Navigator")
    slice_axis = st.selectbox("View axis",
                               ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], index=0)
    axis_map = {"Axial (Z)": 2, "Coronal (Y)": 1, "Sagittal (X)": 0}
    axis = axis_map[slice_axis]
    slice_idx = st.slider("Slice index", 0, 500, 0, 1,
                          help="Actual range clamped to volume size after loading.")

    st.markdown("---")
    st.markdown("### 🎨 Visualization")
    overlay_alpha = st.slider("Candidate overlay opacity", 0.1, 1.0, 0.5, 0.05)
    colorscale = st.selectbox("Colormap", ["Hot", "Plasma", "Viridis", "Turbo"], index=0)


# ── EARLY EXIT ─────────────────────────────────────────────────────────────────
if art_file is None:
    st.markdown("---")
    st.info("⬆️  Upload at least the **Arterial phase** NIfTI to begin.")
    st.markdown("""
    **What this tool does (Stage 1–3):**
    - **Stage 1:** Load multi-phase CT volumes, validate geometry/spacing/affine consistency
    - **Stage 2:** Propose 3D intensity anomaly candidates within the liver mask using raw HU
    - **Stage 3:** Score candidates using cross-phase HU changes (arterial → venous)

    **Output terminology:** All candidates are labeled *"intensity anomaly candidate"* or
    *"enhancement-pattern candidate"*. The term "HCC" never appears in detection output.
    It appears only in the offline evaluation tab when a ground-truth mask is supplied.

    **Liver mask is required for any quantitative detection.**
    Without it, only the geometry viewer and HU histogram are available.
    """)
    st.stop()


# ── LOAD PHASES ────────────────────────────────────────────────────────────────
with st.spinner("Loading NIfTI volumes…"):
    art_nifti    = nifti_from_upload(art_file)
    plain_nifti  = nifti_from_upload(plain_file)
    venous_nifti = nifti_from_upload(venous_file)
    delayed_nifti = nifti_from_upload(delayed_file)
    liver_nifti  = nifti_from_upload(liver_file)
    lesion_nifti = nifti_from_upload(lesion_file)

phases = {
    "arterial": art_nifti,
    "plain":    plain_nifti,
    "venous":   venous_nifti,
    "delayed":  delayed_nifti,
}
loaded_phases = {k: v for k, v in phases.items() if v is not None}


# ── GEOMETRY CHECKS ────────────────────────────────────────────────────────────
geom_warnings: list[str] = []
cross_phase_ok = False

if len(loaded_phases) >= 2:
    geom_warnings += check_geometry_consistency(
        list(loaded_phases.values()), list(loaded_phases.keys())
    )
    cross_phase_ok = phases_are_registered(loaded_phases)

if liver_nifti is not None:
    geom_warnings += check_mask_compatibility(
        art_nifti, liver_nifti, "Arterial CT", "Liver mask"
    )

hu_warnings = check_hu_plausibility(
    art_nifti, "Arterial",
    liver_mask=liver_nifti["data"].astype(np.uint8) if liver_nifti else None,
)
geom_warnings += hu_warnings


# ── TABS ───────────────────────────────────────────────────────────────────────
tab_geom, tab_slice, tab_detect, tab_eval = st.tabs([
    "📐 Geometry", "🔬 2D Viewer", "🎯 3D Candidates", "📊 Evaluation"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — GEOMETRY
# ════════════════════════════════════════════════════════════════════════════════
with tab_geom:
    st.subheader("Geometry Validation")

    if geom_warnings:
        for w in geom_warnings:
            show_alert(w, "warn")
    else:
        show_alert("All geometry checks passed.", "ok")

    # Metrics per loaded phase
    for phase_label, nifti in loaded_phases.items():
        sp = nifti["spacing"]
        sh = nifti["shape"]
        ori = "".join(nifti["orientation"])
        with st.expander(f"📦 {phase_label.capitalize()} phase — {sh}", expanded=(phase_label == "arterial")):
            c1, c2, c3, c4 = st.columns(4)
            metric_card(c1, "Dimensions", f"{sh[0]}×{sh[1]}×{sh[2]}")
            metric_card(c2, "Spacing (mm)", f"{sp[0]:.2f}×{sp[1]:.2f}×{sp[2]:.2f}")
            metric_card(c3, "Orientation", ori)
            metric_card(c4, "Dtype", str(nifti["dtype"]))

    if liver_nifti is not None:
        liver_mask = liver_nifti["data"].astype(np.uint8)
        liver_vox = int(liver_mask.sum())
        sp = art_nifti["spacing"]
        liver_vol = liver_vox * sp[0] * sp[1] * sp[2] / 1000.0

        st.markdown("---")
        st.subheader("Liver Mask Summary")
        lc1, lc2, lc3 = st.columns(3)
        metric_card(lc1, "Liver voxels", f"{liver_vox:,}")
        metric_card(lc2, "Liver volume (cm³)", f"{liver_vol:.1f}")
        metric_card(lc3, "Liver label values", str(np.unique(liver_mask).tolist()))

    if lesion_nifti is not None:
        les_mask = lesion_nifti["data"].astype(np.uint8)
        les_vox = int((les_mask > 0).sum())
        les_vol = les_vox * sp[0] * sp[1] * sp[2] / 1000.0
        st.markdown("---")
        st.subheader("GT Lesion Mask Summary (evaluation only)")
        show_alert("Ground-truth mask loaded — used only in the Evaluation tab, not for detection.", "ok")
        ec1, ec2, ec3 = st.columns(3)
        metric_card(ec1, "GT lesion voxels", f"{les_vox:,}")
        metric_card(ec2, "GT lesion volume (cm³)", f"{les_vol:.2f}")
        metric_card(ec3, "Label values", str(np.unique(les_mask).tolist()))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — 2D VIEWER
# ════════════════════════════════════════════════════════════════════════════════
with tab_slice:
    st.subheader("Synchronized 2D Views")

    # Clamp slice index to volume
    max_idx = art_nifti["data"].shape[axis] - 1
    s_idx = min(slice_idx, max_idx)

    phase_cols = st.columns(len(loaded_phases))
    for col, (phase_label, nifti) in zip(phase_cols, loaded_phases.items()):
        data = nifti["data"]
        slc = get_2d_slice(data, axis, s_idx).T
        fig = px.imshow(
            slc, color_continuous_scale="gray", origin="lower",
            title=f"{phase_label.capitalize()} — slice {s_idx}/{data.shape[axis]-1}",
        )
        # Liver overlay
        if liver_nifti is not None:
            liver_slc = get_2d_slice(liver_nifti["data"], axis, s_idx).T
            liver_overlay = np.ma.masked_where(liver_slc == 0, liver_slc)
            fig.add_trace(go.Heatmap(
                z=liver_overlay,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, f"rgba(96,165,250,{overlay_alpha:.2f})"]],
                showscale=False, name="Liver mask",
            ))
        # GT lesion overlay
        if lesion_nifti is not None:
            les_slc = get_2d_slice(lesion_nifti["data"], axis, s_idx).T
            les_overlay = np.ma.masked_where(les_slc == 0, les_slc)
            fig.add_trace(go.Heatmap(
                z=les_overlay,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(239,68,68,0.6)"]],
                showscale=False, name="GT lesion",
            ))
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            title_font=dict(size=13, color="#94a3b8"),
        )
        col.plotly_chart(fig, use_container_width=True)

    # HU histogram within liver mask on this slice
    if liver_nifti is not None:
        st.markdown("---")
        st.markdown("**HU distribution in liver ROI — this slice (raw HU, no normalization)**")
        art_slc = get_2d_slice(art_nifti["data"], axis, s_idx)
        liver_slc = get_2d_slice(liver_nifti["data"], axis, s_idx)
        liver_hu_vals = art_slc[liver_slc > 0]
        if len(liver_hu_vals) > 0:
            hist_vals, hist_bins = np.histogram(liver_hu_vals, bins=60)
            fig_h = go.Figure(go.Bar(
                x=hist_bins[:-1], y=hist_vals,
                marker=dict(color=hist_vals, colorscale="Plasma", showscale=False),
                name="Liver HU",
            ))
            fig_h.update_layout(
                xaxis_title="HU (raw)",
                yaxis_title="Voxel count",
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#64748b"),
                yaxis=dict(color="#64748b"),
                height=200,
            )
            st.plotly_chart(fig_h, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — 3D CANDIDATES
# ════════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.subheader("3D Intensity Anomaly Candidates")

    if liver_nifti is None:
        show_alert(
            "Liver mask not uploaded. Detection requires a liver mask to confine "
            "candidate proposals to liver parenchyma. Upload a liver mask in the sidebar.",
            "err",
        )
        st.stop()

    # Warn if geometry issues exist that would invalidate cross-phase scoring
    if geom_warnings:
        for w in geom_warnings:
            show_alert(w, "warn")

    liver_mask = liver_nifti["data"].astype(np.uint8)
    spacing = art_nifti["spacing"]
    vol = art_nifti["data"].astype(np.float32)

    # Run detection
    with st.spinner("Running 3D anomaly proposal… (may take 20–60s for large volumes)"):
        candidates = propose_3d_candidates(
            volume=vol,
            liver_mask=liver_mask,
            spacing_mm=spacing,
            z_score_thresh=z_thresh,
            direction=direction_key,
            smooth_sigma_mm=smooth_mm,
            open_radius_mm=open_mm,
            min_vol_mm3=float(min_vol),
            max_vol_mm3=float(max_vol),
            suppress_vessels=suppress_vessels,
        )

    # Multiphase enhancement scoring
    if cross_phase_ok and len(loaded_phases) >= 2:
        show_alert(
            f"Multiphase enhancement scoring enabled — {len(loaded_phases)} registered phases detected.",
            "ok",
        )
        with st.spinner("Computing multiphase enhancement scores…"):
            from scipy.ndimage import label as cc_label_fn
            # Rebuild candidate masks for enhancement sampling
            shape = vol.shape
            labeled_map = np.zeros(shape, dtype=np.int32)
            for c in candidates:
                (z0, z1), (y0, y1), (x0, x1) = c["bbox_voxel"]
                labeled_map[z0:z1+1, y0:y1+1, x0:x1+1] = c["id"]

            updated = []
            for c in candidates:
                cid = c["id"]
                cmask = (labeled_map == cid).astype(np.uint8)
                c_updated = compute_multiphase_enhancement(
                    candidate=c,
                    candidate_mask=cmask,
                    phases=phases,
                    liver_mask=liver_mask,
                    spacing_mm=spacing,
                )
                updated.append(c_updated)
            candidates = updated
    elif len(loaded_phases) >= 2 and not cross_phase_ok:
        show_alert(
            "Multiple phases uploaded but geometry checks failed — "
            "cross-phase enhancement scoring is disabled. All candidates labeled "
            "'single_phase_intensity_anomaly'.", "warn",
        )

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_total = len(candidates)
    n_small = sum(1 for c in candidates if c["size_flag"] == "small")
    n_vessel = sum(1 for c in candidates if c["possible_vessel"])
    n_washout = sum(1 for c in candidates
                    if c.get("enhancement_pattern") == "hyperarterial_washout_candidate")

    st.markdown("")
    mc1, mc2, mc3, mc4 = st.columns(4)
    metric_card(mc1, "Total candidates", str(n_total))
    metric_card(mc2, "Flagged small", str(n_small))
    metric_card(mc3, "Possible vessels", str(n_vessel))
    metric_card(mc4, "Washout candidates", str(n_washout) if cross_phase_ok else "N/A (single-phase)")

    if n_total == 0:
        show_alert("No candidates found. Try lowering the Z-score threshold or "
                   "increasing the detection window.", "warn")
        st.stop()

    # ── Sort + filter UI ──────────────────────────────────────────────────────
    st.markdown("---")
    sort_by = st.selectbox(
        "Sort candidates by",
        ["research_score ↓", "volume_mm3 ↓", "contrast_vs_liver ↓", "sphericity ↓"],
        index=0,
    )
    sort_key = sort_by.split(" ")[0]
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get(sort_key) or 0.0,
        reverse=True,
    )

    score_min = st.slider("Minimum research score to display", 0.0, 1.0, 0.0, 0.05)
    filtered = [c for c in sorted_candidates if (c.get("research_score") or 0.0) >= score_min]

    # ── Heatmap overlay on slice ───────────────────────────────────────────────
    st.markdown("### Candidate overlay on slice")
    heat_vol = np.zeros(vol.shape, dtype=np.float32)
    for c in filtered:
        (z0, z1), (y0, y1), (x0, x1) = c["bbox_voxel"]
        heat_vol[z0:z1+1, y0:y1+1, x0:x1+1] = max(
            heat_vol[z0:z1+1, y0:y1+1, x0:x1+1].max(),
            c.get("research_score") or 0.0,
        )

    max_idx2 = vol.shape[axis] - 1
    s_idx2 = min(slice_idx, max_idx2)

    art_slc2 = get_2d_slice(vol, axis, s_idx2).T
    heat_slc = get_2d_slice(heat_vol, axis, s_idx2).T

    fig_ov = px.imshow(art_slc2, color_continuous_scale="gray", origin="lower",
                        title=f"Arterial — slice {s_idx2} with candidate overlay")
    if heat_slc.max() > 0:
        fig_ov.add_trace(go.Heatmap(
            z=heat_slc,
            colorscale=colorscale,
            opacity=overlay_alpha,
            showscale=True,
            colorbar=dict(title=dict(text="Research score", font=dict(color="#94a3b8")),
                          tickfont=dict(color="#94a3b8"), thickness=12),
            name="Candidate score",
        ))
    fig_ov.update_layout(
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=13, color="#94a3b8"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_ov, use_container_width=True)

    # ── Candidate table ────────────────────────────────────────────────────────
    st.markdown("### Candidate table")
    show_alert(
        "All scores are heuristic research values — not diagnostic probabilities. "
        "Enhancement patterns are not LI-RADS criteria.", "warn",
    )

    import pandas as pd
    rows = []
    for c in filtered:
        rows.append({
            "ID": c["id"],
            "Vol (mm³)": c["volume_mm3"],
            "Diam (mm)": c["equivalent_diameter_mm"],
            "Mean HU (raw)": c["mean_hu_raw"],
            "Liver BG HU": c["liver_bg_hu"],
            "Contrast ΔHU": c["contrast_vs_liver"],
            "Sphericity": c["sphericity"],
            "Washout ΔHU": c.get("relative_washout"),
            "Pattern": c.get("enhancement_pattern", "—"),
            "Research score": c.get("research_score"),
            "Small?": "⚑" if c["size_flag"] == "small" else "",
            "Vessel?": "⚑" if c["possible_vessel"] else "",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=350)

    # ── JSON run config export ─────────────────────────────────────────────────
    run_config = {
        "tool": "V8 Candidate Viewer",
        "version": "0.1-stage123",
        "disclaimer": "Research prototype only. Not for clinical use.",
        "detection_settings": {
            "z_score_thresh": z_thresh,
            "direction": direction_key,
            "smooth_sigma_mm": smooth_mm,
            "open_radius_mm": open_mm,
            "min_vol_mm3": min_vol,
            "max_vol_mm3": max_vol,
            "suppress_vessels": suppress_vessels,
            "score_version": (candidates[0].get("research_score_version") if candidates else "unknown"),
        },
        "phases_available": list(loaded_phases.keys()),
        "cross_phase_scoring": cross_phase_ok,
        "n_candidates": n_total,
        "n_filtered": len(filtered),
    }
    st.download_button(
        "⬇️ Download run config (JSON)",
        data=json.dumps(run_config, indent=2),
        file_name="candidate_viewer_run_config.json",
        mime="application/json",
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION
# ════════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.subheader("Offline Evaluation (GT mask required)")

    if lesion_nifti is None:
        show_alert(
            "No ground-truth lesion mask uploaded. Upload a GT mask in the sidebar "
            "to compute evaluation metrics. This tab is for offline research evaluation only.",
            "warn",
        )
        st.stop()

    if liver_nifti is None:
        show_alert("Liver mask required to run detection before evaluation.", "err")
        st.stop()

    gt_mask = (lesion_nifti["data"] > 0).astype(np.uint8)

    # Need to run detection if not already done (may be first tab opened)
    try:
        n_cands = len(candidates)
    except NameError:
        show_alert("Run detection in the '3D Candidates' tab first.", "warn")
        st.stop()

    # Build binary prediction mask from candidates
    pred_mask = np.zeros(vol.shape, dtype=np.uint8)
    for c in candidates:
        (z0, z1), (y0, y1), (x0, x1) = c["bbox_voxel"]
        pred_mask[z0:z1+1, y0:y1+1, x0:x1+1] = 1

    with st.spinner("Computing evaluation metrics…"):
        vox_metrics = compute_voxelwise_metrics(pred_mask, gt_mask)
        les_metrics = compute_lesion_level_metrics(pred_mask, gt_mask, iou_threshold=0.1)

    show_alert(
        "These metrics evaluate the current threshold + settings against the uploaded GT mask. "
        "This is a single-scan result, not a patient-cohort evaluation. "
        "For FROC analysis, use the batch evaluation script (coming in Stage 4).",
        "warn",
    )

    # ── Voxel metrics ─────────────────────────────────────────────────────────
    st.markdown("#### Voxel-level metrics")
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    def _fmt(v):
        return f"{v:.3f}" if not (isinstance(v, float) and math.isnan(v)) else "N/A"

    metric_card(v1, "Dice", _fmt(vox_metrics["dice"]))
    metric_card(v2, "IoU", _fmt(vox_metrics["iou"]))
    metric_card(v3, "Sensitivity", _fmt(vox_metrics["sensitivity"]))
    metric_card(v4, "Specificity", _fmt(vox_metrics["specificity"]))
    metric_card(v5, "Precision", _fmt(vox_metrics["precision"]))
    metric_card(v6, "FPR", _fmt(vox_metrics["fpr"]))

    st.markdown("")
    tp_col, fp_col, tn_col, fn_col = st.columns(4)
    metric_card(tp_col, "TP voxels", f"{vox_metrics['TP']:,}")
    metric_card(fp_col, "FP voxels", f"{vox_metrics['FP']:,}")
    metric_card(tn_col, "TN voxels", f"{vox_metrics['TN']:,}")
    metric_card(fn_col, "FN voxels", f"{vox_metrics['FN']:,}")

    # ── Lesion-level metrics ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Lesion-level detection metrics")
    ll1, ll2, ll3, ll4 = st.columns(4)
    metric_card(ll1, "GT lesions total", str(les_metrics["total_gt_lesions"]))
    metric_card(ll2, "Detected GT lesions", str(les_metrics["detected_gt_lesions"]))
    metric_card(ll3, "Missed GT lesions", str(les_metrics["missed_gt_lesions"]))
    metric_card(ll4, "FP / scan", _fmt(les_metrics["fp_per_scan"]))

    l_sens = les_metrics["lesion_sensitivity"]
    st.metric("Lesion-level sensitivity", _fmt(l_sens),
              help="Fraction of GT lesion components detected (IoU ≥ 0.1)")

    # ── Comparison overlay ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Prediction vs GT overlay")
    max_idx3 = vol.shape[axis] - 1
    eval_slice = min(slice_idx, max_idx3)

    art_slc3 = get_2d_slice(vol, axis, eval_slice).T
    pred_slc = get_2d_slice(pred_mask, axis, eval_slice).T
    gt_slc = get_2d_slice(gt_mask, axis, eval_slice).T

    fig_eval = px.imshow(art_slc3, color_continuous_scale="gray", origin="lower",
                          title=f"Slice {eval_slice} — prediction (blue) vs GT (red)")
    if pred_slc.max() > 0:
        fig_eval.add_trace(go.Heatmap(
            z=np.ma.masked_where(pred_slc == 0, pred_slc),
            colorscale=[[0,"rgba(0,0,0,0)"], [1,f"rgba(96,165,250,{overlay_alpha})"]],
            showscale=False, name="Prediction",
        ))
    if gt_slc.max() > 0:
        fig_eval.add_trace(go.Heatmap(
            z=np.ma.masked_where(gt_slc == 0, gt_slc),
            colorscale=[[0,"rgba(0,0,0,0)"], [1,"rgba(239,68,68,0.5)"]],
            showscale=False, name="GT lesion",
        ))
    fig_eval.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=13, color="#94a3b8"),
    )
    st.plotly_chart(fig_eval, use_container_width=True)
