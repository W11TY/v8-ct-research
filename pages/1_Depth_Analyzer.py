"""
depth_analyzer.py
─────────────────
V8 — Depth Surface Analyzer
Converts flat 2D grayscale image intensities (e.g. a windowed CT slice) into a
pseudo-depth surface and renders an interactive 3D map.
Also accepts NIfTI (.nii / .nii.gz) volumes and extracts a chosen 2D slice.

Research/visualization only. Not a diagnostic instrument.
"""

import io
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from utils.io import load_image_as_array, load_nifti

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Creator")
st.sidebar.markdown("**AKSHAT TIWARI**")
st.sidebar.markdown("---")

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}

.sub-title {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: -8px;
    margin-bottom: 20px;
}

.metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}

.metric-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-value {
    color: #e2e8f0;
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 2px;
}

.warning-box {
    background: rgba(245, 158, 11, 0.08);
    border-left: 3px solid #f59e0b;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-bottom: 12px;
}

section-header {
    font-size: 1.0rem;
    font-weight: 600;
    color: #cbd5e1;
    letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">V8 — Depth Surface Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Intensity → Pseudo-Depth → Interactive 3D Surface &nbsp;|&nbsp; '
    'Research visualization only — not a diagnostic instrument.</p>',
    unsafe_allow_html=True,
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Input")
    uploaded = st.file_uploader(
        "Upload image or NIfTI volume",
        type=["png", "jpg", "jpeg", "tiff", "tif", "bmp", "nii", "gz"],
        help="Flat images (PNG/JPEG/TIFF) or NIfTI volumes (.nii / .nii.gz).",
    )

    # ── NIfTI slice controls (shown only for .nii/.gz files) ─────────────────
    _is_nifti = uploaded is not None and (
        uploaded.name.endswith(".nii") or uploaded.name.endswith(".nii.gz") or uploaded.name.endswith(".gz")
    )

    if _is_nifti:
        st.markdown("""
        <div class="warning-box">
        📦 NIfTI volume detected — select axis and slice below.
        </div>
        """, unsafe_allow_html=True)
        nii_axis = st.selectbox(
            "Slice axis",
            options=[0, 1, 2],
            format_func=lambda a: {0: "0 — Sagittal (X)", 1: "1 — Coronal (Y)", 2: "2 — Axial (Z)"}[a],
            index=2,
            help="Axis along which to extract a single 2D slice.",
        )
        # placeholder slider — real max set after load, use 0 as temporary default
        nii_slice_idx = st.slider(
            "Slice index",
            min_value=0, max_value=500, value=0, step=1,
            help="Scroll through slices along the chosen axis.",
        )
    else:
        nii_axis = 2
        nii_slice_idx = 0
        st.markdown("""
        <div class="warning-box">
        ⚠️ For 16-bit DICOM exports, pre-window to 8-bit PNG before upload
        (e.g. liver window: W=150, L=60 HU).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Preprocessing")

    sigma = st.slider(
        "Gaussian smoothing σ",
        min_value=0.0, max_value=8.0, value=1.5, step=0.5,
        help="Removes pixel noise before depth conversion. Higher = smoother surface.",
    )
    win_low = st.slider(
        "Intensity window — low percentile",
        min_value=0.0, max_value=10.0, value=0.5, step=0.5,
        help="Clips the darkest outlier pixels (like CT W/L floor).",
    )
    win_high = st.slider(
        "Intensity window — high percentile",
        min_value=90.0, max_value=100.0, value=99.5, step=0.5,
        help="Clips the brightest outlier pixels (like CT W/L ceiling).",
    )
    depth_scale = st.slider(
        "Depth scale",
        min_value=10, max_value=500, value=100, step=10,
        help="Maximum Z-height of the 3D surface. Increase for more dramatic relief.",
    )
    downsample = st.slider(
        "Downsample factor",
        min_value=1, max_value=8, value=2, step=1,
        help="Reduce image resolution for faster 3D rendering. "
             "1 = full resolution (may be slow for large images).",
    )

    st.markdown("---")
    st.markdown("### 🎨 Visualization")
    colorscale = st.selectbox(
        "3D colorscale",
        ["Viridis", "Plasma", "Hot", "Jet", "Turbo", "RdBu", "Magma", "Cividis"],
        index=0,
    )
    show_contours = st.checkbox("Show contour lines on surface", value=True)
    show_floor = st.checkbox("Project contours to floor", value=False)


# ── DEPTH CONVERSION CORE ─────────────────────────────────────────────────────
def intensity_to_depth(
    gray: np.ndarray,
    sigma: float = 1.5,
    win_low: float = 0.5,
    win_high: float = 99.5,
    scale: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2D float grayscale array to a pseudo-depth surface.

    Mapping:  bright pixel  →  high Z (closer / foreground peak)
              dark pixel    →  low  Z (deeper / background)

    Returns
    -------
    depth : (H, W) float array scaled to [0, scale]
    norm  : (H, W) float array normalized to [0, 1] (for colorizing)
    """
    # 1. Smooth
    smoothed = gaussian_filter(gray.astype(np.float64), sigma=sigma) if sigma > 0 else gray.astype(np.float64)

    # 2. Intensity windowing (percentile-based, like CT W/L)
    lo = float(np.percentile(smoothed, win_low))
    hi = float(np.percentile(smoothed, win_high))
    clipped = np.clip(smoothed, lo, hi)

    # 3. Normalize to [0, 1]
    denom = hi - lo
    norm = (clipped - lo) / (denom if denom > 1e-9 else 1.0)

    # 4. bright = high Z  →  norm is already 0→1 where 1 = bright = high
    depth = norm * scale

    return depth.astype(np.float32), norm.astype(np.float32)


def make_depth_heatmap(depth: np.ndarray, colorscale: str) -> go.Figure:
    """2D heatmap of the depth array."""
    fig = px.imshow(
        depth,
        color_continuous_scale=colorscale.lower(),
        origin="upper",
        labels={"color": "Depth"},
        title="Depth Heatmap",
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="Depth", thickness=12, len=0.7),
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=14, color="#94a3b8"),
    )
    return fig


def make_3d_surface(
    depth: np.ndarray,
    norm: np.ndarray,
    colorscale: str,
    show_contours: bool,
    show_floor: bool,
) -> go.Figure:
    """Interactive 3D surface from depth array."""
    H, W = depth.shape
    x_ratio = W / H

    contour_cfg = dict(
        z=dict(
            show=show_contours,
            usecolormap=True,
            highlightcolor="#a78bfa",
            project_z=show_floor,
            width=1,
        )
    )

    surface = go.Surface(
        z=depth,
        surfacecolor=norm,           # colour by original intensity
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text="Intensity", font=dict(color="#94a3b8")),
            thickness=14,
            len=0.65,
            tickfont=dict(color="#94a3b8"),
        ),
        lighting=dict(
            ambient=0.55,
            diffuse=0.85,
            specular=0.45,
            roughness=0.35,
            fresnel=0.2,
        ),
        lightposition=dict(x=200, y=200, z=300),
        contours=contour_cfg,
        hovertemplate="X: %{x}<br>Y: %{y}<br>Depth: %{z:.1f}<extra></extra>",
    )

    fig = go.Figure(data=[surface])
    fig.update_layout(
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=x_ratio, y=1.0, z=0.35),
            xaxis=dict(
                title=dict(text="X (px)", font=dict(color="#94a3b8")),
                showbackground=False, gridcolor="#1e293b",
                tickfont=dict(color="#64748b"),
            ),
            yaxis=dict(
                title=dict(text="Y (px)", font=dict(color="#94a3b8")),
                showbackground=False, gridcolor="#1e293b",
                tickfont=dict(color="#64748b"),
                autorange="reversed",
            ),
            zaxis=dict(
                title=dict(text="Depth", font=dict(color="#94a3b8")),
                showbackground=False, gridcolor="#1e293b",
                tickfont=dict(color="#64748b"),
            ),
            bgcolor="#0f172a",
        ),
        paper_bgcolor="#0f172a",
        margin=dict(l=0, r=0, b=0, t=0),
        title_font=dict(color="#e2e8f0"),
    )
    return fig


# ── MAIN BODY ─────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("---")
    st.info("⬆️  Upload a grayscale image or NIfTI volume using the sidebar to begin.")
    st.markdown("""
    **Supported formats:**
    - **PNG / JPEG / TIFF / BMP** — 2D grayscale CT slice exports
    - **NIfTI (.nii / .nii.gz)** — 3D CT volumes; choose axis + slice in sidebar

    **Workflow:**
    1. Upload a CT slice image or a NIfTI volume
    2. Adjust smoothing and depth scale until lesion peaks are clearly visible
    3. Use the interactive 3D surface to inspect lesion topology spatially

    **Clinical note:** HCC lesions in arterial-phase CT appear hyperintense (bright).
    They will manifest as elevated **peaks** on the depth surface — visually separating
    them from surrounding liver parenchyma.
    """)
    st.stop()

# ── LOAD & PROCESS ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load_image(file_bytes: bytes, name: str) -> np.ndarray:
    """Load a flat image (PNG/JPEG/TIFF) as a 2D grayscale array."""
    return load_image_as_array(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def cached_load_nifti(file_bytes: bytes, name: str) -> dict:
    """Load a NIfTI volume via nibabel (writes to temp file)."""
    import tempfile, os
    suffix = ".nii.gz" if name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        result = load_nifti(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return result


def extract_nifti_slice(nii: dict, axis: int, idx: int) -> np.ndarray:
    """Extract a single 2D slice from a NIfTI volume along the given axis."""
    data = nii["data"]  # shape (X, Y, Z) or (Z, Y, X) depending on orientation
    idx = int(np.clip(idx, 0, data.shape[axis] - 1))
    slices = [slice(None)] * data.ndim
    slices[axis] = idx
    return data[tuple(slices)].astype(np.float32)


nii_meta = None   # populated if NIfTI

if _is_nifti:
    with st.spinner("Loading NIfTI volume…"):
        nii_meta = cached_load_nifti(uploaded.getvalue(), uploaded.name)
    n_slices = nii_meta["data"].shape[nii_axis]
    # Clamp user-chosen slice index to actual volume bounds
    nii_slice_idx = int(np.clip(nii_slice_idx, 0, n_slices - 1))
    raw = extract_nifti_slice(nii_meta, nii_axis, nii_slice_idx)
else:
    with st.spinner("Loading image…"):
        raw = cached_load_image(uploaded.getvalue(), uploaded.name)

# Downsample for performance
if downsample > 1:
    raw_ds = raw[::downsample, ::downsample]
else:
    raw_ds = raw

H, W = raw_ds.shape

with st.spinner("Computing depth surface…"):
    depth, norm = intensity_to_depth(raw_ds, sigma=sigma,
                                     win_low=win_low, win_high=win_high,
                                     scale=float(depth_scale))

# ── METRICS ROW ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

def metric_card(col, label, value):
    col.markdown(
        f'<div class="metric-box"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

metric_card(m1, "Width (px)", f"{W * downsample}")
metric_card(m2, "Height (px)", f"{H * downsample}")
metric_card(m3, "Render size", f"{W} × {H}")
metric_card(m4, "Depth min", f"{depth.min():.1f}")
metric_card(m5, "Depth max", f"{depth.max():.1f}")

if nii_meta is not None:
    ax_name = {0: "Sagittal", 1: "Coronal", 2: "Axial"}[nii_axis]
    sp = nii_meta["spacing"]
    st.caption(
        f"📦 NIfTI | Volume: {nii_meta['shape']} | Spacing: {sp[0]:.2f}×{sp[1]:.2f}×{sp[2]:.2f} mm | "
        f"{ax_name} slice {nii_slice_idx} / {nii_meta['data'].shape[nii_axis] - 1}"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── SIDE-BY-SIDE: ORIGINAL + HEATMAP ─────────────────────────────────────────
col_orig, col_heat = st.columns(2)

with col_orig:
    fig_orig = px.imshow(
        raw_ds, color_continuous_scale="gray", origin="upper",
        title="Original (grayscale)",
    )
    fig_orig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=14, color="#94a3b8"),
    )
    st.plotly_chart(fig_orig, use_container_width=True)

with col_heat:
    fig_heat = make_depth_heatmap(depth, colorscale)
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ── 3D SURFACE ────────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Interactive 3D Depth Surface")
st.caption(
    "Bright regions (high intensity) → elevated peaks. "
    "Drag to rotate • Scroll to zoom • Double-click to reset."
)

with st.spinner("Rendering 3D surface…"):
    fig_3d = make_3d_surface(depth, norm, colorscale, show_contours, show_floor)

st.plotly_chart(fig_3d, use_container_width=True, height=620)

# ── DEPTH STATS EXPANDER ──────────────────────────────────────────────────────
with st.expander("📊 Depth distribution stats"):
    stat_col1, stat_col2 = st.columns([1, 2])

    with stat_col1:
        st.markdown(f"""
| Stat | Value |
|------|-------|
| Mean | `{depth.mean():.2f}` |
| Median | `{float(np.median(depth)):.2f}` |
| Std dev | `{depth.std():.2f}` |
| Min | `{depth.min():.2f}` |
| Max | `{depth.max():.2f}` |
| P25 | `{float(np.percentile(depth, 25)):.2f}` |
| P75 | `{float(np.percentile(depth, 75)):.2f}` |
| P95 | `{float(np.percentile(depth, 95)):.2f}` |
        """)

    with stat_col2:
        hist_vals, hist_bins = np.histogram(depth.ravel(), bins=60)
        fig_hist = go.Figure(
            go.Bar(
                x=hist_bins[:-1],
                y=hist_vals,
                marker=dict(
                    color=hist_vals,
                    colorscale=colorscale,
                    showscale=False,
                ),
                name="Depth distribution",
            )
        )
        fig_hist.update_layout(
            xaxis_title="Depth value",
            yaxis_title="Pixel count",
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#64748b"),
            yaxis=dict(color="#64748b"),
            height=220,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
