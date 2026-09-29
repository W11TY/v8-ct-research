"""
utils/anomaly_3d.py
────────────────────
3D single-phase intensity anomaly proposal engine.
Operates entirely on raw calibrated HU values inside a liver mask.
Output terminology: "intensity anomaly candidate" — never "lesion" or "HCC".
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label as cc_label
from typing import Optional

from utils.geometry_3d import (
    make_spacing_aware_ball,
    spacing_aware_open,
    label_3d_components,
    compute_surface_area_voxels,
)


# ── Robust liver background estimation ───────────────────────────────────────

def estimate_liver_background_hu(
    volume: np.ndarray,
    liver_mask: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
    n_iter: int = 2,
) -> tuple[float, float]:
    """
    Estimate mean and std of liver parenchyma HU, excluding outlier voxels
    (vessels, ducts, candidate regions).

    Uses iterative sigma-clipping: fit mean/std, clip to [μ - 2σ, μ + 2σ], repeat.

    Parameters
    ----------
    volume       : raw HU 3D array
    liver_mask   : binary mask — 1 = liver
    exclude_mask : optional additional exclusion (e.g., current candidate)
    n_iter       : number of sigma-clipping iterations

    Returns
    -------
    (mean_hu, std_hu) — raw calibrated HU, no normalization applied
    """
    roi = liver_mask > 0
    if exclude_mask is not None:
        roi = roi & (exclude_mask == 0)

    if roi.sum() == 0:
        return 0.0, 1.0  # degenerate fallback

    vals = volume[roi].astype(np.float64)

    for _ in range(n_iter):
        mu, sd = vals.mean(), vals.std()
        sd = max(sd, 1e-6)
        vals = vals[(vals >= mu - 2 * sd) & (vals <= mu + 2 * sd)]
        if len(vals) == 0:
            break

    return float(mu), float(sd)


def build_local_liver_annulus(
    candidate_mask: np.ndarray,
    liver_mask: np.ndarray,
    spacing_mm: tuple,
    annulus_width_mm: float = 5.0,
) -> np.ndarray:
    """
    Build a ring of liver voxels immediately surrounding a candidate mask
    (5mm physical annulus), confined to the liver mask.
    Used for local background HU estimation.
    """
    from scipy.ndimage import binary_dilation
    se = make_spacing_aware_ball(annulus_width_mm, spacing_mm)
    dilated = binary_dilation(candidate_mask > 0, structure=se)
    annulus = dilated & ~(candidate_mask > 0) & (liver_mask > 0)
    return annulus.astype(np.uint8)


# ── Frangi 3D vesselness (heuristic) ─────────────────────────────────────────

def _hessian_3d(volume: np.ndarray, sigma: float) -> tuple:
    """Compute 3x3 Hessian matrix components via Gaussian second derivatives."""
    from scipy.ndimage import gaussian_filter

    sm = gaussian_filter(volume.astype(np.float32), sigma=sigma)
    Dzz = gaussian_filter(sm, sigma=sigma, order=[2, 0, 0])
    Dyy = gaussian_filter(sm, sigma=sigma, order=[0, 2, 0])
    Dxx = gaussian_filter(sm, sigma=sigma, order=[0, 0, 2])
    Dzy = gaussian_filter(sm, sigma=sigma, order=[1, 1, 0])
    Dzx = gaussian_filter(sm, sigma=sigma, order=[1, 0, 1])
    Dyx = gaussian_filter(sm, sigma=sigma, order=[0, 1, 1])
    return Dzz, Dyy, Dxx, Dzy, Dzx, Dyx


def compute_frangi_vesselness_3d(
    volume: np.ndarray,
    spacing_mm: tuple,
    sigmas: list = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    c_fraction: float = 0.5,
) -> np.ndarray:
    """
    Compute 3D Frangi vesselness filter over a range of scales.
    Returns a [0,1] tubularity heuristic score per voxel.

    IMPORTANT: This is a heuristic. High score suggests a tubular structure
    but does NOT confirm a blood vessel. Used only for flagging, not exclusion.

    Parameters
    ----------
    volume     : raw HU array (will be normalized internally for eigenvalue analysis)
    spacing_mm : voxel spacing for scale normalization
    sigmas     : list of Gaussian scales in mm (default: [1, 2, 3])
    alpha, beta, c_fraction : Frangi sensitivity parameters
    """
    if sigmas is None:
        sigmas = [1.0, 2.0, 3.0]

    # Normalize volume to [0,1] for eigenvalue analysis only (not used elsewhere)
    v = volume.astype(np.float32)
    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min)

    vesselness = np.zeros_like(v)

    for sigma in sigmas:
        Dzz, Dyy, Dxx, Dzy, Dzx, Dyx = _hessian_3d(v, sigma)

        # Approximate eigenvalues via Frobenius norm (fast approximation)
        # Full eigendecomposition is too slow for large 3D volumes
        Ra = np.abs(Dzy) / (np.abs(Dzz) * np.abs(Dyy) + 1e-9)
        Rb = np.abs(Dzx) / (np.abs(Dzz) * np.abs(Dxx) + 1e-9)
        S = np.sqrt(Dzz**2 + Dyy**2 + Dxx**2 + 2*(Dzy**2 + Dzx**2 + Dyx**2))
        c = c_fraction * S.max()

        v_s = (1 - np.exp(-Ra**2 / (2 * alpha**2))) * \
              np.exp(-Rb**2 / (2 * beta**2)) * \
              (1 - np.exp(-S**2 / (2 * c**2 + 1e-9)))

        # Bright-on-dark vessels only
        v_s[Dzz + Dyy + Dxx > 0] = 0
        vesselness = np.maximum(vesselness, v_s)

    # Clip to [0, 1]
    v_max = vesselness.max()
    if v_max > 0:
        vesselness /= v_max

    return vesselness.astype(np.float32)


# ── Main 3D candidate proposal ────────────────────────────────────────────────

def propose_3d_candidates(
    volume: np.ndarray,
    liver_mask: np.ndarray,
    spacing_mm: tuple,
    z_score_thresh: float = 2.5,
    direction: str = "hyper",          # "hyper" | "hypo" | "both"
    smooth_sigma_mm: float = 1.0,      # physical smoothing before anomaly map
    open_radius_mm: float = 1.5,       # morphological opening to remove noise
    min_vol_mm3: float = 100.0,        # flagged as 'small' below this
    max_vol_mm3: float = 200_000.0,    # flagged as 'large' above this
    suppress_vessels: bool = True,
    vessel_threshold: float = 0.6,     # Frangi score above which = possible_vessel
) -> list[dict]:
    """
    Propose 3D intensity anomaly candidates within the liver mask.

    All measurements use raw calibrated HU. CLAHE is never applied here.
    All size thresholds produce flags, not exclusions — all candidates are returned.

    Parameters
    ----------
    volume          : raw HU float32 array, shape (Z, Y, X) or (X, Y, Z)
    liver_mask      : binary array, same shape as volume
    spacing_mm      : (sz, sy, sx) voxel spacing in mm
    z_score_thresh  : SD threshold for anomaly definition relative to liver bg
    direction       : detect hyperintense, hypointense, or both anomalies
    smooth_sigma_mm : Gaussian pre-smoothing in physical mm
    open_radius_mm  : morphological opening radius in mm (removes speckle)
    min_vol_mm3     : volume below which candidate is flagged 'small'
    max_vol_mm3     : volume above which candidate is flagged 'large'
    suppress_vessels: compute Frangi vesselness to flag tubular candidates
    vessel_threshold: Frangi score above which candidate is flagged 'possible_vessel'

    Returns
    -------
    list of candidate dicts (see plan data contract)
    """
    vol = volume.astype(np.float32)

    # 1. Gentle Gaussian smoothing — physical sigma converted to voxels per axis
    if smooth_sigma_mm > 0:
        sigma_vox = tuple(smooth_sigma_mm / s for s in spacing_mm)
        vol_smooth = gaussian_filter(vol, sigma=sigma_vox)
    else:
        vol_smooth = vol.copy()

    # 2. Estimate liver background (robust, sigma-clipped)
    liver_mu, liver_sd = estimate_liver_background_hu(vol_smooth, liver_mask)

    # 3. Build anomaly map based on direction
    if direction == "hyper":
        anomaly_map = (vol_smooth > liver_mu + z_score_thresh * liver_sd).astype(np.uint8)
    elif direction == "hypo":
        anomaly_map = (vol_smooth < liver_mu - z_score_thresh * liver_sd).astype(np.uint8)
    else:  # "both"
        anomaly_map = (
            (vol_smooth > liver_mu + z_score_thresh * liver_sd) |
            (vol_smooth < liver_mu - z_score_thresh * liver_sd)
        ).astype(np.uint8)

    # 4. Restrict to liver mask
    anomaly_map &= (liver_mask > 0).astype(np.uint8)

    # 5. Morphological opening (removes single-voxel noise, physically sized)
    if open_radius_mm > 0:
        anomaly_map = spacing_aware_open(anomaly_map, open_radius_mm, spacing_mm)

    if anomaly_map.sum() == 0:
        return []

    # 6. Optional: precompute vesselness
    vessel_map = None
    if suppress_vessels:
        vessel_map = compute_frangi_vesselness_3d(vol, spacing_mm)

    # 7. 3D connected components + measurements
    labeled_vol, components = label_3d_components(
        anomaly_map, spacing_mm,
        min_vol_mm3=min_vol_mm3,
        max_vol_mm3=max_vol_mm3,
    )

    # 8. Build final candidate list
    candidates = []
    for comp in components:
        cid = comp["id"]
        mask_i = (labeled_vol == cid)

        # Raw HU statistics (on unsmoothed volume)
        raw_hu_in_candidate = vol[mask_i]
        mean_hu_raw = float(raw_hu_in_candidate.mean())

        # Local liver background annulus
        annulus = build_local_liver_annulus(mask_i, liver_mask, spacing_mm)
        liver_bg_hu = float(vol[annulus > 0].mean()) if annulus.sum() > 0 else liver_mu
        contrast_vs_liver = mean_hu_raw - liver_bg_hu

        # Tubularity flag
        possible_vessel = False
        tubularity_score = None
        if vessel_map is not None:
            tubularity_score = float(vessel_map[mask_i].mean())
            possible_vessel = tubularity_score > vessel_threshold

        candidate = {
            # Identity
            "id": cid,
            "phase_source": direction,
            # Physical measurements
            "volume_mm3": comp["volume_mm3"],
            "equivalent_diameter_mm": comp["equivalent_diameter_mm"],
            "centroid_voxel": comp["centroid_voxel"],
            "bbox_voxel": comp["bbox_voxel"],
            # HU features — raw, no normalization
            "mean_hu_raw": round(mean_hu_raw, 2),
            "liver_bg_hu": round(liver_bg_hu, 2),
            "liver_mu": round(liver_mu, 2),
            "liver_sd": round(liver_sd, 2),
            "contrast_vs_liver": round(contrast_vs_liver, 2),
            # Shape
            "sphericity": comp["sphericity"],
            "size_flag": comp["size_flag"],
            # Vessel heuristic (flagging only, not exclusion)
            "possible_vessel": possible_vessel,
            "tubularity_score": round(tubularity_score, 4) if tubularity_score is not None else None,
            # Multiphase fields — populated later by enhancement.py
            "hu_plain": None,
            "hu_arterial": None,
            "hu_venous": None,
            "hu_delayed": None,
            "delta_art_to_venous": None,
            "liver_delta_art_to_venous": None,
            "relative_washout": None,
            "enhancement_pattern": None,
            # Research score — not a diagnostic probability
            "research_score": None,
            "research_score_version": "v0.1-classical-single-phase",
        }
        candidates.append(candidate)

    # 9. Compute simple research score (contrast + sphericity, no phase info yet)
    _score_candidates_single_phase(candidates, liver_sd)

    return candidates


def _score_candidates_single_phase(candidates: list[dict], liver_sd: float) -> None:
    """
    Assign a simple [0,1] heuristic research score based on single-phase features.
    This is NOT a diagnostic probability. Updated in place.

    Score = 0.6 * normalized_contrast + 0.4 * sphericity
    Vessel-flagged candidates have score penalized by 0.5×.
    All outputs labeled "single_phase_intensity_anomaly".
    """
    if not candidates:
        return

    contrasts = [abs(c["contrast_vs_liver"]) for c in candidates]
    max_c = max(contrasts) if max(contrasts) > 0 else 1.0

    for c in candidates:
        norm_contrast = abs(c["contrast_vs_liver"]) / max_c
        sph = c["sphericity"]
        score = 0.6 * norm_contrast + 0.4 * sph
        if c["possible_vessel"]:
            score *= 0.5
        c["research_score"] = round(float(score), 4)
        c["enhancement_pattern"] = "single_phase_intensity_anomaly"
