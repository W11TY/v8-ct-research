"""
utils/anomaly_3d.py
────────────────────
3D single-phase intensity anomaly proposal engine.
Operates entirely on raw calibrated HU values inside a liver mask.
Output terminology: "intensity anomaly candidate" — never "lesion" or "HCC".
"""

import numpy as np
import math
from scipy.ndimage import gaussian_filter
from typing import Optional

from utils.geometry_3d import (
    make_spacing_aware_ball,
    spacing_aware_open,
    label_3d_components,
)


def estimate_liver_background_hu(
    volume: np.ndarray,
    liver_mask: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
    n_iter: int = 2,
) -> tuple[float, float]:
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
    from scipy.ndimage import binary_dilation
    se = make_spacing_aware_ball(annulus_width_mm, spacing_mm)
    dilated = binary_dilation(candidate_mask > 0, structure=se)
    annulus = dilated & ~(candidate_mask > 0) & (liver_mask > 0)
    return annulus.astype(np.uint8)


def _hessian_3d(volume: np.ndarray, sigma: float) -> tuple:
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
    if sigmas is None:
        sigmas = [1.0, 2.0, 3.0]

    v = volume.astype(np.float32)
    v_min, v_max = v.min(), v.max()
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min)

    vesselness = np.zeros_like(v)

    for sigma in sigmas:
        Dzz, Dyy, Dxx, Dzy, Dzx, Dyx = _hessian_3d(v, sigma)

        Ra = np.abs(Dzy) / (np.abs(Dzz) * np.abs(Dyy) + 1e-9)
        Rb = np.abs(Dzx) / (np.abs(Dzz) * np.abs(Dxx) + 1e-9)
        S = np.sqrt(Dzz**2 + Dyy**2 + Dxx**2 + 2*(Dzy**2 + Dzx**2 + Dyx**2))
        c = c_fraction * S.max()

        v_s = (1 - np.exp(-Ra**2 / (2 * alpha**2))) * \
              np.exp(-Rb**2 / (2 * beta**2)) * \
              (1 - np.exp(-S**2 / (2 * c**2 + 1e-9)))

        v_s[Dzz + Dyy + Dxx > 0] = 0
        vesselness = np.maximum(vesselness, v_s)

    v_max = vesselness.max()
    if v_max > 0:
        vesselness /= v_max

    return vesselness.astype(np.float32)


def propose_3d_candidates(
    volume: np.ndarray,
    liver_mask: np.ndarray,
    spacing_mm: tuple,
    config: dict
) -> list[dict]:
    vol = volume.astype(np.float32)
    
    z_score_thresh = config.get("z_score_thresh", 2.5)
    direction = config.get("direction", "both")
    smooth_sigma_mm = config.get("smooth_sigma_mm", 1.0)
    open_radius_mm = config.get("open_radius_mm", 1.5)
    suppress_vessels = config.get("suppress_vessels", True)
    vessel_threshold = config.get("vessel_threshold", 0.6)

    if smooth_sigma_mm > 0:
        sigma_vox = tuple(smooth_sigma_mm / s for s in spacing_mm)
        vol_smooth = gaussian_filter(vol, sigma=sigma_vox)
    else:
        vol_smooth = vol.copy()

    liver_mu, liver_sd = estimate_liver_background_hu(vol_smooth, liver_mask)

    if direction == "hyper":
        anomaly_map = (vol_smooth > liver_mu + z_score_thresh * liver_sd).astype(np.uint8)
    elif direction == "hypo":
        anomaly_map = (vol_smooth < liver_mu - z_score_thresh * liver_sd).astype(np.uint8)
    else:
        anomaly_map = (
            (vol_smooth > liver_mu + z_score_thresh * liver_sd) |
            (vol_smooth < liver_mu - z_score_thresh * liver_sd)
        ).astype(np.uint8)

    anomaly_map &= (liver_mask > 0).astype(np.uint8)

    if open_radius_mm > 0:
        anomaly_map = spacing_aware_open(anomaly_map, open_radius_mm, spacing_mm)

    if anomaly_map.sum() == 0:
        return []

    vessel_map = None
    if suppress_vessels:
        vessel_map = compute_frangi_vesselness_3d(vol, spacing_mm)

    labeled_vol, components = label_3d_components(anomaly_map, spacing_mm, config)

    # Calculate Z-score map once
    z_map = (vol_smooth - liver_mu) / liver_sd if liver_sd > 0 else np.zeros_like(vol_smooth)
    if direction == "hypo":
        z_map = -z_map # invert so we can treat anomaly as max

    candidates = []
    for comp in components:
        cid = comp["id"]
        mask_i = (labeled_vol == cid)

        raw_hu_in_candidate = vol[mask_i]
        mean_hu_raw = float(raw_hu_in_candidate.mean())
        median_hu_raw = float(np.median(raw_hu_in_candidate))

        annulus = build_local_liver_annulus(mask_i, liver_mask, spacing_mm)
        liver_bg_hu = float(vol[annulus > 0].mean()) if annulus.sum() > 0 else liver_mu
        contrast_vs_liver = mean_hu_raw - liver_bg_hu

        # Z-score stats
        z_vals = z_map[mask_i]
        z_mean = float(z_vals.mean())
        z_max = float(z_vals.max())
        z_95 = float(np.percentile(z_vals, 95))

        possible_vessel = False
        tubularity_score = None
        if vessel_map is not None:
            tubularity_score = float(vessel_map[mask_i].mean())
            possible_vessel = tubularity_score > vessel_threshold

        candidate = {
            "id": cid,
            "phase_source": direction,
            "volume_mm3": comp["volume_mm3"],
            "equivalent_diameter_mm": comp["equivalent_diameter_mm"],
            "centroid_voxel": comp["centroid_voxel"],
            "centroid_mm": comp["centroid_mm"],
            "bbox_voxel": comp["bbox_voxel"],
            "_mask_crop": comp["_mask_crop"],
            "voxel_count": comp["voxel_count"],
            "compactness": comp["compactness"],
            "sphericity": comp["sphericity"],
            "sphericity_fallback": comp["sphericity_fallback"],
            
            "mean_hu_raw": round(mean_hu_raw, 2),
            "median_hu_raw": round(median_hu_raw, 2),
            "liver_bg_hu": round(liver_bg_hu, 2),
            "liver_mu": round(liver_mu, 2),
            "liver_sd": round(liver_sd, 2),
            "contrast_vs_liver": round(contrast_vs_liver, 2),
            
            "z_mean": round(z_mean, 2),
            "z_max": round(z_max, 2),
            "z_95": round(z_95, 2),
            
            "size_flag": comp["size_flag"],
            "possible_vessel": possible_vessel,
            "tubularity_score": round(tubularity_score, 4) if tubularity_score is not None else None,
            
            "hu_plain": None,
            "hu_arterial": None,
            "hu_venous": None,
            "hu_delayed": None,
            "delta_art_to_venous": None,
            "liver_delta_art_to_venous": None,
            "relative_washout": None,
            "enhancement_pattern": None,
            
            "research_score": None,
            "research_score_version": "v0.1-classical-single-phase",
        }
        candidates.append(candidate)

    _score_candidates_single_phase(candidates, liver_sd)
    return candidates


def _score_candidates_single_phase(candidates: list[dict], liver_sd: float) -> None:
    if not candidates:
        return

    contrasts = [abs(c["contrast_vs_liver"]) for c in candidates]
    max_c = max(contrasts) if max(contrasts) > 0 else 1.0

    for c in candidates:
        norm_contrast = abs(c["contrast_vs_liver"]) / max_c
        sph = c["sphericity"] if not math.isnan(c["sphericity"]) else 0.5
        score = 0.6 * norm_contrast + 0.4 * sph
        if c["possible_vessel"]:
            score *= 0.5
        c["research_score"] = round(float(score), 4)
        c["enhancement_pattern"] = "single_phase_intensity_anomaly"
