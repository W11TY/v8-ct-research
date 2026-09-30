"""
utils/enhancement.py
─────────────────────
Multiphase enhancement scoring for 3D candidates.

Computes raw HU at candidate locations across CT phases and calculates
liver-normalized enhancement changes. Does not label output as HCC or washout.

All output terminology: "enhancement-pattern candidate", "research score".
Single-phase output always labeled "single_phase_intensity_anomaly".
"""

import numpy as np
from typing import Optional

from utils.anomaly_3d import build_local_liver_annulus, estimate_liver_background_hu


# ── Thresholds (all configurable via run config JSON) ─────────────────────────
DEFAULT_THRESHOLDS = {
    # Minimum HU enhancement above liver background to be called 'hyperarterial'
    "relative_arterial_enhancement_min_hu": 10.0,
    # Minimum relative washout (lesion loses enhancement faster than liver)
    "relative_washout_min_hu": 5.0,
}


def sample_hu_in_candidate(
    volume: np.ndarray,
    candidate_mask_3d: np.ndarray,
) -> float:
    """
    Return mean raw HU within the candidate mask in a given phase volume.
    No normalization applied.
    """
    vals = volume[candidate_mask_3d > 0]
    if len(vals) == 0:
        return float("nan")
    return float(vals.mean())


def reconstruct_candidate_mask(candidate: dict, shape: tuple) -> np.ndarray:
    """
    Reconstruct a binary candidate mask from the stored bbox and centroid.
    For efficiency, the mask is built from the stored bbox voxel coordinates.
    If the full mask is stored elsewhere, it should be passed directly instead.
    This is a bounding-box approximation — only used when the full mask isn't available.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    (z0, z1), (y0, y1), (x0, x1) = candidate["bbox_voxel"]
    mask[z0:z1+1, y0:y1+1, x0:x1+1] = 1
    return mask


def compute_multiphase_enhancement(
    candidate: dict,
    candidate_mask: np.ndarray,       # actual 3D binary mask, same shape as volumes
    phases: dict,                      # {'plain': nifti_dict, 'arterial': ..., ...}
    liver_mask: np.ndarray,
    spacing_mm: tuple,
    config: Optional[dict] = None,
) -> dict:
    """
    Measure raw HU at a candidate's location across all provided CT phases
    and compute enhancement metrics.

    Parameters
    ----------
    candidate      : candidate dict from propose_3d_candidates()
    candidate_mask : 3D binary mask for this candidate
    phases         : dict of phase label -> nifti_dict (None values are skipped)
    liver_mask     : binary liver mask
    spacing_mm     : voxel spacing in mm
    thresholds     : override default enhancement thresholds

    Returns
    -------
    Updated candidate dict with multiphase fields populated.
    Enhancement pattern is a research label, not a clinical diagnosis.
    """
    if config is None:
        from utils.config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG
    
    thresh = {
        "relative_arterial_enhancement_min_hu": config.get("relative_arterial_enhancement_min_hu", 10.0),
        "relative_washout_min_hu": config.get("relative_washout_min_hu", 5.0)
    }
    updated = dict(candidate)

    # Sample HU in each phase
    phase_hu: dict[str, Optional[float]] = {}
    liver_hu: dict[str, Optional[float]] = {}
    annulus = build_local_liver_annulus(candidate_mask, liver_mask, spacing_mm)

    for phase_name in ("plain", "arterial", "venous", "delayed"):
        nifti = phases.get(phase_name)
        if nifti is None or nifti.get("data") is None:
            phase_hu[phase_name] = None
            liver_hu[phase_name] = None
            continue

        vol = nifti["data"].astype(np.float32)
        phase_hu[phase_name] = sample_hu_in_candidate(vol, candidate_mask)
        liver_hu[phase_name] = float(vol[annulus > 0].mean()) if annulus.sum() > 0 else None

    # Store raw HU per phase
    updated["hu_plain"] = phase_hu.get("plain")
    updated["hu_arterial"] = phase_hu.get("arterial")
    updated["hu_venous"] = phase_hu.get("venous")
    updated["hu_delayed"] = phase_hu.get("delayed")

    # Count available phases
    available = [k for k, v in phase_hu.items() if v is not None]
    n_phases = len(available)

    if n_phases < 2:
        # Single-phase only — no cross-phase arithmetic
        updated["enhancement_pattern"] = "single_phase_intensity_anomaly"
        updated["delta_art_to_venous"] = None
        updated["liver_delta_art_to_venous"] = None
        updated["relative_washout"] = None
        _update_research_score(updated, liver_hu, phase_hu, thresh, single_phase=True)
        return updated

    # ── Cross-phase enhancement metrics ──────────────────────────────────────

    # Arterial enhancement (relative to plain, if available)
    art_enhancement = None
    liver_art_enhancement = None
    if phase_hu["arterial"] is not None and phase_hu["plain"] is not None:
        art_enhancement = phase_hu["arterial"] - phase_hu["plain"]
        if liver_hu["arterial"] is not None and liver_hu["plain"] is not None:
            liver_art_enhancement = liver_hu["arterial"] - liver_hu["plain"]

    # Relative arterial enhancement (lesion enhancement minus liver enhancement)
    relative_art_enh = None
    if art_enhancement is not None and liver_art_enhancement is not None:
        relative_art_enh = art_enhancement - liver_art_enhancement

    # Washout: arterial → venous change
    delta_art_to_venous = None
    liver_delta_art_to_venous = None
    relative_washout = None

    if phase_hu["arterial"] is not None and phase_hu["venous"] is not None:
        delta_art_to_venous = phase_hu["arterial"] - phase_hu["venous"]
        if liver_hu["arterial"] is not None and liver_hu["venous"] is not None:
            liver_delta_art_to_venous = liver_hu["arterial"] - liver_hu["venous"]
            # Relative washout: how much faster does the lesion lose enhancement vs liver?
            # Positive = lesion washes out relative to liver
            relative_washout = delta_art_to_venous - liver_delta_art_to_venous

    updated["delta_art_to_venous"] = (
        round(delta_art_to_venous, 2) if delta_art_to_venous is not None else None
    )
    updated["liver_delta_art_to_venous"] = (
        round(liver_delta_art_to_venous, 2) if liver_delta_art_to_venous is not None else None
    )
    updated["relative_washout"] = (
        round(relative_washout, 2) if relative_washout is not None else None
    )

    # ── Pattern label (research only, not a clinical diagnosis) ──────────────
    is_hyperarterial = (
        relative_art_enh is not None and
        relative_art_enh >= thresh["relative_arterial_enhancement_min_hu"]
    )
    has_washout = (
        relative_washout is not None and
        relative_washout >= thresh["relative_washout_min_hu"]
    )

    if is_hyperarterial and has_washout:
        pattern = "hyperarterial_washout_candidate"
    elif is_hyperarterial and not has_washout:
        pattern = "hyperarterial_no_washout_candidate"
    elif phase_hu["arterial"] is not None and phase_hu["venous"] is not None:
        pattern = "isoarterial_candidate"
    else:
        pattern = "multiphase_intensity_anomaly"

    updated["enhancement_pattern"] = pattern
    _update_research_score(updated, liver_hu, phase_hu, thresh, single_phase=False)

    return updated


def _update_research_score(
    candidate: dict,
    liver_hu: dict,
    phase_hu: dict,
    thresh: dict,
    single_phase: bool,
) -> None:
    """
    Recompute the research score incorporating multiphase enhancement features.
    Modifies candidate dict in place.

    Score components:
      - contrast_vs_liver (normalized) : 0.4 weight
      - sphericity                     : 0.2 weight
      - relative_washout (if available): 0.3 weight
      - arterial enhancement (if avail): 0.1 weight
    Vessel-flagged candidates are penalized 0.5×.

    IMPORTANT: This is a heuristic, not a validated diagnostic probability.
    """
    score = 0.0

    # Contrast component (always available)
    contrast = abs(candidate.get("contrast_vs_liver") or 0.0)
    score += 0.4 * min(1.0, contrast / 100.0)  # normalize by 100 HU

    # Sphericity
    score += 0.2 * candidate.get("sphericity", 0.0)

    if not single_phase:
        # Relative washout component
        rw = candidate.get("relative_washout")
        if rw is not None:
            score += 0.3 * min(1.0, max(0.0, rw / thresh["relative_washout_min_hu"]))

        # Arterial enhancement component
        delta_av = candidate.get("delta_art_to_venous")
        if delta_av is not None:
            score += 0.1 * min(1.0, max(0.0, delta_av / 50.0))
    else:
        # Redistribute to contrast and sphericity in single-phase case
        score = score / 0.6  # renormalize to [0,1]

    if candidate.get("possible_vessel"):
        score *= 0.5

    candidate["research_score"] = round(min(1.0, float(score)), 4)
    candidate["research_score_version"] = (
        "v0.1-classical-single-phase" if single_phase else "v0.1-classical-multiphase"
    )
