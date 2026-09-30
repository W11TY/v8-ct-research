"""
utils/nifti_checks.py
─────────────────────
Geometry validation for multi-phase NIfTI volumes.
"""

import numpy as np
import nibabel as nib
from typing import Optional, List, Dict
from utils.config import load_config

# ── Tolerances ────────────────────────────────────────────────────────────────
SPACING_MAX_MM = 15.0       # sanity ceiling for any voxel dimension
LIVER_HU_MIN = -50.0        # plausible liver HU floor (plain phase, fatty liver)
LIVER_HU_MAX = 250.0        # plausible liver HU ceiling (post-contrast peak)


class PhaseMisalignmentError(ValueError):
    """Raised when phases are not registered properly."""
    pass


def check_geometry_consistency(
    nifti_dicts: List[Dict],
    labels: List[str],
    config: Optional[Dict] = None,
    raise_on_misalignment: bool = True
) -> List[str]:
    """
    Validate geometric compatibility of multiple NIfTI volumes.
    """
    if len(nifti_dicts) != len(labels):
        raise ValueError("nifti_dicts and labels must be the same length.")

    if config is None:
        config = load_config("config.json")
        
    affine_tolerance = config.get("affine_tolerance_mm", 0.5)

    warnings: List[str] = []
    ref = nifti_dicts[0]
    ref_label = labels[0]

    for nifti, label in zip(nifti_dicts[1:], labels[1:]):
        # 1. Shape identity
        if nifti["shape"] != ref["shape"]:
            msg = f"Shape mismatch: '{ref_label}' is {ref['shape']} but '{label}' is {nifti['shape']}."
            if raise_on_misalignment:
                raise PhaseMisalignmentError(msg)
            warnings.append(msg)

        # 2. Affine proximity
        max_affine_diff = float(np.max(np.abs(nifti["affine"] - ref["affine"])))
        if max_affine_diff > affine_tolerance:
            msg = f"Affine mismatch: '{label}' vs '{ref_label}' differs by {max_affine_diff:.3f}mm (tol {affine_tolerance}mm)."
            if raise_on_misalignment:
                raise PhaseMisalignmentError(msg)
            warnings.append(msg)

        # 3. Orientation
        if nifti["orientation"] != ref["orientation"]:
            warnings.append(
                f"Orientation mismatch: '{ref_label}' is {''.join(ref['orientation'])} but '{label}' is {''.join(nifti['orientation'])}."
            )

    # 4. Per-volume spacing sanity
    for nifti, label in zip(nifti_dicts, labels):
        sp = nifti["spacing"]
        if any(s <= 0 or s > SPACING_MAX_MM for s in sp):
            warnings.append(f"Suspicious spacing in '{label}': {sp}.")
        if max(sp) / min(sp) > 10:
            warnings.append(f"Highly anisotropic spacing in '{label}': {sp}.")

    return warnings


def check_mask_compatibility(
    volume: dict,
    mask: dict,
    volume_label: str = "CT volume",
    mask_label: str = "mask",
    config: Optional[Dict] = None
) -> list[str]:
    warnings: list[str] = []
    if config is None:
        config = load_config("config.json")
    affine_tolerance = config.get("affine_tolerance_mm", 0.5)

    if volume["shape"] != mask["shape"]:
        warnings.append(
            f"Shape mismatch: {volume_label} is {volume['shape']} but {mask_label} is {mask['shape']}."
        )

    max_affine_diff = float(np.max(np.abs(volume["affine"] - mask["affine"])))
    if max_affine_diff > affine_tolerance:
        warnings.append(
            f"Affine mismatch between {volume_label} and {mask_label}: {max_affine_diff:.3f}mm deviation."
        )

    return warnings


def check_hu_plausibility(nifti: dict, label: str, liver_mask: Optional[np.ndarray] = None) -> list[str]:
    warnings: list[str] = []
    data = nifti["data"]

    if liver_mask is not None and np.sum(liver_mask) > 0:
        values = data[liver_mask > 0]
        region = "liver ROI"
    else:
        values = data.ravel()
        region = "full volume"

    mean_hu = float(np.mean(values))
    min_hu = float(np.min(values))

    if mean_hu < LIVER_HU_MIN or mean_hu > LIVER_HU_MAX:
        warnings.append(
            f"'{label}' mean HU in {region} is {mean_hu:.1f}, outside expected [{LIVER_HU_MIN}, {LIVER_HU_MAX}] HU."
        )

    if min_hu > 0:
        warnings.append(
            f"'{label}' minimum HU is {min_hu:.1f}. May not be a raw HU volume."
        )

    return warnings


def phases_are_registered(phase_dicts: dict[str, dict], config: Optional[Dict] = None) -> bool:
    if config is None:
        config = load_config("config.json")
    affine_tolerance = config.get("affine_tolerance_mm", 0.5)
    
    items = [(label, d) for label, d in phase_dicts.items() if d is not None]
    if len(items) < 2:
        return False

    ref_label, ref = items[0]
    for label, d in items[1:]:
        if d["shape"] != ref["shape"]:
            return False
        if float(np.max(np.abs(d["affine"] - ref["affine"]))) > affine_tolerance:
            return False
    return True
