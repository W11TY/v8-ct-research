"""
utils/nifti_checks.py
─────────────────────
Geometry validation for multi-phase NIfTI volumes.
All checks return warning strings; empty list = all clear.
"""

import numpy as np
import nibabel as nib
from typing import Optional


# ── Tolerances ────────────────────────────────────────────────────────────────
AFFINE_TOLERANCE_MM = 0.5   # max allowed affine deviation (mm) between phases
SPACING_MAX_MM = 15.0       # sanity ceiling for any voxel dimension
LIVER_HU_MIN = -50.0        # plausible liver HU floor (plain phase, fatty liver)
LIVER_HU_MAX = 250.0        # plausible liver HU ceiling (post-contrast peak)


def check_geometry_consistency(
    nifti_dicts: list[dict],
    labels: list[str],
) -> list[str]:
    """
    Validate geometric compatibility of multiple NIfTI volumes for cross-phase analysis.

    Parameters
    ----------
    nifti_dicts : list of dicts, each from load_nifti()
    labels      : human-readable name for each volume (e.g. ['arterial', 'venous'])

    Returns
    -------
    List of warning strings. Empty list = all checks passed.
    Raises ValueError immediately for hard failures (wrong number of args, etc.)
    """
    if len(nifti_dicts) != len(labels):
        raise ValueError("nifti_dicts and labels must be the same length.")

    warnings: list[str] = []
    ref = nifti_dicts[0]
    ref_label = labels[0]

    for nifti, label in zip(nifti_dicts[1:], labels[1:]):
        # 1. Shape identity
        if nifti["shape"] != ref["shape"]:
            warnings.append(
                f"Shape mismatch: '{ref_label}' is {ref['shape']} but "
                f"'{label}' is {nifti['shape']}. Cross-phase scoring will be disabled."
            )

        # 2. Affine proximity
        max_affine_diff = float(np.max(np.abs(nifti["affine"] - ref["affine"])))
        if max_affine_diff > AFFINE_TOLERANCE_MM:
            warnings.append(
                f"Affine mismatch: '{label}' vs '{ref_label}' differs by "
                f"{max_affine_diff:.3f}mm (tolerance {AFFINE_TOLERANCE_MM}mm). "
                f"Volumes may not be registered. Cross-phase scoring will be disabled."
            )

        # 3. Orientation
        if nifti["orientation"] != ref["orientation"]:
            warnings.append(
                f"Orientation mismatch: '{ref_label}' is "
                f"{''.join(ref['orientation'])} but '{label}' is "
                f"{''.join(nifti['orientation'])}."
            )

    # 4. Per-volume spacing sanity
    for nifti, label in zip(nifti_dicts, labels):
        sp = nifti["spacing"]
        if any(s <= 0 or s > SPACING_MAX_MM for s in sp):
            warnings.append(
                f"Suspicious spacing in '{label}': {sp}. "
                f"Expected all values in (0, {SPACING_MAX_MM}]mm."
            )
        if max(sp) / min(sp) > 10:
            warnings.append(
                f"Highly anisotropic spacing in '{label}': {sp}. "
                f"Morphological operations will use spacing-aware kernels."
            )

    return warnings


def check_mask_compatibility(
    volume: dict,
    mask: dict,
    volume_label: str = "CT volume",
    mask_label: str = "mask",
) -> list[str]:
    """
    Verify that a mask is geometrically compatible with its CT volume.

    Returns list of warning strings; empty = compatible.
    """
    warnings: list[str] = []

    if volume["shape"] != mask["shape"]:
        warnings.append(
            f"Shape mismatch: {volume_label} is {volume['shape']} but "
            f"{mask_label} is {mask['shape']}. Mask cannot be applied."
        )

    max_affine_diff = float(np.max(np.abs(volume["affine"] - mask["affine"])))
    if max_affine_diff > AFFINE_TOLERANCE_MM:
        warnings.append(
            f"Affine mismatch between {volume_label} and {mask_label}: "
            f"{max_affine_diff:.3f}mm deviation."
        )

    if mask["data"].dtype not in (np.uint8, np.int8, np.int16, np.int32,
                                   np.uint16, np.uint32, np.float32, np.float64):
        warnings.append(f"Unexpected mask dtype: {mask['data'].dtype}.")

    unique_vals = np.unique(mask["data"])
    if len(unique_vals) > 10:
        warnings.append(
            f"{mask_label} has {len(unique_vals)} unique values — expected binary or few-label mask."
        )

    return warnings


def check_hu_plausibility(nifti: dict, label: str, liver_mask: Optional[np.ndarray] = None) -> list[str]:
    """
    Sanity-check that HU values in a CT volume are in a plausible range.
    If liver_mask is provided, checks HU within liver only.
    """
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
    max_hu = float(np.max(values))

    if mean_hu < LIVER_HU_MIN or mean_hu > LIVER_HU_MAX:
        warnings.append(
            f"'{label}' mean HU in {region} is {mean_hu:.1f}, outside expected "
            f"liver range [{LIVER_HU_MIN}, {LIVER_HU_MAX}] HU. "
            f"Check windowing or phase assignment."
        )

    if min_hu > 0:
        warnings.append(
            f"'{label}' minimum HU is {min_hu:.1f}. A CT volume should contain "
            f"negative HU values (air, fat). May not be a raw HU volume."
        )

    return warnings


def phases_are_registered(phase_dicts: dict[str, dict]) -> bool:
    """
    Return True only if all provided phases have matching shapes and affines
    within tolerance. Used to gate cross-phase enhancement scoring.
    """
    items = [(label, d) for label, d in phase_dicts.items() if d is not None]
    if len(items) < 2:
        return False

    ref_label, ref = items[0]
    for label, d in items[1:]:
        if d["shape"] != ref["shape"]:
            return False
        if float(np.max(np.abs(d["affine"] - ref["affine"]))) > AFFINE_TOLERANCE_MM:
            return False
    return True
