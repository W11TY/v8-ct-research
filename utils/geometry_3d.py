"""
utils/geometry_3d.py
─────────────────────
Spacing-aware 3D morphology and shape measurements.
All physical quantities are computed in mm or mm³ using the NIfTI spacing metadata.
"""

import numpy as np
import math
from scipy.ndimage import label, center_of_mass, binary_opening, binary_erosion
from typing import Optional


def make_spacing_aware_ball(radius_mm: float, spacing_mm: tuple) -> np.ndarray:
    """
    Create a 3D binary ellipsoid structuring element that represents a sphere
    of physical radius `radius_mm`, respecting anisotropic voxel spacing.

    Parameters
    ----------
    radius_mm  : physical radius in mm
    spacing_mm : (sz, sy, sx) voxel spacing in mm

    Returns
    -------
    3D boolean array — True inside the ellipsoid
    """
    sz, sy, sx = spacing_mm
    # Number of voxels along each axis for the given physical radius
    rz = max(1, int(round(radius_mm / sz)))
    ry = max(1, int(round(radius_mm / sy)))
    rx = max(1, int(round(radius_mm / sx)))

    z = np.arange(-rz, rz + 1) * sz
    y = np.arange(-ry, ry + 1) * sy
    x = np.arange(-rx, rx + 1) * sx

    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    ball = (Z**2 + Y**2 + X**2) <= radius_mm**2
    return ball


def spacing_aware_open(binary_vol: np.ndarray, radius_mm: float, spacing_mm: tuple) -> np.ndarray:
    """
    Binary morphological opening with a physically-sized ball structuring element.
    Removes connected components smaller than radius_mm; smooths boundaries.
    """
    se = make_spacing_aware_ball(radius_mm, spacing_mm)
    return binary_opening(binary_vol, structure=se).astype(np.uint8)


def compute_surface_area_voxels(mask_3d: np.ndarray) -> float:
    """
    Estimate voxel-count surface area of a binary mask using erosion difference.
    Used internally for sphericity; not a true physical surface area.
    """
    eroded = binary_erosion(mask_3d)
    surface = mask_3d.astype(bool) & ~eroded
    return float(surface.sum())


def compute_sphericity(volume_mm3: float, surface_area_vox: float, spacing_mm: tuple) -> float:
    """
    3D sphericity: ratio of surface area of an equivalent sphere to actual surface area.
    sphericity = π^(1/3) * (6V)^(2/3) / A

    Parameters
    ----------
    volume_mm3       : physical volume in mm³
    surface_area_vox : voxel-count surface area (from compute_surface_area_voxels)
    spacing_mm       : (sz, sy, sx) for converting voxel surface area to mm²

    Returns float in (0, 1] — 1.0 = perfect sphere
    """
    if surface_area_vox <= 0 or volume_mm3 <= 0:
        return 0.0

    # Convert voxel surface count to mm² (approximate: each surface voxel contributes
    # an area proportional to the smallest face of the voxel)
    sz, sy, sx = spacing_mm
    min_face = min(sz * sy, sy * sx, sz * sx)
    surface_area_mm2 = surface_area_vox * min_face

    try:
        sphere_surface = math.pi**(1/3) * (6 * volume_mm3)**(2/3)
        return float(min(1.0, sphere_surface / surface_area_mm2))
    except (ZeroDivisionError, ValueError):
        return 0.0


def label_3d_components(
    binary_vol: np.ndarray,
    spacing_mm: tuple,
    min_vol_mm3: float = 0.0,
    max_vol_mm3: float = float("inf"),
) -> tuple[np.ndarray, list[dict]]:
    """
    Label 3D connected components and compute spacing-aware physical measurements.

    Parameters
    ----------
    binary_vol  : 3D binary array
    spacing_mm  : (sz, sy, sx) voxel spacing in mm
    min_vol_mm3 : components below this volume are flagged, not removed
    max_vol_mm3 : components above this volume are flagged, not removed

    Returns
    -------
    labeled_vol : 3D int array — each component gets a unique integer label
    components  : list of component dicts (see below)

    Component dict keys:
        id, voxel_count, volume_mm3, equivalent_diameter_mm,
        centroid_voxel, bbox_voxel (z_range, y_range, x_range),
        sphericity, size_flag ('ok' | 'small' | 'large'),
        surface_area_vox
    """
    sz, sy, sx = spacing_mm
    voxel_vol_mm3 = sz * sy * sx

    labeled_vol, n = label(binary_vol > 0)
    components = []

    for i in range(1, n + 1):
        mask_i = labeled_vol == i
        voxel_count = int(mask_i.sum())
        volume_mm3 = voxel_count * voxel_vol_mm3
        diameter_mm = 2.0 * (3 * volume_mm3 / (4 * math.pi))**(1/3)

        centroid = center_of_mass(mask_i)
        coords = np.argwhere(mask_i)
        z_min, y_min, x_min = coords.min(axis=0).tolist()
        z_max, y_max, x_max = coords.max(axis=0).tolist()

        surface_vox = compute_surface_area_voxels(mask_i)
        sph = compute_sphericity(volume_mm3, surface_vox, spacing_mm)

        if volume_mm3 < min_vol_mm3:
            size_flag = "small"
        elif volume_mm3 > max_vol_mm3:
            size_flag = "large"
        else:
            size_flag = "ok"

        components.append({
            "id": i,
            "voxel_count": voxel_count,
            "volume_mm3": round(volume_mm3, 2),
            "equivalent_diameter_mm": round(diameter_mm, 2),
            "centroid_voxel": tuple(int(round(c)) for c in centroid),
            "bbox_voxel": (
                (z_min, z_max),
                (y_min, y_max),
                (x_min, x_max),
            ),
            "sphericity": round(sph, 4),
            "size_flag": size_flag,
            "surface_area_vox": int(surface_vox),
        })

    return labeled_vol, components
