"""
utils/geometry_3d.py
─────────────────────
Spacing-aware 3D morphology and shape measurements.
All physical quantities are computed in mm or mm³ using the NIfTI spacing metadata.
"""

import numpy as np
import math
import logging
from scipy.ndimage import label, center_of_mass
from skimage.measure import marching_cubes, mesh_surface_area
from typing import Optional, Tuple, List, Dict
from utils.config import load_config

logger = logging.getLogger(__name__)

def make_spacing_aware_ball(radius_mm: float, spacing_mm: tuple) -> np.ndarray:
    """
    Create a 3D binary ellipsoid structuring element that represents a sphere
    of physical radius `radius_mm`, respecting anisotropic voxel spacing.
    """
    sz, sy, sx = spacing_mm
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
    from scipy.ndimage import binary_opening
    se = make_spacing_aware_ball(radius_mm, spacing_mm)
    return binary_opening(binary_vol, structure=se).astype(np.uint8)


def compute_sphericity_marching_cubes(mask_3d: np.ndarray, spacing_mm: tuple, volume_mm3: float, min_voxels: int) -> Tuple[float, bool]:
    """
    Compute sphericity using marching cubes to calculate physical surface area.
    Pads the mask to ensure closed surfaces.
    Falls back to a voxel-based approximation for tiny components.
    
    Returns (sphericity, fallback_flag). NaN if tiny.
    """
    if volume_mm3 <= 0 or mask_3d.sum() < min_voxels:
        logger.debug(f"Component too small for marching cubes (<{min_voxels} voxels). Returning NaN.")
        return float('nan'), True
    
    # Pad mask to close boundaries
    padded_mask = np.pad(mask_3d, pad_width=1, mode='constant', constant_values=0)
    
    try:
        verts, faces, normals, values = marching_cubes(padded_mask, level=0.5, spacing=spacing_mm)
        surface_area_mm2 = mesh_surface_area(verts, faces)
        
        if surface_area_mm2 <= 0:
            return float('nan'), True
            
        sphere_surface = math.pi**(1/3) * (6 * volume_mm3)**(2/3)
        return float(min(1.0, sphere_surface / surface_area_mm2)), False
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Marching cubes failed ({e}), fallback used.")
        return float('nan'), True


def label_3d_components(
    binary_vol: np.ndarray,
    spacing_mm: tuple,
    config: dict
) -> tuple[np.ndarray, list[dict]]:
    sz, sy, sx = spacing_mm
    voxel_vol_mm3 = sz * sy * sx
    
    min_vol_mm3 = config.get("min_vol_mm3", 100.0)
    max_vol_mm3 = config.get("max_vol_mm3", 200000.0)
    sphericity_min_voxels = config.get("sphericity_min_voxels", 4)

    labeled_vol, n = label(binary_vol > 0)
    components = []

    for i in range(1, n + 1):
        mask_i = labeled_vol == i
        voxel_count = int(mask_i.sum())
        volume_mm3 = voxel_count * voxel_vol_mm3
        diameter_mm = 2.0 * (3 * volume_mm3 / (4 * math.pi))**(1/3)

        centroid = center_of_mass(mask_i)
        
        # Bounding box
        coords = np.argwhere(mask_i)
        z_min, y_min, x_min = coords.min(axis=0).tolist()
        z_max, y_max, x_max = coords.max(axis=0).tolist()
        
        # Extract small bounding box for marching cubes
        bbox_mask = mask_i[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        sph, fallback = compute_sphericity_marching_cubes(bbox_mask, spacing_mm, volume_mm3, sphericity_min_voxels)

        if volume_mm3 < min_vol_mm3:
            size_flag = "small"
        elif volume_mm3 > max_vol_mm3:
            size_flag = "large"
        else:
            size_flag = "ok"
            
        # Compactness: volume / bounding_box_volume
        bbox_vol_mm3 = (z_max - z_min + 1)*sz * (y_max - y_min + 1)*sy * (x_max - x_min + 1)*sx
        compactness = volume_mm3 / bbox_vol_mm3 if bbox_vol_mm3 > 0 else 0.0

        components.append({
            "id": i,
            "voxel_count": voxel_count,
            "volume_mm3": round(volume_mm3, 2),
            "equivalent_diameter_mm": round(diameter_mm, 2),
            "centroid_voxel": tuple(int(round(c)) for c in centroid),
            "centroid_mm": (centroid[0]*sz, centroid[1]*sy, centroid[2]*sx),
            "bbox_voxel": (
                (z_min, z_max),
                (y_min, y_max),
                (x_min, x_max),
            ),
            "_mask_crop": bbox_mask,
            "sphericity": round(sph, 4) if not math.isnan(sph) else sph,
            "sphericity_fallback": fallback,
            "compactness": round(compactness, 4),
            "size_flag": size_flag,
        })

    return labeled_vol, components
