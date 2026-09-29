import numpy as np
from scipy.ndimage import label, center_of_mass
import math

def calculate_measurements(mask_data, spacing, affine):
    """
    Calculates physical measurements for a given binary mask.
    spacing: tuple of (dx, dy, dz) in mm
    affine: 4x4 matrix mapping voxel indices to physical space
    """
    if mask_data is None:
        return None
        
    # Voxel volume in cm^3 (spacing is typically in mm)
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0
    
    # Label connected components
    labeled_mask, num_features = label(mask_data > 0)
    
    lesions = []
    
    for i in range(1, num_features + 1):
        component_mask = (labeled_mask == i)
        voxel_count = np.sum(component_mask)
        
        # Volume
        volume_cm3 = voxel_count * voxel_volume_cm3
        
        # Equivalent spherical diameter
        # V = 4/3 * pi * r^3  =>  r = (3V / 4pi)^(1/3) => D = 2 * r
        volume_mm3 = voxel_count * voxel_volume_mm3
        diameter_mm = 2 * math.pow((3 * volume_mm3) / (4 * math.pi), 1.0/3.0)
        
        # Centroid in voxel coordinates
        centroid_voxels = center_of_mass(component_mask)
        
        # Convert centroid to physical coordinates using affine
        centroid_voxels_homog = np.array([centroid_voxels[0], centroid_voxels[1], centroid_voxels[2], 1.0])
        centroid_phys = affine.dot(centroid_voxels_homog)[:3]
        
        # Bounding box in voxel coordinates
        coords = np.argwhere(component_mask)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        
        # Physical bounding box dimensions
        dim_z = (z_max - z_min) * spacing[0]
        dim_y = (y_max - y_min) * spacing[1]
        dim_x = (x_max - x_min) * spacing[2]
        
        lesions.append({
            'id': i,
            'voxel_count': voxel_count,
            'volume_cm3': volume_cm3,
            'diameter_mm': diameter_mm,
            'centroid_voxel': centroid_voxels,
            'centroid_phys': centroid_phys,
            'bbox_dim_mm': (dim_z, dim_y, dim_x),
            'bbox_voxel': ((z_min, z_max), (y_min, y_max), (x_min, x_max)),
            'mask': component_mask
        })
        
    return lesions
