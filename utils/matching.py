"""
utils/matching.py
──────────────────
Ground truth matching and candidate labeling (Phase 2).
"""

import numpy as np
import math
from scipy.ndimage import label as cc_label, center_of_mass, distance_transform_edt
from typing import Dict, List, Tuple
from utils.config import load_config

def match_candidates_to_gt(
    candidates: List[Dict],
    gt_mask: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    config: Optional[Dict] = None
) -> Tuple[List[Dict], Dict]:
    if config is None:
        config = load_config("config.json")

    overlap_threshold = config.get("overlap_threshold", 0.5)
    detection_gt_covered_threshold = config.get("detection_gt_covered_threshold", 0.1)
    distance_margin_mm = config.get("distance_margin_mm", 10.0)
    
    gt_labeled, n_gt = cc_label(gt_mask > 0)
    
    # Compute GT lesion properties and EDT maps
    gt_lesions = {}
    gt_edts = {}
    sz, sy, sx = spacing_mm
    voxel_vol = sz * sy * sx
    
    for i in range(1, n_gt + 1):
        mask_i = gt_labeled == i
        vol = int(mask_i.sum()) * voxel_vol
        centroid_vox = center_of_mass(mask_i)
        centroid_mm = (centroid_vox[0]*sz, centroid_vox[1]*sy, centroid_vox[2]*sx)
        gt_lesions[i] = {
            "id": i,
            "volume_mm3": vol,
            "centroid_mm": centroid_mm,
            "voxel_count": int(mask_i.sum()),
            "detected": False,
            "hit_by": []
        }
        # Surface distance transform (distance from outside to boundary of lesion)
        # distance_transform_edt computes distance from 0 to nearest 1, so we invert the mask
        gt_edts[i] = distance_transform_edt(np.logical_not(mask_i), sampling=spacing_mm)
        
    updated_candidates = []
    
    for cand in candidates:
        c_up = dict(cand)
        c_id = c_up["id"]
        c_vox = c_up["centroid_voxel"]
            
        (z0, z1), (y0, y1), (x0, x1) = c_up["bbox_voxel"]
        cand_crop = c_up.get("_mask_crop")
        if cand_crop is None:
            cand_crop = np.ones((z1-z0+1, y1-y0+1, x1-x0+1), dtype=bool)
            
        gt_crop = gt_labeled[z0:z1+1, y0:y1+1, x0:x1+1]
        
        overlapping_gts = np.unique(gt_crop[cand_crop > 0])
        overlapping_gts = [g for g in overlapping_gts if g > 0]
        
        best_gt_id = None
        best_overlap_frac = 0.0
        best_gt_covered = 0.0
        best_iou = 0.0
        best_intersect_vox = 0
        
        for gt_id in overlapping_gts:
            intersect_vox = int(np.sum((gt_crop == gt_id) & (cand_crop > 0)))
            cand_vox = cand_crop.sum()
            gt_vox = gt_lesions[gt_id]["voxel_count"]
            
            frac_inside_gt = intersect_vox / cand_vox if cand_vox > 0 else 0
            frac_gt_covered = intersect_vox / gt_vox if gt_vox > 0 else 0
            iou = intersect_vox / (cand_vox + gt_vox - intersect_vox) if (cand_vox + gt_vox - intersect_vox) > 0 else 0
            
            if frac_inside_gt > best_overlap_frac:
                best_overlap_frac = frac_inside_gt
                best_gt_id = gt_id
                best_gt_covered = frac_gt_covered
                best_iou = iou
                best_intersect_vox = intersect_vox
                
        # Surface distance: minimum EDT value over all candidate voxels
        min_dist = float('inf')
        nearest_gt_id = None
        if n_gt > 0:
            cand_full_mask = np.zeros_like(gt_mask, dtype=bool)
            cand_full_mask[z0:z1+1, y0:y1+1, x0:x1+1] = (cand_crop > 0)
            
            for gt_id, edt_map in gt_edts.items():
                # Get the min distance from candidate voxels to this GT
                dist = edt_map[cand_full_mask].min()
                if dist < min_dist:
                    min_dist = dist
                    nearest_gt_id = gt_id
                
        c_up["nearest_gt_distance_mm"] = round(float(min_dist), 2) if n_gt > 0 else None
        
        centroid_in_gt = False
        try:
            centroid_in_gt = (gt_mask[c_vox] > 0)
        except IndexError:
            pass
            
        label = "AMBIGUOUS"
        if n_gt == 0:
            label = "NEGATIVE"
        else:
            if best_overlap_frac >= overlap_threshold:
                label = "POSITIVE"
            elif best_overlap_frac == 0 and min_dist > distance_margin_mm:
                label = "NEGATIVE"
            else:
                label = "AMBIGUOUS"
                
        # Detection separate from training label
        is_detected = False
        if best_gt_id is not None:
            if centroid_in_gt or best_gt_covered >= detection_gt_covered_threshold:
                is_detected = True
                gt_lesions[best_gt_id]["hit_by"].append(c_id)
                gt_lesions[best_gt_id]["detected"] = True

        c_up["label"] = label
        c_up["matched_gt_id"] = best_gt_id if is_detected else None
        c_up["intersection_voxels"] = best_intersect_vox
        c_up["fraction_inside_gt"] = round(best_overlap_frac, 4)
        c_up["fraction_gt_covered"] = round(best_gt_covered, 4)
        c_up["iou"] = round(best_iou, 4)
        
        updated_candidates.append(c_up)
        
    lesion_stats = {
        "total_gt_lesions": n_gt,
        "detected_gt_lesions": sum(1 for g in gt_lesions.values() if g["detected"]),
        "missed_gt_lesions": sum(1 for g in gt_lesions.values() if not g["detected"]),
    }
    
    return updated_candidates, lesion_stats
