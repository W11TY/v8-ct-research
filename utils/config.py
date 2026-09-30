"""
utils/config.py
────────────────
Central configuration for V8 CT prototype.
"""

import json
import os

DEFAULT_CONFIG = {
    # Detection
    "z_score_thresh": 2.5,
    "direction": "both",
    "smooth_sigma_mm": 1.0,
    "open_radius_mm": 1.5,
    "min_vol_mm3": 100.0,
    "max_vol_mm3": 200000.0,
    "suppress_vessels": True,
    "vessel_threshold": 0.6,
    "sphericity_min_voxels": 4,
    
    # Phase matching
    "affine_tolerance_mm": 0.5,
    
    # GT Matching
    "overlap_threshold": 0.5,
    "detection_gt_covered_threshold": 0.1,
    "distance_margin_mm": 10.0,
    
    # Patient Split
    "seed": 42,
    "test_fraction": 0.20,
    
    # Random Forest
    "rf_parameters": {
        "n_estimators": 100,
        "max_depth": 5,
        "class_weight": "balanced"
    },
    
    # Evaluation
    "size_bins_mm": [1.0, 3.0, 5.0]
}

def load_config(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return DEFAULT_CONFIG.copy()
    with open(filepath, 'r') as f:
        cfg = json.load(f)
    # Merge with defaults to ensure all keys exist
    merged = DEFAULT_CONFIG.copy()
    merged.update(cfg)
    return merged

def save_config(config: dict, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
