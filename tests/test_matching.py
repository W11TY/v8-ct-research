import numpy as np
from utils.matching import match_candidates_to_gt
from utils.evaluation import compute_froc

def test_matching_synthetic():
    gt_mask = np.zeros((10, 10, 10), dtype=np.uint8)
    gt_mask[2:5, 2:5, 2:5] = 1 # Lesion 1 (id=1, 27 voxels, centroid approx (3,3,3))
    
    spacing_mm = (1.0, 1.0, 1.0)
    config = {"overlap_threshold": 0.5, "distance_margin_mm": 5.0, "detection_gt_covered_threshold": 0.1}
    
    # Candidate 1: Perfect match (POSITIVE)
    cand1 = {
        "id": 1,
        "volume_mm3": 27.0,
        "centroid_voxel": (3, 3, 3),
        "centroid_mm": (3.0, 3.0, 3.0),
        "bbox_voxel": ((2, 4), (2, 4), (2, 4)),
        "_mask_crop": np.ones((3, 3, 3), dtype=np.uint8)
    }
    
    # Candidate 2: Zero overlap, far away (NEGATIVE)
    cand2 = {
        "id": 2,
        "volume_mm3": 1.0,
        "centroid_voxel": (9, 9, 9),
        "centroid_mm": (9.0, 9.0, 9.0),
        "bbox_voxel": ((9, 9), (9, 9), (9, 9)),
        "_mask_crop": np.ones((1, 1, 1), dtype=np.uint8)
    }
    
    # Candidate 3: Adjacent to large lesion, but 0 overlap (AMBIGUOUS or NEGATIVE?)
    # Bbox adjacent to (2:5, 2:5, 2:5) is e.g., (5:6, 3:4, 3:4). Distance should be 1 voxel = 1.0mm.
    # Since 1.0mm <= distance_margin_mm (5.0), it should be AMBIGUOUS.
    cand3 = {
        "id": 3,
        "volume_mm3": 1.0,
        "centroid_voxel": (5, 3, 3), 
        "centroid_mm": (5.0, 3.0, 3.0),
        "bbox_voxel": ((5, 5), (3, 3), (3, 3)),
        "_mask_crop": np.ones((1, 1, 1), dtype=np.uint8)
    }
    
    cands, stats = match_candidates_to_gt([cand1, cand2, cand3], gt_mask, spacing_mm, config)
    
    assert stats["total_gt_lesions"] == 1
    assert stats["detected_gt_lesions"] == 1
    
    labels = {c["id"]: c["label"] for c in cands}
    assert labels[1] == "POSITIVE"
    assert labels[2] == "NEGATIVE"
    assert labels[3] == "AMBIGUOUS"

def test_froc_toy_example():
    candidates_by_patient = {
        "p1": [
            {"id": 1, "label": "POSITIVE", "matched_gt_id": 1, "research_score": 0.9},
            {"id": 2, "label": "NEGATIVE", "research_score": 0.8},
            {"id": 3, "label": "POSITIVE", "matched_gt_id": 1, "research_score": 0.7}, # Duplicate
        ],
        "p2": [
            {"id": 4, "label": "NEGATIVE", "research_score": 0.95},
            {"id": 5, "label": "AMBIGUOUS", "research_score": 0.85},
        ]
    }
    
    res = compute_froc(candidates_by_patient, n_gt_lesions_total=2, fp_thresholds=[0.5, 1.0])
    
    # Primary (AMBIGUOUS counts as FP)
    # Order: 4(N), 1(P), 5(A), 2(N), 3(P,dup)
    # 4(N): FP=1 -> FP/scan=0.5
    # 1(P): FP=1 -> sens=0.5
    # 5(A): FP=2 -> FP/scan=1.0
    # 2(N): FP=3 -> FP/scan=1.5
    pts_primary = res["primary"]["froc_points"]
    assert pts_primary[2] == (1.0, 0.5) # index 2 is cand 5
    
    # Secondary (AMBIGUOUS excluded from FP)
    # 5(A) adds nothing
    # 2(N): FP=2 -> FP/scan=1.0
    pts_secondary = res["secondary_no_ambiguous"]["froc_points"]
    assert pts_secondary[3] == (1.0, 0.5) # index 3 is cand 2
