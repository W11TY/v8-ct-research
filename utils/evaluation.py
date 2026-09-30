"""
utils/evaluation.py
────────────────────
Patient-level offline evaluation metrics.
"""

import math
import json
import numpy as np
from typing import Optional, List, Dict, Tuple
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

# ── Voxel-level metrics (Segmentation only, largely unused for detection) ────
def compute_voxelwise_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    pred = (pred_mask > 0).astype(np.int8)
    gt   = (gt_mask > 0).astype(np.int8)
    TP = int(((pred == 1) & (gt == 1)).sum())
    FP = int(((pred == 1) & (gt == 0)).sum())
    TN = int(((pred == 0) & (gt == 0)).sum())
    FN = int(((pred == 0) & (gt == 1)).sum())
    
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else float("nan")
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else float("nan")
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    specificity = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    fpr = FP / (FP + TN) if (FP + TN) > 0 else float("nan")

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "dice": float(dice), "iou": float(iou),
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "precision": float(precision), "fpr": float(fpr),
    }


# ── FROC using Candidate Labels ──────────────────────────────────────────────
def compute_froc(
    candidates_by_patient: Dict[str, List[Dict]],
    n_gt_lesions_total: int,
    fp_thresholds: Optional[List[float]] = None,
) -> dict:
    if fp_thresholds is None:
        fp_thresholds = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    all_candidates = []
    n_patients = len(candidates_by_patient)
    
    for pid, cands in candidates_by_patient.items():
        for c in cands:
            all_candidates.append({"patient_id": pid, **c})

    if not all_candidates:
        return {
            "primary": {"froc_points": [], "sensitivity_at_thresholds": {t: float("nan") for t in fp_thresholds}},
            "secondary_no_ambiguous": {"froc_points": [], "sensitivity_at_thresholds": {t: float("nan") for t in fp_thresholds}},
            "n_patients": n_patients,
            "n_gt_lesions_total": n_gt_lesions_total,
        }

    all_candidates.sort(
        key=lambda c: c.get("candidate_score", c.get("research_score", 0.0)) or 0.0, 
        reverse=True
    )

    def compute_curve(exclude_ambiguous: bool):
        froc_points = []
        detected_set = set()
        fp_count = 0

        for cand in all_candidates:
            pid = cand["patient_id"]
            label = cand.get("label", "AMBIGUOUS")
            
            if label == "POSITIVE":
                gt_id = cand.get("matched_gt_id")
                if gt_id is not None:
                    detected_set.add((pid, gt_id))
            elif label == "NEGATIVE":
                fp_count += 1
            elif label == "AMBIGUOUS":
                if not exclude_ambiguous:
                    fp_count += 1 # In primary curve, AMBIGUOUS count as FP
                
            sensitivity = len(detected_set) / n_gt_lesions_total if n_gt_lesions_total > 0 else 0.0
            fp_per_scan = fp_count / n_patients if n_patients > 0 else 0.0
            froc_points.append((round(fp_per_scan, 4), round(sensitivity, 4)))

        sens_at = {}
        for t in fp_thresholds:
            found = [s for fp, s in froc_points if fp <= t]
            sens_at[t] = found[-1] if found else float("nan")
            
        return {"froc_points": froc_points, "sensitivity_at_thresholds": sens_at}

    primary = compute_curve(exclude_ambiguous=False)
    secondary = compute_curve(exclude_ambiguous=True)

    return {
        "primary": primary,
        "secondary_no_ambiguous": secondary,
        "n_patients": n_patients,
        "n_gt_lesions_total": n_gt_lesions_total,
    }


# ── Patient-level Split ──────────────────────────────────────────────────────
def generate_patient_split(
    scan_ids: List[str],
    patient_ids: List[str],
    patient_labels: List[str],
    test_fraction: float = 0.20,
    n_folds: int = 5,
    seed: int = 42,
    output_json: str = "patient_split.json"
) -> dict:
    """
    Fixed stratified 20% test split, plus 5-fold GroupKFold on remaining 80%.
    Requires explicit explicit true `patient_ids` mapped to each `scan_id`.
    """
    if len(scan_ids) != len(patient_ids) or len(patient_ids) != len(patient_labels):
        raise ValueError("scan_ids, patient_ids, and patient_labels must have the same length.")
        
    # Check for ambiguity: a patient ID cannot have multiple different labels
    label_map = {}
    for pid, lbl in zip(patient_ids, patient_labels):
        if pid in label_map and label_map[pid] != lbl:
            raise ValueError(f"Ambiguity: Patient {pid} has conflicting labels ({label_map[pid]} and {lbl}).")
        label_map[pid] = lbl
        
    n_splits_test = max(2, int(1.0 / test_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    
    train_val_idx, test_idx = next(sgkf.split(scan_ids, patient_labels, groups=patient_ids))
    
    test_scans = [scan_ids[i] for i in test_idx]
    train_val_scans = [scan_ids[i] for i in train_val_idx]
    train_val_labels = [patient_labels[i] for i in train_val_idx]
    train_val_groups = [patient_ids[i] for i in train_val_idx]
    
    cv_sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    folds = []
    try:
        cv_splits = list(cv_sgkf.split(train_val_scans, train_val_labels, groups=train_val_groups))
    except ValueError:
        cv_gkf = GroupKFold(n_splits=n_folds)
        cv_splits = list(cv_gkf.split(train_val_scans, groups=train_val_groups))
        
    for train_idx, val_idx in cv_splits:
        folds.append({
            "train": [train_val_scans[i] for i in train_idx],
            "val": [train_val_scans[i] for i in val_idx]
        })
        
    split_info = {
        "test": test_scans,
        "cv_folds": folds
    }
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(split_info, f, indent=4)
            
    return split_info
