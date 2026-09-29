"""
utils/evaluation.py
────────────────────
Patient-level offline evaluation metrics.
Only used when ground-truth segmentation masks are available.
Clearly distinguishes detection metrics from segmentation metrics.
"""

import math
import numpy as np
from scipy.ndimage import label as cc_label
from typing import Optional


# ── Voxel-level metrics ───────────────────────────────────────────────────────

def compute_voxelwise_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> dict:
    """
    Voxel-level segmentation metrics.
    Empty prediction and empty ground-truth are handled explicitly — not silently set to 0.

    Parameters
    ----------
    pred_mask : binary 3D prediction
    gt_mask   : binary 3D ground truth

    Returns
    -------
    dict with keys:
        dice, iou, sensitivity, specificity, precision, recall, fpr, tnr
        All float. NaN when undefined (e.g. no positive GT voxels).
    """
    pred = (pred_mask > 0).astype(np.int8)
    gt   = (gt_mask > 0).astype(np.int8)

    TP = int(((pred == 1) & (gt == 1)).sum())
    FP = int(((pred == 1) & (gt == 0)).sum())
    TN = int(((pred == 0) & (gt == 0)).sum())
    FN = int(((pred == 0) & (gt == 1)).sum())

    # Dice
    if (2 * TP + FP + FN) == 0:
        dice = float("nan")   # both empty — undefined
    else:
        dice = 2 * TP / (2 * TP + FP + FN)

    # IoU (Jaccard)
    if (TP + FP + FN) == 0:
        iou = float("nan")
    else:
        iou = TP / (TP + FP + FN)

    # Sensitivity (recall) — undefined if no GT positives
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    recall = sensitivity

    # Specificity — undefined if no GT negatives
    specificity = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    tnr = specificity

    # Precision — undefined if no predicted positives
    precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")

    # False positive rate
    fpr = FP / (FP + TN) if (FP + TN) > 0 else float("nan")

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "recall": float(recall),
        "specificity": float(specificity),
        "tnr": float(tnr),
        "precision": float(precision),
        "fpr": float(fpr),
    }


# ── Lesion-level metrics ──────────────────────────────────────────────────────

def compute_lesion_level_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.1,
) -> dict:
    """
    Lesion-level detection metrics (not segmentation quality).

    For each GT connected component, check if any predicted component overlaps
    it by >= iou_threshold. If yes, the GT lesion is "detected".

    False positives at lesion level = predicted components with no GT overlap.

    Parameters
    ----------
    pred_mask     : binary 3D prediction
    gt_mask       : binary 3D ground truth
    iou_threshold : minimum IoU to count a GT lesion as detected

    Returns
    -------
    dict with:
        lesion_sensitivity, fp_per_scan, total_gt_lesions,
        detected_gt_lesions, missed_gt_lesions, fp_count
    """
    pred_labeled, n_pred = cc_label(pred_mask > 0)
    gt_labeled,   n_gt   = cc_label(gt_mask > 0)

    if n_gt == 0:
        return {
            "total_gt_lesions": 0,
            "detected_gt_lesions": 0,
            "missed_gt_lesions": 0,
            "fp_count": n_pred,
            "fp_per_scan": float(n_pred),
            "lesion_sensitivity": float("nan"),  # undefined — no GT lesions
        }

    detected = 0
    for gt_id in range(1, n_gt + 1):
        gt_comp = (gt_labeled == gt_id)
        for pred_id in range(1, n_pred + 1):
            pred_comp = (pred_labeled == pred_id)
            intersection = int((gt_comp & pred_comp).sum())
            union = int((gt_comp | pred_comp).sum())
            if union > 0 and intersection / union >= iou_threshold:
                detected += 1
                break

    # FP lesions = predicted components that don't overlap any GT lesion
    fp_count = 0
    for pred_id in range(1, n_pred + 1):
        pred_comp = (pred_labeled == pred_id)
        is_tp = False
        for gt_id in range(1, n_gt + 1):
            gt_comp = (gt_labeled == gt_id)
            intersection = int((gt_comp & pred_comp).sum())
            union = int((gt_comp | pred_comp).sum())
            if union > 0 and intersection / union >= iou_threshold:
                is_tp = True
                break
        if not is_tp:
            fp_count += 1

    return {
        "total_gt_lesions": n_gt,
        "detected_gt_lesions": detected,
        "missed_gt_lesions": n_gt - detected,
        "fp_count": fp_count,
        "fp_per_scan": float(fp_count),
        "lesion_sensitivity": detected / n_gt if n_gt > 0 else float("nan"),
    }


# ── FROC ─────────────────────────────────────────────────────────────────────

def compute_froc(
    candidates_by_patient: dict,         # {patient_id: [candidate_dict, ...]}
    gt_masks_by_patient: dict,           # {patient_id: np.ndarray binary mask}
    fp_thresholds: Optional[list] = None,
    iou_threshold: float = 0.1,
) -> dict:
    """
    Free-Response ROC (FROC) curve computation.
    Standard evaluation for lesion detection: sensitivity vs FP/scan.

    Parameters
    ----------
    candidates_by_patient : dict mapping patient_id -> list of candidate dicts
                            Each candidate must have 'research_score' and a mask
                            or bbox to reconstruct one.
    gt_masks_by_patient   : dict mapping patient_id -> binary 3D GT mask
    fp_thresholds         : FP/scan values at which to report sensitivity
                            Default: [0.125, 0.25, 0.5, 1, 2, 4, 8]
    iou_threshold         : IoU threshold for a detection to count as TP

    Returns
    -------
    {
        'froc_points': list of (fp_per_scan, sensitivity) tuples,
        'sensitivity_at_thresholds': {0.5: float, 1: float, 2: float, ...},
        'n_patients': int,
        'n_gt_lesions_total': int,
    }
    """
    if fp_thresholds is None:
        fp_thresholds = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    # Flatten all candidates with their patient source
    all_candidates = []
    for pid, cands in candidates_by_patient.items():
        for c in cands:
            all_candidates.append({"patient_id": pid, **c})

    if not all_candidates:
        return {
            "froc_points": [],
            "sensitivity_at_thresholds": {t: float("nan") for t in fp_thresholds},
            "n_patients": len(gt_masks_by_patient),
            "n_gt_lesions_total": 0,
        }

    # Sort by research_score descending
    all_candidates.sort(key=lambda c: c.get("research_score") or 0.0, reverse=True)

    n_patients = len(gt_masks_by_patient)
    n_gt_total = sum(
        cc_label(m > 0)[1] for m in gt_masks_by_patient.values()
    )

    froc_points = []
    detected_set = set()   # (patient_id, gt_lesion_id) pairs
    fp_count = 0

    for cand in all_candidates:
        pid = cand["patient_id"]
        gt_mask = gt_masks_by_patient.get(pid)
        if gt_mask is None:
            fp_count += 1
            continue

        gt_labeled, n_gt = cc_label(gt_mask > 0)
        # Reconstruct candidate mask from bbox
        shape = gt_mask.shape
        (z0, z1), (y0, y1), (x0, x1) = cand["bbox_voxel"]
        cand_mask = np.zeros(shape, dtype=np.uint8)
        cand_mask[z0:z1+1, y0:y1+1, x0:x1+1] = 1

        matched_gt = None
        for gt_id in range(1, n_gt + 1):
            gt_comp = (gt_labeled == gt_id)
            intersection = int((gt_comp & (cand_mask > 0)).sum())
            union = int((gt_comp | (cand_mask > 0)).sum())
            if union > 0 and intersection / union >= iou_threshold:
                matched_gt = (pid, gt_id)
                break

        if matched_gt and matched_gt not in detected_set:
            detected_set.add(matched_gt)
        else:
            fp_count += 1

        sensitivity = len(detected_set) / n_gt_total if n_gt_total > 0 else 0.0
        fp_per_scan = fp_count / n_patients if n_patients > 0 else 0.0
        froc_points.append((round(fp_per_scan, 4), round(sensitivity, 4)))

    # Sensitivity at standard FP/scan thresholds (linear interpolation)
    sens_at = {}
    for t in fp_thresholds:
        # Find sensitivity at the first point where fp_per_scan >= t
        found = [s for fp, s in froc_points if fp <= t]
        sens_at[t] = found[-1] if found else float("nan")

    return {
        "froc_points": froc_points,
        "sensitivity_at_thresholds": sens_at,
        "n_patients": n_patients,
        "n_gt_lesions_total": int(n_gt_total),
    }


# ── Patient-level split ───────────────────────────────────────────────────────

def make_patient_split(
    patient_ids: list,
    test_fraction: float = 0.20,
    val_fraction: float = 0.10,
    seed: int = 42,
    labels: Optional[dict] = None,    # {patient_id: str label} for stratification
) -> dict:
    """
    Create a reproducible patient-level train/val/test split.
    No patient appears in more than one partition.

    Parameters
    ----------
    patient_ids   : list of unique patient identifiers
    test_fraction : fraction of patients for held-out test set
    val_fraction  : fraction of patients for validation set
    seed          : random seed for reproducibility
    labels        : optional {patient_id: subtype_label} for stratified splitting

    Returns
    -------
    {'train': [...], 'val': [...], 'test': [...]}
    """
    rng = np.random.default_rng(seed)
    ids = list(patient_ids)

    if labels is not None:
        # Stratified split by subtype
        from collections import defaultdict
        groups = defaultdict(list)
        for pid in ids:
            groups[labels.get(pid, "unknown")].append(pid)

        train_ids, val_ids, test_ids = [], [], []
        for group_ids in groups.values():
            shuffled = list(group_ids)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_test = max(1, round(n * test_fraction))
            n_val = max(1, round(n * val_fraction))
            test_ids.extend(shuffled[:n_test])
            val_ids.extend(shuffled[n_test:n_test + n_val])
            train_ids.extend(shuffled[n_test + n_val:])
    else:
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * test_fraction))
        n_val = max(1, round(n * val_fraction))
        test_ids = shuffled[:n_test]
        val_ids = shuffled[n_test:n_test + n_val]
        train_ids = shuffled[n_test + n_val:]

    # Verify no overlap
    assert not (set(train_ids) & set(test_ids)), "Train-test overlap detected!"
    assert not (set(val_ids) & set(test_ids)), "Val-test overlap detected!"
    assert not (set(train_ids) & set(val_ids)), "Train-val overlap detected!"

    return {"train": train_ids, "val": val_ids, "test": test_ids}
