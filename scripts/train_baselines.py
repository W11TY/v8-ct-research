import os
import sys
import json
import math
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config

FEATURES = [
    'volume_mm3', 'equivalent_diameter_mm', 'compactness', 
    'sphericity', 'sphericity_fallback',
    'mean_hu_raw', 'median_hu_raw', 'liver_bg_hu', 'liver_mu', 
    'liver_sd', 'contrast_vs_liver',
    'z_mean', 'z_max', 'z_95'
]

def generate_mock_candidates(split_info: dict) -> pd.DataFrame:
    """Generate synthetic candidates for testing the tuning harness."""
    all_scans = split_info["test"]
    for fold in split_info["cv_folds"]:
        all_scans.extend(fold["train"])
        all_scans.extend(fold["val"])
    all_scans = list(set(all_scans))
    
    rng = np.random.default_rng(42)
    rows = []
    
    for scan in all_scans:
        # Generate 10-50 candidates per scan
        n_cands = rng.integers(10, 50)
        for i in range(n_cands):
            # 10% chance to be POSITIVE
            is_pos = rng.random() < 0.1
            is_ambig = rng.random() < 0.05
            
            label = "POSITIVE" if is_pos else ("AMBIGUOUS" if is_ambig else "NEGATIVE")
            
            row = {
                "scan_id": scan,
                "candidate_id": f"{scan}_c{i}",
                "label": label,
                
                # Mock features
                "volume_mm3": rng.uniform(10, 5000),
                "equivalent_diameter_mm": rng.uniform(2, 50),
                "compactness": rng.uniform(0.1, 1.0),
                # 5% chance of NaN sphericity
                "sphericity": float('nan') if rng.random() < 0.05 else rng.uniform(0.5, 1.0),
                "sphericity_fallback": False,
                
                "mean_hu_raw": rng.uniform(50, 150),
                "median_hu_raw": rng.uniform(50, 150),
                "liver_bg_hu": rng.uniform(40, 80),
                "liver_mu": rng.uniform(40, 80),
                "liver_sd": rng.uniform(5, 20),
                "contrast_vs_liver": rng.uniform(10, 100),
                
                "z_mean": rng.uniform(1, 10),
                "z_max": rng.uniform(2, 15),
                "z_95": rng.uniform(1.5, 12),
            }
            if math.isnan(row["sphericity"]):
                row["sphericity_fallback"] = True
                
            # Make POSITIVE candidates have higher z_max to simulate signal
            if is_pos:
                row["z_max"] += 5.0
                
            rows.append(row)
            
    return pd.DataFrame(rows)

def train_baselines():
    config = load_config("config.json")
    
    if not os.path.exists("patient_split.json"):
        print("patient_split.json not found. Run scripts/run_split.py first.")
        return
        
    with open("patient_split.json", "r") as f:
        split_info = json.load(f)
        
    # In real usage, you load extracted candidate JSONs. 
    # Here we use synthetic mock data.
    df = generate_mock_candidates(split_info)
    
    # Baseline A (Max-Z)
    print("Evaluating Baseline A (max-Z ranking) on CV folds...")
    # Baseline A doesn't require training, just use 'z_max' as score
    
    # Baseline B (Random Forest)
    print("Tuning Baseline B (Random Forest)...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'class_weight': ['balanced']
    }
    
    best_params = None
    best_avg_ap = -1
    
    for params in ParameterGrid(param_grid):
        fold_aps = []
        for fold_idx, fold in enumerate(split_info["cv_folds"]):
            train_scans = fold["train"]
            val_scans = fold["val"]
            
            train_df = df[df["scan_id"].isin(train_scans) & (df["label"] != "AMBIGUOUS")]
            val_df = df[df["scan_id"].isin(val_scans) & (df["label"] != "AMBIGUOUS")]
            
            X_train = train_df[FEATURES]
            y_train = (train_df["label"] == "POSITIVE").astype(int)
            
            X_val = val_df[FEATURES]
            y_val = (val_df["label"] == "POSITIVE").astype(int)
            
            # Imputation fit on training data only
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_val_imp = imputer.transform(X_val)
            
            clf = RandomForestClassifier(**params, random_state=42)
            clf.fit(X_train_imp, y_train)
            
            val_probs = clf.predict_proba(X_val_imp)[:, 1]
            ap = average_precision_score(y_val, val_probs) if len(y_val.unique()) > 1 else 0.0
            fold_aps.append(ap)
            
        avg_ap = np.mean(fold_aps)
        print(f"Params: {params} -> Avg AP: {avg_ap:.4f}")
        if avg_ap > best_avg_ap:
            best_avg_ap = avg_ap
            best_params = params
            
    print(f"Best RF params: {best_params} (AP: {best_avg_ap:.4f})")
    
    # Train final model on ALL tuning data (train+val across all folds, excluding test)
    tuning_scans = []
    for f in split_info["cv_folds"]:
        tuning_scans.extend(f["train"])
        tuning_scans.extend(f["val"])
    tuning_scans = list(set(tuning_scans))
    
    tuning_df = df[df["scan_id"].isin(tuning_scans) & (df["label"] != "AMBIGUOUS")]
    X_tune = tuning_df[FEATURES]
    y_tune = (tuning_df["label"] == "POSITIVE").astype(int)
    
    final_imputer = SimpleImputer(strategy='median')
    X_tune_imp = final_imputer.fit_transform(X_tune)
    
    final_rf = RandomForestClassifier(**best_params, random_state=42)
    final_rf.fit(X_tune_imp, y_tune)
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": final_rf, "imputer": final_imputer, "features": FEATURES}, "models/rf_baseline.joblib")
    
    with open("models/features.json", "w") as f:
        json.dump(FEATURES, f, indent=4)
        
    print("Final model saved to models/rf_baseline.joblib")

if __name__ == "__main__":
    train_baselines()
