import os
import sys
import pandas as pd
import json

# Ensure utils is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation import generate_patient_split
from utils.config import load_config

def create_synthetic_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = []
    for i in range(1, 21):
        pid = f"P{i:03d}"
        label = "cancer" if i % 2 == 0 else "control"
        data.append({
            "Patient ID": pid,
            "Scan ID": f"{pid}_art",
            "Subtype": label,
            "Stage": "Arterial"
        })
        if i % 3 == 0:
            data.append({
                "Patient ID": pid,
                "Scan ID": f"{pid}_ven",
                "Subtype": label,
                "Stage": "Venous"
            })
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"Created synthetic dataset at {path}")

def run():
    config = load_config("config.json")
    csv_path = "data/patient_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Operating in SYNTHETIC mode.")
        create_synthetic_csv(csv_path)
        
    df = pd.read_csv(csv_path)
    
    required_cols = ["Patient ID", "Scan ID", "Subtype"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Extract
    scan_ids = df["Scan ID"].tolist()
    patient_ids = df["Patient ID"].tolist()
    patient_labels = df["Subtype"].tolist()
    
    out_json = "patient_split.json"
    generate_patient_split(
        scan_ids, 
        patient_ids, 
        patient_labels, 
        test_fraction=config.get("test_fraction", 0.20),
        n_folds=5,
        seed=config.get("seed", 42),
        output_json=out_json
    )
    print(f"Split generated successfully and saved to {out_json}")

if __name__ == "__main__":
    run()
