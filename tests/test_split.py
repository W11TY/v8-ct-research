import pytest
from utils.evaluation import generate_patient_split

def test_split_disjointness():
    scan_ids = [
        "scan1", "scan2", "scan3", "scan4", "scan5",
        "scan6", "scan7", "scan8",
        "scan9", "scan10", "scan11", "scan12", "scan13", "scan14", "scan15", "scan16", "scan17", "scan18"
    ]
    patient_ids = [
        "P1", "P1", "P2", "P3", "P4",
        "P5", "P5", "P5",
        "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14", "P15"
    ]
    patient_labels = ["cancer" if int(pid[1:]) % 2 == 0 else "control" for pid in patient_ids]
    
    split_info = generate_patient_split(scan_ids, patient_ids, patient_labels, test_fraction=0.2, n_folds=3, seed=42)
    
    test_set = set(split_info["test"])
    assert len(test_set) > 0
    
    # Map scan to patient
    scan_to_patient = dict(zip(scan_ids, patient_ids))
    
    # Ensure all scans from a patient go together
    test_groups = set([scan_to_patient[scan] for scan in test_set])
    for scan in scan_ids:
        group = scan_to_patient[scan]
        if group in test_groups:
            assert scan in test_set
        else:
            assert scan not in test_set
            
    for fold in split_info["cv_folds"]:
        train_set = set(fold["train"])
        val_set = set(fold["val"])
        
        assert not (train_set & test_set)
        assert not (val_set & test_set)
        assert not (train_set & val_set)
        
        train_groups = set([scan_to_patient[s] for s in train_set])
        val_groups = set([scan_to_patient[s] for s in val_set])
        assert not (train_groups & val_groups)
        assert not (train_groups & test_groups)
        assert not (val_groups & test_groups)

def test_split_ambiguity():
    # Same patient ID, but conflicting labels
    with pytest.raises(ValueError):
        generate_patient_split(["s1", "s2"], ["P1", "P1"], ["cancer", "control"])
