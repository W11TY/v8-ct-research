import nibabel as nib
from PIL import Image
import numpy as np
import streamlit as st

@st.cache_data
def load_nifti(file_path_or_bytes):
    """
    Load a NIfTI file and extract its core properties.
    Using st.cache_data prevents reloading the same file on every UI interaction.
    """
    if hasattr(file_path_or_bytes, 'read'):
        # Uploaded via Streamlit file_uploader
        # Nibabel needs a FileHolder or we can just read bytes via a memory buffer
        # But for nibabel it's easiest to write to a temp file first, 
        # as it heavily relies on file-like random access for large files.
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            tmp.write(file_path_or_bytes.read())
            tmp_path = tmp.name
        
        img = nib.load(tmp_path)
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        
        # Clean up temp file (can defer this but okay for prototype)
        # os.unlink(tmp_path) 
    else:
        img = nib.load(file_path_or_bytes)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

    # Extract voxel spacing (usually from pixdim in header)
    # pixdim[1:4] gives x, y, z physical spacing
    spacing = header.get_zooms()[:3]
    
    return {
        'data': data,
        'affine': affine,
        'spacing': spacing,
        'shape': data.shape,
        'dtype': data.dtype,
        'orientation': nib.aff2axcodes(affine)
    }


def load_image_as_array(file_or_path) -> np.ndarray:
    """
    Load a flat 2D image (PNG, JPEG, TIFF, BMP) as a grayscale float32 array.

    Parameters
    ----------
    file_or_path : str, Path, or file-like object
        Path to the image file or a file-like (e.g. BytesIO from st.file_uploader).

    Returns
    -------
    np.ndarray
        2D float32 array with values in [0, 255], shape (H, W).
    """
    img = Image.open(file_or_path).convert("L")  # 'L' = 8-bit grayscale
    return np.array(img, dtype=np.float32)


def load_plc_study(
    dataset_root: str,
    patient_row: dict,
    phase_col_map: dict | None = None,
) -> dict:
    """
    Load all available phases and masks for one PLC-CECT patient row.

    Parameters
    ----------
    dataset_root  : absolute path to the root folder of the PLC-CECT download
    patient_row   : dict representing one row from patient_data.csv
                    Expected keys (from confirmed PLC-CECT schema):
                      'CT File'         -> arterial phase path (relative to root)
                      'Liver Mask File' -> liver mask path (relative to root)
                      'Mask File'       -> lesion mask path (relative to root, may be NaN)
                      'Stage'           -> phase label (e.g. 'C1' = arterial, etc.)
    phase_col_map : optional override mapping phase labels to file-column names
                    If None, assumes 'CT File' is always the primary scan for that row.

    Returns
    -------
    {
        'phases': {
            'plain':    nifti_dict | None,
            'arterial': nifti_dict | None,
            'venous':   nifti_dict | None,
            'delayed':  nifti_dict | None,
        },
        'liver_mask':  nifti_dict | None,
        'lesion_mask': nifti_dict | None,
        'patient_id':  str,
        'stage':       str,
    }

    Notes
    -----
    The PLC-CECT CSV has one row per patient per stage. To load all phases,
    pass all four rows for the patient and call this function once per row,
    then merge the 'phases' dicts. Alternatively, load all rows at once via
    load_plc_patient_all_phases().
    """
    import os
    import pandas as pd

    result = {
        "phases": {"plain": None, "arterial": None, "venous": None, "delayed": None},
        "liver_mask": None,
        "lesion_mask": None,
        "patient_id": patient_row.get("Patient ID", "unknown"),
        "stage": patient_row.get("Stage", "unknown"),
    }

    def _load_path(rel_path):
        if not rel_path or (isinstance(rel_path, float) and math.isnan(rel_path)):
            return None
        full = os.path.join(dataset_root, str(rel_path).strip())
        if not os.path.exists(full):
            return None
        return load_nifti(full)

    import math

    # Load CT scan for this stage
    ct_path = patient_row.get("CT File")
    nifti_ct = _load_path(ct_path)

    # Map Stage label to phase key
    # PLC-CECT stages: C0=Plain, C1=Arterial, C2=Venous, C3=Delayed (convention varies)
    # Configurable via phase_col_map
    stage_to_phase = (phase_col_map or {
        "C0": "plain",
        "C1": "arterial",
        "C2": "venous",
        "C3": "delayed",
        # Also accept plain English labels
        "plain": "plain",
        "arterial": "arterial",
        "venous": "venous",
        "delayed": "delayed",
    })
    phase_key = stage_to_phase.get(str(result["stage"]), "arterial")
    result["phases"][phase_key] = nifti_ct

    # Load masks
    result["liver_mask"] = _load_path(patient_row.get("Liver Mask File"))
    result["lesion_mask"] = _load_path(patient_row.get("Mask File"))

    return result


def load_plc_patient_all_phases(
    dataset_root: str,
    patient_id: str,
    patient_data_df,  # pandas DataFrame of patient_data.csv
    phase_col_map: dict | None = None,
) -> dict:
    """
    Load all four phases + masks for a single patient from the PLC-CECT dataset.

    Parameters
    ----------
    dataset_root    : path to PLC-CECT root directory
    patient_id      : e.g. 'P0059'
    patient_data_df : full pandas DataFrame of patient_data.csv

    Returns
    -------
    Merged study dict with all four phase slots populated where data exists.
    """
    rows = patient_data_df[patient_data_df["Patient ID"] == patient_id]
    merged = {
        "phases": {"plain": None, "arterial": None, "venous": None, "delayed": None},
        "liver_mask": None,
        "lesion_mask": None,
        "patient_id": patient_id,
    }
    for _, row in rows.iterrows():
        study = load_plc_study(dataset_root, row.to_dict(), phase_col_map)
        for phase, data in study["phases"].items():
            if data is not None:
                merged["phases"][phase] = data
        if study["liver_mask"] is not None:
            merged["liver_mask"] = study["liver_mask"]
        if study["lesion_mask"] is not None:
            merged["lesion_mask"] = study["lesion_mask"]
    return merged
