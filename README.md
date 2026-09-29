# V8 — Liver CT Research Suite

A research toolkit for multiphasic liver CT analysis built with Streamlit.

## Tools

| Tool | Description |
|---|---|
| 🌋 Depth Analyzer | Turns CT slices into interactive 3D surface meshes |
| 🎯 Candidate Viewer | 3D intensity anomaly detection across CT phases with evaluation metrics |

## Run locally

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Dataset

Uses the [PLC-CECT dataset](https://physionet.org) — 361 patients, 4 CT phases each, with expert liver + lesion segmentation masks.

## ⚠️ Disclaimer

Research prototype only. Not validated for clinical use. Not for diagnosis, treatment, or patient management.
