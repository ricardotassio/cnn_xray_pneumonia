# Chest X-Ray Pneumonia (Path B) — Export

This bundle contains:

- `artifacts/transfer_final.keras`, `artifacts/custom_final.keras`
- Thresholds: `artifacts/threshold_transfer.json`, `artifacts/threshold_custom.json`
- Traditional models (.pkl): LR/SVM/RF/KNN for HOG and CNN features
- Plots: PR curves, confusion matrices, and `final_model_comparison_test_ap.png`
- Summaries: `cnn_summary.json/csv`, `traditional_results.json`
- Repro env: `requirements.txt`
- Notebook (if copied here): `Chest_XRay_PathB_Explained.ipynb`

## Quick start (Streamlit)

- Load Keras models from `artifacts/*.keras`
- Load thresholds from `artifacts/threshold_*.json`
- Load traditional baselines from `artifacts/*.pkl`

### Run the Streamlit app

You can run from either this folder (`xray_project_export/`) or the repository root.

1. Install requirements (recommended):

```
pip install -r requirements.txt
```

2. Start Streamlit

Option A — from this folder:

```
streamlit run app.py
```

Option B — from the repository root:

```
python3 -m streamlit run xray_project_export/app.py
```

3. Upload a chest X-ray image and choose a model (HOG or CNN features).

#### Minimal install (HOG models only)

If you only need HOG models (no TensorFlow):

```
python3 -m pip install streamlit==1.36.0 scikit-image==0.25.2 pillow==11.3.0 numpy==2.0.2 scikit-learn==1.6.1
python3 -m streamlit run app.py
```

#### Notes

- HOG models require `scikit-image`. CNN-feature models require `tensorflow` (included in `requirements.txt`).
- If the `streamlit` command is not found, use the module form: `python3 -m streamlit run app.py`.
- Make sure you run the command from `xray_project_export/` or use the path `xray_project_export/app.py` from the repo root.

#### Troubleshooting

- Missing scikit-image: `python3 -m pip install scikit-image==0.25.2`
- First-time EfficientNet weights download requires internet for CNN-feature models.
