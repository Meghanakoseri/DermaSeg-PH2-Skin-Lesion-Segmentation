# DermaSeg - Streamlit Inference Demo

The Streamlit application provides an inference and visualization interface for the trained U-Net models. It does not train models.

## 1. Required checkpoints

Place the selected checkpoints in `models/checkpoints/`:

- `tune_focal_tversky_lr_0005_best.pt` — proposed CE + Focal Tversky model.
- `tune_ce_lr_0005_best.pt` — Cross-Entropy baseline.

Model weights are excluded from Git. Copy the authorized checkpoint files from the original project backup/Google Drive. Do not commit PH² images, masks, archives, or model weights to this repository.

## 2. Install dependencies

Create and activate a Python environment, then run:

```bash
python -m pip install -r requirements.txt
```

For the Lenovo CPU-only laptop, CPU inference is supported. A CUDA GPU is not required to run the single-image demonstration.

## 3. Launch the application

From the repository root:

```bash
python -m streamlit run app.py
```

Open the local Streamlit address shown in the terminal (normally `http://localhost:8501`).

## 4. Demo workflow

1. Select the baseline or proposed checkpoint.
2. Upload a dermoscopic image.
3. Optionally upload the corresponding ground-truth lesion mask for a visual comparison and per-image metrics.
4. Inspect the original image, ground truth (when supplied), predicted mask, overlay and comparison panels.
5. Download the generated prediction/visualization where supported by the interface.

The app uses 256 × 256 model inputs and a lesion-probability threshold initialized to 0.50.

## 5. Limitations

- This is an academic image-segmentation demonstration, not a diagnostic or clinical tool.
- Per-image metrics are meaningful only when the uploaded ground-truth mask belongs to the uploaded image.
- The model was developed using PH²; performance on other datasets, cameras, populations or acquisition conditions is not established.
- The app performs inference only; it does not train the model.
