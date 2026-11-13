# Run the DermaSeg image-upload demo

The Streamlit app is an inference interface. Training is not performed by the app.

## 1. Put the trained checkpoint in the project

Copy the selected trained checkpoint from Google Drive into:

```text
models/checkpoints/candidate_focal_tversky_best.pt
```

Optionally copy the baseline checkpoint to:

```text
models/checkpoints/baseline_ce_best.pt
```

Do not commit the PH² dataset archive or large model checkpoints to an ordinary GitHub repository. The app expects the checkpoint file to be present locally when it starts.

## 2. Install dependencies

Create/activate a Python environment, then install the project requirements:

```bash
pip install -r requirements.txt
```

For the Lenovo laptop, run CPU inference; a CUDA GPU is not required for a single-image demo. If installing PyTorch locally, choose the CPU build from the official PyTorch installation selector.

## 3. Start the local app

From the root of the project folder:

```bash
streamlit run app.py
```

Streamlit opens a local browser page (normally `http://localhost:8501`). Upload one JPG, PNG, BMP or TIFF dermoscopic image. The app displays the input, binary predicted mask and overlay, and lets you download the mask as PNG.

## 4. Limitations

- This is an academic segmentation demo, not a diagnostic or clinical tool.
- The model has been trained and evaluated on PH² images; performance on arbitrary cameras, image sources, skin tones, or acquisition conditions is not established.
- The interface uses the same 256 × 256 resize and 0.5 lesion-probability threshold as the reported evaluation.
- A public Streamlit deployment requires a reliable way to provide the model checkpoint at runtime. Do not put large `.pt` files or PH² data in ordinary Git history. Use an approved model-artifact host or Git LFS only after checking storage/bandwidth limits and dataset terms.
