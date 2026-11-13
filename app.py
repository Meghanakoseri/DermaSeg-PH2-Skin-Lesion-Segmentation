from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from src.model import UNet

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = {
    "Proposed — CE + Focal Tversky": PROJECT_ROOT / "models" / "checkpoints" / "tune_focal_tversky_lr_0005_best.pt",
    "Baseline — Cross-Entropy": PROJECT_ROOT / "models" / "checkpoints" / "tune_ce_lr_0005_best.pt",
}
IMAGE_SIZE = 256
THRESHOLD = 0.5


@st.cache_resource(show_spinner="Loading segmentation model…")
def load_model(checkpoint_path: str, modified_time: float, device_name: str):
    """Load the trained U-Net checkpoint once per app session/cache key."""
    del modified_time  # Included in the cache key so changed weights reload.
    device = torch.device(device_name)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    saved_config = checkpoint.get("config", {})
    project_config = saved_config.get("project", {})
    in_channels = int(project_config.get("in_channels", 3))
    num_classes = int(project_config.get("num_classes", 2))

    model = UNet(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, device, checkpoint


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Match training preprocessing: RGB, 256x256, bilinear, [0,1]."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def predict_mask(model, device: torch.device, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return 256x256 probability map and binary mask."""
    input_tensor = preprocess_image(image).to(device)
    with torch.inference_mode():
        logits = model(input_tensor)
        probability = torch.softmax(logits, dim=1)[0, 1]
        binary = probability >= THRESHOLD
    return (
        probability.detach().cpu().numpy(),
        binary.detach().cpu().numpy().astype(np.uint8),
    )


def make_overlay(image: Image.Image, mask_256: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """Resize prediction to original image size using nearest-neighbour."""
    original = ImageOps.exif_transpose(image).convert("RGB")
    mask_image = Image.fromarray(mask_256 * 255, mode="L")
    mask_image = mask_image.resize(original.size, Image.Resampling.NEAREST)

    rgb = np.asarray(original, dtype=np.uint8).copy()
    binary = np.asarray(mask_image) > 0
    tint = np.array([255, 55, 75], dtype=np.float32)
    rgb[binary] = (
        0.55 * rgb[binary].astype(np.float32) + 0.45 * tint
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(rgb), mask_image


st.set_page_config(
    page_title="DermaSeg | Skin Lesion Segmentation",
    page_icon="🔬",
    layout="wide",
)

st.title("DermaSeg — Skin Lesion Segmentation")
st.caption("U-Net inference demo trained on the PH² dermoscopic dataset")
st.warning(
    "Research demonstration only. This tool is not a medical device, does not diagnose skin cancer, "
    "and must not be used to make clinical decisions."
)

with st.sidebar:
    st.header("Model settings")
    model_choice = st.selectbox(
        "Choose checkpoint",
        list(DEFAULT_CHECKPOINTS.keys()),
        index=0,
    )
    checkpoint_path = DEFAULT_CHECKPOINTS[model_choice]
    st.write(f"**Checkpoint:** `{checkpoint_path.relative_to(PROJECT_ROOT)}`")
    st.write(f"**Input size:** {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.write(f"**Foreground threshold:** {THRESHOLD:.2f}")

if not checkpoint_path.exists():
    st.error(
        f"Trained checkpoint not found: {checkpoint_path}\n\n"
        "Copy the trained `.pt` file from your Google Drive project folder into "
        "`models/checkpoints/`. Do not upload the PH² dataset to GitHub."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="The model predicts a binary lesion mask for one uploaded image.",
)

if uploaded_file is None:
    st.info("Upload one dermoscopic image to generate its predicted lesion mask.")
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
except Exception as exc:
    st.error(f"The uploaded file could not be opened as an image: {exc}")
    st.stop()

if image.width < 32 or image.height < 32:
    st.error("Please upload an image at least 32 × 32 pixels in size.")
    st.stop()

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_mtime = checkpoint_path.stat().st_mtime
    model, device, checkpoint = load_model(
        str(checkpoint_path), checkpoint_mtime, device
    )
    with st.spinner("Generating lesion mask…"):
        probability, mask_256 = predict_mask(model, device, image)
        overlay, full_size_mask = make_overlay(image, mask_256)
except Exception as exc:
    st.exception(exc)
    st.stop()

foreground_fraction = float(mask_256.mean())
col1, col2, col3 = st.columns(3)
with col1:
    st.image(image, caption="Uploaded image", use_container_width=True)
with col2:
    st.image(full_size_mask, caption="Predicted binary lesion mask", use_container_width=True)
with col3:
    st.image(overlay, caption="Prediction overlay", use_container_width=True)

st.metric("Predicted foreground area", f"{foreground_fraction * 100:.2f}%")
st.caption(
    f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')} · "
    f"Best validation IoU: {float(checkpoint.get('best_val_iou', float('nan'))):.4f} · "
    f"Inference device: {device}"
)

mask_buffer = io.BytesIO()
full_size_mask.save(mask_buffer, format="PNG")
st.download_button(
    "Download predicted mask (PNG)",
    data=mask_buffer.getvalue(),
    file_name="derm aseg_predicted_mask.png".replace(" ", ""),
    mime="image/png",
    use_container_width=True,
)

with st.expander("Technical details"):
    st.write(
        "The uploaded image is converted to RGB, resized to 256 × 256 using bilinear "
        "interpolation and scaled to [0, 1], matching the training preprocessing. "
        "The model's lesion-class softmax probability is thresholded at 0.5. "
        "The binary mask is resized back to the uploaded image dimensions with nearest-neighbour interpolation."
    )
