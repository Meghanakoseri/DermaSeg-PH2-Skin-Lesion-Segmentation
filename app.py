from __future__ import annotations

import io
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
DEFAULT_THRESHOLD = 0.50

st.set_page_config(
    page_title="DermaSeg | Skin Lesion Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark workspace styling inspired by the supplied reference screenshot.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ds-bg: #111217;
        --ds-sidebar: #1d1e28;
        --ds-card: #20222d;
        --ds-card-2: #262833;
        --ds-border: #363947;
        --ds-muted: #a6a9b7;
        --ds-text: #f2f3f7;
        --ds-accent: #a5e6bd;
        --ds-green: #35e477;
        --ds-red: #ff5268;
        --ds-blue: #579bff;
    }

    html, body, [class*="css"], .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"] {
        font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
    }
    h1, h2, h3, .ds-brand {
        font-family: 'Manrope', 'Segoe UI', sans-serif !important;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--ds-bg);
        color: var(--ds-text);
    }
    [data-testid="stHeader"] {
        background: rgba(17, 18, 23, 0.94);
    }
    [data-testid="stSidebar"] {
        background: var(--ds-sidebar);
        border-right: 1px solid #30323d;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: var(--ds-sidebar);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #c4c9d6 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        font-size: 0.82rem !important;
        line-height: 1.5 !important;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, label, span, div {
        color: inherit;
    }
    h1 {
        letter-spacing: -0.045em;
        font-size: clamp(2.15rem, 3.2vw, 2.85rem) !important;
        line-height: 1.12 !important;
        font-weight: 800 !important;
        color: #f7f8fc !important;
    }
    h2, h3 {
        letter-spacing: -0.025em;
        font-weight: 750 !important;
    }
    [data-testid="stMarkdownContainer"] p {
        color: #c4c6d0;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--ds-card);
        border: 1px solid var(--ds-border);
        border-radius: 13px;
    }
    [data-testid="stFileUploader"] {
        background: transparent;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--ds-card-2);
        border: 1px dashed #565a6a;
        border-radius: 10px;
        min-height: 105px;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--ds-accent);
    }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] p,
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #e6e9f1 !important;
        opacity: 1 !important;
        font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
    }
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] small * {
        color: #b9c0d0 !important;
        opacity: 1 !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: #151720;
        color: #f5f6fa;
        border: 1px solid #555a69;
        border-radius: 8px;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: var(--ds-accent);
        color: white;
    }
    [data-testid="stMetric"] {
        background: var(--ds-card);
        border: 1px solid var(--ds-border);
        border-radius: 12px;
        padding: 13px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: var(--ds-muted) !important;
    }
    [data-testid="stMetricValue"] {
        color: white !important;
    }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSlider"] [data-baseweb="slider"] {
        color: white;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #292b36;
        border-color: #454957;
    }
    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button {
        border-radius: 9px;
        border: 1px solid #5b806b;
        background: #254232;
        color: #effff4;
        font-weight: 650;
    }
    [data-testid="stButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {
        border-color: #9fe2b7;
        background: #305640;
        color: white;
    }
    [data-testid="stAlert"] {
        border-radius: 11px;
    }
    hr {
        border-color: #363946;
    }
    .ds-brand {
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #f6f7fb;
        padding: 0.35rem 0 0.1rem 0;
    }
    .ds-brand small {
        display: block;
        color: #9ba0b1;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .ds-nav-label, .ds-kicker {
        color: #8e93a4 !important;
        font-size: 0.70rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .ds-nav-active {
        background: #343744;
        border: 1px solid #454959;
        color: #f5f6fa;
        padding: 0.58rem 0.75rem;
        border-radius: 8px;
        font-size: 0.89rem;
        font-weight: 650;
        margin: 0.45rem 0 1rem 0;
    }
    .ds-nav-idle {
        color: #a8abba;
        padding: 0.42rem 0.75rem;
        font-size: 0.88rem;
    }
    .ds-page-subtitle {
        color: #aeb4c4;
        margin-top: -0.35rem;
        margin-bottom: 1.1rem;
        font-size: 0.98rem;
        line-height: 1.6;
    }
    .ds-page-subtitle strong {
        color: #e8ebf3;
        font-weight: 700;
    }
    .ds-disclaimer {
        background: #2c281d;
        border: 1px solid #66552c;
        color: #f4d99a;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0 1.35rem 0;
        font-size: 0.84rem;
    }
    .ds-card-title {
        color: #f1f2f7;
        font-size: 0.91rem;
        font-weight: 700;
        margin-bottom: 0.12rem;
    }
    .ds-card-hint {
        color: #999eaf;
        font-size: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .ds-section-title {
        color: #f1f2f7;
        font-weight: 750;
        font-size: 1.03rem;
        margin: 1.2rem 0 0.75rem 0;
    }
    .ds-section-hint {
        color: #9da2b2;
        font-size: 0.79rem;
        margin-top: -0.55rem;
        margin-bottom: 0.85rem;
    }
    .ds-placeholder {
        background: #171922;
        border: 1px dashed #4a4e5d;
        border-radius: 9px;
        min-height: 210px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 1rem;
        color: #9da2b2;
    }
    .ds-placeholder strong {
        color: #e4e6ed;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .ds-legend {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        color: #b8bbc7;
        font-size: 0.77rem;
        margin: 0.35rem 0 0.75rem 0;
    }
    .ds-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .ds-footnote {
        color: #aeb6c8;
        font-size: 0.78rem;
        line-height: 1.65;
        margin-top: 1rem;
    }
    .ds-metric-note {
        background: #202530;
        border: 1px solid #3d4658;
        color: #c6cede;
        border-radius: 9px;
        padding: 0.65rem 0.85rem;
        font-size: 0.78rem;
        margin: 0.25rem 0 0.9rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading segmentation model…")
def load_model(checkpoint_path: str, modified_time: float, device_name: str):
    """Load a trained U-Net checkpoint once per app session/cache key."""
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
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def predict_mask(
    model: torch.nn.Module,
    device: torch.device,
    image: Image.Image,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 256x256 probability map and binary mask."""
    input_tensor = preprocess_image(image).to(device)
    with torch.inference_mode():
        logits = model(input_tensor)
        probability = torch.softmax(logits, dim=1)[0, 1]
        binary = probability >= threshold
    return (
        probability.detach().cpu().numpy(),
        binary.detach().cpu().numpy().astype(np.uint8),
    )


def resize_binary_mask(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return mask_image.resize(size, Image.Resampling.NEAREST)


def make_overlay(
    image: Image.Image,
    mask_image: Image.Image,
    tint: tuple[int, int, int],
    alpha: float = 0.48,
) -> Image.Image:
    original = ImageOps.exif_transpose(image).convert("RGB")
    binary = np.asarray(mask_image.convert("L")) > 0
    rgb = np.asarray(original, dtype=np.uint8).copy()
    tint_array = np.asarray(tint, dtype=np.float32)
    rgb[binary] = (
        (1.0 - alpha) * rgb[binary].astype(np.float32) + alpha * tint_array
    ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def make_overlap_map(prediction: np.ndarray, ground_truth: np.ndarray) -> Image.Image:
    """Green=TP, red=FP, blue=FN; black=background correctly rejected."""
    pred = prediction.astype(bool)
    truth = ground_truth.astype(bool)
    tp = pred & truth
    fp = pred & ~truth
    fn = ~pred & truth
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    rgb[tp] = (35, 228, 119)
    rgb[fp] = (255, 70, 88)
    rgb[fn] = (75, 145, 255)
    return Image.fromarray(rgb, mode="RGB")


def show_image_panel(title: str, image: Image.Image | None, empty_text: str) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="ds-card-title">{title}</div>', unsafe_allow_html=True)
        if image is None:
            st.markdown(
                f'<div class="ds-placeholder"><strong>Not available</strong>{empty_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.image(image, use_container_width=True)


# Sidebar --------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="ds-brand">◉ DermaSeg<small>PH² · IMAGE ANALYSIS</small></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ds-nav-label">NAVIGATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="ds-nav-idle">⌂ &nbsp; Home</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ds-nav-active">▦ &nbsp; Skin lesion segmentation</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="ds-nav-label">SETTINGS</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Model checkpoint",
        list(DEFAULT_CHECKPOINTS.keys()),
        index=0,
        label_visibility="visible",
    )
    threshold = st.slider(
        "Prediction threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Pixels with lesion probability at or above this value are marked as lesion.",
    )
    checkpoint_path = DEFAULT_CHECKPOINTS[model_choice]
    st.caption(f"Checkpoint: `{checkpoint_path.name}`")
    st.caption(f"Input size: {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.caption("Runtime: automatic CPU / CUDA")
    st.divider()
    st.markdown('<div class="ds-nav-label">ABOUT</div>', unsafe_allow_html=True)
    st.caption("U-Net trained on the PH² dermoscopic dataset.")
    st.caption("Academic research demonstration only.")


# Main workspace -------------------------------------------------------------
st.markdown('<div class="ds-kicker">DERMASEG / RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
st.title("DermaSeg")
st.markdown(
    '<div class="ds-page-subtitle"><strong>Skin lesion segmentation</strong> · Upload a dermoscopic image, generate a lesion mask, and compare it with a ground-truth mask when available.</div>', 
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="ds-disclaimer">⚠ &nbsp; Research demonstration only — not a medical device. It does not diagnose skin cancer and must not be used for clinical decisions.</div>',
    unsafe_allow_html=True,
)

# Three compact upload / output cards, following the reference layout.
up_col1, up_col2, up_col3 = st.columns(3, gap="medium")
with up_col1:
    with st.container(border=True):
        st.markdown('<div class="ds-card-title">01 &nbsp; Original image</div>', unsafe_allow_html=True)
        st.markdown('<div class="ds-card-hint">Upload a dermoscopic image · JPG, PNG, BMP or TIFF</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Original image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="input_image",
            label_visibility="collapsed",
            help="The image that the U-Net will segment.",
        )
with up_col2:
    with st.container(border=True):
        st.markdown('<div class="ds-card-title">02 &nbsp; Ground-truth mask</div>', unsafe_allow_html=True)
        st.markdown('<div class="ds-card-hint">Optional · upload the matching binary lesion mask</div>', unsafe_allow_html=True)
        uploaded_ground_truth = st.file_uploader(
            "Ground-truth mask",
            type=["png", "bmp", "tif", "tiff", "jpg", "jpeg"],
            key="ground_truth_mask",
            label_visibility="collapsed",
            help="Optional. Used only to calculate comparison metrics; it does not affect prediction.",
        )
with up_col3:
    with st.container(border=True):
        st.markdown('<div class="ds-card-title">03 &nbsp; Model prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="ds-card-hint">Generated automatically by the selected U-Net</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ds-placeholder" style="min-height:105px"><strong>Ready for inference</strong>The predicted binary mask appears below after an image is uploaded.</div>',
            unsafe_allow_html=True,
        )

if uploaded_image is None:
    st.markdown(
        '<div class="ds-footnote">Start by uploading an original dermoscopic image. A ground-truth mask is optional and is only needed for overlap metrics.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not checkpoint_path.exists():
    st.error(
        f"Trained checkpoint not found: {checkpoint_path}. Copy the selected .pt file into models/checkpoints/."
    )
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(uploaded_image)).convert("RGB")
except Exception as exc:
    st.error(f"The uploaded image could not be opened: {exc}")
    st.stop()

if image.width < 32 or image.height < 32:
    st.error("Please upload an image at least 32 × 32 pixels in size.")
    st.stop()

try:
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    model, device, checkpoint = load_model(
        str(checkpoint_path), checkpoint_path.stat().st_mtime, runtime_device
    )
    with st.spinner("Generating lesion mask…"):
        probability, prediction_256 = predict_mask(model, device, image, threshold)
        prediction_mask_full = resize_binary_mask(prediction_256, image.size)
        prediction_overlay = make_overlay(
            image, prediction_mask_full, tint=(255, 65, 80), alpha=0.46
        )
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

# Optional ground-truth processing and comparison.
gt_mask_256: np.ndarray | None = None
gt_mask_full: Image.Image | None = None
gt_overlay: Image.Image | None = None
overlap_map: Image.Image | None = None
comparison_metrics: dict[str, float] = {}

if uploaded_ground_truth is not None:
    try:
        gt_source = ImageOps.exif_transpose(Image.open(uploaded_ground_truth)).convert("L")
        if gt_source.size != image.size:
            st.warning(
                "The uploaded image and ground-truth mask have different original dimensions. "
                "They will both be resized to 256 × 256, but please confirm they are the correct matching pair."
            )
        gt_resized = gt_source.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
        gt_mask_256 = (np.asarray(gt_resized, dtype=np.uint8) >= 128).astype(np.uint8)
        gt_mask_full = resize_binary_mask(gt_mask_256, image.size)
        gt_overlay = make_overlay(
            image, gt_mask_full, tint=(35, 228, 119), alpha=0.42
        )
        overlap_map = make_overlap_map(prediction_256, gt_mask_256).resize(
            image.size, Image.Resampling.NEAREST
        )

        pred_bool = prediction_256.astype(bool)
        gt_bool = gt_mask_256.astype(bool)
        tp = int(np.logical_and(pred_bool, gt_bool).sum())
        fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
        fn = int(np.logical_and(~pred_bool, gt_bool).sum())
        union = tp + fp + fn
        comparison_metrics = {
            "IoU": tp / union if union else 1.0,
            "Dice / F1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
            "Precision": tp / (tp + fp) if (tp + fp) else 1.0,
            "Recall": tp / (tp + fn) if (tp + fn) else 1.0,
        }
    except Exception as exc:
        st.error(f"The ground-truth mask could not be processed: {exc}")
        st.stop()

# Results grid: three columns by two rows like the supplied target UI.
st.markdown('<div class="ds-section-title">Visual comparison</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ds-section-hint">Top row: source and binary masks · Bottom row: overlap and colour overlays</div>',
    unsafe_allow_html=True,
)
row1 = st.columns(3, gap="medium")
with row1[0]:
    show_image_panel("Original image", image, "Upload an input image to begin.")
with row1[1]:
    show_image_panel(
        "Ground-truth mask",
        gt_mask_full,
        "No reference mask uploaded. Add the matching PH² lesion mask above to enable comparison.",
    )
with row1[2]:
    show_image_panel("Predicted mask", prediction_mask_full, "Prediction is generated automatically.")

row2 = st.columns(3, gap="medium")
with row2[0]:
    show_image_panel(
        "Overlap map",
        overlap_map,
        "Upload a ground-truth mask to compare correct and missed / extra pixels.",
    )
with row2[1]:
    show_image_panel(
        "Ground-truth overlay",
        gt_overlay,
        "Available after a matching ground-truth mask is uploaded.",
    )
with row2[2]:
    show_image_panel("Prediction overlay", prediction_overlay, "Prediction overlay is generated automatically.")

if gt_mask_256 is not None:
    st.markdown(
        '<div class="ds-legend">'
        '<span><i class="ds-dot" style="background:#23e477"></i>True positive</span>'
        '<span><i class="ds-dot" style="background:#ff4658"></i>False positive</span>'
        '<span><i class="ds-dot" style="background:#4b91ff"></i>False negative</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# Metrics and downloads.
foreground_percent = float(prediction_256.mean() * 100.0)
st.markdown('<div class="ds-section-title">Prediction summary</div>', unsafe_allow_html=True)
if comparison_metrics:
    # Use two rows of three cards so values remain readable at laptop widths.
    metric_rows = [
        [
            ("Predicted lesion area", f"{foreground_percent:.2f}%"),
            ("IoU", f"{comparison_metrics['IoU']:.4f}"),
            ("Dice / F1", f"{comparison_metrics['Dice / F1']:.4f}"),
        ],
        [
            ("Threshold", f"{threshold:.2f}"),
            ("Precision", f"{comparison_metrics['Precision']:.4f}"),
            ("Recall", f"{comparison_metrics['Recall']:.4f}"),
        ],
    ]
    for metric_row in metric_rows:
        metric_cols = st.columns(3, gap="medium")
        for metric_col, (metric_name, metric_value) in zip(metric_cols, metric_row):
            with metric_col:
                st.metric(metric_name, metric_value)
    st.markdown(
        '<div class="ds-metric-note">These metrics compare this single uploaded image with its uploaded ground-truth mask. They are not the overall 30-image test-set results.</div>',
        unsafe_allow_html=True,
    )
else:
    metric_cols = st.columns(2, gap="medium")
    with metric_cols[0]:
        st.metric("Predicted lesion area", f"{foreground_percent:.2f}%")
    with metric_cols[1]:
        st.metric("Threshold", f"{threshold:.2f}")

info_col, detail_col = st.columns([1, 2], gap="medium")
with info_col:
    st.markdown(
        f'<div class="ds-footnote">Selected model: <b>{model_choice}</b><br>'
        f'Checkpoint epoch: <b>{checkpoint.get("epoch", "unknown")}</b><br>'
        f'Best validation IoU: <b>{float(checkpoint.get("best_val_iou", float("nan"))):.4f}</b><br>'
        f'Inference device: <b>{device}</b></div>',
        unsafe_allow_html=True,
    )
with detail_col:
    with st.expander("Preprocessing and evaluation details"):
        st.write(
            "The input image is converted to RGB, resized to 256 × 256 using bilinear interpolation, "
            "and scaled to [0, 1]. The model's lesion-class softmax probability is thresholded using "
            "the selected threshold. The predicted mask is resized to the original image size using "
            "nearest-neighbour interpolation. The optional ground-truth mask is used only for the "
            "displayed overlap and metrics; it is not passed to the model."
        )

mask_buffer = io.BytesIO()
prediction_mask_full.save(mask_buffer, format="PNG")
overlay_buffer = io.BytesIO()
prediction_overlay.save(overlay_buffer, format="PNG")
download_col1, download_col2, _ = st.columns([1, 1, 2], gap="medium")
with download_col1:
    st.download_button(
        "↓  Download predicted mask",
        data=mask_buffer.getvalue(),
        file_name="derm aseg_predicted_mask.png".replace(" ", ""),
        mime="image/png",
        use_container_width=True,
    )
with download_col2:
    st.download_button(
        "↓  Download prediction overlay",
        data=overlay_buffer.getvalue(),
        file_name="derm aseg_prediction_overlay.png".replace(" ", ""),
        mime="image/png",
        use_container_width=True,
    )

st.markdown(
    '<div class="ds-footnote">DermaSeg is an academic segmentation demo, not a diagnostic tool. '
    'Ground-truth comparison metrics are meaningful only when the uploaded mask belongs to the uploaded image.</div>',
    unsafe_allow_html=True,
)
