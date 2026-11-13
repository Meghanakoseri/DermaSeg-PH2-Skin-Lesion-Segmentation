# DermaSeg — PH² Skin Lesion Segmentation

A PyTorch U-Net pipeline for binary lesion segmentation in dermoscopic images. The work compares a Cross-Entropy baseline with compound overlap losses, performs a focal-Tversky ablation, evaluates on a fixed held-out test set, and reports three training seeds.

> **Scope:** this is an academic image-segmentation demonstration, not a skin-cancer classifier, diagnostic system, medical device, or substitute for professional medical care.

## Dataset and protocol

- Dataset: PH², 200 dermoscopic images with corresponding binary lesion masks.
- Audit: 200 readable image–mask pairs; no missing pairs or dimension mismatches; masks contain 0/255 values.
- Preprocessing: RGB images and grayscale masks resized to 256 × 256; bilinear interpolation for images and nearest-neighbour for masks; image values scaled to [0, 1], masks converted to classes {0, 1}.
- Fixed split: 140 training / 30 validation / 30 test, stratified across five lesion-area groups; split seed 42; same case allocation reused for all experiments.
- The PH² archive is not redistributed in this repository. Obtain it from the dataset provider/Kaggle source referenced in the project report and follow its access and reuse terms.

## Model and experiments

The U-Net receives RGB images and predicts two per-pixel classes (background and lesion). The baseline uses two-class Cross-Entropy. The compound-loss candidates use equal weighting (0.5 / 0.5) of Cross-Entropy and one overlap term:

- CE + soft Dice
- CE + soft IoU
- CE + Focal Tversky, with α = β = 0.5 and γ = 0.75

The ablation replaces focal Tversky (γ = 0.75) with standard Tversky (γ = 1.0). Checkpoints are selected by validation IoU. Test metrics are not used for candidate selection.

## Reported results

### Seed-42 held-out test comparison (pixel-aggregated)

| Metric | U-Net + CE | U-Net + CE + Focal Tversky |
|---|---:|---:|
| Precision | 0.86548 | 0.92036 |
| Recall | 0.94630 | 0.94078 |
| F1 | 0.90409 | 0.93046 |
| IoU | 0.82496 | 0.86995 |

### Three-seed test mean ± standard deviation (seeds 42, 43, 44)

| Metric | Baseline CE | Proposed CE + Focal Tversky |
|---|---:|---:|
| Precision | 0.89308 ± 0.02401 | 0.91123 ± 0.01230 |
| Recall | 0.94020 ± 0.00825 | 0.92044 ± 0.02048 |
| F1 | 0.91586 ± 0.01036 | 0.91573 ± 0.01276 |
| IoU | 0.84489 ± 0.01755 | 0.84472 ± 0.02186 |

The seed-42 run favours the proposed loss on F1 and IoU, but the three-seed means show essentially tied F1 and IoU. The project therefore does **not** claim robust multi-seed superiority.

## Environment and training

The recorded primary environment was Google Colab with a Tesla T4 GPU and PyTorch 2.14.1+cu130. The run configuration used AdamW, learning rate 0.001, weight decay 0.0001, batch size 8, up to 60 epochs, early stopping (patience 10), ReduceLROnPlateau, mixed precision and gradient clipping.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Dataset audit and split

Place the extracted PH² dataset in a local folder and run from the repository root:

```bash
python -m src.audit --data-root PATH_TO_PH2_DATASET_IMAGES --output-dir results/audit
python -m src.make_splits --manifest results/audit/pairs.csv --output data/splits/ph2_seed42.json --seed 42
```

The split-generation script in the final experiment version stratifies the cases into five lesion-area groups and checks for case overlap. The split JSON contains file paths from the machine on which it was generated; regenerate it from the local audit manifest on a new machine. The case allocation is reproducible from the same dataset and seed. Keep the case-ID assignment table (`ph2_seed42_details.csv`) as the recorded split evidence.

### Training

```bash
python -m src.train --config configs/experiments.yaml --experiment baseline_ce
python -m src.train --config configs/experiments.yaml --experiment candidate_ce_dice
python -m src.train --config configs/experiments.yaml --experiment candidate_ce_iou
python -m src.train --config configs/experiments.yaml --experiment candidate_focal_tversky
python -m src.train --config configs/experiments.yaml --experiment ablation_ce_tversky
```

### Evaluation

```bash
python -m src.evaluate --config configs/experiments.yaml --checkpoint models/checkpoints/baseline_ce_best.pt --experiment baseline_ce
python -m src.evaluate --config configs/experiments.yaml --checkpoint models/checkpoints/candidate_focal_tversky_best.pt --experiment proposed_focal_tversky
```

### Single-image Streamlit demo

Copy `candidate_focal_tversky_best.pt` into `models/checkpoints/` (and optionally the baseline checkpoint), then run:

```bash
streamlit run app.py
```

See `STREAMLIT_DEMO.md`. Model checkpoints and the dataset are deliberately kept outside ordinary Git commits. The app will not run until a checkpoint file is made available locally.

## Repository contents

- `src/`: dataset handling, U-Net, losses, metrics, training, evaluation, audit and split utilities.
- `configs/`: experiment configuration.
- `data/splits/`: split metadata (regenerate path-bearing manifests locally).
- `results/logs/`, `results/tables/`, `results/graphs/`: selected experiment evidence, when included.
- `app.py`: single-image segmentation interface.

## Limitations

- PH² is small (200 images); the held-out test set contains 30 images.
- Only three training seeds were evaluated; their mean IoU/F1 did not show a consistent proposed-loss advantage.
- Direct square resizing changes aspect ratio for non-square inputs and may affect boundary geometry.
- No clinical validation, diagnosis, malignancy classification or prospective deployment study was performed.
- Boundary-specific metrics and a systematic hyperparameter sweep should be added before making stronger claims about contour quality or tuning robustness.

## References

1. T. Mendonça et al., “PH² - A dermoscopic image database for research and benchmarking,” EMBC, 2013. doi: 10.1109/EMBC.2013.6610779.
2. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, 2015. doi: 10.1007/978-3-319-24574-4_28.
3. S. S. Salehi, D. Erdogmus, and A. Gholipour, “Tversky loss function for image segmentation using 3D fully convolutional deep networks,” arXiv:1706.05721, 2017.
4. N. Abraham and N. M. Khan, “A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation,” ISBI, 2019. doi: 10.1109/ISBI.2019.8759329.
