# DermaSeg - PH² Skin Lesion Segmentation

A PyTorch U-Net pipeline for pixel-wise binary skin-lesion segmentation using the PH² dermoscopic image dataset. The project compares a Cross-Entropy (CE) baseline with compound overlap losses, performs validation-based model selection, studies multi-seed variation, and provides a Streamlit inference demo.

> **Scope:** this is an academic segmentation experiment, not a diagnostic classifier, skin-cancer detector, medical device, or substitute for clinical assessment.

## Project question

With the U-Net architecture and PH² case allocation held fixed, does replacing the CE objective with a compound CE + Focal Tversky objective improve lesion-mask overlap?

## Dataset and protocol

- PH²: 200 dermoscopic images and corresponding lesion masks; ROI masks were excluded as targets.
- Audit: 200 readable matched image-mask pairs; 0 missing/invalid pairs; 0 dimension mismatches; all lesion masks binary (`{0,255}`). Raw widths ranged from 761-769 px and heights from 553-577 px.
- Preprocessing: RGB image and grayscale mask resized to 256 x 256; bilinear interpolation for images and nearest-neighbour interpolation for masks; image values scaled to `[0,1]`; masks mapped to classes `{0,1}`.
- Augmentation: random horizontal and vertical flips on training samples only; no validation/test augmentation.
- Split: fixed seed-42, area-stratified 140 train / 30 validation / 30 test. Five lesion-area bins each contributed 28/6/6 cases.
- The PH² dataset is not included. Obtain it from the dataset provider or the assignment's acquisition source and follow its reuse terms. Do not commit raw images or masks.

**Exact split evidence:** `data/splits/ph2_seed42_details.csv` is included in this package and records all 200 case-to-split assignments with foreground fractions and area bins. It was restored from the existing public GitHub repository and verified against the repository blob hash. Do not substitute `src.make_splits` for this reported allocation: it creates a new split.

## Model and loss functions

Both variants use the same two-class U-Net: RGB input, four encoder levels (32/64/128/256 channels), 512-channel bottleneck, decoder with transposed convolutions and skip connections, and two output logits.

Baseline:

`L = L_CE`

Proposed:

`L = 0.5 * L_CE + 0.5 * (1 - T)^0.75`

`T = (TP + eps) / (TP + 0.5*FP + 0.5*FN + eps)`

The implementation uses `alpha=beta=0.5`, `gamma=0.75`, and `eps=1e-6`. Since alpha and beta are equal, the Tversky term weights false positives and false negatives symmetrically; the focal exponent modulates the overlap penalty.

## Results

### Final selected seed-42 test comparison (LR=0.0005; pixel-aggregated)

| Metric | U-Net + CE | U-Net + CE + Focal Tversky |
|---|---:|---:|
| Precision | 0.90176 | 0.90387 |
| Recall | 0.93964 | 0.95104 |
| F1 | 0.92031 | 0.92686 |
| IoU | 0.85238 | 0.86369 |

**Frozen-test disclosure:** the same 30-case test split was examined in multiple exploratory evaluations (including LR=0.001 runs across seeds 42/43/44 and the later selected LR=0.0005 run). These are exploratory comparative results, not a pristine one-time holdout estimate. The test split must not be evaluated or used for further tuning again.

### Three-seed test results (LR=0.001; mean +/- SD)

| Metric | Baseline CE | Proposed |
|---|---:|---:|
| Precision | 0.89308 +/- 0.02401 | 0.91123 +/- 0.01230 |
| Recall | 0.94020 +/- 0.00825 | 0.92044 +/- 0.02048 |
| F1 | 0.91586 +/- 0.01036 | 0.91573 +/- 0.01276 |
| IoU | 0.84489 +/- 0.01755 | 0.84472 +/- 0.02186 |

The three-seed mean F1 and IoU are effectively tied. The selected single-seed gain therefore does not establish robust multi-seed superiority.

### Fresh LR=0.0005 reproduction (validation only; 14 Nov 2026)

The two selected models were freshly retrained on the original Colab Tesla T4 using seed 42, the same fixed manifest, LR=0.0005, batch size 8 and up to 60 epochs. Both completed 60 epochs and selected epoch 57. Checkpoints and CSV logs were verified; the saved checkpoint epoch and validation IoU matched the maximum-IoU rows in the logs.

| Validation metric at best checkpoint | CE baseline | CE + Focal Tversky |
|---|---:|---:|
| Precision | 0.912588 | 0.927027 |
| Recall | 0.920743 | 0.914295 |
| F1 | 0.916647 | 0.920617 |
| IoU | 0.846120 | 0.852910 |

Training summaries: 122.24 s (CE) and 125.36 s (proposed). The training entry point loads only train and validation splits; the test split was not loaded in these reproduction runs. This confirms repeatability of the selected training/validation pipeline for seed 42, not external generalization or multi-seed replication at LR=0.0005.

The comparison CSV/JSON and plots are included under `results/reproduction_20261114_01_logs/`. The Colab checkpoints and original full per-epoch CSV logs remain in the original Drive project and are not bundled here.

### Other supplementary analyses

- Initial validation loss screening at LR=0.001: CE IoU 0.81042; CE+Dice 0.81482; CE+IoU 0.81592; CE+Focal Tversky 0.81595.
- Standard-Tversky ablation (gamma=1.0): validation IoU 0.81008; focal Tversky (gamma=0.75): 0.81595.
- Validation LR sweep: baseline/proposed IoU at LR=0.001: 0.81042/0.81595; LR=0.0005: 0.84612/0.85291; LR=0.002: 0.81734/0.81384.
- Validation-only perturbation sensitivity: proposed was higher in four of six conditions; baseline was slightly higher under Gaussian noise and translation. Because validation informed model selection, this is descriptive sensitivity analysis, not independent generalization evidence.
- Validation boundary metrics (macro mean +/- SD; 30 images; 3x3 inner-boundary erosion; 2-pixel tolerance): CE boundary P/R/F1 0.51287 +/- 0.30236 / 0.53785 +/- 0.26888 / 0.51802 +/- 0.28742; proposed 0.53913 +/- 0.34028 / 0.55929 +/- 0.31071 / 0.54288 +/- 0.32817.
- Synthetic CPU benchmark on the local Windows laptop (2 Torch threads, 10 warm-ups, 100 iterations): median latency 643.83 ms CE vs 604.58 ms proposed; P95 746.25 ms vs 716.71 ms. This sequential device-specific benchmark does not establish a meaningful speed advantage.

## Reproduction setup

### Requirements

Python 3.10+. Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

A CUDA GPU is recommended for full training. A CPU-only laptop can run code checks and the Streamlit demo, but training will be slower.

### Audit dataset and reconstruct exact split

1. Extract PH² so that each case folder contains the original dermoscopic image and lesion mask.
2. Run the audit:

```bash
python -m src.audit --data-root "PATH_TO_PH2_DATASET_IMAGES" --output-dir results/audit
```

3. The exact `ph2_seed42_details.csv` is already included in `data/splits/`. Reconstruct machine-specific image paths without changing case membership:

```bash
python scripts/create_manifest_from_split_details.py --audit results/audit/pairs.csv --details data/splits/ph2_seed42_details.csv --output data/splits/ph2_seed42.json
python scripts/verify_split.py --manifest data/splits/ph2_seed42.json --details data/splits/ph2_seed42_details.csv --check-paths
```

Do not proceed if audit or split verification reports missing pairs, overlaps, or path failures.

### Train the selected pair

The reproduction configuration is `configs/reproduction/config_reproduction_20261114_01.yaml`:

```bash
python -m src.train --config configs/reproduction/config_reproduction_20261114_01.yaml --experiment reproduction_ce_lr_0005
python -m src.train --config configs/reproduction/config_reproduction_20261114_01.yaml --experiment reproduction_focal_tversky_lr_0005
```

Outputs are configured to separate reproduction checkpoint and log folders. This training entry point uses train and validation records only. **Do not run `src.evaluate` on the PH² test split again.** The test results above are historical and must remain frozen.

### Run static checks

```bash
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/check_config_consistency.py
```

`check_config_consistency.py` checks for the exact split-membership CSV; it passes when run from this package root.

## Streamlit demo

The redesigned app is in `app.py`. See `STREAMLIT_DEMO.md`. Model checkpoint files are intentionally not committed to Git. Copy an authorized checkpoint from your Drive into the path expected by the demo before running:

```bash
streamlit run app.py
```

The demo is for research visualization only and is not a diagnostic tool.

## Environment provenance

- Original Colab training runtime capture: Python 3.13.16, Tesla T4, PyTorch 2.14.1+cu130, CUDA 13.0. The capture was made after training in the still-connected original runtime; it is supporting provenance, not a guaranteed snapshot of every historical run.
- Local Windows demo runtime capture: Python 3.12.4, PyTorch 2.15.0+cpu, 8 logical CPUs, CUDA unavailable.
- The JSON and pip-freeze files were generated in the original workspaces but were not present in the starter ZIP used to assemble this local snapshot. Copy those raw files into `results/audit/` from the original workspaces if required. Keep Colab and laptop dependency locks separate; do not combine them into one cross-platform lock file.

## Limitations

- 200-image dataset and 30-image test split.
- Test split was examined multiple times; the test metrics are not a pristine one-time estimate.
- Only three training seeds were used in the LR=0.001 multi-seed study; the LR=0.0005 pair was freshly reproduced at seed 42 only.
- CUDA deterministic algorithms were not explicitly enabled, so bit-for-bit identical training is not guaranteed.
- No independent external-dataset, clinical, prospective or subgroup-fairness validation was performed.
- Direct 256x256 resizing changes aspect ratio for some images and may affect boundary geometry.

## References

1. T. Mendonça et al., “PH² - A dermoscopic image database for research and benchmarking,” EMBC, 2013. doi: 10.1109/EMBC.2013.6610779.
2. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, 2015. doi: 10.1007/978-3-319-24574-4_28.
3. S. S. Salehi, D. Erdogmus, and A. Gholipour, “Tversky loss function for image segmentation using 3D fully convolutional deep networks,” arXiv:1706.05721, 2017.
4. N. Abraham and N. M. Khan, “A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation,” ISBI, 2019. doi: 10.1109/ISBI.2019.8759329.
