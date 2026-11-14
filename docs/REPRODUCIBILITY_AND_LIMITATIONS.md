# DermaSeg: reproducibility and protocol notes

## Frozen experiment protocol

- Dataset: PH², 200 image/lesion-mask pairs.
- Fixed split: 140 train / 30 validation / 30 test, seed 42, five lesion-area strata.
- Input: RGB 256x256; bilinear image resize; nearest-neighbour mask resize; masks mapped to {0,1}.
- Architecture: same two-class U-Net for CE baseline and compound-loss models.
- Selected LR: 0.0005 by validation IoU.
- Final selected seed-42 test metrics at LR 0.0005: CE P/R/F1/IoU = 0.90176/0.93964/0.92031/0.85238; CE + Focal Tversky = 0.90387/0.95104/0.92686/0.86369.
- Three-seed comparison at LR 0.001: baseline and proposed mean F1/IoU were effectively tied; robust multi-seed superiority was not demonstrated.

## Test-set-use disclosure

The same 30-case test split was examined in multiple exploratory runs: LR=0.001 comparisons, the three-seed comparison (42/43/44 at LR=0.001), and later selected LR=0.0005 comparisons. The learning-rate choice was made using validation IoU, not test results, and no gradient updates were performed using test images or labels. Nevertheless, repeated inspection means the current PH² test set is not an untouched one-time holdout. Current test metrics are exploratory comparative evidence. Do not use these results to tune thresholds, select models, or claim a fresh unbiased estimate. No further test evaluation should be run.

## Fresh LR=0.0005 reproduction (14 November 2026)

Both models were freshly retrained on the original Colab Tesla T4 with seed 42, existing 140/30 train/validation allocation, batch size 8, and up to 60 epochs. Both selected epoch 57. Best validation IoU: CE 0.8461204752; CE + Focal Tversky 0.8529104296. Best-checkpoint validation precision/recall/F1: CE 0.912588/0.920743/0.916647; proposed 0.927027/0.914295/0.920617. Training summaries: 122.24 s and 125.36 s. Checkpoint metadata and 60-row CSV logs were verified in the original Drive project. The training entry point loaded only train and validation records; the test split was not loaded in these reproduction jobs.

## Reproduction scripts and supporting analysis

- `python scripts/verify_split.py`: verifies 140/30/30 counts and no case overlap; no model evaluation.
- `python scripts/eda_ph2.py --data-root "PATH_TO_PH2_DATASET_IMAGES"`: regenerates dataset-card and EDA outputs from the actual audit pairs CSV.
- `python scripts/validation_robustness.py`: descriptive sensitivity analysis on the validation split using frozen checkpoints. Validation has already informed model selection; do not retune based on these results.
- `python scripts/predict_validation.py`: generates validation-only predictions.
- `python scripts/boundary_metrics_from_predictions.py --split validation ...`: computes macro boundary scores using 3x3 erosion and 2-pixel tolerance.
- `python scripts/benchmark_inference.py`: synthetic-input CPU inference timing on the local device; not a clinical-speed claim.
- `python scripts/capture_environment.py`: captures actual versions from the active environment. Keep Colab and laptop records separate.

The summary metrics and reproduction training plots are included in `results/reproduction_20261114_01_logs/`. The package includes the EDA dataset card, data dictionary, EDA summary and raw-dimension distribution. Historical logs, detailed result tables and figures are also retained in the public GitHub repository. The fresh LR=0.0005 reproduction summaries and plots are included in this package. The original Colab checkpoints and per-epoch logs remain in the Google Drive project and are not included here.

## Fixed split source requirement

The exact case-to-split membership CSV `data/splits/ph2_seed42_details.csv` is included and hash-verified against the existing public GitHub repository. Rebuild the path-bearing JSON manifest on each machine from the local audit CSV and this frozen membership file. Do not replace it with a new split and call it the reported allocation.

## Environment provenance

The original Colab capture was made after training in the still-connected runtime: Python 3.13.16, Tesla T4, PyTorch 2.14.1+cu130, CUDA 13.0. The local demo runtime was Windows 11, Python 3.12.4, CPU PyTorch 2.15.0+cpu, 8 logical CPUs, CUDA unavailable. Raw JSON/pip-freeze files were generated in the earlier workspaces but were not present in the starter ZIP used for this local bundle. No package-modification commands were found in the visible Colab notebook history; this does not prove there were no unrecorded external changes.

## Data access, licensing and privacy

PH² images/masks are not included. Follow the source provider and any mirror-specific terms. Do not add PH² image archives, masks, per-case predictions or model checkpoints to a public repository without confirming permission.

## Limitations

- Small, single dataset (200 images; 30 in the fixed test subset).
- Repeated test-set inspection weakens the independence of the reported test comparison.
- LR=0.0005 was freshly reproduced at seed 42 only; it was not repeated across multiple seeds.
- CUDA deterministic algorithms were not explicitly enabled, so bit-for-bit identical training is not guaranteed.
- No independent external dataset, clinical, prospective or subgroup-fairness validation was performed.
- Direct resizing to 256x256 changes aspect ratio for some images and may affect boundary geometry.
- The app is for research visualization, not diagnosis or medical decision-making.
