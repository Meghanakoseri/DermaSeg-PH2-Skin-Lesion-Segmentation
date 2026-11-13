# DermaSeg — PH² Skin Lesion Segmentation

A PyTorch U-Net pipeline for binary skin-lesion segmentation in dermoscopic images. This project compares a Cross-Entropy baseline with compound segmentation losses, performs loss ablation and learning-rate experiments, and evaluates segmentation performance using pixel-level and boundary-level metrics.

> **Important:** This is an academic image-segmentation demonstration. It is not a skin-cancer classifier, diagnostic system, medical device, or substitute for professional medical care.

## 1. Dataset and preprocessing

- Dataset: PH², containing 200 dermoscopic images with corresponding binary lesion masks.
- Dataset audit: 200 readable image-mask pairs; no missing pairs or dimension mismatches; all masks contain binary values (0 and 255).
- Original image dimensions vary slightly across cases.
- Input images are resized to 256 × 256.
- Images use bilinear interpolation; masks use nearest-neighbour interpolation.
- Images are converted to RGB and scaled to [0, 1]. Masks are converted to classes 0 and 1.

### Fixed data split

| Split | Images |
|---|---:|
| Training | 140 |
| Validation | 30 |
| Test | 30 |
| Total | 200 |

The split uses seed 42 and stratification across five lesion-area groups. The same case allocation was used for the experiments.

The PH² dataset is not included in this repository. Obtain it from the dataset provider or the source specified in the project report, and follow its access and reuse terms.

## 2. Model and loss functions

The model is a U-Net that accepts RGB images and predicts two pixel classes: background and lesion.

### Baseline

The baseline uses two-class Cross-Entropy (CE) loss.

### Compound-loss candidates

The following combinations were evaluated:

- CE + soft Dice loss
- CE + soft IoU loss
- CE + Focal Tversky loss

The proposed compound loss is:

L = 0.5 × L_CE + 0.5 × L_FocalTversky

The Tversky index is:

T = (TP + ε) / (TP + 0.5 FP + 0.5 FN + ε)

The focal Tversky term is:

L_FocalTversky = (1 - T)^0.75

Here, TP, FP and FN represent true-positive, false-positive and false-negative soft pixel counts. The implementation computes the overlap term per image and averages it over the batch.

### Ablation

The ablation replaces the focal exponent γ = 0.75 with standard Tversky loss, γ = 1.0.

Checkpoints are selected using validation IoU.

## 3. Learning-rate tuning

Learning rates 0.001, 0.0005 and 0.002 were compared using validation performance.

| Learning rate | Baseline validation IoU | Proposed validation IoU |
|---:|---:|---:|
| 0.001 | 0.81042 | 0.81595 |
| 0.0005 | 0.84612 | 0.85291 |
| 0.002 | 0.81734 | 0.81384 |

The learning rate 0.0005 was selected based on validation IoU.

## 4. Final selected test results

The selected seed-42 models trained with learning rate 0.0005 were evaluated on the 30-case test split.

| Metric | U-Net + CE | U-Net + CE + Focal Tversky |
|---|---:|---:|
| Precision | 0.90176 | 0.90387 |
| Recall | 0.93964 | 0.95104 |
| F1-score | 0.92031 | 0.92686 |
| IoU | 0.85238 | 0.86369 |

For this selected run, the proposed model achieved higher precision, recall, F1 and IoU than the baseline.

### Three-seed comparison at learning rate 0.001

The following separate experiment evaluated seeds 42, 43 and 44 using the original learning rate of 0.001.

| Metric | Baseline CE | CE + Focal Tversky |
|---|---:|---:|
| Precision | 0.89308 ± 0.02401 | 0.91123 ± 0.01230 |
| Recall | 0.94020 ± 0.00825 | 0.92044 ± 0.02048 |
| F1-score | 0.91586 ± 0.01036 | 0.91573 ± 0.01276 |
| IoU | 0.84489 ± 0.01755 | 0.84472 ± 0.02186 |

The three-seed results show essentially tied mean F1 and IoU. Therefore, the project does not claim that the proposed loss has robust multi-seed superiority.

**Test-set evaluation note:** The same 30-case test split was evaluated in earlier experiments, including the three-seed comparison, before the final learning-rate experiment. Learning-rate selection was based on validation IoU, not test metrics. However, the final test comparison should be interpreted as exploratory rather than as a completely untouched, one-time test evaluation.

## 5. Boundary evaluation

An additional boundary-based evaluation was performed for the selected learning-rate-0.0005 models.

The evaluation uses an inner boundary extracted with 3 × 3 erosion and a tolerance of 2 pixels. Scores are macro-averaged across the 30 test images.

| Metric | Baseline CE | CE + Focal Tversky |
|---|---:|---:|
| Boundary precision | 0.41722 ± 0.19417 | 0.45269 ± 0.23576 |
| Boundary recall | 0.43766 ± 0.20214 | 0.47514 ± 0.22272 |
| Boundary F1 | 0.42384 ± 0.19163 | 0.46055 ± 0.22687 |

These are supplementary boundary metrics. They indicate better average boundary scores for the proposed model in this experiment, but the relatively large standard deviations show variation across images.

## 6. Training environment and efficiency

The primary training environment was Google Colab with a Tesla T4 GPU and PyTorch 2.14.1+cu130.

The training configuration included:

- Optimizer: AdamW
- Batch size: 8
- Epoch limit: 60
- Early stopping patience: 10
- Weight decay: 0.0001
- Learning-rate scheduler: ReduceLROnPlateau
- Mixed precision and gradient clipping

For the selected learning-rate-0.0005 runs:

| Detail | Baseline CE | CE + Focal Tversky |
|---|---:|---:|
| Parameters | 7,763,074 | 7,763,074 |
| Training time | 124.20 s | 123.68 s |
| Checkpoint size | ~29.68 MB | ~29.68 MB |
| Best validation IoU | 0.84612 | 0.85291 |

Both models have the same architecture and parameter count. The proposed loss does not provide a meaningful model-size or training-time advantage.

## 7. Reproduction instructions

### Install dependencies

```bash
pip install -r requirements.txt
```

### Dataset audit

Place the extracted PH² dataset on your machine. Run the audit from the repository root:

```bash
python -m src.audit --data-root "PATH_TO_PH2_DATASET_IMAGES" --output-dir results/audit
```

Replace the placeholder with the actual path to the extracted PH² image folders.

### Generate the split

```bash
python -m src.make_splits --manifest results/audit/pairs.csv --output data/splits/ph2_seed42.json --seed 42
```

The generated JSON manifest contains machine-specific paths. Regenerate it on each new machine using the local audit output. The case-ID allocation evidence is stored in `data/splits/ph2_seed42_details.csv`.

### Train the selected models

```bash
python -m src.train --config configs/learning_rate_tuning/config_lr_0005.yaml --experiment tune_ce_lr_0005
```

```bash
python -m src.train --config configs/learning_rate_tuning/config_lr_0005.yaml --experiment tune_focal_tversky_lr_0005
```

### Evaluate the selected checkpoints

```bash
python -m src.evaluate --config configs/learning_rate_tuning/config_lr_0005.yaml --checkpoint models/checkpoints/tune_ce_lr_0005_best.pt --experiment tune_ce_lr_0005
```

```bash
python -m src.evaluate --config configs/learning_rate_tuning/config_lr_0005.yaml --checkpoint models/checkpoints/tune_focal_tversky_lr_0005_best.pt --experiment tune_focal_tversky_lr_0005
```

Evaluation uses the test split. Do not use test results for model or hyperparameter selection.

## 8. Run the Streamlit demo

The application supports single-image segmentation.

Place the trained checkpoints in `models/checkpoints/`:

- `tune_focal_tversky_lr_0005_best.pt`
- `tune_ce_lr_0005_best.pt`

Then install dependencies and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The interface allows a user to upload an image, view the predicted binary lesion mask and overlay, inspect the predicted foreground percentage, and download the mask as a PNG file.

The checkpoint files are not stored in ordinary Git history. The app requires the checkpoint files to be available locally.

See `STREAMLIT_DEMO.md` for details.

## 9. Repository structure

- `src/`: dataset handling, U-Net model, loss functions, metrics, training, evaluation and audit utilities.
- `configs/`: experiment and learning-rate configurations.
- `data/splits/`: split metadata and case-ID allocation.
- `models/checkpoints/`: local model checkpoint location; trained weights are excluded from Git.
- `results/audit/`: dataset audit and data-card information.
- `results/logs/`: training logs and summaries.
- `results/tables/`: metric tables and experiment comparisons.
- `results/graphs/`: generated result visualizations.
- `app.py`: Streamlit inference application.

## 10. Limitations

- PH² contains only 200 images, with 30 cases in the test split.
- The three-seed comparison does not show consistent superiority of the proposed loss.
- The final learning-rate-0.0005 test comparison is exploratory because the same test split was examined in earlier experiments.
- Direct square resizing changes the aspect ratio of non-square images and may affect boundary geometry.
- Boundary scores vary substantially across images.
- Performance on arbitrary cameras, image sources and acquisition conditions has not been established.
- No clinical validation, diagnosis, malignancy classification or prospective deployment study was performed.

## References

1. T. Mendonça et al., “PH² - A dermoscopic image database for research and benchmarking,” EMBC, 2013. doi: 10.1109/EMBC.2013.6610779.
2. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, 2015. doi: 10.1007/978-3-319-24574-4_28.
3. S. S. Salehi, D. Erdogmus, and A. Gholipour, “Tversky loss function for image segmentation using 3D fully convolutional deep networks,” arXiv:1706.05721, 2017.
4. N. Abraham and N. M. Khan, “A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation,” ISBI, 2019. doi: 10.1109/ISBI.2019.8759329.
