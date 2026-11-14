# PH2 dataset card and EDA interpretation

## Dataset and target
- Observational unit: one dermoscopic image and its paired binary lesion mask.
- Cases audited: 200.
- Target: foreground lesion pixels (mask value 255); background pixels (0).
- Raw dimensions: width 761–769 px; height 553–577 px; 24 observed dimension combinations.
- All pairs were readable, dimension-matched and binary in this run.

## Descriptive EDA findings
- Lesion area ranges from 3.14% to 98.31% of the mask, with mean 32.24% and median 23.06% (sample SD 23.43 percentage points).
- The wide lesion-area range indicates substantial variation in foreground size. This motivates reporting overlap metrics (F1/Dice and IoU) in addition to pixel-wise precision and recall; it also supports the previously defined lesion-area stratification for the split.
- Raw image dimensions vary slightly. Images and masks are resized together to 256×256, using bilinear interpolation for RGB images and nearest-neighbour interpolation for masks so the discrete mask labels are preserved.
- The RGB intensity histogram describes image-level average channel brightness/colour. Spearman correlations between lesion area, raw dimensions and image-level RGB means are included as descriptive checks; they are not used for feature selection because the CNN is trained directly on the RGB image.
- No exact duplicate source-image files were detected by SHA-256 hashing in this run.

## Split interpretation
The split-level boxplot checks whether lesion-area distributions are broadly comparable across partitions. Modeling-oriented distributions and summary statistics are separately calculated from training cases only; validation/test labels are not used to guide modeling or preprocessing decisions.

## Source, access and licence note
The original PH² Browser tutorial states that PH² is available for research and educational purposes. The published PH² dataset license notice also states that redistribution and commercial use are not allowed. The Kaggle mirror's own terms were not independently verified; do not redistribute its archive or PH² images/masks. Cite the original PH² paper and follow the original provider terms.

## Data protection and intended use
Only de-identified dataset case IDs and relative paths are recorded by this script; absolute machine-specific paths are not written to the EDA case table. This is an academic segmentation dataset. The model is not a diagnostic or clinical decision tool.

## Outputs
- `eda_summary.json`
- `eda_case_statistics.csv`
- `data_dictionary.csv`
- `raw_dimension_distribution.csv`
- `results/graphs/eda_lesion_area_distribution.png`
- `results/graphs/eda_raw_dimensions.png`
- `results/graphs/eda_rgb_intensity_means.png`
- `eda_spearman_correlations.csv`
- `results/graphs/eda_feature_correlations.png`
- Optional split-level summary and plot when split details are supplied.
