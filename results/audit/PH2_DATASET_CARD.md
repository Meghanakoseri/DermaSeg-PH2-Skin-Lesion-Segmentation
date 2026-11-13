# PH2 Dataset Card — DermaSeg

## Task
Binary semantic segmentation of skin lesions in dermoscopic photographs.

## Dataset source
The project assignment identifies the PH2 dataset and provides this source:
https://www.kaggle.com/datasets/spacesurfer/ph2-dataset

## Dataset contents used
- Original dermoscopic photographs.
- Corresponding expert-provided lesion masks.
- ROI and colour-class masks were not used as lesion targets.

## Dataset audit
- Total image-mask pairs: 200
- Readable pairs: 200
- Missing/unreadable pairs: 0
- Image-mask dimension mismatches: 0
- Non-binary masks: 0
- Duplicate original-image hashes: 0
- Duplicate lesion-mask hashes: 0

## Original image dimensions
- Width range: 761 to 769 pixels.
- Height range: 553 to 577 pixels.
- Distinct observed dimension combinations: 24.

The assignment describes PH2 images as 768 x 560. The downloaded archive
contains some small per-image dimension variations; the actual observed
dimensions are documented in `raw_dimension_distribution.csv`.
Every corresponding image-mask pair has matching dimensions.

## Lesion mask characteristics
- Mask encoding: binary grayscale values 0 and 255.
- Minimum lesion area: 3.14%.
- Maximum lesion area: 98.31%.
- Mean lesion area: 32.24%.
- Median lesion area: 23.06%.

## Split protocol
A fixed split with seed 42 is used in all primary experiments:
- Training: 140 cases.
- Validation: 30 cases.
- Test: 30 cases.

The split is stratified using five lesion-area groups. No case is shared
between the three partitions.

## Preprocessing
Images are converted to RGB and resized to 256 x 256 using bilinear
interpolation. Binary masks are resized using nearest-neighbour interpolation.

## Intended use
Academic research and educational demonstration of binary lesion segmentation.

## Limitations and cautions
- PH2 is a small dataset with 200 cases.
- Performance on this dataset does not establish clinical validity or
  generalization to other populations, devices or acquisition conditions.
- The model is not a diagnostic system.
- Dataset redistribution and licensing conditions must be checked at the
  original source before redistribution.

## Files accompanying this card
- `eda_case_statistics.csv`
- `data_dictionary.csv`
- `raw_dimension_distribution.csv`
- `eda_summary.json`
