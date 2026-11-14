# Fresh reproduction evidence (14 November 2026)

This folder contains the validation-IoU and training-loss plots, plus a comparison summary transcribed from the user's completed Colab runs and artifact-verification output.

## Verified runs

| Item | CE baseline | CE + Focal Tversky |
|---|---:|---:|
| Best epoch | 57 | 57 |
| Best validation IoU | 0.8461204752 | 0.8529104296 |
| Validation precision | 0.912588 | 0.927027 |
| Validation recall | 0.920743 | 0.914295 |
| Validation F1 | 0.916647 | 0.920617 |
| Training summary duration | 122.24 s | 125.36 s |
| Completed epochs | 60 | 60 |
| Checkpoint verified | Yes | Yes |
| 60-row CSV log verified | Yes | Yes |

Both jobs used seed 42, the existing 140/30/30 split manifest, LR=0.0005, batch size 8 and the original Colab Tesla T4. The training entry point loaded only train and validation records. No test split was loaded or evaluated in these fresh reproduction runs.

## Files present here

- `reproduction_comparison.csv` and `.json` — best validation metrics from both runs.
- `reproduction_validation_iou.png` — per-epoch validation IoU curves.
- `reproduction_training_loss.png` — per-epoch training loss curves.

The original 60-row per-epoch CSVs and checkpoint binaries were verified in the original Google Drive project and remain there; they are not included in this local archive. Do not claim they are bundled in this directory.

The best validation IoU values match the earlier LR=0.0005 tuning results to reported precision. This is a single-seed reproduction check, not an external validation or a multi-seed rerun at LR=0.0005. Do not run another PH² test evaluation.
