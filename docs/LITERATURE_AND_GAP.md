# Literature context and project gap

| Reference | Main idea | Relevance to this project | Scope limitation |
|---|---|---|---|
| Mendonça et al. (2013), PH2 database, DOI: 10.1109/EMBC.2013.6610779 | Dermoscopic image database with expert lesion annotations | Defines the dataset and annotation context | Dataset is small; results should not be generalized without external validation |
| Ronneberger et al. (2015), U-Net, DOI: 10.1007/978-3-319-24574-4_28 | Encoder-decoder CNN with skip connections | Provides the segmentation architecture | Original work does not establish performance on this specific PH2 split |
| Salehi et al. (2017), Tversky loss, arXiv:1706.05721 | Overlap objective balancing FP/FN contributions | Motivates overlap-aware loss terms | The chosen alpha=beta=0.5 setting is symmetric, not FP- or FN-weighted |
| Abraham & Khan (2019), Focal Tversky loss, DOI: 10.1109/ISBI.2019.8759329 | Focalized Tversky objective for lesion segmentation | Motivates testing a focal overlap term | This project uses a standard U-Net and does not reproduce their attention U-Net or experimental datasets |
| Ding et al. (2024), CTH-Net, iScience 27(4):109442, DOI: 10.1016/j.isci.2024.109442 | CNN-Transformer hybrid with multi-domain feature fusion and boundary refinement; reports PH² results including U-Net Jaccard 0.840 and CTH-Net 0.908 | Shows architectural approaches to context and boundary refinement, and supplies PH² literature context | The paper uses a different experimental protocol and substantially different architecture; its scores are contextual only and are not directly comparable to this project's fixed 140/30/30 split |

## Project question and bounded contribution

Does replacing half of the pixel-wise Cross-Entropy objective with a focalized, symmetric Tversky/Dice-like overlap term improve PH2 lesion-mask agreement for an otherwise unchanged two-class U-Net?

## Explicit research gap and scope

The cited literature includes architectural advances (U-Net variants, attention models and CNN-Transformer hybrids) and overlap-oriented objectives. These studies do not isolate the effect of the loss under this project's exact fixed PH² allocation and otherwise unchanged compact U-Net. The bounded engineering question is therefore: under one fixed 140/30/30 case allocation, what changes when half of CE is replaced by a focalized symmetric overlap term, with candidate and learning-rate selection based on validation IoU?

The contribution is a controlled loss comparison on one fixed PH² split, with loss screening, a standard-Tversky ablation, validation-based learning-rate selection, per-image error analysis and a three-seed check at the original learning rate. It is not a new architecture or a state-of-the-art benchmark. Reported results from Ding et al. use a different protocol and architecture and are therefore literature context, not a head-to-head score target. The three-seed results do not show a consistent mean F1/IoU advantage, and the test set was inspected repeatedly; therefore, no robust superiority or external generalization claim is made.
