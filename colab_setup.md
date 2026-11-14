# Google Colab execution and reproduction

The final reported results use the existing PH² seed-42 case allocation. **Do not create a new random split for reproducing the reported comparison.** The exact case-membership CSV is included at `data/splits/ph2_seed42_details.csv`.

## 1. Open the project and select a GPU

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/DermaSeg_Work/DermaSeg_Project
!nvidia-smi
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
```

## 2. Verify dataset and exact split assets

Extract PH² to the location used in the original manifest or rebuild machine-specific paths from the exact case-membership CSV. Audit the image/mask pairing:

```bash
python -m src.audit --data-root "PATH_TO_PH2_DATASET_IMAGES" --output-dir results/audit
```

Rebuild the machine-specific manifest from the included exact membership CSV without changing case allocation:

```bash
python scripts/create_manifest_from_split_details.py --audit results/audit/pairs.csv --details data/splits/ph2_seed42_details.csv --output data/splits/ph2_seed42.json
python scripts/verify_split.py --manifest data/splits/ph2_seed42.json --details data/splits/ph2_seed42_details.csv --check-paths
```

Confirm counts 140/30/30, zero overlap, five area bins with 28/6/6 cases, and that every image/mask path exists. Do not use `src.make_splits` for the reported results; that would draw a new split.

## 3. Reproduce the selected LR=0.0005 runs (train/validation only)

The archived reproduction config is `configs/reproduction/config_reproduction_20261114_01.yaml`. It has isolated checkpoint/log folders and separate experiment names:

```bash
python -m src.train --config configs/reproduction/config_reproduction_20261114_01.yaml --experiment reproduction_ce_lr_0005
python -m src.train --config configs/reproduction/config_reproduction_20261114_01.yaml --experiment reproduction_focal_tversky_lr_0005
```

The training entry point loads only `train` and `validation` records. It does not load the test split. The original Colab Drive copy contains the verified checkpoint and per-epoch logs. The local release includes the comparison CSV/JSON and training curves, not the checkpoint binaries.

**Do not evaluate or tune on the PH² test split again.** Its historical metrics were examined in multiple exploratory experiments and that limitation is explicitly disclosed in the report.

## 4. Run checks

```bash
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
python scripts/check_config_consistency.py
```

The consistency checker should pass with the included split-membership CSV. If it does not, stop and inspect the reported path or configuration mismatch; do not bypass the exact split requirement.
