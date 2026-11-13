# Google Colab execution

1. Upload the project ZIP to Google Drive and extract it, or clone the private/public GitHub repository.
2. Set Runtime > Change runtime type > T4 GPU when available.
3. Confirm the GPU:

```python
!nvidia-smi
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

4. Install dependencies:

```python
%cd /content/DermaSeg
!pip install -r requirements.txt
```

5. Place/extract PH2 under `data/raw/PH2/`.
6. Audit the dataset before any training:

```python
!python -m src.audit --data-root data/raw/PH2 --output-dir results/audit
```

7. Inspect `results/audit/pairs.csv` and `audit_issues.txt`. Do not continue if matching issues exist.
8. Create the fixed split:

```python
!python -m src.make_splits --manifest results/audit/pairs.csv --output data/splits/ph2_seed42.json --seed 42
```

9. Run a short technical smoke test only to check code execution, then run the full configured experiments. Smoke-test results must never be reported as final results.
10. Copy checkpoints, logs, split manifest and final outputs to persistent Drive storage after each completed experiment.
