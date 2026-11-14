param(
  [Parameter(Mandatory=$true)][string]$DataRoot
)
$ErrorActionPreference = 'Stop'
Write-Host '1/6 - Dataset audit (data quality only; no model/test metric evaluation)' -ForegroundColor Cyan
python -m src.audit --data-root $DataRoot --output-dir results/audit
if ($LASTEXITCODE -ne 0) { throw 'Dataset audit failed.' }
Write-Host '2/6 - Rebuild exact fixed manifest from committed split membership (does not draw a new split)' -ForegroundColor Cyan
python scripts/create_manifest_from_split_details.py --audit results/audit/pairs.csv --details data/splits/ph2_seed42_details.csv --output data/splits/ph2_seed42.json
if ($LASTEXITCODE -ne 0) { throw 'Manifest reconstruction failed.' }
Write-Host '3/6 - Verify counts, case isolation, strata and local paths' -ForegroundColor Cyan
python scripts/verify_split.py --manifest data/splits/ph2_seed42.json --details data/splits/ph2_seed42_details.csv --check-paths
if ($LASTEXITCODE -ne 0) { throw 'Split verification failed.' }
Write-Host '4/6 - Regenerate descriptive EDA' -ForegroundColor Cyan
python scripts/eda_ph2.py --data-root $DataRoot --manifest results/audit/pairs.csv --output-dir results/audit/eda --split-details data/splits/ph2_seed42_details.csv
if ($LASTEXITCODE -ne 0) { throw 'EDA generation failed.' }
Write-Host '5/6 - Check training code/config consistency' -ForegroundColor Cyan
python scripts/check_config_consistency.py
if ($LASTEXITCODE -ne 0) { throw 'Config/code mismatch detected. Read docs/CODE_CONSISTENCY_PATCH.md.' }
Write-Host '6/6 - Capture this active environment; this is NOT a reconstruction of a past Colab runtime unless run inside that same runtime.' -ForegroundColor Cyan
python scripts/capture_environment.py --name local_demo
if ($LASTEXITCODE -ne 0) { throw 'Environment capture failed.' }
Write-Host 'Done. No model training or test-set model evaluation was performed.' -ForegroundColor Green
Write-Host 'After these checks, run validation-only robustness, validation boundary and synthetic latency scripts separately.' -ForegroundColor Yellow
