#!/usr/bin/env python3
"""Static audit of training/evaluation config consumption and fixed split assets."""
from pathlib import Path
import sys
import yaml
train_path=Path('src/train.py'); eval_path=Path('src/evaluate.py')
if not train_path.is_file() or not eval_path.is_file(): raise SystemExit('Run from project root: src/train.py/src/evaluate.py not found.')
train=train_path.read_text(encoding='utf-8'); evaluate=eval_path.read_text(encoding='utf-8')
config_paths=list(Path('configs').rglob('*.yaml')) if Path('configs').exists() else []
config_docs=[]
for path in config_paths:
    try:
        config_docs.append((path, yaml.safe_load(path.read_text(encoding='utf-8')) or {}))
    except Exception as exc:
        raise SystemExit(f'Could not parse {path}: {exc}')
issues=[]
checks = [
    ('gradient clipping', 'gradient_clip_norm' in train and 'clip_grad_norm_(model.parameters(), gradient_clip_norm)' in train),
    ('optimizer setting', 'config["training"].get("optimizer"' in train and 'optimizer_name != "adamw"' in train),
    ('training pin_memory setting', 'config["data"].get("pin_memory"' in train),
    ('evaluation pin_memory setting', 'config["data"].get("pin_memory"' in evaluate),
]
for label, ok in checks:
    if not ok: issues.append(f'{label} is not consumed from YAML as documented; follow README_PATCH_NOTES.md patch steps.')
for path, config in config_docs:
    if not isinstance(config, dict):
        continue
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    project_cfg = config.get('project', {})
    evaluation_cfg = config.get('evaluation', {})
    if isinstance(data_cfg, dict):
        for dead_key in ['use_cache', 'root']:
            if dead_key in data_cfg:
                issues.append(f'{path}: unused data.{dead_key} remains; the current pipeline uses the split manifest and has no dataset cache.')
        for key in ['split_manifest', 'batch_size']:
            if key in data_cfg and f'config["data"]["{key}"]' not in train and f'config["data"]["{key}"]' not in evaluate:
                issues.append(f'{path}: data.{key} is not consumed by the training/evaluation code.')
        if 'num_workers' in data_cfg and 'config["data"]["num_workers"]' not in train:
            issues.append(f'{path}: data.num_workers is not consumed by src/train.py.')
    if isinstance(training_cfg, dict):
        for key in ['learning_rate','weight_decay','epochs','patience','scheduler_factor','scheduler_patience','min_learning_rate','amp','checkpoint_dir','log_dir']:
            if key in training_cfg and f'config["training"]["{key}"]' not in train:
                issues.append(f'{path}: training.{key} is not consumed by src/train.py.')
        if 'gradient_clip_norm' in training_cfg and 'config["training"].get("gradient_clip_norm"' not in train:
            issues.append(f'{path}: training.gradient_clip_norm is not consumed by src/train.py.')
    if isinstance(project_cfg, dict):
        for key in ['seed','image_size','in_channels','num_classes']:
            if key in project_cfg and f'config["project"]["{key}"]' not in train:
                issues.append(f'{path}: project.{key} is not consumed by src/train.py.')
    if isinstance(evaluation_cfg, dict):
        for key in ['threshold','output_dir']:
            if key in evaluation_cfg and f'config["evaluation"]["{key}"]' not in evaluate:
                issues.append(f'{path}: evaluation.{key} is not consumed by src/evaluate.py.')
if not Path('data/splits/ph2_seed42_details.csv').is_file(): issues.append('Committed case/split allocation CSV is missing.')
if not Path('scripts/create_manifest_from_split_details.py').is_file(): issues.append('Exact split-manifest reconstruction script is missing.')
if issues:
    print('CONFIG / REPRODUCIBILITY WARNINGS:')
    for issue in issues: print('- '+issue)
    sys.exit(1)
print('PASS: optimizer, gradient clipping, pin_memory, training settings and split-reconstruction assets are consistent.')
print('Note: use create_manifest_from_split_details.py to recreate the exact published case allocation; do not draw a new split for reported results.')
