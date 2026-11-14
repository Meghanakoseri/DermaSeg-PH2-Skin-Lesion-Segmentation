#!/usr/bin/env python3
"""Validation-only input corruption sensitivity analysis for two saved models.

This script never loads the test split. It evaluates a frozen checkpoint on the
validation split under deterministic synthetic perturbations. Do not use these
results to choose checkpoints, thresholds or hyperparameters; report them as
exploratory sensitivity analysis only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import PH2Dataset, load_split
from src.metrics import confusion_counts, metrics_from_counts
from src.model import UNet
from src.utils import get_device, load_config


def load_model(path: Path, config: dict, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = UNet(config["project"]["in_channels"], config["project"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, ckpt


def corruptions(x: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    # x in [0,1], shape B,C,H,W. The same deterministic noise is used for
    # both models on the same validation batch.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    noise = torch.randn_like(x) * 0.03
    bright_down = (x * 0.8).clamp(0, 1)
    bright_up = (x * 1.2).clamp(0, 1)
    noisy = (x + noise).clamp(0, 1)
    blurred = F.avg_pool2d(F.pad(x, (1,1,1,1), mode="reflect"), kernel_size=3, stride=1)
    # Translate content +4 px right and +4 px down, with reflection padding.
    shifted = F.pad(x, (4,0,4,0), mode="reflect")[:, :, :x.shape[2], :x.shape[3]]
    return {"clean": x, "brightness_0.8": bright_down, "brightness_1.2": bright_up,
            "gaussian_noise_std_0.03": noisy, "3x3_mean_blur": blurred,
            "translation_4px": shifted}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/learning_rate_tuning/config_lr_0005.yaml")
    p.add_argument("--baseline", default="models/checkpoints/tune_ce_lr_0005_best.pt")
    p.add_argument("--proposed", default="models/checkpoints/tune_focal_tversky_lr_0005_best.pt")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", default="results/robustness_validation")
    args = p.parse_args()
    config = load_config(args.config)
    device = get_device()
    split = load_split(config["data"]["split_manifest"], "validation")
    ds = PH2Dataset(split, config["project"]["image_size"], augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    models = {"baseline_ce": load_model(Path(args.baseline), config, device)[0],
              "proposed_focal_tversky": load_model(Path(args.proposed), config, device)[0]}
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, model in models.items():
        for condition in ["clean", "brightness_0.8", "brightness_1.2", "gaussian_noise_std_0.03", "3x3_mean_blur", "translation_4px"]:
            totals = [0, 0, 0, 0]
            for batch_index, (images, masks, _) in enumerate(dl):
                images = images.to(device); masks = masks.to(device)
                x = corruptions(images, seed=2026 + batch_index)[condition]
                with torch.inference_mode():
                    logits = model(x)
                    pred = torch.softmax(logits, dim=1)[:, 1] >= 0.5
                c = confusion_counts(pred, masks)
                totals = [a + b for a, b in zip(totals, c)]
            scores = metrics_from_counts(*totals)
            rows.append({"model": model_name, "condition": condition,
                         "precision": scores["precision"], "recall": scores["recall"],
                         "f1": scores["f1"], "iou": scores["iou"],
                         "tp": scores["tp"], "fp": scores["fp"], "fn": scores["fn"]})
    df = pd.DataFrame(rows); df.to_csv(output / "validation_corruption_sensitivity.csv", index=False)
    for metric in ["iou", "f1"]:
        pivot = df.pivot(index="condition", columns="model", values=metric)
        ax = pivot.plot(kind="bar", figsize=(9,4.5))
        ax.set_ylabel(metric.upper()); ax.set_ylim(0,1); ax.set_title(f"Validation sensitivity: {metric.upper()}")
        ax.tick_params(axis="x", rotation=25); plt.tight_layout()
        plt.savefig(output / f"validation_sensitivity_{metric}.png", dpi=220); plt.close()
    metadata = {"split":"validation", "n_images":len(split), "threshold":0.5,
        "corruptions":{"brightness_0.8":"multiply RGB values by 0.8", "brightness_1.2":"multiply RGB values by 1.2",
        "gaussian_noise_std_0.03":"add Gaussian noise with standard deviation 0.03 in [0,1] range",
        "3x3_mean_blur":"3x3 average filter", "translation_4px":"translate content 4 pixels right and 4 pixels down with reflected padding"},
        "interpretation":"Exploratory validation-only sensitivity. Validation has already been used for model selection; these results are descriptive and must not be used to retune the model. The test split is not accessed by this script."}
    (output / "validation_robustness_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(df.to_string(index=False)); print(f"Saved: {output.resolve()}")

if __name__ == "__main__":
    main()
