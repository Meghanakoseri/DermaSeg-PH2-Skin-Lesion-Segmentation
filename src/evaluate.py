from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import PH2Dataset, load_split
from src.metrics import confusion_counts, metrics_from_counts
from src.model import UNet
from src.utils import ensure_dirs, get_device, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    records = load_split(config["data"]["split_manifest"], "test")
    dataset = PH2Dataset(records, config["project"]["image_size"], augment=False)
    loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        pin_memory=bool(config["data"].get("pin_memory", device.type == "cuda")) and device.type == "cuda",
    )

    model = UNet(
        config["project"]["in_channels"],
        config["project"]["num_classes"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_root = Path(config["evaluation"]["output_dir"])
    prediction_dir = output_root / "predictions" / args.experiment
    table_dir = output_root / "tables"
    graph_dir = output_root / "graphs"
    ensure_dirs(prediction_dir, table_dir, graph_dir)

    total_counts = [0, 0, 0, 0]
    per_image = []

    with torch.no_grad():
        for images, masks, case_ids in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = probabilities >= config["evaluation"]["threshold"]

            for index, case_id in enumerate(case_ids):
                counts = confusion_counts(predictions[index].cpu(), masks[index].cpu())
                total_counts = [a + b for a, b in zip(total_counts, counts)]
                per_image.append({"case_id": case_id, **metrics_from_counts(*counts)})

                mask = predictions[index].cpu().numpy().astype(np.uint8) * 255
                from PIL import Image
                Image.fromarray(mask).save(prediction_dir / f"{case_id}_prediction.png")

    aggregate = metrics_from_counts(*total_counts)
    aggregate["experiment"] = args.experiment
    aggregate["checkpoint_epoch"] = checkpoint.get("epoch")
    aggregate["best_validation_iou"] = checkpoint.get("best_val_iou")

    with (table_dir / f"{args.experiment}_test_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate.keys()))
        writer.writeheader()
        writer.writerow(aggregate)

    with (table_dir / f"{args.experiment}_per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    names = ["Precision", "Recall", "F1", "IoU"]
    values = [aggregate["precision"], aggregate["recall"], aggregate["f1"], aggregate["iou"]]
    plt.figure(figsize=(7, 4))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Test metrics: {args.experiment}")
    plt.tight_layout()
    plt.savefig(graph_dir / f"{args.experiment}_test_metrics.png", dpi=200)
    plt.close()

    print("Verified test evaluation completed.")
    print(aggregate)


if __name__ == "__main__":
    main()
