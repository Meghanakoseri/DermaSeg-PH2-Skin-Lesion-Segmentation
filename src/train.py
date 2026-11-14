from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PH2Dataset, load_split
from src.losses import SegmentationLoss
from src.metrics import confusion_counts, metrics_from_counts
from src.model import UNet
from src.utils import ensure_dirs, get_device, load_config, save_json, set_seed


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, amp=False, gradient_clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    counts = [0, 0, 0, 0]

    for images, masks, _ in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(amp and device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        loss_sum += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        batch_counts = confusion_counts(predictions, masks)
        counts = [a + b for a, b in zip(counts, batch_counts)]

    metrics = metrics_from_counts(*counts)
    metrics["loss"] = loss_sum / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config["experiments"][args.experiment]
    seed = config["project"]["seed"]
    set_seed(seed)

    device = get_device()
    image_size = config["project"]["image_size"]
    split_path = config["data"]["split_manifest"]
    train_records = load_split(split_path, "train")
    val_records = load_split(split_path, "validation")

    train_data = PH2Dataset(train_records, image_size, augment=True)
    val_data = PH2Dataset(val_records, image_size, augment=False)

    use_cuda = device.type == "cuda"
    workers = config["data"]["num_workers"]
    train_loader = DataLoader(
        train_data,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=bool(config["data"].get("pin_memory", use_cuda)) and use_cuda,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config["data"].get("pin_memory", use_cuda)) and use_cuda,
        persistent_workers=workers > 0,
    )

    model = UNet(
        config["project"]["in_channels"],
        config["project"]["num_classes"],
    ).to(device)
    criterion = SegmentationLoss(exp["loss"], exp.get("dice_weight", 0.5))
    optimizer_name = str(config["training"].get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; this project currently implements AdamW.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["training"]["scheduler_factor"],
        patience=config["training"]["scheduler_patience"],
        min_lr=config["training"]["min_learning_rate"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and config["training"]["amp"])

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    log_dir = Path(config["training"]["log_dir"])
    ensure_dirs(checkpoint_dir, log_dir)
    checkpoint_path = checkpoint_dir / f"{args.experiment}_best.pt"
    log_path = log_dir / f"{args.experiment}.csv"

    best_iou = -1.0
    patience = 0
    started = time.time()

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch", "train_loss", "train_iou", "val_loss",
                "val_precision", "val_recall", "val_f1", "val_iou",
                "learning_rate", "epoch_seconds",
            ],
        )
        writer.writeheader()

        for epoch in range(1, config["training"]["epochs"] + 1):
            epoch_start = time.time()
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler=scaler, amp=config["training"]["amp"],
                gradient_clip_norm=float(config["training"].get("gradient_clip_norm", 1.0)),
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, None, device,
                scaler=None, amp=config["training"]["amp"],
                gradient_clip_norm=float(config["training"].get("gradient_clip_norm", 1.0)),
            )
            scheduler.step(val_metrics["iou"])
            epoch_seconds = time.time() - epoch_start

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_iou": val_metrics["iou"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
            }
            writer.writerow(row)
            handle.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"val IoU {val_metrics['iou']:.4f} | "
                f"{epoch_seconds:.1f}s"
            )

            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                patience = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "experiment": args.experiment,
                    "epoch": epoch,
                    "best_val_iou": best_iou,
                    "config": config,
                }, checkpoint_path)
            else:
                patience += 1

            if patience >= config["training"]["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    summary = {
        "experiment": args.experiment,
        "device": str(device),
        "best_validation_iou": best_iou,
        "training_seconds": time.time() - started,
        "checkpoint": str(checkpoint_path),
        "log": str(log_path),
    }
    save_json(summary, log_dir / f"{args.experiment}_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
