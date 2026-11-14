#!/usr/bin/env python3
"""Recompute boundary metrics from saved prediction PNGs (no model inference).

This script exists to make the earlier supplementary boundary analysis
reproducible. It is not a model-selection tool. Re-running on `test` reopens
test labels; do not do that for further tuning or claims of a fresh holdout.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


def load_binary(path: Path, size=(256, 256)) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != size:
            im = im.resize(size, Image.Resampling.NEAREST)
        return np.asarray(im) > 0


def erode(mask: np.ndarray, size: int = 3) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(im.filter(ImageFilter.MinFilter(size))) > 0


def dilate(mask: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(im.filter(ImageFilter.MaxFilter(size))) > 0


def boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~erode(mask, 3)


def scores(pred: np.ndarray, truth: np.ndarray, tolerance: int) -> tuple[float, float, float, int, int]:
    pb, tb = boundary(pred), boundary(truth)
    kernel = 2 * tolerance + 1
    matched_p = pb & dilate(tb, kernel)
    matched_t = tb & dilate(pb, kernel)
    precision = float(matched_p.sum() / max(int(pb.sum()), 1))
    recall = float(matched_t.sum() / max(int(tb.sum()), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1, int(pb.sum()), int(tb.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-manifest", default="data/splits/ph2_seed42.json")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--baseline-pred-dir", required=True)
    p.add_argument("--proposed-pred-dir", required=True)
    p.add_argument("--output-dir", default="results/boundary_reproducibility")
    p.add_argument("--tolerance", type=int, default=2)
    p.add_argument("--i-understand-test-reuse", action="store_true",
                    help="Required only when --split test; this reruns descriptive analysis on previously inspected test labels.")
    args = p.parse_args()
    if args.split == "test" and not args.i_understand_test_reuse:
        p.error("Test labels were already examined. Use validation for new work. For archival reproduction only, add --i-understand-test-reuse.")
    import json
    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))[args.split]
    dirs = {"baseline": Path(args.baseline_pred_dir), "proposed": Path(args.proposed_pred_dir)}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    records = []
    for item in split:
        cid = str(item["case_id"])
        gt = load_binary(Path(item["mask"]))
        row = {"case_id": cid}
        for model, pred_dir in dirs.items():
            pred_path = pred_dir / f"{cid}_prediction.png"
            if not pred_path.is_file():
                raise FileNotFoundError(f"Missing saved prediction for {cid}: {pred_path}")
            pred = load_binary(pred_path)
            bp, br, bf, npix, ngt = scores(pred, gt, args.tolerance)
            row.update({f"{model}_boundary_precision": bp, f"{model}_boundary_recall": br,
                        f"{model}_boundary_f1": bf, f"{model}_predicted_boundary_pixels": npix,
                        f"{model}_ground_truth_boundary_pixels": ngt})
        records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(out / "boundary_metrics_per_image.csv", index=False)
    summary = []
    for model in dirs:
        for metric in ["boundary_precision", "boundary_recall", "boundary_f1"]:
            values = df[f"{model}_{metric}"]
            summary.append({"model": model, "metric": metric, "mean": values.mean(), "std": values.std(ddof=1)})
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / "boundary_metrics_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    metrics = ["boundary_precision", "boundary_recall", "boundary_f1"]
    x = np.arange(len(metrics)); width = 0.36
    for idx, model in enumerate(dirs):
        vals = [sdf[(sdf.model == model) & (sdf.metric == metric)]["mean"].iloc[0] for metric in metrics]
        ax.bar(x + (idx - .5) * width, vals, width, label=model.title())
    ax.set_xticks(x, ["Boundary precision", "Boundary recall", "Boundary F1"])
    ax.set_ylim(0, 1); ax.set_ylabel("Macro-average per-image score")
    ax.set_title(f"Boundary metrics ({args.split}; tolerance={args.tolerance}px)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "boundary_metrics_comparison.png", dpi=220); plt.close(fig)
    import json
    (out / "boundary_metrics_summary.json").write_text(json.dumps({"split": args.split, "n_images": len(df),
        "inner_boundary": "3x3 erosion", "tolerance_pixels": args.tolerance,
        "averaging": "macro mean and sample standard deviation across images",
        "warning": "Descriptive analysis only; do not use for model selection."}, indent=2), encoding="utf-8")
    print(sdf.to_string(index=False))
    print(f"Saved boundary outputs: {out.resolve()}")

if __name__ == "__main__":
    main()
