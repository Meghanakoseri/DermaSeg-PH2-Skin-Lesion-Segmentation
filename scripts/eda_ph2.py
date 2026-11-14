#!/usr/bin/env python3
"""Reproducible, path-sanitized EDA for the PH2 lesion dataset.

This is descriptive dataset-level EDA. It does not train models, select
hyperparameters, or calculate model performance. Run against the audit CSV
created by `python -m src.audit`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(path_str: str, root: Path) -> str:
    p = Path(path_str).resolve()
    try:
        return p.relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.name


def read_pairs(manifest: Path, data_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest)
    required = {"case_id", "image", "mask"}
    missing_cols = required - set(frame.columns)
    if missing_cols:
        raise ValueError(f"Audit CSV lacks required columns: {sorted(missing_cols)}")
    if frame.case_id.astype(str).duplicated().any():
        dupes = frame.loc[frame.case_id.astype(str).duplicated(), "case_id"].tolist()
        raise ValueError(f"Duplicate case IDs in audit CSV: {dupes[:10]}")
    rows = []
    image_hashes, mask_hashes = [], []
    errors = []
    for row in frame.to_dict("records"):
        image_path, mask_path = Path(row["image"]), Path(row["mask"])
        if not image_path.is_absolute():
            # src.audit writes paths relative to the current project working
            # directory when --data-root is relative. Prefer CWD resolution;
            # fall back to the supplied dataset root only for root-relative CSVs.
            candidates = [Path.cwd() / image_path, data_root / image_path, manifest.parent / image_path]
            image_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())
        if not mask_path.is_absolute():
            candidates = [Path.cwd() / mask_path, data_root / mask_path, manifest.parent / mask_path]
            mask_path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())
        if not image_path.is_file() or not mask_path.is_file():
            errors.append(f"{row['case_id']}: image or mask path missing")
            continue
        try:
            with Image.open(image_path) as im:
                im.load()
                w, h = im.size
                image_mode = im.mode
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            with Image.open(mask_path) as m:
                m.load()
                mask = np.asarray(m.convert("L"))
                mw, mh = m.size
            unique = np.unique(mask)
            if (w, h) != (mw, mh):
                errors.append(f"{row['case_id']}: image/mask dimensions differ")
            if not set(unique.tolist()).issubset({0, 255}):
                errors.append(f"{row['case_id']}: mask is not binary 0/255")
            image_hash = sha256(image_path)
            mask_hash = sha256(mask_path)
            image_hashes.append(image_hash)
            mask_hashes.append(mask_hash)
            fg = float((mask > 0).mean())
            rows.append({
                "case_id": str(row["case_id"]),
                "image_path_relative": relpath(str(image_path), data_root),
                "mask_path_relative": relpath(str(mask_path), data_root),
                "width": int(w), "height": int(h), "image_mode": image_mode,
                "mask_values": ",".join(map(str, unique.tolist())),
                "foreground_pixels": int((mask > 0).sum()),
                "background_pixels": int((mask == 0).sum()),
                "lesion_area_fraction": fg,
                "red_mean": float(rgb[:, :, 0].mean()),
                "green_mean": float(rgb[:, :, 1].mean()),
                "blue_mean": float(rgb[:, :, 2].mean()),
                "red_std": float(rgb[:, :, 0].std()),
                "green_std": float(rgb[:, :, 1].std()),
                "blue_std": float(rgb[:, :, 2].std()),
                "image_sha256": image_hash,
                "mask_sha256": mask_hash,
            })
        except Exception as exc:  # report each damaged pair, don't silently skip
            errors.append(f"{row['case_id']}: {type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError("EDA stopped because data audit failed:\n- " + "\n- ".join(errors[:30]))
    if not rows:
        raise RuntimeError("No valid pairs found.")
    out = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    if out.image_sha256.duplicated().any():
        raise RuntimeError("Exact duplicate original-image files detected; inspect before proceeding.")
    if out.mask_sha256.duplicated().any():
        # Identical masks can occur legitimately, so record rather than stop.
        print("WARNING: duplicate mask file content detected; inspect duplicate mask hashes.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="results/audit/pairs.csv")
    parser.add_argument("--data-root", required=True, help="Root containing PH2 case folders")
    parser.add_argument("--output-dir", default="results/audit/eda")
    parser.add_argument("--split-details", default="data/splits/ph2_seed42_details.csv")
    args = parser.parse_args()

    manifest, root, outdir = Path(args.manifest), Path(args.data_root), Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (Path("results/graphs")).mkdir(parents=True, exist_ok=True)
    cases = read_pairs(manifest, root)
    cases.to_csv(outdir / "eda_case_statistics.csv", index=False)
    data_dictionary = pd.DataFrame([
        {"field": "case_id", "type": "string", "definition": "PH2 case identifier; used to pair image/mask and prevent split overlap", "unit_or_values": "IMD###"},
        {"field": "image_path_relative", "type": "relative path", "definition": "Path to original RGB dermoscopic image relative to data root", "unit_or_values": "path"},
        {"field": "mask_path_relative", "type": "relative path", "definition": "Path to expert lesion segmentation mask relative to data root", "unit_or_values": "path"},
        {"field": "width", "type": "integer", "definition": "Original image width before resizing", "unit_or_values": "pixels"},
        {"field": "height", "type": "integer", "definition": "Original image height before resizing", "unit_or_values": "pixels"},
        {"field": "image_mode", "type": "categorical", "definition": "PIL mode of original image before RGB conversion", "unit_or_values": "RGB/RGBA/etc."},
        {"field": "mask_values", "type": "categorical set", "definition": "Unique grayscale values present in lesion mask", "unit_or_values": "expected 0,255"},
        {"field": "foreground_pixels", "type": "integer", "definition": "Number of mask pixels assigned to lesion", "unit_or_values": "pixels"},
        {"field": "background_pixels", "type": "integer", "definition": "Number of mask pixels assigned to background", "unit_or_values": "pixels"},
        {"field": "lesion_area_fraction", "type": "float", "definition": "Foreground pixels divided by all mask pixels", "unit_or_values": "fraction in [0,1]"},
        {"field": "red_mean/green_mean/blue_mean", "type": "float", "definition": "Per-image mean RGB intensity after conversion to RGB and scaling", "unit_or_values": "0 to 1"},
        {"field": "red_std/green_std/blue_std", "type": "float", "definition": "Per-image standard deviation of RGB intensity", "unit_or_values": "0 to 1"},
        {"field": "image_sha256", "type": "string", "definition": "SHA-256 checksum used to detect exact duplicate image files", "unit_or_values": "hex digest"},
        {"field": "mask_sha256", "type": "string", "definition": "SHA-256 checksum for lesion mask file", "unit_or_values": "hex digest"},
    ])
    data_dictionary.to_csv(outdir / "data_dictionary.csv", index=False)

    dims = (cases.groupby(["width", "height"], as_index=False)
            .agg(count=("case_id", "count"))
            .sort_values(["height", "width"]))
    dims.to_csv(outdir / "raw_dimension_distribution.csv", index=False)

    area = cases.lesion_area_fraction
    summary = {
        "dataset": "PH2",
        "n_cases": int(len(cases)),
        "readable_pairs": int(len(cases)),
        "missing_or_invalid_pairs": 0,
        "unique_case_ids": int(cases.case_id.nunique()),
        "image_sha256_duplicates": int(cases.image_sha256.duplicated().sum()),
        "distinct_raw_dimension_combinations": int(len(dims)),
        "width_min": int(cases.width.min()), "width_max": int(cases.width.max()),
        "height_min": int(cases.height.min()), "height_max": int(cases.height.max()),
        "lesion_area_fraction_min": float(area.min()),
        "lesion_area_fraction_max": float(area.max()),
        "lesion_area_fraction_mean": float(area.mean()),
        "lesion_area_fraction_median": float(area.median()),
        "lesion_area_fraction_std": float(area.std(ddof=1)),
        "foreground_pixels_total": int(cases.foreground_pixels.sum()),
        "background_pixels_total": int(cases.background_pixels.sum()),
        "foreground_pixel_fraction_over_all_pixels": float(cases.foreground_pixels.sum() / (cases.foreground_pixels.sum() + cases.background_pixels.sum())),
        "mean_rgb_intensity": {c: float(cases[f"{c}_mean"].mean()) for c in ["red", "green", "blue"]},
        "mask_encoding": "binary values 0 and 255",
        "input_size": [256, 256],
        "split_protocol": "fixed 140/30/30 seed-42 allocation; see split details file",
    }
    (outdir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(area * 100, bins=20, edgecolor="black")
    ax.set(title="PH2 lesion area distribution", xlabel="Lesion area (% of mask pixels)", ylabel="Number of images")
    fig.tight_layout(); fig.savefig("results/graphs/eda_lesion_area_distribution.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([f"{int(r.width)}×{int(r.height)}" for r in dims.itertuples()], dims["count"])
    ax.set(title="PH2 raw image dimensions", xlabel="Width × height (pixels)", ylabel="Number of images")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout(); fig.savefig("results/graphs/eda_raw_dimensions.png", dpi=220); plt.close(fig)

    # Image-level global RGB channel intensity distributions. These are
    # descriptive image features, not labels and not a basis for test tuning.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for channel, label in [("red_mean", "Red"), ("green_mean", "Green"), ("blue_mean", "Blue")]:
        ax.hist(cases[channel], bins=18, alpha=0.45, label=label)
    ax.set(title="PH2 image-level RGB intensity means", xlabel="Mean channel intensity (scaled 0–1)", ylabel="Number of images")
    ax.legend()
    fig.tight_layout(); fig.savefig("results/graphs/eda_rgb_intensity_means.png", dpi=220); plt.close(fig)

    feature_cols = ["lesion_area_fraction", "width", "height", "red_mean", "green_mean", "blue_mean"]
    corr = cases[feature_cols].corr(method="spearman")
    corr.to_csv(outdir / "eda_spearman_correlations.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(feature_cols)), feature_cols)
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Spearman correlations: image-level descriptors")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig("results/graphs/eda_feature_correlations.png", dpi=220); plt.close(fig)

    split_file = Path(args.split_details)
    split_interpretation = "Split-specific area statistics were not generated because the split-details CSV was unavailable."
    if split_file.is_file():
        split = pd.read_csv(split_file)
        # Accept split-details files containing case_id + split and optional area field.
        feature_columns = ["lesion_area_fraction", "red_mean", "green_mean", "blue_mean", "width", "height"]
        split_for_merge = split.drop(columns=[c for c in feature_columns if c in split.columns], errors="ignore")
        merged = split_for_merge.merge(cases[["case_id", *feature_columns]], on="case_id", how="left", validate="one_to_one")
        split_col = "split" if "split" in merged.columns else ("subset" if "subset" in merged.columns else None)
        if split_col:
            if merged["lesion_area_fraction"].isna().any():
                raise RuntimeError("Some split-detail case IDs did not match audited lesion masks.")
            stats = merged.groupby(split_col).lesion_area_fraction.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
            stats.to_csv(outdir / "split_lesion_area_summary.csv", index=False)
            fig, ax = plt.subplots(figsize=(7, 4.2))
            split_labels = sorted(merged[split_col].dropna().astype(str).unique())
            groups = [merged.loc[merged[split_col].astype(str).str.lower().eq(label.lower()), "lesion_area_fraction"].dropna().to_numpy() * 100
                      for label in split_labels]
            ax.boxplot(groups, showmeans=True)
            ax.set_xticks(range(1, len(split_labels) + 1), split_labels)
            ax.set(title="Lesion area by data split", ylabel="Lesion area (%)", xlabel="Split")
            fig.tight_layout(); fig.savefig("results/graphs/eda_split_area_distribution.png", dpi=220); plt.close(fig)

            # Keep modeling-oriented EDA restricted to the training subset.
            split_norm = merged[split_col].astype(str).str.strip().str.lower().replace({"training": "train"})
            train_cases = merged.loc[split_norm.eq("train")].copy()
            train_area = train_cases["lesion_area_fraction"]
            train_summary = {
                "analysis_split": "train only",
                "n_images": int(len(train_cases)),
                "lesion_area_fraction_min": float(train_area.min()),
                "lesion_area_fraction_max": float(train_area.max()),
                "lesion_area_fraction_mean": float(train_area.mean()),
                "lesion_area_fraction_median": float(train_area.median()),
                "lesion_area_fraction_std": float(train_area.std(ddof=1)),
                "mean_rgb_intensity": {c: float(train_cases[f"{c}_mean"].mean()) for c in ["red", "green", "blue"]},
                "note": "Training-only descriptive EDA. Validation/test labels were not used to guide model or preprocessing decisions."
            }
            (outdir / "eda_training_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.hist(train_area * 100, bins=18, edgecolor="black")
            ax.set(title="Training split: lesion area distribution", xlabel="Lesion area (% of mask pixels)", ylabel="Number of training images")
            fig.tight_layout(); fig.savefig("results/graphs/eda_train_lesion_area_distribution.png", dpi=220); plt.close(fig)
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for channel, label in [("red_mean", "Red"), ("green_mean", "Green"), ("blue_mean", "Blue")]:
                ax.hist(train_cases[channel], bins=16, alpha=0.45, label=label)
            ax.set(title="Training split: image-level RGB intensity means", xlabel="Mean channel intensity (scaled 0–1)", ylabel="Number of training images")
            ax.legend(); fig.tight_layout(); fig.savefig("results/graphs/eda_train_rgb_intensity_means.png", dpi=220); plt.close(fig)
            split_interpretation = "The split-level boxplot checks whether lesion-area distributions are broadly comparable across partitions. Modeling-oriented distributions and summary statistics are separately calculated from training cases only; validation/test labels are not used to guide modeling or preprocessing decisions."

    interpretations = f"""# PH2 dataset card and EDA interpretation

## Dataset and target
- Observational unit: one dermoscopic image and its paired binary lesion mask.
- Cases audited: {summary['n_cases']}.
- Target: foreground lesion pixels (mask value 255); background pixels (0).
- Raw dimensions: width {summary['width_min']}–{summary['width_max']} px; height {summary['height_min']}–{summary['height_max']} px; {summary['distinct_raw_dimension_combinations']} observed dimension combinations.
- All pairs were readable, dimension-matched and binary in this run.

## Descriptive EDA findings
- Lesion area ranges from {summary['lesion_area_fraction_min']*100:.2f}% to {summary['lesion_area_fraction_max']*100:.2f}% of the mask, with mean {summary['lesion_area_fraction_mean']*100:.2f}% and median {summary['lesion_area_fraction_median']*100:.2f}% (sample SD {summary['lesion_area_fraction_std']*100:.2f} percentage points).
- The wide lesion-area range indicates substantial variation in foreground size. This motivates reporting overlap metrics (F1/Dice and IoU) in addition to pixel-wise precision and recall; it also supports the previously defined lesion-area stratification for the split.
- Raw image dimensions vary slightly. Images and masks are resized together to 256×256, using bilinear interpolation for RGB images and nearest-neighbour interpolation for masks so the discrete mask labels are preserved.
- The RGB intensity histogram describes image-level average channel brightness/colour. Spearman correlations between lesion area, raw dimensions and image-level RGB means are included as descriptive checks; they are not used for feature selection because the CNN is trained directly on the RGB image.
- No exact duplicate source-image files were detected by SHA-256 hashing in this run.

## Split interpretation
{split_interpretation}

## Source, access and licence note
The original PH² Browser tutorial states that PH² is available for research and educational purposes. The published PH² dataset license notice also states that redistribution and commercial use are not allowed. The Kaggle mirror's own terms were not independently verified; do not redistribute its archive or PH² images/masks. Cite the original PH² paper and follow the original provider terms.

## Data protection and intended use
Only de-identified dataset case IDs and relative paths are recorded by this script; absolute machine-specific paths are not written to the EDA case table. This is an academic segmentation dataset. The model is not a diagnostic or clinical decision tool.

## Outputs
- `eda_summary.json`
- `eda_case_statistics.csv`
- `data_dictionary.csv`
- `raw_dimension_distribution.csv`
- `results/graphs/eda_lesion_area_distribution.png`
- `results/graphs/eda_raw_dimensions.png`
- `results/graphs/eda_rgb_intensity_means.png`
- `eda_spearman_correlations.csv`
- `results/graphs/eda_feature_correlations.png`
- Optional split-level summary and plot when split details are supplied.
"""
    (outdir / "PH2_DATASET_CARD.md").write_text(interpretations, encoding="utf-8")
    # Keep the conventional project paths expected by README and existing report.
    project_audit = Path("results/audit")
    project_audit.mkdir(parents=True, exist_ok=True)
    for name in ["eda_summary.json", "eda_training_summary.json", "data_dictionary.csv", "raw_dimension_distribution.csv", "PH2_DATASET_CARD.md"]:
        source = outdir / name
        if source.resolve() != (project_audit / name).resolve():
            (project_audit / name).write_bytes(source.read_bytes())
    print(json.dumps(summary, indent=2))
    print(f"EDA outputs written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
