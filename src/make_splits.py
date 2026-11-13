
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/splits/ph2_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    required = {"case_id", "image", "mask"}

    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")

    if len(frame) != 200:
        raise ValueError(f"Expected 200 pairs, found {len(frame)}")

    frame = frame.sort_values("case_id").reset_index(drop=True)

    if frame["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs found.")

    if "same_dimensions" in frame:
        valid = frame["same_dimensions"].astype(str).str.lower().eq("true")
        if not valid.all():
            raise ValueError("Image-mask dimension mismatch detected.")

    # Calculate lesion foreground area from every original mask.
    fractions = []

    for path in frame["mask"]:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"))

        values = set(np.unique(mask).tolist())

        if values != {0, 255}:
            raise ValueError(
                f"Unexpected mask values in {path}: {sorted(values)}"
            )

        fractions.append(float((mask == 255).mean()))

    frame["foreground_fraction"] = fractions

    # Divide the dataset into five equal-sized lesion-area groups.
    ranks = frame["foreground_fraction"].rank(method="first")
    frame["area_bin"] = pd.qcut(
        ranks, q=5, labels=False
    ).astype(int)

    bin_counts = frame["area_bin"].value_counts().sort_index()

    if not (bin_counts == 40).all():
        raise RuntimeError("The five lesion-area groups are not balanced.")

    # 70% training, 30% temporary holdout.
    train, holdout = train_test_split(
        frame,
        test_size=0.30,
        random_state=args.seed,
        shuffle=True,
        stratify=frame["area_bin"],
    )

    # Divide holdout equally into validation and test.
    validation, test = train_test_split(
        holdout,
        test_size=0.50,
        random_state=args.seed,
        shuffle=True,
        stratify=holdout["area_bin"],
    )

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # Check split sizes and prevent leakage.
    expected = {"train": 140, "validation": 30, "test": 30}

    for name, part in splits.items():
        if len(part) != expected[name]:
            raise RuntimeError(f"Incorrect size for {name}: {len(part)}")

    id_sets = {
        name: set(part["case_id"])
        for name, part in splits.items()
    }

    if (
        id_sets["train"] & id_sets["validation"]
        or id_sets["train"] & id_sets["test"]
        or id_sets["validation"] & id_sets["test"]
    ):
        raise RuntimeError("Data leakage: overlapping case IDs.")

    # Build the split manifest used by every experiment.
    manifest = {
        "seed": args.seed,
        "split_strategy": "70/15/15, stratified into five lesion-area groups",
        "train": train[["case_id", "image", "mask"]].to_dict("records"),
        "validation": validation[["case_id", "image", "mask"]].to_dict("records"),
        "test": test[["case_id", "image", "mask"]].to_dict("records"),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Save an audit trail showing each case's split and lesion area.
    details = pd.concat(
        [
            part.assign(split=name)
            for name, part in splits.items()
        ],
        ignore_index=True,
    )

    details = details[
        ["case_id", "split", "foreground_fraction", "area_bin"]
    ]

    details_path = output.parent / "ph2_seed42_details.csv"
    details.to_csv(details_path, index=False)

    # Print the lesion-area distribution in each split.
    summary = pd.crosstab(
        details["area_bin"], details["split"]
    ).reindex(
        index=range(5),
        columns=["train", "validation", "test"],
        fill_value=0,
    )

    expected_distribution = np.array([[28, 6, 6]] * 5)

    if not np.array_equal(summary.values, expected_distribution):
        raise RuntimeError("Unexpected lesion-area distribution.")

    print("FIXED SPLIT CREATED SUCCESSFULLY")
    print("\nSplit sizes:")
    for name, part in splits.items():
        print(f"{name}: {len(part)}")

    print("\nLesion-area group distribution:")
    print(summary.to_string())

    print("\nSplit manifest:", output.resolve())
    print("Split details:", details_path.resolve())
    print("Seed:", args.seed)
    print("Overlapping case IDs: 0")


if __name__ == "__main__":
    main()
