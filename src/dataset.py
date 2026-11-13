from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_HINTS = ("lesion", "mask", "segmentation", "groundtruth", "ground_truth")


def _case_id(path: Path) -> str | None:
    import re
    match = re.search(r"(IMD\d{3})", str(path), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def discover_pairs(root: str | Path) -> tuple[list[dict], list[str]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    candidates = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]

    by_case: dict[str, list[Path]] = {}
    for path in candidates:
        cid = _case_id(path)
        if cid:
            by_case.setdefault(cid, []).append(path)

    pairs = []
    problems = []

    for cid, paths in sorted(by_case.items()):
        mask_candidates = [
            p for p in paths
            if any(hint in (str(p.parent) + " " + p.stem).lower() for hint in MASK_HINTS)
        ]
        image_candidates = [
            p for p in paths
            if "dermoscopic" in (str(p.parent) + " " + p.stem).lower()
            and p not in mask_candidates
        ]

        if not image_candidates:
            image_candidates = [
                p for p in paths
                if p not in mask_candidates and "thumbnail" not in p.stem.lower()
            ]

        if len(image_candidates) != 1 or len(mask_candidates) < 1:
            problems.append(
                f"{cid}: {len(image_candidates)} possible image(s), "
                f"{len(mask_candidates)} possible mask(s)"
            )
            continue

        if len(mask_candidates) > 1:
            exact = [
                p for p in mask_candidates
                if "lesion" in (str(p.parent) + " " + p.stem).lower()
            ]
            if len(exact) == 1:
                mask_candidates = exact
            else:
                problems.append(f"{cid}: ambiguous masks: {[str(p) for p in mask_candidates]}")
                continue

        pairs.append({
            "case_id": cid,
            "image": str(image_candidates[0]),
            "mask": str(mask_candidates[0]),
        })

    return pairs, problems


class PH2Dataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        image_size: int = 256,
        augment: bool = False,
    ):
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record["image"]).convert("RGB")
        mask = Image.open(record["mask"]).convert("L")

        image = TF.resize(
            image, [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask, [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if self.augment:
            if torch.rand(()) < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if torch.rand(()) < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image_tensor = TF.to_tensor(image)
        mask_array = np.asarray(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy((mask_array > 0).astype(np.int64))

        return image_tensor, mask_tensor, record["case_id"]


def load_split(path: str | Path, split: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if split not in manifest:
        raise KeyError(f"Split '{split}' not found in {path}")
    return manifest[split]
