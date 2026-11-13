from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from src.dataset import discover_pairs
from src.utils import ensure_dirs


def audit_dataset(data_root: str, output_dir: str) -> pd.DataFrame:
    pairs, problems = discover_pairs(data_root)
    rows = []

    for record in pairs:
        try:
            with Image.open(record["image"]) as image:
                image_size = image.size
                image_mode = image.mode
            with Image.open(record["mask"]) as mask:
                mask_size = mask.size
                mask_mode = mask.mode
                mask_values = sorted(set(mask.convert("L").getdata()))
            rows.append({
                **record,
                "image_width": image_size[0],
                "image_height": image_size[1],
                "image_mode": image_mode,
                "mask_width": mask_size[0],
                "mask_height": mask_size[1],
                "mask_mode": mask_mode,
                "mask_unique_values": ",".join(map(str, mask_values[:30])),
                "same_dimensions": image_size == mask_size,
            })
        except Exception as exc:
            problems.append(f"{record['case_id']}: unreadable file: {exc}")

    output = Path(output_dir)
    ensure_dirs(output)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pairs.csv", index=False)
    (output / "audit_issues.txt").write_text(
        "\n".join(problems) if problems else "No matching issues detected.",
        encoding="utf-8",
    )
    print(f"Valid matched pairs: {len(frame)}")
    print(f"Matching/read issues: {len(problems)}")
    if len(frame):
        print(f"Image-mask dimension mismatches: {(~frame['same_dimensions']).sum()}")
    print(f"Audit files: {output.resolve()}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="results/audit")
    args = parser.parse_args()
    audit_dataset(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
