import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import scan_directory
from dip.handcrafted import extract_features, FEATURE_NAMES
from dip.preprocessing import run_dip_pipeline

def process_split(records, cache_dir, image_size, dip_options, report_path):
    cache_dir = Path(cache_dir)
    rows = []
    failures = 0

    for index, record in enumerate(records):
        source = Path(record["path"])
        target_dir = cache_dir / record["class_name"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (source.stem + ".npz")

        if target.exists():
            continue

        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"skipping unreadable file {source}")
            failures += 1
            continue

        output = run_dip_pipeline(image, output_size=image_size, **dip_options)
        features, named = extract_features(output)

        np.savez_compressed(
            target,
            gray=output["gray"],
            mask=output["mask"],
            asymmetry=output["asymmetry"],
            features=features,
        )

        row = {"path": str(source), "class_name": record["class_name"]}
        row.update(output["report"])
        rows.append(row)

        if (index + 1) % 200 == 0:
            print(f"  {index + 1}/{len(records)} processed")

    if rows:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows, failures

def summarise(rows):
    if not rows:
        return
    fallback = sum(1 for r in rows if r.get("brain_extraction_failed"))
    low_contrast = sum(1 for r in rows if r.get("low_contrast"))
    brain_fraction = np.array([r.get("brain_fraction", 0.0) for r in rows])
    symmetry = np.array([r.get("symmetry_score", 0.0) for r in rows])

    print(f"  images processed          {len(rows)}")
    print(f"  brain extraction fallback {fallback} ({100.0 * fallback / len(rows):.2f} percent)")
    print(f"  low contrast flagged      {low_contrast} ({100.0 * low_contrast / len(rows):.2f} percent)")
    print(f"  brain fraction mean       {brain_fraction.mean():.4f}")
    print(f"  symmetry score mean       {symmetry.mean():.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="precompute the DIP cache and handcrafted features"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = config["data"]
    dip_options = config.get("dip", {})
    cache_root = Path(data_cfg["cache_dir"])

    for split_name, directory in (("train", data_cfg["train_dir"]),
                                  ("test", data_cfg["test_dir"])):
        if not Path(directory).is_dir():
            print(f"skipping {split_name}, directory not found: {directory}")
            continue
        print(f"processing {split_name} from {directory}")
        records = scan_directory(directory)
        rows, failures = process_split(
            records,
            cache_root / split_name,
            data_cfg["image_size"],
            dip_options,
            cache_root / f"quality_{split_name}.csv",
        )
        summarise(rows)
        if failures:
            print(f"  unreadable files          {failures}")

    print(f"feature vector length {len(FEATURE_NAMES)}")

if __name__ == "__main__":
    main()
