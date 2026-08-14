import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import CLASS_NAMES

def embed_split(root, size=32):
    root = Path(root)
    vectors = []
    meta = []
    for class_name in CLASS_NAMES:
        folder = root / class_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.jpg")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            vector = cv2.resize(image, (size, size),
                                interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            vector -= vector.mean()
            norm = np.linalg.norm(vector)
            if norm < 1e-6:
                continue
            vectors.append(vector / norm)
            meta.append({"class_name": class_name, "filename": path.name,
                         "path": str(path)})
    return np.stack(vectors), meta

def main():
    parser = argparse.ArgumentParser(
        description="find near duplicate images shared between the train and test splits"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--output", default="data/leaked_test_images.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = config["data"]

    train_vectors, train_meta = embed_split(data_cfg["train_dir"])
    test_vectors, test_meta = embed_split(data_cfg["test_dir"])
    print(f"train {train_vectors.shape[0]} images, test {test_vectors.shape[0]} images")

    similarity = test_vectors @ train_vectors.T
    best = similarity.max(axis=1)
    nearest = similarity.argmax(axis=1)

    for threshold in (0.99, 0.995, 0.999, 0.9999):
        count = int((best > threshold).sum())
        print(f"  test images matching a training image above {threshold}: "
              f"{count} ({100.0 * count / len(test_meta):.2f} percent)")

    flagged = [i for i in range(len(test_meta)) if best[i] > args.threshold]
    same_class = sum(
        test_meta[i]["class_name"] == train_meta[nearest[i]]["class_name"]
        for i in flagged
    )
    print(f"\nat threshold {args.threshold}: {len(flagged)} flagged, "
          f"{same_class} share the same class label")

    by_class = {}
    for i in flagged:
        by_class[test_meta[i]["class_name"]] = by_class.get(
            test_meta[i]["class_name"], 0) + 1
    print("flagged per class:", by_class)

    payload = {
        "threshold": args.threshold,
        "method": "32x32 mean centred cosine similarity on raw grayscale",
        "n_test": len(test_meta),
        "n_flagged": len(flagged),
        "n_same_class": same_class,
        "flagged": [
            {
                "test": f"{test_meta[i]['class_name']}/{test_meta[i]['filename']}",
                "train": f"{train_meta[nearest[i]]['class_name']}/"
                         f"{train_meta[nearest[i]]['filename']}",
                "similarity": round(float(best[i]), 6),
            }
            for i in flagged
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written {output}")

if __name__ == "__main__":
    main()
