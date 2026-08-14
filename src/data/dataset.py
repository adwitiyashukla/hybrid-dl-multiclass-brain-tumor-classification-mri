import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan_directory(root):
    root = Path(root)
    records = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({
                    "path": str(path),
                    "label": CLASS_TO_INDEX[class_name],
                    "class_name": class_name,
                })
    if not records:
        raise FileNotFoundError(
            f"no images found under {root}. Expected subfolders named "
            f"{CLASS_NAMES}"
        )
    return records


def class_counts(records, n_classes=4):
    counts = [0] * n_classes
    for record in records:
        counts[record["label"]] += 1
    return counts


def split_train_val(records, val_fraction=0.15, seed=42):
    rng = np.random.default_rng(seed)
    by_class = {i: [] for i in range(len(CLASS_NAMES))}
    for record in records:
        by_class[record["label"]].append(record)

    train, val = [], []
    for label, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        cut = int(round(len(items) * val_fraction))
        val.extend(items[:cut])
        train.extend(items[cut:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def save_split(train, val, path):
    payload = {
        "train": [r["path"] for r in train],
        "val": [r["path"] for r in val],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


class BrainMRIDataset(Dataset):
    def __init__(self, records, cache_dir=None, n_channels=3,
                 use_handcrafted=True, image_size=256, augment=False,
                 dip_options=None):
        self.records = records
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.n_channels = n_channels
        self.use_handcrafted = use_handcrafted
        self.image_size = image_size
        self.augment = augment
        self.dip_options = dip_options or {}

    def __len__(self):
        return len(self.records)

    def _cache_path(self, record):
        if self.cache_dir is None:
            return None
        source = Path(record["path"])
        return self.cache_dir / record["class_name"] / (source.stem + ".npz")

    def _load_cached(self, record):
        path = self._cache_path(record)
        if path is None or not path.exists():
            return None
        with np.load(path) as data:
            return {
                "gray": data["gray"],
                "mask": data["mask"],
                "asymmetry": data["asymmetry"],
                "features": data["features"],
            }

    def _compute(self, record):
        from dip.preprocessing import run_dip_pipeline
        from dip.handcrafted import extract_features

        image = cv2.imread(record["path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"could not read {record['path']}")

        output = run_dip_pipeline(
            image, output_size=self.image_size, **self.dip_options
        )
        features, _ = extract_features(output)
        return {
            "gray": output["gray"],
            "mask": output["mask"],
            "asymmetry": output["asymmetry"],
            "features": features,
        }

    def _apply_augmentation(self, stacked, rng):
        if rng.random() < 0.5:
            stacked = stacked[:, ::-1, :]
        angle = rng.uniform(-12.0, 12.0)
        shift_x = rng.uniform(-0.05, 0.05) * stacked.shape[1]
        shift_y = rng.uniform(-0.05, 0.05) * stacked.shape[0]
        scale = rng.uniform(0.92, 1.08)

        h, w = stacked.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y
        stacked = cv2.warpAffine(
            stacked, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        if stacked.ndim == 2:
            stacked = stacked[:, :, None]

        brightness = rng.uniform(-12.0, 12.0)
        contrast = rng.uniform(0.9, 1.1)
        stacked = np.clip(stacked.astype(np.float32) * contrast + brightness, 0, 255)
        return stacked.astype(np.uint8)

    def __getitem__(self, index):
        import torch

        record = self.records[index]
        data = self._load_cached(record)
        if data is None:
            data = self._compute(record)

        channels = [data["gray"]]
        if self.n_channels >= 2:
            channels.append(data["mask"])
        if self.n_channels >= 3:
            channels.append(data["asymmetry"])
        stacked = np.dstack(channels)

        if self.augment:
            rng = np.random.default_rng()
            stacked = self._apply_augmentation(np.ascontiguousarray(stacked), rng)

        stacked = stacked.astype(np.float32) / 255.0
        stacked = (stacked - 0.5) / 0.5
        image = torch.from_numpy(np.ascontiguousarray(stacked.transpose(2, 0, 1)))

        item = {
            "image": image,
            "label": torch.tensor(record["label"], dtype=torch.long),
        }
        if self.use_handcrafted:
            item["handcrafted"] = torch.from_numpy(
                np.ascontiguousarray(data["features"].astype(np.float32))
            )
        return item
