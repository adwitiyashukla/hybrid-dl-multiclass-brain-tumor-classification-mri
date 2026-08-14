import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import (BrainMRIDataset, CLASS_NAMES, scan_directory,
                          split_train_val)
from decision import apply_offsets, coordinate_search
from engine import evaluate
from losses import ClassBalancedFocalLoss
from metrics import compute_metrics, format_report
from models.fusion_net import build_model

def main():
    parser = argparse.ArgumentParser(
        description="tune per class decision offsets on the validation split"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--objective", default="macro_f1",
                        choices=["macro_f1", "tumor_sensitivity"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = build_model(payload.get("config", config)).to(device)
    model.load_state_dict(payload.get("ema") or payload["model"])
    model.eval()

    records = scan_directory(data_cfg["train_dir"])
    _, val_records = split_train_val(
        records, data_cfg["val_fraction"], exp_cfg["seed"]
    )

    dataset = BrainMRIDataset(
        val_records, cache_dir=Path(data_cfg["cache_dir"]) / "train",
        n_channels=model_cfg["in_chans"],
        use_handcrafted=model_cfg["use_handcrafted"],
        image_size=data_cfg["image_size"], augment=False,
        dip_options=config.get("dip", {}),
    )
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"],
                        shuffle=False, num_workers=data_cfg["num_workers"])

    criterion = ClassBalancedFocalLoss().to(device)
    stats = evaluate(model, loader, criterion, device, model_cfg["use_handcrafted"])
    labels, probs = stats["labels"], stats["probs"]

    notumor_index = CLASS_NAMES.index("notumor")

    print("validation, argmax decision rule")
    print(format_report(compute_metrics(labels, probs)))

    offsets, score = coordinate_search(labels, probs, args.objective, notumor_index)

    print(f"\ntuned offsets ({args.objective} objective, zero centred)")
    for name, value in zip(CLASS_NAMES, offsets):
        print(f"  {name:12s} {value:+.2f}")

    print("\nvalidation, tuned decision rule")
    tuned_metrics = compute_metrics(labels, apply_offsets(probs, offsets))
    print(format_report(tuned_metrics))

    output = Path(args.output or Path(args.checkpoint).parent / "offsets.json")
    output.write_text(json.dumps({
        "class_names": CLASS_NAMES,
        "offsets": offsets.tolist(),
        "objective": args.objective,
        "tuned_on": "validation split",
        "validation_macro_f1_argmax": compute_metrics(labels, probs)["macro_f1"],
        "validation_macro_f1_tuned": tuned_metrics["macro_f1"],
    }, indent=2), encoding="utf-8")
    print(f"\nwritten {output}")

if __name__ == "__main__":
    main()
