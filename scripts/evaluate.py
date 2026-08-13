import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import BrainMRIDataset, scan_directory
from engine import evaluate
from losses import ClassBalancedFocalLoss
from metrics import bootstrap_ci, compute_metrics, confusion, format_report
from models.fusion_net import build_model

def main():
    parser = argparse.ArgumentParser(description="evaluate a checkpoint on the test set")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = config["data"]
    model_cfg = config["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = build_model(payload.get("config", config)).to(device)
    state = payload.get("ema") or payload["model"]
    model.load_state_dict(state)
    model.eval()

    records = scan_directory(data_cfg["test_dir"])
    dataset = BrainMRIDataset(
        records, cache_dir=Path(data_cfg["cache_dir"]) / "test",
        n_channels=model_cfg["in_chans"],
        use_handcrafted=model_cfg["use_handcrafted"],
        image_size=data_cfg["image_size"], augment=False,
        dip_options=config.get("dip", {}),
    )
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"],
                        shuffle=False, num_workers=data_cfg["num_workers"])

    criterion = ClassBalancedFocalLoss().to(device)
    stats = evaluate(model, loader, criterion, device, model_cfg["use_handcrafted"])
    metrics = compute_metrics(stats["labels"], stats["probs"])

    print(format_report(metrics))

    ci_f1 = bootstrap_ci(stats["labels"], stats["probs"], "macro_f1", args.bootstrap)
    ci_bal = bootstrap_ci(stats["labels"], stats["probs"], "balanced_accuracy",
                          args.bootstrap)
    print(f"macro_f1          {ci_f1['mean']:.4f} [{ci_f1['lo']:.4f}, {ci_f1['hi']:.4f}]")
    print(f"balanced_accuracy {ci_bal['mean']:.4f} [{ci_bal['lo']:.4f}, {ci_bal['hi']:.4f}]")

    matrix = confusion(stats["labels"], stats["probs"].argmax(1), model_cfg["n_classes"])
    print("confusion matrix, rows true, columns predicted")
    print(matrix)

    if stats["gates"] is not None:
        print(f"mean fusion gate {stats['gates'].mean():.4f}")

    output = {
        "metrics": metrics,
        "macro_f1_ci": ci_f1,
        "balanced_accuracy_ci": ci_bal,
        "confusion_matrix": matrix.tolist(),
    }
    out_path = Path(args.checkpoint).parent / "test_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"written {out_path}")

if __name__ == "__main__":
    main()
