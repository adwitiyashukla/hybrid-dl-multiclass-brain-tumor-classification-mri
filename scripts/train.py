import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import (BrainMRIDataset, class_counts, save_split,
                          scan_directory, split_train_val)
from engine import (ModelEma, cosine_schedule_with_warmup, evaluate,
                    load_checkpoint, make_scaler, save_checkpoint,
                    train_one_epoch, write_history)
from losses import ClassBalancedFocalLoss, class_balanced_weights
from metrics import compute_metrics, format_report
from models.fusion_net import build_model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_sampler(records, counts):
    weights_per_class = 1.0 / np.array(counts, dtype=np.float64)
    sample_weights = [weights_per_class[r["label"]] for r in records]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(records), replacement=True
    )


def main():
    parser = argparse.ArgumentParser(description="train the Brain Tumor MRI Classifier model")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", default="auto")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    loss_cfg = config.get("loss", {})

    set_seed(exp_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    output_dir = Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_directory(data_cfg["train_dir"])
    train_records, val_records = split_train_val(
        records, data_cfg["val_fraction"], exp_cfg["seed"]
    )
    save_split(train_records, val_records, output_dir / "split.json")

    counts = class_counts(train_records, model_cfg["n_classes"])
    print(f"train {len(train_records)} val {len(val_records)} counts {counts}")

    cache_root = Path(data_cfg["cache_dir"]) / "train"
    use_handcrafted = model_cfg["use_handcrafted"]

    train_dataset = BrainMRIDataset(
        train_records, cache_dir=cache_root, n_channels=model_cfg["in_chans"],
        use_handcrafted=use_handcrafted, image_size=data_cfg["image_size"],
        augment=True, dip_options=config.get("dip", {}),
    )
    val_dataset = BrainMRIDataset(
        val_records, cache_dir=cache_root, n_channels=model_cfg["in_chans"],
        use_handcrafted=use_handcrafted, image_size=data_cfg["image_size"],
        augment=False, dip_options=config.get("dip", {}),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        sampler=build_sampler(train_records, counts),
        num_workers=data_cfg["num_workers"], pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"], pin_memory=(device.type == "cuda"),
    )

    model = build_model(config).to(device)

    weights = class_balanced_weights(counts, beta=loss_cfg.get("beta", 0.9999))
    criterion = ClassBalancedFocalLoss(
        weights=weights.to(device),
        gamma=loss_cfg.get("gamma", 2.0),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    steps_per_epoch = max(1, len(train_loader) // train_cfg["grad_accum_steps"])
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=train_cfg["warmup_epochs"] * steps_per_epoch,
        total_steps=train_cfg["epochs"] * steps_per_epoch,
    )
    scaler = make_scaler(device, train_cfg["amp"] and device.type == "cuda")
    ema = ModelEma(model, decay=train_cfg["ema_decay"]) if train_cfg["ema_decay"] else None

    start_epoch = 0
    best_metric = -1.0
    history = []

    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"

    if args.resume == "auto" and last_path.exists():
        payload = load_checkpoint(last_path, model, optimizer, scheduler,
                                  scaler, ema, device)
        start_epoch = payload["epoch"] + 1
        best_metric = payload["best_metric"]
        history = payload.get("history", [])
        print(f"resumed from epoch {start_epoch}")
    elif args.resume not in ("auto", "none") and Path(args.resume).exists():
        payload = load_checkpoint(Path(args.resume), model, optimizer, scheduler,
                                  scaler, ema, device)
        start_epoch = payload["epoch"] + 1
        best_metric = payload["best_metric"]
        history = payload.get("history", [])
        print(f"resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        print(f"epoch {epoch + 1}/{train_cfg['epochs']}")
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, use_handcrafted, train_cfg["grad_accum_steps"], ema,
            train_cfg["max_grad_norm"],
        )

        raw_stats = evaluate(model, val_loader, criterion, device, use_handcrafted)
        raw_metrics = compute_metrics(raw_stats["labels"], raw_stats["probs"])

        if ema is not None:
            val_stats = evaluate(ema.module, val_loader, criterion, device,
                                 use_handcrafted)
            val_metrics = compute_metrics(val_stats["labels"], val_stats["probs"])
        else:
            val_stats, val_metrics = raw_stats, raw_metrics

        print(f"  train loss {train_stats['loss']:.4f} acc {train_stats['accuracy']:.4f} "
              f"({train_stats['seconds']:.0f}s)")
        print(f"  val raw  loss {raw_stats['loss']:.4f} "
              f"macro_f1 {raw_metrics['macro_f1']:.4f} "
              f"bal_acc {raw_metrics['balanced_accuracy']:.4f}")
        if ema is not None:
            print(f"  val ema  loss {val_stats['loss']:.4f} "
                  f"macro_f1 {val_metrics['macro_f1']:.4f} "
                  f"bal_acc {val_metrics['balanced_accuracy']:.4f} "
                  f"(decay {ema.current_decay():.4f})")
            gap = raw_metrics["macro_f1"] - val_metrics["macro_f1"]
            if gap > 0.15:
                print(f"  WARNING ema trails raw model by {gap:.3f} macro_f1, "
                      f"ema may be under converged")
        if val_stats["gates"] is not None:
            print(f"  mean fusion gate {val_stats['gates'].mean():.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_raw_macro_f1": raw_metrics["macro_f1"],
        })

        save_checkpoint(last_path, model, optimizer, scheduler, scaler, ema,
                        epoch, best_metric, config, history)

        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, ema,
                            epoch, best_metric, config, history)
            print(f"  new best macro_f1 {best_metric:.4f}")

        write_history(output_dir / "history.json", history)

    print(format_report(val_metrics))
    print(f"best val macro_f1 {best_metric:.4f}")


if __name__ == "__main__":
    main()
