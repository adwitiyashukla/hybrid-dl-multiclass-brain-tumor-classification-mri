import copy
import json
import time
from pathlib import Path

import numpy as np
import torch


def make_autocast(device, enabled):
    device_type = "cuda" if device.type == "cuda" else "cpu"
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def make_scaler(device, enabled):
    try:
        return torch.amp.GradScaler(
            "cuda" if device.type == "cuda" else "cpu", enabled=enabled
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class ModelEma:
    def __init__(self, model, decay=0.999, warmup=True):
        self.decay = decay
        self.warmup = warmup
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    def current_decay(self):
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.current_decay()
        for ema_param, param in zip(self.module.state_dict().values(),
                                    model.state_dict().values()):
            if ema_param.dtype.is_floating_point:
                ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
            else:
                ema_param.copy_(param)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_scale=0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, progress)
        return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, use_handcrafted, grad_accum_steps=1,
                    ema=None, max_grad_norm=1.0, log_every=50):
    model.train()
    running_loss = 0.0
    seen = 0
    correct = 0
    start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        handcrafted = None
        if use_handcrafted:
            handcrafted = batch["handcrafted"].to(device, non_blocking=True)

        with make_autocast(device, scaler.is_enabled()):
            output = model(images, handcrafted)
            loss = criterion(output["logits"], labels) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        batch_size = labels.size(0)
        running_loss += loss.item() * grad_accum_steps * batch_size
        seen += batch_size
        correct += (output["logits"].argmax(1) == labels).sum().item()

        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}/{len(loader)} "
                  f"loss {running_loss / seen:.4f} "
                  f"acc {correct / seen:.4f}")

    return {
        "loss": running_loss / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "seconds": time.time() - start,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_handcrafted):
    model.eval()
    total_loss = 0.0
    seen = 0
    all_probs = []
    all_labels = []
    all_gates = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        handcrafted = None
        if use_handcrafted:
            handcrafted = batch["handcrafted"].to(device, non_blocking=True)

        output = model(images, handcrafted)
        loss = criterion(output["logits"], labels)

        total_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        all_probs.append(torch.softmax(output["logits"].float(), dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if output["gate"] is not None:
            all_gates.append(output["gate"].float().cpu().numpy())

    return {
        "loss": total_loss / max(seen, 1),
        "probs": np.concatenate(all_probs) if all_probs else np.zeros((0, 4)),
        "labels": np.concatenate(all_labels) if all_labels else np.zeros((0,)),
        "gates": np.concatenate(all_gates) if all_gates else None,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, ema, epoch,
                    best_metric, config, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "ema_updates": ema.updates if ema is not None else 0,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "history": history,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    ema=None, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    if ema is not None and payload.get("ema"):
        ema.load_state_dict(payload["ema"])
        ema.updates = payload.get("ema_updates", 0)
    return payload


def write_history(path, history):
    Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")
