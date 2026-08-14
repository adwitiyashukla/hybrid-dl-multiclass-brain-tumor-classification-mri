import json
from pathlib import Path

import numpy as np


def apply_offsets(probs, offsets):
    logp = np.log(np.clip(probs, 1e-12, 1.0)) + np.asarray(offsets)[None, :]
    logp = logp - logp.max(axis=1, keepdims=True)
    adjusted = np.exp(logp)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def objective_value(labels, probs, offsets, objective, notumor_index):
    from metrics import compute_metrics
    adjusted = apply_offsets(probs, offsets)
    metrics = compute_metrics(labels, adjusted)
    if objective == "macro_f1":
        return metrics["macro_f1"]
    preds = adjusted.argmax(axis=1)
    true_tumor = labels != notumor_index
    pred_tumor = preds != notumor_index
    sensitivity = (pred_tumor & true_tumor).sum() / max(true_tumor.sum(), 1)
    return metrics["macro_f1"] + 0.5 * sensitivity


def coordinate_search(labels, probs, objective, notumor_index,
                      span=2.0, step=0.1, passes=3):
    n_classes = probs.shape[1]
    offsets = np.zeros(n_classes)
    best = objective_value(labels, probs, offsets, objective, notumor_index)
    grid = np.arange(-span, span + 1e-9, step)

    for _ in range(passes):
        improved = False
        for index in range(n_classes):
            current = offsets[index]
            for candidate in grid:
                offsets[index] = candidate
                value = objective_value(labels, probs, offsets, objective,
                                        notumor_index)
                if value > best + 1e-9:
                    best = value
                    current = candidate
                    improved = True
            offsets[index] = current
        if not improved:
            break

    return offsets - offsets.mean(), best


def load_offsets(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["offsets"]
