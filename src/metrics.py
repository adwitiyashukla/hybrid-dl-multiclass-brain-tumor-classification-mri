from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASS_NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        bin_accuracy = accuracies[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += (count / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    if class_names is None:
        class_names = CLASS_NAMES

    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = probs.argmax(axis=1)
    n_classes = probs.shape[1]

    out: Dict[str, float] = {}

    out["accuracy"] = float(accuracy_score(labels, preds))
    out["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
    out["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    out["weighted_f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
    out["cohen_kappa"] = float(cohen_kappa_score(labels, preds))

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(n_classes)), zero_division=0
    )
    for i, name in enumerate(class_names[:n_classes]):
        out[f"precision_{name}"] = float(precision[i])
        out[f"recall_{name}"] = float(recall[i])
        out[f"f1_{name}"] = float(f1[i])
        out[f"support_{name}"] = int(support[i])

    try:
        out["macro_auc"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        out["macro_auc"] = float("nan")

    out["ece"] = expected_calibration_error(probs, labels)

    if n_classes == 4 and "notumor" in list(class_names):
        notumor_idx = list(class_names).index("notumor")
        true_tumor = labels != notumor_idx
        pred_tumor = preds != notumor_idx
        if true_tumor.sum() > 0:
            out["tumor_sensitivity"] = float(
                (pred_tumor & true_tumor).sum() / true_tumor.sum()
            )
        if (~true_tumor).sum() > 0:
            out["tumor_specificity"] = float(
                ((~pred_tumor) & (~true_tumor)).sum() / (~true_tumor).sum()
            )

    return out


def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n = len(labels)

    values: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        value = compute_metrics(labels[idx], probs[idx]).get(metric, np.nan)
        if not np.isnan(value):
            values.append(value)

    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}

    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.percentile(arr, 100 * alpha / 2)),
        "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "std": float(arr.std()),
    }


def paired_bootstrap(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)

    diffs: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        a = compute_metrics(labels[idx], probs_a[idx]).get(metric, np.nan)
        b = compute_metrics(labels[idx], probs_b[idx]).get(metric, np.nan)
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(b - a)

    if not diffs:
        return {"mean_diff": float("nan")}

    arr = np.array(diffs)
    return {
        "mean_diff": float(arr.mean()),
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "prob_b_better": float((arr > 0).mean()),
    }


def confusion(labels: np.ndarray, preds: np.ndarray, n_classes: int = 4) -> np.ndarray:
    return confusion_matrix(labels, preds, labels=list(range(n_classes)))


def format_report(metrics: Dict[str, float],
                  class_names: Optional[Sequence[str]] = None) -> str:
    if class_names is None:
        class_names = CLASS_NAMES

    lines = [
        "-" * 58,
        f"{'Balanced accuracy':<26} {metrics.get('balanced_accuracy', float('nan')):.4f}",
        f"{'Macro F1':<26} {metrics.get('macro_f1', float('nan')):.4f}",
        f"{'Macro AUC':<26} {metrics.get('macro_auc', float('nan')):.4f}",
        f"{'Cohen kappa':<26} {metrics.get('cohen_kappa', float('nan')):.4f}",
        f"{'Expected calib. error':<26} {metrics.get('ece', float('nan')):.4f}",
        f"{'Accuracy (deprioritised)':<26} {metrics.get('accuracy', float('nan')):.4f}",
    ]
    if "tumor_sensitivity" in metrics:
        lines += [
            "-" * 58,
            f"{'Tumor sensitivity':<26} {metrics['tumor_sensitivity']:.4f}",
            f"{'Tumor specificity':<26} {metrics.get('tumor_specificity', float('nan')):.4f}",
        ]
    lines += ["-" * 58, f"{'class':<14}{'prec':>8}{'recall':>9}{'f1':>8}{'n':>7}"]
    for name in class_names:
        if f"f1_{name}" in metrics:
            lines.append(
                f"{name:<14}{metrics[f'precision_{name}']:>8.3f}"
                f"{metrics[f'recall_{name}']:>9.3f}"
                f"{metrics[f'f1_{name}']:>8.3f}"
                f"{metrics[f'support_{name}']:>7d}"
            )
    lines.append("-" * 58)
    return "\n".join(lines)
