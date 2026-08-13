from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GLCM_DISTANCES = (1, 3)

def _build_feature_names() -> List[str]:
    names: List[str] = []

    names += [
        "int_mean", "int_std", "int_skew", "int_kurtosis", "int_entropy",
        "int_p10", "int_p25", "int_p50", "int_p75", "int_p90",
        "int_range_iqr", "int_cv",
    ]

    names += [
        "geom_area_frac", "geom_eccentricity", "geom_solidity",
        "geom_extent", "geom_aspect_ratio",
    ]

    names += [
        "asym_mean", "asym_std", "asym_max", "asym_p95",
        "asym_area_frac", "asym_centroid_dx", "asym_centroid_dy",
        "asym_q_upper_left", "asym_q_upper_right",
        "asym_q_lower_left", "asym_q_lower_right",
        "asym_lr_ratio", "asym_blob_area_frac", "asym_blob_compactness",
    ]

    for dist in GLCM_DISTANCES:
        for prop in GLCM_PROPS:
            names.append(f"glcm_d{dist}_{prop}")

    return names

FEATURE_NAMES: List[str] = _build_feature_names()
N_FEATURES: int = len(FEATURE_NAMES)

def intensity_features(gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    brain = mask > 0
    out: Dict[str, float] = {}

    if brain.sum() < 50:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("int_")}

    values = gray[brain].astype(np.float64)

    out["int_mean"] = float(values.mean())
    out["int_std"] = float(values.std())
    if values.std() < 1e-6:
        out["int_skew"] = 0.0
        out["int_kurtosis"] = 0.0
    else:
        out["int_skew"] = float(stats.skew(values))
        out["int_kurtosis"] = float(stats.kurtosis(values))

    hist = np.bincount(values.astype(np.int64), minlength=256).astype(np.float64)
    probs = hist / max(hist.sum(), 1.0)
    nonzero = probs[probs > 0]
    out["int_entropy"] = float(-(nonzero * np.log2(nonzero)).sum())

    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    out["int_p10"] = float(p10)
    out["int_p25"] = float(p25)
    out["int_p50"] = float(p50)
    out["int_p75"] = float(p75)
    out["int_p90"] = float(p90)
    out["int_range_iqr"] = float(p75 - p25)
    out["int_cv"] = float(values.std() / max(values.mean(), 1e-6))

    return out

def geometry_features(mask: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    area = float(binary.sum())

    if area < 50:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("geom_")}

    out["geom_area_frac"] = area / float(h * w)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("geom_")}

    contour = max(contours, key=cv2.contourArea)

    if len(contour) >= 5:
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
        if major > 1e-6:
            ratio = min(minor / major, 1.0)
            out["geom_eccentricity"] = float(np.sqrt(max(0.0, 1.0 - ratio ** 2)))
            out["geom_aspect_ratio"] = float(minor / major)
        else:
            out["geom_eccentricity"] = 0.0
            out["geom_aspect_ratio"] = 0.0
    else:
        out["geom_eccentricity"] = 0.0
        out["geom_aspect_ratio"] = 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    out["geom_solidity"] = float(area / hull_area) if hull_area > 1e-6 else 0.0

    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = float(bw * bh)
    out["geom_extent"] = float(area / bbox_area) if bbox_area > 1e-6 else 0.0

    return out

def asymmetry_features(asymmetry: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    brain = mask > 0

    if brain.sum() < 50 or asymmetry.max() == 0:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("asym_")}

    values = asymmetry[brain].astype(np.float64)
    h, w = asymmetry.shape

    out["asym_mean"] = float(values.mean())
    out["asym_std"] = float(values.std())
    out["asym_max"] = float(values.max())
    out["asym_p95"] = float(np.percentile(values, 95))

    masked_asym = np.where(brain, asymmetry, 0).astype(np.uint8)
    thresh_val, _ = cv2.threshold(
        masked_asym[brain].reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    blob = ((masked_asym > thresh_val) & brain).astype(np.uint8)

    brain_area = float(brain.sum())
    out["asym_area_frac"] = float(blob.sum()) / brain_area

    moments = cv2.moments(blob, binaryImage=True)
    if moments["m00"] > 1e-6:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        out["asym_centroid_dx"] = float((cx - w / 2.0) / (w / 2.0))
        out["asym_centroid_dy"] = float((cy - h / 2.0) / (h / 2.0))
    else:
        out["asym_centroid_dx"] = 0.0
        out["asym_centroid_dy"] = 0.0

    mid_y, mid_x = h // 2, w // 2
    global_mean = max(float(values.mean()), 1e-6)

    quadrants = {
        "asym_q_upper_left": (slice(0, mid_y), slice(0, mid_x)),
        "asym_q_upper_right": (slice(0, mid_y), slice(mid_x, w)),
        "asym_q_lower_left": (slice(mid_y, h), slice(0, mid_x)),
        "asym_q_lower_right": (slice(mid_y, h), slice(mid_x, w)),
    }
    quad_means: Dict[str, float] = {}
    for name, (ys, xs) in quadrants.items():
        sub_mask = brain[ys, xs]
        if sub_mask.sum() > 10:
            quad_means[name] = float(asymmetry[ys, xs][sub_mask].mean())
        else:
            quad_means[name] = 0.0
        out[name] = quad_means[name] / global_mean

    left = quad_means["asym_q_upper_left"] + quad_means["asym_q_lower_left"]
    right = quad_means["asym_q_upper_right"] + quad_means["asym_q_lower_right"]
    out["asym_lr_ratio"] = float((left + 1e-6) / (right + 1e-6))

    num, labels, stat_rows, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
    if num > 1:
        areas = stat_rows[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        largest_area = float(areas.max())
        out["asym_blob_area_frac"] = largest_area / brain_area

        blob_mask = (labels == largest_idx).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 1e-6:
                out["asym_blob_compactness"] = float(
                    4.0 * np.pi * largest_area / (perimeter ** 2)
                )
            else:
                out["asym_blob_compactness"] = 0.0
        else:
            out["asym_blob_compactness"] = 0.0
    else:
        out["asym_blob_area_frac"] = 0.0
        out["asym_blob_compactness"] = 0.0

    return out

def glcm_features(gray: np.ndarray, mask: np.ndarray, levels: int = 32
                  ) -> Dict[str, float]:
    out: Dict[str, float] = {}
    brain = mask > 0

    if brain.sum() < 100:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("glcm_")}

    quantised = (gray.astype(np.float32) / 256.0 * (levels - 1)).astype(np.uint8) + 1
    quantised = np.where(brain, quantised, 0).astype(np.uint8)

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    for dist in GLCM_DISTANCES:
        glcm = graycomatrix(
            quantised, distances=[dist], angles=angles,
            levels=levels + 1, symmetric=True, normed=False,
        )
        glcm = glcm[1:, 1:, :, :].astype(np.float64)
        totals = glcm.sum(axis=(0, 1), keepdims=True)
        totals[totals == 0] = 1.0
        glcm = glcm / totals

        for prop in GLCM_PROPS:
            try:
                vals = graycoprops(glcm, prop)
                out[f"glcm_d{dist}_{prop}"] = float(np.mean(vals))
            except (ValueError, IndexError):
                out[f"glcm_d{dist}_{prop}"] = 0.0

    return out

def extract_features(dip_output: dict) -> Tuple[np.ndarray, Dict[str, float]]:
    gray = dip_output["gray"]
    mask = dip_output["mask"]
    asymmetry = dip_output["asymmetry"]

    named: Dict[str, float] = {}
    named.update(intensity_features(gray, mask))
    named.update(geometry_features(mask))
    named.update(asymmetry_features(asymmetry, mask))
    named.update(glcm_features(gray, mask))

    vector = np.array([named.get(name, 0.0) for name in FEATURE_NAMES],
                      dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    return vector, named
