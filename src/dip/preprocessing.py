from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Tuple

import cv2
import numpy as np

@dataclass
class DIPReport:
    brain_fraction: float = 0.0
    brain_extraction_failed: bool = False
    symmetry_angle: float = 0.0
    symmetry_shift: float = 0.0
    symmetry_score: float = 0.0
    asymmetry_peak: float = 0.0
    low_contrast: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

def extract_brain(
    gray: np.ndarray,
    min_area_frac: float = 0.03,
    max_area_frac: float = 0.95,
) -> Tuple[np.ndarray, DIPReport]:
    report = DIPReport()
    h, w = gray.shape[:2]

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_thresh, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    fg = blurred[binary > 0]
    bg = blurred[binary == 0]
    if fg.size > 0 and bg.size > 0:
        report.low_contrast = float(fg.mean() - bg.mean()) < 20.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    mask = _largest_component(opened)
    mask = _fill_holes(mask)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, erode_kernel, iterations=1)
    mask = _largest_component(mask)

    area_frac = float((mask > 0).sum()) / float(h * w)
    report.brain_fraction = area_frac

    if area_frac < min_area_frac or area_frac > max_area_frac:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            mask, (w // 2, h // 2), (int(w * 0.38), int(h * 0.42)),
            0, 0, 360, 255, -1,
        )
        report.brain_extraction_failed = True
        report.brain_fraction = float((mask > 0).mean())

    return mask, report

def _largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8) * 255

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    flooded = padded.copy()
    cv2.floodFill(flooded, ff_mask, (0, 0), 1)
    holes = flooded == 0
    filled = (padded | holes).astype(np.uint8)
    return filled[1:-1, 1:-1] * 255

def correct_bias_field(
    gray: np.ndarray,
    mask: np.ndarray,
    sigma: float = 30.0,
) -> np.ndarray:
    img = gray.astype(np.float32)
    brain = mask > 0

    if brain.sum() < 100:
        return gray.copy()

    brain_mean = float(img[brain].mean())
    filled = np.where(brain, img, brain_mean).astype(np.float32)

    bias = cv2.GaussianBlur(filled, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bias = np.maximum(bias, 1e-3)

    corrected = img / bias * brain_mean
    corrected = np.where(brain, corrected, 0.0)

    return np.clip(corrected, 0, 255).astype(np.uint8)

def denoise(gray: np.ndarray, strength: float = 6.0) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=strength,
                                    templateWindowSize=7, searchWindowSize=21)

def normalise_intensity(
    gray: np.ndarray,
    mask: np.ndarray,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    img = gray.astype(np.float32)
    brain = mask > 0

    if brain.sum() < 100:
        return gray.copy()

    values = img[brain]
    lo, hi = np.percentile(values, clip_percentiles)
    if hi - lo < 1e-6:
        return gray.copy()

    img = np.clip(img, lo, hi)
    values = img[brain]

    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        return gray.copy()

    z = (img - mean) / std
    scaled = (z + 3.0) * (255.0 / 6.0)
    scaled = np.where(brain, scaled, 0.0)

    return np.clip(scaled, 0, 255).astype(np.uint8)

def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0,
                tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)

def symmetry_analysis(
    gray: np.ndarray,
    mask: np.ndarray,
    angle_range: float = 12.0,
    angle_step: float = 2.0,
    shift_range: int = 16,
    shift_step: int = 2,
    search_size: int = 64,
) -> Tuple[np.ndarray, DIPReport]:
    report = DIPReport()
    h, w = gray.shape[:2]

    small = cv2.resize(gray, (search_size, search_size), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (search_size, search_size),
                            interpolation=cv2.INTER_NEAREST) > 0

    if small_mask.sum() < 50:
        return np.zeros((h, w), dtype=np.uint8), report

    scale = search_size / float(w)
    best = (np.inf, 0.0, 0.0)

    angles = np.arange(-angle_range, angle_range + 1e-6, angle_step)
    shifts = np.arange(-shift_range, shift_range + 1, shift_step) * scale

    small_f = small.astype(np.float32)

    for angle in angles:
        rotated = _warp(small_f, angle, 0.0)
        rot_mask = _warp(small_mask.astype(np.float32), angle, 0.0) > 0.5
        for shift in shifts:
            shifted = _warp(rotated, 0.0, shift)
            sh_mask = _warp(rot_mask.astype(np.float32), 0.0, shift) > 0.5

            mirrored = shifted[:, ::-1]
            mirror_mask = sh_mask[:, ::-1]
            both = sh_mask & mirror_mask
            if both.sum() < 50:
                continue

            score = float(np.abs(shifted[both] - mirrored[both]).mean())
            if score < best[0]:
                best = (score, float(angle), float(shift))

    _, best_angle, best_shift_small = best
    best_shift = best_shift_small / scale

    report.symmetry_angle = best_angle
    report.symmetry_shift = best_shift
    report.symmetry_score = float(best[0]) if np.isfinite(best[0]) else 0.0

    full = _warp(gray.astype(np.float32), best_angle, best_shift * scale * (w / search_size))
    full_mask = _warp((mask > 0).astype(np.float32), best_angle,
                      best_shift * scale * (w / search_size)) > 0.5

    mirrored = full[:, ::-1]
    mirror_mask = full_mask[:, ::-1]
    valid = full_mask & mirror_mask

    diff = np.abs(full - mirrored)
    diff = np.where(valid, diff, 0.0)

    diff = _warp(diff, -best_angle, -best_shift * scale * (w / search_size))

    peak = float(diff.max())
    report.asymmetry_peak = peak

    if peak > 1e-6:
        diff = diff * (255.0 / peak)

    return np.clip(diff, 0, 255).astype(np.uint8), report

def _warp(img: np.ndarray, angle_deg: float, shift_x: float) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    matrix[0, 2] += shift_x
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def run_dip_pipeline(
    image: np.ndarray,
    do_skull_strip: bool = True,
    do_bias_correction: bool = True,
    do_denoise: bool = True,
    do_normalise: bool = True,
    do_clahe: bool = True,
    do_symmetry: bool = True,
    output_size: int = 256,
) -> dict:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"expected a 2D or 3D array, got shape {image.shape}")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    gray = cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_AREA)

    report = DIPReport()

    if do_skull_strip:
        mask, report = extract_brain(gray)
        gray = np.where(mask > 0, gray, 0).astype(np.uint8)
    else:
        mask = np.full(gray.shape, 255, dtype=np.uint8)

    if do_bias_correction:
        gray = correct_bias_field(gray, mask)

    if do_denoise:
        gray = denoise(gray)

    if do_normalise:
        gray = normalise_intensity(gray, mask)

    if do_clahe:
        gray = apply_clahe(gray)
        gray = np.where(mask > 0, gray, 0).astype(np.uint8)

    if do_symmetry:
        asymmetry, sym_report = symmetry_analysis(gray, mask)
        sym_report.brain_fraction = report.brain_fraction
        sym_report.brain_extraction_failed = report.brain_extraction_failed
        sym_report.low_contrast = report.low_contrast
        report = sym_report
    else:
        asymmetry = np.zeros(gray.shape, dtype=np.uint8)

    return {
        "gray": gray,
        "mask": mask,
        "asymmetry": asymmetry,
        "report": report.as_dict(),
    }

def stack_channels(dip_output: dict, n_channels: int = 3) -> np.ndarray:
    gray = dip_output["gray"]
    if n_channels == 1:
        return gray[:, :, None]
    if n_channels == 2:
        return np.dstack([gray, dip_output["mask"]])
    if n_channels == 3:
        return np.dstack([gray, dip_output["mask"], dip_output["asymmetry"]])
    raise ValueError(f"n_channels must be 1, 2 or 3, got {n_channels}")
