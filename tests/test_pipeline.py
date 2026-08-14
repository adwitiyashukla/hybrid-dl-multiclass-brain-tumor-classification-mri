import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from dip.handcrafted import FEATURE_NAMES, N_FEATURES, extract_features
from decision import apply_offsets, coordinate_search
from dip.preprocessing import (correct_bias_field, extract_brain,
                               run_dip_pipeline, stack_channels,
                               symmetry_analysis)
from make_phantom import make_phantom

try:
    import torch
    import timm

    from models.cbam import CBAM
    from models.fusion_net import HybridTumorNet

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch and timm are required for model tests"
)


@pytest.fixture(scope="module")
def tumour_case():
    image, centre = make_phantom(tumour=True, tumour_side="right", seed=1)
    return image, centre


@pytest.fixture(scope="module")
def healthy_case():
    image, _ = make_phantom(tumour=False, seed=2)
    return image


def test_apply_offsets_returns_valid_distribution():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1, (20, 4))
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    adjusted = apply_offsets(probs, [0.5, -0.2, 0.0, 0.1])

    assert adjusted.shape == probs.shape
    assert np.allclose(adjusted.sum(axis=1), 1.0)
    assert (adjusted >= 0).all()


def test_zero_offsets_leave_predictions_unchanged():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, (50, 4))
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    adjusted = apply_offsets(probs, [0.0, 0.0, 0.0, 0.0])

    assert np.allclose(adjusted, probs, atol=1e-9)


def test_offset_search_recovers_a_suppressed_class():
    rng = np.random.default_rng(2)
    n = 600
    labels = rng.integers(0, 4, n)
    logits = rng.normal(0, 1, (n, 4))
    logits[np.arange(n), labels] += 2.0
    logits[labels == 0, 0] -= 1.2
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)

    from metrics import compute_metrics

    before = compute_metrics(labels, probs)
    offsets, _ = coordinate_search(labels, probs, "macro_f1", 2)
    after = compute_metrics(labels, apply_offsets(probs, offsets))

    assert after["macro_f1"] >= before["macro_f1"]
    assert after["recall_glioma"] > before["recall_glioma"]
    assert abs(float(np.mean(offsets))) < 1e-9


def test_brain_extraction_plausible(tumour_case):
    image, _ = tumour_case
    mask, report = extract_brain(image)
    fraction = (mask > 0).mean()
    assert 0.25 < fraction < 0.75
    assert not report.brain_extraction_failed


def test_bias_correction_reduces_left_right_ramp(tumour_case):
    image, _ = tumour_case
    mask, _ = extract_brain(image)
    brain = mask > 0
    masked = np.where(brain, image, 0).astype(np.uint8)
    width = image.shape[1]

    left = masked[:, : width // 2][brain[:, : width // 2]].mean()
    right = masked[:, width // 2 :][brain[:, width // 2 :]].mean()
    before = abs(right - left) / max(left, right)

    corrected = correct_bias_field(masked, mask)
    left_c = corrected[:, : width // 2][brain[:, : width // 2]].mean()
    right_c = corrected[:, width // 2 :][brain[:, width // 2 :]].mean()
    after = abs(right_c - left_c) / max(left_c, right_c)

    assert after < before


def test_symmetry_localises_tumour(tumour_case):
    image, centre = tumour_case
    output = run_dip_pipeline(image, output_size=256)
    asymmetry = output["asymmetry"]
    positive = asymmetry[asymmetry > 0]
    assert positive.size > 0

    rows, cols = np.where(asymmetry > np.percentile(positive, 97))
    distance = np.hypot(cols.mean() - centre[0], rows.mean() - centre[1])
    assert distance < 45


def peak_region_centroid(asymmetry, percentile=97):
    positive = asymmetry[asymmetry > 0]
    if positive.size < 10:
        return float("nan")
    threshold = np.percentile(positive, percentile)
    rows, cols = np.where(asymmetry >= threshold)
    return float(cols.mean())


def test_asymmetry_map_tracks_lesion_side():
    centroids = {}
    positions = {}
    for side in ("left", "right"):
        image, centre = make_phantom(tumour=True, tumour_side=side, seed=1)
        output = run_dip_pipeline(image, output_size=256)
        centroids[side] = peak_region_centroid(output["asymmetry"].astype(np.float64))
        positions[side] = centre[0]

    observed = centroids["right"] - centroids["left"]
    expected = positions["right"] - positions["left"]

    assert observed > 0
    assert observed > 0.25 * expected


def test_tumour_more_asymmetric_than_healthy(tumour_case, healthy_case):
    image, _ = tumour_case
    tumour_output = run_dip_pipeline(image, output_size=256)
    healthy_output = run_dip_pipeline(healthy_case, output_size=256)
    assert tumour_output["asymmetry"].mean() > healthy_output["asymmetry"].mean()


def test_feature_vector_shape_and_validity(tumour_case):
    image, _ = tumour_case
    output = run_dip_pipeline(image, output_size=256)
    vector, named = extract_features(output)

    assert vector.shape == (N_FEATURES,)
    assert vector.dtype == np.float32
    assert not np.isnan(vector).any()
    assert not np.isinf(vector).any()
    assert len(named) == len(FEATURE_NAMES)


def test_feature_names_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_channel_stacking(tumour_case, n_channels):
    image, _ = tumour_case
    output = run_dip_pipeline(image, output_size=256)
    stacked = stack_channels(output, n_channels)
    assert stacked.shape == (256, 256, n_channels)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((256, 256), np.uint8),
        np.full((256, 256), 255, np.uint8),
        np.zeros((16, 16), np.uint8),
    ],
)


def test_degenerate_inputs_do_not_crash(image):
    output = run_dip_pipeline(image, output_size=128)
    vector, _ = extract_features(output)
    assert not np.isnan(vector).any()


def test_symmetry_analysis_returns_report(tumour_case):
    image, _ = tumour_case
    mask, _ = extract_brain(image)
    asymmetry, report = symmetry_analysis(image, mask)
    assert asymmetry.shape == image.shape
    assert -15.0 <= report.symmetry_angle <= 15.0


def test_pipeline_switches_disable_stages(tumour_case):
    image, _ = tumour_case
    output = run_dip_pipeline(
        image, do_symmetry=False, do_clahe=False, output_size=128
    )
    assert output["asymmetry"].max() == 0


@requires_torch
def test_cbam_preserves_shape():
    block = CBAM(32)
    x = torch.randn(2, 32, 16, 16)
    assert block(x).shape == x.shape


@requires_torch
def test_model_forward_and_backward():
    model = HybridTumorNet(
        backbone="resnet18", n_classes=4, in_chans=3,
        n_handcrafted=N_FEATURES, pretrained=False,
    )
    images = torch.randn(2, 3, 128, 128)
    handcrafted = torch.randn(2, N_FEATURES)

    output = model(images, handcrafted)
    assert output["logits"].shape == (2, 4)
    assert output["gate"].shape == (2,)
    assert output["fmap"].ndim == 4

    output["logits"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


@requires_torch
def test_model_without_handcrafted_branch():
    model = HybridTumorNet(
        backbone="resnet18", n_classes=4, in_chans=1,
        use_handcrafted=False, use_cbam=False, pretrained=False,
    )
    output = model(torch.randn(2, 1, 128, 128))
    assert output["logits"].shape == (2, 4)
    assert output["gate"] is None


@requires_torch
def test_class_balanced_weights_favour_rare_classes():
    from losses import ClassBalancedFocalLoss, class_balanced_weights

    weights = class_balanced_weights([1000, 100, 50, 500])
    assert weights[2] > weights[0]
    assert abs(float(weights.sum()) - 4.0) < 1e-4

    criterion = ClassBalancedFocalLoss(weights=weights, gamma=2.0)
    loss = criterion(torch.randn(8, 4), torch.randint(0, 4, (8,)))
    assert torch.isfinite(loss)


@requires_torch
def test_gradcam_matches_image_size():
    from explain import GradCAMPlusPlus

    model = HybridTumorNet(
        backbone="resnet18", n_classes=4, in_chans=3,
        n_handcrafted=N_FEATURES, pretrained=False,
    )
    engine = GradCAMPlusPlus(model)
    cam, indices = engine(torch.randn(2, 3, 128, 128), torch.randn(2, N_FEATURES))
    engine.remove()

    assert cam.shape == (2, 128, 128)
    assert indices.shape == (2,)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
