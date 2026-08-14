# Hybrid DL for Multi-Class Brain Tumor Classification using MRI: Healthcare Management System

Four-class brain tumor classification from axial MRI slices, combining a classical
digital image processing stage with a convolutional neural network, wrapped in a
patient record and reporting system.

Live demo: https://huggingface.co/spaces/adwitiyashukla/hybrid-dl-multiclass-brain-tumor-classification-mri

## Overview

The system classifies a brain MRI slice as glioma, meningioma, pituitary adenoma, or
no tumor. Rather than passing raw pixels to a pretrained network, it first runs a six
stage image processing pipeline that removes acquisition artifacts and derives an
explicit anatomical prior, then fuses those signals with learned features.

The central idea is bilateral symmetry. A healthy brain is approximately symmetric
about the midsagittal plane, and a space occupying lesion breaks that symmetry. The
pipeline locates the midsagittal plane by searching over small rotations and shifts
for the transform that minimises the difference between the slice and its mirror,
then uses the residual difference map as a tumor saliency prior. That map is supplied
to the network as a dedicated input channel and is also summarised into handcrafted
features.

## Pipeline

| Stage | Method | Purpose |
|---|---|---|
| Skull stripping | Otsu threshold, morphological opening, largest component, hole fill | Removes skull, scalp and background so the model cannot key on head size or field of view |
| Bias field correction | Homomorphic low pass division | Corrects smooth multiplicative coil sensitivity falloff |
| Denoising | Non-local means | Suppresses Rician noise without blurring lesion boundaries |
| Intensity normalisation | Z-score inside the brain mask with percentile clipping | MRI has no absolute intensity scale, so raw values are not comparable across scanners |
| Contrast enhancement | CLAHE | Amplifies local lesion boundary contrast without global histogram domination |
| Symmetry analysis | Midsagittal search, mirror, difference map | Produces the bilateral asymmetry prior |

Stage order matters. Bias correction precedes normalisation because a multiplicative
field distorts the very statistics that normalisation uses, and symmetry analysis runs
last because an uncorrected left to right intensity gradient would otherwise appear as
false asymmetry.

## Architecture

```
input slice (1, 256, 256)
        |
  DIP pipeline
        |
  3 channels: processed slice, brain mask, asymmetry map
        |                                  43 handcrafted features
  EfficientNet-B0 (timm, ImageNet init)         |
        |                                  BatchNorm + MLP 43 -> 256 -> 128
  CBAM attention                                |
        |                                       |
  global average pool -> 1280                   |
        |                                       |
        +-------------------+-------------------+
                            |
                     gated fusion -> 256
                            |
                  linear -> 4 class logits
                            |
             class balanced focal loss
```

The fusion gate is a learned per sample mixing weight between the CNN stream and the
handcrafted stream. It is returned with every prediction, so the system reports not
only what it predicted but how much it relied on each source of evidence.

Handcrafted features cover four groups: first order intensity statistics inside the
brain mask, brain mask geometry, bilateral asymmetry descriptors including quadrant
distribution and blob compactness, and GLCM Haralick texture at two distances
averaged over four angles.

## Dataset

Brain Tumor MRI Dataset, 7200 T1 weighted contrast enhanced slices across four
classes, assembled from the figshare, SARTAJ and Br35H collections. The release used
here is class balanced: 1400 training and 400 test images per class.

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Download and extract so the layout is:

```
data/
  Training/
    glioma/  meningioma/  notumor/  pituitary/
  Testing/
    glioma/  meningioma/  notumor/  pituitary/
```

The dataset ships with its own train and test split. That split is used as provided
rather than reshuffled, because the source collections contain multiple adjacent
slices from the same patient without patient identifiers. Re-splitting at the image
level would place near duplicate slices of one patient in both train and test and
inflate the reported scores.

## Preprocessing quality

Measured over all 7200 images by `scripts/precompute_dip.py`, which writes a per image
quality report to `data/cache/quality_<split>.csv`.

| Metric | Training (5600) | Testing (1600) |
|---|---|---|
| Brain extraction fallback | 77 images, 1.38 percent | 22 images, 1.38 percent |
| Low contrast flagged | 0, 0.00 percent | 0, 0.00 percent |
| Mean brain fraction | 0.299 | 0.304 |
| Mean symmetry score | 21.37 | 22.00 |

Otsu based skull stripping is not perfect. When the resulting mask covers an
implausible fraction of the frame the pipeline falls back to a centred ellipse and
records a flag, rather than passing a bad mask silently downstream. The fallback rate
is reported here because a pipeline that fails on some fraction of its inputs without
measuring it is worse than one that measures and states the number. The two splits
producing the same rate independently indicates the behaviour is stable rather than
tuned to one subset.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Precompute the image processing cache and handcrafted features once:

```
python scripts/precompute_dip.py --config configs/base.yaml
```

Train:

```
python scripts/train.py --config configs/base.yaml --resume auto
```

Training checkpoints every epoch and `--resume auto` restores model, optimiser,
scheduler, gradient scaler, exponential moving average and epoch counter. This is
required for free tier Colab sessions, which disconnect.

Evaluate on the held out test split:

```
python scripts/evaluate.py --config configs/base.yaml --checkpoint runs/e6_full/best.pt
```

Evaluation reports macro F1, balanced accuracy, per class precision and recall,
macro AUC, Cohen kappa, expected calibration error, tumor sensitivity and
specificity, and bootstrap confidence intervals over 2000 resamples.

## A note on the loss

The training objective is class balanced focal loss. On this balanced release the
class balanced weights reduce to uniform and the weighted sampler becomes a no op, so
the objective is effectively plain focal loss with label smoothing. Both mechanisms
are retained because the original unbalanced release of this dataset and the BraTS
volumes targeted as a follow up are not balanced, and because the ablation rows change
the training subset composition.

## Ablations

Each row is reproduced by editing `configs/base.yaml`. Run three seeds per row and
report mean and standard deviation.

| Row | Change | Config keys |
|---|---|---|
| E0 | No image processing, raw slice | all `dip` flags false, `in_chans: 1` |
| E1 | Add skull stripping | `do_skull_strip: true` |
| E2 | Add bias correction and normalisation | `do_bias_correction`, `do_normalise` |
| E3 | Add CLAHE and denoising | `do_clahe`, `do_denoise` |
| E4 | Add asymmetry channel | `do_symmetry: true`, `in_chans: 3` |
| E5 | Add handcrafted branch | `use_handcrafted: true` |
| E6 | Add CBAM attention | `use_cbam: true` |
| E7 | Remove ImageNet initialisation | `pretrained: false` |

## Results

Measured on the 1600 image held out test split, using the checkpoint selected by
validation macro F1. Confidence intervals are bootstrap percentile intervals over 2000
resamples of the test set. This is the configuration deployed in the live demo.

| Metric | Value |
|---|---|
| Macro F1 | 0.9407, 95 percent CI [0.9286, 0.9515] |
| Balanced accuracy | 0.9419, 95 percent CI [0.9309, 0.9519] |
| Macro AUC | 0.9878 |
| Cohen kappa | 0.9225 |
| Expected calibration error | 0.0098 |
| Tumor sensitivity | 0.9750 |
| Tumor specificity | 0.9950 |
| Mean fusion gate | 0.476 |

Per class:

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| glioma | 0.991 | 0.820 | 0.897 | 400 |
| meningioma | 0.899 | 0.958 | 0.927 | 400 |
| notumor | 0.930 | 0.995 | 0.961 | 400 |
| pituitary | 0.959 | 0.995 | 0.977 | 400 |

Confusion matrix, rows are true class, columns are predicted:

```
              glioma  mening  notumor  pituit
glioma           328      40       25       7
meningioma         2     383        5      10
notumor            0       2      398       0
pituitary          1       1        0     398
```

### Reading these results honestly

**Validation reported 0.9798 balanced accuracy and the test set gives 0.9419.** The gap
is expected, since the checkpoint was chosen by validation score. The test number is
the one quoted here and the validation number is not.

**Glioma is the weakest class at 0.820 recall.** 72 of 400 gliomas were misclassified,
40 of them as meningioma. Gliomas are infiltrative and heterogeneous in appearance, so
this is the expected direction of failure.

**25 gliomas were classified as no tumor.** These are the clinically consequential
errors, since a missed tumor carries far more cost than a confusion between two tumor
types. Overall tumor sensitivity is 0.9750, meaning 30 of 1200 tumor cases were called
healthy, and 25 of those 30 were gliomas. Any future work should target glioma recall
specifically rather than overall accuracy.

**Calibration is good.** An expected calibration error of 0.0098 means predicted
confidence tracks observed accuracy closely, so the confidence value shown in the
application is meaningful rather than decorative.

### Experiment: does the asymmetry map need to preserve lesion side?

The bilateral asymmetry map was originally computed as `abs(I - mirror(I))`. That
quantity is symmetric by construction, since the value at position x and at its mirror
position are identical, so the map highlights the lesion and its mirror image equally
and carries no information about which side the lesion is actually on.

Measured on a synthetic phantom with a tumor at a known position, moving the tumor 76
pixels across the midline moved the peak of the `abs` map by 0.0 pixels, under bias
field swings from 0 to 60 percent. Replacing it with `relu(I - mirror(I))`, which keeps
only the hyperintense side, moved the peak by 35 to 44 pixels, recovering roughly half
the true displacement and remaining stable across the same bias range.

Both formulations were then trained end to end under identical conditions. The
preprocessing quality metrics were byte identical between the two runs, so the
asymmetry map was the only variable.

| Metric | `abs`, no laterality | `relu`, keeps laterality |
|---|---|---|
| Macro F1 | 0.9435 [0.9324, 0.9545] | 0.9407 [0.9286, 0.9515] |
| Balanced accuracy | 0.9444 | 0.9419 |
| Macro AUC | 0.9905 | 0.9878 |
| Expected calibration error | 0.0135 | 0.0098 |
| Tumor sensitivity | 0.9833 | 0.9750 |
| Gliomas called no tumor | 18 | 25 |

**The change did not improve accuracy.** Macro F1 moved by -0.0028 against a confidence
interval roughly 0.022 wide, so the two configurations are not distinguishable on a
single seed each. Calibration improved slightly and tumor sensitivity degraded slightly,
and neither movement is resolvable at this sample size.

The most likely explanation is that the laterality information was never missing from
the model's point of view. The asymmetry map is supplied as one of three input channels
alongside the processed slice itself, and the slice plainly shows which side the lesion
is on, so the convolutional stream can recover laterality without help from the
asymmetry channel.

The `relu` formulation is the one deployed, on the grounds that it is the better
motivated quantity and calibrates slightly better, not on the grounds that it is more
accurate. Settling the question properly would require the three seeds per configuration
prescribed in the ablation protocol below, which was not run.

## Tests

```
pytest
```

The suite verifies that skull stripping produces a plausible brain fraction, that
bias correction measurably reduces a synthetic left to right intensity ramp, that
the asymmetry map localises a known tumor position on a synthetic phantom, that
feature vectors are finite and correctly shaped, that degenerate inputs do not
crash the pipeline, and that the model performs a valid forward and backward pass.

## Project structure

```
configs/          experiment configuration
scripts/          precompute, train, evaluate entry points
src/dip/          classical image processing and handcrafted features
src/models/       CBAM attention and the hybrid fusion network
src/data/         dataset, splitting and caching
src/engine.py     training loop, mixed precision, EMA, checkpointing
src/metrics.py    evaluation metrics and bootstrap intervals
src/explain.py    Grad-CAM++ and attribution overlays
tests/            pytest suite and synthetic phantom generator
```

## Limitations

This is a research prototype and not a medical device. It has not been clinically
validated and must not be used to inform diagnosis or treatment.

- Trained on preprocessed 2D slices rather than full volumes, so it does not use
  through plane context that a radiologist would use.
- Single dataset with no external validation on data from a different institution.
- Patient identifiers are absent from the public dataset, so patient level leakage
  between the provided train and test splits cannot be fully ruled out.
- The bilateral symmetry prior is weakest for midline lesions. Pituitary adenomas sit
  on the midline at the skull base and produce approximately symmetric abnormality,
  so the asymmetry channel is expected to contribute least for that class.
- The model predicts four classes only and has no mechanism to flag an input that
  falls outside them.

## License

MIT. See LICENSE.
