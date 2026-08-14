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

### Train and test overlap in this dataset

Before any metric below is read, this has to be stated. **About 26 percent of the
supplied test split consists of duplicates of training images.**

`scripts/find_duplicates.py` embeds every image as a 32x32 mean centred vector and
computes cosine similarity between the two splits. At a threshold of 0.999, 420 of 1600
test images match a training image, and the closest pairs sit at similarity 1.000000,
meaning pixel identical. 418 of those 420 pairs carry the same class label, which rules
out coincidental anatomical similarity: were these chance matches between different
scans, class agreement would be near 25 percent, not 99.5 percent. The duplicates carry
different filenames in each split, which is why a filename comparison finds nothing.

The overlap is very unevenly distributed:

| Class | Duplicated from training | Clean test images remaining |
|---|---|---|
| notumor | 310 of 400, 77.5 percent | 90 |
| meningioma | 103 of 400, 25.8 percent | 297 |
| glioma | 7 of 400, 1.8 percent | 393 |
| pituitary | 0 of 400, 0.0 percent | 400 |

Reproduce with:

```
python scripts/find_duplicates.py --config configs/base.yaml
```

### Primary result, leak free test split

Computed on the 1180 test images with no near duplicate in the training split, using
tuned decision offsets fitted on validation. This is the honest number for this model.

| Metric | Value |
|---|---|
| Macro F1 | 0.9190, 95 percent CI [0.9011, 0.9365] |
| Balanced accuracy | 0.9428, 95 percent CI [0.9312, 0.9540] |
| Macro AUC | 0.9867 |
| Cohen kappa | 0.8987 |
| Expected calibration error | 0.0186 |
| Tumor sensitivity | 0.9798 |
| Tumor specificity | 1.0000 |
| Mean fusion gate | 0.452 |

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| glioma | 0.994 | 0.824 | 0.901 | 393 |
| meningioma | 0.875 | 0.946 | 0.909 | 297 |
| notumor | 0.804 | 1.000 | 0.891 | 90 |
| pituitary | 0.950 | 1.000 | 0.974 | 400 |

```
              glioma  mening  notumor  pituit
glioma           324      40       21       8
meningioma         2     281        1      13
notumor            0       0       90       0
pituitary          0       0        0     400
```

### What deduplication changed, and what it did not

| Metric | Full test split, 1600 | Leak free split, 1180 |
|---|---|---|
| Macro F1 | 0.9412 | 0.9190 |
| Balanced accuracy | 0.9425 | 0.9428 |
| Macro AUC | 0.9881 | 0.9867 |
| Tumor sensitivity | 0.9800 | 0.9798 |
| notumor precision | 0.943 | 0.804 |
| notumor support | 400 | 90 |

**Balanced accuracy barely moved, from 0.9425 to 0.9428.** That is not a contradiction.
Balanced accuracy is mean per class recall, and recall measures performance on each
class in isolation, so removing duplicated examples of a class the model already handles
well leaves it essentially unchanged.

**Macro F1 fell by 0.022** because it incorporates precision. Removing 310 of 400
notumor images left only 90 true negatives to absorb the false positives arriving from
other classes, so notumor precision dropped from 0.943 to 0.804 and dragged the macro
average down.

An honest caveat on the leak free subset: because the duplication was concentrated in
one class, the remaining subset is itself imbalanced at 393, 297, 90 and 400. Part of
the shift in precision based metrics is attributable to that changed class balance
rather than to leakage alone. Separating those two effects cleanly would require a
dataset where the overlap is uniform across classes, which this one is not.

The reported drop is therefore best read as a lower bound on the optimism in the naive
protocol. The perceptual hash used is also a lower bound in a second sense: it detects
duplicates and near duplicates, but a rotated or reflected copy of a training image
would not be matched by it and would remain undetected in the test split.

### Decision offsets and what they can and cannot fix

The default decision rule is argmax over the four class probabilities. That is only one
possible operating point, and it is not obviously the right one here: glioma precision
is 0.991 against recall 0.820, so the model is highly reluctant to predict glioma and
almost always correct when it does. In a clinical setting a missed tumor costs more than
a confusion between two tumor types, which argues for trading some of that precision.

`scripts/tune_thresholds.py` fits one additive log-space offset per class on the
validation split by coordinate ascent, then `scripts/evaluate.py --offsets` applies them
once to the test split. The offsets are never fitted on test data.

| Metric | argmax | tuned offsets |
|---|---|---|
| Macro F1 | 0.9407 [0.9286, 0.9515] | 0.9412 [0.9292, 0.9520] |
| Balanced accuracy | 0.9419 | 0.9425 |
| Tumor sensitivity | 0.9750 | 0.9800 |
| Missed tumors, of 1200 | 30 | 24 |
| Gliomas called no tumor | 25 | 21 |
| Glioma recall | 0.820 | 0.823 |
| Expected calibration error | 0.0098 | 0.0118 |

Fitted offsets: glioma -0.12, meningioma +0.07, notumor -0.32, pituitary +0.38. The
largest adjustment suppresses the no tumor class, which is the intended direction.

**Missed tumors fell by 20 percent at no cost to macro F1, so the offsets are applied in
the deployed model.** Calibration degraded slightly, which is expected since shifting
the decision rule moves probabilities away from the values the network was trained to
produce.

**Glioma recall was not fixed.** It moved 0.820 to 0.823. The reason is visible in the
confusion matrix: 42 gliomas are still classified as meningioma, and those cases are not
near a decision boundary, so no threshold can recover them. The model represents them as
meningioma-like. Fixing that requires changing what the model learns, not how its outputs
are thresholded, and it is the single most valuable direction for further work.

A related observation worth recording: **validation glioma recall is 0.990 while test
glioma recall is 0.820.** The validation split is carved from the same `Training` folder
and therefore shares its distribution, whereas the supplied `Testing` folder is
measurably different. The glioma weakness is not visible on validation at all, which is
why it could not be tuned away, and it is direct evidence of a distribution shift between
the two folders rather than a simple random holdout.

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
