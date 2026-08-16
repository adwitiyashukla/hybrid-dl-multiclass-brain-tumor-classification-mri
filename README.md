# Hybrid DL for Multi-Class Brain Tumor Classification using MRI: Healthcare Management System

A four class brain tumor classifier for MRI slices, with a classical image processing
stage in front of the network and a small web app around it for storing scans against
patient records.

Live demo: https://huggingface.co/spaces/adwitiyashukla/hybrid-dl-multiclass-brain-tumor-classification-mri

## Overview

The obvious version of this project takes the Kaggle brain tumor dataset, fine-tunes a
pretrained ResNet on it, reports something like 99 percent accuracy and finishes in an
afternoon. I started that way. It works, and I don't think it means very much.

Two reasons. MRI has no absolute intensity scale. A CT scanner gives you Hounsfield units,
where 0 is water and -1000 is air, and those mean the same thing on every machine. MRI
intensity depends on the scanner, the coil, the sequence settings and the gain, so the same
tissue in the same patient can come out as completely different numbers on two different
days. Feed raw pixels to a network and part of what it learns is the scanner.

The dataset is also stitched together from three separate collections that differ in head
size, field of view and how much neck is in the frame. Those correlate with which collection
an image came from, which correlates with the class balance, so a model can score well by
learning the collection instead of the tumor.

Most of the work here is therefore in front of the network. Then I spent a while checking
whether my own test numbers were real, and found they were not.

## Dataset

Brain Tumor MRI Dataset from Kaggle, 7200 T1 weighted contrast enhanced slices in four
classes: glioma, meningioma, pituitary adenoma, and no tumor.

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The copy I downloaded is class balanced, 1400 training and 400 test images per class. It
ships with its own Training and Testing folders and I used those as given rather than
reshuffling, because the source collections contain several adjacent slices from the same
patient and there are no patient IDs to group on. Splitting at the image level myself would
have put near identical slices of one patient on both sides.

Extract it so the layout is:

```
data/
  Training/
    glioma/  meningioma/  notumor/  pituitary/
  Testing/
    glioma/  meningioma/  notumor/  pituitary/
```

## Data leakage

This is the thing I would want someone to read if they only read one section.

I got suspicious when the code picking example images for the demo skipped past a file named
`Te-aug-me_1.jpg`. The `aug` told me the test folder contained augmented images, which are
derived from other images, so I went looking for how much of the test split was really new.

Filenames were no help, there are zero identical names across the two folders. So I embedded
every image as a 32x32 mean centred vector and computed cosine similarity between the splits.

| Similarity threshold | Test images with a training match |
|---|---|
| above 0.99 | 449 of 1600 |
| above 0.995 | 435 of 1600 |
| above 0.999 | 420 of 1600 |
| above 0.9999 | 279 of 1600 |

The closest pairs sit at similarity 1.000000, which means pixel identical. What convinced me
these were genuine duplicates rather than brains that happen to look alike is that 418 of the
420 pairs above 0.999 carry the same class label. If they were coincidental matches between
different scans, class agreement would sit near 25 percent by chance. It is 99.5 percent.

The overlap is very lopsided:

| Class | Duplicated from training | Clean test images left |
|---|---|---|
| notumor | 310 of 400 | 90 |
| meningioma | 103 of 400 | 297 |
| glioma | 7 of 400 | 393 |
| pituitary | 0 of 400 | 400 |

Over three quarters of the no tumor class is a copy of something the model trained on. So
every score I had at that point was too generous, and most generous exactly where the model
looked strongest.

Everything reported below is measured on the 1180 test images that have no near duplicate in
training. Reproduce the check with:

```
python scripts/find_duplicates.py --config configs/base.yaml
```

One thing that check does not do is catch a rotated or flipped copy of a training image,
because those do not match on pixel similarity. So 26 percent is a floor, not a ceiling.

## Pipeline

Six stages, all classical, all in `src/dip/preprocessing.py`.

| Stage | What it does | Why |
|---|---|---|
| Skull stripping | Otsu threshold, morphological opening, largest connected component, hole fill | Removes skull, scalp and background so head size and field of view cannot be used as shortcuts |
| Bias field correction | Divide by a heavily blurred copy of the image | MRI coils are not uniformly sensitive, so brightness drifts smoothly across the image |
| Denoising | Non-local means | MRI carries Rician noise. Non-local means averages over similar patches instead of a local window, so it does not blur the tumor edge |
| Intensity normalisation | Z-score inside the brain mask, with percentile clipping | The scale problem from the opening section. This is the step that matters most |
| Contrast enhancement | CLAHE | A small tumor contributes almost nothing to a whole image histogram, so global equalisation ignores it. CLAHE works on local tiles |
| Symmetry analysis | Align the midline, mirror, subtract | See the next section |

The order is not arbitrary. Bias correction runs before normalisation because a brightness
drift distorts the mean and standard deviation that z-scoring uses. Denoising runs before
CLAHE because CLAHE amplifies whatever is there, and amplifying noise first then trying to
remove it does not work. Symmetry runs last, on the cleanest image, because a leftover
left-to-right brightness drift shows up as a difference between the two halves that has
nothing to do with any tumor.

## Preprocessing quality

Otsu does not always work. When the resulting mask covers an implausible fraction of the
frame the code falls back to a centred ellipse and records a flag, rather than passing a bad
mask quietly down the line. Over all 7200 images:

| Metric | Training, 5600 images | Testing, 1600 images |
|---|---|---|
| Fell back to an ellipse | 77, 1.38 percent | 22, 1.38 percent |
| Flagged as low contrast | 0 | 0 |
| Mean brain fraction | 0.2990 | 0.3042 |
| Mean symmetry score | 21.3687 | 21.9967 |

Both splits landing on 1.38 percent independently is a reasonable sign the behaviour is
stable rather than tuned to one subset. The pipeline is deterministic, and I confirmed that
by re-running it from scratch after losing a Colab session and getting identical numbers.

## Symmetry

A healthy brain is roughly symmetric about the midsagittal plane, and radiologists compare
left against right as a first pass. A mass breaks that symmetry. So if you mirror the slice
about its own midline and subtract, healthy tissue mostly cancels and abnormal tissue
survives.

The midline is not exactly vertical or exactly centred, because head positioning varies. So
the code searches over small rotations and horizontal shifts for the transform that minimises
the difference between the slice and its mirror, restricted to the brain mask. It searches on
a downsampled copy for speed, then applies the winning transform at full resolution. The
result becomes a third input channel alongside the processed slice and the brain mask.

I originally computed this as `abs(I - mirror(I))`. That turns out to be a mistake, and I
only noticed it by looking at my own demo output, where the tumor was showing up twice, once
in the right place and once mirrored across the midline. It should have been obvious from the
formula: the value at a pixel and at its mirror position are the same number, so the map is
symmetric by construction and cannot say which side anything is on.

I tested it on a synthetic phantom with the tumor at a coordinate I chose, then measured how
far the peak of the map moved when I moved the tumor 76 pixels across the midline.

| Version | Peak moves by | Under bias drift of 0 to 60 percent |
|---|---|---|
| `abs(I - mirror(I))` | 0.0 px | unchanged |
| `relu(I - mirror(I))` | 34.9 to 44.0 px | stable |

Keeping only the positive difference recovers about half the true displacement. So I retrained
with it. Macro F1 went from 0.9435 to 0.9407, which is a change of -0.0028 against a confidence
interval about 0.022 wide. In other words it made no measurable difference.

My best guess is that the laterality was never missing as far as the model was concerned. The
asymmetry map is one of three channels and it sits right next to the processed slice, which
plainly shows which side the lesion is on, so the network could already work it out. I kept
the `relu` version because it is the more sensible quantity and it calibrated slightly better,
not because it scored higher.

## Architecture

```
input slice (1, 256, 256)
        |
   image processing
        |
   3 channels: processed slice, brain mask, asymmetry map
        |                                  43 handcrafted features
   EfficientNet-B0, ImageNet weights            |
        |                                  BatchNorm + MLP 43 -> 256 -> 128
   CBAM attention                               |
        |                                       |
   global average pool -> 1280                  |
        |                                       |
        +-------------------+-------------------+
                            |
                     gated fusion -> 256
                            |
                    linear -> 4 logits
```

The handcrafted branch takes 43 features in four groups: intensity statistics inside the brain
mask, brain mask geometry, descriptors of the asymmetry map including per quadrant distribution
and blob compactness, and GLCM Haralick texture. These are what radiomics used before deep
learning, and I wanted to see whether they still carried anything.

The two streams are combined by a learned gate rather than concatenated. Concatenation works
fine but gives you no way to ask which stream the model used. The gate is a mixing weight you
can read straight off the forward pass, and the app shows it with every prediction. It settled
at 0.452 to 0.480 across three separate training runs. Below 0.5 means the model leaned slightly
on the handcrafted features over the learned ones, which is the main reason I think the hybrid
design earned its place.

CBAM is there because a brain slice is mostly normal tissue and the tumor is a small part of
it, so a plain global average pool dilutes the interesting region against a large uninteresting
one.

Training uses class balanced focal loss. On this balanced copy of the dataset the class weights
reduce to uniform and the weighted sampler does nothing, so it behaves like plain focal loss
with label smoothing. I kept both because the original release of this dataset is not balanced.

Thirty epochs, batch size 32, 256 pixel input, mixed precision, cosine schedule with warmup,
exponential moving average of the weights. Checkpoints save every epoch and `--resume auto`
restores the model, optimiser, scheduler, scaler, EMA and epoch counter, which I needed more than
once when Colab dropped the session.

## Results

Measured on the 1180 test images with no near duplicate in training, with decision offsets fitted
on validation. Confidence intervals are bootstrap percentile intervals over 2000 resamples.

| Metric | Value |
|---|---|
| Macro F1 | 0.9190, 95 percent CI [0.9011, 0.9365] |
| Balanced accuracy | 0.9426, 95 percent CI [0.9312, 0.9540] |
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

Two of those rows are perfect and I want to say what they do not prove. The notumor row is 90 of
90, but only 90 clean test images survived deduplication out of 400, so it is a small sample and
the precision of 0.804 is the more informative number for that class. The pituitary row is 400 of
400 with no images removed at all, which is a real result on a full class, though pituitary
adenomas sit in a characteristic place at the skull base and are the easiest of the four to
recognise from position alone.

Glioma is the weak class at 0.824 recall. Of 393 clean glioma images, 40 were called meningioma
and 21 were called no tumor. Those 21 are the errors that would matter, since missing a tumor
costs more than mixing up two tumor types.

If I evaluate on the full 1600 image test split instead, including the duplicates, macro F1 comes
out at 0.9413. Balanced accuracy barely moves either way, 0.9425 against 0.9426, and that
difference between the two metrics is worth understanding: balanced accuracy is mean recall,
computed within each class, so deleting duplicated examples of a class the model already handles
well changes almost nothing. Macro F1 includes precision, and removing 310 of 400 notumor images
left only 90 true negatives to absorb false positives arriving from the other classes, so notumor
precision fell from 0.943 to 0.804 and pulled the average down.

Because the duplication sat in one class, the clean subset is itself imbalanced at 393, 297, 90
and 400, so some of the precision shift comes from that rather than from leakage alone.

I have not run the stage by stage ablations. Every stage can be switched off through the `dip`
flags in `configs/base.yaml`, so isolating each one is a config change and a retrain, but I have
not done it, and nothing above should be read as evidence that any particular stage helped.

## Decision offsets

Taking the argmax over four probabilities is only one way to decide. Looking at the per class
numbers, glioma had precision 0.991 against recall 0.820, meaning the model was very reluctant to
say glioma and nearly always right when it did. In a medical setting missing a tumor costs more
than confusing two tumor types, so there was a case for trading some of that precision.

`scripts/tune_thresholds.py` fits one additive offset per class on the validation split by
coordinate ascent, and `scripts/evaluate.py --offsets` applies them once to test. They are never
fitted on test data. The fitted values were glioma -0.12, meningioma +0.07, notumor -0.32,
pituitary +0.38, so the biggest adjustment suppresses the no tumor class, which is the direction
I wanted.

On the full test split that took missed tumors from 30 of 1200 down to 24 of 1200, a fifth fewer,
with macro F1 unchanged at 0.9407 against 0.9413. Glioma recall barely moved, 0.820 to 0.823, and
the reason is visible in the confusion matrix. The gliomas being called meningioma are not sitting
near a decision boundary, so no threshold reaches them. The model represents them as meningioma
shaped. That needs different features, not a different cutoff.

Something else showed up here. Validation glioma recall is 0.990 while test glioma recall is
0.824. The validation split is carved out of the same Training folder, so it shares that
distribution, and the supplied Testing folder is measurably different. The glioma weakness does
not appear on validation at all, which is why tuning on validation could not fix it.

## What went wrong

Eight things broke while I was building this. These three were worth writing down.

### The EMA bug

Four epochs in, training accuracy was 94 percent and climbing, and validation balanced accuracy
was sitting at exactly 0.2500. With four classes that is chance, and it means the model was
predicting one class for every image.

The obvious reading is overfitting, but the validation loss was going down: 0.8229, 0.8042,
0.7933, 0.7839. Overfitting drives validation loss up. Two numbers that cannot both be true meant
something else was wrong.

It was the exponential moving average. I was validating `ema.module`, not the model being trained.
With decay 0.999 and only 148 steps per epoch, after four epochs the EMA had absorbed
`1 - 0.999^592`, about 45 percent, of the trained weights. More than half of what I was evaluating
was still random initialisation, so it collapsed to one class. The trained model was fine the whole
time.

I added a warmup so the decay ramps in as `min(decay, (1 + step) / (10 + step))`. First epoch after
the fix: validation balanced accuracy 0.8667, validation loss 0.1764. I also made the training loop
print raw and EMA validation separately with a warning when they diverge, because the thing that
cost me the run was not the bug, it was that the bug was silent.

### A bad duplicate check

When I first went looking for train and test overlap I used an 8x8 average hash, which is a
standard trick: shrink the image to 64 pixels, threshold at the mean, treat the bit pattern as a
fingerprint. It reported that 62 percent of the test set collided with training images.

That number was wrong, and the reason was in the output. One collision paired a `notumor` training
image with a `glioma` test image. Those are different images by definition. Another paired
`meningioma` with `notumor`. A duplicate detector reporting duplicates across classes is not
detecting duplicates.

The check that settled it: 5600 training images produced only 2964 unique hashes. Training images
are not 47 percent copies of each other. Every brain MRI is a bright oval on a black background, so
at 8x8 they all collapse to nearly the same bit pattern. I confirmed it on synthetic phantoms, where
24 clearly different images produced 2 unique hashes.

Switching to 32x32 mean centred cosine similarity fixed it. Identical images score 1.00000, a four
pixel edit scores 0.99985, and two genuinely different scans score 0.97585. That is the measure
behind the 26 percent figure earlier, and the version that lives in `scripts/find_duplicates.py`.

### A broken gitignore

I put `data/` in `.gitignore` to keep the dataset out of the repo. Then I noticed the first commit
was staging 21 files when I expected 23.

An unanchored pattern in `.gitignore` matches a folder of that name at any depth, so `data/` was
also excluding `src/data/`, which holds `dataset.py`. Both `scripts/train.py` and
`scripts/evaluate.py` import from it. The code would have run perfectly on my machine and thrown
`ModuleNotFoundError` on the first command in this file for anyone else.

The fix is one character: `/data/` anchors the pattern to the repository root. I caught it by
reading `git status` before committing rather than after pushing, which is now the only way I do it.

## Demo

The Space is a Streamlit app in a Docker container, deployed on Hugging Face. It has five tabs:
a dashboard, an upload and prediction screen, patient records with a clinician review and override
step, exportable reports, and a page explaining the method.

Uploading a slice runs the full processing pipeline in the browser session and shows all three
stages, then the prediction with its probabilities, the fusion gate value, and a Grad-CAM++
attribution map. Four example scans are included so it can be tried without finding an MRI first.
Records are stored in SQLite.

A scheduled GitHub Action pings the Space every six hours, because Hugging Face suspends a free
Space after 48 hours of no traffic and a cold start makes a visitor wait about a minute. It polls
the Hub API for `runtime.stage` rather than curling the app, since a sleeping Space serves a holding
page that returns HTTP 200 while nothing is actually running.

## Setup

Windows, from the project folder. Command Prompt or PowerShell both work.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Cache the processed images and handcrafted features once. This is the slow step, about 17 minutes
for 7200 images, which is why it is cached rather than done inside the data loader.

```
python scripts/precompute_dip.py --config configs/base.yaml
```

Train. Around 30 minutes on a Colab T4.

```
python scripts/train.py --config configs/base.yaml --resume auto
```

Fit the decision offsets on validation, then evaluate on the deduplicated test split:

```
python scripts/find_duplicates.py --config configs/base.yaml
python scripts/tune_thresholds.py --config configs/base.yaml --checkpoint runs/e6_full/best.pt
python scripts/evaluate.py --config configs/base.yaml --checkpoint runs/e6_full/best.pt --offsets runs/e6_full/offsets.json --exclude data/leaked_test_images.json
```

Drop the `--exclude` flag to see the inflated numbers on the full test split.

## Project structure

```
configs/base.yaml            every setting, one file
scripts/precompute_dip.py    runs the image processing over the dataset and caches it
scripts/train.py             training loop entry point
scripts/evaluate.py          test set metrics, bootstrap intervals, confusion matrix
scripts/tune_thresholds.py   fits per class decision offsets on validation
scripts/find_duplicates.py   near duplicate detection between train and test
src/dip/preprocessing.py     the six processing stages
src/dip/handcrafted.py       the 43 features
src/models/fusion_net.py     backbone, gated fusion, classifier head
src/models/cbam.py           channel and spatial attention
src/data/dataset.py          dataset, splitting, caching
src/engine.py                training loop, mixed precision, EMA, checkpointing
src/metrics.py               metrics, bootstrap intervals, calibration error
src/decision.py              decision offset search and application
src/explain.py               Grad-CAM++ and overlays
tests/                       pytest suite and the synthetic phantom generator
```

## Tests

```
pytest
```

23 tests. Five of them need torch and timm and skip without them, so the image processing tests
still run on a machine with nothing heavy installed.

The suite does not just check that functions return without error. Most of it runs on a synthetic
brain phantom where the tumor position is something I chose, so the tests can check the pipeline
found the right thing rather than merely finding something.

| Test | What it pins down |
|---|---|
| `test_symmetry_localises_tumor` | The asymmetry peak lands within 45 px of a tumor placed at a known coordinate |
| `test_asymmetry_map_tracks_lesion_side` | Moving the tumor across the midline moves the map, which the old `abs` version failed |
| `test_bias_correction_reduces_left_right_ramp` | A synthetic brightness drift measurably shrinks, 0.233 down to 0.088 |
| `test_degenerate_inputs_do_not_crash` | All black, all white and 16x16 images fall back cleanly instead of producing NaN |
| `test_offset_search_recovers_a_suppressed_class` | Threshold tuning raises recall on a class artificially pushed down |
| `test_model_forward_and_backward` | Shapes are right and gradients reach the parameters |

## License

MIT, see LICENSE.

Python, PyTorch, timm, OpenCV, scikit-image, scikit-learn, Streamlit, Docker.
