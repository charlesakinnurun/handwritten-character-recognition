# Model Card — Handwriting Intelligence (character classifier)

This card follows the model-card format (Mitchell et al., 2019). It describes the
character-classification models produced by this repository. Fill-in-the-blank
numbers are populated from **actual** runs recorded in
[`docs/experiments.md`](experiments.md); nothing here is estimated or invented.

## 1. Model details

| | |
| --- | --- |
| Developed by | Charles Akinnurun (portfolio project) |
| Model type | Supervised convolutional neural network, image classification |
| Architectures | `CNNBaseline` (compact conv + MLP), `CNNImproved` (residual CNN + GAP) |
| Framework | PyTorch (CPU or CUDA) |
| Inputs | Single-channel 28x28 grayscale image, normalized |
| Outputs | Logits over the dataset's class set; softmax for confidence |
| License | MIT (code). Datasets retain their own terms — see `docs/data.md`. |

## 2. Intended use

**Intended**
* Classifying a single, isolated handwritten digit or character from a cropped
  image.
* Education, demos, and as a reference implementation of a production-shaped
  ML pipeline.

**Not intended**
* Reading words, sentences, or pages (see the CRNN extension — separate, untrained).
* Documents, forms, signatures, or handwriting with layout.
* Any decision with legal, medical, financial, or safety consequences.
* Re-identifying individuals from handwriting.

## 3. Factors

Relevant sources of variation: image polarity (dark-on-light vs light-on-dark),
stroke thickness, image size and framing, scanning/compression artefacts, and
domain shift away from the datasets (pens, paper, fonts, camera photos).

## 4. Metrics

Reported per split on the held-out **test** set unless noted:

* Accuracy
* Precision / Recall / F1 (macro and weighted — see `docs/experiments.md` for why)
* Top-5 accuracy (when ≥ 5 classes)
* Per-class precision/recall/F1 and support
* Confusion matrix
* Majority-class baseline accuracy
* Inference latency (ms/image, CPU) and model size (KiB)

## 5. Training & evaluation data

See `docs/data.md`. In short: MNIST (60k train / 10k test) and EMNIST Balanced
(47 classes). Validation is carved from training with a fixed seed; the test set
is untouched until final evaluation.

## 6. Quantitative analysis

Actual results are recorded in [`docs/experiments.md`](experiments.md) and in
`reports/<experiment>/`. Confidence behaviour is analysed there: a well-behaved
model is measurably *less* confident when it is wrong.

## 7. Ethical considerations

* The datasets are historical collections of handwriting and may not represent
  all scripts, ages, or writing styles. Performance on out-of-distribution input
  is worse and should be measured before any real use.
* No personally identifying information is used or produced.
* The API never logs image contents; only metadata and timing.

## 8. Caveats and recommendations

* **Isolated characters only.** Feeding a multi-character image yields a single
  prediction with no segmentation and is *not* word recognition.
* **Confidence ≠ correctness.** Softmax confidence is a model score, not a
  guarantee; inspect top-k and calibration before acting on it.
* **Distribution shift.** Camera photos, coloured backgrounds, and unusual stroke
  weights are harder than the training distribution. Test on your own data.
* **Not calibrated by default.** No temperature scaling / Platt scaling is applied.
* Retrain and re-evaluate before domain changes; do not assume the published
  numbers transfer.
