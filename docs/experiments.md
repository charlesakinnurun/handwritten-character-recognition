# Experiments

This document is the experimental log for Handwriting Intelligence. Every result
below was produced by the commands shown, on the hardware noted, with the recorded
seeds. Test-set results are only ever reported **after** all model selection was
done on the validation split.

**Hardware/software for all runs below:** Windows, CPU-only (Intel, 2 PyTorch
threads), Python 3.14, PyTorch 2.13.0+cpu, torchvision 0.28.0. GPU paths exist but
were not exercised.

## 1. Method

* **Splits.** The validation set is a deterministic 10% slice of the training set
  (`seed: 42`). The test set is the official one where available; the local EMNIST
  balanced JPEG corpus ships training data only, so a held-out slice is carved from
  the training data and flagged `official_test: false`.
* **Model selection.** The best checkpoint is chosen by the configured monitor
  (`val_loss` by default). Early stopping uses the same metric.
* **Averaging.** Macro-averaged precision/recall/F1 weight each class equally and
  are the headline numbers; weighted averages weight by class support and
  approximate overall sample-level performance. Both are reported.
* **Baseline.** A majority-class ("always predict the most frequent training
  class") model is reported alongside each experiment as the honest lower bound.
* **Reproducibility.** Each run records `git_commit`, Python/library versions,
  dataset identity, split seed and timing in `runs/<name>/environment.json` and
  `summary.json`.

### Commands

```bash
# MNIST baseline
python -m handwriting_intelligence.cli.train --config configs/mnist_cnn.yaml
python -m handwriting_intelligence.cli.evaluate --artifact models/mnist_cnn/artifact.pt \
    --config configs/mnist_cnn.yaml --split test

# MNIST improved
python -m handwriting_intelligence.cli.train --config configs/mnist_cnn_improved.yaml --epochs 5
python -m handwriting_intelligence.cli.evaluate --artifact models/mnist_cnn_improved/artifact.pt \
    --config configs/mnist_cnn_improved.yaml --split test

# EMNIST balanced (CPU quick pass, capped samples)
python -m handwriting_intelligence.cli.train --config configs/emnist_cnn.yaml \
    --max-train-samples 30000 --max-val-samples 3000 --epochs 3
python -m handwriting_intelligence.cli.evaluate --artifact models/emnist_balanced_cnn/artifact.pt \
    --config configs/emnist_cnn.yaml --split test
```

## 2. MNIST — baseline vs improved

Both models share the identical training/validation split (`seed: 42`,
`val_split: 0.1`) and the same 10,000-image official test set.

The **baseline** (`cnn_baseline`, 5 epochs, Adam, no augmentation) reached
**val_loss 0.0420 / val_acc 0.9873** at epoch 4. Its official test results:

| Metric | Value |
| --- | --- |
| Accuracy | **0.9907** |
| F1 (macro) | 0.9907 |
| F1 (weighted) | 0.9907 |
| Precision (macro / weighted) | 0.9907 / 0.9908 |
| Recall (macro / weighted) | 0.9906 / 0.9907 |
| Top-5 accuracy | 1.0000 |
| Majority-class baseline | 0.1135 |
| Model size | 1,648.2 KiB (421,738 params) |
| Inference latency (est., CPU) | 0.933 ms/image |
| Mean confidence (correct / wrong) | 0.9934 / 0.7337 |

The baseline's hardest classes by F1 were `9` (0.985), `8` (0.987) and `6` (0.989);
`9↔8`, `6↔8` are the expected confusions. Note the confidence gap: errors are
still fairly confident (0.73), so raw confidence alone is not a reliable
correctness filter.

The **improved** model (`cnn_improved`: three residual blocks + global average
pooling, AdamW, cosine LR schedule, affine augmentation, label smoothing 0.05;
5 epochs) reached **val_loss 0.3134 / val_acc 0.9928** at epoch 4.

### Baseline vs improved (identical splits, official test set)

| Metric | Baseline `cnn_baseline` | Improved `cnn_improved` |
| --- | --- | --- |
| Trainable parameters | 421,738 | **342,378** |
| Model size | 1,648.2 KiB | **1,342.7 KiB** |
| Epochs run | 5 | 5 |
| Best validation accuracy | 0.9873 | **0.9928** |
| **Test accuracy** | 0.9907 | **0.9953** |
| Test F1 (macro) | 0.9907 | **0.9953** |
| Test F1 (weighted) | 0.9907 | **0.9953** |
| Test precision (macro / weighted) | 0.9907 / 0.9908 | **0.9953 / 0.9953** |
| Test recall (macro / weighted) | 0.9906 / 0.9907 | **0.9952 / 0.9953** |
| Top-5 accuracy | 1.0000 | 1.0000 |
| Majority-class baseline | 0.1135 | 0.1135 |
| Inference latency (est., ms/image) | **0.933** | 1.792 |
| Mean confidence (correct / wrong) | 0.9934 / 0.7337 | 0.9579 / **0.5627** |

**Takeaways**

* The improved model is **+0.46 percentage points** more accurate on the official
  test set (0.9953 vs 0.9907) while using **~19% fewer parameters** — the change
  was justified by evidence, not complexity for its own sake.
* It is better **calibrated**: its mean confidence on *incorrect* predictions is
  0.5627 versus 0.7337 for the baseline, i.e. it is noticeably less confident when
  it is wrong. That makes a confidence threshold far more useful.
* **Validation loss is not comparable across these two runs.** Label smoothing
  (0.05) deliberately raises the reported cross-entropy, so the improved run's
  higher `val_loss` (0.3134) does not mean it trained worse. Accuracy is the
  apples-to-apples comparison here — a reminder to compare like with like.
* The trade-off is latency: +0.86 ms/image (deeper network, higher-resolution
  feature maps). On this CPU-only setup both are comfortably sub-2 ms.

## 3. EMNIST Balanced

The 47-class `emnist_balanced` split is dramatically harder than MNIST: several
classes are genuinely ambiguous (`O/0`, `I/l/1`, `L/1`), and the local JPEG corpus
ships only training data.

> **This is a quick CPU verification pass, not a tuned result.** It trained
> `cnn_improved` for 3 epochs on a capped 30,000-image subset (3,000 validation),
> with a held-out split carved from the training data (`official_test: false`).
> Full-data, longer training would do better; the point here is to prove the
> pipeline works end-to-end on a real 47-class problem.

Best validation accuracy was **0.7987** (epoch 2, still improving). Held-out test:

| Metric | Value |
| --- | --- |
| Accuracy | **0.8126** |
| F1 (macro) | 0.8084 |
| F1 (weighted) | 0.8092 |
| Precision (macro / weighted) | 0.8195 / 0.8202 |
| Recall (macro / weighted) | 0.8115 / 0.8126 |
| Top-5 accuracy | 0.9821 |
| Majority-class baseline | 0.0166 |
| Model size | 1,379.9 KiB (351,887 params) |
| Inference latency (est., CPU) | 3.156 ms/image |
| Mean confidence (correct / wrong) | 0.7157 / 0.4404 |

**Worst classes by F1** — confirming the ambiguity is in the label space, not the
pipeline:

| Class | F1 | Precision | Recall | Support |
| --- | --- | --- | --- | --- |
| `L` | 0.430 | 0.779 | 0.297 | 249 |
| `f` | 0.494 | 0.602 | 0.418 | 239 |
| `I` | 0.508 | 0.502 | 0.514 | 222 |
| `q` | 0.531 | 0.594 | 0.479 | 263 |
| `O` | 0.533 | 0.555 | 0.513 | 187 |
| `g` | 0.534 | 0.643 | 0.456 | 241 |
| `1` | 0.560 | 0.460 | 0.713 | 237 |
| `0` | 0.652 | 0.606 | 0.706 | 238 |

`L`/`1`, `I`/`l`/`1` and `O`/`0` dominate, exactly as expected for this label
space. Top-5 accuracy (0.9821) being far above top-1 is the signature of a model
that "knows" the plausible alternatives but cannot always pick the right one — the
expected outcome for a capped run.

## 4. Error analysis

### MNIST (official test set, 10,000 samples)

| Model | Errors | Accuracy |
| --- | --- | --- |
| `cnn_baseline` | 93 / 10000 | 0.9907 |
| `cnn_improved` | 47 / 10000 | 0.9953 |

**Most frequent confusions**

*Baseline (`cnn_baseline`):*

| True → Predicted | Count |
| --- | --- |
| 9 → 8 | 7 |
| 9 → 4 | 6 |
| 6 → 0 | 5 |
| 3 → 8 | 5 |
| 3 → 5 | 4 |
| 6 → 1 | 4 |

*Improved (`cnn_improved`):*

| True → Predicted | Count |
| --- | --- |
| 5 → 3 | 5 |
| 9 → 4 | 5 |
| 7 → 2 | 4 |
| 2 → 7 | 3 |
| 6 → 0 | 3 |
| 8 → 2 | 3 |

The confusions are the expected shape errors: closed-vs-open loops (`9/8`, `9/4`,
`6/0`) and stroke-count ambiguity (`5/3`, `7/2`). There is no single "broken" class
— the improved model's errors are spread thinly across the class space, which is
the signature of working near the ceiling of this dataset.

**Confidence / calibration**

| Model | Mean conf (correct) | Mean conf (wrong) | frac ≥ 0.99 | frac ≥ 0.9 |
| --- | --- | --- | --- | --- |
| `cnn_baseline` | 0.9934 | 0.7337 | 0.9343 | 0.9752 |
| `cnn_improved` | 0.9579 | 0.5627 | 0.0069 | 0.9453 |

Two observations matter operationally:

1. **Both models are less confident when wrong**, but the improved model's error
   confidence (0.56) is much lower than the baseline's (0.73). A confidence
   threshold such as "route to a human if confidence < 0.8" would catch more of the
   improved model's errors.
2. **The improved model is not more confident — it is better calibrated.** Its
   `frac ≥ 0.99` drops from 0.93 to ~0.01, yet its accuracy *rises*. This is the
   regularizing effect of label smoothing and augmentation: the softmax is flatter
   but more honest. Do not read "higher confidence" as "better model".

Confusion-matrix images and correct/misclassified example grids are written to
`reports/<experiment>/` by the evaluation command.

## 5. Reproducibility

To reproduce any row: check out the recorded `git_commit` from the run's
`environment.json`, install `pip install -e ".[dev]"`, and run the command above.
The validation split is fixed by `seed`, so the train/val partition is identical
across machines.

## 6. CRNN / sequence extension (not yet trained)

The CRNN (`models/crnn.py`) is implemented with a CNN feature extractor, a
bidirectional LSTM and CTC loss with greedy decoding. It is **not trained here**
and no CER/WER is reported, because it requires a word/line-level dataset (IAM)
under a research license. Training it and reporting CER/WER on a held-out writer
split is the next milestone. See `docs/architecture.md`.
