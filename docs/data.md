# Data

This document is the source of truth for every dataset the project can consume:
how it is laid out on disk, how it is validated, how labels are mapped, and what
its licensing constraints are.

> **Never commit datasets.** `data/` is git-ignored. Obtain data using the steps
> below and keep it local.

## 1. Supported datasets

| Config `data.name` | Classes | Source | Notes |
| --- | --- | --- | --- |
| `mnist` | 10 | IDX files (local) or the official mirror | Baseline dataset; official test set preserved. |
| `emnist_balanced` | 47 | local JPEG corpus | Digits, uppercase `A-Z`, 11 non-merged lowercase letters. |
| `emnist_bymerge` | 47 | local JPEG corpus or torchvision | Same label space as `balanced`. |
| `emnist_letters` | 27 (index 0 unused) | torchvision | Raw labels `1..26` = `a..z`; index 0 is a placeholder. |
| `emnist_digits` | 10 | torchvision | EMNIST digits split. |
| `emnist_byclass` | 62 | torchvision | Full alphanumeric set. |

The project **never silently substitutes** one dataset for another. If the
requested dataset is unavailable and `download: false`, it raises
`DatasetNotFoundError` naming the exact path it searched.

## 2. Label conventions (the important part)

MNIST and the EMNIST splits do **not** share label meanings:

* **MNIST / EMNIST digits** — labels `0..9` are the characters `'0'..'9'`.
* **EMNIST letters** — labels are `1..26` for `'a'..'z'`; **index 0 is unused**.
  Treating it as a normal 26-class problem shifts every class by one. This project
  models it explicitly as 27 classes with `labels[0] == "N/A"`.
* **EMNIST balanced / bymerge** — 47 classes. Uppercase `A-Z` and digits are kept;
  15 lowercase letters that are visually confusable with uppercase
  (`c i j k l m o p s u v w x y z`) are **merged into their uppercase form**. The
  index order is `sorted(digits + letters − merged)`:
  `0-9`, `A-Z`, then `a b d e f g h n q r t`.
* **EMNIST byclass** — all 62 alphanumeric classes.

The mapping is implemented once, in `handwriting_intelligence/classes.py`, and is
unit-tested against the reference merge rule. It is serialized into every
checkpoint and artifact, so a prediction can never be silently relabelled.

## 3. Acquisition

### 3.1 MNIST

Place the four IDX files anywhere under `data/` (a common layout is
`data/MINST/` or `data/MNIST/raw/`):

```
train-images.idx3-ubyte   train-labels.idx1-ubyte
t10k-images.idx3-ubyte    t10k-labels.idx1-ubyte
```

The loader discovers files by name, validates the IDX magic number and the exact
byte count, and ignores truncated placeholders. If nothing is found and
`data.download: true`, it fetches the files from the community mirror
(`https://ossci-datasets.s3.amazonaws.com/mnist/`), **verifies their MD5 sums**,
decompresses them into `data/mnist/raw/`, and validates them again.

### 3.2 EMNIST balanced (local JPEG corpus)

The project reads an extracted EMNIST JPEG corpus laid out as one folder per
class plus a CSV manifest:

```
data/EMNIST/EMNIST/
├── emnist.csv             # img_name,label   e.g.  "45/char_0.jpeg,45"
├── 0/  char_*.jpeg        # 2400 images per class ...
├── 1/
├── ...
└── 46/
```

* The manifest is authoritative — only files it lists are read, so stray files
  (e.g. macOS `._` resource forks) are ignored.
* Every image must be 28x28; the loader fails loudly otherwise.
* **Orientation**: EMNIST images are distributed transposed (rotated 90°). The
  loader transposes them to the upright orientation. This is verified visually in
  the test suite (`class 1` becomes a `1`).
* The decoded array is cached to `data/cache/<name>_corpus.npz` so loading is
  fast on subsequent runs (`data.cache: true`).

### 3.3 Other EMNIST splits

Splits that are not shipped as a local corpus (`letters`, `digits`, `byclass`)
are obtained through `torchvision.datasets.EMNIST`, which downloads the ~500 MB
official archive. Enable with `data.download: true`. Images are transposed to the
upright orientation after loading.

## 4. Splits and integrity

* **Validation** is carved from the **training** split using a seeded permutation
  (`data.val_split`, `seed`). The same split is used by every experiment so
  baseline/improved comparisons are apples-to-apples.
* **Test** is the official test set when one is available (MNIST, torchvision
  EMNIST splits). The local EMNIST *balanced JPEG corpus ships training data only*;
  for it, a held-out split is carved from the training data and the run metadata
  records `official_test: false` so results are never overstated.
* The test set is used **only** for final evaluation. Hyperparameters are never
  tuned against it.
* Optional caps (`max_train_samples`, `max_val_samples`, `max_test_samples`) exist
  for quick CPU verification passes and are recorded in the run metadata.

## 5. Preprocessing

Training/evaluation data is already a centred 28x28 glyph, so the dataset
transform only converts to a tensor and normalizes (`mean=0.1307`,
`std=0.3081` for MNIST). Arbitrary uploads go through a fuller pipeline:
grayscale → polarity normalization (auto-invert) → optional denoise/threshold →
crop-to-content → aspect-preserving resize to a 20x20 box → centre on a 28x28
canvas → normalize. The same normalization constants are shared, which is what
keeps training and serving distributions aligned.

## 6. Licensing and citation

* **MNIST** — Yann LeCun, Corinna Cortes, Christopher Burges. Distributed for
  research use; see the dataset page for terms.
* **EMNIST** — Cohen, Afshar, Tapson, van Schaik, *EMNIST: an extension of MNIST
  to handwritten letters* (2017).
* **IAM Handwriting Database** (CRNN extension only) — distributed by the
  University of Bern under a research license and requires registration. It is
  **not** bundled or downloaded by this project. See `docs/architecture.md`.

If you use these datasets, cite the original authors. This project adds no
additional restrictions on top of theirs.
