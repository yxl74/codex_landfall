# Benign-Only Anomaly Experiments (AE + Deep SVDD)

This document captures the benign-only anomaly experiments run so far, explains the full training
pipeline, and describes each feature used. It is written for readers without an ML background.

## Goal and constraints

- Goal: learn what *normal TIFF/DNG structure* looks like from benign files only, then flag files
  that deviate as suspicious (LandFall + other TIFF malware).
- Constraint: only 6 LandFall samples exist, so we must avoid training on LandFall and instead
  rely on benign-only anomaly detection.

## Data and splits

Data sources (after hash de-duplication; priority label: landfall > general_mal > benign):

| Split | Total | Benign | LandFall | General TIFF malware |
| --- | --- | --- | --- | --- |
| All | 731 | 630 | 6 | 95 |
| Train (70%) | 511 | 440 | 4 | 67 |
| Val (15%) | 109 | 92 | 1 | 16 |
| Test (15%) | 111 | 98 | 1 | 12 |

Notes:
- The **LandFall sample count is tiny**. Test-only recall can swing between 0 and 1 depending on
  whether the single LandFall file in the test split is detected.
- We **train only on benign** data; malicious files are used only for evaluation.

## Training pipeline (benign-only anomaly detection)

1. **Feature extraction** from each file:
   - Byte-level prefix/suffix features (Magika-style).
   - Structural TIFF/DNG features (IFD layout, opcodes, sizes).
   - Entropy features (randomness of bytes in header/tail).
2. **De-duplication** by SHA-256 hash (keep malicious label if a duplicate appears in multiple sets).
3. **Train/val/test split** with a fixed seed (42).
4. **Normalization**:
   - Count-like features are transformed with `log1p` to shrink large values.
   - All features are standardized using the benign training mean and std.
5. **Model training** using **benign-only data**:
   - Autoencoder (AE) minimizes reconstruction error.
   - Deep SVDD learns a center and minimizes distance to it.
6. **Anomaly score**:
   - AE: mean squared reconstruction error.
   - Deep SVDD: squared distance to the learned center.
7. **Thresholding**:
   - Threshold = percentile of **val-benign** scores (e.g., p97).
   - A file is **flagged** if `score >= threshold`.

## What the model “sees” (features)

### Byte-level features (Magika-style)

| Feature | Description | Why it helps |
| --- | --- | --- |
| `magika_prefix` | First 1024 bytes (after trimming whitespace), encoded as a 257-bin histogram. | Captures magic bytes and early structure. |
| `magika_suffix` | Last 1024 bytes (after trimming whitespace), encoded as a 257-bin histogram. | Captures footer patterns and padding behavior. |

Notes:
- We use **histograms** (not raw bytes) for stability: two 257-bin histograms are concatenated.
- Bin 256 represents padding tokens when a file is short.

### Structural TIFF/DNG features

These are extracted by parsing TIFF IFDs and DNG opcode lists.

| Feature | Description | Why it helps |
| --- | --- | --- |
| `is_tiff` | 1 if TIFF header is valid. | Sanity check; non-TIFF bytes stand out. |
| `is_dng` | 1 if DNG version tag exists. | Differentiates DNG-like structures. |
| `min_width` | Minimum width across IFDs. | Outliers often have strange dimensions. |
| `min_height` | Minimum height across IFDs. | Same as above. |
| `max_width` | Maximum width across IFDs. | Large sizes can be suspicious. |
| `max_height` | Maximum height across IFDs. | Large sizes can be suspicious. |
| `total_pixels` | `max_width * max_height`. | Normalizes size vs file length. |
| `file_size` | Size in bytes. | Captures unusually large/small files. |
| `bytes_per_pixel_milli` | `(file_size / total_pixels) * 1000`. | Compressed vs uncompressed ratio clues. |
| `pixels_per_mb` | `(total_pixels / file_size) * 1,000,000`. | Inverse of bytes-per-pixel. |
| `ifd_entry_max` | Max number of IFD entries. | Abnormally large IFDs can be malicious. |
| `subifd_count_sum` | Count of sub-IFD offsets. | Unusual multi-IFD layout can be a signal. |
| `new_subfile_types_unique` | Unique NewSubfileType values. | Multiple variants can be abnormal. |
| `total_opcodes` | Total opcodes across lists. | DNG opcode usage may be abnormal. |
| `unknown_opcodes` | Count of opcodes with id > 14. | LandFall uses unusual opcodes. |
| `max_opcode_id` | Maximum opcode id observed. | Large opcode ids are suspicious. |
| `opcode_list1_bytes` | Byte size of OpcodeList1. | Abnormal opcode sizes stand out. |
| `opcode_list2_bytes` | Byte size of OpcodeList2. | Same. |
| `opcode_list3_bytes` | Byte size of OpcodeList3. | Same. |
| `opcode_list_bytes_total` | Sum of opcode list bytes. | Total opcode footprint. |
| `opcode_list_bytes_max` | Largest opcode list size. | Captures extreme list sizes. |
| `opcode_list_present_count` | Number of opcode lists present. | Non-standard list usage. |
| `opcode_bytes_ratio_permille` | `(opcode_bytes / file_size) * 1000`. | Opcode footprint relative to file size. |
| `opcode_bytes_per_opcode_milli` | `(opcode_bytes / total_opcodes) * 1000`. | Average opcode payload size. |
| `unknown_opcode_ratio_permille` | `(unknown_opcodes / total_opcodes) * 1000`. | Ratio of unknown opcodes. |
| `has_opcode_list1` | 1 if list1 present. | Presence/absence signal. |
| `has_opcode_list2` | 1 if list2 present. | Presence/absence signal. |
| `has_opcode_list3` | 1 if list3 present. | Presence/absence signal. |

### Entropy features

| Feature | Description | Why it helps |
| --- | --- | --- |
| `header_entropy` | Shannon entropy of first 4 KB. | High entropy suggests packed/random payload. |
| `tail_entropy` | Shannon entropy of last 4 KB. | Detects unusual tail content. |
| `overall_entropy` | Shannon entropy of first 64 KB. | Coarse “randomness” of file. |
| `entropy_gradient` | `abs(header_entropy - tail_entropy)`. | Large differences can indicate injected data. |

## Preprocessing details

- The following features are `log1p` transformed before training:
  `min_width`, `min_height`, `ifd_entry_max`, `subifd_count_sum`, `new_subfile_types_unique`,
  `total_opcodes`, `unknown_opcodes`, `max_opcode_id`, `opcode_list1_bytes`,
  `opcode_list2_bytes`, `opcode_list3_bytes`, `max_width`, `max_height`, `total_pixels`,
  `file_size`, `bytes_per_pixel_milli`, `pixels_per_mb`, `opcode_list_bytes_total`,
  `opcode_list_bytes_max`, `opcode_list_present_count`, `opcode_bytes_ratio_permille`,
  `opcode_bytes_per_opcode_milli`, `unknown_opcode_ratio_permille`.
- All features are standardized using **benign training** mean/std.

## Models (intuitive view)

### Autoencoder (AE)

An autoencoder is trained to **reconstruct** benign feature vectors.  
If a file is unusual, reconstruction error is larger.

Architecture (Dense): `input -> 256 -> 64 -> 256 -> output`

### Deep SVDD

Deep SVDD learns a center `c` in latent space and **pulls benign samples toward it**.  
Files far from the center are flagged.

Architecture (Dense): `input -> 256 -> 64 -> 32`

## How we decide “malicious”

We set a **threshold** using the benign validation set:
- At percentile **p97**, 97% of val-benign scores fall below the threshold.
- Any file with score **>= threshold** is flagged as suspicious.

This makes the **false positive rate (FPR)** roughly the chosen percentile on benign validation
data, but the **test FPR** can be higher due to small sample sizes and distribution shift.

## Metrics (plain language)

- **Recall** (malicious): fraction of malicious files that are correctly flagged.
- **False Positive Rate (FPR)**: fraction of benign files incorrectly flagged.
- **Threshold**: a cutoff on the anomaly score; higher threshold = fewer false positives.

## Experiment 1: Percentile sweep (all data)

This shows rates over *all deduplicated samples*, using a val-benign threshold.

### Autoencoder (AE)

| Percentile | Benign FPR | LandFall recall | General malware recall |
| --- | --- | --- | --- |
| 99 | 0.001587 | 0.000000 | 0.852632 |
| 98 | 0.004762 | 0.333333 | 0.884211 |
| 97 | 0.007937 | 1.000000 | 0.947368 |
| 95 | 0.011111 | 1.000000 | 0.947368 |
| 90 | 0.031746 | 1.000000 | 0.947368 |

### Deep SVDD

| Percentile | Benign FPR | LandFall recall | General malware recall |
| --- | --- | --- | --- |
| 99 | 0.001587 | 0.000000 | 0.884211 |
| 98 | 0.006349 | 0.333333 | 0.884211 |
| 97 | 0.007937 | 0.333333 | 0.915789 |
| 95 | 0.012698 | 1.000000 | 0.915789 |
| 90 | 0.034921 | 1.000000 | 0.947368 |

## Experiment 2: Percentile sweep (test-only)

This shows rates on **test data only** (no train/val).  
The test split contains **1 LandFall sample**, so LandFall recall is very unstable.

### Autoencoder (AE)

| Percentile | Benign FPR | LandFall recall | General malware recall |
| --- | --- | --- | --- |
| 99 | 0.000000 | 0.000000 | 0.916667 |
| 98 | 0.010204 | 0.000000 | 0.916667 |
| 97 | 0.020408 | 1.000000 | 0.916667 |
| 95 | 0.020408 | 1.000000 | 0.916667 |
| 90 | 0.091837 | 1.000000 | 0.916667 |

### Deep SVDD

| Percentile | Benign FPR | LandFall recall | General malware recall |
| --- | --- | --- | --- |
| 99 | 0.010204 | 0.000000 | 0.916667 |
| 98 | 0.020408 | 0.000000 | 0.916667 |
| 97 | 0.020408 | 0.000000 | 0.916667 |
| 95 | 0.040816 | 1.000000 | 0.916667 |
| 90 | 0.102041 | 1.000000 | 0.916667 |

## Interpretation

- **AE at p97** is the most attractive tradeoff so far:
  - All-data: FPR ~0.8%, LandFall recall 6/6, general malware recall ~95%.
  - Test-only: FPR ~2.0%, LandFall recall 1/1, general malware recall ~92%.
- **Deep SVDD** needs a lower threshold (p95) to catch LandFall, but then FPR rises.

Given the tiny LandFall set, we should treat these numbers as **promising but uncertain**.

## Repro steps

```bash
# 1) Feature extraction
.venv-tf/bin/python analysis/anomaly_feature_extract.py \
  --data-root data \
  --output-npz outputs/anomaly_features.npz \
  --output-csv outputs/anomaly_features.csv

# 2) Train + evaluate AE
.venv-tf/bin/python analysis/ae_train_eval.py \
  --input-npz outputs/anomaly_features.npz

# 3) Train + evaluate Deep SVDD
.venv-tf/bin/python analysis/deepsvdd_train_eval.py \
  --input-npz outputs/anomaly_features.npz

# 4) Threshold sweep (all-data + thresholds from val-benign)
.venv-tf/bin/python analysis/anomaly_threshold_sweep.py \
  --model ae \
  --input-npz outputs/anomaly_features.npz \
  --percentiles 99,98,97,95,90

.venv-tf/bin/python analysis/anomaly_threshold_sweep.py \
  --model deepsvdd \
  --input-npz outputs/anomaly_features.npz \
  --percentiles 99,98,97,95,90
```

## Current recommendation

If we accept ~1% FPR as a target, **AE with a p97 threshold** is the best current setting.
We should still expect FPR to vary across devices and benign sources.
