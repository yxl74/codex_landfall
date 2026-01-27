# AE vs Hybrid TIFF/DNG Detection - Combined Report

## Executive summary
This document compares two detection approaches built on the same TIFF/DNG parsing pipeline:

- **Hybrid (supervised)**: logistic regression + rule flags, trained on benign + malicious data.
- **Autoencoder (AE, benign-only)**: reconstruction-error anomaly detector trained only on benign data.

Key findings (Linux evaluation):
- On the baseline 731-sample dataset, **Hybrid** achieves test ROC-AUC 0.998 with 100% recall on the test split, and **AE (p97)** achieves ROC-AUC 0.965 with 92% recall and lower FPR (2.0% vs 3.1%).
- On the expanded **FiveK benign dataset**, **AE** can trade FPR and recall by threshold: p99 yields ~0.5% FPR (all-data) while keeping LandFall recall 6/6.
- LandFall counts are **very small** (6 total, 1 in the test split), so all LandFall recall numbers should be treated as directional.

---

## 1) ML basics (short primer)

- **Binary classification**: model outputs a score; above a threshold = malicious.
- **Threshold (Thr)**: cutoff to turn a score into a label.
- **TP/TN/FP/FN**:
  - TP: malicious correctly flagged.
  - TN: benign correctly ignored.
  - FP: benign flagged as malicious.
  - FN: malicious missed.
- **Accuracy**: (TP + TN) / total.
- **Precision**: TP / (TP + FP). "When the model says malicious, how often is it right?"
- **Recall**: TP / (TP + FN). "How many real malicious files are caught?"
- **F1**: balance of precision and recall.
- **ROC-AUC**: threshold-free ranking score; 1.0 is perfect.
- **FPR (false positive rate)**: FP / benign total.

---

## 2) Datasets and splits (Linux)

### 2.1 Baseline dataset (731 total, after dedup)
Sources: `data/benign_data`, `data/LandFall`, `data/general_mal`.

| Class | Count |
| --- | --- |
| LandFall | 6 |
| general_mal | 95 |
| benign | 630 |

Split (seed=42): 70% train / 15% val / 15% test

| Split | Total | LandFall | general_mal | benign |
| --- | --- | --- | --- | --- |
| Train | 511 | 4 | 67 | 440 |
| Val | 109 | 1 | 16 | 92 |
| Test | 111 | 1 | 12 | 98 |

### 2.2 FiveK benign expansion (5974 total, after dedup)
Adds `data/benign_data/fivek_dataset` (~5000 DNG photos).

| Class | Count |
| --- | --- |
| LandFall | 6 |
| general_mal | 95 |
| benign | 5873 |

Split (seed=42): 70% train / 15% val / 15% test

| Split | Total | LandFall | general_mal | benign |
| --- | --- | --- | --- | --- |
| Train | 4181 | 1 | 59 | 4121 |
| Val | 896 | 4 | 20 | 872 |
| Test | 897 | 1 | 16 | 880 |

Notes:
- **LandFall is tiny** (6 total, 1 in the test split). Test recall can swing wildly.
- **DNG-heavy training** can increase false positives on classic TIFFs (accepted per project goal).

---

## 3) Feature extraction (shared pipeline)
We do **structure-only parsing**; we never decode pixels.

### 3.1 Feature groups (summary)

| Group | What it captures | Used by |
| --- | --- | --- |
| Byte histograms | Magika-style head/tail byte distributions (257-bin hist x2). | Hybrid + AE |
| TIFF/DNG structure | IFD counts, opcode lists, sizes, DNG tags. | Hybrid + AE |
| Entropy | Randomness of header/tail bytes. | AE only |
| Rule flags | Known exploit patterns (opcode anomalies, ZIP polyglot). | Hybrid only |
| File-type mismatch | DNG/TIFF magic vs extension or Magika output. | Hybrid only |

### 3.2 Structural feature glossary

| Feature(s) | Meaning | Used by |
| --- | --- | --- |
| `is_tiff`, `is_dng` | Valid TIFF header, DNG tag presence. | Hybrid + AE |
| `min_width`, `min_height`, `max_width`, `max_height` | IFD dimensions range. | AE (min shared with Hybrid) |
| `total_pixels`, `file_size` | Size and scale signals. | AE |
| `bytes_per_pixel_milli`, `pixels_per_mb` | Density vs size (compression hints). | AE |
| `ifd_entry_max` | Max IFD entry count. | Hybrid + AE |
| `subifd_count_sum`, `new_subfile_types_unique` | DNG layout complexity. | Hybrid + AE |
| `total_opcodes`, `unknown_opcodes`, `max_opcode_id` | DNG opcode structure; unknowns are suspicious. | Hybrid + AE |
| `opcode_list1_bytes`, `opcode_list2_bytes`, `opcode_list3_bytes` | Per-list opcode sizes. | Hybrid + AE |
| `opcode_list_bytes_total`, `opcode_list_bytes_max`, `opcode_list_present_count` | Opcode footprint summary. | AE |
| `opcode_bytes_ratio_permille`, `opcode_bytes_per_opcode_milli`, `unknown_opcode_ratio_permille` | Opcode density and ratios. | AE |
| `has_opcode_list1/2/3` | Presence of each list. | AE |
| `header_entropy`, `tail_entropy`, `overall_entropy`, `entropy_gradient` | Byte randomness and asymmetry. | AE |
| `zip_eocd_near_end`, `zip_local_in_tail` | ZIP polyglot indicators. | Hybrid |
| `flag_*` (opcode anomaly, tiny dims, ZIP polyglot, DNG/JPEG mismatch) | High-precision rule triggers. | Hybrid |

Why these are effective:
- LandFall abuses **DNG opcode lists**; opcode size/count anomalies are strong indicators.
- Crafted TIFF malware often has unusual **IFD layouts** or tiny dimensions.
- **Byte histograms** capture format/tooling fingerprints even when structure appears valid.
- **Entropy** can reveal injected payloads or compressed data patterns.

---

## 4) Model and training pipeline

### 4.1 Hybrid (supervised)
1. Extract byte histograms + structural features + rule flags.
2. Log1p-transform heavy-tailed fields; standardize.
3. Train **logistic regression** on benign + malicious.
4. Choose threshold on validation to maximize F1 (Thr = 0.20).

### 4.2 Autoencoder (benign-only)
1. Extract byte histograms + structural features + entropy.
2. Log1p-transform heavy-tailed fields; standardize using benign train stats.
3. Train **autoencoder** on benign only.
   - Architecture: `input -> 256 -> 64 -> 256 -> output`.
4. Compute **reconstruction error** as anomaly score.
5. Threshold by **val-benign percentile** (p97 or p99).

---

## 5) Linux evaluation (side-by-side)

### 5.1 Baseline dataset (731) - test split

| Model | Thr | Acc | Prec | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid (supervised) | 0.20 | 0.973 | 0.813 | 1.000 | 0.897 | 0.998 |
| AE (p97, benign-only) | 4.484 | 0.973 | 0.857 | 0.923 | 0.889 | 0.965 |

### 5.2 Test split: LandFall vs general malware (baseline dataset)

| Model | LandFall recall (1) | general_mal recall (12) | benign FPR (98) |
| --- | --- | --- | --- |
| Hybrid | 1.00 | 1.00 | 0.031 |
| AE (p97) | 1.00 | 0.917 | 0.020 |

Notes:
- Test split includes **1 LandFall sample**; treat recall as directional.
- Hybrid has higher general-mal recall; AE yields lower FPR.

### 5.3 AE (p97) all-data rates (baseline dataset, more stable)

| Class | Total | Flagged | Recall / FPR |
| --- | --- | --- | --- |
| LandFall | 6 | 6 | Recall 1.00 |
| general_mal | 95 | 90 | Recall 0.947 |
| benign | 630 | 5 | FPR 0.008 |

---

## 6) AE with FiveK benign expansion

This reflects the DNG-heavy benign dataset, which is more realistic for DNG scanning.

### 6.1 Test split results

| AE Threshold | Thr | Acc | Prec | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| p97 | 1.136 | 0.954 | 0.293 | 1.000 | 0.453 | 0.999 |
| p99 | 3.881 | 0.979 | 0.472 | 1.000 | 0.642 | 0.999 |

### 6.2 Test split: LandFall vs general malware (FiveK)

| AE Threshold | LandFall recall (1) | general_mal recall (16) | benign FPR (880) |
| --- | --- | --- | --- |
| p97 | 1.00 | 1.00 | 0.047 |
| p99 | 1.00 | 1.00 | 0.022 |

### 6.3 All-data sweep (FiveK, more stable)

| Percentile | Benign FPR | LandFall recall (6) | general_mal recall (95) |
| --- | --- | --- | --- |
| 99 | 0.0046 | 1.00 | 0.947 |
| 98 | 0.0083 | 1.00 | 0.968 |
| 97 | 0.0172 | 1.00 | 0.979 |
| 95 | 0.0422 | 1.00 | 1.00 |
| 90 | 0.0819 | 1.00 | 1.00 |

Interpretation:
- **p99** is a good low-FPR option for DNG-heavy benign data.
- Lower thresholds improve general-mal recall but increase false positives.

---

## 7) On-device evaluation (Android)

Bench set (same as `docs/experiment_report.md`):
`/sdcard/Android/data/com.landfall.hybriddetector/files/bench_full/`

| Metric | Hybrid | AE (FiveK p99) |
| --- | --- | --- |
| Files tested | 206 | 206 |
| LandFall samples | 6 | 6 |
| general_mal samples | 100 | 100 |
| benign samples | 100 | 100 |
| Avg extract time | 15.968 ms | 17.217 ms |
| Avg infer time | 0.063 ms | 0.090 ms |

On-device class breakdown for Hybrid at Thr=0.20:

| Class | Total | Flagged | Recall / FPR |
| --- | --- | --- | --- |
| LandFall | 6 | 6 | Recall 1.00 |
| general_mal | 100 | 98 | Recall 0.98 |
| benign | 100 | 2 | FPR 0.02 |

On-device class breakdown for AE (FiveK p99, Thr=4.259):

| Class | Total | Flagged | Recall / FPR |
| --- | --- | --- | --- |
| LandFall | 6 | 6 | Recall 1.00 |
| general_mal | 100 | 97 | Recall 0.97 |
| benign | 100 | 13 | FPR 0.13 |

Log source: `outputs/device_log_ae_p99.txt`

---

## 8) Tests performed (Linux)

- Feature extraction (Hybrid): `analysis/hybrid_extract.py`
- Hybrid training/eval: `analysis/train_hybrid.py`
- Ablation study: `analysis/run_ablation.py`
- Feature extraction (AE): `analysis/anomaly_feature_extract.py`
- AE training/eval: `analysis/ae_train_eval.py`
- AE threshold sweep: `analysis/anomaly_threshold_sweep.py`
- AE TFLite export: `analysis/export_anomaly_ae_tflite.py`
- On-device parity: `analysis/compare_device_local.py`

---

## 9) Limitations and recommendations

- **LandFall sample scarcity** (6 total) limits statistical confidence.
- DNG-heavy benign training can **raise FPR on classic TIFFs**; this is expected.
- For production, keep **two thresholds**:
  - Low-FPR (p99) for large-scale scanning.
  - High-recall (p97) for manual triage.
- Gather more benign TIFF varieties (scientific, multi-page, BigTIFF) to reduce FPR drift.

---

## 10) Repro commands (Linux)

```bash
# Baseline features (AE)
.venv-tf/bin/python analysis/anomaly_feature_extract.py \
  --data-root data \
  --output-npz outputs/anomaly_features.npz \
  --output-csv outputs/anomaly_features.csv

# AE train/eval (p97 default)
.venv-tf/bin/python analysis/ae_train_eval.py \
  --input-npz outputs/anomaly_features.npz \
  --output-json outputs/ae_metrics_anomaly_p97.json

# FiveK features + AE train/eval (p99 example)
.venv-tf/bin/python analysis/anomaly_feature_extract.py \
  --data-root data \
  --output-npz outputs/anomaly_features_fivek.npz \
  --output-csv outputs/anomaly_features_fivek.csv

.venv-tf/bin/python analysis/ae_train_eval.py \
  --input-npz outputs/anomaly_features_fivek.npz \
  --threshold-percentile 99 \
  --output-json outputs/ae_metrics_fivek_p99.json

# Hybrid feature extraction + training
python3 analysis/hybrid_extract.py --data-root data --output-dir outputs
python3 analysis/train_hybrid.py --input-npz outputs/hybrid_features.npz --output-model outputs/hybrid_model.npz --output-metrics outputs/hybrid_metrics.json
```
