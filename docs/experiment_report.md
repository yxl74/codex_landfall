# Hybrid TIFF/DNG Detection - Comprehensive Experiment Report

## Executive summary
This document explains the full detection pipeline end-to-end, from file parsing and feature extraction to ML training, evaluation, and on-device testing. The system uses a hybrid approach: structure-aware TIFF/DNG parsing plus Magika-style byte statistics, combined with lightweight rule flags and a simple, interpretable logistic regression model.

Key outcomes:
- Linux evaluation: ROC-AUC 0.998 on test with threshold 0.20 (TP=13, FP=3, FN=0).
- On-device evaluation: 206-file bench set, average 11.211 ms feature extraction and 0.062 ms inference per file.
- Device vs local parity: max absolute score difference 4.992e-05 across the entire bench set.

---

## 1) ML basics (quick primer)
This section explains evaluation terms in plain language.

- Binary classification: the model outputs a score between 0 and 1. Higher means "more likely malicious."
- Threshold (Thr): the cutoff to convert a score into a label. Example: Thr=0.20 means scores >= 0.20 are labeled "malicious."
- Confusion matrix terms:
  - TP (true positives): malicious files correctly flagged.
  - TN (true negatives): benign files correctly ignored.
  - FP (false positives): benign files incorrectly flagged.
  - FN (false negatives): malicious files missed.
- Accuracy: (TP + TN) / total. Can look good even if malware is rare.
- Precision: TP / (TP + FP). "When the model says malicious, how often is it right?"
- Recall: TP / (TP + FN). "How many of the actual malicious files did we catch?"
- F1: harmonic mean of precision and recall (useful for imbalanced data).
- ROC-AUC: threshold-independent ranking score; 1.0 is perfect, 0.5 is random.

Why we use a threshold:
- Lower threshold: higher recall, more false positives.
- Higher threshold: higher precision, more false negatives.
- In this experiment, Thr=0.20 is chosen on validation to maximize F1.

---

## 2) TIFF/DNG primer (short and practical)
- TIFF files start with a 2-byte endian marker (II or MM) and a magic number 42.
- Metadata lives in Image File Directories (IFDs). Each IFD is a list of tag entries.
- DNG is a TIFF-based RAW format with extra tags:
  - SubIFDs (tag 330) to store preview and main images.
  - Opcode lists (tags 51008, 51009, 51022) for image processing operations.
- Landfall targets DNG opcode parsing, so opcode counts and IDs are critical signals.

---

## 3) Data and splits
- Total samples: 731
- Breakdown: 630 benign, 100 general malware, 6 Landfall samples
- Deduplication: file-hash based before splitting
- Split: 70% train / 15% val / 15% test (random, seed=42)
- Train counts: 511 (71 malicious, 440 benign)
- Validation counts: 109 (17 malicious, 92 benign)
- Test counts: 111 (13 malicious, 98 benign)

---

## 4) Pipeline overview (training and inference)
1. Bytes features: read file head/tail and build Magika-style histograms.
2. Structure features: parse TIFF/DNG metadata and extract structural statistics.
3. Rule flags: high-precision heuristics for known exploit signals.
4. Model: logistic regression with standardization and log1p on heavy-tailed fields.
5. TFLite deployment: identical preprocessing embedded in the graph for on-device parity.

---

## 5) Feature extraction in detail

### 5.1 Bytes features (Magika-style)
We read the first and last 4096 bytes, trim leading/trailing ASCII whitespace, then build 1024-byte histograms.

| Feature group | Representation | Dimension | What it captures |
| --- | --- | --- | --- |
| Byte histograms | 257-bin histogram for head + tail (includes padding token 256) | 514 | Format fingerprints and encoder behavior in headers/trailers |

Why it helps:
- File formats and families have characteristic byte distributions at the start and end.
- Malicious samples can carry unusual or non-standard patterns even if they are valid TIFF/DNG.

### 5.2 Structural TIFF/DNG features
These are computed from TIFF headers and IFDs (no full image decoding).

| Feature name | Meaning | Why it helps |
| --- | --- | --- |
| is_tiff | TIFF header detected | Fast validation of file type |
| is_dng | DNG tag detected | Landfall targets DNG opcodes |
| min_width | Minimum width across IFDs | Tiny dimensions can be suspicious |
| min_height | Minimum height across IFDs | Same as above |
| ifd_entry_max | Maximum number of entries in any IFD | Malformed files often have very small IFDs |
| subifd_count_sum | Total SubIFD offsets found | DNG layout signal |
| new_subfile_types_unique | Unique NewSubfileType values | DNG layout signal |
| total_opcodes | Total opcodes in DNG opcode lists | Landfall abuses large opcode counts |
| unknown_opcodes | Opcode IDs > 14 | Unknown IDs are strong anomaly signals |
| max_opcode_id | Largest opcode ID found | Captures out-of-range opcode IDs |
| opcode_list1_bytes | Bytes in OpcodeList1 | Structure and payload size hints |
| opcode_list2_bytes | Bytes in OpcodeList2 | Same as above |
| opcode_list3_bytes | Bytes in OpcodeList3 | Same as above |
| zip_eocd_near_end | ZIP EOCD near file tail | Indicates possible polyglot |
| zip_local_in_tail | ZIP local header in tail | Confirms potential ZIP payload |
| flag_opcode_anomaly | DNG + (opcodes > 100 or unknown opcodes) | High-precision DNG anomaly rule |
| flag_tiny_dims_low_ifd | non-DNG tiny dims + small IFD | Flags malformed TIFFs |
| flag_zip_polyglot | EOCD + ZIP local header in tail | Strong polyglot indicator |
| flag_dng_jpeg_mismatch | Extension says JPEG, header is TIFF + DNG | Disguised DNG signal |
| flag_magika_ext_mismatch | Magika says TIFF/DNG but extension says JPEG | Optional mismatch rule |
| flag_any | OR of all rule flags | High-confidence "red flag" |

### 5.3 Why these features work
- DNG opcode lists are the core of Landfall's exploit chain; unusual opcodes stand out.
- Small dimensions + tiny IFDs are common in crafted TIFF malware.
- ZIP polyglot flags catch smuggled payloads appended to otherwise valid TIFFs.
- Byte histograms capture global format and toolchain differences even when structural metadata looks plausible.

---

## 6) Model and training
We use logistic regression for transparency and speed.

Training steps:
1. Hash-based deduplication to prevent leakage.
2. 70/15/15 train/val/test split (seed=42).
3. Log1p transform on heavy-tailed structural fields.
4. Standardization with train-set mean/std.
5. Threshold selected on validation to maximize F1.

Model outputs:
- Continuous score (0..1)
- Thresholded label at Thr=0.20

---

## 7) Linux evaluation
Model: outputs/hybrid_model.npz (bytes_mode=hist), Thr=0.20
Metrics from outputs/hybrid_metrics.json:

| Split | Acc | Prec | Recall | F1 | ROC-AUC | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Val | 0.982 | 1.000 | 0.882 | 0.938 | - | 15 | 92 | 0 | 2 |
| Test | 0.973 | 0.812 | 1.000 | 0.897 | 0.998 | 13 | 95 | 3 | 0 |

### 7.1 Ablation study (outputs/ablation_results.md)
| Model | Acc | Prec | Recall | F1 | ROC-AUC | Thr |
| --- | --- | --- | --- | --- | --- | --- |
| rules_flag_any | 0.991 | 1.000 | 0.923 | 0.960 | 0.000 | 0.50 |
| struct_only | 0.883 | 0.500 | 1.000 | 0.667 | 0.990 | 0.10 |
| bytes_only_hist | 0.973 | 0.812 | 1.000 | 0.897 | 0.997 | 0.40 |
| bytes_only_raw | 0.982 | 0.923 | 0.923 | 0.923 | 0.976 | 0.55 |
| hybrid_hist | 0.973 | 0.812 | 1.000 | 0.897 | 0.998 | 0.20 |
| hybrid_raw | 0.982 | 0.923 | 0.923 | 0.923 | 0.958 | 0.50 |

Interpretation:
- Bytes-only and hybrid are strong on this dataset; hybrid_hist matches bytes_only_hist.
- Structural-only yields high recall but lower precision.
- Rules-only are high precision but miss some samples.

---

## 8) On-device evaluation
Bench set path: /sdcard/Android/data/com.landfall.hybriddetector/files/bench_full/

| Metric | Value |
| --- | --- |
| Total samples | 206 |
| Benign samples | 100 |
| Malicious samples | 106 |
| Avg feature extraction time | 11.211 ms / file |
| Avg inference time | 0.062 ms / file |
| Log source | outputs/device_log.txt |

### 8.1 Device vs local TFLite parity
Comparison uses the same feature extractor logic and the same TFLite model.

| Metric | Value |
| --- | --- |
| Files compared | 206 |
| Max abs diff | 4.992e-05 |
| Mean abs diff | 1.421e-05 |
| Diff report | outputs/device_local_compare.csv |

This confirms device and Linux inference parity within ~5e-5.

---

## 9) Tests performed
- Linux feature extraction + training + evaluation: analysis/hybrid_extract.py, analysis/train_hybrid.py
- Ablation study: analysis/run_ablation.py
- TFLite conversion: analysis/convert_to_tflite.py
- On-device benchmark: Android app benchmark run, logcat captured in outputs/device_log.txt
- Device vs local parity: analysis/compare_device_local.py

---

## 10) Findings and interpretation
- Perfect recall on test at Thr=0.20 with a small FP count (3).
- Byte features dominate on this dataset; structural features are helpful but not sufficient alone.
- Rules provide high-precision alarms and are a good guardrail for known exploit patterns.
- On-device inference is fast enough for batch scanning.

---

## 11) False positives (why they happen)
False positives often appear when benign files share characteristics with malicious ones:
- Byte distributions similar to malicious samples.
- Structural quirks in benign TIFF encodings (e.g., very small dimensions or unusual opcodes).

Mitigation strategies:
- Increase benign diversity (camera RAW, lab TIFFs, BigTIFF).
- Tune the threshold upward if false positives are costly.
- Add a second validation set from in-the-wild benign sources.

---

## 12) Limitations
- Only 6 Landfall samples; generalization is not fully proven.
- Random split is not stratified by campaign or source.
- Magika mismatch rule is not enabled on-device (flag is 0).

---

## 13) Reproducibility (Linux)
1. Extract features:
   - python3 analysis/hybrid_extract.py --data-root data --output-dir outputs
2. Train and evaluate:
   - python3 analysis/train_hybrid.py --input-npz outputs/hybrid_features.npz --output-model outputs/hybrid_model.npz --output-metrics outputs/hybrid_metrics.json
3. Ablation:
   - python3 analysis/run_ablation.py --input-npz outputs/hybrid_features.npz --output-json outputs/ablation_results.json --output-md outputs/ablation_results.md
4. Convert to TFLite:
   - .venv-tf/bin/python analysis/convert_to_tflite.py --model-npz outputs/hybrid_model.npz --output-tflite outputs/hybrid_model.tflite --output-meta outputs/hybrid_model_meta.json

---

## 14) Recommendations
- Collect more Landfall variants and diversify general TIFF malware.
- Expand benign TIFF/DNG coverage to reduce FP rate.
- Consider a second-stage classifier or adaptive thresholds for high-risk environments.
