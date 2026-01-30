# MediaThreatDetector — Architecture & Implementation

This project is an **on-device “media file zero-day detection”** prototype focused on catching exploit-like files via:
- **Type identification** (Magika)
- **Static high-confidence rules** (CVE-style patterns)
- **Benign-only anomaly detection** (autoencoders for JPEG/TIFF, GRU-AE for DNG)

The Android demo app name is **MediaThreatDetector** and the UI title is **Media Zero-day Detection**.

---

## 1) Verdict policy (3-way)

The app emits one of:
- **BENIGN**: detector ran and score < threshold, and no high-confidence rule match.
- **SUSPICIOUS**: anomaly score ≥ threshold, detector error/unsupported, or low-confidence static anomaly.
- **MALICIOUS**: high-confidence static CVE rule match.

Rationale: for B2B deployments where **false positives must be near zero**, we avoid calling something “malicious” unless we have a strong, explainable static match.

---

## 2) On-device pipeline (Android)

Entry point: `android/HybridDetectorApp/app/src/main/java/com/landfall/hybriddetector/MainActivity.kt`

### 2.1 Type routing (Magika)
1) Run Magika ONNX to classify file type (e.g., `jpeg`, `tiff`).
2) Compare Magika output with extension-implied type:
   - DNG (`.dng`) is treated as TIFF-compatible (`tiff` ↔ `dng` is allowed).
   - If incompatible → **SUSPICIOUS (type_mismatch)**.

### 2.2 TIFF/DNG path
1) Parse TIFF/DNG structure: `TiffParser`.
2) Run static rules: `CveDetector`.
   - If any **MALICIOUS** rule hits → **MALICIOUS**.
   - If any **SUSPICIOUS** static anomaly hits → **SUSPICIOUS**.
3) Route by DNG:
   - If DNG: run **TagSeq GRU-AE** (required). Failure → **SUSPICIOUS**.
   - Else: run **TIFF AE**. Failure → **SUSPICIOUS**.

### 2.3 JPEG path
1) Parse JPEG header/segments: `JpegParser`.
2) Run **JPEG AE**. Failure → **SUSPICIOUS**.

### 2.4 Unsupported types
If Magika is not `jpeg` or `tiff` → **SUSPICIOUS (unsupported_type)**.

---

## 3) Detectors “in production” (assets shipped in the APK)

Android assets live under: `android/HybridDetectorApp/app/src/main/assets/`

### 3.1 Magika (type identification)
- Model: `magika/model.onnx` (pretrained, not trained in this repo)
- Runner: `MagikaModelRunner`

### 3.2 TIFF AE (non‑DNG TIFF anomaly detector)
- Model: `anomaly_ae.tflite`
- Meta: `anomaly_model_meta.json` (includes threshold + feature schema)
- Input: **550 floats** = `[514 byte-hist features] + [36 structural features]`
- Output: **1 float** anomaly score (MSE in normalized feature space)

Feature extraction code:
- Offline: `analysis/anomaly_feature_extract.py`
- Android: `AnomalyFeatureExtractor` + `TiffParser`

### 3.3 DNG TagSeq GRU‑AE (DNG anomaly detector)
- Model: `tagseq_gru_ae.tflite` (requires TF Lite Flex ops)
- Meta: `tagseq_gru_ae_meta.json` (threshold + max sequence length)
- Inputs (multiple tensors):
  - `features`: `[max_seq_len, 12]` floats per IFD-entry
  - `tag_ids`, `type_ids`, `ifd_kinds`: `[max_seq_len]` integer sequences
- Output: `[max_seq_len, 12]` reconstructed features
- Score (Android): average per-timestep SSE over the true (unpadded) length

Important robustness detail:
- TIFF `type_id` is mapped to 1..12 else 13 (UNK); 0 is padding.

### 3.4 JPEG AE (JPEG anomaly detector)
- Model: `jpeg_ae.tflite`
- Meta: `jpeg_model_meta.json` (threshold + feature schema)
- Input: **35 floats** of JPEG structural features
- Output: **1 float** anomaly score (MSE in normalized feature space)

Feature extraction code:
- Offline: `analysis/jpeg_feature_extract.py`
- Android: `JpegParser` + `JpegFeatureExtractor`

---

## 4) Static rules (CVE-style)

File: `android/HybridDetectorApp/app/src/main/java/com/landfall/hybriddetector/CveDetector.kt`

### 4.1 MALICIOUS (high confidence)
- `CVE-2025-21043`: declared opcode list count > 1,000,000
- `CVE-2025-43300`: SOF3 component mismatch (DNG SubIFD with JPEG lossless)

### 4.2 SUSPICIOUS (lower confidence)
- `TILE-CONFIG`: tile offsets/counts inconsistencies and “unexpected tile count”
- `TILE-DIM`: extreme dimension threshold

Rationale: real-world TIFFs can legitimately trigger tile anomalies, so we treat these as **warnings**.

---

## 5) Training / export pipeline (current)

Most training inputs are intentionally **not committed** (see `.gitignore`).

### 5.1 Environments
- `python3` (feature extraction / utilities)
- `.venv-tf/bin/python3` (TensorFlow training + TFLite export)

### 5.2 Data layout (local, gitignored)
- TIFF/DNG:
  - `data/benign_data/**`
  - `data/general_mal/**` (evaluation)
  - `data/LandFall/**` (evaluation)
- JPEG:
  - `JPG_dataset/**` (benign)
  - `JPEG_malware/**` (evaluation)

Public real-world TIFF fixtures downloader:
```bash
python3 analysis/download_realworld_tiffs.py
```

### 5.3 Feature extraction
TIFF:
```bash
python3 analysis/anomaly_feature_extract.py \
  --data-root data \
  --output-npz outputs/anomaly_features.npz \
  --output-csv outputs/anomaly_features.csv
```

DNG TagSeq:
```bash
python3 analysis/build_dng_tagseq_lists.py --benign-root data/benign_data --landfall-root data/LandFall

python3 analysis/tag_sequence_extract.py --list-file outputs/dng_tagseq_train_list.txt   --dng-only --max-seq-len 512 --output outputs/tagseq_dng_train.npz
python3 analysis/tag_sequence_extract.py --list-file outputs/dng_tagseq_holdout_list.txt --dng-only --max-seq-len 512 --output outputs/tagseq_dng_holdout.npz
python3 analysis/tag_sequence_extract.py --list-file outputs/dng_tagseq_landfall_list.txt --dng-only --max-seq-len 512 --output outputs/tagseq_dng_landfall.npz
```

JPEG:
```bash
python3 analysis/jpeg_feature_extract.py \
  --benign-root JPG_dataset \
  --malware-root JPEG_malware \
  --output-dir outputs
```

### 5.4 Train + export TFLite
TIFF AE:
```bash
.venv-tf/bin/python3 analysis/export_anomaly_ae_tflite.py \
  --input-npz outputs/anomaly_features.npz \
  --train-scope tiff_non_dng \
  --bytes-mode hist \
  --threshold-percentile 97.0 \
  --output-tflite outputs/anomaly_ae.tflite \
  --output-meta outputs/anomaly_model_meta.json
```

JPEG AE:
```bash
.venv-tf/bin/python3 analysis/export_jpeg_ae_tflite.py \
  --input-npz outputs/jpeg_features.npz \
  --threshold-percentile 99.0 \
  --output-tflite outputs/jpeg_ae.tflite \
  --output-meta outputs/jpeg_model_meta.json
```

DNG TagSeq GRU-AE:
```bash
.venv-tf/bin/python3 analysis/tagseq_gru_ae_train.py
.venv-tf/bin/python3 analysis/export_tagseq_gru_ae_tflite.py
.venv-tf/bin/python3 analysis/calibrate_tagseq_tflite_threshold.py
```

Deploy to Android assets:
```bash
cp outputs/anomaly_ae.tflite android/HybridDetectorApp/app/src/main/assets/anomaly_ae.tflite
cp outputs/anomaly_model_meta.json android/HybridDetectorApp/app/src/main/assets/anomaly_model_meta.json

cp outputs/jpeg_ae.tflite android/HybridDetectorApp/app/src/main/assets/jpeg_ae.tflite
cp outputs/jpeg_model_meta.json android/HybridDetectorApp/app/src/main/assets/jpeg_model_meta.json

cp outputs/tagseq_gru_ae.tflite android/HybridDetectorApp/app/src/main/assets/tagseq_gru_ae.tflite
cp outputs/tagseq_gru_ae_meta.json android/HybridDetectorApp/app/src/main/assets/tagseq_gru_ae_meta.json
```

---

## 6) Evaluation & tests

### 6.1 Offline (local) three-way verdict evaluation
Runs the **current shipped TFLite assets** against the local NPZ feature sets and reports
`BENIGN / SUSPICIOUS / MALICIOUS` rates:
```bash
.venv-tf/bin/python3 analysis/eval_three_way_verdict.py --with-cve --output-json outputs/three_way_verdict_eval.json
```

### 6.2 Unit tests (static rules)
```bash
python3 analysis/test_cve_rules.py
```

---

## 7) Android build + demo usage

Build:
```bash
cd android/HybridDetectorApp
JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 ./gradlew assembleDebug
```

Install:
```bash
adb install -r android/HybridDetectorApp/app/build/outputs/apk/debug/app-debug.apk
```

Demo options:
- “Choose file” uses Android’s document picker (copied into app cache before scanning).
- “Choose from samples” expects a folder under `/sdcard/Download/...` (see the app UI).
- “Run benchmark” reads `/sdcard/bench_list.txt` or the app’s `externalFilesDir/bench_list.txt` and logs per-file decisions.

---

## 8) Known limitations / production notes

- **Thresholds must be calibrated on a large, source-heldout benign set** for low false positives.
- **Real-world TIFF variance is huge** (GeoTIFF/OME/multipage/tiling) → expect many “rare-but-benign” outliers.
- **BigTIFF (magic 43)** appears in public corpora; current lightweight parsing/features are primarily for classic TIFF (magic 42).
- `MANAGE_EXTERNAL_STORAGE` is used for the demo; for Play Store distribution, prefer Storage Access Framework instead.

