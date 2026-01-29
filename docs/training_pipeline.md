# Training Pipeline (Current Implementation)

This document describes the **current** end-to-end training/export pipeline for the models shipped in the Android app:
- `anomaly_ae.tflite` (TIFF, non‑DNG)
- `tagseq_gru_ae.tflite` (DNG)
- `jpeg_ae.tflite` (JPEG)
- Magika `model.onnx` (file type identification; pre-trained, not trained here)

For feature definitions, see `docs/model_features.md`.

> Note: The repository contains legacy/experiment artifacts (e.g., hybrid LR) and older reports. This document focuses on what’s currently deployed.

---

## 0) Environments / prerequisites

Two Python environments are used:

1) **System Python** (`python3`) for feature extraction and dataset utilities (no TensorFlow required).
2) **TensorFlow venv** (`.venv-tf/bin/python3`) for training/export of neural models to TFLite.

Android build requires Java 17:
- `JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64`

---

## 1) Data layout and provenance (inputs)

All raw data lives in local folders and is intentionally not committed (see `.gitignore`).

### 1.1 TIFF / DNG data (under `data/`)

**Benign**
- `data/benign_data/raw_pixls/**/*.dng`  
  Primary benign DNG corpus (used for TagSeq GRU-AE).
- `data/benign_data/ome_tiff/**/*.tiff`  
  Real-world OME-TIFF (microscopy).
- `data/benign_data/nasa_neo/**/*.TIFF`  
  Real-world TIFF (remote sensing).
- `data/benign_data/generated_tiff_from_jpeg*/**/*.tiff`  
  Generated benign TIFFs (from `JPG_dataset/`) to broaden non‑DNG TIFF coverage.
- `data/benign_data/real_world_tiff/**/*.(tif|tiff)`  
  Downloaded public TIFF fixtures (libtiff, GeoTIFF samples, GDAL autotests).  
  Downloader: `analysis/download_realworld_tiffs.py`.

**Malicious / evaluation**
- `data/general_mal/*`  
  “General malware” TIFF samples (used for evaluation; not used for benign-only training).
- `data/LandFall/*`  
  LandFall DNG samples (used for evaluation of the DNG model).

> Limitation: BigTIFF (TIFF magic 43) is present in some public corpora but is not fully parsed by the current TIFF parser/features. These files currently behave like “unsupported/unknown TIFF” for feature extraction.

### 1.2 JPEG data (top-level folders)

**Benign**
- `JPG_dataset/**` (camera JPEG corpus; large)

**Malicious / evaluation**
- `JPEG_malware/**`

---

## 2) Routing and model roles (what is “in production”)

The Android app does:
1) **Magika**: classify file type (`jpeg` vs `tiff` vs other).
2) **Static rule**: if extension-implied type conflicts with Magika (with `dng` treated as TIFF-compatible), mark as malware.
3) If `jpeg`: run **JPEG AE**.
4) If `tiff`: run **CVE rules** first; if no CVE hit:
   - If DNG: run **TagSeq GRU-AE** (**required**, no TIFF fallback).
   - Else: run **TIFF AE**.

Runtime code: `android/HybridDetectorApp/app/src/main/java/com/landfall/hybriddetector/MainActivity.kt`.

---

## 3) Feature extraction (offline)

### 3.1 TIFF AE feature extraction

Script: `analysis/anomaly_feature_extract.py`

Outputs:
- `outputs/anomaly_features.npz`
- `outputs/anomaly_features.csv`

What it contains:
- `X_bytes`: Magika-like 2048 tokens (head/tail bytes) used to form histograms later.
- `X_struct`: 36 structural/entropy features from a lightweight TIFF parse.
- `labels`: inferred from path (`benign` / `general_mal` / `landfall`).

Run:
```bash
python3 analysis/anomaly_feature_extract.py \
  --data-root data \
  --output-npz outputs/anomaly_features.npz \
  --output-csv outputs/anomaly_features.csv
```

### 3.2 DNG TagSeq feature extraction

Scripts:
- `analysis/build_dng_tagseq_lists.py` (builds deterministic lists)
- `analysis/tag_sequence_extract.py` (extracts padded sequences)

Key outputs:
- `outputs/dng_tagseq_train_list.txt`
- `outputs/dng_tagseq_holdout_list.txt`
- `outputs/dng_tagseq_landfall_list.txt`
- `outputs/tagseq_dng_train.npz`
- `outputs/tagseq_dng_holdout.npz`
- `outputs/tagseq_dng_landfall.npz`

Run:
```bash
python3 analysis/build_dng_tagseq_lists.py --benign-root data/benign_data --landfall-root data/LandFall

python3 analysis/tag_sequence_extract.py --list-file outputs/dng_tagseq_train_list.txt   --dng-only --max-seq-len 512 --output outputs/tagseq_dng_train.npz
python3 analysis/tag_sequence_extract.py --list-file outputs/dng_tagseq_holdout_list.txt --dng-only --max-seq-len 512 --output outputs/tagseq_dng_holdout.npz
python3 analysis/tag_sequence_extract.py --list-file outputs/dng_tagseq_landfall_list.txt --dng-only --max-seq-len 512 --output outputs/tagseq_dng_landfall.npz
```

Important preprocessing details:
- `type_ids` are mapped to **TIFF types 1..12**, else **13 (UNK)**; **0 is padding**.
- Sequences are padded/truncated to `max_seq_len=512`.

### 3.3 JPEG feature extraction

Script: `analysis/jpeg_feature_extract.py`

Output:
- `outputs/jpeg_features.npz`
- `outputs/jpeg_features.csv`

Run:
```bash
python3 analysis/jpeg_feature_extract.py \
  --benign-root JPG_dataset \
  --malware-root JPEG_malware \
  --output-dir outputs
```

---

## 4) Training + export (TFLite models)

All three detectors are **benign-only anomaly detectors**:
- Train on benign samples only.
- Score = reconstruction error (MSE).
- Threshold is chosen on a benign validation set, and is calibrated using **TFLite outputs** (to match on-device scoring).

### 4.1 TIFF anomaly AE (non‑DNG TIFF)

Script: `analysis/export_anomaly_ae_tflite.py`

Technique:
- Dense autoencoder over a concatenation of:
  - byte histograms (514 dims: head+tail 1024-byte histograms)
  - TIFF structural features (36 dims)
- Training: benign-only MSE reconstruction
- Score: per-sample mean squared error in normalized feature space
- Threshold: percentile (default p97) over benign validation scores, computed using the exported TFLite model

Run:
```bash
.venv-tf/bin/python3 analysis/export_anomaly_ae_tflite.py \
  --input-npz outputs/anomaly_features.npz \
  --train-scope tiff_non_dng \
  --bytes-mode hist \
  --threshold-percentile 97.0 \
  --output-tflite outputs/anomaly_ae.tflite \
  --output-meta outputs/anomaly_model_meta.json
```

Deploy:
```bash
cp outputs/anomaly_ae.tflite android/HybridDetectorApp/app/src/main/assets/anomaly_ae.tflite
cp outputs/anomaly_model_meta.json android/HybridDetectorApp/app/src/main/assets/anomaly_model_meta.json
```

### 4.2 DNG TagSeq GRU-AE

Training script: `analysis/tagseq_gru_ae_train.py`  
Export script: `analysis/export_tagseq_gru_ae_tflite.py`  
TFLite threshold calibration helper: `analysis/calibrate_tagseq_tflite_threshold.py`

Technique:
- GRU autoencoder over **IFD-entry sequences**.
- Inputs:
  - numeric per-entry features (12 floats)
  - embeddings for `tag_ids`, `type_ids`, `ifd_kinds`
- Loss masking:
  - training uses `sample_weight` to ignore padded timesteps
  - encoder GRU also receives an explicit mask derived from padding (tag_id=0)
- Score:
  - average reconstruction SSE per timestep over the true (unpadded) length

Train:
```bash
.venv-tf/bin/python3 analysis/tagseq_gru_ae_train.py \
  --train-npz outputs/tagseq_dng_train.npz \
  --holdout-npz outputs/tagseq_dng_holdout.npz \
  --landfall-npz outputs/tagseq_dng_landfall.npz \
  --out-model outputs/tagseq_gru_ae.keras \
  --out-metrics outputs/tagseq_gru_ae_metrics.json
```

Export:
```bash
.venv-tf/bin/python3 analysis/export_tagseq_gru_ae_tflite.py \
  --keras outputs/tagseq_gru_ae.keras \
  --out outputs/tagseq_gru_ae.tflite
```

Calibrate threshold on TFLite output (recommended):
```bash
.venv-tf/bin/python3 analysis/calibrate_tagseq_tflite_threshold.py \
  --tflite outputs/tagseq_gru_ae.tflite \
  --holdout-npz outputs/tagseq_dng_holdout.npz \
  --landfall-npz outputs/tagseq_dng_landfall.npz \
  --threshold-percentile 99.0 \
  --output-meta outputs/tagseq_gru_ae_meta.json
```

Deploy:
```bash
cp outputs/tagseq_gru_ae.tflite android/HybridDetectorApp/app/src/main/assets/tagseq_gru_ae.tflite
cp outputs/tagseq_gru_ae_meta.json android/HybridDetectorApp/app/src/main/assets/tagseq_gru_ae_meta.json
```

> Note: TagSeq uses Select TF ops (“Flex”) and therefore requires `tensorflow-lite-select-tf-ops` in the Android app dependencies (`android/HybridDetectorApp/app/build.gradle`).

### 4.3 JPEG anomaly AE

Script: `analysis/export_jpeg_ae_tflite.py`

Technique:
- Dense autoencoder on **JPEG structural features** (marker/segment statistics)
- Training: benign-only MSE reconstruction
- Augmentation (benign-only, optional but enabled in the script):
  - resegment combined DQT/DHT blocks into per-table segments
  - APP marker insertion to match camera JPEG distributions
  - append small tail bytes after EOI (common in real camera outputs)
- Threshold: percentile on benign validation scores computed using exported TFLite output

Train/export:
```bash
.venv-tf/bin/python3 analysis/export_jpeg_ae_tflite.py \
  --input-npz outputs/jpeg_features.npz \
  --output-tflite outputs/jpeg_ae.tflite \
  --output-meta outputs/jpeg_model_meta.json
```

Deploy:
```bash
cp outputs/jpeg_ae.tflite android/HybridDetectorApp/app/src/main/assets/jpeg_ae.tflite
cp outputs/jpeg_model_meta.json android/HybridDetectorApp/app/src/main/assets/jpeg_model_meta.json
```

---

## 5) Tests and validation currently in the repo

### 5.1 Unit tests (CVE rules)

File: `analysis/test_cve_rules.py` (unittest; no pytest required)

Run:
```bash
python3 analysis/test_cve_rules.py
```

### 5.2 Offline sanity checks / evaluations

- Each export script prints evaluation summaries and writes thresholds into `*_meta.json`.
  - TIFF AE: `analysis/export_anomaly_ae_tflite.py` prints benign test FPR + malware recall (for files within the training scope).
  - JPEG AE: `analysis/export_jpeg_ae_tflite.py` prints benign test FPR + malware recall.
  - TagSeq: `analysis/tagseq_gru_ae_train.py` writes `outputs/tagseq_gru_ae_metrics.json`; `analysis/calibrate_tagseq_tflite_threshold.py` writes calibrated meta JSON.
- Ablations:
  - `analysis/eval_anomaly_ae_ablation.py` compares bytes-only vs struct-only vs combined AEs to quantify histogram dominance.

### 5.3 On-device testing (benchmark mode)

The Android app supports a benchmark mode that scans a list of paths and logs decisions:
- Trigger: `adb shell am start -n com.landfall.hybriddetector/.MainActivity --ez auto true`
- List file searched in:
  - `/sdcard/Android/data/com.landfall.hybriddetector/files/bench_list.txt`
  - or `/sdcard/bench_list.txt`
- Results: `logcat` tag `HybridDetector` (per-file decisions + timing stats)

---

## 6) Known limitations / open technical debt

- BigTIFF parsing is incomplete (magic=43); current pipelines target classic TIFF (magic=42).
- The TIFF AE anomaly score is dimension-weighted; byte histograms are 514/550 dims and therefore dominate the scalar MSE unless explicitly reweighted.
- “General malware” labels are not a perfect exploitability ground truth; see `docs/ground_truth_plan.md` for a crash-oracle-based path forward.

