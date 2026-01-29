# LandFall Detector: Architecture & Implementation

## 1. Overview

> This file contains a mix of current architecture notes and legacy experiment history.
> For the **current training/export pipeline** (what’s actually shipped in the Android app), see:
> `docs/training_pipeline.md`.

LandFall Detector is a defensive security system for detecting malicious TIFF/DNG files on Android devices. It combines deterministic CVE-pattern rules with an ensemble of machine learning models to identify both known exploit patterns (e.g., CVE-2025-21043, CVE-2025-43300) and novel structural anomalies in image files.

The system is built around a **two-tier detection architecture**:

- **Tier 1 (CVE Rules)**: Fast deterministic pattern matching against known vulnerability signatures. If any rule triggers, the file is immediately flagged without ML inference.
- **Tier 2 (ML Ensemble)**: Four complementary models that analyze byte-level and structural features to catch malicious files that evade signature-based detection.

All ML models run on-device as TFLite graphs with no server dependency.

---

## 2. Detection Architecture

### 2.1 Tier 1: CVE Rule Engine

The CVE rule engine (`CveDetector.kt`) evaluates pre-parsed TIFF structural features against known vulnerability patterns. Rules are stateless and execute in sub-millisecond time.

**Active rules:**

| Rule ID | CVE | Trigger Condition |
|---------|-----|-------------------|
| Rule 1 | CVE-2025-21043 | Declared opcode list count > 1,000,000 |
| Rule 2 | CVE-2025-43300 | JPEG SOF3 component count != SamplesPerPixel (with SPP=2, Compression=7) |
| Rule 3a | TILE-CONFIG | tile_offsets_count != tile_byte_counts_count |
| Rule 3b | TILE-CONFIG | tile_offsets_count != expected count from image/tile geometry |
| Rule 3c | TILE-DIM | Any dimension (image or tile) > 0xFFFE7960 |

Unit tests: `analysis/test_cve_rules.py` (11 tests covering all rules and edge cases).

### 2.2 Tier 2: ML Anomaly Detectors

The Android app deploys three benign-only anomaly detectors (one per file family), routed by Magika:

| Model | Type | Input Dims | Scope | Role |
|-------|------|-----------|-------|------|
| TIFF AE | Unsupervised autoencoder | 550 | All TIFF | Reconstruction-error anomaly detector (primary for non-DNG TIFF) |
| TagSeq GRU-AE | Unsupervised GRU autoencoder | 512 x 12 | DNG only | Tag-sequence anomaly detector |
| JPEG AE | Unsupervised autoencoder | 35 | JPEG only | JPEG structural anomaly detector |
| Magika | Supervised DNN (Google) | 512 x 257 | All files | File-type identification |

**Decision logic** (single-file mode):

- **If Magika predicts `jpeg`**: run JPEG AE and flag MALICIOUS if `jpeg_score >= threshold`.
- **If Magika predicts `tiff`**:
  - Run Tier 1 CVE rules first; any CVE hit is MALICIOUS.
  - If the file is detected as DNG (via TagSeq parsing), use TagSeq GRU-AE and flag MALICIOUS if `tagseq_score >= threshold`.
  - Otherwise, use TIFF AE and flag MALICIOUS if `tiff_ae_score >= threshold`.
  - If DNG is suspected but TagSeq parsing/inference fails, fall back to TIFF AE.

---

## 3. Dataset

### 3.1 Composition

| Split | Benign | General Malware | LandFall | Total |
|-------|--------|----------------|----------|-------|
| Training | 5,773 | 45 | 0 | 5,818 |
| Holdout | 100 | 50 | 6 | 156 |

- **Benign**: Legitimate TIFF/DNG files from camera manufacturers, raw photography collections, and public image datasets.
- **General malware**: Malicious TIFF files from malware repositories (VirusTotal, MalwareBazaar) with diverse exploit techniques.
- **LandFall**: Samples specifically crafted to exploit CVE-2025-21043 (opcode overflow in DNG processing).

### 3.2 Data Pipeline

```
raw files
  |
  v
dedup_dataset.py          SHA-256 deduplication
  |
  v
build_bench_lists.py      Stratified train/holdout split
  |
  v
hybrid_extract.py         Feature extraction (hybrid model)
anomaly_feature_extract.py  Feature extraction (anomaly models)
tag_sequence_extract.py   Tag-sequence extraction (DNG only)
  |
  v
*.npz / *.csv             Feature matrices + metadata
```

Key scripts:
- `analysis/dedup_dataset.py` — SHA-256 hash deduplication across benign/malware pools
- `analysis/build_bench_lists.py` — Generates `train_list.txt` and `bench_holdout_list.txt`
- `analysis/build_dng_tagseq_lists.py` — Filters DNG-only subsets for TagSeq training

---

## 4. Feature Engineering

### 4.1 Byte Histogram Features (514 dims)

Computed by both `hybrid_extract.py` and `anomaly_feature_extract.py`:

- **First 1024 bytes**: 257-bin histogram (256 byte values + file-shorter-than-1024 flag)
- **Last 1024 bytes**: 257-bin histogram (256 byte values + file-shorter-than-1024 flag)
- Total: 514 dimensions, L1-normalized per region

### 4.2 Hybrid Structural Features (25 dims)

Extracted from TIFF IFD traversal in `hybrid_extract.py`:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `is_tiff` | File starts with TIFF magic bytes (II/MM) |
| 2 | `is_dng` | Contains DNG version tag (50706) |
| 3 | `min_width` | Minimum ImageWidth across IFDs |
| 4 | `min_height` | Minimum ImageLength across IFDs |
| 5 | `ifd_entry_max` | Maximum IFD entry count in any single IFD |
| 6 | `subifd_count_sum` | Total SubIFD pointers across all IFDs |
| 7 | `new_subfile_types_unique` | Count of distinct NewSubfileType values |
| 8 | `total_opcodes` | Total opcodes parsed from OpcodeList1/2/3 |
| 9 | `unknown_opcodes` | Opcodes with ID outside DNG spec range |
| 10 | `max_opcode_id` | Maximum opcode ID encountered |
| 11 | `opcode_list1_bytes` | Byte size of OpcodeList1 data |
| 12 | `opcode_list2_bytes` | Byte size of OpcodeList2 data |
| 13 | `opcode_list3_bytes` | Byte size of OpcodeList3 data |
| 14 | `max_declared_opcode_count` | Max big-endian uint32 at start of opcode list (declared count) |
| 15 | `spp_max` | Maximum SamplesPerPixel value across all IFDs |
| 16 | `compression_variety` | Count of distinct Compression tag values |
| 17 | `tile_count_ratio` | tile_offsets_count / expected_tile_count (0 if no tiles) |
| 18 | `zip_eocd_near_end` | ZIP EOCD signature found near file end |
| 19 | `zip_local_in_tail` | ZIP local file header found in tail region |
| 20 | `flag_opcode_anomaly` | Heuristic: unknown opcodes or extreme opcode list sizes |
| 21 | `flag_tiny_dims_low_ifd` | Heuristic: very small dimensions with few IFD entries |
| 22 | `flag_zip_polyglot` | Heuristic: both ZIP markers detected (polyglot file) |
| 23 | `flag_dng_jpeg_mismatch` | Heuristic: DNG with unexpected JPEG content |
| 24 | `flag_magika_ext_mismatch` | Magika file-type disagrees with TIFF extension |
| 25 | `flag_any` | Logical OR of all heuristic flags |

Features 14-17 are **CVE-derived features** added to improve detection of specific exploit patterns.

### 4.3 Anomaly Structural Features (36 dims)

Extracted in `anomaly_feature_extract.py`. Includes all hybrid structural features plus additional ratio and entropy features:

| Additional Feature | Description |
|-------------------|-------------|
| `max_width`, `max_height` | Maximum image dimensions |
| `total_pixels` | Product of max dimensions |
| `file_size` | Total file size in bytes |
| `bytes_per_pixel_milli` | file_size / total_pixels * 1000 |
| `pixels_per_mb` | total_pixels / (file_size / 1MB) |
| `opcode_list_bytes_total` | Sum of all opcode list sizes |
| `opcode_list_bytes_max` | Maximum opcode list size |
| `opcode_list_present_count` | How many of the 3 opcode lists exist |
| `opcode_bytes_ratio_permille` | opcode_bytes / file_size * 1000 |
| `opcode_bytes_per_opcode_milli` | opcode_bytes / total_opcodes * 1000 |
| `unknown_opcode_ratio_permille` | unknown_opcodes / total_opcodes * 1000 |
| `has_opcode_list1/2/3` | Binary presence flags |
| `header_entropy` | Shannon entropy of first 1024 bytes |
| `tail_entropy` | Shannon entropy of last 1024 bytes |
| `overall_entropy` | Shannon entropy of entire file |
| `entropy_gradient` | tail_entropy - header_entropy |

### 4.4 Tag Sequence Features (512 x 12 dims)

Extracted in `analysis/tag_sequence_extract.py` for DNG files only. Each IFD entry produces a 12-dimensional feature vector:

- Tag ID (normalized)
- Type ID
- Value count (log-scaled)
- Data offset (normalized by file size)
- IFD depth
- IFD index within depth
- Entry position within IFD
- SubIFD indicator
- Is-standard-DNG-tag flag
- Value magnitude (log-scaled)
- Inter-tag gap
- Offset-to-previous ratio

Sequences are zero-padded or truncated to 512 entries.

### 4.5 Preprocessing

**Hybrid model**: Structural features undergo `log1p` transformation for right-skewed fields (`max_declared_opcode_count`, `opcode_list1_bytes`, etc.), then z-score standardization. The normalization parameters (mean, std) are baked into the TFLite graph.

**Anomaly AE**: Same `log1p` + z-score pipeline, with parameters embedded in the TFLite model via preprocessing layers.

**TagSeq GRU-AE**: Raw normalized sequences fed directly to GRU encoder.

---

## 5. Model Architectures

### 5.1 Hybrid Logistic Regression (Legacy / Not Deployed)

- **Input**: 539 floats (514 byte histogram + 25 structural)
- **Architecture**: Single-layer logistic regression (sigmoid output)
- **Training**: SGD with L2 regularization (lambda=1e-4), 300 epochs, lr=0.2
- **Threshold**: 0.5 (training), 0.20 (Android deployment for higher sensitivity)
- **Output**: Probability of malicious class
- **Export**: `convert_to_tflite.py` creates TFLite graph with embedded log1p mask and z-score normalization

Training script: `analysis/train_hybrid.py` (kept for research; not used by the Android app)

### 5.2 Anomaly Autoencoder

- **Input**: 550 floats (514 byte histogram + 36 structural)
- **Architecture**: Dense encoder-decoder
  - Encoder: 550 → 256 (ReLU) → 64 (ReLU)
  - Decoder: 64 → 256 (ReLU) → 550 (linear)
- **Training**: Benign-only, MSE reconstruction loss
- **Scoring**: MSE between input and reconstruction
- **Threshold**: p97 = 2.510 (for deployment), p99 = 5.822 (for evaluation)
- **Export**: `export_anomaly_ae_tflite.py` creates TFLite graph with embedded preprocessing

Training script: `analysis/ae_train_eval.py`

### 5.3 TagSeq GRU Autoencoder

- **Input**: 512 x 12 sequence (DNG tag entries)
- **Architecture**: GRU encoder-decoder
  - Encoder: GRU (hidden=64)
  - Decoder: GRU (hidden=64) → Dense(12)
- **Training**: Benign DNG files only, MSE reconstruction loss, 30 epochs
- **Scoring**: Per-timestep MSE averaged over non-padded positions
- **Threshold**: p99 = 0.742
- **Scope**: DNG files only (returns N/A for non-DNG TIFF)
- **Export**: `export_tagseq_gru_ae_tflite.py` (requires Flex delegate on Android)

Training script: `analysis/tagseq_gru_ae_train.py`

### 5.4 Baseline Models (Evaluation Only)

Two additional models are trained for comparison but not deployed to Android:

**Isolation Forest** (`analysis/if_svm_train_eval.py`):
- Feature set: structural + byte histogram
- Trained on benign data only
- Threshold: p99 anomaly score

**One-Class SVM** (`analysis/if_svm_train_eval.py`):
- Feature set: structural + byte histogram
- RBF kernel, trained on benign data only
- Threshold: p99 decision function

### 5.5 Magika (File-Type Classifier)

Google's Magika model for file-type identification, used as a supplementary signal:
- Detects file-type mismatches (e.g., TIFF extension but JPEG content)
- Contributes to `flag_magika_ext_mismatch` structural feature
- Does not participate in the binary malicious/benign decision

---

## 6. Training Pipeline

### 6.1 Environment

- Python 3.x with TensorFlow, scikit-learn, NumPy
- Virtual environment: `.venv-tf/` (required for TF/sklearn-dependent scripts)
- System Python: sufficient for pure-NumPy scripts (hybrid_extract.py, train_hybrid.py)

### 6.2 End-to-End Workflow

```
Step 1: Feature Extraction
    hybrid_extract.py        → outputs/hybrid_features.npz
    anomaly_feature_extract.py → outputs/anomaly_features.npz
    tag_sequence_extract.py  → outputs/tagseq_features.npz

Step 2: Model Training
    train_hybrid.py          → outputs/hybrid_model.npz
    ae_train_eval.py         → outputs/ae_model_p99_holdout.keras
    if_svm_train_eval.py     → outputs/iforest_model_struct_bytes.pkl
    if_svm_train_eval.py     → outputs/ocsvm_model_struct_bytes.pkl
    tagseq_gru_ae_train.py   → outputs/tagseq_gru_ae.keras

Step 3: TFLite Export
    convert_to_tflite.py     → outputs/hybrid_model.tflite
    export_model_params.py   → android/.../hybrid_model_params.json
    export_anomaly_ae_tflite.py → outputs/anomaly_ae.tflite
    export_tagseq_gru_ae_tflite.py → outputs/tagseq_gru_ae.tflite

Step 4: Holdout Evaluation
    eval_holdout_models.py   → outputs/holdout_eval_models.json
```

### 6.3 Feature Normalization

The `log1p` transform is applied to right-skewed integer features before z-score standardization. The following fields use `log1p`:

**Hybrid model** (`log_fields` in `train_hybrid.py`):
`opcode_list1_bytes`, `opcode_list2_bytes`, `opcode_list3_bytes`, `max_declared_opcode_count`

**Anomaly models** (`log_fields` in `ae_train_eval.py`, `if_svm_train_eval.py`, `export_anomaly_ae_tflite.py`):
Same fields plus additional ratio features that are already in a reasonable range.

The TFLite export scripts bake a `log_mask` (boolean vector indicating which features get `log1p`) and the normalization parameters (mean, std) directly into the TFLite graph, so the Android app feeds raw feature values.

---

## 7. Evaluation Results

### 7.1 Holdout Benchmark

Evaluated on 156 held-out files (100 benign, 50 general malware, 6 LandFall):

| Model | Threshold | FPR | General Mal Recall | LandFall Recall | F1 |
|-------|-----------|-----|-------------------|----------------|-----|
| **Hybrid LR** | 0.500 | **0%** (0/100) | **98%** (49/50) | **100%** (6/6) | **0.991** |
| **Anomaly AE** (p99) | 5.822 | **0%** (0/100) | **94%** (47/50) | **100%** (6/6) | **0.972** |
| Isolation Forest (p99) | 0.692 | **0%** (0/100) | 4% (2/50) | **100%** (6/6) | — |
| One-Class SVM (p99) | 0.092 | 1% (1/100) | **100%** (50/50) | **100%** (6/6) | — |

### 7.2 Score Distributions

**Hybrid LR** (holdout):
- Benign: min=0.005, mean=0.035, max=0.068
- General malware: min=0.012, mean=0.977, max=1.000
- LandFall: min=0.997, mean=0.999, max=1.000

**Anomaly AE** (holdout):
- Benign: min=0.016, mean=0.164, max=1.434
- General malware: min=0.732, mean=803.4, max=1371.7
- LandFall: min=29.6, mean=30.8, max=33.0

The hybrid model produces well-separated score distributions: benign files cluster below 0.07 while malware is above 0.97, giving a large decision margin. The anomaly AE shows an even wider gap — benign scores are under 1.5 while malware reconstruction errors are orders of magnitude higher.

### 7.3 Training Set Metrics

**Hybrid LR** (train/val/test split of 5,818 files):
- Val: accuracy=0.999, precision=1.0, recall=0.90, F1=0.947
- Test: accuracy=0.999, precision=1.0, recall=0.857, F1=0.923, AUC=0.857

### 7.4 LandFall-Specific Detection

All models achieve 100% recall on LandFall samples, with high-confidence scores:
- Hybrid scores > 0.997 (max possible: 1.0)
- AE reconstruction error > 29.5 (threshold: 5.8)
- These files exhibit strong structural anomalies (extreme opcode counts, unusual dimensions) that both supervised and unsupervised models detect reliably.

---

## 8. Android Deployment

### 8.1 App Structure

The Android app (`android/HybridDetectorApp/`) is a native Kotlin application targeting Android API 26+.

**Kotlin classes** (14 files in `com.landfall.hybriddetector`):

| Class | Role |
|-------|------|
| `MainActivity` | UI, file selection, benchmark mode, orchestration |
| `TiffParser` | Binary TIFF/DNG parser — IFD traversal, tag extraction, opcode parsing |
| `CveDetector` | Tier 1 CVE rule evaluation |
| `AnomalyFeatureExtractor` | TIFF AE feature extraction |
| `AnomalyModelRunner` | TIFF AE TFLite inference |
| `AnomalyMeta` | TIFF AE metadata loader |
| `TagSequenceExtractor` | DNG tag-sequence extraction |
| `TagSeqModelRunner` | TagSeq GRU-AE TFLite inference (Flex delegate) |
| `TagSeqMeta` | TagSeq model metadata loader |
| `JpegParser` | JPEG structural parser |
| `JpegFeatureExtractor` | JPEG AE feature extraction |
| `JpegModelRunner` | JPEG AE TFLite inference |
| `JpegMeta` | JPEG AE metadata loader |
| `MagikaModelRunner` | Google Magika file-type classification |

### 8.2 Assets

| File | Size | Description |
|------|------|-------------|
| `anomaly_ae.tflite` | 346 KB | Anomaly AE with embedded preprocessing |
| `anomaly_model_meta.json` | 1.1 KB | Feature names, architecture, threshold |
| `jpeg_ae.tflite` | 11 KB | JPEG AE with embedded preprocessing |
| `jpeg_model_meta.json` | 1.1 KB | JPEG feature names, architecture, threshold |
| `tagseq_gru_ae.tflite` | 5.7 MB | TagSeq GRU-AE (requires Flex delegate) |
| `tagseq_gru_ae_meta.json` | 165 B | Sequence length, feature dim, threshold |
| `magika/` | — | Google Magika model and config |

### 8.3 Detection Flow (Single File)

```
User selects file via SAF picker
  |
  v
Copy to cache (avoid repeated content-resolver reads)
  |
  v
MagikaModelRunner.classify(path) → file-type label
  |
  v
Branch by type:
  +-- jpeg → JPEG parser/features → JPEG AE → jpeg score
  |
  +-- tiff → continue
  |
  +-- other → skip
  |
  v
TiffParser.parse(path) → TiffFeatures
  |
  v
TIER 1: CveDetector.evaluate(features)
  |
  +-- CVE hit? → Display "MALICIOUS (CVE rule match)", skip ML
  |
  v (no CVE hit)
TIER 2: Run ML models:
  |- TagSequenceExtractor.extract() → TagSeqRunner.predict()           → tagseq score (DNG only)
  |- AnomalyFeatureExtractor.extractFromParsed() → AnomalyRunner.predict() → tiff_ae score (non-DNG TIFF)
  |
  v
Decision:
  - DNG: tagseq_score >= threshold → MALICIOUS
  - non-DNG TIFF: tiff_ae_score >= threshold → MALICIOUS
```

### 8.4 Benchmark Mode

The app supports batch evaluation via `bench_list.txt` (placed on `/sdcard/` or in app-specific storage). In benchmark mode, all tiers run on every file (even if Tier 1 triggers) to allow full comparison. Per-file timing is reported for extract and inference phases.

### 8.5 TiffParser Feature Map

`TiffParser.toFeatureMap()` returns a `Map<String, Int>` with all structural features needed by both hybrid and anomaly extractors. The 4 CVE-derived features are computed from fields already parsed for CVE detection:

```kotlin
"max_declared_opcode_count" to maxDeclaredOpcodeCount,
"spp_max" to (sppValues.maxOrNull() ?: 0),
"compression_variety" to compressionValues.size,
"tile_count_ratio" to if (expectedTileCount > 0)
    (tileOffsetsCount * 1000 / expectedTileCount) else 0,
```

Note: `tile_count_ratio` is stored as permille (x1000) to fit the integer map type.

---

## 9. Project Structure

```
codex_landfall/
├── analysis/                    # Python ML pipeline
│   ├── hybrid_extract.py        # Hybrid feature extraction
│   ├── train_hybrid.py          # Hybrid LR training
│   ├── convert_to_tflite.py     # Hybrid → TFLite export
│   ├── export_model_params.py   # Hybrid weights → JSON (for attribution)
│   ├── anomaly_feature_extract.py # Anomaly feature extraction
│   ├── ae_train_eval.py         # Anomaly AE training + evaluation
│   ├── export_anomaly_ae_tflite.py # Anomaly AE → TFLite export
│   ├── deepsvdd_train_eval.py   # Deep SVDD experiment (not deployed)
│   ├── if_svm_train_eval.py     # Isolation Forest / OCSVM baselines
│   ├── tag_sequence_extract.py  # DNG tag-sequence feature extraction
│   ├── tagseq_gru_ae_train.py   # TagSeq GRU-AE training
│   ├── export_tagseq_gru_ae_tflite.py # TagSeq → TFLite export
│   ├── eval_holdout_models.py   # Unified holdout evaluation
│   ├── cve_rule_validation.py   # CVE rule validation on dataset
│   ├── test_cve_rules.py        # CVE rule unit tests (11 tests)
│   ├── cve_feature_ml_eval.py   # CVE feature impact evaluation
│   ├── dedup_dataset.py         # SHA-256 dataset deduplication
│   ├── build_bench_lists.py     # Train/holdout split generation
│   ├── build_dng_tagseq_lists.py # DNG-only subset for TagSeq
│   ├── analyze_benign_fp.py     # False-positive analysis
│   ├── anomaly_threshold_sweep.py # AE threshold sweep
│   ├── tagseq_threshold_sweep.py  # TagSeq threshold sweep
│   ├── tag_sequence_stats.py    # Tag sequence statistics
│   ├── run_ablation.py          # Feature ablation study
│   └── compare_device_local.py  # Device vs local score comparison
│
├── android/HybridDetectorApp/   # Android app
│   ├── app/src/main/
│   │   ├── java/com/landfall/hybriddetector/  # 13 Kotlin classes
│   │   ├── assets/              # TFLite models + metadata JSONs
│   │   └── res/layout/          # UI layout
│   ├── app/build.gradle         # Dependencies (TFLite, Flex delegate)
│   └── apk/                     # Prebuilt debug APK
│
├── outputs/                     # Training artifacts
│   ├── hybrid_features.npz      # Hybrid feature matrix (train)
│   ├── hybrid_features_holdout.npz
│   ├── anomaly_features.npz     # Anomaly feature matrix (train)
│   ├── anomaly_features_holdout.npz
│   ├── hybrid_model.npz         # Trained hybrid model
│   ├── hybrid_model.tflite      # Exported TFLite
│   ├── hybrid_metrics.json      # Training metrics
│   ├── ae_model_p99_holdout.keras
│   ├── anomaly_ae.tflite
│   ├── anomaly_model_meta.json
│   ├── tagseq_gru_ae.tflite
│   ├── tagseq_gru_ae_meta.json
│   ├── iforest_model_struct_bytes.pkl
│   ├── ocsvm_model_struct_bytes.pkl
│   ├── holdout_eval_models.json  # Final holdout comparison
│   ├── train_list.txt            # 5,818 training file paths
│   └── bench_holdout_list.txt    # 156 holdout file paths
│
├── data/                        # Raw sample files (not in repo)
│   ├── benign/
│   ├── general_mal/
│   └── LandFall/
│
└── docs/                        # Documentation
```

---

## 10. Reproducing Results

### 10.1 Feature Extraction

```bash
cd /path/to/codex_landfall

# Hybrid features
python3 analysis/hybrid_extract.py \
  --list-file outputs/train_list.txt \
  --output-dir outputs

# Anomaly features
.venv-tf/bin/python3 analysis/anomaly_feature_extract.py \
  --list-file outputs/train_list.txt \
  --output-npz outputs/anomaly_features.npz \
  --output-csv outputs/anomaly_features.csv

# Holdout features (repeat both with bench_holdout_list.txt)
```

### 10.2 Training

```bash
# Hybrid LR
python3 analysis/train_hybrid.py \
  --input-npz outputs/hybrid_features.npz \
  --output-model outputs/hybrid_model.npz \
  --output-metrics outputs/hybrid_metrics.json \
  --bytes-mode hist --epochs 300 --lr 0.2 --l2 1e-4

# Anomaly AE
.venv-tf/bin/python3 analysis/ae_train_eval.py \
  --input-npz outputs/anomaly_features.npz \
  --output-json outputs/ae_metrics_p99_holdout.json \
  --output-model outputs/ae_model_p99_holdout.keras \
  --bytes-mode hist --threshold-percentile 99.0

# IF/OCSVM baselines
.venv-tf/bin/python3 analysis/if_svm_train_eval.py \
  --input-npz outputs/anomaly_features.npz \
  --output-json outputs/iforest_metrics_struct_bytes.json \
  --output-model outputs/iforest_model_struct_bytes.pkl \
  --model iforest --feature-set struct_bytes \
  --eval-npz outputs/anomaly_features_holdout.npz
```

### 10.3 TFLite Export

```bash
.venv-tf/bin/python3 analysis/export_anomaly_ae_tflite.py \
  --input-npz outputs/anomaly_features.npz \
  --output-tflite outputs/anomaly_ae.tflite \
  --output-meta outputs/anomaly_model_meta.json \
  --bytes-mode hist --threshold-percentile 97.0
```

### 10.4 Deploy to Android

```bash
cp outputs/anomaly_ae.tflite android/HybridDetectorApp/app/src/main/assets/
cp outputs/anomaly_model_meta.json android/HybridDetectorApp/app/src/main/assets/
cp outputs/tagseq_gru_ae.tflite android/HybridDetectorApp/app/src/main/assets/
cp outputs/tagseq_gru_ae_meta.json android/HybridDetectorApp/app/src/main/assets/
cp outputs/jpeg_ae.tflite android/HybridDetectorApp/app/src/main/assets/
cp outputs/jpeg_model_meta.json android/HybridDetectorApp/app/src/main/assets/
```

### 10.5 Unit Tests

```bash
cd /path/to/codex_landfall
python3 -m pytest analysis/test_cve_rules.py -v
```
