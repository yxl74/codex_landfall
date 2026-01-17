# Hybrid TIFF/DNG Detection Design

This document explains the hybrid detection system, the features we extract, how the ML model is trained, and the ablation study results. It is written as an educational guide with minimal ML assumptions.

## Goals and constraints

- Detect Landfall-style DNG exploits and generic TIFF malware.
- Run on-device (Android), so feature extraction and inference must be fast.
- Keep false positives low on diverse benign TIFF/DNG images.
- Provide interpretable signals for debugging and tuning.

## Data summary

- Benign: 630 files in `data/benign_data`
- Landfall: 6 files in `data/LandFall`
- General malware: 100 files in `data/general_mal`
- There are 5 cross-label duplicates (Landfall appears in general_mal). We deduplicate by SHA-256 before training to avoid leakage.

## TIFF and DNG background (short version)

- TIFF files start with a 2-byte endian marker (`II` or `MM`) and a magic number (42).
- TIFF stores metadata in Image File Directories (IFDs). Each IFD is a list of tag entries.
- DNG is a TIFF-based format with extra tags:
  - SubIFDs (tag 330) to store multiple images (preview, main).
  - Opcode lists (tags 51008/51009/51022) with per-image processing instructions.
- Landfall abuses DNG opcode lists (huge counts, unknown opcode IDs) and often embeds a ZIP payload (polyglot).

## Threat patterns observed in this dataset

- Landfall DNGs have thousands of opcodes, including unknown opcode IDs (e.g., ID 23).
- Many Landfall samples are polyglots (ZIP EOCD in last 64KB).
- General malware TIFFs tend to have:
  - Very small or zero dimensions.
  - Very small IFD entry counts.
- Benign DNGs have small opcode counts and no unknown opcode IDs.

## Hybrid pipeline overview

We combine three sources of evidence:

1. **Magika-style bytes features** (fast, content-only)
2. **Structural TIFF/DNG features** (parsing-based, semantic)
3. **Rules and type mismatch checks** (high precision)

The final model is a lightweight logistic regression classifier trained on the combined features. Rules can be used as a high-confidence gate.

## Feature extraction

### 1) Magika-style byte features

We mirror Magika’s idea: use only the first and last 1024 bytes of the file (after stripping leading/trailing whitespace). Two representations are supported:

- **Raw**: 2048 bytes, normalized to [0, 1].
- **Histogram**: 257-bin histogram (0..255 + padding token) for the first and last blocks.

This is fast and robust to large files because we never read the full file.

### 2) Structural TIFF/DNG features

We parse only the header and IFDs (no pixel data), and extract:

- `is_tiff`, `is_dng`
- `min_width`, `min_height`
- `ifd_entry_max`
- `subifd_count_sum`
- `new_subfile_types_unique`
- `total_opcodes`, `unknown_opcodes`, `max_opcode_id`
- `opcode_list1_bytes`, `opcode_list2_bytes`, `opcode_list3_bytes`
- `zip_eocd_near_end`, `zip_local_in_tail`

These are fast to compute and capture the semantics of DNG/TIFF structures.

### 3) Type mismatch checks

Two mismatch checks are available:

- **Header vs extension**: `.jpg` but TIFF/DNG header.
- **Magika vs extension**: Magika says DNG/TIFF but extension is `.jpg`.

The second check requires Magika inference. It is optional and only used if Magika is available.

## Rules (high-confidence)

These rules are built into `analysis/hybrid_extract.py`:

- **Opcode anomaly**: DNG + (total opcodes > 100 OR unknown opcode IDs > 0)
- **Tiny dims + low IFD** (non-DNG TIFF): `ifd_entry_max <= 10` AND min dimension <= 16
- **ZIP polyglot**: EOCD within last 64KB + local ZIP header in tail
- **DNG disguised as JPEG** (extension/magic mismatch)
- **Magika mismatch** (if enabled)

`flag_any` is the OR of all rule flags and can be used as a strong baseline.

## ML model

We use logistic regression (a simple linear classifier). It is:

- Fast to train and easy to deploy.
- Interpretable (weights reflect feature importance).
- Easy to calibrate by changing the decision threshold.

### Training details

1. **Dedup by hash** to avoid leakage (same file in multiple labels).
2. **Split**: 70% train, 15% validation, 15% test.
3. **Feature scaling**:
   - Log1p for heavy-tailed counts (opcodes, byte sizes).
   - Standardize using train mean/std.
4. **Class weighting** to reduce imbalance bias.
5. **Threshold** chosen on validation set by best F1.

Scripts:

- Feature extraction: `analysis/hybrid_extract.py`
- Training: `analysis/train_hybrid.py`
- Ablation: `analysis/run_ablation.py`

## Ablation study

Test-set metrics (hash-deduped split, seed 42):

| model | acc | prec | recall | f1 | roc_auc | thr |
| --- | --- | --- | --- | --- | --- | --- |
| rules_flag_any | 0.991 | 1.000 | 0.923 | 0.960 | 0.000 | 0.50 |
| struct_only | 0.883 | 0.500 | 1.000 | 0.667 | 0.990 | 0.10 |
| bytes_only_hist | 0.973 | 0.812 | 1.000 | 0.897 | 0.997 | 0.40 |
| bytes_only_raw | 0.982 | 0.923 | 0.923 | 0.923 | 0.976 | 0.55 |
| hybrid_hist | 0.973 | 0.812 | 1.000 | 0.897 | 0.998 | 0.20 |
| hybrid_raw | 0.982 | 0.923 | 0.923 | 0.923 | 0.958 | 0.50 |

Interpretation:

- **Rules** are very strong (high precision, high recall).
- **Bytes-only** models are strong, but can still produce false positives.
- **Structural-only** has perfect recall but lower precision.
- **Hybrid** is a good balance and helps recover cases missed by rules.

If you want very low false positives, the rules-only path is a good high-confidence gate. If you want maximum recall, use the hybrid model with a tuned threshold.

## On-device deployment approach

1. Extract Magika-style bytes (first/last 1024).
2. Parse TIFF/DNG headers for structural features.
3. Apply rule flags (fast).
4. If not flagged, run the logistic regression model (small).

This keeps latency low and avoids reading the full file.

## How to reproduce

1. Extract features:
   - `python3 analysis/hybrid_extract.py --data-root data --output-dir outputs`
2. Train and evaluate:
   - `python3 analysis/train_hybrid.py --input-npz outputs/hybrid_features.npz --output-model outputs/hybrid_model.npz --output-metrics outputs/hybrid_metrics.json`
3. Ablation:
   - `python3 analysis/run_ablation.py --input-npz outputs/hybrid_features.npz --output-json outputs/ablation_results.json --output-md outputs/ablation_results.md`

## Notes and caveats

- The dataset is small. Results will change with more benign TIFF/DNG varieties.
- General malware contains some Landfall duplicates; dedup by hash is required.
- Magika inference is optional and needs onnxruntime; the mismatch rule still works using header magic alone.

## Next steps

- Expand benign TIFF/DNG diversity (camera RAW, OME-TIFF, BigTIFF).
- Add more generic malicious TIFF samples.
- Validate on-device performance and memory usage.
- Move the model to TFLite or ONNX with quantized weights.
