# Holdout Benchmark Report (DNG-only benign)

This document captures the most recent end-to-end experiment after dataset cleanup, with a
hash-based holdout split and a DNG-only benign benchmark on device.

---

## 1) Goals
- Remove duplicate samples across folders (LandFall should not live in general_mal).
- Create a **clean holdout benchmark** with no overlap with training (by hash).
- On-device evaluation uses **benign DNG files only**.

---

## 2) Dataset cleanup and dedup
Tool: `analysis/dedup_dataset.py`

Policy:
- Keep one copy per hash.
- Prefer LandFall > general_mal > benign_data.
- Move duplicates to `data/_duplicates/...` (do not delete).

Result summary (after dedup):
- benign: 5873
- LandFall: 6
- general_mal: 95
- duplicates moved: 5 (LandFall hashes removed from general_mal)

---

## 3) Holdout benchmark design (hash-based)
Tool: `analysis/build_bench_lists.py`

Holdout policy:
- LandFall: **all** samples (unique by hash).
- general_mal: **50** randomly selected by hash (seed=42).
- benign: **100 DNG** files only (hash-based, seed=42).

Artifacts:
- `outputs/bench_holdout_manifest.json`
- `outputs/bench_holdout_list.txt`
- `outputs/train_list.txt`

Counts:
- Bench: LandFall 6, general_mal 50, benign DNG 100 (156 total)
- Training list: 5818 files (benign 5773, general_mal 45, LandFall 0)

---

## 4) Training (Linux)

### 4.1 Hybrid (supervised)
Command:
```
python3 analysis/hybrid_extract.py --data-root data --output-dir outputs --list-file outputs/train_list.txt
python3 analysis/train_hybrid.py --input-npz outputs/hybrid_features.npz --output-model outputs/hybrid_model.npz --output-metrics outputs/hybrid_metrics.json
.venv-tf/bin/python analysis/convert_to_tflite.py --model-npz outputs/hybrid_model.npz --output-tflite outputs/hybrid_model.tflite --output-meta outputs/hybrid_model_meta.json
.venv-tf/bin/python analysis/export_model_params.py --model-npz outputs/hybrid_model.npz --output-json android/HybridDetectorApp/app/src/main/assets/hybrid_model_params.json
```

Hybrid metrics (from `outputs/hybrid_metrics.json`):

| Split | Thr | Acc | Prec | Recall | F1 | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Val | 0.45 | 0.999 | 1.000 | 0.900 | 0.947 | 9 | 862 | 0 | 1 |
| Test | 0.45 | 0.999 | 1.000 | 0.857 | 0.923 | 6 | 867 | 0 | 1 |

Note: LandFall is **excluded from training**, since all LandFall samples are in the holdout bench.

### 4.2 Autoencoder (benign-only)
Command:
```
.venv-tf/bin/python analysis/anomaly_feature_extract.py \
  --data-root data \
  --list-file outputs/train_list.txt \
  --output-npz outputs/anomaly_features.npz \
  --output-csv outputs/anomaly_features.csv

.venv-tf/bin/python analysis/ae_train_eval.py \
  --input-npz outputs/anomaly_features.npz \
  --threshold-percentile 99 \
  --output-json outputs/ae_metrics_p99_holdout.json \
  --output-model outputs/ae_model_p99_holdout.keras

.venv-tf/bin/python analysis/export_anomaly_ae_tflite.py \
  --input-npz outputs/anomaly_features.npz \
  --output-tflite outputs/anomaly_ae.tflite \
  --output-meta outputs/anomaly_model_meta.json \
  --threshold-percentile 99 \
  --seed 42
```

AE metrics (from `outputs/ae_metrics_p99_holdout.json`):

| Split | Thr (p99) | Acc | Prec | Recall | F1 | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Val | 6.232 | 0.989 | 0.500 | 0.900 | 0.643 | 9 | 853 | 9 | 1 |
| Test | 6.232 | 0.993 | 0.538 | 1.000 | 0.700 | 7 | 861 | 6 | 0 |

For on-device testing, AE threshold is **manually set to 30.0** in:
`android/HybridDetectorApp/app/src/main/assets/anomaly_model_meta.json`

---

## 5) On-device benchmark (holdout, benign DNG only)

Bench location (device):
`/storage/emulated/0/Android/data/com.landfall.hybriddetector/files/bench_holdout/`

List file (device):
`/sdcard/Android/data/com.landfall.hybriddetector/files/bench_list.txt`
Contains **relative paths** like `bench_holdout/benign_dng/<hash>.dng` for scoped storage safety.

Note: We set ownership so the app can read all files:
```
adb shell su 0 chown -R 10335:10335 /sdcard/Android/data/com.landfall.hybriddetector/files/bench_holdout
```

Results (from `outputs/device_log_holdout.txt`):

| Class | Total | Hybrid flagged | Hybrid Recall/FPR | AE flagged (Thr=30) | AE Recall/FPR |
| --- | --- | --- | --- | --- | --- |
| LandFall | 6 | 6 | 1.00 | 6 | 1.00 |
| general_mal | 50 | 49 | 0.98 | 33 | 0.66 |
| benign DNG | 100 | 0 | 0.00 | 0 | 0.00 |

Performance (avg per file):
- Hybrid: extract 4.252 ms, infer 0.039 ms
- AE: extract 4.032 ms, infer 0.063 ms

---

## 6) Artifacts (latest)

Training/holdout:
- `outputs/train_list.txt`
- `outputs/bench_holdout_manifest.json`
- `outputs/bench_holdout_list.txt`

Models:
- `outputs/hybrid_model.npz`
- `outputs/hybrid_model.tflite`
- `outputs/ae_model_p99_holdout.keras`
- `outputs/anomaly_ae.tflite`
- `android/HybridDetectorApp/app/src/main/assets/*` (deployed)

Device logs:
- `outputs/device_log_holdout.txt`

---

## 7) Notes and caveats
- This holdout bench is intentionally **DNG-only for benign**, so FPR numbers here
  reflect DNG-only performance and should not be generalized to classic TIFFs.
- AE threshold 30 is **manual** and chosen to control DNG false positives; it reduces
  general_mal recall relative to the p99 threshold from training.
