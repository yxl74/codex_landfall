# Thresholding Plan (On-device Anomaly Models)

This project uses **benign-trained anomaly detectors** (AEs) and therefore the thresholding policy is the main control knob for the false positive rate (FPR).

The goal of this plan is to make thresholds:
- **Domain-robust** (generalize beyond the training sources),
- **Device-aligned** (calibrated on *TFLite output* using the same scoring code path as Android),
- **Auditable** (repeatable split + recorded calibration set).

## 1) Split strategy (avoid leakage)

Random train/val splits can leak distribution (same camera/app/pipeline) into both sets and inflate results.

Recommended: **grouped splits** where groups approximate “source domains”.

### DNG (TagSeq GRU-AE)
- Group key: `raw_pixls/<vendor>` (camera maker folder).
- Split groups into:
  - Train groups (benign training),
  - Calibration groups (benign threshold tuning),
  - Test groups (benign performance reporting).

### TIFF (general TIFF AE, non‑DNG)
- Group key: data source / generator pipeline, e.g.:
  - `ome_tiff`, `nasa_neo` (real-world),
  - `generated_tiff_from_jpeg_*` (PIL pipeline variants),
  - any future added sources (GeoTIFF, scanner, microscopy, etc).
- Keep at least one entire **real-world** group out of training for calibration/test.

### JPEG (JPEG AE)
- Group key: camera/device/app family if available, else directory prefix.
- Keep at least one group containing **phone camera JPEGs** out of training to prevent “all phone JPEGs look anomalous” failures.

## 2) Threshold selection protocol

For each model:
1. **Train** on benign `train` split only.
2. **Export** to TFLite.
3. **Score** the benign `calibration` split using the same logic as Android.
4. Choose a threshold that hits a target FPR on the calibration split.
5. **Report** performance on:
   - benign `test` split (FPR),
   - malicious evaluation set (recall), if available.

### Pick an explicit operating point
Percentile thresholds (p97 / p99) are fine, but they should map to an explicit FPR target:
- “Demo mode”: 1% benign FPR (more sensitive)
- “Strict mode”: 0.1% benign FPR (fewer false alarms)

Consider shipping **two thresholds**:
- `suspicious_threshold` (lower; p99-ish),
- `malicious_threshold` (higher; p99.9-ish),
and render these as separate UI states.

## 3) Device-aligned calibration (important)

Always calibrate thresholds using **TFLite model outputs**:
- Numerical differences (Keras vs TFLite, delegates, optimizations) can shift score distributions.
- This project already does this for the TIFF AE; TagSeq and JPEG should follow the same pattern.

## 4) Optional: tail modeling (EVT) instead of raw percentiles

For tighter control at extreme FPR targets:
- Fit a Generalized Pareto Distribution (GPD) to the upper tail of benign calibration scores.
- Choose a threshold at a desired tail probability (e.g., 1e‑3).

This reduces sensitivity to calibration set size, but requires careful validation.

## 5) What to store in meta JSON

In each model’s meta JSON (assets), store:
- threshold + threshold method (percentile or EVT),
- calibration dataset identifier (hash of file list),
- basic calibration stats (mean/p95/p99).

This makes regressions debuggable when benign drift happens on-device.

