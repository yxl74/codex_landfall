# Isolation Forest / One-Class SVM Pipeline (Standalone)

This document describes the standalone anomaly baselines built for comparison
against AE and Hybrid. These models are **benign-only** and do not use the
supervised rules or logits.

---

## 1) Feature sets (design choices)

We intentionally avoid very high-dimensional bytes for OCSVM/Isolation Forest:

| Feature set | Dim (approx) | Contents | When to use |
| --- | --- | --- | --- |
| `struct` | 32 | Structural TIFF/DNG + entropy only | Most stable, fastest |
| `struct_bytes` | 47 | Structural + entropy + byte summary stats | Recommended default |
| `full_hist` | 546 | Structural + full 2x257 byte hist | High dimensional, heavier |

Byte summary stats (for `struct_bytes`):
- Head/tail 1KB entropy
- Zero/0xFF ratio
- ASCII ratio
- Whitespace ratio
- Padding ratio
- Unique byte ratio
- Entropy gap (head vs tail)

Rationale:
- OCSVM is sensitive to high dimensionality; `struct_bytes` provides signal
  without blowing up features.
- Isolation Forest can handle more dimensions but still benefits from a compact,
  well-behaved feature space.

---

## 2) Training protocol

All models:
1. Load features from `outputs/anomaly_features.npz` built from **train list** only.
2. Log1p-transform heavy-tailed structural fields.
3. Standardize using **benign train** mean/std.
4. Train on **benign only**.
5. Pick threshold using benign validation percentile (p99 default).

---

## 3) Commands

Feature extraction (train list):
```
.venv-tf/bin/python analysis/anomaly_feature_extract.py \
  --data-root data \
  --list-file outputs/train_list.txt \
  --output-npz outputs/anomaly_features.npz \
  --output-csv outputs/anomaly_features.csv
```

Isolation Forest:
```
python3 analysis/if_svm_train_eval.py \
  --input-npz outputs/anomaly_features.npz \
  --model iforest \
  --feature-set struct_bytes \
  --threshold-percentile 99 \
  --output-json outputs/iforest_metrics.json \
  --output-model outputs/iforest_model.pkl
```

One-Class SVM:
```
python3 analysis/if_svm_train_eval.py \
  --input-npz outputs/anomaly_features.npz \
  --model ocsvm \
  --feature-set struct_bytes \
  --threshold-percentile 99 \
  --nu 0.01 \
  --gamma scale \
  --output-json outputs/ocsvm_metrics.json \
  --output-model outputs/ocsvm_model.pkl
```

Optional: evaluate on holdout NPZ (if you build one from `outputs/bench_holdout_list.txt`):
```
python3 analysis/if_svm_train_eval.py \
  --input-npz outputs/anomaly_features.npz \
  --model iforest \
  --feature-set struct_bytes \
  --threshold-percentile 99 \
  --eval-npz outputs/anomaly_features_holdout.npz \
  --output-json outputs/iforest_metrics.json
```

---

## 4) Outputs

Each run produces a JSON with:
- counts (train/val/test)
- val/test metrics
- per-class rates
- threshold used

Example outputs:
- `outputs/iforest_metrics.json`
- `outputs/ocsvm_metrics.json`

---

## 5) Dependencies

Requires scikit-learn:
```
pip install scikit-learn
```
