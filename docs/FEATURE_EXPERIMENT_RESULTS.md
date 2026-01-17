# Feature Engineering Experiment Results

**Date:** 2026-01-17
**Objective:** Evaluate current ML features and test combinations for improved detection

---

## 1. Dataset Summary

| Class | Samples | Parseable | Notes |
|-------|---------|-----------|-------|
| LANDFALL | 6 | 6 (100%) | Targeted DNG malware |
| general_mal | 1 | 1 (100%) | **Duplicate of LANDFALL sample** |
| Benign | 629 | 455 (72%) | Training corpus |

**Note:** The general_mal sample (`b06dec10...`) is identical to a LANDFALL sample. Effectively, we have 6 unique malware samples.

---

## 2. Individual Feature Discrimination

Ranked by separation ratio: `|mean_mal - mean_benign| / (std_mal + std_benign)`

| Rank | Feature | Separation | Mal Mean | Ben Mean | Direction |
|------|---------|------------|----------|----------|-----------|
| 1 | strip_entropy_variance | **4.92** | 2.38 | 0.45 | MAL > BEN |
| 2 | overlay_entropy | **2.74** | 8.00 | 1.21 | MAL > BEN |
| 3 | header_entropy | **2.05** | 1.83 | 4.81 | MAL < BEN |
| 4 | payload_capacity_ratio | **2.01** | 0.00 | 0.79 | MAL < BEN |
| 5 | bytes_per_pixel | **1.89** | 40,576 | 62 | MAL > BEN |
| 6 | entropy_gradient | 1.82 | 6.04 | 0.86 | MAL > BEN |
| 7 | strip_entropy_mean | 1.14 | 5.39 | 6.86 | MAL < BEN |
| 8 | total_pixels | 1.00 | 206 | 17M | MAL < BEN |
| 9 | pixels_per_mb | 0.74 | 32 | 824K | MAL < BEN |
| 10 | file_size | 0.62 | 6.7M | 33.7M | MAL < BEN |
| 11 | overlay_ratio | 0.38 | 0.06 | 0.02 | MAL > BEN |
| 12 | reference_coverage | 0.38 | 0.94 | 0.98 | MAL < BEN |
| 13 | overall_entropy | 0.14 | 7.30 | 7.13 | MAL > BEN |
| 14 | ifd_count | 0.13 | 3.00 | 3.70 | MAL < BEN |

### Key Insights:
- **strip_entropy_variance** is the best single discriminator (4.92 separation)
- **overlay_entropy** at 8.0 for malware indicates encrypted/compressed payloads
- **header_entropy** is low for malware (1.83 vs 4.81) - sparse/fake headers
- **payload_capacity_ratio** at 0% for malware - no real image data declared

---

## 3. Combined Feature Analysis

Created 4 combined features with improved discrimination:

| Combined Feature | Formula | Separation | Rationale |
|-----------------|---------|------------|-----------|
| **variance × (8-header_ent)** | strip_entropy_variance × (8 - header_entropy) | **6.60** | High entropy variance + sparse header |
| **(1-payload_cap) × overlay_ent** | (1 - payload_capacity_ratio) × overlay_entropy | **5.40** | Missing declared data + encrypted overlay |
| **entropy_cv** | strip_entropy_variance / strip_entropy_mean | **4.65** | Normalized entropy inconsistency |
| **log(file_size/pixels)** | log(file_size) - log(total_pixels) | **3.68** | Log-scale size bloat |

### Best Combined Feature:
`variance × (8-header_ent)` achieves **6.60 separation** - 34% better than the best individual feature!

---

## 4. Model Comparison

Trained Isolation Forest (contamination=5%) with different feature sets:

| Feature Set | Features | Detection | FP Rate | Min Score | Mean Score |
|-------------|----------|-----------|---------|-----------|------------|
| All 14 original | 14 | 7/7 | 5.05% | 0.1225 | 0.1245 |
| Top 5 discriminators | 5 | 7/7 | 5.05% | 0.1761 | 0.1761 |
| 14 + 4 combined | 18 | 7/7 | 5.05% | 0.1905 | 0.1920 |
| Top 5 + 4 combined | 9 | 7/7 | 5.05% | 0.1995 | 0.2009 |
| **4 combined only** | 4 | **7/7** | 5.05% | **0.2232** | **0.2241** |

### Key Finding:
**4 combined features alone outperform 14 original features** in anomaly score magnitude while maintaining 100% detection.

---

## 5. Detection Robustness

Using 4 combined features:

| FP Rate Threshold | Detection | Malware Score Range |
|-------------------|-----------|---------------------|
| 5% | 7/7 (100%) | 0.2232 - 0.2265 |
| 3% | 7/7 (100%) | 0.2232 - 0.2265 |
| **1%** | **7/7 (100%)** | 0.2232 - 0.2265 |

Benign score distribution: Min=-0.25, Max=0.10, Mean=-0.18

**All malware samples score well above the 1% FP threshold (0.069)**

---

## 6. Recommendations

### 6.1 Feature Set Update
Replace current 14 features with optimized set:

**Recommended 9-feature set (Top 5 + 4 combined):**
1. strip_entropy_variance
2. overlay_entropy
3. header_entropy
4. payload_capacity_ratio
5. bytes_per_pixel
6. variance × (8-header_ent) [NEW]
7. (1-payload_cap) × overlay_ent [NEW]
8. entropy_cv [NEW]
9. log(file_size/pixels) [NEW]

### 6.2 Alternative: Minimal Feature Set
For maximum efficiency, use only 4 combined features:
1. variance × (8-header_ent)
2. (1-payload_cap) × overlay_ent
3. entropy_cv
4. log(file_size/pixels)

### 6.3 DNG Validation Features (Future Work)
Based on DNG SDK analysis, additional features to implement:
- `tag_sort_score` - % of IFDs with sorted tags
- `valid_bps_count` - Count of invalid BitsPerSample values
- `critical_dng_tags` - Count of required DNG tags present
- `offset_alignment_score` - % of aligned offsets

---

## 7. Limitations

1. **Small malware dataset**: Only 6 unique LANDFALL samples
2. **No diverse general malware**: general_mal is a LANDFALL duplicate
3. **Single malware family**: All samples are LANDFALL variants
4. **Need validation on other TIFF malware types**

---

## 8. Next Steps

1. ✅ Feature discrimination analysis complete
2. ✅ Combined features identified and tested
3. ⬜ Implement combined features in `ml_features.py`
4. ⬜ Acquire diverse TIFF malware samples for testing
5. ⬜ Implement DNG validation features
6. ⬜ Retrain and validate with expanded feature set
