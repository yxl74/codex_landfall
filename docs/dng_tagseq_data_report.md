# DNG Tag-Sequence Data Report (Feature Evaluation)

This report evaluates the proposed **tag‑sequence feature engineering** for a
DNG‑only anomaly model and summarizes the extracted data statistics.

## Dataset Split

- **Benign DNG pool:** 5,192 files (detected via DNGVersion tag).
- **Train:** 4,692 benign DNG files.
- **Holdout:** 500 benign DNG files.
- **LandFall:** 6 files (DNG).
- **General TIFF malware:** no DNGVersion tag → excluded from DNG‑only pipeline.

Split creation:
- Script: `analysis/build_dng_tagseq_lists.py`
- Seed: `1337`
- Holdout size: `500`

## Extraction

- Script: `analysis/tag_sequence_extract.py`
- Max sequence length: `512` (longer sequences are truncated).

Outputs:
- `outputs/tagseq_dng_train.npz`
- `outputs/tagseq_dng_holdout.npz`
- `outputs/tagseq_dng_landfall.npz`

## Sequence Lengths

| Split | Median tags | P95 tags | P99 tags |
| --- | --- | --- | --- |
| Train (benign) | 108 | 114 | 115 |
| Holdout (benign) | 108 | 114 | 512* |
| LandFall | 512* | 512* | 512* |

*512 indicates truncation at the configured max sequence length.

## Structural Signal Strength (File‑Level Rates)

These are **file‑level** rates: a file is marked if *any* tag triggers the flag.

| Split | Order violation rate | Invalid offset rate | Type mismatch rate |
| --- | --- | --- | --- |
| Train (benign) | 1.07% | 0.75% | 0.32% |
| Holdout (benign) | 1.60% | 1.60% | 0.40% |
| LandFall | 100% | 100% | 100% |

This is exactly what we want for anomaly detection:
- Benign DNGs have **very low** structural violations.
- LandFall files show **universal** violations.

## Key Takeaways (Feature Evaluation)

The proposed tag‑sequence features are **well‑suited** for DNG‑only anomaly detection:

- **Order logic** (tag sort violations) is stable in benign DNGs and highly
  anomalous in LandFall.
- **Offset validity** is a strong signal: benign DNGs are overwhelmingly valid,
  while LandFall samples are consistently invalid.
- **Type mismatch** also separates LandFall from benign DNGs.

Additional features added to the proposal (implemented):
- `is_immediate` (inline vs offset value)
- `log1p(byte_count)` (captures buffer size)
- `type_mismatch` + `type_id_norm` (type constraints)
- `ifd_kind_norm` (main/subifd/exif/gps/interop context)

These are critical for models that must learn **constraints** rather than raw
values.

## Recommended Data to Collect (for robust DNG model)

To improve generalization, collect **benign DNGs** from more sources:

- **Camera diversity:** Canon, Nikon, Sony, Fujifilm, Panasonic, Leica,
  PhaseOne, Hasselblad (native DNG or converted).
- **Adobe DNG Converter outputs** (varied settings).
- **Adobe DNG SDK samples** (official spec coverage).
- **Mobile DNG captures** (Android camera apps producing DNG).

Avoid mixing non‑DNG TIFF/RAW formats in this model. RAW formats such as
NEF/CR2/ARW/PEF are structurally different and should be handled by a separate
pipeline if needed.

## Next Step

Proceed to **train a GRU‑AE** on these tag‑sequence features, then evaluate on
benign holdout + LandFall.
