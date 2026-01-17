# Hybrid TIFF/DNG Detection — End-to-End Experiment Report

## Executive summary
This experiment validates a hybrid detector that combines TIFF/DNG structural parsing with Magika-style byte features and light rules. The trained logistic regression model achieves strong offline detection (ROC-AUC ~0.998 on test) and matches on-device TFLite inference with extremely tight parity (max absolute diff ~5e-5 over 206 files). On-device performance is lightweight (~11.2 ms feature extraction and ~0.06 ms inference per file on the connected device).

## ML basics (quick primer)
This section explains the evaluation terms used in the report in plain language.

- **Binary classification**: the model outputs a score between 0 and 1. Higher means “more likely malicious.”
- **Threshold (Thr)**: the cutoff used to convert the score into a label. Example: Thr=0.20 means scores >= 0.20 are labeled “malicious.”
- **Confusion matrix terms**:
  - **TP (true positives)**: malicious files correctly flagged.
  - **TN (true negatives)**: benign files correctly ignored.
  - **FP (false positives)**: benign files incorrectly flagged (your current pain point).
  - **FN (false negatives)**: malicious files missed.
- **Accuracy**: (TP + TN) / total. Can look good even if the model misses rare malicious files.
- **Precision**: TP / (TP + FP). “When the model says malicious, how often is it right?”
- **Recall**: TP / (TP + FN). “How many of the actual malicious files did we catch?”
- **F1**: harmonic mean of precision and recall. Useful when the dataset is imbalanced.
- **ROC-AUC**: threshold‑independent score; 1.0 is perfect, 0.5 is random. It measures how well the model ranks malicious higher than benign, not where the threshold is.

## Why we use a threshold
The model’s raw score is continuous. We choose a threshold to convert it into a decision:
- **Lower threshold**: higher recall, more false positives.
- **Higher threshold**: higher precision, more false negatives.
In this experiment, the threshold is chosen on the validation set to maximize F1 (balanced precision/recall).

## Dataset and splits
- Total samples: 731
- Breakdown: 630 benign, 100 general malware, 6 Landfall samples
- Deduplication: file-hash based before splitting
- Split: 70% train / 15% val / 15% test (random, seed=42)
- Train counts (from metrics): 511 total (71 malicious, 440 benign)
- Validation counts (derived from confusion): 109 total (17 malicious, 92 benign)
- Test counts (derived from confusion): 111 total (13 malicious, 98 benign)

## Feature pipeline (training + inference)
1. **Bytes features (Magika-style)**
   - Read first/last 4096 bytes.
   - Trim leading/trailing ASCII whitespace.
   - Convert to 1024-byte histograms (257 bins including padding token 256).
   - Why it helps: many file formats and file families have characteristic byte distributions near the start/end of files (headers, metadata blocks, trailers). Malicious samples can carry anomalous or non‑standard patterns even if they are valid TIFF/DNG.
2. **TIFF/DNG structural features**
   - Endian + TIFF magic, IFD size (max entries), min width/height.
   - SubIFD counts, NewSubfileType unique count.
   - DNG opcode list sizes, opcode IDs, unknown opcode counts.
   - ZIP polyglot signals near end of file.
   - Why it helps: Landfall targets DNG opcode processing. Unusual opcode counts or unknown opcodes are strong signals. Structural inconsistencies (e.g., tiny dimensions with small IFDs) can indicate crafted files rather than real camera output.
3. **Rule flags**
   - DNG opcode anomalies, tiny non-DNG dimensions + small IFD, ZIP polyglot.
   - DNG pretending to be JPEG based on extension vs magic.
   - Why it helps: rules give high‑precision “red flags” for known exploit techniques and file-type mismatches.
4. **Model**
   - Logistic regression with standardization.
   - Log1p on heavy-tail structural fields.
   - Threshold chosen by max F1 on validation (0.20).
   - Why it helps: logistic regression is simple, interpretable, and robust with limited data. Standardization keeps features on comparable scales. Log1p reduces the impact of extremely large counts (opcode sizes, offsets) that otherwise dominate training.
5. **TFLite**
   - Same log1p + normalization baked into the graph.
   - Device pipeline mirrors feature extraction exactly.

### Feature details (what each group captures)
- **Header/tail byte histograms**: coarse “fingerprint” of format and encoder behavior. This is especially useful when files share structure but differ in metadata/compression patterns.
- **IFD size and dimensions**: real camera files tend to have consistent IFD sizes and plausible dimensions. Crafted files often have anomalously small dimensions or minimal metadata.
- **DNG opcode lists**: core to Landfall; these are strong signals for malicious DNGs. Unknown opcode IDs are particularly suspicious.
- **ZIP polyglot flags**: compressed data appended to a TIFF can indicate a polyglot or smuggling attempt.
- **Rules**: fast, explainable triggers that complement ML. They reduce the risk of missing known exploit patterns.

## Offline evaluation
Model: `outputs/hybrid_model.npz` (bytes_mode=hist), threshold=0.20  
Metrics from `outputs/hybrid_metrics.json`:

| Split | Acc | Prec | Recall | F1 | ROC-AUC | TP | TN | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Val | 0.982 | 1.000 | 0.882 | 0.938 | - | 15 | 92 | 0 | 2 |
| Test | 0.973 | 0.812 | 1.000 | 0.897 | 0.998 | 13 | 95 | 3 | 0 |

### Ablation findings (from `outputs/ablation_results.md`)
| Model | Acc | Prec | Recall | F1 | ROC-AUC | Thr |
| --- | --- | --- | --- | --- | --- | --- |
| rules_flag_any | 0.991 | 1.000 | 0.923 | 0.960 | 0.000 | 0.50 |
| struct_only | 0.883 | 0.500 | 1.000 | 0.667 | 0.990 | 0.10 |
| bytes_only_hist | 0.973 | 0.812 | 1.000 | 0.897 | 0.997 | 0.40 |
| bytes_only_raw | 0.982 | 0.923 | 0.923 | 0.923 | 0.976 | 0.55 |
| hybrid_hist | 0.973 | 0.812 | 1.000 | 0.897 | 0.998 | 0.20 |
| hybrid_raw | 0.982 | 0.923 | 0.923 | 0.923 | 0.958 | 0.50 |

**Key observations:**
- Byte features carry most of the signal for this dataset; hybrid_hist matches bytes_only_hist.
- Structural-only features yield high recall but lower precision (more false positives).
- Rules alone have high precision but lower recall than the hybrid model.

## On-device evaluation
**Bench set:** 206 files (100 benign + 106 malicious)  
**Device path:** `/sdcard/Android/data/com.landfall.hybriddetector/files/bench_full/`  
**Average runtime (from `outputs/device_log.txt`):**
- Feature extraction: 11.211 ms / file
- Inference: 0.062 ms / file

## Device vs local TFLite parity
Comparison uses the same feature extractor logic on local files and the same TFLite model used on-device.
- Files compared: 206
- Max abs diff: 4.992e-05
- Mean abs diff: 1.421e-05
- Diff report: `outputs/device_local_compare.csv`

This confirms local vs device inference parity within ~5e-5 across the entire bench set.

## Findings and interpretation
- The model hits perfect recall on the test set at the chosen threshold but produces a small number of false positives (3) on the test split.
- Structural-only features are helpful but insufficient alone; byte distribution carries much of the discriminative power for current data.
- Rules are a strong precision guardrail and can be used as an immediate “high confidence” malicious indicator.
- On-device inference is fast enough for batch scanning on modern devices.

## How to interpret “false positives”
False positives happen when a benign file’s features resemble malicious samples. In this system that often means:
- The byte histogram looks similar to the malicious set.
- The file is structurally valid, but the model learned a pattern that overlaps with certain benign TIFF encodings.
This is a signal to **expand benign diversity** and **add more structure-aware features** to reduce reliance on byte distribution alone.

## Limitations
- Landfall samples are few (n=6), so generalization to new Landfall variants is not fully validated.
- Train/val/test split is random (not stratified by family or source), which can overstate generalization.
- The Magika mismatch rule is not used in on-device inference (flag is 0); it can be enabled later if a trusted file-type classifier is embedded.

## Recommendations
- Collect additional Landfall samples and rotate the split by campaign/source for more robust evaluation.
- Expand benign TIFF/DNG/RAW coverage to reduce false positives.
- Add a second validation set from “in-the-wild” non-malicious RAW/TIFF images (phone cameras, lab instruments, etc.).
- Consider threshold tuning for operational preference (e.g., higher precision if false positives are costly).
