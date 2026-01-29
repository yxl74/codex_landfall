# Stronger Ground Truth (Without CVE Signature Matching)

Signatures (CVE rules) are useful for known-bad, but they do not provide a scalable way to label “zero‑day exploitability”.
For ML evaluation and iterative improvements, the best proxy ground truth is **behavioral**: does the file trigger memory-unsafe behavior in real decoders?

## 1) Crash oracle (recommended)

Create labels by running media files through **instrumented decoders** and treating any memory safety issue as “malicious”.

### What counts as “malicious” for labeling
- ASAN/UBSAN/MSAN finding (heap OOB, stack OOB, UAF, integer overflow, etc.)
- Segfault/abort
- Hang/timeout (e.g., >5s decode / >N MB allocations)

### Practical setup
- Build multiple decoders with sanitizers:
  - TIFF: libtiff, ImageMagick/GraphicsMagick, OpenCV (if applicable)
  - JPEG: libjpeg-turbo, mozjpeg, ImageMagick
  - RAW/DNG: libraw, dcraw-compatible tooling, Adobe DNG SDK (if available)
- Run each file across **multiple implementations**:
  - If any crashes: label malicious
  - If all decode cleanly: label benign (or “unknown” if only weak coverage)

This yields a dataset that is **not signature-based** and is directly tied to “memory exploit surface”.

## 2) Differential testing oracle (secondary)

Even without crashes, inconsistent parser behavior is a strong signal:
- Decoder A accepts; decoder B rejects
- Parsed dimensions differ
- Output image hash differs (for deterministic decoders)

These are “suspicious” labels (not as strong as crash), useful for hard-negative mining and triage.

## 3) Fuzz-generated malicious set (scales best)

Instead of hunting real exploits:
- Run coverage-guided fuzzers against instrumented decoders (OSS-Fuzz style).
- Collect, minimize, and deduplicate crashers.
- Label crashers as malicious.

Benefits:
- Produces large volumes of exploit-relevant samples
- Covers the exact code paths that lead to memory errors

## 4) Dataset hygiene

To make ML evaluation meaningful:
- Dedup by file hash and by crash stacktrace (for fuzz crashers)
- Keep source-heldout test sets (camera/vendor/app/toolchain)
- Separate “corrupt-but-safe” (parse error) vs “memory-unsafe” (crash/ASAN)

## 5) Where ML fits in this loop

Once you have a crash oracle:
- Train anomaly models on broad benign corpora
- Evaluate recall on crashers
- Use “benign but high-score” samples as candidates to expand benign coverage (reduce false positives)

