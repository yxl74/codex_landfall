# Landscape Survey & Feature Gap Analysis for LandFall Detector

## 1. Introduction

LandFall Detector is a defensive security system for identifying malicious TIFF/DNG files on Android devices. It combines deterministic CVE-pattern rules with an ensemble of machine learning models (supervised logistic regression, unsupervised autoencoders, and a GRU sequence model) to detect both known exploit signatures and novel structural anomalies in image files.

This document surveys the broader ecosystem of file-based exploit detection tools, situates LandFall within that landscape, and identifies concrete techniques and capabilities from comparable projects that could strengthen the system.

**Scope.** The survey covers defensive tools that detect exploitation attempts embedded in files — particularly image and document formats — before or after they reach a target device. It excludes general-purpose antivirus engines (beyond ClamAV's media heuristics), network-layer intrusion detection, and runtime exploit mitigation (ASLR, sandboxing, etc.).

---

## 2. Landscape Taxonomy

### 2.1 Projects Surveyed

#### Tier 1: File-Based Mobile Exploit Detection

| Project | Author | Language | Approach | File Formats | Deployment |
|---------|--------|----------|----------|-------------|------------|
| **ELEGANTBOUNCER** | Matt Suiche | Rust | Structural impossibility detection (no ML) | PDF, WebP, TTF/OTF, TIFF/DNG | CLI desktop |
| **LandFall Detector** | (this project) | Python + Kotlin | CVE rules + ML ensemble (LR, AE, GRU-AE) | TIFF/DNG | Android on-device |

#### Tier 2: Mobile Spyware Forensics (Post-Compromise Detection)

| Project | Author | Approach | Scope |
|---------|--------|----------|-------|
| **MVT** (Mobile Verification Toolkit) | Amnesty International | IOC matching on device backups/logs | iOS & Android forensics |
| **iVerify Mobile Threat Hunting** | iVerify (ex-Trail of Bits) | Signatures + heuristics + ML on sysdiagnose | iOS & Android runtime |
| **PiRogue Tool Suite** | PTS Project / OTF | Network DPI + MVT integration + Suricata | Network + device forensics |
| **Citizen Lab malware-indicators** | Citizen Lab | STIX IOC sharing | IOC repository |
| **iMazing Spyware Analyzer** | DigiDNA | GUI wrapper around MVT | iOS via desktop |

#### Tier 3: File-Type & Format Analysis

| Project | Author | Approach | Relevance |
|---------|--------|----------|-----------|
| **Google Magika** | Google | Deep learning file-type classifier (1 MB model, 200+ types) | Already integrated in LandFall as supplementary signal |
| **PolyFile** | Trail of Bits | Polyglot-aware semantic file parser | Polyglot detection methodology |
| **PolyConv** | Koch et al. (WWW 2025) | ML polyglot detector (F1 = 99.2%) | ML approach to polyglot files |
| **Binwalk** | ReFirmLabs | Entropy + signature-based embedded file extraction | File structure analysis patterns |
| **Sherloq** | GuidoBartoli | Image forensics toolset | Image format analysis techniques |

#### Tier 4: Signature/Rule-Based Scanning

| Project | Approach | Relevance |
|---------|----------|-----------|
| **ClamAV** | Signature DB + `Heuristics.Broken.Media` for TIFF/JPEG/PNG/GIF | Broken media heuristics, image fuzzy hashing |
| **YARA** + rule repos (Neo23x0, Yara-Rules, ReversingLabs) | Pattern matching rules | CVE-specific rule templates |
| **YARAify** (abuse.ch) | Online YARA scanning platform | Rule aggregation model |

#### Tier 5: ML-Based Malicious Document Detection (Academic)

| Work | Venue | Approach | Relevance |
|------|-------|----------|-----------|
| **VAPD** | USENIX Security 2025 | VAE anomaly detection for malicious PDFs | AE-based anomaly detection on file structure |
| **uitPDF-MalDe** | Eng. App. of AI 2025 | Multi-model ML PDF classifier | Ensemble approach comparison |
| **Temporal PDF Classification** | Applied Soft Computing 2025 | Temporal drift-aware malicious PDF detection | Concept drift handling |
| **Hidost** | EURASIP 2016 | Static ML on file logical structure (PDF + SWF) | Multi-format structural ML |

### 2.2 Detection Approach Categories

**Structural/Deterministic** — Rules that identify conditions impossible in legitimate files.

- ELEGANTBOUNCER: Checks for structurally impossible values in PDF JBIG2 streams, WebP Huffman tables, and TIFF tile configurations.
- ClamAV `Heuristics.Broken.Media`: Validates internal consistency of TIFF, JPEG, PNG, and GIF headers.
- YARA rules: Byte-pattern matching for known exploit payloads.
- LandFall Tier 1: CVE-specific deterministic rules for opcode overflow (CVE-2025-21043), JPEG SOF3 mismatch (CVE-2025-43300), and tile configuration anomalies.

**ML/Statistical** — Learned models that generalize from training data.

- LandFall Tier 2: Supervised LR on byte histogram + structural features; unsupervised AE on structural + entropy features; GRU-AE on DNG tag sequences.
- VAPD: Variational autoencoder on PDF structural features.
- iVerify: Undisclosed ML models on iOS sysdiagnose telemetry.
- Magika: Deep learning file-type classifier (integrated in LandFall).
- PolyConv: CNN-based polyglot file detector.

**IOC/Forensic** — Post-compromise indicator matching.

- MVT: Matches file hashes, domain names, and process names against known Pegasus/Predator IOCs.
- PiRogue Tool Suite: Network traffic analysis + MVT integration.
- Citizen Lab indicators: STIX-formatted IOC repository for targeted surveillance malware.

### 2.3 Deployment Target Categories

**On-device mobile:**
- LandFall Detector — Android app, all inference on-device via TFLite (~6 MB total model size).
- iVerify — iOS & Android app, combines on-device heuristics with cloud analysis.

**Desktop CLI:**
- ELEGANTBOUNCER — Rust CLI, scans individual files or iOS backup directories.
- MVT — Python CLI, processes iOS/Android backups and logs.
- Binwalk — Python CLI, firmware and embedded file extraction.

**Server/Cloud:**
- ClamAV — Daemon mode for mail/proxy scanning; also CLI.
- YARAify — Cloud-hosted YARA scanning service.
- Magika — Used at Google scale internally; also available as Python library.

### 2.4 File Format Coverage

**Image-focused:**
- LandFall: TIFF/DNG only (deep structural parsing with IFD traversal, opcode analysis, tag sequences).
- ELEGANTBOUNCER: TIFF/DNG + WebP + GIF (structural impossibility checks across multiple image exploit families).

**Document-focused:**
- VAPD, uitPDF-MalDe, Hidost: PDF-specific detection.
- ELEGANTBOUNCER: PDF (JBIG2 stream analysis for FORCEDENTRY detection).

**Multi-format:**
- ClamAV: Broad format coverage including TIFF, JPEG, PNG, GIF, PDF, Office documents.
- YARA: Format-agnostic byte pattern matching.
- Magika: 200+ file types classified (identification, not exploit detection).
- PolyFile: Format-aware parsing for polyglot detection across dozens of formats.

---

## 3. Deep Comparison with ELEGANTBOUNCER

ELEGANTBOUNCER is the most directly comparable project to LandFall. Both analyze file structure to detect mobile exploits in image formats, and both cover TIFF/DNG. The projects differ fundamentally in detection philosophy.

### 3.1 Detection Philosophy

| Aspect | ELEGANTBOUNCER | LandFall Detector |
|--------|---------------|-------------------|
| **Core approach** | Structural impossibility — conditions that cannot occur in legitimate files | CVE rules + ML anomaly detection — pattern matching and learned deviation from benign distribution |
| **ML dependency** | None | Central (4-model ensemble) |
| **Training data** | Not required | 5,818 training files (5,773 benign + 45 malware) |
| **False positive model** | Deterministic: either structurally impossible or not | Statistical: score thresholds with configurable sensitivity |
| **Novel exploit detection** | Only if the exploit creates a structural impossibility that overlaps with existing checks | Unsupervised models (AE, GRU-AE) can flag novel anomalies without rule updates |
| **Confidence output** | Binary (flagged or not) | Continuous scores (hybrid probability, reconstruction error) with interpretable feature attribution |

### 3.2 Exploit Family Coverage

| Exploit Family | CVEs | ELEGANTBOUNCER | LandFall |
|---------------|------|----------------|----------|
| **FORCEDENTRY** (NSO, iMessage) | CVE-2021-30860 | Yes — JBIG2 page segment analysis in PDF | No — requires PDF parser |
| **BLASTPASS** (NSO, iMessage) | CVE-2023-41064 | Yes — WebP VP8L Huffman table validation | No — requires WebP parser |
| **TRIANGULATION** (Kaspersky) | CVE-2023-41990 | Yes — TrueType font instruction validation | No — requires TTF/OTF parser |
| **LandFall opcode overflow** | CVE-2025-21043 | Not specifically, but may detect via structural checks | Yes — dedicated CVE rule + ML detection |
| **DNG JPEG SOF3 mismatch** | CVE-2025-43300 | Yes — TIFF structural analysis | Yes — dedicated CVE rule |
| **TIFF tile anomalies** | Generic | Partially — tile configuration checks | Yes — three tile-related rules (3a/3b/3c) |

### 3.3 Strengths Comparison

**ELEGANTBOUNCER advantages:**

- **No training data required.** Deterministic rules derived from format specifications mean no data collection, labeling, or model drift concerns.
- **Multi-format coverage.** Handles PDF (FORCEDENTRY), WebP (BLASTPASS), TTF/OTF (TRIANGULATION), and TIFF/DNG — four distinct exploit families.
- **Zero false positive rate by design.** Structural impossibilities cannot occur in legitimate files (assuming correct implementation).
- **iOS backup scanning.** Can scan files extracted from iOS backups, including iMessage attachment databases.
- **Messaging app integration.** Scans attachment databases from iMessage, WhatsApp, Signal, and Telegram.
- **Rust implementation.** Memory-safe, fast, single-binary deployment.

**LandFall advantages:**

- **Novel anomaly detection.** Unsupervised autoencoders (AE and GRU-AE) can flag structurally anomalous files that don't match any known CVE pattern, providing coverage against zero-day exploits that create unusual but not strictly impossible structures.
- **On-device Android deployment.** Runs as an Android app with TFLite inference, enabling real-time scanning on the target platform.
- **Quantitative confidence scores.** Hybrid LR outputs calibrated probabilities; AE outputs reconstruction error magnitudes. These enable tunable sensitivity (the Android threshold of 0.20 is more aggressive than the training threshold of 0.50).
- **Interpretable attribution.** The `AttributionModel` decomposes hybrid LR predictions into per-feature contributions, identifying which structural features drive a detection.
- **Deep DNG analysis.** The GRU-AE tag sequence model analyzes the ordering and properties of all IFD entries (up to 512), capturing structural patterns that point-wise rules miss.
- **Entropy analysis.** Header, tail, and overall Shannon entropy features detect unusual byte distributions that may indicate embedded payloads.
- **Byte-level features.** 514-dimensional byte histograms from file head and tail capture content-level anomalies beyond structural metadata.

### 3.4 Complementary Nature

The two systems address different failure modes:

- ELEGANTBOUNCER catches exploits that create **structurally impossible** file states — these are high-confidence, zero-FP detections but are limited to conditions the developer anticipated.
- LandFall catches exploits that create **statistically unusual** file states — these have configurable sensitivity and can generalize to unseen exploit families, but require training data and accept a nonzero (if empirically zero on the holdout set) FP risk.

A combined system would provide defense in depth: deterministic rules for known impossible states, plus ML models for anomalies that fall outside rule coverage.

---

## 4. Feature Gap Analysis

Each gap is assessed with the following fields:

- **Gap**: What capability is missing.
- **Demonstrated by**: Which project(s) implement it.
- **Priority**: High / Medium / Low, based on impact relative to implementation effort.
- **Notes**: Rationale and considerations.

### 4.1 File Format & Exploit Coverage

| # | Gap | Demonstrated By | Priority | Notes |
|---|-----|----------------|----------|-------|
| 1 | **Multi-format coverage** (PDF, WebP, TTF/OTF exploits) | ELEGANTBOUNCER | Medium | LandFall only handles TIFF/DNG. Adding even one additional format (e.g., WebP, which is common on Android) would significantly broaden coverage. Each format requires a new parser and feature extractor, but the ML pipeline (byte histograms, anomaly AE) could partially transfer. |
| 2 | **FORCEDENTRY-style JBIG2 detection** | ELEGANTBOUNCER | Low | Requires a PDF parser and JBIG2 stream decoder. Different domain from LandFall's image-file focus, and the FORCEDENTRY vulnerability (CVE-2021-30860) is patched on current iOS/Android. |
| 3 | **BLASTPASS WebP Huffman table analysis** | ELEGANTBOUNCER | Low | Requires a WebP VP8L parser. The vulnerability (CVE-2023-41064) is iOS-specific and patched since iOS 16.6.1. |
| 4 | **iOS backup scanning** | ELEGANTBOUNCER, MVT | Medium | Scanning TIFF/DNG files extracted from iOS backups would extend LandFall's reach to forensic workflows. Does not require new ML models — just a file discovery front-end that understands backup directory structure. |
| 5 | **Messaging app attachment scanning** | ELEGANTBOUNCER | Medium | Scanning iMessage/WhatsApp/Signal/Telegram attachment databases is the primary delivery vector for targeted image exploits. Requires understanding each app's storage format (SQLite DBs, blob directories). |

### 4.2 Detection Techniques

| # | Gap | Demonstrated By | Priority | Notes |
|---|-----|----------------|----------|-------|
| 6 | **Polyglot file detection (expanded)** | PolyFile, PolyConv | High | LandFall has `flag_zip_polyglot` (ZIP-in-TIFF detection) but this covers only one polyglot type. PolyFile demonstrates detection of TIFF-PDF, TIFF-ELF, and other polyglot combinations. Expanding polyglot detection would strengthen LandFall's coverage of container-based evasion techniques. The existing Magika integration already provides a secondary file-type signal that could support this. |
| 7 | **Broken media heuristics** | ClamAV | Medium | ClamAV's `Heuristics.Broken.Media` validates internal consistency of TIFF structures beyond CVE-specific patterns — e.g., impossible compression/bit-depth combinations, invalid color space declarations, truncated strip/tile data. LandFall's CVE rules target specific vulnerabilities; broader structural validation would catch malformed files that don't match any known CVE but are nonetheless invalid. |
| 8 | **Image fuzzy hashing** | ClamAV | Low | Content-level similarity detection (e.g., ssdeep, TLSH) across malware families. Useful for clustering exploit variants but limited value for zero-day detection. LandFall's byte histograms already provide a coarse content fingerprint. |
| 9 | **VAE-based anomaly detection** | VAPD (USENIX 2025) | Medium | LandFall uses a standard autoencoder (MSE reconstruction error). A variational autoencoder would provide a structured latent space and probabilistic scoring via the ELBO, potentially improving separation between benign and anomalous files. VAPD demonstrates this approach on PDF structural features with strong results. The change would modify `ae_train_eval.py` and the TFLite export but preserve the existing feature pipeline. |
| 10 | **Temporal/concept drift handling** | Temporal PDF Classification (2025) | Low | As new camera models and DNG software produce novel but legitimate structural patterns, LandFall's anomaly models may produce false positives on unseen benign distributions. Drift detection (monitoring score distributions over time) and periodic retraining could address this. Low priority because LandFall's holdout FPR is currently 0% with a wide decision margin. |

### 4.3 Integration & Interoperability

| # | Gap | Demonstrated By | Priority | Notes |
|---|-----|----------------|----------|-------|
| 11 | **YARA rule export** | YARA ecosystem | Medium | Exporting LandFall's CVE rules as YARA rules would enable integration with existing security toolchains (ClamAV, VirusTotal, YARA scanners). The three CVE rules are deterministic pattern checks that translate directly to YARA syntax. This would increase the project's practical reach without changing the detection engine. |
| 12 | **STIX/IOC export** | MVT, Citizen Lab | Medium | Exporting detection results as STIX indicators (file hashes, structural properties of flagged files) would enable sharing with SOC platforms, threat intelligence feeds, and tools like MVT. Relevant for organizations performing forensic analysis of targeted attacks. |
| 13 | **Batch/directory scanning mode** | ELEGANTBOUNCER, ClamAV | Low | LandFall has a benchmark mode driven by file lists. Recursive directory scanning with output formats (JSON, CSV) would be useful for desktop/server deployment scenarios. The Android app's detection pipeline is already modular enough to support this via a Python CLI wrapper. |

### 4.4 Deployment & Platform

| # | Gap | Demonstrated By | Priority | Notes |
|---|-----|----------------|----------|-------|
| 14 | **Desktop/server deployment** | ELEGANTBOUNCER, ClamAV | Medium | LandFall is Android-only. A Python CLI that reuses the analysis scripts (feature extraction + model inference) would enable desktop scanning. A Rust port would match ELEGANTBOUNCER's performance profile. The Python path is lower effort since the training pipeline already runs on desktop. |

### 4.5 Explainability & Debugging

| # | Gap | Demonstrated By | Priority | Notes |
|---|-----|----------------|----------|-------|
| 15 | **Explainability via SHAP/attribution** | PDFMal XAI (2024) | Low | LandFall already has an `AttributionModel` that decomposes hybrid LR predictions into per-feature contributions using model weights. SHAP would provide model-agnostic feature importance that works across all models (including the AE and GRU-AE). Low priority because the existing attribution covers the primary classifier and the autoencoder scores are inherently interpretable (high reconstruction error = unusual structure). |
| 16 | **Entropy-based file region analysis** | Binwalk | Low | LandFall computes header, tail, and overall Shannon entropy as features. Binwalk-style entropy visualization (byte-by-byte entropy across the file) would aid debugging of detections but is an analysis tool, not a detection improvement. Could be added to a desktop CLI variant. |

---

## 5. Recommendations

### 5.1 Prioritized Gaps

The gaps are ranked by the product of (detection impact) x (feasibility) x (alignment with LandFall's research goals).

**High priority — strengthen core detection:**

1. **Expanded polyglot detection** (Gap #6). LandFall already detects ZIP-in-TIFF polyglots and integrates Magika for file-type classification. Extending this to detect additional polyglot combinations (TIFF-PDF, TIFF-ELF, embedded executables) is high-value because polyglot files are a documented evasion technique and the infrastructure is partially in place. Implementation: add format-specific magic byte checks beyond ZIP EOCD, and consider integrating PolyFile's parsing grammar for comprehensive polyglot identification.

2. **Broken media heuristics** (Gap #7). Adding structural validation rules beyond CVE-specific patterns would catch malformed TIFF files that are invalid per the specification but don't match any known exploit. This complements the ML anomaly detection with deterministic structural checks. Implementation: validate compression-type/bits-per-sample consistency, strip/tile data completeness, and IFD pointer validity.

**Medium priority — broaden reach and integration:**

3. **YARA rule export** (Gap #11). Low implementation cost, high integration value. The three CVE rules are straightforward to express in YARA syntax. A YARA rule file would let LandFall's detection patterns be consumed by any YARA-compatible scanner.

4. **VAE anomaly detection** (Gap #9). Replacing the standard AE with a VAE could improve latent space structure and provide probabilistic anomaly scores. Worth experimenting with since the feature pipeline is unchanged — only the model architecture and training loop need modification.

5. **Desktop CLI deployment** (Gap #14). A Python CLI wrapping the existing analysis scripts would enable desktop/server scanning with minimal new code. The feature extractors and model inference code already run on desktop Python.

6. **STIX/IOC export** (Gap #12). Useful for forensic workflows where LandFall detections need to feed into broader threat intelligence pipelines.

**Lower priority — future extensions:**

7. **iOS backup scanning** (Gap #4). Extends LandFall's applicability to forensic analysis scenarios without requiring new ML models.
8. **Multi-format coverage** (Gap #1). High impact but high effort — each new format requires a parser, feature extractor, and potentially new training data.
9. **Image fuzzy hashing** (Gap #8), **Concept drift** (Gap #10), **SHAP** (Gap #15), **Entropy visualization** (Gap #16). These are refinements rather than capability gaps. They would improve specific aspects of the system but don't address fundamental coverage limitations.

### 5.2 Short-Term vs. Long-Term

**Short-term** (incremental improvements to the current system):
- Export CVE rules as YARA rules.
- Expand polyglot detection beyond ZIP-in-TIFF.
- Add structural validation heuristics (broken media checks) to the CVE rule engine.
- Experiment with VAE architecture in the anomaly detection pipeline.

**Long-term** (architectural extensions):
- Desktop/server CLI deployment using existing Python pipeline.
- Multi-format parser support (starting with WebP, the most relevant Android image format).
- STIX/IOC export for threat intelligence integration.
- iOS backup scanning for forensic workflows.

### 5.3 Research vs. Practical Impact

For strengthening the **research contribution**, the most impactful gaps are:
- **VAE anomaly detection** (#9) — directly comparable to VAPD (USENIX 2025) and would position the AE component within the current academic literature.
- **Expanded polyglot detection** (#6) — connects to PolyConv (WWW 2025) and PolyFile research on file format abuse.
- **Concept drift handling** (#10) — addresses a known limitation of static ML models in evolving file format ecosystems.

For strengthening **practical utility**, the most impactful gaps are:
- **YARA rule export** (#11) — immediate integration with existing security infrastructure.
- **Broken media heuristics** (#7) — catches a broader class of malformed files.
- **Desktop CLI deployment** (#14) — makes the tool accessible beyond Android.

---

## 6. References

### Tools & Projects

| Project | URL |
|---------|-----|
| ELEGANTBOUNCER | https://github.com/comaeio/ELEGANTBOUNCER |
| MVT (Mobile Verification Toolkit) | https://github.com/mvt-project/mvt |
| iVerify | https://iverify.io |
| PiRogue Tool Suite | https://pts-project.org |
| Citizen Lab malware-indicators | https://github.com/citizenlab/malware-indicators |
| iMazing Spyware Analyzer | https://imazing.com/spyware-analyzer |
| Google Magika | https://github.com/google/magika |
| PolyFile | https://github.com/trailofbits/polyfile |
| Binwalk | https://github.com/ReFirmLabs/binwalk |
| Sherloq | https://github.com/GuidoBartoli/sherloq |
| ClamAV | https://github.com/Cisco-Talos/clamav |
| YARA | https://github.com/VirusTotal/yara |
| YARAify | https://yaraify.abuse.ch |

### Academic Works

| Work | Reference |
|------|-----------|
| VAPD | Chen et al., "VAPD: Anomaly Detection for Malicious PDF via VAE," USENIX Security 2025 |
| uitPDF-MalDe | "Multi-model ML framework for PDF malware detection," Engineering Applications of AI, 2025 |
| Temporal PDF Classification | "Temporal drift-aware malicious PDF detection," Applied Soft Computing, 2025 |
| Hidost | Srndic & Laskov, "Hidost: a static machine-learning-based detector of malicious files," EURASIP J. on Information Security, 2016 |
| PolyConv | Koch et al., "PolyConv: ML-based polyglot file detection," WWW 2025 |

### CVEs Referenced

| CVE | Description | Detected By |
|-----|-------------|-------------|
| CVE-2021-30860 | FORCEDENTRY — CoreGraphics JBIG2 integer overflow (iOS) | ELEGANTBOUNCER |
| CVE-2023-41064 | BLASTPASS — ImageIO WebP heap buffer overflow (iOS) | ELEGANTBOUNCER |
| CVE-2023-41990 | TRIANGULATION — FontParser TrueType instruction exploitation (iOS) | ELEGANTBOUNCER |
| CVE-2025-21043 | DNG opcode list count overflow (Android) | LandFall |
| CVE-2025-43300 | DNG JPEG SOF3 component count mismatch (Android) | LandFall, ELEGANTBOUNCER |
