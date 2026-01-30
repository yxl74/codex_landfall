# MediaThreatDetector

**MediaThreatDetector** is an Android demo app for **on-device media file “zero-day” detection**. It combines:
- **Magika** file type identification (ONNX)
- **Static CVE-style rules** (high-confidence patterns)
- **Benign-only anomaly detection** models for JPEG/TIFF/DNG

Verdict is **3-way**:
- **BENIGN**: normal
- **SUSPICIOUS**: anomaly/unknown/error/type mismatch
- **MALICIOUS**: high-confidence CVE rule match

## Docs

See `docs/architecture.md` for the full architecture + implementation + training pipeline.

## Quickstart

### Android build
```bash
cd android/HybridDetectorApp
JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 ./gradlew assembleDebug
```

Install to a connected device:
```bash
adb install -r android/HybridDetectorApp/app/build/outputs/apk/debug/app-debug.apk
```

### Offline evaluation (local datasets)
Requires TensorFlow venv:
```bash
.venv-tf/bin/python3 analysis/eval_three_way_verdict.py --with-cve --output-json outputs/three_way_verdict_eval.json
```

### Static-rule unit tests
```bash
python3 analysis/test_cve_rules.py
```

## Repository layout

- `android/HybridDetectorApp/` — Android demo app + shipped model assets
- `analysis/` — dataset utilities, feature extraction, training/export scripts
- `data/`, `outputs/`, `JPG_dataset/`, `JPEG_malware/` — local data/artifacts (gitignored)

