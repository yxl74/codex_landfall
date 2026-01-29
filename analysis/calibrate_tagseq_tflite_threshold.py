#!/usr/bin/env python3
"""
Calibrate TagSeq GRU-AE threshold using the exported TFLite model outputs.

Why: threshold should match on-device scoring, not just Keras float math.

Requires TensorFlow (use .venv-tf).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit("TensorFlow is required. Use the .venv-tf environment.") from exc


def find_input_index(in_details, contains: str) -> int:
    for d in in_details:
        if contains in d["name"]:
            return int(d["index"])
    raise KeyError(f"Missing input containing: {contains}")


def score_npz(tflite_path: str, npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    tag_ids = data["tag_ids"].astype(np.int32)
    type_ids = data["type_ids"].astype(np.int32)
    ifd_kinds = data["ifd_kinds"].astype(np.int32)
    lengths = data["lengths"].astype(np.int32)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    idx_features = find_input_index(in_details, "features")
    idx_tag = find_input_index(in_details, "tag_ids")
    idx_type = find_input_index(in_details, "type_ids")
    idx_ifd = find_input_index(in_details, "ifd_kinds")
    out_idx = int(out_details[0]["index"])

    scores = np.zeros((features.shape[0],), dtype=np.float32)
    for i in range(features.shape[0]):
        interpreter.set_tensor(idx_features, features[i : i + 1])
        interpreter.set_tensor(idx_tag, tag_ids[i : i + 1])
        interpreter.set_tensor(idx_type, type_ids[i : i + 1])
        interpreter.set_tensor(idx_ifd, ifd_kinds[i : i + 1])
        interpreter.invoke()
        recon = interpreter.get_tensor(out_idx)[0]
        L = int(lengths[i])
        if L <= 0:
            scores[i] = 0.0
            continue
        diff = features[i, :L, :] - recon[:L, :]
        scores[i] = float(np.sum(diff * diff) / float(L))
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default="outputs/tagseq_gru_ae.tflite")
    ap.add_argument("--holdout-npz", default="outputs/tagseq_dng_holdout.npz")
    ap.add_argument("--landfall-npz", default="")
    ap.add_argument("--threshold-percentile", type=float, default=99.0)
    ap.add_argument("--output-meta", default="outputs/tagseq_gru_ae_meta.json")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--feature-dim", type=int, default=12)
    ap.add_argument("--requires-flex", action="store_true", default=True)
    args = ap.parse_args()

    hold_scores = score_npz(args.tflite, args.holdout_npz)
    thr = float(np.percentile(hold_scores, args.threshold_percentile))
    hold_fpr = float(np.mean(hold_scores >= thr))

    land_recall: Optional[float] = None
    land_mean: Optional[float] = None
    if args.landfall_npz:
        land_scores = score_npz(args.tflite, args.landfall_npz)
        land_recall = float(np.mean(land_scores >= thr))
        land_mean = float(np.mean(land_scores))

    meta = {
        "model": "tagseq_gru_ae",
        "threshold": thr,
        "threshold_percentile": float(args.threshold_percentile),
        "max_seq_len": int(args.max_seq_len),
        "feature_dim": int(args.feature_dim),
        "requires_flex": bool(args.requires_flex),
        "calibration": {
            "holdout_npz": str(args.holdout_npz),
            "holdout_fpr": hold_fpr,
            "holdout_mean": float(np.mean(hold_scores)),
            "holdout_p95": float(np.percentile(hold_scores, 95)),
        },
    }
    if land_recall is not None:
        meta["calibration"]["landfall_npz"] = str(args.landfall_npz)
        meta["calibration"]["landfall_recall"] = land_recall
        meta["calibration"]["landfall_mean"] = land_mean

    Path(args.output_meta).write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.output_meta}")
    print(f"Threshold (p{args.threshold_percentile:.1f}) = {thr:.6f}")
    print(f"Holdout FPR: {hold_fpr*100:.2f}% (n={hold_scores.size})")
    if land_recall is not None:
        print(f"LandFall recall: {land_recall*100:.2f}% (n={land_scores.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

