#!/usr/bin/env python3
"""
Evaluate the current *production* detectors under the 3-way verdict policy:

  - MALICIOUS: CVE rules hit (TIFF/DNG only)
  - SUSPICIOUS: unsupported/type mismatch/errors/anomaly score >= threshold
  - BENIGN: otherwise

This script focuses on quantifying how many files land in each bucket for the
existing local datasets and currently-shipped TFLite models.

Requires TensorFlow (use .venv-tf/bin/python3).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from cve_rule_validation import parse_cve_features, rule_cve_2025_21043, rule_cve_2025_43300, rule_tile_config


ASSETS_DIR = "android/HybridDetectorApp/app/src/main/assets"


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def magika_bytes_to_hist_features(X_bytes: np.ndarray) -> np.ndarray:
    """Convert raw 2048 head+tail bytes to the 514-dim histogram features."""
    beg = X_bytes[:, :1024]
    end = X_bytes[:, 1024:]
    bins = 257
    feats = []
    for block in (beg, end):
        h = np.zeros((block.shape[0], bins), dtype=np.float32)
        for i in range(block.shape[0]):
            counts = np.bincount(block[i].astype(np.int64), minlength=bins)
            h[i] = counts / float(block.shape[1])
        feats.append(h)
    return np.concatenate(feats, axis=1)


def tflite_score_2d(model_bytes: bytes, X: np.ndarray, num_threads: int) -> np.ndarray:
    """Run a TFLite model that maps [N, D] -> [N, 1] scores."""
    if X.size == 0:
        return np.zeros((0,), dtype=np.float32)
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=num_threads)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.resize_tensor_input(input_index, [int(X.shape[0]), int(X.shape[1])])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, X.astype(np.float32))
    interpreter.invoke()
    out = interpreter.get_tensor(output_index)
    return out.reshape(-1).astype(np.float32)


def find_input_index(interpreter: tf.lite.Interpreter, contains: str) -> Optional[int]:
    for d in interpreter.get_input_details():
        name = str(d.get("name", ""))
        if contains in name:
            return int(d["index"])
    return None


def tagseq_scores(
    model_bytes: bytes,
    features: np.ndarray,
    tag_ids: np.ndarray,
    type_ids: np.ndarray,
    ifd_kinds: np.ndarray,
    lengths: np.ndarray,
    num_threads: int,
) -> np.ndarray:
    """Compute TagSeq GRU-AE scores (mean SSE per timestep), matching Android."""
    n, max_len, feature_dim = features.shape
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=num_threads)
    interpreter.allocate_tensors()

    idx_features = find_input_index(interpreter, "features")
    idx_tag_ids = find_input_index(interpreter, "tag_ids")
    idx_type_ids = find_input_index(interpreter, "type_ids")
    idx_ifd_kinds = find_input_index(interpreter, "ifd_kinds")
    if idx_features is None or idx_tag_ids is None or idx_type_ids is None or idx_ifd_kinds is None:
        raise RuntimeError("Could not locate TagSeq input tensors by name.")

    out_detail = interpreter.get_output_details()[0]
    out_idx = int(out_detail["index"])

    scores: List[float] = []
    for i in range(n):
        length = int(lengths[i])
        if length <= 0:
            scores.append(float("nan"))
            continue

        interpreter.set_tensor(idx_features, features[i : i + 1].astype(np.float32, copy=False))
        interpreter.set_tensor(idx_tag_ids, tag_ids[i : i + 1].astype(np.int32, copy=False))
        interpreter.set_tensor(idx_type_ids, type_ids[i : i + 1].astype(np.int32, copy=False))
        interpreter.set_tensor(idx_ifd_kinds, ifd_kinds[i : i + 1].astype(np.int32, copy=False))
        interpreter.invoke()
        out = interpreter.get_tensor(out_idx)[0]

        inp = features[i, :length, :].astype(np.float32, copy=False)
        pred = out[:length, :].astype(np.float32, copy=False)
        diff = inp - pred
        per_timestep = np.sum(diff * diff, axis=1)
        scores.append(float(np.mean(per_timestep)))

    return np.array(scores, dtype=np.float32)


def static_findings(path: str) -> Tuple[List[str], List[str]]:
    feat = parse_cve_features(path)
    if feat is None:
        return [], []

    malicious: List[str] = []
    suspicious: List[str] = []
    if rule_cve_2025_21043(feat):
        malicious.append("CVE-2025-21043")
    if rule_cve_2025_43300(feat):
        malicious.append("CVE-2025-43300")
    if rule_tile_config(feat):
        suspicious.append("TILE-CONFIG")
    return malicious, suspicious


def summarize_three_way(
    labels: np.ndarray,
    verdicts: np.ndarray,
    scores: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for label in sorted(set(labels.astype(str))):
        mask = labels.astype(str) == label
        v = verdicts[mask].astype(str)
        counts = Counter(v)
        total = int(mask.sum())

        row: Dict[str, object] = {
            "total": total,
            "benign": int(counts.get("BENIGN", 0)),
            "suspicious": int(counts.get("SUSPICIOUS", 0)),
            "malicious": int(counts.get("MALICIOUS", 0)),
        }
        if total:
            row["suspicious_rate"] = float(row["suspicious"] / total)
            row["malicious_rate"] = float(row["malicious"] / total)

        if scores is not None and total:
            sc = scores[mask]
            sc = sc[np.isfinite(sc)]
            if sc.size:
                row["score_min"] = float(np.min(sc))
                row["score_mean"] = float(np.mean(sc))
                row["score_p95"] = float(np.percentile(sc, 95))
                row["score_max"] = float(np.max(sc))

        out[label] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", default=ASSETS_DIR)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--with-cve", action="store_true", help="Evaluate TIFF/DNG CVE rules (slower).")
    parser.add_argument("--output-json", default="", help="Optional path to write JSON summary.")
    args = parser.parse_args()

    assets_dir = args.assets_dir
    num_threads = args.num_threads

    # --- Load production meta/models ---
    tiff_meta = load_json(os.path.join(assets_dir, "anomaly_model_meta.json"))
    tiff_thr = float(tiff_meta["threshold"])
    tiff_model = load_bytes(os.path.join(assets_dir, "anomaly_ae.tflite"))

    jpeg_meta = load_json(os.path.join(assets_dir, "jpeg_model_meta.json"))
    jpeg_thr = float(jpeg_meta["threshold"])
    jpeg_model = load_bytes(os.path.join(assets_dir, "jpeg_ae.tflite"))

    tagseq_meta = load_json(os.path.join(assets_dir, "tagseq_gru_ae_meta.json"))
    tagseq_thr = float(tagseq_meta["threshold"])
    tagseq_model = load_bytes(os.path.join(assets_dir, "tagseq_gru_ae.tflite"))

    report: Dict[str, object] = {
        "policy": {
            "malicious": "CVE hit (TIFF/DNG)",
            "suspicious": "type mismatch / unsupported / error / anomaly score >= threshold",
            "benign": "otherwise",
        },
        "thresholds": {
            "tiff_ae": tiff_thr,
            "jpeg_ae": jpeg_thr,
            "tagseq": tagseq_thr,
        },
        "models": {
            "tiff_ae_tflite_bytes": os.path.getsize(os.path.join(assets_dir, "anomaly_ae.tflite")),
            "jpeg_ae_tflite_bytes": os.path.getsize(os.path.join(assets_dir, "jpeg_ae.tflite")),
            "tagseq_tflite_bytes": os.path.getsize(os.path.join(assets_dir, "tagseq_gru_ae.tflite")),
        },
        "results": {},
    }

    # --- JPEG ---
    jpeg_npz = "outputs/jpeg_features.npz"
    if os.path.exists(jpeg_npz):
        D = np.load(jpeg_npz, allow_pickle=True)
        X_struct = D["X_struct"].astype(np.float32)
        labels = D["labels"].astype(str)
        names = [str(x) for x in D["struct_feature_names"]]

        order = [names.index(n) for n in jpeg_meta["struct_feature_names"]]
        X = X_struct[:, order]
        scores = tflite_score_2d(jpeg_model, X, num_threads=num_threads)
        verdicts = np.where(scores >= jpeg_thr, "SUSPICIOUS", "BENIGN").astype(object)
        report["results"]["jpeg"] = {
            "npz": jpeg_npz,
            "summary": summarize_three_way(labels, verdicts, scores),
        }
    else:
        report["results"]["jpeg"] = {"error": f"Missing {jpeg_npz}. Run: python3 analysis/jpeg_feature_extract.py"}

    # --- TIFF AE (non-DNG scope) ---
    tiff_npz = "outputs/anomaly_features.npz"
    if os.path.exists(tiff_npz):
        D = np.load(tiff_npz, allow_pickle=True)
        X_bytes = D["X_bytes"]
        X_struct = D["X_struct"].astype(np.float32)
        labels = D["labels"].astype(str)
        paths = D["paths"].astype(str)
        struct_names = [str(x) for x in D["struct_feature_names"]]

        idx_is_tiff = struct_names.index("is_tiff")
        idx_is_dng = struct_names.index("is_dng")
        scope = (X_struct[:, idx_is_tiff].astype(np.int64) == 1) & (X_struct[:, idx_is_dng].astype(np.int64) == 0)

        X_bytes = X_bytes[scope]
        X_struct = X_struct[scope]
        labels = labels[scope]
        paths = paths[scope]

        X_hist = magika_bytes_to_hist_features(X_bytes)
        X = np.concatenate([X_hist, X_struct], axis=1).astype(np.float32)
        scores = tflite_score_2d(tiff_model, X, num_threads=num_threads)
        verdicts = np.where(scores >= tiff_thr, "SUSPICIOUS", "BENIGN").astype(object)

        cve_hits = 0
        static_warns = 0
        if args.with_cve:
            for i, p in enumerate(paths):
                mal, sus = static_findings(p)
                if mal:
                    verdicts[i] = "MALICIOUS"
                    cve_hits += 1
                elif sus:
                    verdicts[i] = "SUSPICIOUS"
                    static_warns += 1

        report["results"]["tiff_non_dng"] = {
            "npz": tiff_npz,
            "scope": {"is_tiff": True, "is_dng": False},
            "with_cve": bool(args.with_cve),
            "cve_hits": int(cve_hits),
            "static_warns": int(static_warns),
            "summary": summarize_three_way(labels, verdicts, scores),
        }
    else:
        report["results"]["tiff_non_dng"] = {"error": f"Missing {tiff_npz}. Run: python3 analysis/anomaly_feature_extract.py"}

    # --- DNG TagSeq (holdout + landfall) ---
    def eval_tagseq(npz_path: str) -> Dict[str, object]:
        if not os.path.exists(npz_path):
            return {"error": f"Missing {npz_path}. Run: python3 analysis/tag_sequence_extract.py"}
        D = np.load(npz_path, allow_pickle=True)
        feats = D["features"]
        tag_ids = D["tag_ids"]
        type_ids = D["type_ids"]
        ifd_kinds = D["ifd_kinds"]
        lengths = D["lengths"]
        labels = D["labels"].astype(str)
        paths = D["paths"].astype(str)

        scores = tagseq_scores(
            tagseq_model,
            feats,
            tag_ids,
            type_ids,
            ifd_kinds,
            lengths,
            num_threads=num_threads,
        )
        verdicts = np.where(scores >= tagseq_thr, "SUSPICIOUS", "BENIGN").astype(object)

        cve_hits = 0
        static_warns = 0
        if args.with_cve:
            for i, p in enumerate(paths):
                mal, sus = static_findings(p)
                if mal:
                    verdicts[i] = "MALICIOUS"
                    cve_hits += 1
                elif sus:
                    verdicts[i] = "SUSPICIOUS"
                    static_warns += 1

        return {
            "npz": npz_path,
            "with_cve": bool(args.with_cve),
            "cve_hits": int(cve_hits),
            "static_warns": int(static_warns),
            "summary": summarize_three_way(labels, verdicts, scores),
        }

    report["results"]["dng_holdout"] = eval_tagseq("outputs/tagseq_dng_holdout.npz")
    report["results"]["dng_landfall"] = eval_tagseq("outputs/tagseq_dng_landfall.npz")

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"\nWrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
