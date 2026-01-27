#!/usr/bin/env python3
"""
Analyze benign files flagged as malicious by the anomaly model.

Outputs:
  - Markdown report with per-file outlier features
  - JSON with raw scores and z-stats
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


def load_anomaly_extract():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "anomaly_feature_extract", os.path.join("analysis", "anomaly_feature_extract.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


LOG_FIELDS = {
    "min_width",
    "min_height",
    "ifd_entry_max",
    "subifd_count_sum",
    "new_subfile_types_unique",
    "total_opcodes",
    "unknown_opcodes",
    "max_opcode_id",
    "opcode_list1_bytes",
    "opcode_list2_bytes",
    "opcode_list3_bytes",
    "max_width",
    "max_height",
    "total_pixels",
    "file_size",
    "bytes_per_pixel_milli",
    "pixels_per_mb",
    "opcode_list_bytes_total",
    "opcode_list_bytes_max",
    "opcode_list_present_count",
    "opcode_bytes_ratio_permille",
    "opcode_bytes_per_opcode_milli",
    "unknown_opcode_ratio_permille",
}


def magika_bytes_to_features(X_bytes: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return X_bytes.astype(np.float32) / 256.0
    if mode != "hist":
        raise ValueError(f"Unknown bytes mode: {mode}")

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


def build_struct_features(X_struct: np.ndarray, names: List[str]) -> np.ndarray:
    X = X_struct.astype(np.float32).copy()
    name_to_idx = {n: i for i, n in enumerate(names)}
    for name in LOG_FIELDS:
        idx = name_to_idx.get(name)
        if idx is not None:
            X[:, idx] = np.log1p(X[:, idx])
    return X


@dataclass
class FileStats:
    path: str
    size: int
    is_tiff: int
    is_dng: int
    anomaly_score: float
    anomaly_flag: bool
    bytes_z_max: float
    bytes_z_mean: float
    top_struct: List[Dict[str, float]]


def compute_benign_stats(
    npz_path: str,
    bytes_mode: str,
    struct_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    X_bytes = data["X_bytes"]
    X_struct = data["X_struct"]
    labels = data["labels"]

    X_bytes_feat = magika_bytes_to_features(X_bytes, bytes_mode)
    X_struct_feat = build_struct_features(X_struct, struct_names)
    X = np.concatenate([X_bytes_feat, X_struct_feat], axis=1)

    benign_mask = labels == "benign"
    X_benign = X[benign_mask]
    mean = X_benign.mean(axis=0)
    std = X_benign.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def make_struct_vector(
    tiff_feat: Dict[str, int],
    struct_names: List[str],
    entropy: Tuple[float, float, float, float],
    file_size: int,
) -> np.ndarray:
    header_entropy, tail_entropy, overall_entropy, entropy_gradient = entropy
    struct_map = dict(tiff_feat)
    struct_map["file_size"] = file_size
    struct_map["header_entropy"] = header_entropy
    struct_map["tail_entropy"] = tail_entropy
    struct_map["overall_entropy"] = overall_entropy
    struct_map["entropy_gradient"] = entropy_gradient
    return np.array([float(struct_map.get(name, 0)) for name in struct_names], dtype=np.float32)


def compute_entropy(module, path: str) -> Tuple[float, float, float, float]:
    file_size = os.path.getsize(path)
    header = b""
    tail = b""
    overall = b""
    with open(path, "rb") as f:
        header = f.read(min(4096, file_size))
        if file_size > 0:
            f.seek(max(0, file_size - min(4096, file_size)))
            tail = f.read(min(4096, file_size))
        f.seek(0)
        overall = f.read(min(65536, file_size))

    header_entropy = module.byte_entropy(header)
    tail_entropy = module.byte_entropy(tail)
    overall_entropy = module.byte_entropy(overall)
    entropy_gradient = abs(header_entropy - tail_entropy)
    return header_entropy, tail_entropy, overall_entropy, entropy_gradient


def tflite_score(model_path: str, features: np.ndarray) -> float:
    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = features.astype(np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return float(output[0][0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/benign_fp_samples")
    parser.add_argument("--anomaly-model", default="outputs/anomaly_ae.tflite")
    parser.add_argument("--anomaly-meta", default="outputs/anomaly_model_meta.json")
    parser.add_argument("--benign-npz", default="outputs/anomaly_features.npz")
    parser.add_argument("--output-md", default="outputs/benign_fp_analysis.md")
    parser.add_argument("--output-json", default="outputs/benign_fp_analysis.json")
    args = parser.parse_args()

    with open(args.anomaly_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    bytes_mode = meta["bytes_mode"]
    struct_names = [str(x) for x in meta["struct_feature_names"]]
    threshold = float(meta["threshold"])

    extractor = load_anomaly_extract()
    mean, std = compute_benign_stats(args.benign_npz, bytes_mode, struct_names)
    bytes_dim = 514 if bytes_mode == "hist" else 2048

    files = sorted(
        os.path.join(args.input_dir, name)
        for name in os.listdir(args.input_dir)
        if os.path.isfile(os.path.join(args.input_dir, name))
    )
    results: List[FileStats] = []

    for path in files:
        tiff_feat = extractor.parse_tiff_struct(path)
        if tiff_feat is None:
            tiff_feat = {name: 0 for name in struct_names if name not in (
                "header_entropy", "tail_entropy", "overall_entropy", "entropy_gradient"
            )}
            tiff_feat["is_tiff"] = 0
            tiff_feat["is_dng"] = 0

        entropy = compute_entropy(extractor, path)
        struct_vec = make_struct_vector(tiff_feat, struct_names, entropy, os.path.getsize(path))

        byte_raw = extractor.magika_like_bytes(path)
        byte_feat = magika_bytes_to_features(byte_raw.reshape(1, -1), bytes_mode)[0]
        features = np.concatenate([byte_feat, struct_vec], axis=0)

        score = tflite_score(args.anomaly_model, features)
        flagged = score >= threshold

        struct_adj = struct_vec.copy()
        for i, name in enumerate(struct_names):
            if name in LOG_FIELDS:
                struct_adj[i] = np.log1p(struct_adj[i])
        combined = np.concatenate([byte_feat, struct_adj], axis=0)
        z = (combined - mean) / std
        z_bytes = z[:bytes_dim]
        z_struct = z[bytes_dim:]

        top_idx = np.argsort(np.abs(z_struct))[::-1][:8]
        top_struct = []
        for idx in top_idx:
            name = struct_names[idx]
            top_struct.append(
                {
                    "name": name,
                    "value": float(struct_vec[idx]),
                    "z": float(z_struct[idx]),
                }
            )

        results.append(
            FileStats(
                path=path,
                size=os.path.getsize(path),
                is_tiff=int(tiff_feat.get("is_tiff", 0)),
                is_dng=int(tiff_feat.get("is_dng", 0)),
                anomaly_score=score,
                anomaly_flag=bool(flagged),
                bytes_z_max=float(np.max(np.abs(z_bytes))),
                bytes_z_mean=float(np.mean(np.abs(z_bytes))),
                top_struct=top_struct,
            )
        )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    lines = []
    lines.append("# Benign FP analysis\n")
    lines.append(f"Threshold: {threshold:.6f}\n")
    lines.append("")
    for r in results:
        lines.append(f"## {os.path.basename(r.path)}")
        lines.append(f"- path: `{r.path}`")
        lines.append(f"- size: {r.size} bytes")
        lines.append(f"- tiff: {r.is_tiff}  dng: {r.is_dng}")
        lines.append(f"- anomaly_score: {r.anomaly_score:.6f}  flagged: {r.anomaly_flag}")
        lines.append(f"- bytes_z_max: {r.bytes_z_max:.3f}  bytes_z_mean: {r.bytes_z_mean:.3f}")
        lines.append("")
        lines.append("Top structural outliers (abs z):")
        for item in r.top_struct:
            lines.append(f"- {item['name']}: value={item['value']:.4f} z={item['z']:.3f}")
        lines.append("")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote:", args.output_md)
    print("Wrote:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
