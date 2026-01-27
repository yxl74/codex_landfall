#!/usr/bin/env python3
"""
Evaluate Hybrid and AE models on a holdout NPZ set.

Inputs:
  - hybrid_features_holdout.npz
  - anomaly_features_holdout.npz
  - hybrid_model.npz
  - ae_model_p99_holdout.keras
  - ae_metrics_p99_holdout.json (threshold)
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import tensorflow as tf


LOG_FIELDS_HYBRID = {
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
    "max_declared_opcode_count",
}


LOG_FIELDS_AE = {
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
    "max_declared_opcode_count",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def magika_bytes_to_features(X_bytes: np.ndarray) -> np.ndarray:
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


def build_struct_features(X_struct: np.ndarray, names: List[str], log_fields: set) -> np.ndarray:
    X = X_struct.astype(np.float32).copy()
    name_to_idx = {n: i for i, n in enumerate(names)}
    for name in log_fields:
        idx = name_to_idx.get(name)
        if idx is not None:
            X[:, idx] = np.log1p(X[:, idx])
    return X


def split_indices(n: int, seed: int, splits=(0.7, 0.15, 0.15)):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = scores >= threshold
    tp = int(np.sum((y_true == 1) & preds))
    tn = int(np.sum((y_true == 0) & (~preds)))
    fp = int(np.sum((y_true == 0) & preds))
    fn = int(np.sum((y_true == 1) & (~preds)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / max(1, len(y_true))
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def per_class_stats(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name in sorted(set(labels)):
        mask = labels == name
        sc = scores[mask]
        flagged = int(np.sum(sc >= threshold))
        total = int(mask.sum())
        if name == "benign":
            rate = flagged / total if total else 0.0
            out[name] = {
                "total": total,
                "flagged": flagged,
                "fpr": rate,
                "score_min": float(sc.min()) if total else 0.0,
                "score_mean": float(sc.mean()) if total else 0.0,
                "score_max": float(sc.max()) if total else 0.0,
            }
        else:
            recall = flagged / total if total else 0.0
            out[name] = {
                "total": total,
                "flagged": flagged,
                "recall": recall,
                "score_min": float(sc.min()) if total else 0.0,
                "score_mean": float(sc.mean()) if total else 0.0,
                "score_max": float(sc.max()) if total else 0.0,
            }
    return out


def eval_hybrid(holdout_npz: str, model_npz: str, metrics_json: str) -> Dict[str, object]:
    data = np.load(holdout_npz, allow_pickle=True)
    X_bytes = data["X_bytes"]
    X_struct = data["X_struct"]
    labels = data["labels"].astype(str)
    struct_names = [str(x) for x in data["struct_feature_names"]]

    model = np.load(model_npz, allow_pickle=True)
    w = model["w"].astype(np.float32)
    b = float(model["b"])
    mean = model["mean"].astype(np.float32)
    std = model["std"].astype(np.float32)

    X_bytes_feat = magika_bytes_to_features(X_bytes)
    X_struct_feat = build_struct_features(X_struct, struct_names, LOG_FIELDS_HYBRID)
    X = np.concatenate([X_bytes_feat, X_struct_feat], axis=1)
    X_norm = (X - mean) / np.where(std == 0, 1.0, std)
    scores = sigmoid(X_norm @ w + b).astype(np.float32)

    with open(metrics_json, "r", encoding="utf-8") as f:
        thr = json.load(f)["threshold"]

    y_true = np.array([0 if l == "benign" else 1 for l in labels], dtype=np.int64)
    return {
        "threshold": float(thr),
        "metrics": metrics_at_threshold(y_true, scores, thr),
        "per_class": per_class_stats(labels, scores, thr),
    }


def eval_ae(
    train_npz: str,
    holdout_npz: str,
    model_path: str,
    metrics_json: str,
    seed: int,
) -> Dict[str, object]:
    train = np.load(train_npz, allow_pickle=True)
    X_bytes = train["X_bytes"]
    X_struct = train["X_struct"]
    labels = train["labels"].astype(str)
    struct_names = [str(x) for x in train["struct_feature_names"]]

    X_bytes_feat = magika_bytes_to_features(X_bytes)
    X_struct_feat = build_struct_features(X_struct, struct_names, LOG_FIELDS_AE)
    X_train = np.concatenate([X_bytes_feat, X_struct_feat], axis=1)

    train_idx, _, _ = split_indices(len(labels), seed)
    benign_train = labels[train_idx] == "benign"
    X_train = X_train[train_idx]
    mean = X_train[benign_train].mean(axis=0).astype(np.float32)
    std = X_train[benign_train].std(axis=0).astype(np.float32)
    std = np.where(std == 0, 1.0, std)

    holdout = np.load(holdout_npz, allow_pickle=True)
    Hb = holdout["X_bytes"]
    Hs = holdout["X_struct"]
    Hlabels = holdout["labels"].astype(str)
    Hstruct_names = [str(x) for x in holdout["struct_feature_names"]]

    Hb_feat = magika_bytes_to_features(Hb)
    Hs_feat = build_struct_features(Hs, Hstruct_names, LOG_FIELDS_AE)
    H = np.concatenate([Hb_feat, Hs_feat], axis=1)
    H_norm = (H - mean) / std

    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(H_norm, verbose=0)
    scores = np.mean((preds - H_norm) ** 2, axis=1).astype(np.float32)

    with open(metrics_json, "r", encoding="utf-8") as f:
        thr = json.load(f)["threshold"]

    y_true = np.array([0 if l == "benign" else 1 for l in Hlabels], dtype=np.int64)
    return {
        "threshold": float(thr),
        "metrics": metrics_at_threshold(y_true, scores, thr),
        "per_class": per_class_stats(Hlabels, scores, thr),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-holdout", default="outputs/hybrid_features_holdout.npz")
    parser.add_argument("--hybrid-model", default="outputs/hybrid_model.npz")
    parser.add_argument("--hybrid-metrics", default="outputs/hybrid_metrics.json")
    parser.add_argument("--ae-train-npz", default="outputs/anomaly_features.npz")
    parser.add_argument("--ae-holdout", default="outputs/anomaly_features_holdout.npz")
    parser.add_argument("--ae-model", default="outputs/ae_model_p99_holdout.keras")
    parser.add_argument("--ae-metrics", default="outputs/ae_metrics_p99_holdout.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="outputs/holdout_eval_models.json")
    args = parser.parse_args()

    results = {
        "hybrid": eval_hybrid(args.hybrid_holdout, args.hybrid_model, args.hybrid_metrics),
        "ae": eval_ae(args.ae_train_npz, args.ae_holdout, args.ae_model, args.ae_metrics, args.seed),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
