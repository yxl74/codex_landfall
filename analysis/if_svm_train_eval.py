#!/usr/bin/env python3
"""
Train and evaluate Isolation Forest / One-Class SVM on benign-only data.

Feature sets:
  - struct: structural + entropy only (low-dim, stable).
  - struct_bytes: struct + entropy + byte summary stats (recommended).
  - full_hist: struct + entropy + full byte histogram (high-dim).

Thresholding:
  - Uses a benign-only validation percentile (p99 by default).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - dependency check
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn\n"
        f"Import error: {exc}"
    )


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
    "max_declared_opcode_count",
}


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


def build_struct_features(X_struct: np.ndarray, names: List[str]) -> np.ndarray:
    X = X_struct.astype(np.float32).copy()
    name_to_idx = {n: i for i, n in enumerate(names)}
    for name in LOG_FIELDS:
        idx = name_to_idx.get(name)
        if idx is not None:
            X[:, idx] = np.log1p(X[:, idx])
    return X


def shannon_entropy(block: np.ndarray) -> float:
    data = block[block < 256]
    if data.size == 0:
        return 0.0
    counts = np.bincount(data.astype(np.int64), minlength=256).astype(np.float32)
    probs = counts / float(data.size)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def block_stats(block: np.ndarray) -> List[float]:
    total = float(block.size)
    if total == 0:
        return [0.0] * 7
    zeros = np.mean(block == 0)
    ff = np.mean(block == 255)
    ascii_ratio = np.mean((block >= 0x20) & (block <= 0x7E))
    ws = np.mean(np.isin(block, [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20]))
    padding = np.mean(block == 256)
    unique = block[block < 256]
    unique_ratio = float(len(np.unique(unique)) / 256.0) if unique.size else 0.0
    entropy = shannon_entropy(block)
    return [entropy, zeros, ff, ascii_ratio, ws, padding, unique_ratio]


def bytes_summary_features(X_bytes: np.ndarray) -> np.ndarray:
    feats = []
    for i in range(X_bytes.shape[0]):
        head = X_bytes[i, :1024]
        tail = X_bytes[i, 1024:]
        head_stats = block_stats(head)
        tail_stats = block_stats(tail)
        gap = abs(head_stats[0] - tail_stats[0])
        feats.append(head_stats + tail_stats + [gap])
    return np.array(feats, dtype=np.float32)


def build_feature_matrix(
    X_bytes: np.ndarray,
    X_struct: np.ndarray,
    struct_names: List[str],
    feature_set: str,
) -> np.ndarray:
    struct_feat = build_struct_features(X_struct, struct_names)
    if feature_set == "struct":
        return struct_feat
    if feature_set == "struct_bytes":
        byte_feat = bytes_summary_features(X_bytes)
        return np.concatenate([struct_feat, byte_feat], axis=1)
    if feature_set == "full_hist":
        byte_hist = magika_bytes_to_features(X_bytes)
        return np.concatenate([byte_hist, struct_feat], axis=1)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def split_indices(n: int, seed: int, splits: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
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
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", default="outputs/anomaly_features.npz")
    parser.add_argument("--output-json", default="outputs/if_svm_metrics.json")
    parser.add_argument("--output-model", default="")
    parser.add_argument("--model", choices=["iforest", "ocsvm"], default="iforest")
    parser.add_argument("--feature-set", choices=["struct", "struct_bytes", "full_hist"], default="struct_bytes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-samples", default="auto")
    parser.add_argument("--eval-npz", default="")
    args = parser.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    X_bytes = data["X_bytes"]
    X_struct = data["X_struct"]
    labels = data["labels"].astype(str)
    paths = data["paths"].astype(str)
    struct_names = [str(x) for x in data["struct_feature_names"]]

    X_all = build_feature_matrix(X_bytes, X_struct, struct_names, args.feature_set)
    y_all = np.array([0 if l == "benign" else 1 for l in labels], dtype=np.int64)

    train_idx, val_idx, test_idx = split_indices(len(labels), args.seed, (0.7, 0.15, 0.15))
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]
    labels_train = labels[train_idx]
    labels_val = labels[val_idx]
    labels_test = labels[test_idx]

    benign_train = labels_train == "benign"
    scaler = StandardScaler().fit(X_train[benign_train])
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    if args.model == "iforest":
        max_samples = args.max_samples
        if max_samples != "auto":
            max_samples = int(max_samples)
        model = IsolationForest(
            n_estimators=args.n_estimators,
            max_samples=max_samples,
            contamination="auto",
            random_state=args.seed,
        )
        model.fit(X_train_s[benign_train])
        score_train = -model.score_samples(X_train_s)
        score_val = -model.score_samples(X_val_s)
        score_test = -model.score_samples(X_test_s)
        score_all = -model.score_samples(scaler.transform(X_all))
    else:
        model = OneClassSVM(kernel="rbf", nu=args.nu, gamma=args.gamma)
        model.fit(X_train_s[benign_train])
        score_train = -model.decision_function(X_train_s).ravel()
        score_val = -model.decision_function(X_val_s).ravel()
        score_test = -model.decision_function(X_test_s).ravel()
        score_all = -model.decision_function(scaler.transform(X_all)).ravel()

    val_benign_scores = score_val[labels_val == "benign"]
    threshold = float(np.percentile(val_benign_scores, args.threshold_percentile))

    metrics = {
        "model": args.model,
        "feature_set": args.feature_set,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "counts": {
            "total": int(len(labels)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "train_benign": int(benign_train.sum()),
            "val_benign": int((labels_val == "benign").sum()),
        },
        "val_metrics": metrics_from_scores(np.array([0 if l == "benign" else 1 for l in labels_val]), score_val, threshold),
        "test_metrics": metrics_from_scores(np.array([0 if l == "benign" else 1 for l in labels_test]), score_test, threshold),
        "val_per_class": per_class_stats(labels_val, score_val, threshold),
        "test_per_class": per_class_stats(labels_test, score_test, threshold),
        "all_per_class": per_class_stats(labels, score_all, threshold),
    }

    if args.eval_npz:
        eval_data = np.load(args.eval_npz, allow_pickle=True)
        Xb = eval_data["X_bytes"]
        Xs = eval_data["X_struct"]
        labels_eval = eval_data["labels"].astype(str)
        struct_names_eval = [str(x) for x in eval_data["struct_feature_names"]]
        X_eval = build_feature_matrix(Xb, Xs, struct_names_eval, args.feature_set)
        X_eval_s = scaler.transform(X_eval)
        if args.model == "iforest":
            eval_scores = -model.score_samples(X_eval_s)
        else:
            eval_scores = -model.decision_function(X_eval_s).ravel()
        metrics["eval_per_class"] = per_class_stats(labels_eval, eval_scores, threshold)

    if args.output_model:
        os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
        with open(args.output_model, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "feature_set": args.feature_set}, f)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
