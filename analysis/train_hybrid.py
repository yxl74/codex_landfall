#!/usr/bin/env python3
"""
Train a simple logistic regression model on hybrid features (bytes + structure).

This script:
  - Loads outputs/hybrid_features.npz
  - Deduplicates by file hash to avoid leakage
  - Splits into train/val/test by hash group
  - Trains a logistic regression model (numpy)
  - Selects threshold on val (best F1)
  - Evaluates on test (accuracy/precision/recall/F1/ROC-AUC)
  - Saves model + normalization stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.0

    order = np.argsort(y_score)
    y_score_sorted = y_score[order]
    y_true_sorted = y_true[order]

    # Compute ranks with ties averaged
    ranks = np.zeros_like(y_score_sorted, dtype=np.float64)
    i = 0
    while i < len(y_score_sorted):
        j = i
        while j + 1 < len(y_score_sorted) and y_score_sorted[j + 1] == y_score_sorted[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based
        ranks[i : j + 1] = avg_rank
        i = j + 1

    sum_ranks_pos = np.sum(ranks[y_true_sorted == 1])
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_score >= thr).astype(np.int64)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best = None
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        m = metrics_at_threshold(y_true, y_score, thr)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best = m
    assert best is not None
    return best["threshold"], best


def logreg_train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
    class_weight: Tuple[float, float],
) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float32)
    b = 0.0
    w_pos, w_neg = class_weight

    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        weights = np.where(y == 1, w_pos, w_neg)
        error = (p - y) * weights
        grad_w = (X.T @ error) / n + l2 * w
        grad_b = float(np.mean(error))
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def magika_bytes_to_features(X_bytes: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return X_bytes.astype(np.float32) / 256.0
    if mode != "hist":
        raise ValueError(f"Unknown bytes mode: {mode}")

    # Histogram for first 1024 and last 1024 (257 bins including padding token 256)
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


def build_struct_features(X_struct: np.ndarray, names: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    # Apply log1p to scale-heavy fields
    log_fields = {
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
    X = X_struct.astype(np.float32).copy()
    name_to_idx = {n: i for i, n in enumerate(names)}
    for name in log_fields:
        idx = name_to_idx.get(name)
        if idx is not None:
            X[:, idx] = np.log1p(X[:, idx])
    return X, name_to_idx


def compute_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def deduplicate_by_hash(
    paths: np.ndarray,
    X_bytes: np.ndarray,
    X_struct: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group = {}
    for i, p in enumerate(paths):
        h = compute_hash(str(p))
        group.setdefault(h, []).append(i)

    kept = []
    for h, idxs in group.items():
        # Label is malicious if any file in group is malicious
        label = int(np.max(y[idxs]))
        # Keep first entry, but override label if needed
        i0 = idxs[0]
        kept.append((i0, label))

    kept.sort(key=lambda x: x[0])
    indices = [i for i, _ in kept]
    y_new = np.array([label for _, label in kept], dtype=np.int64)
    return X_bytes[indices], X_struct[indices], y_new, paths[indices]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", default="outputs/hybrid_features.npz")
    parser.add_argument("--output-model", default="outputs/hybrid_model.npz")
    parser.add_argument("--output-metrics", default="outputs/hybrid_metrics.json")
    parser.add_argument("--bytes-mode", choices=["raw", "hist"], default="hist")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.7,0.15,0.15")
    args = parser.parse_args()

    data = np.load(args.input_npz, allow_pickle=True)
    X_bytes = data["X_bytes"]
    X_struct = data["X_struct"]
    y = data["y"]
    paths = data["paths"]
    names = list(map(str, data.get("struct_feature_names", [])))

    # Deduplicate by hash
    X_bytes, X_struct, y, paths = deduplicate_by_hash(paths, X_bytes, X_struct, y)

    # Build features
    Xb = magika_bytes_to_features(X_bytes, args.bytes_mode)
    Xs, name_to_idx = build_struct_features(X_struct, names)
    X = np.concatenate([Xb, Xs], axis=1)

    # Normalize
    split_vals = tuple(float(x) for x in args.split.split(","))
    train_idx, val_idx, test_idx = split_indices(len(y), args.seed, split_vals)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Class weights
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    w_pos = (neg + pos) / (2.0 * max(1, pos))
    w_neg = (neg + pos) / (2.0 * max(1, neg))

    w, b = logreg_train(X_train, y_train, args.lr, args.epochs, args.l2, (w_pos, w_neg))

    # Threshold selection on val
    val_scores = sigmoid(X_val @ w + b)
    thr, val_metrics = choose_threshold(y_val, val_scores)

    # Test evaluation
    test_scores = sigmoid(X_test @ w + b)
    test_metrics = metrics_at_threshold(y_test, test_scores, thr)
    test_metrics["roc_auc"] = roc_auc(y_test, test_scores)

    # Rule baseline using flag_any if present
    rule_metrics = {}
    if "flag_any" in name_to_idx:
        flag_idx = name_to_idx["flag_any"]
        y_rule = (X_struct[:, flag_idx] > 0).astype(np.int64)
        y_rule_test = y_rule[test_idx]
        rule_metrics = metrics_at_threshold(y_test, y_rule_test.astype(np.float32), 0.5)

    metrics = {
        "counts": {
            "total": int(len(y)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "train_pos": int(np.sum(y_train == 1)),
            "train_neg": int(np.sum(y_train == 0)),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "rule_metrics": rule_metrics,
        "threshold": thr,
        "bytes_mode": args.bytes_mode,
    }

    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        args.output_model,
        w=w,
        b=b,
        mean=mean,
        std=std,
        bytes_mode=args.bytes_mode,
        struct_feature_names=np.array(names),
    )

    print(json.dumps(metrics, indent=2))
    print("Saved model:", args.output_model)
    print("Saved metrics:", args.output_metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
