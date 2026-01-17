#!/usr/bin/env python3
"""
Run ablations on hybrid features and write a report.
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

    ranks = np.zeros_like(y_score_sorted, dtype=np.float64)
    i = 0
    while i < len(y_score_sorted):
        j = i
        while j + 1 < len(y_score_sorted) and y_score_sorted[j + 1] == y_score_sorted[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
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
        label = int(np.max(y[idxs]))
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


def eval_model(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    lr: float,
    epochs: int,
    l2: float,
) -> Dict[str, float]:
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

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    w_pos = (neg + pos) / (2.0 * max(1, pos))
    w_neg = (neg + pos) / (2.0 * max(1, neg))

    w, b = logreg_train(X_train, y_train, lr, epochs, l2, (w_pos, w_neg))

    val_scores = sigmoid(X_val @ w + b)
    thr, val_metrics = choose_threshold(y_val, val_scores)

    test_scores = sigmoid(X_test @ w + b)
    test_metrics = metrics_at_threshold(y_test, test_scores, thr)
    test_metrics["roc_auc"] = roc_auc(y_test, test_scores)
    test_metrics["threshold"] = thr

    return {
        "val": val_metrics,
        "test": test_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", default="outputs/hybrid_features.npz")
    parser.add_argument("--output-json", default="outputs/ablation_results.json")
    parser.add_argument("--output-md", default="outputs/ablation_results.md")
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

    split_vals = tuple(float(x) for x in args.split.split(","))
    train_idx, val_idx, test_idx = split_indices(len(y), args.seed, split_vals)

    # Prepare features
    Xs, name_to_idx = build_struct_features(X_struct, names)
    Xb_hist = magika_bytes_to_features(X_bytes, "hist")
    Xb_raw = magika_bytes_to_features(X_bytes, "raw")

    results: Dict[str, Dict[str, float]] = {}

    # Rules baseline
    if "flag_any" in name_to_idx:
        flag_idx = name_to_idx["flag_any"]
        y_rule = (X_struct[:, flag_idx] > 0).astype(np.int64)
        rule_metrics = metrics_at_threshold(y[test_idx], y_rule[test_idx].astype(np.float32), 0.5)
        results["rules_flag_any"] = {
            "accuracy": rule_metrics["accuracy"],
            "precision": rule_metrics["precision"],
            "recall": rule_metrics["recall"],
            "f1": rule_metrics["f1"],
            "roc_auc": 0.0,
            "threshold": 0.5,
        }

    # Structural only
    struct_metrics = eval_model(Xs, y, train_idx, val_idx, test_idx, args.lr, args.epochs, args.l2)
    results["struct_only"] = {
        "accuracy": struct_metrics["test"]["accuracy"],
        "precision": struct_metrics["test"]["precision"],
        "recall": struct_metrics["test"]["recall"],
        "f1": struct_metrics["test"]["f1"],
        "roc_auc": struct_metrics["test"]["roc_auc"],
        "threshold": struct_metrics["test"]["threshold"],
    }

    # Bytes only (hist)
    bytes_hist_metrics = eval_model(Xb_hist, y, train_idx, val_idx, test_idx, args.lr, args.epochs, args.l2)
    results["bytes_only_hist"] = {
        "accuracy": bytes_hist_metrics["test"]["accuracy"],
        "precision": bytes_hist_metrics["test"]["precision"],
        "recall": bytes_hist_metrics["test"]["recall"],
        "f1": bytes_hist_metrics["test"]["f1"],
        "roc_auc": bytes_hist_metrics["test"]["roc_auc"],
        "threshold": bytes_hist_metrics["test"]["threshold"],
    }

    # Bytes only (raw)
    bytes_raw_metrics = eval_model(Xb_raw, y, train_idx, val_idx, test_idx, args.lr, args.epochs, args.l2)
    results["bytes_only_raw"] = {
        "accuracy": bytes_raw_metrics["test"]["accuracy"],
        "precision": bytes_raw_metrics["test"]["precision"],
        "recall": bytes_raw_metrics["test"]["recall"],
        "f1": bytes_raw_metrics["test"]["f1"],
        "roc_auc": bytes_raw_metrics["test"]["roc_auc"],
        "threshold": bytes_raw_metrics["test"]["threshold"],
    }

    # Hybrid (hist + struct)
    X_hybrid_hist = np.concatenate([Xb_hist, Xs], axis=1)
    hybrid_hist_metrics = eval_model(X_hybrid_hist, y, train_idx, val_idx, test_idx, args.lr, args.epochs, args.l2)
    results["hybrid_hist"] = {
        "accuracy": hybrid_hist_metrics["test"]["accuracy"],
        "precision": hybrid_hist_metrics["test"]["precision"],
        "recall": hybrid_hist_metrics["test"]["recall"],
        "f1": hybrid_hist_metrics["test"]["f1"],
        "roc_auc": hybrid_hist_metrics["test"]["roc_auc"],
        "threshold": hybrid_hist_metrics["test"]["threshold"],
    }

    # Hybrid (raw + struct)
    X_hybrid_raw = np.concatenate([Xb_raw, Xs], axis=1)
    hybrid_raw_metrics = eval_model(X_hybrid_raw, y, train_idx, val_idx, test_idx, args.lr, args.epochs, args.l2)
    results["hybrid_raw"] = {
        "accuracy": hybrid_raw_metrics["test"]["accuracy"],
        "precision": hybrid_raw_metrics["test"]["precision"],
        "recall": hybrid_raw_metrics["test"]["recall"],
        "f1": hybrid_raw_metrics["test"]["f1"],
        "roc_auc": hybrid_raw_metrics["test"]["roc_auc"],
        "threshold": hybrid_raw_metrics["test"]["threshold"],
    }

    report = {
        "counts": {
            "total": int(len(y)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "results": results,
        "bytes_mode_default": args.bytes_mode,
        "seed": args.seed,
        "split": split_vals,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write simple markdown table
    headers = ["model", "acc", "prec", "recall", "f1", "roc_auc", "thr"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, m in results.items():
        lines.append(
            "| {name} | {acc:.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {auc:.3f} | {thr:.2f} |".format(
                name=name,
                acc=m["accuracy"],
                prec=m["precision"],
                rec=m["recall"],
                f1=m["f1"],
                auc=m["roc_auc"],
                thr=m["threshold"],
            )
        )
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(report, indent=2))
    print("Saved:", args.output_json)
    print("Saved:", args.output_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
