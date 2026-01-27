#!/usr/bin/env python3
"""
Train a benign-only anomaly model once and sweep thresholds.

Supports:
  - autoencoder
  - deepsvdd

Uses the anomaly-focused feature set from outputs/anomaly_features.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


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
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    priority = {"landfall": 2, "general_mal": 1, "benign": 0}
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(paths):
        h = compute_hash(str(p))
        groups.setdefault(h, []).append(i)

    kept: List[Tuple[int, str]] = []
    for _, idxs in groups.items():
        group_labels = [labels[i] for i in idxs]
        label = max(group_labels, key=lambda x: priority.get(x, -1))
        i0 = idxs[0]
        kept.append((i0, label))

    kept.sort(key=lambda x: x[0])
    indices = [i for i, _ in kept]
    labels_new = np.array([label for _, label in kept])
    return X_bytes[indices], X_struct[indices], labels_new, paths[indices]


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
    X = X_struct.astype(np.float32).copy()
    name_to_idx = {n: i for i, n in enumerate(names)}
    for name in log_fields:
        idx = name_to_idx.get(name)
        if idx is not None:
            X[:, idx] = np.log1p(X[:, idx])
    return X


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


def per_class_rates(labels: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cls in ("landfall", "general_mal", "benign"):
        mask = labels == cls
        if not np.any(mask):
            continue
        cls_scores = scores[mask]
        flagged = int(np.sum(cls_scores >= thr))
        total = int(mask.sum())
        if cls == "benign":
            out[cls] = {
                "total": total,
                "flagged": flagged,
                "fpr": float(flagged / max(1, total)),
            }
        else:
            out[cls] = {
                "total": total,
                "flagged": flagged,
                "recall": float(flagged / max(1, total)),
            }
    return out


def build_autoencoder(input_dim: int, hidden: List[int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    for h in reversed(hidden[:-1]):
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_network(input_dim: int, hidden: List[int], latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    outputs = tf.keras.layers.Dense(latent_dim, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", default="outputs/anomaly_features.npz")
    parser.add_argument("--output-json", default="outputs/anomaly_threshold_sweep.json")
    parser.add_argument("--model", choices=["ae", "deepsvdd"], default="ae")
    parser.add_argument("--bytes-mode", choices=["raw", "hist"], default="hist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.7,0.15,0.15")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", default="256,64")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--percentiles", default="99,98,97,95,90")
    args = parser.parse_args()

    splits = tuple(float(x) for x in args.split.split(","))
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    percentiles = [float(x) for x in args.percentiles.split(",") if x.strip()]

    data = np.load(args.input_npz, allow_pickle=True)
    X_bytes = data["X_bytes"]
    X_struct = data["X_struct"]
    labels = data["labels"]
    paths = data["paths"]
    struct_names = [str(x) for x in data["struct_feature_names"]]

    X_bytes, X_struct, labels, _ = deduplicate_by_hash(paths, X_bytes, X_struct, labels)

    X_bytes_feat = magika_bytes_to_features(X_bytes, args.bytes_mode)
    X_struct_feat = build_struct_features(X_struct, struct_names)
    X = np.concatenate([X_bytes_feat, X_struct_feat], axis=1)

    n = X.shape[0]
    train_idx, val_idx, test_idx = split_indices(n, args.seed, splits)

    benign_mask = labels == "benign"
    train_benign_idx = train_idx[benign_mask[train_idx]]
    val_benign_idx = val_idx[benign_mask[val_idx]]

    if train_benign_idx.size == 0:
        raise SystemExit("No benign samples in training split.")

    mean = X[train_benign_idx].mean(axis=0)
    std = X[train_benign_idx].std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    Xn = (X - mean) / std_safe

    if args.model == "ae":
        model = build_autoencoder(X.shape[1], hidden)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss="mse")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ]
        model.fit(
            Xn[train_benign_idx],
            Xn[train_benign_idx],
            validation_data=(Xn[val_benign_idx], Xn[val_benign_idx]),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        recon = model.predict(Xn, batch_size=args.batch_size, verbose=0)
        scores = np.mean((Xn - recon) ** 2, axis=1)
    else:
        model = build_network(X.shape[1], hidden, args.latent_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        z_init = model.predict(Xn[train_benign_idx], batch_size=args.batch_size, verbose=0)
        c = z_init.mean(axis=0).astype(np.float32)
        c_tf = tf.constant(c, dtype=tf.float32)
        train_ds = tf.data.Dataset.from_tensor_slices(Xn[train_benign_idx]).shuffle(1024).batch(args.batch_size)
        for _ in range(args.epochs):
            for batch in train_ds:
                with tf.GradientTape() as tape:
                    z = model(batch, training=True)
                    loss = tf.reduce_mean(tf.reduce_sum(tf.square(z - c_tf), axis=1))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
        z_all = model.predict(Xn, batch_size=args.batch_size, verbose=0)
        scores = np.sum((z_all - c.reshape(1, -1)) ** 2, axis=1)

    results = {
        "model": args.model,
        "bytes_mode": args.bytes_mode,
        "hidden": hidden,
        "latent_dim": args.latent_dim if args.model == "deepsvdd" else None,
        "percentiles": percentiles,
        "counts": {
            "total": int(n),
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
            "train_benign": int(train_benign_idx.size),
            "val_benign": int(val_benign_idx.size),
        },
        "sweep": [],
    }

    print("percentile,threshold,benign_fpr_all,landfall_recall_all,general_recall_all")
    for pct in percentiles:
        thr = float(np.percentile(scores[val_benign_idx], pct))
        all_rates = per_class_rates(labels, scores, thr)
        row = {
            "percentile": pct,
            "threshold": thr,
            "all_rates": all_rates,
        }
        results["sweep"].append(row)
        benign_fpr = all_rates.get("benign", {}).get("fpr", 0.0)
        land_recall = all_rates.get("landfall", {}).get("recall", 0.0)
        gen_recall = all_rates.get("general_mal", {}).get("recall", 0.0)
        print(f"{pct:.1f},{thr:.6f},{benign_fpr:.6f},{land_recall:.6f},{gen_recall:.6f}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Wrote:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
