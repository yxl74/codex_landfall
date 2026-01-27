#!/usr/bin/env python3
"""
Train a benign-only autoencoder on anomaly features and export a TFLite
model that outputs an anomaly score (MSE) from raw features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def build_log_mask(struct_names: List[str], bytes_dim: int, total_dim: int) -> np.ndarray:
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
        "max_declared_opcode_count",
    }
    mask = np.zeros((total_dim,), dtype=np.float32)
    for i, name in enumerate(struct_names):
        if name in log_fields:
            mask[bytes_dim + i] = 1.0
    return mask


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


def build_autoencoder(input_dim: int, hidden: List[int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features_norm")
    x = inputs
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    for h in reversed(hidden[:-1]):
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ae_core")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", default="outputs/anomaly_features.npz")
    parser.add_argument("--output-tflite", default="outputs/anomaly_ae.tflite")
    parser.add_argument("--output-meta", default="outputs/anomaly_model_meta.json")
    parser.add_argument("--bytes-mode", choices=["raw", "hist"], default="hist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.7,0.15,0.15")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", default="256,64")
    parser.add_argument("--threshold-percentile", type=float, default=97.0)
    args = parser.parse_args()

    set_seeds(args.seed)

    splits = tuple(float(x) for x in args.split.split(","))
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]

    data = np.load(args.input_npz, allow_pickle=True)
    X_bytes = data["X_bytes"]
    X_struct = data["X_struct"].astype(np.float32)
    labels = data["labels"]
    paths = data["paths"]
    struct_names = [str(x) for x in data["struct_feature_names"]]

    X_bytes, X_struct, labels, paths = deduplicate_by_hash(paths, X_bytes, X_struct, labels)

    X_bytes_feat = magika_bytes_to_features(X_bytes, args.bytes_mode)
    X_raw = np.concatenate([X_bytes_feat, X_struct], axis=1).astype(np.float32)

    bytes_dim = 514 if args.bytes_mode == "hist" else 2048
    log_mask = build_log_mask(struct_names, bytes_dim, X_raw.shape[1])
    log_idx = np.where(log_mask > 0)[0]

    X_log = X_raw.copy()
    if log_idx.size > 0:
        X_log[:, log_idx] = np.log1p(X_log[:, log_idx])

    n = X_log.shape[0]
    train_idx, val_idx, _ = split_indices(n, args.seed, splits)
    benign_mask = labels == "benign"
    train_benign_idx = train_idx[benign_mask[train_idx]]
    val_benign_idx = val_idx[benign_mask[val_idx]]
    if train_benign_idx.size == 0:
        raise SystemExit("No benign samples in training split.")

    mean = X_log[train_benign_idx].mean(axis=0).astype(np.float32)
    std = X_log[train_benign_idx].std(axis=0).astype(np.float32)
    std_safe = np.where(std == 0, 1.0, std).astype(np.float32)
    X_norm = (X_log - mean) / std_safe

    ae = build_autoencoder(X_norm.shape[1], hidden)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss="mse")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    ae.fit(
        X_norm[train_benign_idx],
        X_norm[train_benign_idx],
        validation_data=(X_norm[val_benign_idx], X_norm[val_benign_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    recon = ae.predict(X_norm, batch_size=args.batch_size, verbose=0)
    errors = np.mean((X_norm - recon) ** 2, axis=1)
    threshold = float(np.percentile(errors[val_benign_idx], args.threshold_percentile))

    inputs = tf.keras.Input(shape=(X_raw.shape[1],), name="features_raw")
    mask_tf = tf.constant(log_mask, dtype=tf.float32)
    x_log = tf.keras.layers.Lambda(lambda t: tf.math.log1p(t), name="log1p")(inputs)
    x = inputs * (1.0 - mask_tf) + x_log * mask_tf
    x = (x - tf.constant(mean)) / tf.constant(std_safe)
    recon_out = ae(x)
    diff = tf.keras.layers.Subtract(name="recon_diff")([x, recon_out])
    sq = tf.keras.layers.Lambda(lambda t: tf.square(t), name="recon_sq")(diff)
    mse = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),
        name="recon_mse",
    )(sq)
    export_model = tf.keras.Model(inputs=inputs, outputs=mse, name="ae_anomaly_score")

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.output_tflite), exist_ok=True)
    with open(args.output_tflite, "wb") as f:
        f.write(tflite_model)

    meta = {
        "model": "autoencoder_anomaly_score",
        "feature_dim": int(X_raw.shape[1]),
        "bytes_mode": args.bytes_mode,
        "struct_feature_names": struct_names,
        "hidden": hidden,
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
    }
    os.makedirs(os.path.dirname(args.output_meta), exist_ok=True)
    with open(args.output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", args.output_tflite)
    print("Wrote:", args.output_meta)
    print(f"Threshold (p{args.threshold_percentile:.1f}) = {threshold:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
