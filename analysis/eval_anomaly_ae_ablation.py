#!/usr/bin/env python3
"""
Evaluate whether the TIFF anomaly AE is dominated by the bytes histogram features.

This script trains small AEs under different feature sets and reports:
- benign test FPR (at percentile threshold on benign val)
- malware test recall
- (for combined) relative contribution of bytes vs struct groups

Requires TensorFlow (use .venv-tf).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit("TensorFlow is required. Use the .venv-tf environment.") from exc


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def magika_bytes_to_hist(X_bytes: np.ndarray) -> np.ndarray:
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


def build_ae(input_dim: int, hidden: List[int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="x")
    x = inputs
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    for h in reversed(hidden[:-1]):
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    out = tf.keras.layers.Dense(input_dim, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=out)


@dataclass(frozen=True)
class Result:
    name: str
    thr: float
    test_ben_fpr: float
    test_mal_recall: float
    extra: dict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-npz", default="outputs/anomaly_features.npz")
    ap.add_argument("--train-scope", choices=["all_tiff", "tiff_non_dng"], default="tiff_non_dng")
    ap.add_argument("--split", default="0.7,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold-percentile", type=float, default=97.0)
    ap.add_argument("--hidden-bytes", default="256,64")
    ap.add_argument("--hidden-struct", default="64,16")
    ap.add_argument("--hidden-combined", default="256,64")
    args = ap.parse_args()

    set_seeds(args.seed)

    splits = tuple(float(x) for x in args.split.split(","))
    hidden_bytes = [int(x) for x in args.hidden_bytes.split(",") if x.strip()]
    hidden_struct = [int(x) for x in args.hidden_struct.split(",") if x.strip()]
    hidden_combined = [int(x) for x in args.hidden_combined.split(",") if x.strip()]

    data = np.load(args.input_npz, allow_pickle=True)
    X_bytes = data["X_bytes"]
    X_struct = data["X_struct"].astype(np.float32)
    labels = data["labels"]
    struct_names = [str(x) for x in data["struct_feature_names"]]

    idx_is_tiff = struct_names.index("is_tiff")
    idx_is_dng = struct_names.index("is_dng")
    is_tiff = X_struct[:, idx_is_tiff].astype(np.int64)
    is_dng = X_struct[:, idx_is_dng].astype(np.int64)
    if args.train_scope == "tiff_non_dng":
        mask_scope = (is_tiff == 1) & (is_dng == 0)
    else:
        mask_scope = is_tiff == 1

    X_bytes = X_bytes[mask_scope]
    X_struct = X_struct[mask_scope]
    labels = labels[mask_scope]

    # Build base feature matrices.
    X_bytes_hist = magika_bytes_to_hist(X_bytes)
    bytes_dim = X_bytes_hist.shape[1]  # 514
    struct_dim = X_struct.shape[1]
    X_combined_raw = np.concatenate([X_bytes_hist, X_struct], axis=1).astype(np.float32)

    # Prepare splits.
    n = X_combined_raw.shape[0]
    train_idx, val_idx, test_idx = split_indices(n, args.seed, splits)
    benign = labels == "benign"
    train_ben = train_idx[benign[train_idx]]
    val_ben = val_idx[benign[val_idx]]
    test_ben = test_idx[benign[test_idx]]
    test_mal = test_idx[~benign[test_idx]]

    if train_ben.size == 0 or val_ben.size == 0:
        raise SystemExit("Not enough benign samples for train/val splits.")

    def train_eval(name: str, X_raw: np.ndarray, bytes_dim_local: int, struct_names_local: List[str], hidden: List[int]) -> Result:
        log_mask = build_log_mask(struct_names_local, bytes_dim_local, X_raw.shape[1])
        log_idx = np.where(log_mask > 0)[0]
        X_log = X_raw.copy()
        if log_idx.size:
            X_log[:, log_idx] = np.log1p(X_log[:, log_idx])

        mean = X_log[train_ben].mean(axis=0).astype(np.float32)
        std = X_log[train_ben].std(axis=0).astype(np.float32)
        std_safe = np.where(std == 0, 1.0, std).astype(np.float32)
        X_norm = (X_log - mean) / std_safe

        model = build_ae(X_norm.shape[1], hidden)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss="mse")
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        model.fit(
            X_norm[train_ben],
            X_norm[train_ben],
            validation_data=(X_norm[val_ben], X_norm[val_ben]),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            callbacks=cb,
        )

        recon = model.predict(X_norm, batch_size=args.batch_size, verbose=0)
        sq = (X_norm - recon) ** 2
        per_sample_mse = np.mean(sq, axis=1)
        thr = float(np.percentile(per_sample_mse[val_ben], args.threshold_percentile))
        fpr = float(np.mean(per_sample_mse[test_ben] >= thr)) if test_ben.size else 0.0
        recall = float(np.mean(per_sample_mse[test_mal] >= thr)) if test_mal.size else 0.0

        extra = {}
        # For combined only, quantify per-dim error + group contribution.
        if name.startswith("combined") and bytes_dim_local > 0:
            total_dim = X_norm.shape[1]
            struct_dim_local = total_dim - bytes_dim_local
            def group_stats(idxs: np.ndarray) -> dict:
                if idxs.size == 0:
                    return {}
                sq_mean = sq[idxs].mean(axis=0)
                bytes_per_dim = float(sq_mean[:bytes_dim_local].mean())
                struct_per_dim = float(sq_mean[bytes_dim_local:].mean()) if struct_dim_local > 0 else 0.0
                return {
                    "bytes_per_dim_mse": bytes_per_dim,
                    "struct_per_dim_mse": struct_per_dim,
                    "bytes_contrib_to_score": bytes_per_dim * bytes_dim_local / total_dim,
                    "struct_contrib_to_score": struct_per_dim * struct_dim_local / total_dim,
                }
            extra["benign_test_group_stats"] = group_stats(test_ben)
            extra["mal_test_group_stats"] = group_stats(test_mal)
        return Result(name=name, thr=thr, test_ben_fpr=fpr, test_mal_recall=recall, extra=extra)

    results: List[Result] = []
    # Combined
    results.append(train_eval("combined(bytes+struct)", X_combined_raw, bytes_dim, struct_names, hidden_combined))
    # Bytes-only
    results.append(train_eval("bytes_only(hist514)", X_bytes_hist.astype(np.float32), bytes_dim_local=0, struct_names_local=[], hidden=hidden_bytes))
    # Struct-only
    results.append(train_eval("struct_only(36)", X_struct.astype(np.float32), bytes_dim_local=0, struct_names_local=struct_names, hidden=hidden_struct))

    # For combined, quantify contribution from bytes vs struct in normalized space.
    # NOTE: the scalar MSE is averaged across all dims, so bytes dims have a fixed ~93.45% weight.
    # Here we report per-group mean squared error per dimension to see if struct dims are more "active".
    print(f"Scope={args.train_scope} n={n} (ben={int(benign.sum())} mal={int((~benign).sum())})")
    print(f"bytes_dim={bytes_dim} struct_dim={struct_dim} total_dim={bytes_dim + struct_dim}")
    print(f"Score weighting by dim-count: bytes={(bytes_dim/(bytes_dim+struct_dim))*100:.2f}% struct={(struct_dim/(bytes_dim+struct_dim))*100:.2f}%")
    print("")
    for r in results:
        print(f"{r.name}: thr(p{args.threshold_percentile:.1f})={r.thr:.6f} | test_ben_fpr={r.test_ben_fpr*100:.2f}% | test_mal_recall={r.test_mal_recall*100:.2f}%")
        if r.extra:
            for k, v in r.extra.items():
                print(f"  - {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
