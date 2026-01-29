#!/usr/bin/env python3
"""
Train a benign-only autoencoder on JPEG structural features and export a
TFLite model that outputs an anomaly score (MSE) from raw JPEG features.

Input:
  - outputs/jpeg_features.npz (from analysis/jpeg_feature_extract.py)

Outputs:
  - outputs/jpeg_ae.tflite
  - outputs/jpeg_model_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="jpeg_ae_core")


def build_log_mask(struct_names: List[str]) -> np.ndarray:
    # Log-scale fields with long-tailed distributions.
    log_fields = {
        "file_size",
        "bytes_after_eoi",
        "app_bytes",
        "com_bytes",
        "dqt_bytes",
        "dht_bytes",
        "app_max_len",
        "dqt_max_len",
        "dht_max_len",
        "max_seg_len",
        "width",
        "height",
        "dri_interval",
    }
    mask = np.zeros((len(struct_names),), dtype=np.float32)
    for i, name in enumerate(struct_names):
        if name in log_fields:
            mask[i] = 1.0
    return mask


def tflite_scores(tflite_model: bytes, X: np.ndarray, num_threads: int = 2) -> np.ndarray:
    """Run the exported TFLite model on a batch of raw feature vectors."""
    if X.size == 0:
        return np.zeros((0,), dtype=np.float32)
    interpreter = tf.lite.Interpreter(model_content=tflite_model, num_threads=num_threads)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.resize_tensor_input(input_index, [int(X.shape[0]), int(X.shape[1])])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, X.astype(np.float32))
    interpreter.invoke()
    out = interpreter.get_tensor(output_index)
    return out.reshape(-1).astype(np.float32)


def _u16be(buf: bytes, off: int) -> Optional[int]:
    if off + 2 > len(buf):
        return None
    return (buf[off] << 8) | buf[off + 1]


def _parse_dqt_tables(payload: bytes) -> Optional[List[bytes]]:
    tables: List[bytes] = []
    p = 0
    n = len(payload)
    while p < n:
        if p >= n:
            break
        pq_tq = payload[p]
        p += 1
        pq = (pq_tq >> 4) & 0x0F
        table_bytes = 64 * (2 if pq else 1)
        if p + table_bytes > n:
            return None
        tables.append(bytes([pq_tq]) + payload[p : p + table_bytes])
        p += table_bytes
    return tables


def _parse_dht_tables(payload: bytes) -> Optional[List[bytes]]:
    tables: List[bytes] = []
    p = 0
    n = len(payload)
    while p < n:
        if p + 1 + 16 > n:
            return None
        tc_th = payload[p]
        p += 1
        counts16 = payload[p : p + 16]
        p += 16
        total_syms = int(sum(counts16))
        if p + total_syms > n:
            return None
        syms = payload[p : p + total_syms]
        p += total_syms
        tables.append(bytes([tc_th]) + counts16 + syms)
    return tables


def _make_segment(marker: int, payload: bytes) -> bytes:
    seglen = len(payload) + 2
    if seglen > 0xFFFF:
        raise ValueError("Segment too large")
    return bytes([0xFF, marker, (seglen >> 8) & 0xFF, seglen & 0xFF]) + payload


def _insert_app_segments(data: bytes, count: int, marker: int = 0xE2, payload_len: int = 64) -> bytes:
    """Insert `count` APPn segments right after SOI.

    This is a benign augmentation to broaden APP marker statistics (common in phone camera JPEGs).
    """
    if count <= 0:
        return data
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return data

    payload = (b"LAND_FALL_DETECTOR_AUGMENTATION" * 8)[:payload_len]
    seg = _make_segment(marker, payload)
    return data[:2] + (seg * count) + data[2:]


def _append_tail_bytes(data: bytes, count: int, fill: int = 0x00) -> bytes:
    """Append `count` bytes after EOI (benign augmentation).

    Many real-world camera JPEGs contain a small amount of trailing bytes after the final EOI marker.
    Appending bytes after EOI keeps the JPEG decodable (bytes are ignored by decoders) while broadening
    the benign distribution of `bytes_after_eoi` features.
    """
    if count <= 0:
        return data
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return data
    if data.rfind(b"\xff\xd9") == -1:
        return data
    return data + (bytes([fill]) * count)


def _resegment_dqt_dht(data: bytes) -> Optional[bytes]:
    """Split combined DQT/DHT segments into 1-table-per-segment variants.

    This creates a benign structural variant without decoding/re-encoding pixels.
    """
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None

    n = len(data)
    pos = 2
    out = bytearray(b"\xff\xd8")

    while pos < n - 1:
        if data[pos] != 0xFF:
            # Unexpected (header corruption). Keep bytes verbatim and continue.
            out.append(data[pos])
            pos += 1
            continue

        # Skip fill bytes
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break

        marker = data[pos]
        pos += 1

        if marker == 0x00:
            continue
        if marker == 0xD9:
            out += b"\xff\xd9"
            out += data[pos:]
            return bytes(out)
        if 0xD0 <= marker <= 0xD7:
            out += bytes([0xFF, marker])
            continue
        if marker in (0xD8, 0x01):
            out += bytes([0xFF, marker])
            continue

        seglen = _u16be(data, pos)
        if seglen is None or seglen < 2:
            return None
        pos += 2
        payload_len = seglen - 2
        if pos + payload_len > n:
            return None
        payload = data[pos : pos + payload_len]
        pos += payload_len

        if marker == 0xDA:
            # SOS: copy as-is and keep the rest intact.
            out += _make_segment(marker, payload)
            out += data[pos:]
            return bytes(out)

        if marker == 0xDB:
            tables = _parse_dqt_tables(payload)
            if tables is None:
                out += _make_segment(marker, payload)
            else:
                for t in tables:
                    out += _make_segment(marker, t)
            continue

        if marker == 0xC4:
            tables = _parse_dht_tables(payload)
            if tables is None:
                out += _make_segment(marker, payload)
            else:
                for t in tables:
                    out += _make_segment(marker, t)
            continue

        out += _make_segment(marker, payload)

    return bytes(out)


def _tail_has_magic(tail: bytes) -> Tuple[int, int, int]:
    tail_zip = int((b"PK\x03\x04" in tail) or (b"PK\x05\x06" in tail))
    tail_pdf = int(b"%PDF" in tail)
    tail_elf = int(b"\x7fELF" in tail)
    return tail_zip, tail_pdf, tail_elf


def parse_jpeg_struct_bytes(data: bytes) -> Optional[Dict[str, float]]:
    """Parse JPEG structure from bytes (mirrors jpeg_feature_extract.py)."""
    n = len(data)
    if n < 4 or data[0:2] != b"\xff\xd8":
        return None

    eoi_pos = data.rfind(b"\xff\xd9")
    has_eoi = 1 if eoi_pos != -1 else 0
    bytes_after_eoi = (n - (eoi_pos + 2)) if has_eoi else n
    bytes_after_eoi_ratio_permille = int(bytes_after_eoi * 1000 / n) if n > 0 else 0

    tail = data[max(0, n - min(4096, n)) :]
    tail_zip, tail_pdf, tail_elf = _tail_has_magic(tail)

    app_count = 0
    app_bytes = 0
    app_max_len = 0
    com_count = 0
    com_bytes = 0
    dqt_count = 0
    dqt_bytes = 0
    dqt_max_len = 0
    dht_count = 0
    dht_bytes = 0
    dht_max_len = 0
    sof0_count = 0
    sof2_count = 0
    sos_count = 0
    max_seg_len = 0

    dqt_tables = 0
    dqt_invalid = 0
    dht_tables = 0
    dht_invalid = 0

    width = 0
    height = 0
    components = 0
    precision = 0
    dri_interval = 0
    invalid_len = 0

    # Markers that start a frame header (SOF*)
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    }

    pos = 2
    while pos < n - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue

        # Skip 0xFF fill bytes
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break

        marker = data[pos]
        pos += 1

        if marker == 0x00:
            continue
        if marker == 0xD9:
            break
        if 0xD0 <= marker <= 0xD7:
            continue

        if 0xE0 <= marker <= 0xEF:
            app_count += 1
        if marker == 0xFE:
            com_count += 1
        if marker == 0xDB:
            dqt_count += 1
        if marker == 0xC4:
            dht_count += 1
        if marker == 0xC0:
            sof0_count += 1
        if marker == 0xC2:
            sof2_count += 1
        if marker == 0xDA:
            sos_count += 1

        if marker in (0xD8, 0x01):
            continue

        seglen = _u16be(data, pos)
        if seglen is None or seglen < 2:
            invalid_len += 1
            break
        seg_end = pos + seglen
        if seg_end > n:
            invalid_len += 1
            break

        if seglen > max_seg_len:
            max_seg_len = seglen

        if 0xE0 <= marker <= 0xEF:
            app_bytes += seglen
            if seglen > app_max_len:
                app_max_len = seglen
        if marker == 0xFE:
            com_bytes += seglen
        if marker == 0xDB:
            dqt_bytes += seglen
            if seglen > dqt_max_len:
                dqt_max_len = seglen
        if marker == 0xC4:
            dht_bytes += seglen
            if seglen > dht_max_len:
                dht_max_len = seglen

        payload_off = pos + 2
        payload_len = seglen - 2

        if marker in sof_markers and width == 0 and height == 0 and payload_len >= 6:
            precision = data[payload_off]
            h = _u16be(data, payload_off + 1)
            w = _u16be(data, payload_off + 3)
            if h is not None:
                height = h
            if w is not None:
                width = w
            components = data[payload_off + 5]

        if marker == 0xDD and payload_len >= 2:
            v = _u16be(data, payload_off)
            if v is not None:
                dri_interval = v

        if marker == 0xDB:
            off = payload_off
            end = payload_off + payload_len
            while off < end:
                if off >= end:
                    break
                pq_tq = data[off]
                off += 1
                pq = (pq_tq >> 4) & 0x0F
                table_bytes = 64 * (2 if pq else 1)
                if off + table_bytes > end:
                    dqt_invalid += 1
                    break
                dqt_tables += 1
                off += table_bytes

        if marker == 0xC4:
            off = payload_off
            end = payload_off + payload_len
            while off < end:
                if off + 1 + 16 > end:
                    dht_invalid += 1
                    break
                off += 1  # tc/th
                counts16 = data[off : off + 16]
                off += 16
                total_syms = int(sum(counts16))
                if off + total_syms > end:
                    dht_invalid += 1
                    break
                dht_tables += 1
                off += total_syms

        if marker == 0xDA:
            break

        pos = seg_end

    dht_bytes_ratio_permille = int(dht_bytes * 1000 / n) if n > 0 else 0
    dqt_bytes_ratio_permille = int(dqt_bytes * 1000 / n) if n > 0 else 0
    app_bytes_ratio_permille = int(app_bytes * 1000 / n) if n > 0 else 0

    return {
        "file_size": float(n),
        "has_eoi": float(has_eoi),
        "bytes_after_eoi": float(bytes_after_eoi),
        "bytes_after_eoi_ratio_permille": float(bytes_after_eoi_ratio_permille),
        "tail_zip_magic": float(tail_zip),
        "tail_pdf_magic": float(tail_pdf),
        "tail_elf_magic": float(tail_elf),
        "invalid_len": float(invalid_len),
        "width": float(width),
        "height": float(height),
        "components": float(components),
        "precision": float(precision),
        "dri_interval": float(dri_interval),
        "app_count": float(app_count),
        "app_bytes": float(app_bytes),
        "app_bytes_ratio_permille": float(app_bytes_ratio_permille),
        "app_max_len": float(app_max_len),
        "com_count": float(com_count),
        "com_bytes": float(com_bytes),
        "dqt_count": float(dqt_count),
        "dqt_bytes": float(dqt_bytes),
        "dqt_bytes_ratio_permille": float(dqt_bytes_ratio_permille),
        "dqt_max_len": float(dqt_max_len),
        "dht_count": float(dht_count),
        "dht_bytes": float(dht_bytes),
        "dht_bytes_ratio_permille": float(dht_bytes_ratio_permille),
        "dht_max_len": float(dht_max_len),
        "sof0_count": float(sof0_count),
        "sof2_count": float(sof2_count),
        "sos_count": float(sos_count),
        "max_seg_len": float(max_seg_len),
        "dqt_tables": float(dqt_tables),
        "dqt_invalid": float(dqt_invalid),
        "dht_tables": float(dht_tables),
        "dht_invalid": float(dht_invalid),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", default="outputs/jpeg_features.npz")
    parser.add_argument("--output-tflite", default="outputs/jpeg_ae.tflite")
    parser.add_argument("--output-meta", default="outputs/jpeg_model_meta.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.8,0.1,0.1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", default="32,8")
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument(
        "--augment-resegment-max",
        type=int,
        default=2000,
        help="Max benign augmented samples to add by splitting combined DQT/DHT segments (0 to disable)",
    )
    parser.add_argument(
        "--augment-appcount-target",
        type=int,
        default=6,
        help="If >0, insert small APP segments during augmentation to reach at least this APP count",
    )
    parser.add_argument(
        "--augment-tailbytes-min",
        type=int,
        default=128,
        help="If >0, append at least this many bytes after EOI during augmentation (0 to disable)",
    )
    parser.add_argument(
        "--augment-tailbytes-max",
        type=int,
        default=256,
        help="Upper bound (inclusive) for bytes appended after EOI during augmentation",
    )
    args = parser.parse_args()

    set_seeds(args.seed)

    splits = tuple(float(x) for x in args.split.split(","))
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]

    data = np.load(args.input_npz, allow_pickle=True)
    X_struct = data["X_struct"].astype(np.float32)
    y = data["y"].astype(np.int64)
    labels = data["labels"]
    struct_names = [str(x) for x in data["struct_feature_names"].tolist()]
    paths = data["paths"]

    benign_orig_mask = y == 0
    if int(np.sum(benign_orig_mask)) == 0:
        raise SystemExit("No benign samples found in input.")

    # Optional benign augmentation: resegment DQT/DHT into multiple segments (mimics common camera encoders).
    aug_max = int(args.augment_resegment_max)
    if aug_max > 0:
        app_idx = struct_names.index("app_count") if "app_count" in struct_names else -1
        app_target = int(args.augment_appcount_target)
        tail_min = max(0, int(args.augment_tailbytes_min))
        tail_max = max(0, int(args.augment_tailbytes_max))
        if tail_min > tail_max:
            tail_min, tail_max = tail_max, tail_min
        benign_indices_all = np.where(benign_orig_mask)[0]
        aug_n = min(aug_max, benign_indices_all.size)
        rng = np.random.RandomState(args.seed + 1337)
        chosen = rng.choice(benign_indices_all, size=aug_n, replace=False)

        aug_struct: List[List[float]] = []
        aug_paths: List[str] = []
        for idx in chosen.tolist():
            p = str(paths[idx])
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "rb") as f:
                    raw = f.read()
                transformed = _resegment_dqt_dht(raw)
                if transformed is None:
                    continue
                if app_target > 0 and app_idx >= 0:
                    orig_app = int(X_struct[idx, app_idx])
                    need = max(0, app_target - orig_app)
                    transformed = _insert_app_segments(transformed, need)
                if tail_min > 0 and tail_max > 0:
                    tail_len = int(rng.randint(tail_min, tail_max + 1))
                    transformed = _append_tail_bytes(transformed, tail_len)
                feat = parse_jpeg_struct_bytes(transformed)
                if feat is None:
                    continue
                aug_struct.append([float(feat.get(name, 0.0)) for name in struct_names])
                aug_paths.append(p + "::resegmented_appaug")
            except Exception:
                continue

        if aug_struct:
            X_struct = np.concatenate([X_struct, np.array(aug_struct, dtype=np.float32)], axis=0)
            y = np.concatenate([y, np.zeros((len(aug_struct),), dtype=np.int64)], axis=0)
            labels = np.concatenate([labels, np.array(["benign_aug"] * len(aug_struct))], axis=0)
            paths = np.concatenate([paths, np.array(aug_paths)], axis=0)

    labels = labels.astype(str)
    is_aug = labels == "benign_aug"
    benign_mask = (y == 0) & (~is_aug)

    if int(np.sum(benign_mask)) == 0:
        raise SystemExit("No original benign samples found after augmentation.")

    # Split ONLY on original benign samples for thresholding and reporting.
    # Augmented samples are used for training but excluded from threshold selection.
    benign_indices = np.where(benign_mask)[0]
    train_idx_b, val_idx_b, test_idx_b = split_indices(int(benign_indices.size), args.seed, splits)
    train_idx = benign_indices[train_idx_b]
    val_idx = benign_indices[val_idx_b]
    test_idx = benign_indices[test_idx_b]

    aug_idx = np.where(is_aug)[0]
    if aug_idx.size > 0:
        train_idx = np.concatenate([train_idx, aug_idx], axis=0)

    log_mask = build_log_mask(struct_names)
    log_idx = np.where(log_mask > 0)[0]
    X_log = X_struct.copy()
    if log_idx.size > 0:
        X_log[:, log_idx] = np.log1p(X_log[:, log_idx])

    mean = X_log[train_idx].mean(axis=0).astype(np.float32)
    std = X_log[train_idx].std(axis=0).astype(np.float32)
    std_safe = np.where(std == 0.0, 1.0, std).astype(np.float32)
    X_norm = (X_log - mean) / std_safe

    ae = build_autoencoder(X_norm.shape[1], hidden)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss="mse")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    ae.fit(
        X_norm[train_idx],
        X_norm[train_idx],
        validation_data=(X_norm[val_idx], X_norm[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    # Export model: raw -> log1p(mask) -> zscore -> ae -> mse
    inputs = tf.keras.Input(shape=(X_struct.shape[1],), name="features_raw")
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
    export_model = tf.keras.Model(inputs=inputs, outputs=mse, name="jpeg_ae_anomaly_score")

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.output_tflite), exist_ok=True)
    with open(args.output_tflite, "wb") as f:
        f.write(tflite_model)

    # Threshold + eval MUST be computed on the exported TFLite model output, because TFLite
    # optimizations (e.g., dynamic range quantization) can shift the anomaly-score scale.
    val_scores = tflite_scores(tflite_model, X_struct[val_idx])
    threshold = float(np.percentile(val_scores, args.threshold_percentile))

    test_scores = tflite_scores(tflite_model, X_struct[test_idx])
    benign_fpr = float(np.mean(test_scores >= threshold))

    malware_mask = y == 1
    malware_scores = tflite_scores(tflite_model, X_struct[malware_mask]) if np.any(malware_mask) else np.zeros((0,))
    malware_recall = float(np.mean(malware_scores >= threshold)) if malware_scores.size > 0 else 0.0

    meta = {
        "model": "jpeg_autoencoder_anomaly_score",
        "feature_dim": int(X_struct.shape[1]),
        "struct_feature_names": struct_names,
        "hidden": hidden,
        "threshold_percentile": float(args.threshold_percentile),
        "threshold": float(threshold),
        "seed": int(args.seed),
        "split": [float(s) for s in splits],
        "eval": {
            "benign_test_fpr": benign_fpr,
            "malware_recall": malware_recall,
            "benign_count": int(np.sum(benign_mask)),
            "benign_aug_count": int(np.sum(is_aug)),
            "malware_count": int(np.sum(malware_mask)),
        },
    }
    os.makedirs(os.path.dirname(args.output_meta), exist_ok=True)
    with open(args.output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", args.output_tflite)
    print("Wrote:", args.output_meta)
    print(f"Threshold (p{args.threshold_percentile:.1f}) = {threshold:.6f}")
    print(f"Benign test FPR = {benign_fpr:.6f}")
    print(f"Malware recall = {malware_recall:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
