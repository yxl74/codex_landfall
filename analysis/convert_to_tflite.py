#!/usr/bin/env python3
"""
Convert the trained hybrid logistic regression model to TFLite.

Requires TensorFlow (use .venv-tf).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-npz", default="outputs/hybrid_model.npz")
    parser.add_argument("--output-tflite", default="outputs/hybrid_model.tflite")
    parser.add_argument("--output-meta", default="outputs/hybrid_model_meta.json")
    args = parser.parse_args()

    data = np.load(args.model_npz, allow_pickle=True)
    w = data["w"].astype(np.float32)
    b = float(data["b"])
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    bytes_mode = str(data["bytes_mode"])
    struct_feature_names = [str(x) for x in data.get("struct_feature_names", [])]

    d = w.shape[0]
    assert mean.shape[0] == d and std.shape[0] == d, "mean/std size mismatch"

    # Build log1p mask for structural features (bytes features are left as-is).
    if bytes_mode == "hist":
        bytes_dim = 514
    elif bytes_mode == "raw":
        bytes_dim = 2048
    else:
        raise ValueError(f"Unknown bytes_mode: {bytes_mode}")

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
    mask = np.zeros((d,), dtype=np.float32)
    for i, name in enumerate(struct_feature_names):
        if name in log_fields:
            mask[bytes_dim + i] = 1.0

    inputs = tf.keras.Input(shape=(d,), dtype=tf.float32, name="features")
    mask_tf = tf.constant(mask, dtype=tf.float32)
    x_log = tf.keras.layers.Lambda(lambda t: tf.math.log1p(t), name="log1p")(inputs)
    x = inputs * (1.0 - mask_tf) + x_log * mask_tf
    x = (x - tf.constant(mean)) / tf.constant(std)
    dense = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True, name="logreg")
    outputs = dense(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Set weights
    kernel = w.reshape(d, 1)
    bias = np.array([b], dtype=np.float32)
    dense.set_weights([kernel, bias])

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.output_tflite), exist_ok=True)
    with open(args.output_tflite, "wb") as f:
        f.write(tflite_model)

    meta = {
        "feature_dim": d,
        "bytes_mode": bytes_mode,
        "struct_feature_names": struct_feature_names,
    }
    with open(args.output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", args.output_tflite)
    print("Wrote:", args.output_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
