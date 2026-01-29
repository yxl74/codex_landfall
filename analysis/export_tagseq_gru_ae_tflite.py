#!/usr/bin/env python3
"""Export tag-sequence GRU-AE Keras model to TFLite."""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", default="outputs/tagseq_gru_ae.keras")
    ap.add_argument("--out", default="outputs/tagseq_gru_ae.tflite")
    args = ap.parse_args()

    # These models are trained locally; allow loading Lambda-free and (older) Lambda-based artifacts.
    model = tf.keras.models.load_model(args.keras, safe_mode=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.experimental_enable_resource_variables = True
    tflite = converter.convert()
    Path(args.out).write_bytes(tflite)
    print(f"Wrote {args.out} ({len(tflite)} bytes)")


if __name__ == "__main__":
    main()
