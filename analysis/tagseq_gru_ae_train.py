#!/usr/bin/env python3
"""Train a GRU autoencoder on DNG tag-sequence features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit("TensorFlow is required. Use the .venv-tf environment.") from exc


def load_npz(path: str) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def build_model(
    max_len: int,
    feature_dim: int,
    tag_vocab: int,
    type_vocab: int,
    ifd_vocab: int,
    tag_emb_dim: int,
    type_emb_dim: int,
    ifd_emb_dim: int,
    gru_units: int,
    latent_dim: int,
) -> tf.keras.Model:
    feat_in = tf.keras.Input(shape=(max_len, feature_dim), name="features")
    tag_in = tf.keras.Input(shape=(max_len,), dtype="int32", name="tag_ids")
    type_in = tf.keras.Input(shape=(max_len,), dtype="int32", name="type_ids")
    ifd_in = tf.keras.Input(shape=(max_len,), dtype="int32", name="ifd_kinds")

    tag_emb = tf.keras.layers.Embedding(
        tag_vocab,
        tag_emb_dim,
        mask_zero=True,
        name="tag_emb",
    )(tag_in)
    type_emb = tf.keras.layers.Embedding(
        type_vocab,
        type_emb_dim,
        mask_zero=False,
        name="type_emb",
    )(type_in)
    ifd_emb = tf.keras.layers.Embedding(
        ifd_vocab,
        ifd_emb_dim,
        mask_zero=False,
        name="ifd_emb",
    )(ifd_in)

    merged = tf.keras.layers.Concatenate(name="concat")([feat_in, tag_emb, type_emb, ifd_emb])

    encoded, state = tf.keras.layers.GRU(
        gru_units,
        return_state=True,
        name="encoder_gru",
    )(merged)
    latent = tf.keras.layers.Dense(latent_dim, activation="tanh", name="latent")(state)

    repeated = tf.keras.layers.RepeatVector(max_len, name="repeat")(latent)
    decoded = tf.keras.layers.GRU(
        gru_units,
        return_sequences=True,
        name="decoder_gru",
    )(repeated)
    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(feature_dim),
        name="recon",
    )(decoded)

    model = tf.keras.Model(inputs=[feat_in, tag_in, type_in, ifd_in], outputs=out)
    return model


def build_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    idx = np.arange(max_len)[None, :]
    mask = (idx < lengths[:, None]).astype(np.float32)
    return mask


def reconstruction_error(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = (y_true - y_pred) ** 2
    diff = diff.sum(axis=-1)
    diff = diff * mask
    denom = mask.sum(axis=1) + 1e-8
    return diff.sum(axis=1) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", default="outputs/tagseq_dng_train.npz")
    ap.add_argument("--holdout-npz", default="outputs/tagseq_dng_holdout.npz")
    ap.add_argument("--landfall-npz", default="outputs/tagseq_dng_landfall.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tag-emb-dim", type=int, default=16)
    ap.add_argument("--type-emb-dim", type=int, default=4)
    ap.add_argument("--ifd-emb-dim", type=int, default=4)
    ap.add_argument("--gru-units", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--threshold-percentile", type=float, default=99.0)
    ap.add_argument("--out-model", default="outputs/tagseq_gru_ae.keras")
    ap.add_argument("--out-metrics", default="outputs/tagseq_gru_ae_metrics.json")
    ap.add_argument("--export-tflite", default="")
    args = ap.parse_args()

    train = load_npz(args.train_npz)
    holdout = load_npz(args.holdout_npz)
    landfall = load_npz(args.landfall_npz)

    x_train = train["features"]
    tag_train = train["tag_ids"].astype(np.int32)
    type_train = train["type_ids"].astype(np.int32)
    ifd_train = train["ifd_kinds"].astype(np.int32)
    len_train = train["lengths"].astype(np.int32)

    x_hold = holdout["features"]
    tag_hold = holdout["tag_ids"].astype(np.int32)
    type_hold = holdout["type_ids"].astype(np.int32)
    ifd_hold = holdout["ifd_kinds"].astype(np.int32)
    len_hold = holdout["lengths"].astype(np.int32)

    x_land = landfall["features"]
    tag_land = landfall["tag_ids"].astype(np.int32)
    type_land = landfall["type_ids"].astype(np.int32)
    ifd_land = landfall["ifd_kinds"].astype(np.int32)
    len_land = landfall["lengths"].astype(np.int32)

    max_len = x_train.shape[1]
    feature_dim = x_train.shape[2]
    tag_vocab = int(max(tag_train.max(), tag_hold.max(), tag_land.max()) + 1)
    type_vocab = int(max(type_train.max(), type_hold.max(), type_land.max()) + 1)
    ifd_vocab = int(max(ifd_train.max(), ifd_hold.max(), ifd_land.max()) + 1)

    model = build_model(
        max_len=max_len,
        feature_dim=feature_dim,
        tag_vocab=tag_vocab,
        type_vocab=type_vocab,
        ifd_vocab=ifd_vocab,
        tag_emb_dim=args.tag_emb_dim,
        type_emb_dim=args.type_emb_dim,
        ifd_emb_dim=args.ifd_emb_dim,
        gru_units=args.gru_units,
        latent_dim=args.latent_dim,
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    train_mask = build_mask(len_train, max_len)
    hold_mask = build_mask(len_hold, max_len)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        [x_train, tag_train, type_train, ifd_train],
        x_train,
        sample_weight=train_mask,
        validation_data=([
            x_hold,
            tag_hold,
            type_hold,
            ifd_hold,
        ], x_hold, hold_mask),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    hold_pred = model.predict([x_hold, tag_hold, type_hold, ifd_hold], batch_size=args.batch_size, verbose=0)
    land_pred = model.predict([x_land, tag_land, type_land, ifd_land], batch_size=args.batch_size, verbose=0)

    hold_err = reconstruction_error(x_hold, hold_pred, hold_mask)
    land_mask = build_mask(len_land, max_len)
    land_err = reconstruction_error(x_land, land_pred, land_mask)

    threshold = float(np.percentile(hold_err, args.threshold_percentile))
    hold_fpr = float(np.mean(hold_err >= threshold))
    land_recall = float(np.mean(land_err >= threshold))

    metrics = {
        "train_files": int(len(len_train)),
        "holdout_files": int(len(len_hold)),
        "landfall_files": int(len(len_land)),
        "max_seq_len": int(max_len),
        "feature_dim": int(feature_dim),
        "tag_vocab": int(tag_vocab),
        "type_vocab": int(type_vocab),
        "ifd_vocab": int(ifd_vocab),
        "threshold_percentile": float(args.threshold_percentile),
        "threshold": threshold,
        "holdout_fpr": hold_fpr,
        "landfall_recall": land_recall,
        "holdout_error_mean": float(np.mean(hold_err)),
        "holdout_error_p95": float(np.percentile(hold_err, 95)),
        "landfall_error_mean": float(np.mean(land_err)),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }

    Path(args.out_metrics).write_text(json.dumps(metrics, indent=2))
    model.save(args.out_model)

    if args.export_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite = converter.convert()
        Path(args.export_tflite).write_bytes(tflite)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
