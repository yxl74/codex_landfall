#!/usr/bin/env python3
"""Sweep thresholds for tag-sequence GRU-AE reconstruction errors."""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path


def load_errors(npz_path: str, model_path: str, batch_size: int) -> np.ndarray:
    import tensorflow as tf
    data = np.load(npz_path, allow_pickle=True)
    x = data["features"]
    tag_ids = data["tag_ids"].astype(np.int32)
    type_ids = data["type_ids"].astype(np.int32)
    ifd_kinds = data["ifd_kinds"].astype(np.int32)
    lengths = data["lengths"].astype(np.int32)
    max_len = x.shape[1]

    idx = np.arange(max_len)[None, :]
    mask = (idx < lengths[:, None]).astype(np.float32)

    model = tf.keras.models.load_model(model_path)
    pred = model.predict([x, tag_ids, type_ids, ifd_kinds], batch_size=batch_size, verbose=0)

    diff = (x - pred) ** 2
    diff = diff.sum(axis=-1) * mask
    denom = mask.sum(axis=1) + 1e-8
    err = diff.sum(axis=1) / denom
    return err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/tagseq_gru_ae.keras")
    ap.add_argument("--holdout", default="outputs/tagseq_dng_holdout.npz")
    ap.add_argument("--landfall", default="outputs/tagseq_dng_landfall.npz")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pcts", default="98,98.5,99,99.5,99.9")
    ap.add_argument("--out", default="outputs/tagseq_gru_ae_threshold_sweep.json")
    args = ap.parse_args()

    hold_err = load_errors(args.holdout, args.model, args.batch_size)
    land_err = load_errors(args.landfall, args.model, args.batch_size)

    pcts = [float(x) for x in args.pcts.split(",") if x.strip()]
    results = []
    for p in pcts:
        thr = float(np.percentile(hold_err, p))
        fpr = float(np.mean(hold_err >= thr))
        recall = float(np.mean(land_err >= thr))
        results.append({
            "percentile": p,
            "threshold": thr,
            "holdout_fpr": fpr,
            "landfall_recall": recall,
        })

    payload = {
        "holdout_files": int(len(hold_err)),
        "landfall_files": int(len(land_err)),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
