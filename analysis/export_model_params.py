#!/usr/bin/env python3
"""
Export hybrid model parameters to JSON for on-device feature attribution.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-npz", default="outputs/hybrid_model.npz")
    parser.add_argument(
        "--output-json",
        default="android/HybridDetectorApp/app/src/main/assets/hybrid_model_params.json",
    )
    args = parser.parse_args()

    data = np.load(args.model_npz, allow_pickle=True)
    w = data["w"].astype(np.float32).tolist()
    b = float(data["b"])
    mean = data["mean"].astype(np.float32).tolist()
    std = data["std"].astype(np.float32).tolist()
    bytes_mode = str(data["bytes_mode"])
    struct_feature_names = [str(x) for x in data.get("struct_feature_names", [])]

    payload = {
        "bytes_mode": bytes_mode,
        "feature_dim": len(w),
        "struct_feature_names": struct_feature_names,
        "weights": w,
        "bias": b,
        "mean": mean,
        "std": std,
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("Wrote:", args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
