#!/usr/bin/env python3
"""
Quick investigation helpers for JPEG benign vs malware datasets.

Usage:
  python3 analysis/jpeg_feature_extract.py --output-dir outputs
  python3 analysis/jpeg_investigate.py --npz outputs/jpeg_features.npz
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np


def _q(v: np.ndarray, p: float) -> float:
    if v.size == 0:
        return 0.0
    vs = np.sort(v)
    idx = int(p * (len(vs) - 1))
    return float(vs[idx])


def summarize_feature(name: str, ben: np.ndarray, mal: np.ndarray) -> None:
    print(f"\n{name}")
    for label, v in (("ben", ben), ("mal", mal)):
        if v.size == 0:
            continue
        nz = float(np.mean(v != 0.0) * 100.0)
        print(
            f"{label:3s} min={v.min():.6g} p50={_q(v,0.5):.6g} p90={_q(v,0.9):.6g} "
            f"p99={_q(v,0.99):.6g} max={v.max():.6g} nz%={nz:.3g}"
        )


def rule_stats(y: np.ndarray, hit: np.ndarray, name: str) -> None:
    ben = (y == 0)
    mal = (y == 1)
    ben_fp = int(np.sum(hit & ben))
    ben_total = int(np.sum(ben))
    mal_tp = int(np.sum(hit & mal))
    mal_total = int(np.sum(mal))
    fpr = ben_fp / max(1, ben_total)
    tpr = mal_tp / max(1, mal_total)
    print(f"{name}: benign_fp={ben_fp}/{ben_total} fpr={fpr:.6g}  mal_tp={mal_tp}/{mal_total} tpr={tpr:.6g}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="outputs/jpeg_features.npz", help="NPZ produced by jpeg_feature_extract.py")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X = data["X_struct"].astype(np.float32)
    y = data["y"].astype(np.int64)
    names = [str(x) for x in data["struct_feature_names"].tolist()]

    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(names)}

    def col(n: str) -> np.ndarray:
        return X[:, name_to_idx[n]]

    ben = (y == 0)
    mal = (y == 1)

    print("samples:", len(y), "benign:", int(ben.sum()), "malware:", int(mal.sum()), "dims:", X.shape[1])

    keys: List[str] = [
        "file_size",
        "has_eoi",
        "bytes_after_eoi",
        "bytes_after_eoi_ratio_permille",
        "dht_bytes",
        "dqt_bytes",
        "dht_tables",
        "dqt_tables",
        "dht_invalid",
        "dqt_invalid",
        "sof2_count",
        "invalid_len",
        "tail_zip_magic",
    ]

    for k in keys:
        if k not in name_to_idx:
            continue
        summarize_feature(k, col(k)[ben], col(k)[mal])

    # Candidate rules (tuned on current dataset for investigation only)
    after_ratio_permille = col("bytes_after_eoi_ratio_permille")
    after_bytes = col("bytes_after_eoi")
    dht_bytes = col("dht_bytes")
    dqt_bytes = col("dqt_bytes")
    tail_zip = col("tail_zip_magic") > 0.5
    invalid_len = col("invalid_len") > 0.5

    print("\nRule checks (dataset-specific):")
    rule_stats(y, after_ratio_permille > 10, "after_ratio_permille>10 ( >1% )")
    rule_stats(y, after_bytes > 20000, "bytes_after_eoi>20000")
    rule_stats(y, tail_zip, "tail_zip_magic==1")
    rule_stats(y, invalid_len, "invalid_len>0")

    # Header table size anomalies
    rule_stats(y, dht_bytes != 418, "dht_bytes!=418")
    rule_stats(y, dqt_bytes > 500, "dqt_bytes>500")

    # A conservative union intended to be low-FP on this dataset
    union = (after_ratio_permille > 10) | (after_bytes > 20000) | tail_zip | invalid_len | (dqt_bytes > 500)
    rule_stats(y, union, "UNION(after_ratio|after_bytes|tail_zip|invalid|dqt>500)")

    print("\nNotes:")
    print("- JPG_dataset appears to be mostly large camera photos; size/metadata bias is likely.")
    print("- dht_bytes/dqt_bytes are unusually constant in JPG_dataset; expect more variance in real-world JPEGs.")
    print("- Treat these rules as investigation starting points, not production thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

