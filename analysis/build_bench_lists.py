#!/usr/bin/env python3
"""
Build training/benchmark lists with a clean holdout split.

Benchmark policy:
  - LandFall: all samples (unique by hash).
  - general_mal: random sample by hash.
  - benign: DNG files only, random sample by hash.
  - All bench hashes are excluded from training lists (no leakage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


SUBDIRS = ("benign_data", "LandFall", "general_mal")


def compute_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: str) -> List[str]:
    files: List[str] = []
    for sub in SUBDIRS:
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                if os.path.isfile(path):
                    files.append(path)
    return files


def category_for_path(path: str, data_root: str) -> str:
    rel = os.path.relpath(path, data_root)
    return rel.split(os.sep)[0]


def is_dng(path: str) -> bool:
    return path.lower().endswith(".dng")


def choose_rep_path(paths: List[str]) -> str:
    return sorted(paths, key=lambda p: (len(p), p))[0]


def build_hash_index(paths: List[str], data_root: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    hash_to_paths: Dict[str, List[str]] = defaultdict(list)
    hash_to_cat: Dict[str, str] = {}
    for path in paths:
        h = compute_hash(path)
        hash_to_paths[h].append(path)
        cat = category_for_path(path, data_root)
        hash_to_cat[h] = cat
    return hash_to_paths, hash_to_cat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--bench-benign-dng", type=int, default=100)
    parser.add_argument("--bench-general-mal", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    files = iter_files(args.data_root)
    hash_to_paths, hash_to_cat = build_hash_index(files, args.data_root)

    landfall_hashes = [h for h, c in hash_to_cat.items() if c == "LandFall"]
    general_hashes = [h for h, c in hash_to_cat.items() if c == "general_mal"]
    benign_hashes = [h for h, c in hash_to_cat.items() if c == "benign_data"]

    benign_dng_hashes = []
    for h in benign_hashes:
        if any(is_dng(p) for p in hash_to_paths[h]):
            benign_dng_hashes.append(h)

    landfall_hashes = sorted(set(landfall_hashes))
    general_hashes = sorted(set(general_hashes))
    benign_dng_hashes = sorted(set(benign_dng_hashes))

    rng.shuffle(general_hashes)
    rng.shuffle(benign_dng_hashes)

    bench_hashes = set(landfall_hashes)
    bench_general = general_hashes[: min(args.bench_general_mal, len(general_hashes))]
    bench_hashes.update(bench_general)
    bench_benign = benign_dng_hashes[: min(args.bench_benign_dng, len(benign_dng_hashes))]
    bench_hashes.update(bench_benign)

    bench_paths: List[str] = []
    bench_records: List[Dict[str, str]] = []
    for h in sorted(bench_hashes):
        path = choose_rep_path(hash_to_paths[h])
        cat = hash_to_cat[h]
        bench_paths.append(path)
        bench_records.append({"hash": h, "category": cat, "path": path})

    train_paths: List[str] = []
    for h, paths in hash_to_paths.items():
        if h in bench_hashes:
            continue
        train_paths.extend(paths)

    bench_list_path = os.path.join(args.output_dir, "bench_holdout_list.txt")
    train_list_path = os.path.join(args.output_dir, "train_list.txt")
    manifest_path = os.path.join(args.output_dir, "bench_holdout_manifest.json")

    with open(bench_list_path, "w", encoding="utf-8") as f:
        for path in sorted(bench_paths):
            f.write(path + "\n")

    with open(train_list_path, "w", encoding="utf-8") as f:
        for path in sorted(train_paths):
            f.write(path + "\n")

    summary = {
        "seed": args.seed,
        "bench_benign_dng": len(bench_benign),
        "bench_general_mal": len(bench_general),
        "bench_landfall": len(landfall_hashes),
        "train_files": len(train_paths),
        "bench_files": len(bench_paths),
        "bench_list": bench_list_path,
        "train_list": train_list_path,
        "records": bench_records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Bench list:", bench_list_path)
    print("Train list:", train_list_path)
    print("Bench counts:", {"landfall": len(landfall_hashes), "general_mal": len(bench_general), "benign_dng": len(bench_benign)})
    print("Train files:", len(train_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
