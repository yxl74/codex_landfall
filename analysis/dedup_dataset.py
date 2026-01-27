#!/usr/bin/env python3
"""
Deduplicate dataset files across benign_data, LandFall, and general_mal.

Policy:
  - Keep one copy per hash.
  - Prefer keeping in LandFall > general_mal > benign_data.
  - Move duplicates into data/_duplicates/<category>/<hash>/...
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


PRIORITY = {"LandFall": 2, "general_mal": 1, "benign_data": 0}


def compute_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def category_for_path(path: str, data_root: str) -> str:
    rel = os.path.relpath(path, data_root)
    top = rel.split(os.sep)[0]
    return top if top in PRIORITY else "other"


def iter_files(root: str, subdirs: List[str]) -> List[str]:
    files: List[str] = []
    for sub in subdirs:
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                if os.path.isfile(path):
                    files.append(path)
    return files


def choose_keep(paths: List[str], data_root: str) -> str:
    def key(p: str) -> Tuple[int, int, str]:
        cat = category_for_path(p, data_root)
        priority = PRIORITY.get(cat, -1)
        return (priority, -len(p), p)

    return max(paths, key=key)


def unique_path(dest_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate = os.path.join(dest_dir, f"{base}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


@dataclass
class DedupStats:
    total_files: int = 0
    duplicate_groups: int = 0
    moved_files: int = 0
    kept_by_category: Dict[str, int] = None
    moved_by_category: Dict[str, int] = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dup-dir", default="data/_duplicates")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subdirs = ["benign_data", "LandFall", "general_mal"]
    files = iter_files(args.data_root, subdirs)
    hashes: Dict[str, List[str]] = defaultdict(list)
    for path in files:
        try:
            h = compute_hash(path)
        except OSError:
            continue
        hashes[h].append(path)

    kept_by_category = Counter()
    moved_by_category = Counter()
    moved = 0
    groups = 0

    for h, paths in hashes.items():
        if len(paths) <= 1:
            cat = category_for_path(paths[0], args.data_root)
            kept_by_category[cat] += 1
            continue
        groups += 1
        keep = choose_keep(paths, args.data_root)
        kept_by_category[category_for_path(keep, args.data_root)] += 1
        for path in sorted(paths):
            if path == keep:
                continue
            cat = category_for_path(path, args.data_root)
            dest_dir = os.path.join(args.dup_dir, cat, h)
            dest = unique_path(dest_dir, os.path.basename(path))
            moved_by_category[cat] += 1
            moved += 1
            if args.dry_run:
                continue
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(path, dest)

    stats = DedupStats(
        total_files=len(files),
        duplicate_groups=groups,
        moved_files=moved,
        kept_by_category=dict(kept_by_category),
        moved_by_category=dict(moved_by_category),
    )

    print("Total files:", stats.total_files)
    print("Duplicate groups:", stats.duplicate_groups)
    print("Moved duplicates:", stats.moved_files)
    print("Kept by category:", stats.kept_by_category)
    print("Moved by category:", stats.moved_by_category)
    if args.dry_run:
        print("Dry-run only; no files moved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
