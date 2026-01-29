#!/usr/bin/env python3
"""Build deterministic train/holdout lists for DNG tag-sequence modeling."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.tag_sequence_extract import parse_tag_sequence


def iter_dng_candidates(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".dng"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign-root", default="data/benign_data")
    ap.add_argument("--landfall-root", default="data/LandFall")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument(
        "--holdout-mode",
        choices=["random", "vendor"],
        default="random",
        help="How to create holdout split. 'vendor' reduces leakage for raw_pixls by holding out camera makers.",
    )
    ap.add_argument("--out-train", default="outputs/dng_tagseq_train_list.txt")
    ap.add_argument("--out-holdout", default="outputs/dng_tagseq_holdout_list.txt")
    ap.add_argument("--out-landfall", default="outputs/dng_tagseq_landfall_list.txt")
    args = ap.parse_args()

    benign_root = Path(args.benign_root)
    landfall_root = Path(args.landfall_root)

    candidates = iter_dng_candidates(benign_root)
    dng_files: List[str] = []
    for path in sorted(candidates):
        seq = parse_tag_sequence(str(path))
        if seq is None or not seq.has_dng:
            continue
        dng_files.append(str(path))

    if len(dng_files) <= args.holdout:
        raise SystemExit("Not enough DNG files for requested holdout")

    random.seed(args.seed)
    if args.holdout_mode == "random":
        random.shuffle(dng_files)
        holdout = sorted(dng_files[: args.holdout])
        train = sorted(dng_files[args.holdout :])
    else:
        # Group holdout by vendor to avoid splitting the same camera/model distribution
        # across train/holdout. This is best-effort and relies on benign_root/raw_pixls/<vendor>/...
        by_group = {}
        for p in dng_files:
            rel = Path(p).relative_to(benign_root)
            parts = rel.parts
            if len(parts) >= 2 and parts[0] == "raw_pixls":
                group = f"raw_pixls/{parts[1]}"
            else:
                group = parts[0] if parts else "unknown"
            by_group.setdefault(group, []).append(p)

        groups = sorted(by_group.keys())
        random.shuffle(groups)
        holdout_groups = []
        holdout_files: List[str] = []
        for g in groups:
            if len(holdout_files) >= args.holdout:
                break
            holdout_groups.append(g)
            holdout_files.extend(by_group[g])

        holdout = sorted(holdout_files)
        train = sorted([p for g, files in by_group.items() if g not in holdout_groups for p in files])

    landfall = sorted(str(p) for p in landfall_root.rglob("*") if p.is_file())

    Path(args.out_train).write_text("\n".join(train) + "\n")
    Path(args.out_holdout).write_text("\n".join(holdout) + "\n")
    Path(args.out_landfall).write_text("\n".join(landfall) + "\n")

    print(f"DNG files: {len(dng_files)}")
    print(f"Train: {len(train)} Holdout: {len(holdout)}")
    print(f"LandFall: {len(landfall)}")


if __name__ == "__main__":
    main()
