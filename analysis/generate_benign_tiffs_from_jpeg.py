#!/usr/bin/env python3
"""
Generate a benign non-DNG TIFF corpus by converting a sampled subset of benign JPEGs.

Why: the current anomaly AE training set is dominated by benign DNG (which will be routed to
TagSeq GRU-AE). This script creates a larger benign TIFF (non-DNG) dataset under
data/benign_data/ so it is automatically picked up by analysis/anomaly_feature_extract.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_COMPRESSIONS = (
    # Common "photo" TIFF
    "tiff_lzw",
    "tiff_deflate",
    "tiff_adobe_deflate",
    "packbits",
    "raw",
    # JPEG-in-TIFF (exercises libjpeg paths in many stacks)
    "tiff_jpeg",
    # Document-style bilevel compressions
    "group3",
    "group4",
)
DEFAULT_MAX_SIDES = (256, 512, 768, 1024, 1536)
DEFAULT_MODES = (
    # Color + grayscale photos
    "RGB",
    "L",
    # Bilevel (document workflows)
    "1",
    # 16-bit grayscale (scientific-ish)
    "I;16",
)


@dataclass(frozen=True)
class Variant:
    compression: str
    mode: str
    max_side: int


def iter_jpegs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in (".jpg", ".jpeg"):
            yield p


def stable_id(path: Path, variant: Variant) -> str:
    h = hashlib.sha1()
    h.update(str(path).encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(f"{variant.compression}|{variant.mode}|{variant.max_side}".encode("utf-8"))
    return h.hexdigest()[:12]


def pick_variant(rng: random.Random, compressions: Sequence[str], modes: Sequence[str], max_sides: Sequence[int]) -> Variant:
    compression = rng.choice(list(compressions))
    mode = rng.choice(list(modes))

    # Constrain mode/compression combos to what PIL can reliably write.
    if compression in ("group3", "group4"):
        mode = "1"
    if compression in ("tiff_jpeg", "jpeg"):
        if mode not in ("RGB", "L"):
            mode = rng.choice(["RGB", "L"])

    return Variant(compression=compression, mode=mode, max_side=int(rng.choice(list(max_sides))))


def downscale(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def to_mode(im: Image.Image, mode: str) -> Image.Image:
    if mode == "RGB":
        return im.convert("RGB")
    if mode == "L":
        return im.convert("L")
    if mode == "1":
        # Bilevel via thresholding the luminance channel.
        gray = im.convert("L")
        arr = np.array(gray, dtype=np.uint8)
        thr = int(np.percentile(arr, 60))
        bw = (arr > thr).astype(np.uint8) * 255
        return Image.fromarray(bw, mode="L").convert("1")
    if mode == "I;16":
        gray = im.convert("L")
        arr = np.array(gray, dtype=np.uint8).astype(np.uint16)
        arr16 = arr * 257  # 0..255 -> 0..65535
        return Image.fromarray(arr16, mode="I;16")
    raise ValueError(f"Unsupported mode: {mode}")


def make_pages(base: Image.Image, max_pages: int, rng: random.Random) -> List[Image.Image]:
    pages: List[Image.Image] = [base]
    if max_pages <= 1:
        return pages
    n = rng.randint(2, max_pages)
    ops = [
        lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        lambda im: im.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        lambda im: im.rotate(90, expand=True),
        lambda im: im.rotate(180, expand=True),
        lambda im: im.rotate(270, expand=True),
    ]
    for _ in range(n - 1):
        op = rng.choice(ops)
        pages.append(op(base))
    return pages


def save_tiff_pages(pages: List[Image.Image], dst: Path, compression: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if len(pages) == 1:
        pages[0].save(dst, format="TIFF", compression=compression)
        return
    first, rest = pages[0], pages[1:]
    first.save(dst, format="TIFF", compression=compression, save_all=True, append_images=rest)


def convert_one(src: Path, dst: Path, variant: Variant, multipage_prob: float, multipage_max_pages: int, rng: random.Random) -> Tuple[int, int, int]:
    with Image.open(src) as im:
        im = to_mode(im, variant.mode)

        im = downscale(im, variant.max_side)
        w, h = im.size

        pages = [im]
        if multipage_max_pages > 1 and rng.random() < multipage_prob:
            pages = make_pages(im, multipage_max_pages, rng)

        save_tiff_pages(pages, dst, variant.compression)
        return w, h, len(pages)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jpeg-root", default="JPG_dataset")
    parser.add_argument("--output-dir", default="data/benign_data/generated_tiff_from_jpeg")
    parser.add_argument("--count", type=int, default=2500, help="Number of TIFFs to generate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compressions", default=",".join(DEFAULT_COMPRESSIONS))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--max-sides", default=",".join(str(x) for x in DEFAULT_MAX_SIDES))
    parser.add_argument("--multipage-prob", type=float, default=0.15)
    parser.add_argument("--multipage-max-pages", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jpeg_root = Path(args.jpeg_root)
    out_root = Path(args.output_dir)

    compressions = [x.strip() for x in str(args.compressions).split(",") if x.strip()]
    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    max_sides = [int(x) for x in str(args.max_sides).split(",") if str(x).strip()]
    if not jpeg_root.exists():
        raise SystemExit(f"Missing jpeg root: {jpeg_root}")
    if not compressions or not modes or not max_sides:
        raise SystemExit("compressions/modes/max-sides must be non-empty")

    rng = random.Random(args.seed)
    jpeg_paths: List[Path] = list(iter_jpegs(jpeg_root))
    if not jpeg_paths:
        raise SystemExit(f"No JPEGs found under: {jpeg_root}")
    rng.shuffle(jpeg_paths)

    manifest_path = out_root / "manifest.csv"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    ok = 0
    skipped = 0
    failed = 0

    for src in jpeg_paths:
        if ok >= args.count:
            break
        variant = pick_variant(rng, compressions, modes, max_sides)
        sid = stable_id(src, variant)
        subdir = f"comp={variant.compression}_mode={variant.mode}_max={variant.max_side}"
        dst = out_root / subdir / f"{sid}_{src.stem}.tiff"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            w, h, pages = convert_one(
                src,
                dst,
                variant,
                multipage_prob=args.multipage_prob,
                multipage_max_pages=args.multipage_max_pages,
                rng=rng,
            )
        except Exception:
            failed += 1
            continue
        ok += 1
        rows.append(
            {
                "src": str(src),
                "dst": str(dst),
                "compression": variant.compression,
                "mode": variant.mode,
                "max_side": str(variant.max_side),
                "width": str(w),
                "height": str(h),
                "pages": str(pages),
                "dst_bytes": str(dst.stat().st_size),
            }
        )

    if rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Generated: {ok}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Wrote:     {manifest_path}")
    print(f"Output:    {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
