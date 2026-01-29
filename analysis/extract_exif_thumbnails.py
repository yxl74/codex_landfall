#!/usr/bin/env python3
"""
Extract embedded Exif JPEG thumbnails from camera JPEGs.

Why: JPG_dataset appears to be entirely camera photos with large Exif APP1 blocks.
The embedded Exif thumbnails provide a cheap way to create a benign set of *small*
JPEGs (2–10KB range) without any image decoding or re-encoding, which helps reduce
file-size/metadata bias when training JPEG detectors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import mmap
import os
import struct
from typing import Iterator, Optional, Tuple


def iter_paths(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def _u16be(mm: mmap.mmap, off: int) -> Optional[int]:
    if off + 2 > len(mm):
        return None
    return (mm[off] << 8) | mm[off + 1]


def _read_u16(mm: mmap.mmap, off: int, little_endian: bool) -> Optional[int]:
    if off + 2 > len(mm):
        return None
    fmt = "<H" if little_endian else ">H"
    return struct.unpack_from(fmt, mm, off)[0]


def _read_u32(mm: mmap.mmap, off: int, little_endian: bool) -> Optional[int]:
    if off + 4 > len(mm):
        return None
    fmt = "<I" if little_endian else ">I"
    return struct.unpack_from(fmt, mm, off)[0]


def extract_exif_thumbnail(path: str) -> Optional[bytes]:
    """Return embedded thumbnail bytes if present (JPEG SOI..), else None."""
    size = os.path.getsize(path)
    if size < 4:
        return None

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if mm[0:2] != b"\xff\xd8":
                return None

            pos = 2
            while pos < size - 1:
                if mm[pos] != 0xFF:
                    pos += 1
                    continue

                while pos < size and mm[pos] == 0xFF:
                    pos += 1
                if pos >= size:
                    break

                marker = mm[pos]
                pos += 1

                if marker == 0x00:
                    continue
                if marker == 0xD9:
                    break
                if marker in (0xD8, 0x01) or (0xD0 <= marker <= 0xD7):
                    continue

                seglen = _u16be(mm, pos)
                if seglen is None or seglen < 2:
                    return None
                seg_end = pos + seglen
                if seg_end > size:
                    return None

                payload_off = pos + 2
                payload_len = seglen - 2

                # APP1 Exif
                if marker == 0xE1 and payload_len > 14 and mm[payload_off : payload_off + 6] == b"Exif\x00\x00":
                    tiff_off = payload_off + 6
                    endian = mm[tiff_off : tiff_off + 2]
                    if endian not in (b"II", b"MM"):
                        return None
                    le = endian == b"II"

                    magic = _read_u16(mm, tiff_off + 2, le)
                    if magic != 42:
                        return None
                    ifd0_rel = _read_u32(mm, tiff_off + 4, le)
                    if ifd0_rel is None or ifd0_rel == 0:
                        return None

                    # Bounds within APP1 payload
                    payload_end = payload_off + payload_len

                    ifd0 = tiff_off + ifd0_rel
                    if ifd0 + 2 > payload_end:
                        return None
                    num0 = _read_u16(mm, ifd0, le)
                    if num0 is None:
                        return None
                    ifd0_entries_end = ifd0 + 2 + num0 * 12
                    if ifd0_entries_end + 4 > payload_end:
                        return None

                    ifd1_rel = _read_u32(mm, ifd0_entries_end, le)
                    if ifd1_rel is None or ifd1_rel == 0:
                        return None

                    ifd1 = tiff_off + ifd1_rel
                    if ifd1 + 2 > payload_end:
                        return None
                    num1 = _read_u16(mm, ifd1, le)
                    if num1 is None:
                        return None

                    thumb_off_rel: Optional[int] = None
                    thumb_len: Optional[int] = None
                    for i in range(num1):
                        ent = ifd1 + 2 + i * 12
                        if ent + 12 > payload_end:
                            break
                        tag = _read_u16(mm, ent, le)
                        val = _read_u32(mm, ent + 8, le)
                        if tag is None or val is None:
                            continue
                        if tag == 0x0201:  # JPEGInterchangeFormat
                            thumb_off_rel = val
                        elif tag == 0x0202:  # JPEGInterchangeFormatLength
                            thumb_len = val

                    if thumb_off_rel is None or thumb_len is None:
                        return None

                    thumb_abs = tiff_off + thumb_off_rel
                    if thumb_abs < payload_off or thumb_abs + thumb_len > payload_end:
                        return None

                    thumb = bytes(mm[thumb_abs : thumb_abs + thumb_len])
                    if not thumb.startswith(b"\xff\xd8"):
                        return None
                    return thumb

                if marker == 0xDA:
                    break

                pos = seg_end

            return None
        finally:
            mm.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="JPG_dataset", help="Root directory of camera JPEGs")
    parser.add_argument("--output-dir", default="outputs/jpg_thumbnails", help="Directory to write extracted thumbnails")
    parser.add_argument("--output-csv", default="outputs/jpg_thumbnails.csv", help="CSV mapping inputs to outputs")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images to process")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    rows = []
    seen = set()

    processed = 0
    extracted = 0

    for p in iter_paths(args.input_root):
        processed += 1
        thumb = extract_exif_thumbnail(p)
        if thumb is not None:
            h = hashlib.sha256(thumb).hexdigest()
            out_path = os.path.join(args.output_dir, f"{h}.jpg")
            if h not in seen and not os.path.exists(out_path):
                with open(out_path, "wb") as out:
                    out.write(thumb)
                seen.add(h)
            extracted += 1
            rows.append(
                {
                    "source_path": p,
                    "thumb_sha256": h,
                    "thumb_path": out_path,
                    "thumb_size": str(len(thumb)),
                }
            )

        if args.limit and processed >= args.limit:
            break

    if rows:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("Processed:", processed)
    print("Extracted thumbnails:", extracted)
    print("Unique thumbnails:", len(seen))
    print("Wrote:", args.output_dir)
    print("Wrote:", args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

