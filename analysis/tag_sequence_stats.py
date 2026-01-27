#!/usr/bin/env python3
"""Summarize TIFF tag-structure statistics for tag-sequence modeling."""
from __future__ import annotations

import argparse
import mmap
import os
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TIFF_TYPES = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    6: 1,   # SBYTE
    7: 1,   # UNDEFINED
    8: 2,   # SSHORT
    9: 4,   # SLONG
    10: 8,  # SRATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
}

TAG_DNG_VERSION = 50706
TAG_SUBIFD = 330
TAG_EXIF_IFD = 34665
TAG_GPS_IFD = 34853
TAG_INTEROP_IFD = 40965

POINTER_TAGS = {TAG_SUBIFD, TAG_EXIF_IFD, TAG_GPS_IFD, TAG_INTEROP_IFD}


@dataclass
class FileStats:
    file_size: int
    total_tags: int
    ifd_count: int
    ifd_entry_max: int
    has_order_violation: bool
    order_violations: int
    invalid_offsets: int
    has_invalid_offset: bool
    pointer_tags: int
    unique_tags: set
    has_dng_tag: bool


def read_u16(mm: mmap.mmap, off: int, endian: str) -> Optional[int]:
    if off < 0 or off + 2 > len(mm):
        return None
    return struct.unpack_from(endian + "H", mm, off)[0]


def read_u32(mm: mmap.mmap, off: int, endian: str) -> Optional[int]:
    if off < 0 or off + 4 > len(mm):
        return None
    return struct.unpack_from(endian + "I", mm, off)[0]


def parse_tiff(path: Path) -> Optional[FileStats]:
    try:
        size = path.stat().st_size
        if size < 8:
            return None
        with path.open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                hdr = mm[:4]
                if hdr[:2] == b"II":
                    endian = "<"
                    magic = struct.unpack_from("<H", mm, 2)[0]
                elif hdr[:2] == b"MM":
                    endian = ">"
                    magic = struct.unpack_from(">H", mm, 2)[0]
                else:
                    return None

                if magic == 43:
                    # BigTIFF not handled here.
                    return None
                if magic != 42:
                    return None

                first_ifd = read_u32(mm, 4, endian)
                if first_ifd is None or first_ifd <= 0 or first_ifd >= size:
                    return None

                visited = set()
                stack = [first_ifd]
                total_tags = 0
                ifd_count = 0
                ifd_entry_max = 0
                order_violations = 0
                invalid_offsets = 0
                pointer_tags = 0
                unique_tags = set()
                has_dng_tag = False

                while stack:
                    off = stack.pop()
                    if off in visited or off <= 0 or off + 2 > size:
                        continue
                    visited.add(off)
                    count = read_u16(mm, off, endian)
                    if count is None:
                        continue
                    base = off + 2
                    if base + count * 12 > size:
                        continue

                    ifd_count += 1
                    ifd_entry_max = max(ifd_entry_max, count)
                    total_tags += count

                    prev_tag = None
                    for i in range(count):
                        entry_off = base + i * 12
                        tag = read_u16(mm, entry_off, endian)
                        type_id = read_u16(mm, entry_off + 2, endian)
                        val_count = read_u32(mm, entry_off + 4, endian)
                        value_or_offset = read_u32(mm, entry_off + 8, endian)
                        if tag is None or type_id is None or val_count is None or value_or_offset is None:
                            continue

                        unique_tags.add(tag)
                        if tag == TAG_DNG_VERSION:
                            has_dng_tag = True
                        if tag in POINTER_TAGS:
                            pointer_tags += 1

                        if prev_tag is not None and tag < prev_tag:
                            order_violations += 1
                        prev_tag = tag

                        type_size = TIFF_TYPES.get(type_id, 0)
                        data_size = val_count * type_size
                        if data_size > 4:
                            if value_or_offset + data_size > size:
                                invalid_offsets += 1

                        # Follow pointer tags for nested IFDs.
                        if tag in POINTER_TAGS and value_or_offset != 0 and value_or_offset < size:
                            stack.append(value_or_offset)
                        elif tag == TAG_SUBIFD and val_count and data_size > 4:
                            # SubIFD array of offsets.
                            for j in range(val_count):
                                sub_off = read_u32(mm, value_or_offset + j * 4, endian)
                                if sub_off and sub_off < size:
                                    stack.append(sub_off)

                    next_ifd = read_u32(mm, base + count * 12, endian)
                    if next_ifd:
                        stack.append(next_ifd)

                return FileStats(
                    file_size=size,
                    total_tags=total_tags,
                    ifd_count=ifd_count,
                    ifd_entry_max=ifd_entry_max,
                    has_order_violation=order_violations > 0,
                    order_violations=order_violations,
                    invalid_offsets=invalid_offsets,
                    has_invalid_offset=invalid_offsets > 0,
                    pointer_tags=pointer_tags,
                    unique_tags=unique_tags,
                    has_dng_tag=has_dng_tag,
                )
            finally:
                mm.close()
    except Exception:
        return None


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from (c for c in p.rglob("*") if c.is_file())
        elif p.is_file():
            yield p


def percentile(sorted_vals: List[int], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def summarize(stats_list: List[FileStats]) -> Dict[str, float]:
    totals = sorted(s.total_tags for s in stats_list)
    ifd_counts = sorted(s.ifd_count for s in stats_list)
    invalid_rate = sum(1 for s in stats_list if s.has_invalid_offset) / len(stats_list)
    order_rate = sum(1 for s in stats_list if s.has_order_violation) / len(stats_list)
    return {
        "files": len(stats_list),
        "tags_median": percentile(totals, 50),
        "tags_p95": percentile(totals, 95),
        "tags_max": max(totals) if totals else 0,
        "ifd_median": percentile(ifd_counts, 50),
        "ifd_max": max(ifd_counts) if ifd_counts else 0,
        "order_violation_rate": order_rate,
        "invalid_offset_rate": invalid_rate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Files or directories to scan")
    args = ap.parse_args()

    groups: Dict[str, List[FileStats]] = defaultdict(list)
    unique_tag_sets: Dict[str, set] = defaultdict(set)

    for path in iter_files([Path(p) for p in args.paths]):
        stats = parse_tiff(path)
        if stats is None:
            continue

        parts = path.parts
        if "LandFall" in parts:
            group = "landfall"
        elif "general_mal" in parts:
            group = "general_mal"
        elif "benign_data" in parts:
            group = "benign_dng" if stats.has_dng_tag else "benign_tiff"
        else:
            group = "other"

        groups[group].append(stats)
        unique_tag_sets[group].update(stats.unique_tags)

    for group, stats_list in groups.items():
        if not stats_list:
            continue
        summary = summarize(stats_list)
        uniq_tags = len(unique_tag_sets[group])
        print(f"[{group}] files={summary['files']} unique_tags={uniq_tags} tags_med={summary['tags_median']:.0f} "
              f"tags_p95={summary['tags_p95']:.0f} tags_max={summary['tags_max']} "
              f"ifd_med={summary['ifd_median']:.0f} ifd_max={summary['ifd_max']} "
              f"order_violation_rate={summary['order_violation_rate']:.3f} "
              f"invalid_offset_rate={summary['invalid_offset_rate']:.3f}")


if __name__ == "__main__":
    main()
