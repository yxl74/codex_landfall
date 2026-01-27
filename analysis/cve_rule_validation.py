#!/usr/bin/env python3
"""
CVE rule validation against the full dataset.

Extracts CVE-relevant structural features from TIFF/DNG files and evaluates
three static detection rules:

  CVE-2025-21043:  Declared opcode list count > 1,000,000
  CVE-2025-43300:  SubIFD has SPP=2, Compression=7 (JPEG Lossless),
                   AND embedded JPEG SOF3 marker has component_count=1
  TILE-CONFIG:     tile_offsets_count != tile_byte_counts_count,
                   OR actual tiles != expected from geometry,
                   OR any tile/image dimension > 0xFFFE7960

Reports per-class trigger rates and overall FPR.

Outputs:
  - outputs/cve_rule_validation_report.json
  - outputs/cve_structural_features.npz
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import struct
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# --- TIFF constants ---

TIFF_TYPES = {
    1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
    8: 2, 9: 4, 10: 8, 11: 4, 12: 8,
}

TAG_NEW_SUBFILE_TYPE = 254
TAG_WIDTH = 256
TAG_HEIGHT = 257
TAG_COMPRESSION = 259
TAG_SAMPLES_PER_PIXEL = 277
TAG_STRIP_OFFSETS = 273
TAG_STRIP_BYTE_COUNTS = 279
TAG_TILE_WIDTH = 322
TAG_TILE_HEIGHT = 323
TAG_TILE_OFFSETS = 324
TAG_TILE_BYTE_COUNTS = 325
TAG_SUBIFD = 330
TAG_EXIF_IFD = 34665
TAG_DNG_VERSION = 50706
TAG_OPCODE_LIST1 = 51008
TAG_OPCODE_LIST2 = 51009
TAG_OPCODE_LIST3 = 51022

EXTREME_DIM_THRESHOLD = 0xFFFE7960


# --- Low-level readers ---

def _read_u16(buf: mmap.mmap, off: int, endian: str) -> Optional[int]:
    if off + 2 > len(buf):
        return None
    fmt = "<H" if endian == "II" else ">H"
    return struct.unpack_from(fmt, buf, off)[0]


def _read_u32(buf: mmap.mmap, off: int, endian: str) -> Optional[int]:
    if off + 4 > len(buf):
        return None
    fmt = "<I" if endian == "II" else ">I"
    return struct.unpack_from(fmt, buf, off)[0]


def _read_u32be(buf: mmap.mmap, off: int) -> Optional[int]:
    if off + 4 > len(buf):
        return None
    return struct.unpack_from(">I", buf, off)[0]


def _read_values(
    buf: mmap.mmap,
    endian: str,
    file_size: int,
    type_id: int,
    count: int,
    value_or_offset: int,
) -> List[int]:
    if count <= 0:
        return []
    elem_size = TIFF_TYPES.get(type_id, 1)
    size_bytes = elem_size * count
    if size_bytes <= 4:
        raw = value_or_offset.to_bytes(4, "little" if endian == "II" else "big")
        if type_id == 3:
            fmt = "<H" if endian == "II" else ">H"
            return [struct.unpack_from(fmt, raw, i * 2)[0] for i in range(min(count, 2))]
        if type_id == 4:
            fmt = "<I" if endian == "II" else ">I"
            return [struct.unpack_from(fmt, raw, 0)[0]]
        return []
    if value_or_offset + size_bytes > file_size:
        return []
    if type_id == 3:
        fmt = ("<" if endian == "II" else ">") + ("H" * count)
        return list(struct.unpack_from(fmt, buf, value_or_offset))
    if type_id == 4:
        fmt = ("<" if endian == "II" else ">") + ("I" * count)
        return list(struct.unpack_from(fmt, buf, value_or_offset))
    return []


# --- TIFF/DNG extended parsing ---

def parse_cve_features(path: str) -> Optional[Dict]:
    """Parse TIFF structure with CVE-relevant fields."""
    file_size = os.path.getsize(path)
    if file_size < 8:
        return None

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            endian = mm[0:2].decode("latin-1", errors="ignore")
            if endian not in ("II", "MM"):
                return None
            magic = _read_u16(mm, 2, endian)
            if magic != 42:
                return None
            root = _read_u32(mm, 4, endian)
            if root is None:
                return None

            is_dng = 0
            widths: List[int] = []
            heights: List[int] = []
            subifd_offsets: List[int] = []
            exif_offset = 0
            opcode_lists: Dict[int, Tuple[int, int]] = {}

            # CVE-relevant per-IFD data
            compression_values: set = set()
            spp_values: set = set()
            tile_offsets_counts: List[int] = []
            tile_byte_counts_counts: List[int] = []
            tile_widths: List[int] = []
            tile_heights: List[int] = []

            # Per-IFD tuples for SOF3 scanning: (compression, spp, strip_offset)
            ifd_sof3_candidates: List[Tuple[int, int, int]] = []

            visited = set()
            stack: List[int] = [root]

            # Track current IFD local data
            def process_ifd_batch(off: int, entries: List[Tuple[int, int, int, int]]):
                """Process all entries from one IFD together for per-IFD logic."""
                nonlocal is_dng, exif_offset
                local_compression = None
                local_spp = None
                local_strip_offset = None
                local_tile_offsets_count = 0
                local_tile_byte_counts_count = 0

                for tag, type_id, val_count, value_or_offset in entries:
                    if tag == TAG_DNG_VERSION:
                        is_dng = 1

                    if tag in (TAG_WIDTH, TAG_HEIGHT):
                        vals = _read_values(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            if tag == TAG_WIDTH:
                                widths.extend(vals)
                            else:
                                heights.extend(vals)

                    if tag == TAG_COMPRESSION:
                        vals = _read_values(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            compression_values.update(vals)
                            local_compression = vals[0]

                    if tag == TAG_SAMPLES_PER_PIXEL:
                        vals = _read_values(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            spp_values.update(vals)
                            local_spp = vals[0]

                    if tag == TAG_STRIP_OFFSETS:
                        vals = _read_values(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            local_strip_offset = vals[0]

                    if tag == TAG_TILE_OFFSETS:
                        tile_offsets_counts.append(val_count)

                    if tag == TAG_TILE_BYTE_COUNTS:
                        tile_byte_counts_counts.append(val_count)

                    if tag == TAG_TILE_WIDTH:
                        vals = _read_values(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            tile_widths.extend(vals)

                    if tag == TAG_TILE_HEIGHT:
                        vals = _read_values(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            tile_heights.extend(vals)

                    if tag == TAG_SUBIFD:
                        size_bytes = TIFF_TYPES.get(type_id, 1) * val_count
                        if size_bytes <= 4:
                            subifd_offsets.append(value_or_offset)
                        else:
                            if value_or_offset + size_bytes <= file_size:
                                for j in range(val_count):
                                    subifd_offsets.append(
                                        _read_u32(mm, value_or_offset + j * 4, endian) or 0
                                    )

                    if tag == TAG_EXIF_IFD:
                        exif_offset = value_or_offset

                    if tag in (TAG_OPCODE_LIST1, TAG_OPCODE_LIST2, TAG_OPCODE_LIST3):
                        size_bytes = TIFF_TYPES.get(type_id, 1) * val_count
                        opcode_lists[tag] = (value_or_offset, size_bytes)

                # Track SOF3 candidate IFDs
                if local_compression is not None and local_spp is not None and local_strip_offset is not None:
                    ifd_sof3_candidates.append((local_compression, local_spp, local_strip_offset))

            # Traverse root + chained IFDs
            while stack:
                off = stack.pop()
                if off == 0 or off in visited or off >= file_size:
                    continue
                visited.add(off)
                count = _read_u16(mm, off, endian)
                if count is None:
                    continue
                entry_base = off + 2
                entries = []
                for i in range(count):
                    entry_off = entry_base + i * 12
                    if entry_off + 12 > file_size:
                        break
                    tag = _read_u16(mm, entry_off, endian)
                    type_id = _read_u16(mm, entry_off + 2, endian)
                    val_count = _read_u32(mm, entry_off + 4, endian)
                    value_or_offset = _read_u32(mm, entry_off + 8, endian)
                    if tag is None or type_id is None or val_count is None or value_or_offset is None:
                        continue
                    entries.append((tag, type_id, val_count, value_or_offset))

                process_ifd_batch(off, entries)

                next_ptr_off = entry_base + count * 12
                next_ifd = _read_u32(mm, next_ptr_off, endian)
                if next_ifd:
                    stack.append(next_ifd)

            # Traverse SubIFDs and Exif
            for off in list(subifd_offsets):
                stack.append(off)
            if exif_offset:
                stack.append(exif_offset)

            while stack:
                off = stack.pop()
                if off == 0 or off in visited or off >= file_size:
                    continue
                visited.add(off)
                count = _read_u16(mm, off, endian)
                if count is None:
                    continue
                entry_base = off + 2
                entries = []
                for i in range(count):
                    entry_off = entry_base + i * 12
                    if entry_off + 12 > file_size:
                        break
                    tag = _read_u16(mm, entry_off, endian)
                    type_id = _read_u16(mm, entry_off + 2, endian)
                    val_count = _read_u32(mm, entry_off + 4, endian)
                    value_or_offset = _read_u32(mm, entry_off + 8, endian)
                    if tag is None or type_id is None or val_count is None or value_or_offset is None:
                        continue
                    entries.append((tag, type_id, val_count, value_or_offset))

                process_ifd_batch(off, entries)

                next_ptr_off = entry_base + count * 12
                next_ifd = _read_u32(mm, next_ptr_off, endian)
                if next_ifd:
                    stack.append(next_ifd)

            # --- Parse opcode list declared counts ---
            max_declared_opcode_count = 0
            for tag, (offset, size_bytes) in opcode_lists.items():
                if offset == 0 or offset + size_bytes > file_size or size_bytes < 4:
                    continue
                declared = _read_u32be(mm, offset)
                if declared is not None:
                    max_declared_opcode_count = max(max_declared_opcode_count, declared)

            # --- SOF3 scanning ---
            sof3_component_mismatch = False
            MAX_SOF3_SCAN = 65536
            for comp_val, spp_val, strip_off in ifd_sof3_candidates:
                if comp_val == 7 and spp_val == 2:
                    scan_end = min(strip_off + MAX_SOF3_SCAN, file_size)
                    scan_len = scan_end - strip_off
                    if scan_len < 4 or strip_off >= file_size:
                        continue
                    data = mm[strip_off:strip_off + scan_len]
                    # Search for SOF3 marker: 0xFF 0xC3
                    pos = 0
                    while pos < len(data) - 1:
                        idx = data.find(b"\xff\xc3", pos)
                        if idx == -1:
                            break
                        # SOF3 header: FF C3 Lh Ll P Y(2) X(2) Nf
                        # Nf (component count) is at offset idx+9 from marker start
                        if idx + 10 <= len(data):
                            nf = data[idx + 9]
                            if nf != spp_val:
                                sof3_component_mismatch = True
                                break
                        pos = idx + 2
                    if sof3_component_mismatch:
                        break

            # --- Tile geometry ---
            total_tile_offsets_count = sum(tile_offsets_counts)
            total_tile_byte_counts_count = sum(tile_byte_counts_counts)

            # Compute expected tile count from geometry
            # We pair widths/heights with tile_widths/tile_heights as available
            expected_tile_count = 0
            if tile_widths and tile_heights and widths and heights:
                tw = tile_widths[0]
                th = tile_heights[0]
                if tw > 0 and th > 0:
                    # Use max image dimensions
                    w = max(widths) if widths else 0
                    h = max(heights) if heights else 0
                    expected_tile_count = math.ceil(w / tw) * math.ceil(h / th)

            # Max dimension values for extreme-dim check
            all_dims = widths + heights + tile_widths + tile_heights
            max_dim = max(all_dims) if all_dims else 0

            return {
                "is_dng": is_dng,
                "max_declared_opcode_count": max_declared_opcode_count,
                "compression_values": sorted(compression_values),
                "spp_values": sorted(spp_values),
                "sof3_component_mismatch": sof3_component_mismatch,
                "tile_offsets_count": total_tile_offsets_count,
                "tile_byte_counts_count": total_tile_byte_counts_count,
                "expected_tile_count": expected_tile_count,
                "tile_widths": tile_widths,
                "tile_heights": tile_heights,
                "max_dim": max_dim,
                "widths": widths,
                "heights": heights,
            }
        finally:
            mm.close()


# --- CVE Rules ---

def rule_cve_2025_21043(feat: Dict) -> bool:
    """Declared opcode list count > 1,000,000."""
    return feat["max_declared_opcode_count"] > 1_000_000


def rule_cve_2025_43300(feat: Dict) -> bool:
    """SubIFD has SPP=2, Compression=7, AND SOF3 component_count mismatch."""
    return (
        2 in feat["spp_values"]
        and 7 in feat["compression_values"]
        and feat["sof3_component_mismatch"]
    )


def rule_tile_config(feat: Dict) -> bool:
    """Tile offset/bytecount mismatch, geometry mismatch, or extreme dimensions."""
    # Mismatch between tile offsets count and tile byte counts count
    if (feat["tile_offsets_count"] > 0 or feat["tile_byte_counts_count"] > 0):
        if feat["tile_offsets_count"] != feat["tile_byte_counts_count"]:
            return True

    # Actual tile count != expected from geometry
    if feat["expected_tile_count"] > 0 and feat["tile_offsets_count"] > 0:
        if feat["tile_offsets_count"] != feat["expected_tile_count"]:
            return True

    # Extreme dimensions
    if feat["max_dim"] > EXTREME_DIM_THRESHOLD:
        return True

    return False


# --- Labelling ---

def get_label_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "benign_data" in parts:
        return "benign"
    if "LandFall" in parts:
        return "landfall"
    if "general_mal" in parts:
        return "general_mal"
    return "unknown"


# --- Main ---

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate CVE detection rules against the full dataset."
    )
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--list-file",
        default="",
        help="Optional file list; if set, only these paths are processed",
    )
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    paths: List[str] = []
    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip()
                if path and os.path.isfile(path):
                    paths.append(path)
    else:
        for sub in ("benign_data", "LandFall", "general_mal"):
            base = os.path.join(data_root, sub)
            if not os.path.isdir(base):
                continue
            for dirpath, _, filenames in os.walk(base):
                for fn in filenames:
                    paths.append(os.path.join(dirpath, fn))

    paths.sort()

    # Collect results
    per_class: Dict[str, Dict] = defaultdict(
        lambda: {
            "count": 0,
            "tiff_count": 0,
            "cve_2025_21043": 0,
            "cve_2025_43300": 0,
            "tile_config": 0,
            "any_cve": 0,
        }
    )

    feature_rows = []
    file_paths_out = []
    labels_out = []

    for i, p in enumerate(paths):
        label = get_label_from_path(p)
        feat = parse_cve_features(p)

        if feat is None:
            per_class[label]["count"] += 1
            feature_rows.append({
                "max_declared_opcode_count": 0,
                "spp_max": 0,
                "compression_variety": 0,
                "tile_count_ratio": 0.0,
                "sof3_component_mismatch": 0,
                "tile_offsets_count": 0,
                "tile_byte_counts_count": 0,
                "expected_tile_count": 0,
                "max_dim": 0,
            })
            file_paths_out.append(p)
            labels_out.append(label)
            continue

        per_class[label]["count"] += 1
        per_class[label]["tiff_count"] += 1

        r1 = rule_cve_2025_21043(feat)
        r2 = rule_cve_2025_43300(feat)
        r3 = rule_tile_config(feat)

        if r1:
            per_class[label]["cve_2025_21043"] += 1
        if r2:
            per_class[label]["cve_2025_43300"] += 1
        if r3:
            per_class[label]["tile_config"] += 1
        if r1 or r2 or r3:
            per_class[label]["any_cve"] += 1

        spp_max = max(feat["spp_values"]) if feat["spp_values"] else 0
        compression_variety = len(feat["compression_values"])
        tile_count_ratio = 0.0
        if feat["expected_tile_count"] > 0:
            tile_count_ratio = feat["tile_offsets_count"] / feat["expected_tile_count"]

        feature_rows.append({
            "max_declared_opcode_count": feat["max_declared_opcode_count"],
            "spp_max": spp_max,
            "compression_variety": compression_variety,
            "tile_count_ratio": tile_count_ratio,
            "sof3_component_mismatch": int(feat["sof3_component_mismatch"]),
            "tile_offsets_count": feat["tile_offsets_count"],
            "tile_byte_counts_count": feat["tile_byte_counts_count"],
            "expected_tile_count": feat["expected_tile_count"],
            "max_dim": feat["max_dim"],
        })
        file_paths_out.append(p)
        labels_out.append(label)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(paths)} files...")

    # --- Compute FPR ---
    benign_stats = per_class.get("benign", {"count": 0, "any_cve": 0})
    benign_total = benign_stats["count"]
    benign_fp = benign_stats["any_cve"]
    fpr = benign_fp / benign_total if benign_total > 0 else 0.0

    # --- Build report ---
    report = {
        "total_files": len(paths),
        "benign_total": benign_total,
        "benign_false_positives": benign_fp,
        "overall_fpr": fpr,
        "per_class": dict(per_class),
    }

    report_path = os.path.join(output_dir, "cve_rule_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # --- Save structural features as NPZ ---
    feature_names = [
        "log_max_declared_opcode_count",
        "spp_max",
        "compression_variety",
        "tile_count_ratio",
        "sof3_component_mismatch",
        "tile_offsets_count",
        "tile_byte_counts_count",
        "expected_tile_count",
        "max_dim",
    ]

    X = np.zeros((len(feature_rows), len(feature_names)), dtype=np.float64)
    for i, row in enumerate(feature_rows):
        X[i, 0] = np.log1p(row["max_declared_opcode_count"])
        X[i, 1] = row["spp_max"]
        X[i, 2] = row["compression_variety"]
        X[i, 3] = row["tile_count_ratio"]
        X[i, 4] = row["sof3_component_mismatch"]
        X[i, 5] = row["tile_offsets_count"]
        X[i, 6] = row["tile_byte_counts_count"]
        X[i, 7] = row["expected_tile_count"]
        X[i, 8] = row["max_dim"]

    npz_path = os.path.join(output_dir, "cve_structural_features.npz")
    np.savez_compressed(
        npz_path,
        X_cve=X,
        feature_names=np.array(feature_names),
        paths=np.array(file_paths_out),
        labels=np.array(labels_out),
    )

    # --- Print summary ---
    print(f"\nCVE Rule Validation Report")
    print(f"{'='*60}")
    print(f"Total files: {len(paths)}")
    print(f"Overall FPR: {fpr:.6f} ({benign_fp}/{benign_total} benign)")
    print()
    for cls in ("benign", "landfall", "general_mal", "unknown"):
        if cls not in per_class:
            continue
        s = per_class[cls]
        print(f"  {cls:15s}: {s['count']:6d} files ({s['tiff_count']} TIFF)")
        print(f"    CVE-2025-21043: {s['cve_2025_21043']:4d}")
        print(f"    CVE-2025-43300: {s['cve_2025_43300']:4d}")
        print(f"    TILE-CONFIG:    {s['tile_config']:4d}")
        print(f"    Any CVE:        {s['any_cve']:4d}")
    print()
    print(f"Wrote: {report_path}")
    print(f"Wrote: {npz_path}")

    if fpr > 0:
        print(f"\nWARNING: FPR > 0 ({fpr:.6f}). Rules need tightening before deployment.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
