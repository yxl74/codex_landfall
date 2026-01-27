#!/usr/bin/env python3
"""
Hybrid feature extraction for TIFF/DNG anomaly detection.

Outputs:
  - CSV with metadata + structural features + detection flags
  - NPZ with Magika-style byte features (beg/end) + structural features
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import mmap
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np


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

TAG_WIDTH = 256
TAG_HEIGHT = 257
TAG_SUBIFD = 330
TAG_EXIF_IFD = 34665
TAG_DNG_VERSION = 50706
TAG_NEW_SUBFILE_TYPE = 254
TAG_COMPRESSION = 259
TAG_SAMPLES_PER_PIXEL = 277
TAG_TILE_WIDTH = 322
TAG_TILE_HEIGHT = 323
TAG_TILE_OFFSETS = 324
TAG_TILE_BYTE_COUNTS = 325
TAG_OPCODE_LIST1 = 51008
TAG_OPCODE_LIST2 = 51009
TAG_OPCODE_LIST3 = 51022


def is_whitespace(b: int) -> bool:
    return b in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20)


def strip_prefix(data: bytes) -> bytes:
    i = 0
    while i < len(data) and is_whitespace(data[i]):
        i += 1
    return data[i:]


def strip_suffix(data: bytes) -> bytes:
    i = len(data)
    while i > 0 and is_whitespace(data[i - 1]):
        i -= 1
    return data[:i]


def magika_like_bytes(
    path: str,
    beg_size: int = 1024,
    end_size: int = 1024,
    block_size: int = 4096,
    padding_token: int = 256,
) -> np.ndarray:
    file_size = os.path.getsize(path)
    if file_size == 0:
        return np.full((beg_size + end_size,), padding_token, dtype=np.int16)

    buffer_size = min(block_size, file_size)
    with open(path, "rb") as f:
        beg_block = f.read(buffer_size)
        beg = strip_prefix(beg_block)
        if file_size >= buffer_size:
            f.seek(max(0, file_size - buffer_size))
        end_block = f.read(buffer_size)
        end = strip_suffix(end_block)

    features = np.full((beg_size + end_size,), padding_token, dtype=np.int16)

    beg_len = min(len(beg), beg_size)
    if beg_len:
        features[:beg_len] = np.frombuffer(beg[:beg_len], dtype=np.uint8).astype(np.int16)

    end_len = min(len(end), end_size)
    if end_len:
        features[beg_size : beg_size + end_len] = np.frombuffer(
            end[-end_len:], dtype=np.uint8
        ).astype(np.int16)

    return features


def detect_magic_type(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(16)
    if len(head) >= 4 and head[:2] in (b"II", b"MM") and head[2:4] in (b"\x2a\x00", b"\x00\x2a"):
        return "tiff"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG"):
        return "png"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "gif"
    if head.startswith(b"BM"):
        return "bmp"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    return "unknown"


def zip_polyglot_flags(path: str, tail_window: int = 1024 * 1024, eocd_tail: int = 65536) -> Tuple[int, int]:
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0
    start = max(0, size - tail_window)
    with open(path, "rb") as f:
        f.seek(start)
        tail = f.read(size - start)
    eocd_pos = tail.find(b"PK\x05\x06")
    if eocd_pos == -1:
        return 0, 0
    eocd_abs = start + eocd_pos
    if size - eocd_abs > eocd_tail:
        return 0, 0
    local_pos = tail.find(b"PK\x03\x04")
    return 1, 1 if local_pos != -1 else 0


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


def _read_value(
    buf: mmap.mmap,
    endian: str,
    file_size: int,
    type_id: int,
    count: int,
    value_or_offset: int,
) -> Optional[List[int]]:
    size_bytes = TIFF_TYPES.get(type_id, 1) * count
    if size_bytes <= 4:
        raw = value_or_offset.to_bytes(4, "little" if endian == "II" else "big")
        if type_id == 3:
            fmt = "<H" if endian == "II" else ">H"
            return [struct.unpack_from(fmt, raw, 0)[0]]
        if type_id == 4:
            fmt = "<I" if endian == "II" else ">I"
            return [struct.unpack_from(fmt, raw, 0)[0]]
        return None
    if value_or_offset + size_bytes > file_size:
        return None
    if type_id == 3:
        fmt = ("<" if endian == "II" else ">") + ("H" * count)
        return list(struct.unpack_from(fmt, buf, value_or_offset))
    if type_id == 4:
        fmt = ("<" if endian == "II" else ">") + ("I" * count)
        return list(struct.unpack_from(fmt, buf, value_or_offset))
    return None


def parse_tiff_struct(path: str) -> Optional[Dict[str, int]]:
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(mm) < 8:
                return None
            endian = mm[0:2].decode("latin-1", errors="ignore")
            if endian not in ("II", "MM"):
                return None
            magic = _read_u16(mm, 2, endian)
            if magic != 42:
                return None
            root = _read_u32(mm, 4, endian)
            if root is None:
                return None

            widths: List[int] = []
            heights: List[int] = []
            ifd_entry_max = 0
            subifd_offsets: List[int] = []
            exif_offset = 0
            new_subfile_types: List[int] = []
            opcode_lists: Dict[int, Tuple[int, int]] = {}
            is_dng = 0
            compression_values: set = set()
            spp_values: List[int] = []
            tile_offsets_count = 0
            tile_widths: List[int] = []
            tile_heights: List[int] = []

            visited = set()
            stack: List[int] = [root]

            # Parse root and chained IFDs
            while stack:
                off = stack.pop()
                if off == 0 or off in visited or off >= file_size:
                    continue
                visited.add(off)
                count = _read_u16(mm, off, endian)
                if count is None:
                    continue
                ifd_entry_max = max(ifd_entry_max, count)
                entry_base = off + 2
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
                    if tag == TAG_DNG_VERSION:
                        is_dng = 1
                    if tag in (TAG_WIDTH, TAG_HEIGHT, TAG_NEW_SUBFILE_TYPE):
                        vals = _read_value(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            if tag == TAG_WIDTH:
                                widths.extend(vals)
                            elif tag == TAG_HEIGHT:
                                heights.extend(vals)
                            else:
                                new_subfile_types.extend(vals)
                    if tag == TAG_SUBIFD:
                        size_bytes = TIFF_TYPES.get(type_id, 1) * val_count
                        if size_bytes <= 4:
                            subifd_offsets = [value_or_offset]
                        else:
                            if value_or_offset + size_bytes <= file_size:
                                for j in range(val_count):
                                    subifd_offsets.append(_read_u32(mm, value_or_offset + j * 4, endian) or 0)
                    if tag == TAG_EXIF_IFD:
                        exif_offset = value_or_offset
                    if tag in (TAG_OPCODE_LIST1, TAG_OPCODE_LIST2, TAG_OPCODE_LIST3):
                        size_bytes = TIFF_TYPES.get(type_id, 1) * val_count
                        opcode_lists[tag] = (value_or_offset, size_bytes)
                    if tag == TAG_COMPRESSION:
                        vals = _read_value(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            compression_values.update(vals)
                    if tag == TAG_SAMPLES_PER_PIXEL:
                        vals = _read_value(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            spp_values.extend(vals)
                    if tag == TAG_TILE_OFFSETS:
                        tile_offsets_count += val_count
                    if tag in (TAG_TILE_WIDTH, TAG_TILE_HEIGHT):
                        vals = _read_value(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            if tag == TAG_TILE_WIDTH:
                                tile_widths.extend(vals)
                            else:
                                tile_heights.extend(vals)

                next_ptr_off = entry_base + count * 12
                next_ifd = _read_u32(mm, next_ptr_off, endian)
                if next_ifd:
                    stack.append(next_ifd)

            # Parse subIFDs and Exif IFDs to find opcode lists
            for off in subifd_offsets:
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
                ifd_entry_max = max(ifd_entry_max, count)
                entry_base = off + 2
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
                    if tag == TAG_DNG_VERSION:
                        is_dng = 1
                    if tag in (TAG_WIDTH, TAG_HEIGHT, TAG_NEW_SUBFILE_TYPE):
                        vals = _read_value(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            if tag == TAG_WIDTH:
                                widths.extend(vals)
                            elif tag == TAG_HEIGHT:
                                heights.extend(vals)
                            else:
                                new_subfile_types.extend(vals)
                    if tag == TAG_SUBIFD:
                        size_bytes = TIFF_TYPES.get(type_id, 1) * val_count
                        if size_bytes <= 4:
                            subifd_offsets.append(value_or_offset)
                        else:
                            if value_or_offset + size_bytes <= file_size:
                                for j in range(val_count):
                                    subifd_offsets.append(_read_u32(mm, value_or_offset + j * 4, endian) or 0)
                    if tag == TAG_EXIF_IFD:
                        exif_offset = value_or_offset
                    if tag in (TAG_OPCODE_LIST1, TAG_OPCODE_LIST2, TAG_OPCODE_LIST3):
                        size_bytes = TIFF_TYPES.get(type_id, 1) * val_count
                        opcode_lists[tag] = (value_or_offset, size_bytes)
                    if tag == TAG_COMPRESSION:
                        vals = _read_value(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            compression_values.update(vals)
                    if tag == TAG_SAMPLES_PER_PIXEL:
                        vals = _read_value(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            spp_values.extend(vals)
                    if tag == TAG_TILE_OFFSETS:
                        tile_offsets_count += val_count
                    if tag in (TAG_TILE_WIDTH, TAG_TILE_HEIGHT):
                        vals = _read_value(mm, endian, file_size, type_id, val_count, value_or_offset)
                        if vals:
                            if tag == TAG_TILE_WIDTH:
                                tile_widths.extend(vals)
                            else:
                                tile_heights.extend(vals)

                next_ptr_off = entry_base + count * 12
                next_ifd = _read_u32(mm, next_ptr_off, endian)
                if next_ifd:
                    stack.append(next_ifd)

            min_width = min(widths) if widths else 0
            min_height = min(heights) if heights else 0

            # Parse opcode lists (DNG)
            total_opcodes = 0
            unknown_opcodes = 0
            max_opcode_id = 0
            opcode_list1_bytes = 0
            opcode_list2_bytes = 0
            opcode_list3_bytes = 0
            max_declared_opcode_count = 0

            for tag, (offset, size_bytes) in opcode_lists.items():
                if offset == 0 or offset + size_bytes > file_size or size_bytes < 4:
                    continue
                opcode_count = _read_u32be(mm, offset)
                if opcode_count is None:
                    continue
                max_declared_opcode_count = max(max_declared_opcode_count, opcode_count)
                pos = offset + 4
                parsed = 0
                while parsed < opcode_count and pos + 16 <= offset + size_bytes:
                    opcode_id = _read_u32be(mm, pos)
                    data_size = _read_u32be(mm, pos + 12)
                    if opcode_id is None or data_size is None:
                        break
                    pos += 16
                    if pos + data_size > offset + size_bytes:
                        break
                    parsed += 1
                    total_opcodes += 1
                    max_opcode_id = max(max_opcode_id, opcode_id)
                    if opcode_id > 14:
                        unknown_opcodes += 1
                    pos += data_size

                if tag == TAG_OPCODE_LIST1:
                    opcode_list1_bytes = size_bytes
                elif tag == TAG_OPCODE_LIST2:
                    opcode_list2_bytes = size_bytes
                elif tag == TAG_OPCODE_LIST3:
                    opcode_list3_bytes = size_bytes

            # Compute CVE-derived features
            spp_max = max(spp_values) if spp_values else 0
            compression_variety = len(compression_values)
            expected_tile_count = 0
            if tile_widths and tile_heights and widths and heights:
                tw = tile_widths[0]
                th = tile_heights[0]
                if tw > 0 and th > 0:
                    mw = max(widths)
                    mh = max(heights)
                    expected_tile_count = math.ceil(mw / tw) * math.ceil(mh / th)
            tile_count_ratio = (tile_offsets_count / expected_tile_count) if expected_tile_count > 0 else 0.0

            return {
                "is_tiff": 1,
                "is_dng": is_dng,
                "min_width": min_width,
                "min_height": min_height,
                "ifd_entry_max": ifd_entry_max,
                "subifd_count_sum": len(subifd_offsets),
                "new_subfile_types_unique": len(set(new_subfile_types)) if new_subfile_types else 0,
                "total_opcodes": total_opcodes,
                "unknown_opcodes": unknown_opcodes,
                "max_opcode_id": max_opcode_id,
                "opcode_list1_bytes": opcode_list1_bytes,
                "opcode_list2_bytes": opcode_list2_bytes,
                "opcode_list3_bytes": opcode_list3_bytes,
                "max_declared_opcode_count": max_declared_opcode_count,
                "spp_max": spp_max,
                "compression_variety": compression_variety,
                "tile_count_ratio": tile_count_ratio,
            }
        finally:
            mm.close()


def get_label_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "benign_data" in parts:
        return "benign"
    if "LandFall" in parts:
        return "landfall"
    if "general_mal" in parts:
        return "general_mal"
    return "unknown"


def ext_claimed_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "jpeg"
    if ext in ("tif", "tiff"):
        return "tiff"
    if ext == "dng":
        return "dng"
    if ext == "png":
        return "png"
    if ext == "gif":
        return "gif"
    return "unknown"


def _init_magika(magika_path: Optional[str]) -> Optional[object]:
    if magika_path:
        sys.path.insert(0, magika_path)
    try:
        from magika import Magika  # type: ignore

        return Magika()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--list-file",
        default="",
        help="Optional file list; if set, only these paths are processed",
    )
    parser.add_argument(
        "--magika-path",
        default="",
        help="Optional path to magika/python/src to enable Magika output",
    )
    parser.add_argument(
        "--use-magika",
        action="store_true",
        help="Enable Magika inference (requires onnxruntime and models)",
    )
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    magika_session = _init_magika(args.magika_path) if args.use_magika else None

    paths: List[str] = []
    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                if os.path.isfile(path):
                    paths.append(path)
    else:
        for sub in ("benign_data", "LandFall", "general_mal"):
            base = os.path.join(data_root, sub)
            if not os.path.isdir(base):
                continue
            for dirpath, _, filenames in os.walk(base):
                for fn in filenames:
                    paths.append(os.path.join(dirpath, fn))

    meta_rows: List[Dict[str, str]] = []
    byte_features: List[np.ndarray] = []
    struct_features: List[List[int]] = []
    labels: List[int] = []
    label_names: List[str] = []
    file_paths: List[str] = []

    for p in sorted(paths):
        label = get_label_from_path(p)
        y = 0 if label == "benign" else 1

        magic_type = detect_magic_type(p)
        claimed_type = ext_claimed_type(p)

        tiff_feat = parse_tiff_struct(p)
        if tiff_feat is None:
            tiff_feat = {
                "is_tiff": 0,
                "is_dng": 0,
                "min_width": 0,
                "min_height": 0,
                "ifd_entry_max": 0,
                "subifd_count_sum": 0,
                "new_subfile_types_unique": 0,
                "total_opcodes": 0,
                "unknown_opcodes": 0,
                "max_opcode_id": 0,
                "opcode_list1_bytes": 0,
                "opcode_list2_bytes": 0,
                "opcode_list3_bytes": 0,
                "max_declared_opcode_count": 0,
                "spp_max": 0,
                "compression_variety": 0,
                "tile_count_ratio": 0.0,
            }

        zip_eocd_near_end, zip_local_in_tail = zip_polyglot_flags(p)

        magika_label = ""
        if magika_session is not None:
            try:
                magika_label = magika_session.identify_path(p).output.label
            except Exception:
                magika_label = ""

        # Detection rules
        flag_opcode_anomaly = int(
            tiff_feat["is_dng"] == 1
            and (tiff_feat["total_opcodes"] > 100 or tiff_feat["unknown_opcodes"] > 0)
        )
        flag_tiny_dims_low_ifd = int(
            tiff_feat["is_tiff"] == 1
            and tiff_feat["is_dng"] == 0
            and tiff_feat["ifd_entry_max"] <= 10
            and (tiff_feat["min_width"] <= 16 or tiff_feat["min_height"] <= 16)
        )
        flag_zip_polyglot = int(zip_eocd_near_end == 1 and zip_local_in_tail == 1)
        flag_dng_jpeg_mismatch = int(
            (claimed_type == "jpeg")
            and (magic_type == "tiff")
            and (tiff_feat["is_dng"] == 1)
        )
        flag_magika_ext_mismatch = int(
            magika_label in ("dng", "tiff")
            and claimed_type == "jpeg"
        )

        flag_any = int(
            flag_opcode_anomaly
            or flag_tiny_dims_low_ifd
            or flag_zip_polyglot
            or flag_dng_jpeg_mismatch
            or flag_magika_ext_mismatch
        )

        # Magika-like byte features
        bytes_feat = magika_like_bytes(p)

        byte_features.append(bytes_feat)
        struct_features.append(
            [
                tiff_feat["is_tiff"],
                tiff_feat["is_dng"],
                tiff_feat["min_width"],
                tiff_feat["min_height"],
                tiff_feat["ifd_entry_max"],
                tiff_feat["subifd_count_sum"],
                tiff_feat["new_subfile_types_unique"],
                tiff_feat["total_opcodes"],
                tiff_feat["unknown_opcodes"],
                tiff_feat["max_opcode_id"],
                tiff_feat["opcode_list1_bytes"],
                tiff_feat["opcode_list2_bytes"],
                tiff_feat["opcode_list3_bytes"],
                tiff_feat["max_declared_opcode_count"],
                tiff_feat["spp_max"],
                tiff_feat["compression_variety"],
                tiff_feat["tile_count_ratio"],
                zip_eocd_near_end,
                zip_local_in_tail,
                flag_opcode_anomaly,
                flag_tiny_dims_low_ifd,
                flag_zip_polyglot,
                flag_dng_jpeg_mismatch,
                flag_magika_ext_mismatch,
                flag_any,
            ]
        )

        meta_rows.append(
            {
                "path": p,
                "label": label,
                "y": str(y),
                "size": str(os.path.getsize(p)),
                "claimed_type": claimed_type,
                "magic_type": magic_type,
                "magika_label": magika_label or "unavailable",
                "magika_available": str(1 if magika_session is not None else 0),
                "is_tiff": str(tiff_feat["is_tiff"]),
                "is_dng": str(tiff_feat["is_dng"]),
                "min_width": str(tiff_feat["min_width"]),
                "min_height": str(tiff_feat["min_height"]),
                "ifd_entry_max": str(tiff_feat["ifd_entry_max"]),
                "subifd_count_sum": str(tiff_feat["subifd_count_sum"]),
                "new_subfile_types_unique": str(tiff_feat["new_subfile_types_unique"]),
                "total_opcodes": str(tiff_feat["total_opcodes"]),
                "unknown_opcodes": str(tiff_feat["unknown_opcodes"]),
                "max_opcode_id": str(tiff_feat["max_opcode_id"]),
                "opcode_list1_bytes": str(tiff_feat["opcode_list1_bytes"]),
                "opcode_list2_bytes": str(tiff_feat["opcode_list2_bytes"]),
                "opcode_list3_bytes": str(tiff_feat["opcode_list3_bytes"]),
                "max_declared_opcode_count": str(tiff_feat["max_declared_opcode_count"]),
                "spp_max": str(tiff_feat["spp_max"]),
                "compression_variety": str(tiff_feat["compression_variety"]),
                "tile_count_ratio": str(tiff_feat["tile_count_ratio"]),
                "zip_eocd_near_end": str(zip_eocd_near_end),
                "zip_local_in_tail": str(zip_local_in_tail),
                "flag_opcode_anomaly": str(flag_opcode_anomaly),
                "flag_tiny_dims_low_ifd": str(flag_tiny_dims_low_ifd),
                "flag_zip_polyglot": str(flag_zip_polyglot),
                "flag_dng_jpeg_mismatch": str(flag_dng_jpeg_mismatch),
                "flag_magika_ext_mismatch": str(flag_magika_ext_mismatch),
                "flag_any": str(flag_any),
            }
        )

        labels.append(y)
        label_names.append(label)
        file_paths.append(p)

    # Write CSV
    csv_path = os.path.join(output_dir, "hybrid_features.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)

    # Write NPZ
    npz_path = os.path.join(output_dir, "hybrid_features.npz")
    struct_feature_names = [
        "is_tiff",
        "is_dng",
        "min_width",
        "min_height",
        "ifd_entry_max",
        "subifd_count_sum",
        "new_subfile_types_unique",
        "total_opcodes",
        "unknown_opcodes",
        "max_opcode_id",
        "opcode_list1_bytes",
        "opcode_list2_bytes",
        "opcode_list3_bytes",
        "max_declared_opcode_count",
        "spp_max",
        "compression_variety",
        "tile_count_ratio",
        "zip_eocd_near_end",
        "zip_local_in_tail",
        "flag_opcode_anomaly",
        "flag_tiny_dims_low_ifd",
        "flag_zip_polyglot",
        "flag_dng_jpeg_mismatch",
        "flag_magika_ext_mismatch",
        "flag_any",
    ]

    np.savez_compressed(
        npz_path,
        X_bytes=np.stack(byte_features, axis=0),
        X_struct=np.array(struct_features, dtype=np.float64),
        y=np.array(labels, dtype=np.int64),
        labels=np.array(label_names),
        paths=np.array(file_paths),
        struct_feature_names=np.array(struct_feature_names),
    )

    # Summary
    summary = {}
    for row in meta_rows:
        lab = row["label"]
        summary.setdefault(lab, {"count": 0, "flag_any": 0})
        summary[lab]["count"] += 1
        if row["flag_any"] == "1":
            summary[lab]["flag_any"] += 1

    print("Wrote:", csv_path)
    print("Wrote:", npz_path)
    print("Summary:", json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
