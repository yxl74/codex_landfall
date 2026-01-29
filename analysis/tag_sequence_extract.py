#!/usr/bin/env python3
"""Extract per-tag sequence features for DNG/TIFF anomaly models."""
from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

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

TAG_SUBIFD = 330
TAG_EXIF_IFD = 34665
TAG_GPS_IFD = 34853
TAG_INTEROP_IFD = 40965
TAG_DNG_VERSION = 50706
TAG_OPCODE_LIST1 = 51008
TAG_OPCODE_LIST2 = 51009
TAG_OPCODE_LIST3 = 51022

POINTER_TAGS = {TAG_SUBIFD, TAG_EXIF_IFD, TAG_GPS_IFD, TAG_INTEROP_IFD}

EXPECTED_TYPES: Dict[int, Tuple[int, ...]] = {
    256: (3, 4),  # ImageWidth
    257: (3, 4),  # ImageLength
    258: (3,),    # BitsPerSample
    259: (3,),    # Compression
    262: (3,),    # PhotometricInterpretation
    271: (2,),    # Make
    272: (2,),    # Model
    273: (3, 4),  # StripOffsets
    277: (3,),    # SamplesPerPixel
    278: (3, 4),  # RowsPerStrip
    279: (3, 4),  # StripByteCounts
    282: (5,),    # XResolution
    283: (5,),    # YResolution
    284: (3,),    # PlanarConfiguration
    296: (3,),    # ResolutionUnit
    305: (2,),    # Software
    306: (2,),    # DateTime
    513: (4,),    # JPEGInterchangeFormat
    514: (4,),    # JPEGInterchangeFormatLength
    TAG_SUBIFD: (4,),
    TAG_EXIF_IFD: (4,),
    TAG_GPS_IFD: (4,),
    TAG_INTEROP_IFD: (4,),
    TAG_DNG_VERSION: (1,),
    TAG_OPCODE_LIST1: (7, 1),
    TAG_OPCODE_LIST2: (7, 1),
    TAG_OPCODE_LIST3: (7, 1),
}

IFD_MAIN = 0
IFD_SUB = 1
IFD_EXIF = 2
IFD_GPS = 3
IFD_INTEROP = 4


@dataclass
class TagSequence:
    tag_ids: List[int]
    type_ids: List[int]
    ifd_kinds: List[int]
    features: List[List[float]]
    has_dng: bool
    total_tags: int
    ifd_count: int


def get_label_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if "LandFall" in parts:
        return "landfall"
    if "general_mal" in parts:
        return "general_mal"
    if "benign_data" in parts:
        return "benign"
    return "unknown"


def read_u16(mm: mmap.mmap, off: int, endian: str) -> Optional[int]:
    if off < 0 or off + 2 > len(mm):
        return None
    return struct.unpack_from(endian + "H", mm, off)[0]


def read_u32(mm: mmap.mmap, off: int, endian: str) -> Optional[int]:
    if off < 0 or off + 4 > len(mm):
        return None
    return struct.unpack_from(endian + "I", mm, off)[0]


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def log1p_clamped(x: float, max_log: float) -> float:
    return min(math.log1p(max(0.0, x)), max_log)


def parse_tag_sequence(path: str) -> Optional[TagSequence]:
    file_size = os.path.getsize(path)
    if file_size < 8:
        return None

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(mm) < 8:
                return None
            endian_bytes = mm[0:2]
            if endian_bytes == b"II":
                endian = "<"
            elif endian_bytes == b"MM":
                endian = ">"
            else:
                return None

            magic = read_u16(mm, 2, endian)
            if magic != 42:
                return None
            root = read_u32(mm, 4, endian)
            if root is None or root == 0 or root >= file_size:
                return None

            tag_ids: List[int] = []
            type_ids: List[int] = []
            ifd_kinds: List[int] = []
            feats: List[List[float]] = []
            has_dng = False
            total_tags = 0
            ifd_count = 0

            visited = set()
            queue: Deque[Tuple[int, int]] = deque()
            queue.append((root, IFD_MAIN))

            while queue:
                off, ifd_kind = queue.popleft()
                if off in visited or off == 0 or off + 2 > file_size:
                    continue
                visited.add(off)

                count = read_u16(mm, off, endian)
                if count is None:
                    continue

                ifd_count += 1
                total_tags += count
                base = off + 2
                if base + count * 12 > file_size:
                    continue

                prev_tag = None
                for i in range(count):
                    entry_off = base + i * 12
                    tag = read_u16(mm, entry_off, endian)
                    type_id = read_u16(mm, entry_off + 2, endian)
                    val_count = read_u32(mm, entry_off + 4, endian)
                    value_or_offset = read_u32(mm, entry_off + 8, endian)
                    if tag is None or type_id is None or val_count is None or value_or_offset is None:
                        continue

                    # TIFF type IDs are defined in 1..12. In the wild (and especially when
                    # parsers walk corrupted pointer graphs) you can see arbitrary 16-bit values.
                    # For ML, mapping those to a single UNK bucket avoids exploding the embedding.
                    type_id_raw = type_id
                    type_id = type_id if 1 <= type_id <= 12 else 13

                    if tag == TAG_DNG_VERSION:
                        has_dng = True

                    type_size = TIFF_TYPES.get(type_id_raw, 0)
                    byte_count = val_count * type_size
                    is_immediate = 1.0 if byte_count <= 4 else 0.0
                    offset_norm = 0.0
                    offset_valid = 1.0
                    if byte_count > 4:
                        offset_norm = clamp(value_or_offset / file_size, 0.0, 1.0)
                        offset_valid = 1.0 if (0 < value_or_offset and value_or_offset + byte_count <= file_size) else 0.0

                    coarse_bucket_norm = clamp((tag // 256) / 255.0, 0.0, 1.0)
                    log_count = log1p_clamped(val_count, 16.0) / 16.0
                    log_bytes = log1p_clamped(byte_count, 24.0) / 24.0

                    order_delta = 0.0
                    order_violation = 0.0
                    if prev_tag is not None:
                        delta = tag - prev_tag
                        order_violation = 1.0 if delta < 0 else 0.0
                        order_delta = clamp(delta, -128.0, 128.0) / 128.0
                    prev_tag = tag

                    is_pointer = 1.0 if tag in POINTER_TAGS else 0.0
                    if tag == TAG_SUBIFD:
                        is_pointer = 1.0

                    expected = EXPECTED_TYPES.get(tag)
                    type_mismatch = 0.0
                    if expected is not None and type_id_raw not in expected:
                        type_mismatch = 1.0

                    type_norm = clamp(type_id_raw / 12.0, 0.0, 1.0)
                    ifd_kind_norm = clamp(ifd_kind / 4.0, 0.0, 1.0)

                    tag_ids.append(tag)
                    type_ids.append(type_id)
                    ifd_kinds.append(ifd_kind)
                    feats.append([
                        coarse_bucket_norm,
                        log_count,
                        log_bytes,
                        offset_norm,
                        offset_valid,
                        is_immediate,
                        is_pointer,
                        order_delta,
                        order_violation,
                        type_mismatch,
                        type_norm,
                        ifd_kind_norm,
                    ])

                    if tag in POINTER_TAGS and byte_count >= 4 and value_or_offset < file_size:
                        queue.append((value_or_offset, _ifd_kind_for_tag(tag)))
                    if tag == TAG_SUBIFD:
                        if byte_count <= 4:
                            if value_or_offset < file_size:
                                queue.append((value_or_offset, IFD_SUB))
                        else:
                            for j in range(val_count):
                                sub_off = read_u32(mm, value_or_offset + j * 4, endian)
                                if sub_off and sub_off < file_size:
                                    queue.append((sub_off, IFD_SUB))

                next_ifd = read_u32(mm, base + count * 12, endian)
                if next_ifd and next_ifd < file_size:
                    queue.append((next_ifd, ifd_kind))

            if not tag_ids:
                return None

            return TagSequence(
                tag_ids=tag_ids,
                type_ids=type_ids,
                ifd_kinds=ifd_kinds,
                features=feats,
                has_dng=has_dng,
                total_tags=total_tags,
                ifd_count=ifd_count,
            )
        finally:
            mm.close()


def _ifd_kind_for_tag(tag: int) -> int:
    if tag == TAG_EXIF_IFD:
        return IFD_EXIF
    if tag == TAG_GPS_IFD:
        return IFD_GPS
    if tag == TAG_INTEROP_IFD:
        return IFD_INTEROP
    return IFD_SUB


def iter_paths(paths: Iterable[str]) -> Iterable[str]:
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    yield str(child)
        elif path.is_file():
            yield str(path)


def load_list_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def pad_sequence(seq: List, max_len: int, pad_value) -> List:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-file", help="Text file with paths to process")
    ap.add_argument("--input", nargs="*", help="Files or directories to scan")
    ap.add_argument("--output", required=True, help="Output npz path")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--dng-only", action="store_true", help="Keep only files with DNGVersion tag")
    ap.add_argument("--include-unknown", action="store_true", help="Include files without labels")
    args = ap.parse_args()

    paths: List[str] = []
    if args.list_file:
        paths.extend(load_list_file(args.list_file))
    if args.input:
        paths.extend(args.input)

    if not paths:
        raise SystemExit("No input paths provided")

    sequences = []
    tag_ids = []
    type_ids = []
    ifd_kinds = []
    lengths = []
    labels = []
    file_features = []
    kept_paths = []

    for path in iter_paths(paths):
        seq = parse_tag_sequence(path)
        if seq is None:
            continue
        if args.dng_only and not seq.has_dng:
            continue
        label = get_label_from_path(path)
        if label == "unknown" and not args.include_unknown:
            continue

        seq_len = len(seq.tag_ids)
        lengths.append(min(seq_len, args.max_seq_len))
        kept_paths.append(path)
        labels.append(label)

        file_size = os.path.getsize(path)
        file_features.append([
            log1p_clamped(file_size, 32.0) / 32.0,
            min(seq.total_tags, 2048) / 2048.0,
            min(seq.ifd_count, 64) / 64.0,
        ])

        tag_ids.append(pad_sequence(seq.tag_ids, args.max_seq_len, 0))
        type_ids.append(pad_sequence(seq.type_ids, args.max_seq_len, 0))
        ifd_kinds.append(pad_sequence(seq.ifd_kinds, args.max_seq_len, 0))
        features = [np.array(f, dtype=np.float32) for f in seq.features]
        if seq_len < args.max_seq_len:
            pad_row = np.zeros((len(seq.features[0]),), dtype=np.float32)
            features.extend([pad_row] * (args.max_seq_len - seq_len))
        else:
            features = features[: args.max_seq_len]
        sequences.append(features)

    np.savez_compressed(
        args.output,
        features=np.array(sequences, dtype=np.float32),
        tag_ids=np.array(tag_ids, dtype=np.int32),
        type_ids=np.array(type_ids, dtype=np.int32),
        ifd_kinds=np.array(ifd_kinds, dtype=np.int32),
        lengths=np.array(lengths, dtype=np.int32),
        labels=np.array(labels, dtype=object),
        paths=np.array(kept_paths, dtype=object),
        file_features=np.array(file_features, dtype=np.float32),
        schema=json.dumps(get_schema(), indent=2),
    )


def get_schema() -> Dict[str, object]:
    return {
        "features": [
            "coarse_tag_bucket_norm",
            "log1p_count_norm",
            "log1p_byte_count_norm",
            "offset_norm",
            "offset_valid",
            "is_immediate",
            "is_pointer",
            "order_delta_norm",
            "order_violation",
            "type_mismatch",
            "type_id_norm",
            "ifd_kind_norm",
        ],
        "tag_ids": "raw tag ids for embedding",
        "type_ids": "raw type ids for embedding",
        "ifd_kinds": "0 main, 1 subifd, 2 exif, 3 gps, 4 interop",
        "lengths": "sequence length before padding",
        "file_features": [
            "log1p_file_size_norm",
            "total_tags_norm",
            "ifd_count_norm",
        ],
        "normalization": {
            "log1p_count_norm": "log1p(count) / 16, clipped at 16",
            "log1p_byte_count_norm": "log1p(count*type_size) / 24, clipped at 24",
            "offset_norm": "offset/file_size clipped to [0,1], 0 for immediate",
            "order_delta_norm": "clip(tag_delta, -128, 128) / 128",
            "type_id_norm": "type_id / 12",
            "ifd_kind_norm": "ifd_kind / 4",
            "file_size_norm": "log1p(file_size)/32, clipped at 32",
            "total_tags_norm": "min(total_tags, 2048)/2048",
            "ifd_count_norm": "min(ifd_count, 64)/64",
        },
    }


if __name__ == "__main__":
    main()
