#!/usr/bin/env python3
"""
Compare on-device TFLite scores vs local TFLite scores line-by-line.

Inputs:
  - Device logcat output containing "HybridDetector: score=..."
  - Device bench list (paths in /sdcard/Android/data/.../bench_full)
  - Local mapping for benign_100 samples (outputs/bench_benign_100_map.txt)
  - Local TFLite model + meta json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import struct
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np


TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
}

TAG_WIDTH = 256
TAG_HEIGHT = 257
TAG_SUBIFD = 330
TAG_EXIF_IFD = 34665
TAG_DNG_VERSION = 50706
TAG_NEW_SUBFILE_TYPE = 254
TAG_OPCODE_LIST1 = 51008
TAG_OPCODE_LIST2 = 51009
TAG_OPCODE_LIST3 = 51022


def load_tflite_interpreter(model_path: str, feature_dim: int):
    try:
        import tensorflow as tf  # type: ignore

        interpreter = tf.lite.Interpreter(model_path=model_path)
    except Exception:
        try:
            import tflite_runtime.interpreter as tflite  # type: ignore

            interpreter = tflite.Interpreter(model_path=model_path)
        except Exception as exc:
            raise RuntimeError(
                "TensorFlow/TFLite runtime not available. "
                "Use .venv-tf/bin/python or install tflite_runtime."
            ) from exc

    input_details = interpreter.get_input_details()[0]
    input_index = input_details["index"]
    interpreter.resize_tensor_input(input_index, [1, feature_dim])
    interpreter.allocate_tensors()
    output_index = interpreter.get_output_details()[0]["index"]
    return interpreter, input_index, output_index


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


def magika_bytes_to_features(x_bytes: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return x_bytes.astype(np.float32) / 256.0
    if mode != "hist":
        raise ValueError(f"Unknown bytes mode: {mode}")

    beg = x_bytes[:1024]
    end = x_bytes[1024:]
    bins = 257
    feats = []
    for block in (beg, end):
        counts = np.bincount(block.astype(np.int64), minlength=bins)
        feats.append(counts.astype(np.float32) / float(block.shape[0]))
    return np.concatenate(feats, axis=0)


def extract_bytes_hist_features(path: str) -> np.ndarray:
    file_size = os.path.getsize(path)
    if file_size == 0:
        return np.zeros(514, dtype=np.float32)

    block_size = min(4096, file_size)
    with open(path, "rb") as f:
        beg_block = f.read(block_size)
        beg = strip_prefix(beg_block)
        if file_size >= block_size:
            f.seek(max(0, file_size - block_size))
        end_block = f.read(block_size)
        end = strip_suffix(end_block)

    def histogram1024(data: bytes) -> np.ndarray:
        hist = np.zeros(257, dtype=np.float32)
        length = min(1024, len(data))
        pad = 1024 - length
        hist[256] = float(pad)
        for i in range(length):
            hist[data[i]] += 1.0
        hist /= 1024.0
        return hist

    beg_hist = histogram1024(beg)
    end_hist = histogram1024(end)
    return np.concatenate([beg_hist, end_hist], axis=0)


def parse_device_scores(log_text: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    pattern = re.compile(r"score=([0-9eE+\\.-]+).*? (/.+)$")
    for line in log_text.splitlines():
        if "HybridDetector" not in line or "score=" not in line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        score = float(match.group(1))
        path = match.group(2).strip()
        scores[path] = score
    return scores


def load_benign_map(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            local, device = parts
            mapping[device] = local
    return mapping


def device_to_local(
    device_path: str,
    device_base: str,
    data_root: str,
    benign_map: Dict[str, str],
) -> Optional[str]:
    if device_path in benign_map:
        return benign_map[device_path]
    landfall_prefix = f"{device_base}/LandFall/"
    gen_prefix = f"{device_base}/general_mal/"
    if device_path.startswith(landfall_prefix):
        rel = device_path[len(landfall_prefix) :]
        return os.path.join(data_root, "LandFall", rel)
    if device_path.startswith(gen_prefix):
        rel = device_path[len(gen_prefix) :]
        return os.path.join(data_root, "general_mal", rel)
    return None


def read_u16(f, offset: int, order: str) -> Optional[int]:
    f.seek(offset)
    buf = f.read(2)
    if len(buf) < 2:
        return None
    return struct.unpack(order + "H", buf)[0]


def to_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return value - 0x100000000
    return value


def read_u32(f, offset: int, order: str) -> Optional[int]:
    f.seek(offset)
    buf = f.read(4)
    if len(buf) < 4:
        return None
    return to_int32(struct.unpack(order + "I", buf)[0])


def read_u32_array(f, offset: int, count: int, order: str) -> List[int]:
    f.seek(offset)
    buf = f.read(count * 4)
    if len(buf) < count * 4:
        return []
    fmt = order + ("I" * count)
    return [to_int32(x) for x in struct.unpack(fmt, buf)]


def read_u16_array(f, offset: int, count: int, order: str) -> List[int]:
    f.seek(offset)
    buf = f.read(count * 2)
    if len(buf) < count * 2:
        return []
    fmt = order + ("H" * count)
    return list(struct.unpack(fmt, buf))


def read_values(
    f,
    order: str,
    file_size: int,
    type_id: int,
    val_count: int,
    value_or_offset: int,
) -> List[int]:
    if val_count <= 0:
        return []
    size_bytes = TYPE_SIZES.get(type_id, 1) * val_count
    if size_bytes <= 4:
        buf = struct.pack(order + "i", to_int32(value_or_offset))
        if type_id == 3:
            out = []
            for i in range(val_count):
                out.append(struct.unpack(order + "H", buf[i * 2 : i * 2 + 2])[0])
            return out
        if type_id == 4:
            return [struct.unpack(order + "i", buf)[0]]
        return []
    off = value_or_offset
    if off + size_bytes > file_size:
        return []
    if type_id == 3:
        return read_u16_array(f, off, val_count, order)
    if type_id == 4:
        return read_u32_array(f, off, val_count, order)
    return []


def read_bytes(f, offset: int, size: int) -> bytes:
    f.seek(offset)
    return f.read(size)


def read_u32be(buf: bytes, offset: int) -> int:
    value = (
        ((buf[offset] & 0xFF) << 24)
        | ((buf[offset + 1] & 0xFF) << 16)
        | ((buf[offset + 2] & 0xFF) << 8)
        | (buf[offset + 3] & 0xFF)
    )
    return to_int32(value)


def magic_type(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(16)
    if len(head) < 4:
        return "unknown"
    is_tiff = (head[0] == ord("I") and head[1] == ord("I") and head[2] == 0x2A) or (
        head[0] == ord("M") and head[1] == ord("M") and head[3] == 0x2A
    )
    if is_tiff:
        return "tiff"
    if head[0] == 0xFF and head[1] == 0xD8 and head[2] == 0xFF:
        return "jpeg"
    if head[0] == 0x89 and head[1] == 0x50:
        return "png"
    if head[0] == 0x47 and head[1] == 0x49:
        return "gif"
    if head[0] == 0x42 and head[1] == 0x4D:
        return "bmp"
    if head[0] == 0x50 and head[1] == 0x4B:
        return "zip"
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


def zip_polyglot_flags(f, size: int) -> Tuple[int, int]:
    tail_window = 1024 * 1024
    eocd_tail = 65536
    start = max(0, size - tail_window)
    tail_size = size - start
    f.seek(start)
    tail = f.read(tail_size)

    eocd_pos = tail.find(b"PK\x05\x06")
    if eocd_pos == -1:
        return 0, 0
    eocd_abs = start + eocd_pos
    if size - eocd_abs > eocd_tail:
        return 0, 0
    local_pos = tail.find(b"PK\x03\x04")
    return 1, 1 if local_pos != -1 else 0


def parse_tiff_struct(path: str) -> Optional[Dict[str, int]]:
    size = os.path.getsize(path)
    if size < 8:
        return None
    with open(path, "rb") as f:
        endian = f.read(2)
        if endian == b"II":
            order = "<"
        elif endian == b"MM":
            order = ">"
        else:
            return None

        magic = read_u16(f, 2, order)
        if magic != 42:
            return None
        root = read_u32(f, 4, order)
        if root is None or root <= 0:
            return None

        is_dng = 0
        widths: List[int] = []
        heights: List[int] = []
        ifd_entry_max = 0
        subifd_offsets: List[int] = []
        exif_offset = 0
        new_subfile_types: set[int] = set()
        opcode_lists: Dict[int, Tuple[int, int]] = {}

        visited = set()
        stack: List[int] = [root]

        def parse_stack() -> None:
            nonlocal is_dng, ifd_entry_max, exif_offset
            while stack:
                off = stack.pop()
                if off <= 0 or off >= size or off in visited:
                    continue
                visited.add(off)
                count = read_u16(f, off, order)
                if count is None or count <= 0:
                    continue
                ifd_entry_max = max(ifd_entry_max, count)
                entry_base = off + 2
                for i in range(count):
                    entry_off = entry_base + i * 12
                    if entry_off + 12 > size:
                        break
                    tag = read_u16(f, entry_off, order)
                    type_id = read_u16(f, entry_off + 2, order)
                    val_count = read_u32(f, entry_off + 4, order)
                    value_or_offset = read_u32(f, entry_off + 8, order)
                    if tag is None or type_id is None or val_count is None or value_or_offset is None:
                        continue
                    if tag == 0:
                        continue
                    if tag == TAG_DNG_VERSION:
                        is_dng = 1
                    if tag in (TAG_WIDTH, TAG_HEIGHT, TAG_NEW_SUBFILE_TYPE):
                        vals = read_values(f, order, size, type_id, val_count, value_or_offset)
                        if vals:
                            if tag == TAG_WIDTH:
                                widths.extend(vals)
                            elif tag == TAG_HEIGHT:
                                heights.extend(vals)
                            else:
                                new_subfile_types.update(vals)
                    if tag == TAG_SUBIFD:
                        size_bytes = TYPE_SIZES.get(type_id, 1) * val_count
                        if size_bytes <= 4:
                            subifd_offsets.append(value_or_offset)
                        else:
                            offs = read_u32_array(f, value_or_offset, val_count, order)
                            subifd_offsets.extend(offs)
                    if tag == TAG_EXIF_IFD:
                        exif_offset = value_or_offset
                    if tag in (TAG_OPCODE_LIST1, TAG_OPCODE_LIST2, TAG_OPCODE_LIST3):
                        size_bytes = TYPE_SIZES.get(type_id, 1) * val_count
                        opcode_lists[tag] = (value_or_offset, size_bytes)

                next_ifd = read_u32(f, entry_base + count * 12, order)
                if next_ifd:
                    stack.append(next_ifd)

        parse_stack()
        for off in subifd_offsets:
            stack.append(off)
        if exif_offset > 0:
            stack.append(exif_offset)
        parse_stack()

        min_width = min(widths) if widths else 0
        min_height = min(heights) if heights else 0

        total_opcodes = 0
        unknown_opcodes = 0
        max_opcode_id = 0
        list1_bytes = 0
        list2_bytes = 0
        list3_bytes = 0

        for tag, pair in opcode_lists.items():
            offset, size_bytes = pair
            if offset <= 0 or size_bytes < 4 or offset + size_bytes > size:
                continue
            buf = read_bytes(f, offset, size_bytes)
            if len(buf) < 4:
                continue
            opcode_count = read_u32be(buf, 0)
            pos = 4
            parsed = 0
            while parsed < opcode_count and pos + 16 <= len(buf):
                opcode_id = read_u32be(buf, pos)
                data_size = read_u32be(buf, pos + 12)
                pos += 16
                if pos + data_size > len(buf):
                    break
                parsed += 1
                total_opcodes += 1
                max_opcode_id = max(max_opcode_id, opcode_id)
                if opcode_id > 14:
                    unknown_opcodes += 1
                pos += data_size
            if tag == TAG_OPCODE_LIST1:
                list1_bytes = size_bytes
            if tag == TAG_OPCODE_LIST2:
                list2_bytes = size_bytes
            if tag == TAG_OPCODE_LIST3:
                list3_bytes = size_bytes

        return {
            "is_tiff": 1,
            "is_dng": is_dng,
            "min_width": min_width,
            "min_height": min_height,
            "ifd_entry_max": ifd_entry_max,
            "subifd_count_sum": len(subifd_offsets),
            "new_subfile_types_unique": len(new_subfile_types),
            "total_opcodes": total_opcodes,
            "unknown_opcodes": unknown_opcodes,
            "max_opcode_id": max_opcode_id,
            "opcode_list1_bytes": list1_bytes,
            "opcode_list2_bytes": list2_bytes,
            "opcode_list3_bytes": list3_bytes,
        }


def extract_features(
    path: str,
    bytes_mode: str,
    struct_feature_names: List[str],
) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    if bytes_mode == "hist":
        byte_features = extract_bytes_hist_features(path)
    elif bytes_mode == "raw":
        x_bytes = magika_like_bytes(path)
        byte_features = magika_bytes_to_features(x_bytes, bytes_mode)
    else:
        raise ValueError(f"Unknown bytes mode: {bytes_mode}")

    tiff_feat = parse_tiff_struct(path)
    if tiff_feat is None:
        struct_map = {name: 0 for name in struct_feature_names}
    else:
        magic = magic_type(path)
        claimed = ext_claimed_type(path)
        with open(path, "rb") as f:
            zip_eocd_near_end, zip_local_in_tail = zip_polyglot_flags(f, os.path.getsize(path))
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
            (claimed == "jpeg")
            and (magic == "tiff")
            and (tiff_feat["is_dng"] == 1)
        )
        flag_magika_ext_mismatch = 0
        flag_any = int(
            flag_opcode_anomaly
            or flag_tiny_dims_low_ifd
            or flag_zip_polyglot
            or flag_dng_jpeg_mismatch
            or flag_magika_ext_mismatch
        )

        struct_map = dict(tiff_feat)
        struct_map.update(
            {
                "zip_eocd_near_end": zip_eocd_near_end,
                "zip_local_in_tail": zip_local_in_tail,
                "flag_opcode_anomaly": flag_opcode_anomaly,
                "flag_tiny_dims_low_ifd": flag_tiny_dims_low_ifd,
                "flag_zip_polyglot": flag_zip_polyglot,
                "flag_dng_jpeg_mismatch": flag_dng_jpeg_mismatch,
                "flag_magika_ext_mismatch": flag_magika_ext_mismatch,
                "flag_any": flag_any,
            }
        )

    struct_features = np.array(
        [struct_map.get(name, 0) for name in struct_feature_names], dtype=np.float32
    )
    return np.concatenate([byte_features, struct_features], axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-log", help="Logcat file with HybridDetector lines")
    parser.add_argument("--adb-logcat", action="store_true", help="Read from adb logcat -d")
    parser.add_argument("--device-list", default="outputs/bench_list_device_full.txt")
    parser.add_argument("--benign-map", default="outputs/bench_benign_100_map.txt")
    parser.add_argument(
        "--device-base",
        default="/sdcard/Android/data/com.landfall.hybriddetector/files/bench_full",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model", default="outputs/hybrid_model.tflite")
    parser.add_argument("--meta", default="outputs/hybrid_model_meta.json")
    parser.add_argument("--output", default="outputs/device_local_compare.csv")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    bytes_mode = meta["bytes_mode"]
    struct_feature_names = list(meta["struct_feature_names"])
    feature_dim = int(meta["feature_dim"])

    if args.device_log:
        with open(args.device_log, "r", encoding="utf-8", errors="ignore") as f:
            log_text = f.read()
    elif args.adb_logcat:
        log_text = subprocess.check_output(["adb", "logcat", "-d"], text=True, errors="ignore")
    else:
        raise SystemExit("Provide --device-log or --adb-logcat.")

    device_scores = parse_device_scores(log_text)
    if not device_scores:
        raise SystemExit("No device scores found in log.")

    benign_map = load_benign_map(args.benign_map)
    with open(args.device_list, "r", encoding="utf-8") as f:
        device_paths = [line.strip() for line in f if line.strip()]

    interpreter, input_index, output_index = load_tflite_interpreter(args.model, feature_dim)

    rows = []
    diffs = []
    missing_device = 0
    missing_local = 0
    for device_path in device_paths:
        local_path = device_to_local(
            device_path, args.device_base, args.data_root, benign_map
        )
        if local_path is None or not os.path.exists(local_path):
            missing_local += 1
            rows.append(
                {
                    "device_path": device_path,
                    "local_path": local_path or "",
                    "device_score": "",
                    "local_score": "",
                    "abs_diff": "",
                    "within_tol": "false",
                    "note": "missing_local",
                }
            )
            continue

        features = extract_features(local_path, bytes_mode, struct_feature_names)
        if features is None or features.shape[0] != feature_dim:
            missing_local += 1
            rows.append(
                {
                    "device_path": device_path,
                    "local_path": local_path,
                    "device_score": "",
                    "local_score": "",
                    "abs_diff": "",
                    "within_tol": "false",
                    "note": "feature_error",
                }
            )
            continue

        input_data = features.reshape(1, -1).astype(np.float32)
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        local_score = float(output.reshape(-1)[0])

        device_score = device_scores.get(device_path)
        if device_score is None:
            missing_device += 1
            rows.append(
                {
                    "device_path": device_path,
                    "local_path": local_path,
                    "device_score": "",
                    "local_score": f"{local_score:.8f}",
                    "abs_diff": "",
                    "within_tol": "false",
                    "note": "missing_device",
                }
            )
            continue

        diff = abs(device_score - local_score)
        diffs.append(diff)
        rows.append(
            {
                "device_path": device_path,
                "local_path": local_path,
                "device_score": f"{device_score:.8f}",
                "local_score": f"{local_score:.8f}",
                "abs_diff": f"{diff:.8f}",
                "within_tol": "true" if diff <= args.tolerance else "false",
                "note": "",
            }
        )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "device_path",
                "local_path",
                "device_score",
                "local_score",
                "abs_diff",
                "within_tol",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    compared = len(diffs)
    max_diff = max(diffs) if diffs else 0.0
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    print(f"Rows: {len(rows)}")
    print(f"Compared: {compared}")
    print(f"Missing local: {missing_local}")
    print(f"Missing device: {missing_device}")
    print(f"Max abs diff: {max_diff:.8f}")
    print(f"Mean abs diff: {mean_diff:.8f}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
