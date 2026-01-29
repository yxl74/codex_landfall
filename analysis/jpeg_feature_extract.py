#!/usr/bin/env python3
"""
JPEG feature extraction for malicious JPEG detection experiments.

This mirrors the TIFF/DNG pipeline style:
  - Extracts Magika-like head/tail byte tokens (2048 int16 values)
  - Extracts fast structural JPEG features (marker stats + basic validity checks)
  - Writes NPZ + CSV for downstream ML training/evaluation

Datasets (defaults):
  - Benign JPEGs: JPG_dataset/
  - Malicious JPEGs: JPEG_malware/
"""

from __future__ import annotations

import argparse
import csv
import mmap
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


PADDING_TOKEN = 256  # matches existing magika-like pipelines (0..256)

# Markers that start a frame header (SOF*). See JPEG spec.
SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3,
    0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB,
    0xCD, 0xCE, 0xCF,
}


def iter_paths(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def head_tail_tokens(
    path: str,
    beg_size: int = 1024,
    end_size: int = 1024,
    block_size: int = 4096,
    padding_token: int = PADDING_TOKEN,
) -> np.ndarray:
    """Return 2048 int16 tokens: first/last 1024 bytes with padding token."""
    file_size = os.path.getsize(path)
    if file_size <= 0:
        return np.full((beg_size + end_size,), padding_token, dtype=np.int16)

    buf_size = min(block_size, file_size)
    out = np.full((beg_size + end_size,), padding_token, dtype=np.int16)

    with open(path, "rb") as f:
        beg_block = f.read(buf_size)
        if beg_block:
            beg = beg_block[:beg_size]
            out[: len(beg)] = np.frombuffer(beg, dtype=np.uint8).astype(np.int16)

        # Tail
        if file_size > buf_size:
            f.seek(file_size - buf_size)
        end_block = f.read(buf_size)
        if end_block:
            end = end_block[-end_size:]
            out[beg_size : beg_size + len(end)] = np.frombuffer(end, dtype=np.uint8).astype(np.int16)

    return out


def _u16be(mm: mmap.mmap, off: int) -> Optional[int]:
    if off + 2 > len(mm):
        return None
    return (mm[off] << 8) | mm[off + 1]


def _tail_has_magic(tail: bytes) -> Tuple[int, int, int]:
    tail_zip = int((b"PK\x03\x04" in tail) or (b"PK\x05\x06" in tail))
    tail_pdf = int(b"%PDF" in tail)
    tail_elf = int(b"\x7fELF" in tail)
    return tail_zip, tail_pdf, tail_elf


def parse_jpeg_struct(path: str) -> Optional[Dict[str, float]]:
    """Parse JPEG header structure up to first SOS and extract robust features."""
    size = os.path.getsize(path)
    if size < 4:
        return None

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if mm[0:2] != b"\xff\xd8":
                return None

            n = size

            eoi_pos = mm.rfind(b"\xff\xd9")
            has_eoi = 1 if eoi_pos != -1 else 0
            bytes_after_eoi = (n - (eoi_pos + 2)) if has_eoi else n
            bytes_after_eoi_ratio_permille = (
                int(bytes_after_eoi * 1000 / n) if n > 0 else 0
            )

            tail = mm[max(0, n - min(4096, n)) :]
            tail_zip, tail_pdf, tail_elf = _tail_has_magic(tail)

            # Marker stats (pre-SOS)
            app_count = 0
            app_bytes = 0
            app_max_len = 0
            com_count = 0
            com_bytes = 0
            dqt_count = 0
            dqt_bytes = 0
            dqt_max_len = 0
            dht_count = 0
            dht_bytes = 0
            dht_max_len = 0
            sof0_count = 0
            sof2_count = 0
            sos_count = 0
            max_seg_len = 0

            dqt_tables = 0
            dqt_invalid = 0
            dht_tables = 0
            dht_invalid = 0

            width = 0
            height = 0
            components = 0
            precision = 0
            dri_interval = 0
            invalid_len = 0

            pos = 2
            while pos < n - 1:
                if mm[pos] != 0xFF:
                    pos += 1
                    continue

                # Skip 0xFF fill bytes
                while pos < n and mm[pos] == 0xFF:
                    pos += 1
                if pos >= n:
                    break

                marker = mm[pos]
                pos += 1

                if marker == 0x00:
                    # stuffed in entropy-coded data; ignore here
                    continue

                if marker == 0xD9:
                    # EOI
                    break

                if 0xD0 <= marker <= 0xD7:
                    # RST markers (no length)
                    continue

                # Track common marker counts
                if 0xE0 <= marker <= 0xEF:
                    app_count += 1
                if marker == 0xFE:
                    com_count += 1
                if marker == 0xDB:
                    dqt_count += 1
                if marker == 0xC4:
                    dht_count += 1
                if marker == 0xC0:
                    sof0_count += 1
                if marker == 0xC2:
                    sof2_count += 1
                if marker == 0xDA:
                    sos_count += 1

                # no-length markers
                if marker in (0xD8, 0x01):
                    continue

                seglen = _u16be(mm, pos)
                if seglen is None or seglen < 2:
                    invalid_len += 1
                    break
                seg_end = pos + seglen
                if seg_end > n:
                    invalid_len += 1
                    break

                if seglen > max_seg_len:
                    max_seg_len = seglen

                # Marker-specific byte stats
                if 0xE0 <= marker <= 0xEF:
                    app_bytes += seglen
                    if seglen > app_max_len:
                        app_max_len = seglen
                if marker == 0xFE:
                    com_bytes += seglen
                if marker == 0xDB:
                    dqt_bytes += seglen
                    if seglen > dqt_max_len:
                        dqt_max_len = seglen
                if marker == 0xC4:
                    dht_bytes += seglen
                    if seglen > dht_max_len:
                        dht_max_len = seglen

                payload_off = pos + 2
                payload_len = seglen - 2

                # Parse SOF dimensions (first SOF only)
                if marker in SOF_MARKERS and width == 0 and height == 0 and payload_len >= 6:
                    precision = mm[payload_off]
                    h = _u16be(mm, payload_off + 1)
                    w = _u16be(mm, payload_off + 3)
                    if h is not None:
                        height = h
                    if w is not None:
                        width = w
                    components = mm[payload_off + 5]

                # DRI: restart interval
                if marker == 0xDD and payload_len >= 2:
                    v = _u16be(mm, payload_off)
                    if v is not None:
                        dri_interval = v

                # DQT table parsing
                if marker == 0xDB:
                    off = payload_off
                    end = payload_off + payload_len
                    while off < end:
                        if off >= end:
                            break
                        pq_tq = mm[off]
                        off += 1
                        pq = (pq_tq >> 4) & 0x0F
                        table_bytes = 64 * (2 if pq else 1)
                        if off + table_bytes > end:
                            dqt_invalid += 1
                            break
                        dqt_tables += 1
                        off += table_bytes

                # DHT table parsing
                if marker == 0xC4:
                    off = payload_off
                    end = payload_off + payload_len
                    while off < end:
                        if off + 1 + 16 > end:
                            dht_invalid += 1
                            break
                        off += 1  # tc/th
                        counts16 = mm[off : off + 16]
                        off += 16
                        total_syms = int(sum(counts16))
                        if off + total_syms > end:
                            dht_invalid += 1
                            break
                        dht_tables += 1
                        off += total_syms

                # SOS starts entropy-coded data
                if marker == 0xDA:
                    break

                pos = seg_end

            dht_bytes_ratio_permille = int(dht_bytes * 1000 / n) if n > 0 else 0
            dqt_bytes_ratio_permille = int(dqt_bytes * 1000 / n) if n > 0 else 0
            app_bytes_ratio_permille = int(app_bytes * 1000 / n) if n > 0 else 0

            return {
                "file_size": float(n),
                "has_eoi": float(has_eoi),
                "bytes_after_eoi": float(bytes_after_eoi),
                "bytes_after_eoi_ratio_permille": float(bytes_after_eoi_ratio_permille),
                "tail_zip_magic": float(tail_zip),
                "tail_pdf_magic": float(tail_pdf),
                "tail_elf_magic": float(tail_elf),
                "invalid_len": float(invalid_len),
                "width": float(width),
                "height": float(height),
                "components": float(components),
                "precision": float(precision),
                "dri_interval": float(dri_interval),
                "app_count": float(app_count),
                "app_bytes": float(app_bytes),
                "app_bytes_ratio_permille": float(app_bytes_ratio_permille),
                "app_max_len": float(app_max_len),
                "com_count": float(com_count),
                "com_bytes": float(com_bytes),
                "dqt_count": float(dqt_count),
                "dqt_bytes": float(dqt_bytes),
                "dqt_bytes_ratio_permille": float(dqt_bytes_ratio_permille),
                "dqt_max_len": float(dqt_max_len),
                "dht_count": float(dht_count),
                "dht_bytes": float(dht_bytes),
                "dht_bytes_ratio_permille": float(dht_bytes_ratio_permille),
                "dht_max_len": float(dht_max_len),
                "sof0_count": float(sof0_count),
                "sof2_count": float(sof2_count),
                "sos_count": float(sos_count),
                "max_seg_len": float(max_seg_len),
                "dqt_tables": float(dqt_tables),
                "dqt_invalid": float(dqt_invalid),
                "dht_tables": float(dht_tables),
                "dht_invalid": float(dht_invalid),
            }
        finally:
            mm.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign-root", default="JPG_dataset", help="Benign JPEG root directory")
    parser.add_argument("--malware-root", default="JPEG_malware", help="Malicious JPEG directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--list-file", default="", help="Optional list of file paths to process")
    parser.add_argument("--limit", type=int, default=0, help="Optional max files (for quick tests)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paths: List[str] = []
    labels: List[str] = []
    y: List[int] = []

    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p or not os.path.isfile(p):
                    continue
                paths.append(p)
                # best-effort label
                if os.path.normpath(args.malware_root) in os.path.normpath(p):
                    labels.append("malware")
                    y.append(1)
                else:
                    labels.append("benign")
                    y.append(0)
    else:
        for p in iter_paths(args.benign_root):
            paths.append(p)
            labels.append("benign")
            y.append(0)
        for p in iter_paths(args.malware_root):
            paths.append(p)
            labels.append("malware")
            y.append(1)

    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
        labels = labels[: args.limit]
        y = y[: args.limit]

    meta_rows: List[Dict[str, str]] = []
    byte_features: List[np.ndarray] = []
    struct_features: List[List[float]] = []

    struct_feature_names: Optional[List[str]] = None

    kept_paths: List[str] = []
    kept_labels: List[str] = []
    kept_y: List[int] = []

    for p, lab, yi in zip(paths, labels, y):
        feat = parse_jpeg_struct(p)
        if feat is None:
            continue
        if struct_feature_names is None:
            struct_feature_names = list(feat.keys())
        # Enforce stable ordering
        vec = [float(feat[name]) for name in struct_feature_names]
        struct_features.append(vec)
        byte_features.append(head_tail_tokens(p))

        kept_paths.append(p)
        kept_labels.append(lab)
        kept_y.append(yi)

        meta_rows.append(
            {
                "path": p,
                "label": lab,
                "y": str(yi),
                "file_size": str(int(os.path.getsize(p))),
                "bytes_after_eoi": str(int(feat["bytes_after_eoi"])),
                "bytes_after_eoi_ratio_permille": str(int(feat["bytes_after_eoi_ratio_permille"])),
                "dht_bytes": str(int(feat["dht_bytes"])),
                "dqt_bytes": str(int(feat["dqt_bytes"])),
                "dht_tables": str(int(feat["dht_tables"])),
                "dqt_tables": str(int(feat["dqt_tables"])),
                "tail_zip_magic": str(int(feat["tail_zip_magic"])),
            }
        )

    if struct_feature_names is None:
        raise SystemExit("No JPEG files parsed successfully.")

    # Write CSV
    csv_path = os.path.join(args.output_dir, "jpeg_features.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)

    # Write NPZ
    npz_path = os.path.join(args.output_dir, "jpeg_features.npz")
    np.savez_compressed(
        npz_path,
        X_bytes=np.stack(byte_features, axis=0),
        X_struct=np.array(struct_features, dtype=np.float32),
        y=np.array(kept_y, dtype=np.int64),
        labels=np.array(kept_labels),
        paths=np.array(kept_paths),
        struct_feature_names=np.array(struct_feature_names),
    )

    print("Wrote:", csv_path)
    print("Wrote:", npz_path)
    print("Parsed:", len(kept_y), "files (benign=", sum(1 for v in kept_y if v == 0), "malware=", sum(1 for v in kept_y if v == 1), ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
