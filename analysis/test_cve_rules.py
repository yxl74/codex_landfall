#!/usr/bin/env python3
"""
Unit tests for CVE detection rules.

Crafts minimal TIFF binaries in-memory to verify each CVE rule
triggers correctly on malicious patterns and does not trigger on
clean benign structures.
"""

from __future__ import annotations

import mmap
import os
import struct
import tempfile
import unittest

from cve_rule_validation import (
    parse_cve_features,
    rule_cve_2025_21043,
    rule_cve_2025_43300,
    rule_tile_config,
)


def _u16(v: int, endian: str = "II") -> bytes:
    return struct.pack("<H" if endian == "II" else ">H", v)


def _u32(v: int, endian: str = "II") -> bytes:
    return struct.pack("<I" if endian == "II" else ">I", v)


def _u32be(v: int) -> bytes:
    return struct.pack(">I", v)


def _ifd_entry(tag: int, type_id: int, count: int, value: int, endian: str = "II") -> bytes:
    return _u16(tag, endian) + _u16(type_id, endian) + _u32(count, endian) + _u32(value, endian)


def _build_minimal_tiff(
    endian: str = "II",
    width: int = 100,
    height: int = 100,
    extra_entries: list | None = None,
    extra_data: bytes = b"",
    extra_data_offset: int | None = None,
) -> bytes:
    """Build a minimal valid TIFF with configurable IFD entries.

    Returns raw bytes that can be written to a temp file.
    """
    entries = [
        # TAG_WIDTH (256), type SHORT (3), count 1
        _ifd_entry(256, 3, 1, width, endian),
        # TAG_HEIGHT (257), type SHORT (3), count 1
        _ifd_entry(257, 3, 1, height, endian),
    ]
    if extra_entries:
        entries.extend(extra_entries)

    count = len(entries)
    # Header: 2 (endian) + 2 (magic) + 4 (root offset) = 8
    # IFD starts at offset 8
    ifd_offset = 8
    ifd_data = _u16(count, endian)
    for entry in entries:
        ifd_data += entry
    ifd_data += _u32(0, endian)  # next IFD = 0

    # Extra data appended after IFD
    data_start = ifd_offset + len(ifd_data)
    if extra_data_offset is not None:
        # Pad to align extra_data at the given offset
        pad = extra_data_offset - data_start
        if pad > 0:
            ifd_data += b"\x00" * pad
            data_start = extra_data_offset

    header = endian.encode("ascii") + _u16(42, endian) + _u32(ifd_offset, endian)
    return header + ifd_data + extra_data


def _write_temp(data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".tif")
    os.write(fd, data)
    os.close(fd)
    return path


class TestCveRules(unittest.TestCase):
    def setUp(self):
        self._temps: list[str] = []

    def tearDown(self):
        for p in self._temps:
            if os.path.exists(p):
                os.unlink(p)

    def _temp(self, data: bytes) -> str:
        path = _write_temp(data)
        self._temps.append(path)
        return path

    # --- CVE-2025-21043: opcode overflow ---

    def test_cve_2025_21043_triggers(self):
        """Opcode list with declared count=2,000,000 should trigger."""
        # Build opcode list data: 4 bytes big-endian count
        opcode_count = 2_000_000
        opcode_data = _u32be(opcode_count)
        # No actual opcode entries (will fail to parse any, but declared count is huge)

        # Place opcode data after IFD
        # OpcodeList1 tag = 51008, type = UNDEFINED (7)
        data_offset = 200  # safe offset past IFD
        entry = _ifd_entry(51008, 7, len(opcode_data), data_offset)

        tiff = _build_minimal_tiff(
            extra_entries=[entry],
            extra_data=opcode_data,
            extra_data_offset=data_offset,
        )
        path = self._temp(tiff)
        feat = parse_cve_features(path)
        self.assertIsNotNone(feat)
        self.assertEqual(feat["max_declared_opcode_count"], opcode_count)
        self.assertTrue(rule_cve_2025_21043(feat))

    def test_cve_2025_21043_benign(self):
        """Normal opcode count (3) should not trigger."""
        opcode_count = 3
        # Build 3 minimal opcode entries: id(4) + version(4) + flags(4) + dataSize(4) = 16 each
        opcode_data = _u32be(opcode_count)
        for i in range(opcode_count):
            opcode_data += _u32be(1)   # opcode id
            opcode_data += _u32be(1)   # version
            opcode_data += _u32be(0)   # flags
            opcode_data += _u32be(0)   # data size = 0

        data_offset = 200
        entry = _ifd_entry(51008, 7, len(opcode_data), data_offset)

        tiff = _build_minimal_tiff(
            extra_entries=[entry],
            extra_data=opcode_data,
            extra_data_offset=data_offset,
        )
        path = self._temp(tiff)
        feat = parse_cve_features(path)
        self.assertIsNotNone(feat)
        self.assertEqual(feat["max_declared_opcode_count"], opcode_count)
        self.assertFalse(rule_cve_2025_21043(feat))

    # --- CVE-2025-43300: SOF3 component mismatch ---

    def test_cve_2025_43300_triggers(self):
        """SPP=2, Compression=7, SOF3 with Nf=1 should trigger."""
        # Build strip data with SOF3 marker
        # SOF3: FF C3 00 0B 08 00 10 00 10 01 ...
        #   Lh Ll = 00 0B (11 bytes)
        #   P = 08 (precision)
        #   Y = 00 10 (height 16)
        #   X = 00 10 (width 16)
        #   Nf = 01 (component count = 1, mismatch with SPP=2)
        sof3_data = b"\xFF\xC3\x00\x0B\x08\x00\x10\x00\x10\x01\x00"
        strip_data = b"\x00" * 10 + sof3_data + b"\x00" * 50

        strip_offset = 300
        entries = [
            # Compression=7 (JPEG), tag 259, type SHORT
            _ifd_entry(259, 3, 1, 7),
            # SamplesPerPixel=2, tag 277, type SHORT
            _ifd_entry(277, 3, 1, 2),
            # StripOffsets, tag 273, type LONG, 1 strip
            _ifd_entry(273, 4, 1, strip_offset),
        ]

        tiff = _build_minimal_tiff(
            extra_entries=entries,
            extra_data=strip_data,
            extra_data_offset=strip_offset,
        )
        path = self._temp(tiff)
        feat = parse_cve_features(path)
        self.assertIsNotNone(feat)
        self.assertTrue(feat["sof3_component_mismatch"])
        self.assertTrue(rule_cve_2025_43300(feat))

    def test_cve_2025_43300_matching_sof3(self):
        """SPP=2, Compression=7, SOF3 with Nf=2 should NOT trigger."""
        # SOF3 with Nf=2 (matches SPP)
        sof3_data = b"\xFF\xC3\x00\x0B\x08\x00\x10\x00\x10\x02\x00"
        strip_data = b"\x00" * 10 + sof3_data + b"\x00" * 50

        strip_offset = 300
        entries = [
            _ifd_entry(259, 3, 1, 7),
            _ifd_entry(277, 3, 1, 2),
            _ifd_entry(273, 4, 1, strip_offset),
        ]

        tiff = _build_minimal_tiff(
            extra_entries=entries,
            extra_data=strip_data,
            extra_data_offset=strip_offset,
        )
        path = self._temp(tiff)
        feat = parse_cve_features(path)
        self.assertIsNotNone(feat)
        self.assertFalse(feat["sof3_component_mismatch"])
        self.assertFalse(rule_cve_2025_43300(feat))

    def test_cve_2025_43300_no_sof3(self):
        """SPP=2, Compression=7, but no SOF3 marker should NOT trigger."""
        strip_data = b"\x00" * 100

        strip_offset = 300
        entries = [
            _ifd_entry(259, 3, 1, 7),
            _ifd_entry(277, 3, 1, 2),
            _ifd_entry(273, 4, 1, strip_offset),
        ]

        tiff = _build_minimal_tiff(
            extra_entries=entries,
            extra_data=strip_data,
            extra_data_offset=strip_offset,
        )
        path = self._temp(tiff)
        feat = parse_cve_features(path)
        self.assertIsNotNone(feat)
        self.assertFalse(feat["sof3_component_mismatch"])
        self.assertFalse(rule_cve_2025_43300(feat))

    # --- TILE-CONFIG: tile offset/bytecount mismatch ---

    def test_tile_config_offset_mismatch(self):
        """tile_offsets_count != tile_byte_counts_count should trigger."""
        # TileOffsets: tag 324, type LONG, count=4 (inline won't work, need pointer)
        # TileByteCounts: tag 325, type LONG, count=3
        # For inline (count=1) just use value directly
        entries = [
            # TileWidth, tag 322, type SHORT, count=1, value=64
            _ifd_entry(322, 3, 1, 64),
            # TileHeight, tag 323, type SHORT, count=1, value=64
            _ifd_entry(323, 3, 1, 64),
            # TileOffsets, tag 324, type LONG, count=4
            _ifd_entry(324, 4, 4, 400),
            # TileByteCounts, tag 325, type LONG, count=3
            _ifd_entry(325, 4, 3, 500),
        ]
        # Need actual data at offsets 400 and 500
        tile_offsets_data = _u32(1000) + _u32(2000) + _u32(3000) + _u32(4000)
        tile_bc_data = _u32(100) + _u32(100) + _u32(100)
        # Pad to offset 400
        extra = b"\x00" * (400 - 200) + tile_offsets_data
        # Pad to offset 500
        extra += b"\x00" * (500 - 400 - len(tile_offsets_data)) + tile_bc_data

        tiff = _build_minimal_tiff(
            extra_entries=entries,
            extra_data=extra,
            extra_data_offset=200,
        )
        path = self._temp(tiff)
        feat = parse_cve_features(path)
        self.assertIsNotNone(feat)
        self.assertEqual(feat["tile_offsets_count"], 4)
        self.assertEqual(feat["tile_byte_counts_count"], 3)
        self.assertTrue(rule_tile_config(feat))

    def test_tile_config_geometry_mismatch(self):
        """Tile count doesn't match expected from geometry."""
        # Image: 128x128, Tile: 64x64 → expected = 4 tiles
        # But declare TileOffsets count = 6
        entries = [
            _ifd_entry(322, 3, 1, 64),   # TileWidth
            _ifd_entry(323, 3, 1, 64),   # TileHeight
            _ifd_entry(324, 4, 6, 400),  # TileOffsets count=6
            _ifd_entry(325, 4, 6, 500),  # TileByteCounts count=6
        ]
        tile_offsets_data = _u32(1000) * 6
        tile_bc_data = _u32(100) * 6
        extra = b"\x00" * (400 - 200) + tile_offsets_data
        extra += b"\x00" * (500 - 400 - len(tile_offsets_data)) + tile_bc_data

        tiff = _build_minimal_tiff(
            width=128,
            height=128,
            extra_entries=entries,
            extra_data=extra,
            extra_data_offset=200,
        )
        path = self._temp(tiff)
        feat = parse_cve_features(path)
        self.assertIsNotNone(feat)
        # Expected tile count: ceil(128/64) * ceil(128/64) = 4
        self.assertEqual(feat["expected_tile_count"], 4)
        self.assertEqual(feat["tile_offsets_count"], 6)
        self.assertTrue(rule_tile_config(feat))

    def test_tile_dim_extreme(self):
        """Extreme dimension > 0xFFFE7960 should trigger."""
        # Use a very large width value
        extreme_w = 0xFFFE7961  # just over threshold
        entries = []
        # We can't fit this in a SHORT, use type LONG (4)
        entries.append(_ifd_entry(322, 4, 1, extreme_w))  # TileWidth
        entries.append(_ifd_entry(323, 3, 1, 64))  # TileHeight normal

        tiff = _build_minimal_tiff(
            extra_entries=entries,
        )
        path = self._temp(tiff)
        feat = parse_cve_features(path)
        self.assertIsNotNone(feat)
        self.assertGreater(feat["max_dim"], 0xFFFE7960)
        self.assertTrue(rule_tile_config(feat))

    # --- Clean benign DNG ---

    def test_benign_dng_triggers_nothing(self):
        """Clean DNG-like file should trigger no rules."""
        # DNG version tag
        entries = [
            _ifd_entry(50706, 1, 4, 0x01040000),  # DNG version
            _ifd_entry(259, 3, 1, 1),              # Compression: no compression
            _ifd_entry(277, 3, 1, 3),              # SPP = 3
        ]

        tiff = _build_minimal_tiff(
            width=4000,
            height=3000,
            extra_entries=entries,
        )
        path = self._temp(tiff)
        feat = parse_cve_features(path)
        self.assertIsNotNone(feat)
        self.assertFalse(rule_cve_2025_21043(feat))
        self.assertFalse(rule_cve_2025_43300(feat))
        self.assertFalse(rule_tile_config(feat))

    def test_non_tiff_returns_none(self):
        """Non-TIFF file should return None from parsing."""
        data = b"This is not a TIFF file at all"
        path = self._temp(data)
        feat = parse_cve_features(path)
        self.assertIsNone(feat)

    def test_too_small_file(self):
        """File smaller than 8 bytes should return None."""
        path = self._temp(b"\x00\x01\x02")
        feat = parse_cve_features(path)
        self.assertIsNone(feat)


if __name__ == "__main__":
    unittest.main()
