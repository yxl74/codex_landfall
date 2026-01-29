# Model Feature Documentation (Production)

This document describes the **feature inputs** for each on-device model currently shipped in the Android app assets.

| Model (asset) | Detects | Input shape | Feature sources |
|---|---|---:|---|
| `anomaly_ae.tflite` + `anomaly_model_meta.json` | **TIFF (non-DNG)** anomaly detection | 550 floats | Byte-histogram (header+tail) + TIFF structural + entropy |
| `tagseq_gru_ae.tflite` + `tagseq_gru_ae_meta.json` | **DNG/TIFF** (DNG is a TIFF) anomaly detection | sequence ≤512 | TIFF IFD **tag-sequence** (per-entry features + IDs) |
| `jpeg_ae.tflite` + `jpeg_model_meta.json` | **JPEG** anomaly detection | 35 floats | JPEG marker/segment structural features |

## 1) TIFF AE (`anomaly_ae.tflite`)

| Feature group | Features | Why these features help |
|---|---|---|
| Byte histograms (514 dims) | Two normalized 257-bin byte histograms: **first 1024 bytes** + **last 1024 bytes** (includes a padding bin) | Captures file “shape” without decoding: magic/headers, ASCII vs binary balance, compression artifacts, and trailing payloads/polyglots. Very cheap and robust. |
| Type flags | `is_tiff`, `is_dng` | Routes the model away from non-TIFF inputs and provides context for DNG-only structures (opcode tags). |
| Dimensions & density | `min_width`, `min_height`, `max_width`, `max_height`, `total_pixels`, `file_size`, `bytes_per_pixel_milli`, `pixels_per_mb` | Many memory bugs are driven by allocation math and size mismatches. These summarize “how big is the image” vs “how big is the file”. |
| IFD graph complexity | `ifd_entry_max`, `subifd_count_sum`, `new_subfile_types_unique` | Malformed/hostile files often have unusual IFD structures: too many entries, weird SubIFD fanout, or strange NewSubFileType patterns. |
| DNG opcode surface | `total_opcodes`, `unknown_opcodes`, `max_opcode_id`, `opcode_list1_bytes`, `opcode_list2_bytes`, `opcode_list3_bytes`, `opcode_list_bytes_total`, `opcode_list_bytes_max`, `opcode_list_present_count`, `has_opcode_list1`, `has_opcode_list2`, `has_opcode_list3`, `max_declared_opcode_count`, `opcode_bytes_ratio_permille`, `opcode_bytes_per_opcode_milli`, `unknown_opcode_ratio_permille` | DNG opcode parsing is a high-risk area (complex nested structures). These features capture “how much opcode surface exists” and whether it looks malformed/unusual. |
| Encoding/tiling hints | `spp_max`, `compression_variety`, `tile_count_ratio` | Compression and tiling choices affect decoder code paths. Unusual combinations can indicate fuzzing or rare-path triggers. |
| Entropy (3 windows) | `header_entropy`, `tail_entropy`, `overall_entropy`, `entropy_gradient` | Entropy spikes can indicate embedded payloads, encrypted/compressed blobs, or abnormal structure (e.g., large binary metadata). Gradient catches header-vs-tail asymmetry common in polyglots. |

## 2) DNG TagSeq GRU-AE (`tagseq_gru_ae.tflite`)

This model treats DNG as a **TIFF with DNG-specific tags** and learns “normal” IFD/tag-sequences.

| Input | What it is | Why it helps |
|---|---|---|
| `tag_ids` (sequence) | Raw TIFF tag IDs per IFD entry | Lets the model learn normal tag co-occurrence and ordering patterns. |
| `type_ids` (sequence) | TIFF type IDs per entry | Many malformed files use unexpected types for known tags; this captures those schema violations. |
| `ifd_kinds` (sequence) | Which IFD an entry came from (main/sub/exif/gps/interop) | Same tag can be normal in one IFD and suspicious in another; this gives structural context. |
| `features` (12 floats per entry) | Per-entry numeric features: tag bucket, log(value count), log(byte size), normalized offset, offset-valid flag, immediate-vs-offset, pointer flag, order delta, order-violation flag, expected-type mismatch flag, normalized type, normalized IFD kind | Captures “shape” of each IFD entry and pointer behavior: invalid offsets, unusual sizes/counts, and out-of-order tags are common exploit/fuzz signals. |
| `length` (mask) | Actual number of observed entries (≤512) | Ensures scoring reflects only real entries. |

## 3) JPEG AE (`jpeg_ae.tflite`)

| Feature group | Features | Why these features help |
|---|---|---|
| File termination / polyglot | `file_size`, `has_eoi`, `bytes_after_eoi`, `bytes_after_eoi_ratio_permille`, `tail_zip_magic`, `tail_pdf_magic`, `tail_elf_magic` | Many malicious samples are JPEG+payload polyglots, or contain suspicious trailers. These features capture that without decoding pixels. |
| Parse correctness | `invalid_len` | Malformed segment length fields are a common trigger for memory issues; this flags obvious structural corruption. |
| Image header context | `width`, `height`, `components`, `precision`, `dri_interval` | Geometry and restart intervals drive decoder allocation/state-machine behavior and can trigger rare paths. |
| Metadata surface | `app_count`, `app_bytes`, `app_bytes_ratio_permille`, `app_max_len`, `com_count`, `com_bytes` | APP/COM blocks carry EXIF/XMP/ICC and other parsers; large/unusual metadata is a frequent attack surface. |
| Codec tables (DQT/DHT) | `dqt_count`, `dqt_bytes`, `dqt_bytes_ratio_permille`, `dqt_max_len`, `dqt_tables`, `dqt_invalid`, `dht_count`, `dht_bytes`, `dht_bytes_ratio_permille`, `dht_max_len`, `dht_tables`, `dht_invalid` | Quantization/Huffman tables are core parser hotspots. These features measure table complexity and structural validity. |
| Scan/frame structure | `sof0_count`, `sof2_count`, `sos_count`, `max_seg_len` | Progressive JPEGs and unusual scan structure exercise more complex decode paths; `max_seg_len` catches extreme segments. |

