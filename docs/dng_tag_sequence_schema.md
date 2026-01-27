# DNG Tag-Sequence Feature Schema (GRU/AE)

This document defines the per-tag feature vector used for a DNG-only
sequence autoencoder and the extraction pipeline implemented in
`analysis/tag_sequence_extract.py`.

The goal is to encode **relationships and constraints** inside TIFF/DNG
structure, rather than raw integer values that are too noisy for a small
on-device model.

## Sequence Definition

- One sequence step = one IFD entry (tag record).
- Tags are emitted in traversal order: root IFD, chained IFDs, then any
  SubIFD/Exif/GPS/Interop IFDs discovered via pointer tags.
- Each tag produces:
  - `tag_id` (int) for embedding.
  - `type_id` (int) for embedding.
  - `ifd_kind` (int) for embedding.
  - `features` (float vector).

## Feature Vector (per tag)

Feature list and exact normalization/clipping rules:

| Index | Feature | Range / Encoding | Notes |
| --- | --- | --- | --- |
| 0 | `coarse_tag_bucket_norm` | `(tag_id // 256) / 255` | Coarse tag range (0..1). |
| 1 | `log1p_count_norm` | `log1p(count)` clipped to 16, then `/ 16` | Normalizes large counts. |
| 2 | `log1p_byte_count_norm` | `log1p(count * type_size)` clipped to 24, then `/ 24` | Captures buffer size. |
| 3 | `offset_norm` | `offset / file_size` clipped to [0,1] | Set to 0 if immediate. |
| 4 | `offset_valid` | `1` if `offset + byte_count <= file_size` else `0` | For immediate, set to 1. |
| 5 | `is_immediate` | `1` if `byte_count <= 4` else `0` | Inline vs pointer. |
| 6 | `is_pointer` | `1` if tag is SubIFD/Exif/GPS/Interop else `0` | Pointer semantics. |
| 7 | `order_delta_norm` | `clip(tag_id - prev_tag, -128, 128) / 128` | Per-IFD ordering signal. |
| 8 | `order_violation` | `1` if `tag_id < prev_tag` else `0` | Sorted tag rule. |
| 9 | `type_mismatch` | `1` if type_id not in expected types | Only for known tags. |
| 10 | `type_id_norm` | `type_id / 12` clipped to [0,1] | Backup signal if no embedding. |
| 11 | `ifd_kind_norm` | `ifd_kind / 4` clipped to [0,1] | Main/SubIFD/Exif/GPS/Interop. |

### Expected type map (subset)

The extractor enforces type checks for common tags:

- `ImageWidth(256)`, `ImageLength(257)`: SHORT or LONG
- `BitsPerSample(258)`: SHORT
- `Compression(259)`, `PhotometricInterpretation(262)`: SHORT
- `StripOffsets(273)`, `StripByteCounts(279)`: SHORT or LONG
- `RowsPerStrip(278)`: SHORT or LONG
- `ExifIFD(34665)`, `GPSIFD(34853)`, `SubIFD(330)`: LONG
- `DNGVersion(50706)`: BYTE
- `OpcodeList1/2/3(51008/51009/51022)`: UNDEFINED or BYTE

Unknown tags do not emit `type_mismatch = 1`.

## Categorical Inputs (Embeddings)

- `tag_id`: raw TIFF tag ID (int32). Use embedding and an UNKNOWN token if desired.
- `type_id`: TIFF type ID (int32). Optionally embed or use `type_id_norm`.
- `ifd_kind`: 0=main, 1=subifd, 2=exif, 3=gps, 4=interop.

## File-Level Features (optional)

The extractor also emits per-file features:

| Index | Feature | Range / Encoding | Notes |
| --- | --- | --- | --- |
| 0 | `log1p_file_size_norm` | `log1p(file_size)` clipped to 32, then `/ 32` | File scale. |
| 1 | `total_tags_norm` | `min(total_tags, 2048) / 2048` | Tag volume. |
| 2 | `ifd_count_norm` | `min(ifd_count, 64) / 64` | IFD depth/complexity. |

## Extraction CLI

Example (DNG-only training list):

```bash
python3 analysis/tag_sequence_extract.py \
  --list-file outputs/train_list.txt \
  --dng-only \
  --max-seq-len 512 \
  --output outputs/tagseq_dng_train.npz
```

NPZ fields:
- `features`: float32, shape `(N, max_len, 12)`
- `tag_ids`: int32, shape `(N, max_len)`
- `type_ids`: int32, shape `(N, max_len)`
- `ifd_kinds`: int32, shape `(N, max_len)`
- `lengths`: int32, shape `(N,)`
- `labels`: string labels inferred from path
- `paths`: original paths
- `file_features`: float32, shape `(N, 3)`
- `schema`: JSON string of the feature schema

## Why these features matter for LandFall

- LandFall samples show invalid opcode offsets and tag order violations.
- The model learns **constraints**: offsets usually stay within file bounds,
  DNG tags are ordered, and opcode lists have plausible sizes.
- When those constraints break, reconstruction error rises.
