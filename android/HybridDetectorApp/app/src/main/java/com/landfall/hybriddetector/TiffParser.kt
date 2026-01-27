package com.landfall.hybriddetector

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.ceil

data class TiffFeatures(
    val isTiff: Int,
    val isDng: Int,
    val minWidth: Int,
    val minHeight: Int,
    val ifdEntryMax: Int,
    val subifdCountSum: Int,
    val newSubfileTypesUnique: Int,
    val totalOpcodes: Int,
    val unknownOpcodes: Int,
    val maxOpcodeId: Int,
    val opcodeList1Bytes: Int,
    val opcodeList2Bytes: Int,
    val opcodeList3Bytes: Int,
    val maxWidth: Int = 0,
    val maxHeight: Int = 0,
    val totalPixels: Int = 0,
    val fileSize: Int = 0,
    val bytesPerPixelMilli: Int = 0,
    val pixelsPerMb: Int = 0,
    val opcodeListBytesTotal: Int = 0,
    val opcodeListBytesMax: Int = 0,
    val opcodeListPresentCount: Int = 0,
    val opcodeBytesRatioPermille: Int = 0,
    val opcodeBytesPerOpcodeMilli: Int = 0,
    val unknownOpcodeRatioPermille: Int = 0,
    val hasOpcodeList1: Int = 0,
    val hasOpcodeList2: Int = 0,
    val hasOpcodeList3: Int = 0,
    val zipEocdNearEnd: Int,
    val zipLocalInTail: Int,
    val flagOpcodeAnomaly: Int,
    val flagTinyDimsLowIfd: Int,
    val flagZipPolyglot: Int,
    val flagDngJpegMismatch: Int,
    val flagMagikaExtMismatch: Int,
    val flagAny: Int,
    // CVE-relevant fields
    val maxDeclaredOpcodeCount: Int = 0,
    val compressionValues: Set<Int> = emptySet(),
    val sppValues: Set<Int> = emptySet(),
    val sof3ComponentMismatch: Boolean = false,
    val tileOffsetsCount: Int = 0,
    val tileByteCountsCount: Int = 0,
    val expectedTileCount: Int = 0,
    val tileWidths: List<Int> = emptyList(),
    val tileHeights: List<Int> = emptyList(),
) {
    fun toFeatureMap(): Map<String, Int> {
        return mapOf(
            "is_tiff" to isTiff,
            "is_dng" to isDng,
            "min_width" to minWidth,
            "min_height" to minHeight,
            "ifd_entry_max" to ifdEntryMax,
            "subifd_count_sum" to subifdCountSum,
            "new_subfile_types_unique" to newSubfileTypesUnique,
            "total_opcodes" to totalOpcodes,
            "unknown_opcodes" to unknownOpcodes,
            "max_opcode_id" to maxOpcodeId,
            "opcode_list1_bytes" to opcodeList1Bytes,
            "opcode_list2_bytes" to opcodeList2Bytes,
            "opcode_list3_bytes" to opcodeList3Bytes,
            "max_width" to maxWidth,
            "max_height" to maxHeight,
            "total_pixels" to totalPixels,
            "file_size" to fileSize,
            "bytes_per_pixel_milli" to bytesPerPixelMilli,
            "pixels_per_mb" to pixelsPerMb,
            "opcode_list_bytes_total" to opcodeListBytesTotal,
            "opcode_list_bytes_max" to opcodeListBytesMax,
            "opcode_list_present_count" to opcodeListPresentCount,
            "opcode_bytes_ratio_permille" to opcodeBytesRatioPermille,
            "opcode_bytes_per_opcode_milli" to opcodeBytesPerOpcodeMilli,
            "unknown_opcode_ratio_permille" to unknownOpcodeRatioPermille,
            "has_opcode_list1" to hasOpcodeList1,
            "has_opcode_list2" to hasOpcodeList2,
            "has_opcode_list3" to hasOpcodeList3,
            "zip_eocd_near_end" to zipEocdNearEnd,
            "zip_local_in_tail" to zipLocalInTail,
            "flag_opcode_anomaly" to flagOpcodeAnomaly,
            "flag_tiny_dims_low_ifd" to flagTinyDimsLowIfd,
            "flag_zip_polyglot" to flagZipPolyglot,
            "flag_dng_jpeg_mismatch" to flagDngJpegMismatch,
            "flag_magika_ext_mismatch" to flagMagikaExtMismatch,
            "flag_any" to flagAny,
            "max_declared_opcode_count" to maxDeclaredOpcodeCount,
            "spp_max" to (sppValues.maxOrNull() ?: 0),
            "compression_variety" to compressionValues.size,
            "tile_count_ratio" to if (expectedTileCount > 0) (tileOffsetsCount * 1000 / expectedTileCount) else 0,
        )
    }
}

object TiffParser {
    private val TYPE_SIZES = mapOf(
        1 to 1, 2 to 1, 3 to 2, 4 to 4, 5 to 8, 6 to 1, 7 to 1,
        8 to 2, 9 to 4, 10 to 8, 11 to 4, 12 to 8
    )

    private const val TAG_WIDTH = 256
    private const val TAG_HEIGHT = 257
    private const val TAG_COMPRESSION = 259
    private const val TAG_SAMPLES_PER_PIXEL = 277
    private const val TAG_STRIP_OFFSETS = 273
    private const val TAG_TILE_WIDTH = 322
    private const val TAG_TILE_HEIGHT = 323
    private const val TAG_TILE_OFFSETS = 324
    private const val TAG_TILE_BYTE_COUNTS = 325
    private const val TAG_SUBIFD = 330
    private const val TAG_EXIF_IFD = 34665
    private const val TAG_DNG_VERSION = 50706
    private const val TAG_NEW_SUBFILE_TYPE = 254
    private const val TAG_OPCODE_LIST1 = 51008
    private const val TAG_OPCODE_LIST2 = 51009
    private const val TAG_OPCODE_LIST3 = 51022

    private const val MAX_SOF3_SCAN = 65536

    fun parse(path: String): TiffFeatures {
        val file = File(path)
        val size = file.length()
        if (size < 8) {
            return emptyFeatures()
        }

        val claimedType = claimedType(path)
        val magicType = magicType(path)

        RandomAccessFile(file, "r").use { raf ->
            val endianBytes = ByteArray(2)
            raf.readFully(endianBytes)
            val endianStr = String(endianBytes)
            val order = when (endianStr) {
                "II" -> ByteOrder.LITTLE_ENDIAN
                "MM" -> ByteOrder.BIG_ENDIAN
                else -> return emptyFeatures()
            }
            val magic = readU16(raf, 2, order)
            if (magic != 42) {
                return emptyFeatures()
            }
            val root = readU32(raf, 4, order).toLong()
            if (root <= 0L) {
                return emptyFeatures()
            }

            var isDng = 0
            val widths = mutableListOf<Int>()
            val heights = mutableListOf<Int>()
            var ifdEntryMax = 0
            val subifdOffsets = mutableListOf<Long>()
            var exifOffset = 0L
            val newSubfileTypes = mutableSetOf<Int>()
            val opcodeLists = mutableMapOf<Int, Pair<Long, Int>>()

            // CVE-relevant collectors
            val compressionValues = mutableSetOf<Int>()
            val sppValues = mutableSetOf<Int>()
            val tileOffsetsCountList = mutableListOf<Int>()
            val tileByteCountsCountList = mutableListOf<Int>()
            val tileWidthsList = mutableListOf<Int>()
            val tileHeightsList = mutableListOf<Int>()
            // Per-IFD (compression, spp, stripOffset) for SOF3 scanning
            data class Sof3Candidate(val compression: Int, val spp: Int, val stripOffset: Long)
            val sof3Candidates = mutableListOf<Sof3Candidate>()

            val visited = mutableSetOf<Long>()
            val stack = ArrayDeque<Long>()
            stack.add(root)

            fun parseStack() {
                while (stack.isNotEmpty()) {
                    val off = stack.removeLast()
                    if (off <= 0 || off >= size || visited.contains(off)) continue
                    visited.add(off)

                    val count = readU16(raf, off, order)
                    if (count <= 0) continue
                    ifdEntryMax = max(ifdEntryMax, count)

                    // Per-IFD locals for SOF3 candidate tracking
                    var localCompression: Int? = null
                    var localSpp: Int? = null
                    var localStripOffset: Long? = null

                    val entryBase = off + 2
                    for (i in 0 until count) {
                        val entryOff = entryBase + i * 12L
                        if (entryOff + 12 > size) break
                        val tag = readU16(raf, entryOff, order)
                        val typeId = readU16(raf, entryOff + 2, order)
                        val valCount = readU32(raf, entryOff + 4, order)
                        val valueOrOffset = readU32(raf, entryOff + 8, order)
                        if (tag == 0) continue

                        if (tag == TAG_DNG_VERSION) {
                            isDng = 1
                        }

                        if (tag == TAG_WIDTH || tag == TAG_HEIGHT || tag == TAG_NEW_SUBFILE_TYPE) {
                            val vals = readValues(raf, order, size, typeId, valCount, valueOrOffset)
                            if (vals.isNotEmpty()) {
                                if (tag == TAG_WIDTH) widths.addAll(vals)
                                if (tag == TAG_HEIGHT) heights.addAll(vals)
                                if (tag == TAG_NEW_SUBFILE_TYPE) newSubfileTypes.addAll(vals)
                            }
                        }

                        if (tag == TAG_COMPRESSION) {
                            val vals = readValues(raf, order, size, typeId, valCount, valueOrOffset)
                            if (vals.isNotEmpty()) {
                                compressionValues.addAll(vals)
                                localCompression = vals[0]
                            }
                        }

                        if (tag == TAG_SAMPLES_PER_PIXEL) {
                            val vals = readValues(raf, order, size, typeId, valCount, valueOrOffset)
                            if (vals.isNotEmpty()) {
                                sppValues.addAll(vals)
                                localSpp = vals[0]
                            }
                        }

                        if (tag == TAG_STRIP_OFFSETS) {
                            val vals = readValues(raf, order, size, typeId, valCount, valueOrOffset)
                            if (vals.isNotEmpty()) {
                                localStripOffset = vals[0].toLong() and 0xFFFFFFFFL
                            }
                        }

                        if (tag == TAG_TILE_OFFSETS) {
                            tileOffsetsCountList.add(valCount)
                        }

                        if (tag == TAG_TILE_BYTE_COUNTS) {
                            tileByteCountsCountList.add(valCount)
                        }

                        if (tag == TAG_TILE_WIDTH) {
                            val vals = readValues(raf, order, size, typeId, valCount, valueOrOffset)
                            if (vals.isNotEmpty()) tileWidthsList.addAll(vals)
                        }

                        if (tag == TAG_TILE_HEIGHT) {
                            val vals = readValues(raf, order, size, typeId, valCount, valueOrOffset)
                            if (vals.isNotEmpty()) tileHeightsList.addAll(vals)
                        }

                        if (tag == TAG_SUBIFD) {
                            val sizeBytes = TYPE_SIZES.getOrDefault(typeId, 1) * valCount
                            if (sizeBytes <= 4) {
                                subifdOffsets.add(valueOrOffset.toLong())
                            } else {
                                val offs = readU32Array(raf, valueOrOffset.toLong(), valCount, order)
                                subifdOffsets.addAll(offs.map { it.toLong() })
                            }
                        }

                        if (tag == TAG_EXIF_IFD) {
                            exifOffset = valueOrOffset.toLong()
                        }

                        if (tag == TAG_OPCODE_LIST1 || tag == TAG_OPCODE_LIST2 || tag == TAG_OPCODE_LIST3) {
                            val sizeBytes = TYPE_SIZES.getOrDefault(typeId, 1) * valCount
                            opcodeLists[tag] = Pair(valueOrOffset.toLong(), sizeBytes)
                        }
                    }

                    // Record SOF3 candidate if we have both compression and SPP
                    if (localCompression != null && localSpp != null && localStripOffset != null) {
                        sof3Candidates.add(Sof3Candidate(localCompression, localSpp, localStripOffset))
                    }

                    val nextIfd = readU32(raf, entryBase + count * 12L, order)
                    if (nextIfd > 0) stack.add(nextIfd.toLong())
                }
            }

            parseStack()
            for (off in subifdOffsets) stack.add(off)
            if (exifOffset > 0) stack.add(exifOffset)
            parseStack()

            val minWidth = widths.minOrNull() ?: 0
            val minHeight = heights.minOrNull() ?: 0
            val maxWidth = widths.maxOrNull() ?: 0
            val maxHeight = heights.maxOrNull() ?: 0

            var totalOpcodes = 0
            var unknownOpcodes = 0
            var maxOpcodeId = 0
            var list1Bytes = 0
            var list2Bytes = 0
            var list3Bytes = 0
            var maxDeclaredOpcodeCount = 0

            for ((tag, pair) in opcodeLists) {
                val offset = pair.first
                val sizeBytes = pair.second
                if (offset <= 0 || sizeBytes < 4 || offset + sizeBytes > size) continue
                val buf = readBytes(raf, offset, sizeBytes)
                val declaredCount = readU32BE(buf, 0)
                maxDeclaredOpcodeCount = max(maxDeclaredOpcodeCount, declaredCount)
                var pos = 4
                var parsed = 0
                while (parsed < declaredCount && pos + 16 <= buf.size) {
                    val opcodeId = readU32BE(buf, pos)
                    val dataSize = readU32BE(buf, pos + 12)
                    pos += 16
                    if (pos + dataSize > buf.size) break
                    parsed += 1
                    totalOpcodes += 1
                    maxOpcodeId = max(maxOpcodeId, opcodeId)
                    if (opcodeId > 14) unknownOpcodes += 1
                    pos += dataSize
                }
                if (tag == TAG_OPCODE_LIST1) list1Bytes = sizeBytes
                if (tag == TAG_OPCODE_LIST2) list2Bytes = sizeBytes
                if (tag == TAG_OPCODE_LIST3) list3Bytes = sizeBytes
            }

            // SOF3 scanning: for IFDs with compression=7 AND spp=2
            var sof3ComponentMismatch = false
            for (candidate in sof3Candidates) {
                if (candidate.compression == 7 && candidate.spp == 2) {
                    if (scanForSof3Mismatch(raf, candidate.stripOffset, MAX_SOF3_SCAN, candidate.spp, size)) {
                        sof3ComponentMismatch = true
                        break
                    }
                }
            }

            // Tile geometry
            val totalTileOffsetsCount = tileOffsetsCountList.sum()
            val totalTileByteCountsCount = tileByteCountsCountList.sum()
            var expectedTileCount = 0
            if (tileWidthsList.isNotEmpty() && tileHeightsList.isNotEmpty() && widths.isNotEmpty() && heights.isNotEmpty()) {
                val tw = tileWidthsList[0]
                val th = tileHeightsList[0]
                if (tw > 0 && th > 0) {
                    val w = widths.max()
                    val h = heights.max()
                    expectedTileCount = ceil(w.toDouble() / tw).toInt() * ceil(h.toDouble() / th).toInt()
                }
            }

            val listBytesTotal = list1Bytes + list2Bytes + list3Bytes
            val listBytesMax = max(list1Bytes, max(list2Bytes, list3Bytes))
            val listPresentCount = (if (list1Bytes > 0) 1 else 0) +
                (if (list2Bytes > 0) 1 else 0) +
                (if (list3Bytes > 0) 1 else 0)

            val totalPixels = clampToInt(maxWidth.toLong() * maxHeight.toLong())
            val fileSize = clampToInt(size)
            val bytesPerPixelMilli = if (totalPixels > 0) {
                clampToInt(size * 1000L / totalPixels.toLong())
            } else {
                0
            }
            val pixelsPerMb = if (size > 0) {
                clampToInt(totalPixels.toLong() * 1_000_000L / size)
            } else {
                0
            }
            val opcodeBytesRatioPermille = if (size > 0) {
                clampToInt(listBytesTotal.toLong() * 1000L / size)
            } else {
                0
            }
            val opcodeBytesPerOpcodeMilli = if (totalOpcodes > 0) {
                clampToInt(listBytesTotal.toLong() * 1000L / totalOpcodes.toLong())
            } else {
                0
            }
            val unknownOpcodeRatioPermille = if (totalOpcodes > 0) {
                clampToInt(unknownOpcodes.toLong() * 1000L / totalOpcodes.toLong())
            } else {
                0
            }

            val (zipEocdNearEnd, zipLocalInTail) = zipPolyglotFlags(raf, size)
            val flagOpcodeAnomaly = if (isDng == 1 && (totalOpcodes > 100 || unknownOpcodes > 0)) 1 else 0
            val flagTinyDimsLowIfd = if (
                isDng == 0 && ifdEntryMax <= 10 && (minWidth <= 16 || minHeight <= 16)
            ) 1 else 0
            val flagZipPolyglot = if (zipEocdNearEnd == 1 && zipLocalInTail == 1) 1 else 0
            val flagDngJpegMismatch = if (claimedType == "jpeg" && magicType == "tiff" && isDng == 1) 1 else 0
            val flagMagikaExtMismatch = 0
            val flagAny = if (
                flagOpcodeAnomaly == 1 || flagTinyDimsLowIfd == 1 || flagZipPolyglot == 1 ||
                flagDngJpegMismatch == 1 || flagMagikaExtMismatch == 1
            ) 1 else 0

            return TiffFeatures(
                isTiff = 1,
                isDng = isDng,
                minWidth = minWidth,
                minHeight = minHeight,
                ifdEntryMax = ifdEntryMax,
                subifdCountSum = subifdOffsets.size,
                newSubfileTypesUnique = newSubfileTypes.size,
                totalOpcodes = totalOpcodes,
                unknownOpcodes = unknownOpcodes,
                maxOpcodeId = maxOpcodeId,
                opcodeList1Bytes = list1Bytes,
                opcodeList2Bytes = list2Bytes,
                opcodeList3Bytes = list3Bytes,
                maxWidth = maxWidth,
                maxHeight = maxHeight,
                totalPixels = totalPixels,
                fileSize = fileSize,
                bytesPerPixelMilli = bytesPerPixelMilli,
                pixelsPerMb = pixelsPerMb,
                opcodeListBytesTotal = listBytesTotal,
                opcodeListBytesMax = listBytesMax,
                opcodeListPresentCount = listPresentCount,
                opcodeBytesRatioPermille = opcodeBytesRatioPermille,
                opcodeBytesPerOpcodeMilli = opcodeBytesPerOpcodeMilli,
                unknownOpcodeRatioPermille = unknownOpcodeRatioPermille,
                hasOpcodeList1 = if (list1Bytes > 0) 1 else 0,
                hasOpcodeList2 = if (list2Bytes > 0) 1 else 0,
                hasOpcodeList3 = if (list3Bytes > 0) 1 else 0,
                zipEocdNearEnd = zipEocdNearEnd,
                zipLocalInTail = zipLocalInTail,
                flagOpcodeAnomaly = flagOpcodeAnomaly,
                flagTinyDimsLowIfd = flagTinyDimsLowIfd,
                flagZipPolyglot = flagZipPolyglot,
                flagDngJpegMismatch = flagDngJpegMismatch,
                flagMagikaExtMismatch = flagMagikaExtMismatch,
                flagAny = flagAny,
                maxDeclaredOpcodeCount = maxDeclaredOpcodeCount,
                compressionValues = compressionValues.toSet(),
                sppValues = sppValues.toSet(),
                sof3ComponentMismatch = sof3ComponentMismatch,
                tileOffsetsCount = totalTileOffsetsCount,
                tileByteCountsCount = totalTileByteCountsCount,
                expectedTileCount = expectedTileCount,
                tileWidths = tileWidthsList.toList(),
                tileHeights = tileHeightsList.toList(),
            )
        }
    }

    private fun scanForSof3Mismatch(
        raf: RandomAccessFile,
        stripOffset: Long,
        maxScan: Int,
        expectedComponents: Int,
        fileSize: Long
    ): Boolean {
        if (stripOffset >= fileSize || stripOffset < 0) return false
        val scanLen = min(maxScan.toLong(), fileSize - stripOffset).toInt()
        if (scanLen < 4) return false
        val data = readBytes(raf, stripOffset, scanLen)
        // Search for SOF3 marker: 0xFF 0xC3
        var pos = 0
        while (pos < data.size - 1) {
            val idx = indexOf(data, byteArrayOf(0xFF.toByte(), 0xC3.toByte()), pos)
            if (idx == -1) break
            // SOF3 header: FF C3 Lh Ll P Y(2) X(2) Nf
            // Nf (component count) at idx+9
            if (idx + 10 <= data.size) {
                val nf = data[idx + 9].toInt() and 0xFF
                if (nf != expectedComponents) {
                    return true
                }
            }
            pos = idx + 2
        }
        return false
    }

    private fun emptyFeatures(): TiffFeatures {
        return TiffFeatures(
            isTiff = 0,
            isDng = 0,
            minWidth = 0,
            minHeight = 0,
            ifdEntryMax = 0,
            subifdCountSum = 0,
            newSubfileTypesUnique = 0,
            totalOpcodes = 0,
            unknownOpcodes = 0,
            maxOpcodeId = 0,
            opcodeList1Bytes = 0,
            opcodeList2Bytes = 0,
            opcodeList3Bytes = 0,
            maxWidth = 0,
            maxHeight = 0,
            totalPixels = 0,
            fileSize = 0,
            bytesPerPixelMilli = 0,
            pixelsPerMb = 0,
            opcodeListBytesTotal = 0,
            opcodeListBytesMax = 0,
            opcodeListPresentCount = 0,
            opcodeBytesRatioPermille = 0,
            opcodeBytesPerOpcodeMilli = 0,
            unknownOpcodeRatioPermille = 0,
            hasOpcodeList1 = 0,
            hasOpcodeList2 = 0,
            hasOpcodeList3 = 0,
            zipEocdNearEnd = 0,
            zipLocalInTail = 0,
            flagOpcodeAnomaly = 0,
            flagTinyDimsLowIfd = 0,
            flagZipPolyglot = 0,
            flagDngJpegMismatch = 0,
            flagMagikaExtMismatch = 0,
            flagAny = 0,
            maxDeclaredOpcodeCount = 0,
            compressionValues = emptySet(),
            sppValues = emptySet(),
            sof3ComponentMismatch = false,
            tileOffsetsCount = 0,
            tileByteCountsCount = 0,
            expectedTileCount = 0,
            tileWidths = emptyList(),
            tileHeights = emptyList(),
        )
    }

    private fun claimedType(path: String): String {
        val ext = path.substringAfterLast('.', "").lowercase()
        return when (ext) {
            "jpg", "jpeg" -> "jpeg"
            "tif", "tiff" -> "tiff"
            "dng" -> "dng"
            "png" -> "png"
            "gif" -> "gif"
            else -> "unknown"
        }
    }

    private fun magicType(path: String): String {
        val f = File(path)
        val head = ByteArray(16)
        RandomAccessFile(f, "r").use { raf ->
            raf.readFully(head)
        }
        val isTiff = (head[0].toInt() == 'I'.code && head[1].toInt() == 'I'.code && head[2] == 0x2A.toByte()) ||
            (head[0].toInt() == 'M'.code && head[1].toInt() == 'M'.code && head[3] == 0x2A.toByte())
        if (isTiff) return "tiff"
        if (head[0] == 0xFF.toByte() && head[1] == 0xD8.toByte() && head[2] == 0xFF.toByte()) return "jpeg"
        if (head[0] == 0x89.toByte() && head[1] == 0x50.toByte()) return "png"
        if (head[0] == 0x47.toByte() && head[1] == 0x49.toByte()) return "gif"
        if (head[0] == 0x42.toByte() && head[1] == 0x4D.toByte()) return "bmp"
        if (head[0] == 0x50.toByte() && head[1] == 0x4B.toByte()) return "zip"
        return "unknown"
    }

    private fun readU16(raf: RandomAccessFile, offset: Long, order: ByteOrder): Int {
        val buf = ByteArray(2)
        raf.seek(offset)
        raf.readFully(buf)
        return ByteBuffer.wrap(buf).order(order).short.toInt() and 0xFFFF
    }

    private fun readU32(raf: RandomAccessFile, offset: Long, order: ByteOrder): Int {
        val buf = ByteArray(4)
        raf.seek(offset)
        raf.readFully(buf)
        return ByteBuffer.wrap(buf).order(order).int
    }

    private fun readU32Array(raf: RandomAccessFile, offset: Long, count: Int, order: ByteOrder): List<Int> {
        val buf = ByteArray(count * 4)
        raf.seek(offset)
        raf.readFully(buf)
        val bb = ByteBuffer.wrap(buf).order(order)
        val out = ArrayList<Int>(count)
        repeat(count) {
            out.add(bb.int)
        }
        return out
    }

    private fun readU16Array(raf: RandomAccessFile, offset: Long, count: Int, order: ByteOrder): List<Int> {
        val buf = ByteArray(count * 2)
        raf.seek(offset)
        raf.readFully(buf)
        val bb = ByteBuffer.wrap(buf).order(order)
        val out = ArrayList<Int>(count)
        repeat(count) {
            out.add(bb.short.toInt() and 0xFFFF)
        }
        return out
    }

    private fun readValues(
        raf: RandomAccessFile,
        order: ByteOrder,
        fileSize: Long,
        typeId: Int,
        valCount: Int,
        valueOrOffset: Int
    ): List<Int> {
        if (valCount <= 0) return emptyList()
        val sizeBytes = TYPE_SIZES.getOrDefault(typeId, 1) * valCount
        return if (sizeBytes <= 4) {
            val buf = ByteBuffer.allocate(4).order(order).putInt(valueOrOffset)
            buf.flip()
            when (typeId) {
                3 -> {
                    val out = ArrayList<Int>(valCount)
                    repeat(valCount) {
                        out.add(buf.short.toInt() and 0xFFFF)
                    }
                    out
                }
                4 -> listOf(buf.int)
                else -> emptyList()
            }
        } else {
            val off = valueOrOffset.toLong()
            if (off + sizeBytes > fileSize) return emptyList()
            when (typeId) {
                3 -> readU16Array(raf, off, valCount, order)
                4 -> readU32Array(raf, off, valCount, order)
                else -> emptyList()
            }
        }
    }

    private fun readBytes(raf: RandomAccessFile, offset: Long, size: Int): ByteArray {
        val buf = ByteArray(size)
        raf.seek(offset)
        raf.readFully(buf)
        return buf
    }

    private fun readU32BE(buf: ByteArray, offset: Int): Int {
        return ((buf[offset].toInt() and 0xFF) shl 24) or
            ((buf[offset + 1].toInt() and 0xFF) shl 16) or
            ((buf[offset + 2].toInt() and 0xFF) shl 8) or
            (buf[offset + 3].toInt() and 0xFF)
    }

    private fun clampToInt(value: Long): Int {
        return when {
            value > Int.MAX_VALUE -> Int.MAX_VALUE
            value < Int.MIN_VALUE -> Int.MIN_VALUE
            else -> value.toInt()
        }
    }

    private fun zipPolyglotFlags(raf: RandomAccessFile, size: Long): Pair<Int, Int> {
        val tailWindow = 1024 * 1024
        val eocdTail = 65536
        val start = max(0, (size - tailWindow).toInt())
        val tailSize = (size - start).toInt()
        val tail = ByteArray(tailSize)
        raf.seek(start.toLong())
        raf.readFully(tail)

        val eocd = indexOf(tail, byteArrayOf(0x50, 0x4B, 0x05, 0x06))
        if (eocd == -1) {
            return Pair(0, 0)
        }
        val eocdAbs = start + eocd
        if (size - eocdAbs > eocdTail) {
            return Pair(0, 0)
        }
        val local = indexOf(tail, byteArrayOf(0x50, 0x4B, 0x03, 0x04))
        return Pair(1, if (local != -1) 1 else 0)
    }

    private fun indexOf(haystack: ByteArray, needle: ByteArray, startFrom: Int = 0): Int {
        if (needle.isEmpty() || haystack.size < needle.size) return -1
        outer@ for (i in startFrom..haystack.size - needle.size) {
            for (j in needle.indices) {
                if (haystack[i + j] != needle[j]) continue@outer
            }
            return i
        }
        return -1
    }
}
