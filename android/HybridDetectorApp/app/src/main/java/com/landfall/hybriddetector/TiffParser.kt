package com.landfall.hybriddetector

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

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
    val zipEocdNearEnd: Int,
    val zipLocalInTail: Int,
    val flagOpcodeAnomaly: Int,
    val flagTinyDimsLowIfd: Int,
    val flagZipPolyglot: Int,
    val flagDngJpegMismatch: Int,
    val flagMagikaExtMismatch: Int,
    val flagAny: Int,
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
            "zip_eocd_near_end" to zipEocdNearEnd,
            "zip_local_in_tail" to zipLocalInTail,
            "flag_opcode_anomaly" to flagOpcodeAnomaly,
            "flag_tiny_dims_low_ifd" to flagTinyDimsLowIfd,
            "flag_zip_polyglot" to flagZipPolyglot,
            "flag_dng_jpeg_mismatch" to flagDngJpegMismatch,
            "flag_magika_ext_mismatch" to flagMagikaExtMismatch,
            "flag_any" to flagAny,
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
    private const val TAG_SUBIFD = 330
    private const val TAG_EXIF_IFD = 34665
    private const val TAG_DNG_VERSION = 50706
    private const val TAG_NEW_SUBFILE_TYPE = 254
    private const val TAG_OPCODE_LIST1 = 51008
    private const val TAG_OPCODE_LIST2 = 51009
    private const val TAG_OPCODE_LIST3 = 51022

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

            var totalOpcodes = 0
            var unknownOpcodes = 0
            var maxOpcodeId = 0
            var list1Bytes = 0
            var list2Bytes = 0
            var list3Bytes = 0

            for ((tag, pair) in opcodeLists) {
                val offset = pair.first
                val sizeBytes = pair.second
                if (offset <= 0 || sizeBytes < 4 || offset + sizeBytes > size) continue
                val buf = readBytes(raf, offset, sizeBytes)
                val opcodeCount = readU32BE(buf, 0)
                var pos = 4
                var parsed = 0
                while (parsed < opcodeCount && pos + 16 <= buf.size) {
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
                zipEocdNearEnd = zipEocdNearEnd,
                zipLocalInTail = zipLocalInTail,
                flagOpcodeAnomaly = flagOpcodeAnomaly,
                flagTinyDimsLowIfd = flagTinyDimsLowIfd,
                flagZipPolyglot = flagZipPolyglot,
                flagDngJpegMismatch = flagDngJpegMismatch,
                flagMagikaExtMismatch = flagMagikaExtMismatch,
                flagAny = flagAny,
            )
        }
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
            zipEocdNearEnd = 0,
            zipLocalInTail = 0,
            flagOpcodeAnomaly = 0,
            flagTinyDimsLowIfd = 0,
            flagZipPolyglot = 0,
            flagDngJpegMismatch = 0,
            flagMagikaExtMismatch = 0,
            flagAny = 0,
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

    private fun indexOf(haystack: ByteArray, needle: ByteArray): Int {
        if (needle.isEmpty() || haystack.size < needle.size) return -1
        outer@ for (i in 0..haystack.size - needle.size) {
            for (j in needle.indices) {
                if (haystack[i + j] != needle[j]) continue@outer
            }
            return i
        }
        return -1
    }
}
