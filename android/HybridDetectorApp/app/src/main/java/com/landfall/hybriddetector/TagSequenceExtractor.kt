package com.landfall.hybriddetector

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

private data class IfdEntry(val offset: Long, val kind: Int)

const val TAGSEQ_FEATURE_DIM = 12

private const val IFD_MAIN = 0
private const val IFD_SUB = 1
private const val IFD_EXIF = 2
private const val IFD_GPS = 3
private const val IFD_INTEROP = 4

private const val TAG_SUBIFD = 330
private const val TAG_EXIF_IFD = 34665
private const val TAG_GPS_IFD = 34853
private const val TAG_INTEROP_IFD = 40965
private const val TAG_DNG_VERSION = 50706
private const val TAG_OPCODE_LIST1 = 51008
private const val TAG_OPCODE_LIST2 = 51009
private const val TAG_OPCODE_LIST3 = 51022

private val POINTER_TAGS = setOf(TAG_SUBIFD, TAG_EXIF_IFD, TAG_GPS_IFD, TAG_INTEROP_IFD)

private val TYPE_SIZES = mapOf(
    1 to 1, 2 to 1, 3 to 2, 4 to 4, 5 to 8, 6 to 1, 7 to 1,
    8 to 2, 9 to 4, 10 to 8, 11 to 4, 12 to 8
)

private val EXPECTED_TYPES = mapOf(
    256 to setOf(3, 4),
    257 to setOf(3, 4),
    258 to setOf(3),
    259 to setOf(3),
    262 to setOf(3),
    271 to setOf(2),
    272 to setOf(2),
    273 to setOf(3, 4),
    277 to setOf(3),
    278 to setOf(3, 4),
    279 to setOf(3, 4),
    282 to setOf(5),
    283 to setOf(5),
    284 to setOf(3),
    296 to setOf(3),
    305 to setOf(2),
    306 to setOf(2),
    513 to setOf(4),
    514 to setOf(4),
    TAG_SUBIFD to setOf(4),
    TAG_EXIF_IFD to setOf(4),
    TAG_GPS_IFD to setOf(4),
    TAG_INTEROP_IFD to setOf(4),
    TAG_DNG_VERSION to setOf(1),
    TAG_OPCODE_LIST1 to setOf(1, 7),
    TAG_OPCODE_LIST2 to setOf(1, 7),
    TAG_OPCODE_LIST3 to setOf(1, 7),
)


data class TagSeqInput(
    val features: Array<FloatArray>,
    val tagIds: IntArray,
    val typeIds: IntArray,
    val ifdKinds: IntArray,
    val length: Int,
    val isDng: Boolean,
)

object TagSequenceExtractor {
    fun extract(path: String, maxLen: Int): TagSeqInput? {
        val file = File(path)
        val size = file.length()
        if (size < 8) return null

        RandomAccessFile(file, "r").use { raf ->
            val order = readByteOrder(raf) ?: return null
            val magic = readU16(raf, 2L, order)
            if (magic != 42) return null
            val root = readU32(raf, 4L, order).toLong()
            if (root <= 0 || root >= size) return null

            val features = Array(maxLen) { FloatArray(TAGSEQ_FEATURE_DIM) }
            val tagIds = IntArray(maxLen)
            val typeIds = IntArray(maxLen)
            val ifdKinds = IntArray(maxLen)

            var length = 0
            var hasDng = false

            val visited = mutableSetOf<Long>()
            val queue = ArrayDeque<IfdEntry>()
            queue.add(IfdEntry(root, IFD_MAIN))

            while (queue.isNotEmpty()) {
                val entry = queue.removeFirst()
                val off = entry.offset
                val ifdKind = entry.kind
                if (off <= 0 || off >= size || visited.contains(off)) continue
                visited.add(off)

                val count = readU16(raf, off, order)
                if (count <= 0) continue
                val entryBase = off + 2
                if (entryBase + count * 12L > size) continue

                var prevTag: Int? = null
                for (i in 0 until count) {
                    val entryOff = entryBase + i * 12L
                    if (entryOff + 12 > size) break
                    val tag = readU16(raf, entryOff, order)
                    val typeIdRaw = readU16(raf, entryOff + 2, order)
                    val valCountRaw = readU32(raf, entryOff + 4, order)
                    val valueOrOffsetRaw = readU32(raf, entryOff + 8, order)
                    val valCount = if (valCountRaw > Int.MAX_VALUE) Int.MAX_VALUE else valCountRaw.toInt()

                    if (tag == TAG_DNG_VERSION) {
                        hasDng = true
                    }

                    // TIFF type IDs are defined in 1..12. Map unexpected values to a single UNK
                    // bucket for the embedding to avoid a huge vocab while preserving a separate
                    // "invalid" signal via typeNorm/typeMismatch.
                    val typeId = if (typeIdRaw in 1..12) typeIdRaw else 13

                    val typeSize = TYPE_SIZES[typeIdRaw] ?: 0
                    val byteCount = valCount.toLong() * typeSize.toLong()
                    val isImmediate = if (byteCount <= 4) 1.0f else 0.0f
                    val offsetNorm = if (byteCount > 4 && size > 0) {
                        clampFloat(valueOrOffsetRaw.toFloat() / size.toFloat(), 0.0f, 1.0f)
                    } else {
                        0.0f
                    }
                    val offsetValid = if (byteCount <= 4) {
                        1.0f
                    } else {
                        if (valueOrOffsetRaw > 0 && valueOrOffsetRaw + byteCount <= size) 1.0f else 0.0f
                    }

                    val coarseBucket = clampFloat((tag / 256.0f) / 255.0f, 0.0f, 1.0f)
                    val logCount = log1pNorm(valCount.toFloat(), 16.0f)
                    val logBytes = log1pNorm(byteCount.toFloat(), 24.0f)

                    var orderDelta = 0.0f
                    var orderViolation = 0.0f
                    if (prevTag != null) {
                        val delta = tag - prevTag
                        orderViolation = if (delta < 0) 1.0f else 0.0f
                        orderDelta = clampFloat(delta.toFloat(), -128.0f, 128.0f) / 128.0f
                    }
                    prevTag = tag

                    val isPointer = if (POINTER_TAGS.contains(tag)) 1.0f else 0.0f
                    val expected = EXPECTED_TYPES[tag]
                    val typeMismatch = if (expected != null && !expected.contains(typeIdRaw)) 1.0f else 0.0f

                    val typeNorm = clampFloat(typeIdRaw / 12.0f, 0.0f, 1.0f)
                    val ifdKindNorm = clampFloat(ifdKind / 4.0f, 0.0f, 1.0f)

                    if (length < maxLen) {
                        tagIds[length] = tag
                        typeIds[length] = typeId
                        ifdKinds[length] = ifdKind
                        features[length][0] = coarseBucket
                        features[length][1] = logCount
                        features[length][2] = logBytes
                        features[length][3] = offsetNorm
                        features[length][4] = offsetValid
                        features[length][5] = isImmediate
                        features[length][6] = isPointer
                        features[length][7] = orderDelta
                        features[length][8] = orderViolation
                        features[length][9] = typeMismatch
                        features[length][10] = typeNorm
                        features[length][11] = ifdKindNorm
                        length += 1
                    }

                    if (POINTER_TAGS.contains(tag) && byteCount >= 4 && valueOrOffsetRaw > 0) {
                        queue.add(IfdEntry(valueOrOffsetRaw, kindForPointer(tag)))
                    }

                    if (tag == TAG_SUBIFD) {
                        if (byteCount <= 4) {
                            if (valueOrOffsetRaw > 0) queue.add(IfdEntry(valueOrOffsetRaw, IFD_SUB))
                        } else {
                            val offs = readU32Array(raf, valueOrOffsetRaw, valCount, order)
                            offs.forEach { offVal ->
                                if (offVal > 0) queue.add(IfdEntry(offVal, IFD_SUB))
                            }
                        }
                    }
                }

                val nextIfd = readU32(raf, entryBase + count * 12L, order)
                if (nextIfd > 0) {
                    queue.add(IfdEntry(nextIfd, ifdKind))
                }
            }

            if (length == 0) return null
            return TagSeqInput(
                features = features,
                tagIds = tagIds,
                typeIds = typeIds,
                ifdKinds = ifdKinds,
                length = length,
                isDng = hasDng,
            )
        }
    }

    private fun readByteOrder(raf: RandomAccessFile): ByteOrder? {
        val buf = ByteArray(2)
        raf.seek(0L)
        raf.readFully(buf)
        val endian = String(buf)
        return when (endian) {
            "II" -> ByteOrder.LITTLE_ENDIAN
            "MM" -> ByteOrder.BIG_ENDIAN
            else -> null
        }
    }

    private fun readU16(raf: RandomAccessFile, offset: Long, order: ByteOrder): Int {
        val buf = ByteArray(2)
        raf.seek(offset)
        raf.readFully(buf)
        return ByteBuffer.wrap(buf).order(order).short.toInt() and 0xFFFF
    }

    private fun readU32(raf: RandomAccessFile, offset: Long, order: ByteOrder): Long {
        val buf = ByteArray(4)
        raf.seek(offset)
        raf.readFully(buf)
        return ByteBuffer.wrap(buf).order(order).int.toLong() and 0xFFFFFFFFL
    }

    private fun readU32Array(raf: RandomAccessFile, offset: Long, count: Int, order: ByteOrder): List<Long> {
        if (count <= 0) return emptyList()
        val buf = ByteArray(count * 4)
        raf.seek(offset)
        raf.readFully(buf)
        val bb = ByteBuffer.wrap(buf).order(order)
        val out = ArrayList<Long>(count)
        repeat(count) {
            out.add(bb.int.toLong() and 0xFFFFFFFFL)
        }
        return out
    }

    private fun clampFloat(value: Float, lo: Float, hi: Float): Float {
        return max(lo, min(hi, value))
    }

    private fun log1pNorm(value: Float, maxLog: Float): Float {
        val v = if (value < 0f) 0f else value
        val logVal = ln(1.0f + v)
        val clipped = min(logVal, maxLog)
        return clipped / maxLog
    }

    private fun kindForPointer(tag: Int): Int {
        return when (tag) {
            TAG_EXIF_IFD -> IFD_EXIF
            TAG_GPS_IFD -> IFD_GPS
            TAG_INTEROP_IFD -> IFD_INTEROP
            else -> IFD_SUB
        }
    }
}
