package com.landfall.hybriddetector

import java.io.RandomAccessFile
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.min

class AnomalyFeatureExtractor(
    private val bytesMode: String,
    private val structFeatureNames: List<String>,
) {
    companion object {
        private const val LN2 = 0.6931471805599453

        fun fromMeta(meta: AnomalyMeta): AnomalyFeatureExtractor {
            return AnomalyFeatureExtractor(meta.bytesMode, meta.structFeatureNames)
        }
    }

    fun extract(path: String): FloatArray? {
        val file = java.io.File(path)
        val exists = file.exists()
        val canRead = file.canRead()
        if (!exists || !canRead) {
            android.util.Log.w(
                "HybridDetector",
                "Missing or unreadable file: $path exists=$exists canRead=$canRead"
            )
            return null
        }
        return try {
            val tiff = TiffParser.parse(path)
            extractFromParsed(tiff, path)
        } catch (e: Exception) {
            android.util.Log.w("HybridDetector", "Failed to extract anomaly features for $path: ${e.message}")
            null
        }
    }

    fun extractFromParsed(tiff: TiffFeatures, path: String): FloatArray? {
        return try {
            val byteFeatures = when (bytesMode) {
                "hist" -> extractBytesHist(path)
                "raw" -> extractBytesRaw(path)
                else -> {
                    android.util.Log.w("HybridDetector", "Unknown bytes_mode=$bytesMode")
                    return null
                }
            }

            val structValues = HashMap<String, Float>()
            for ((k, v) in tiff.toFeatureMap()) {
                structValues[k] = v.toFloat()
            }

            val file = java.io.File(path)
            val fileSize = file.length()
            structValues["file_size"] = clampToFloat(fileSize)

            val entropy = computeEntropies(path, fileSize)
            structValues["header_entropy"] = entropy.header
            structValues["tail_entropy"] = entropy.tail
            structValues["overall_entropy"] = entropy.overall
            structValues["entropy_gradient"] = abs(entropy.header - entropy.tail)

            val structFeatures = FloatArray(structFeatureNames.size)
            for (i in structFeatureNames.indices) {
                val name = structFeatureNames[i]
                structFeatures[i] = structValues[name] ?: 0f
            }

            val out = FloatArray(byteFeatures.size + structFeatures.size)
            System.arraycopy(byteFeatures, 0, out, 0, byteFeatures.size)
            System.arraycopy(structFeatures, 0, out, byteFeatures.size, structFeatures.size)
            out
        } catch (e: Exception) {
            android.util.Log.w("HybridDetector", "Failed to extract anomaly features for $path: ${e.message}")
            null
        }
    }

    private fun extractBytesRaw(path: String): FloatArray {
        RandomAccessFile(path, "r").use { f ->
            val size = f.length()
            if (size <= 0) {
                return FloatArray(2048) { 256.0f / 256.0f }
            }
            val blockSize = min(4096, clampToInt(size))
            val begBlock = ByteArray(blockSize)
            f.seek(0)
            f.readFully(begBlock)
            val endBlock = ByteArray(blockSize)
            f.seek(size - blockSize.toLong())
            f.readFully(endBlock)

            val beg = stripPrefix(begBlock)
            val end = stripSuffix(endBlock)

            val out = FloatArray(2048)
            val begLen = min(1024, beg.size)
            for (i in 0 until begLen) {
                out[i] = (beg[i].toInt() and 0xFF) / 256.0f
            }
            for (i in begLen until 1024) {
                out[i] = 256.0f / 256.0f
            }

            val endLen = min(1024, end.size)
            val start = end.size - endLen
            for (i in 0 until endLen) {
                out[1024 + i] = (end[start + i].toInt() and 0xFF) / 256.0f
            }
            for (i in endLen until 1024) {
                out[1024 + i] = 256.0f / 256.0f
            }
            return out
        }
    }

    private fun extractBytesHist(path: String): FloatArray {
        RandomAccessFile(path, "r").use { f ->
            val size = f.length()
            if (size <= 0) {
                val empty = histogram1024(ByteArray(0), false)
                val out = FloatArray(514)
                System.arraycopy(empty, 0, out, 0, 257)
                System.arraycopy(empty, 0, out, 257, 257)
                return out
            }
            val blockSize = min(4096, clampToInt(size))
            val begBlock = ByteArray(blockSize)
            f.seek(0)
            f.readFully(begBlock)
            val endBlock = ByteArray(blockSize)
            f.seek(size - blockSize.toLong())
            f.readFully(endBlock)

            val beg = stripPrefix(begBlock)
            val end = stripSuffix(endBlock)

            val begHist = histogram1024(beg, false)
            val endHist = histogram1024(end, true)
            val out = FloatArray(514)
            System.arraycopy(begHist, 0, out, 0, 257)
            System.arraycopy(endHist, 0, out, 257, 257)
            return out
        }
    }

    private fun histogram1024(data: ByteArray, fromTail: Boolean): FloatArray {
        val hist = FloatArray(257)
        val len = min(1024, data.size)
        val start = if (fromTail) data.size - len else 0
        val pad = 1024 - len
        hist[256] = pad.toFloat()
        for (i in 0 until len) {
            val v = data[start + i].toInt() and 0xFF
            hist[v] += 1.0f
        }
        for (i in hist.indices) {
            hist[i] /= 1024.0f
        }
        return hist
    }

    private fun stripPrefix(data: ByteArray): ByteArray {
        var i = 0
        while (i < data.size && isWhitespace(data[i])) {
            i++
        }
        return data.copyOfRange(i, data.size)
    }

    private fun stripSuffix(data: ByteArray): ByteArray {
        var i = data.size
        while (i > 0 && isWhitespace(data[i - 1])) {
            i--
        }
        return data.copyOfRange(0, i)
    }

    private fun isWhitespace(b: Byte): Boolean {
        val v = b.toInt() and 0xFF
        return v == 0x09 || v == 0x0A || v == 0x0B || v == 0x0C || v == 0x0D || v == 0x20
    }

    private data class EntropyResult(
        val header: Float,
        val tail: Float,
        val overall: Float,
    )

    private fun computeEntropies(path: String, fileSize: Long): EntropyResult {
        val sizeInt = clampToInt(fileSize)
        val headerSize = min(4096, sizeInt)
        val tailSize = min(4096, sizeInt)
        val overallSize = min(65536, sizeInt)
        RandomAccessFile(path, "r").use { raf ->
            val header = ByteArray(headerSize)
            if (headerSize > 0) {
                raf.seek(0)
                raf.readFully(header)
            }
            val tail = ByteArray(tailSize)
            if (tailSize > 0) {
                raf.seek(fileSize - tailSize.toLong())
                raf.readFully(tail)
            }
            val overall = ByteArray(overallSize)
            if (overallSize > 0) {
                raf.seek(0)
                raf.readFully(overall)
            }
            return EntropyResult(
                header = byteEntropy(header),
                tail = byteEntropy(tail),
                overall = byteEntropy(overall),
            )
        }
    }

    private fun byteEntropy(data: ByteArray): Float {
        if (data.isEmpty()) return 0f
        val counts = IntArray(256)
        for (b in data) {
            counts[b.toInt() and 0xFF]++
        }
        val total = data.size.toDouble()
        var sum = 0.0
        for (c in counts) {
            if (c == 0) continue
            val p = c / total
            sum += -p * (ln(p) / LN2)
        }
        return sum.toFloat()
    }

    private fun clampToInt(value: Long): Int {
        return when {
            value > Int.MAX_VALUE -> Int.MAX_VALUE
            value < Int.MIN_VALUE -> Int.MIN_VALUE
            else -> value.toInt()
        }
    }

    private fun clampToFloat(value: Long): Float {
        val clamped = clampToInt(value).toLong()
        return clamped.toFloat()
    }

}
