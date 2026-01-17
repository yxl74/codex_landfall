package com.landfall.hybriddetector

import android.content.Context
import org.json.JSONObject
import java.io.RandomAccessFile
import kotlin.math.min

class FeatureExtractor(
    private val bytesMode: String,
    private val structFeatureNames: List<String>,
) {
    companion object {
        fun fromAssets(context: Context): FeatureExtractor {
            val metaJson = context.assets.open("hybrid_model_meta.json").bufferedReader().use { it.readText() }
            val meta = JSONObject(metaJson)
            val bytesMode = meta.getString("bytes_mode")
            val namesJson = meta.getJSONArray("struct_feature_names")
            val names = mutableListOf<String>()
            for (i in 0 until namesJson.length()) {
                names.add(namesJson.getString(i))
            }
            return FeatureExtractor(bytesMode, names)
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
            val byteFeatures = when (bytesMode) {
                "hist" -> extractBytesHist(path)
                "raw" -> extractBytesRaw(path)
                else -> {
                    android.util.Log.w("HybridDetector", "Unknown bytes_mode=$bytesMode")
                    return null
                }
            }

            val tiff = TiffParser.parse(path)
            val structMap = tiff.toFeatureMap()
            val structFeatures = FloatArray(structFeatureNames.size)
            for (i in structFeatureNames.indices) {
                val name = structFeatureNames[i]
                structFeatures[i] = (structMap[name] ?: 0).toFloat()
            }

            val out = FloatArray(byteFeatures.size + structFeatures.size)
            System.arraycopy(byteFeatures, 0, out, 0, byteFeatures.size)
            System.arraycopy(structFeatures, 0, out, byteFeatures.size, structFeatures.size)
            out
        } catch (e: Exception) {
            android.util.Log.w("HybridDetector", "Failed to extract features for $path: ${e.message}")
            null
        }
    }

    private fun extractBytesRaw(path: String): FloatArray {
        val f = RandomAccessFile(path, "r")
        val size = f.length().toInt()
        val blockSize = min(4096, size)
        val begBlock = ByteArray(blockSize)
        f.seek(0)
        f.readFully(begBlock)
        val endBlock = ByteArray(blockSize)
        f.seek((size - blockSize).toLong())
        f.readFully(endBlock)
        f.close()

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

    private fun extractBytesHist(path: String): FloatArray {
        val f = RandomAccessFile(path, "r")
        val size = f.length().toInt()
        val blockSize = min(4096, size)
        val begBlock = ByteArray(blockSize)
        f.seek(0)
        f.readFully(begBlock)
        val endBlock = ByteArray(blockSize)
        f.seek((size - blockSize).toLong())
        f.readFully(endBlock)
        f.close()

        val beg = stripPrefix(begBlock)
        val end = stripSuffix(endBlock)

        val begHist = histogram1024(beg)
        val endHist = histogram1024(end)
        val out = FloatArray(514)
        System.arraycopy(begHist, 0, out, 0, 257)
        System.arraycopy(endHist, 0, out, 257, 257)
        return out
    }

    private fun histogram1024(data: ByteArray): FloatArray {
        val hist = FloatArray(257)
        val len = min(1024, data.size)
        val pad = 1024 - len
        hist[256] = pad.toFloat()
        for (i in 0 until len) {
            val v = data[i].toInt() and 0xFF
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
}
