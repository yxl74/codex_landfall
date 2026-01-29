package com.landfall.hybriddetector

import android.util.Log
import java.io.File

class JpegFeatureExtractor(private val structFeatureNames: List<String>) {
    companion object {
        fun fromMeta(meta: JpegMeta): JpegFeatureExtractor {
            return JpegFeatureExtractor(meta.structFeatureNames)
        }
    }

    fun extract(path: String): FloatArray? {
        val file = File(path)
        val exists = file.exists()
        val canRead = file.canRead()
        if (!exists || !canRead) {
            Log.w("HybridDetector", "Missing or unreadable file: $path exists=$exists canRead=$canRead")
            return null
        }
        return try {
            val parsed = JpegParser.parse(path) ?: return null
            val featureMap = parsed.toFeatureMap()
            val out = FloatArray(structFeatureNames.size)
            for (i in structFeatureNames.indices) {
                val name = structFeatureNames[i]
                out[i] = (featureMap[name] ?: 0).toFloat()
            }
            out
        } catch (e: Exception) {
            Log.w("HybridDetector", "Failed to extract JPEG features for $path: ${e.message}")
            null
        }
    }
}

