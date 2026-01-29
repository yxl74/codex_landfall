package com.landfall.hybriddetector

import android.content.Context
import org.json.JSONObject

data class JpegMeta(
    val structFeatureNames: List<String>,
    val threshold: Float,
) {
    companion object {
        fun fromAssets(context: Context): JpegMeta {
            val metaJson = context.assets.open("jpeg_model_meta.json")
                .bufferedReader().use { it.readText() }
            val meta = JSONObject(metaJson)
            val namesJson = meta.getJSONArray("struct_feature_names")
            val names = mutableListOf<String>()
            for (i in 0 until namesJson.length()) {
                names.add(namesJson.getString(i))
            }
            val threshold = meta.getDouble("threshold").toFloat()
            return JpegMeta(structFeatureNames = names, threshold = threshold)
        }
    }
}

