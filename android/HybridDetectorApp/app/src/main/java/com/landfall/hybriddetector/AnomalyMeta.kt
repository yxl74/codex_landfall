package com.landfall.hybriddetector

import android.content.Context
import org.json.JSONObject

data class AnomalyMeta(
    val bytesMode: String,
    val structFeatureNames: List<String>,
    val threshold: Float,
) {
    companion object {
        fun fromAssets(context: Context): AnomalyMeta {
            val metaJson = context.assets.open("anomaly_model_meta.json")
                .bufferedReader().use { it.readText() }
            val meta = JSONObject(metaJson)
            val bytesMode = meta.getString("bytes_mode")
            val namesJson = meta.getJSONArray("struct_feature_names")
            val names = mutableListOf<String>()
            for (i in 0 until namesJson.length()) {
                names.add(namesJson.getString(i))
            }
            val threshold = meta.getDouble("threshold").toFloat()
            return AnomalyMeta(bytesMode, names, threshold)
        }
    }
}
