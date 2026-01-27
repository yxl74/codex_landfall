package com.landfall.hybriddetector

import android.content.Context
import org.json.JSONObject

data class TagSeqMeta(
    val threshold: Float,
    val thresholdPercentile: Float,
    val maxSeqLen: Int,
    val featureDim: Int,
) {
    companion object {
        fun fromAssets(context: Context): TagSeqMeta {
            val metaJson = context.assets.open("tagseq_gru_ae_meta.json")
                .bufferedReader().use { it.readText() }
            val meta = JSONObject(metaJson)
            val threshold = meta.getDouble("threshold").toFloat()
            val thresholdPct = meta.getDouble("threshold_percentile").toFloat()
            val maxSeqLen = meta.getInt("max_seq_len")
            val featureDim = meta.getInt("feature_dim")
            return TagSeqMeta(threshold, thresholdPct, maxSeqLen, featureDim)
        }
    }
}
