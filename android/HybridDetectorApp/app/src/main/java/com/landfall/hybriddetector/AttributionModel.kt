package com.landfall.hybriddetector

import android.content.Context
import org.json.JSONObject
import kotlin.math.ln

data class Contribution(
    val name: String,
    val value: Float,
    val standardized: Float,
    val weight: Float,
    val contribution: Float,
)

data class AttributionResult(
    val logit: Float,
    val score: Float,
    val topPositive: List<Contribution>,
    val topNegative: List<Contribution>,
)

class AttributionModel(
    private val bytesMode: String,
    private val structFeatureNames: List<String>,
    private val weights: FloatArray,
    private val bias: Float,
    private val mean: FloatArray,
    private val std: FloatArray,
) {
    companion object {
        fun fromAssets(context: Context): AttributionModel {
            val jsonText = context.assets.open("hybrid_model_params.json")
                .bufferedReader().use { it.readText() }
            val root = JSONObject(jsonText)
            val bytesMode = root.getString("bytes_mode")
            val namesJson = root.getJSONArray("struct_feature_names")
            val names = mutableListOf<String>()
            for (i in 0 until namesJson.length()) {
                names.add(namesJson.getString(i))
            }
            val weights = toFloatArray(root.getJSONArray("weights"))
            val mean = toFloatArray(root.getJSONArray("mean"))
            val std = toFloatArray(root.getJSONArray("std"))
            val bias = root.getDouble("bias").toFloat()
            return AttributionModel(bytesMode, names, weights, bias, mean, std)
        }

        private fun toFloatArray(array: org.json.JSONArray): FloatArray {
            val out = FloatArray(array.length())
            for (i in 0 until array.length()) {
                out[i] = array.getDouble(i).toFloat()
            }
            return out
        }
    }

    private val featureNames: List<String> = buildFeatureNames()
    private val log1pMask: BooleanArray = buildLog1pMask()

    fun computeTopContributions(features: FloatArray, topK: Int): AttributionResult? {
        if (features.size != weights.size || mean.size != weights.size || std.size != weights.size) {
            return null
        }
        val contributions = ArrayList<Contribution>(features.size)
        var logit = bias
        for (i in features.indices) {
            val raw = features[i]
            val adjusted = if (log1pMask[i]) ln(1.0 + raw.toDouble()).toFloat() else raw
            val denom = if (std[i] == 0f) 1f else std[i]
            val standardized = (adjusted - mean[i]) / denom
            val contrib = standardized * weights[i]
            logit += contrib
            contributions.add(
                Contribution(
                    name = featureNames[i],
                    value = raw,
                    standardized = standardized,
                    weight = weights[i],
                    contribution = contrib,
                )
            )
        }

        val topPositive = contributions
            .filter { it.contribution > 0f }
            .sortedByDescending { it.contribution }
            .take(topK)
        val topNegative = contributions
            .filter { it.contribution < 0f }
            .sortedBy { it.contribution }
            .take(topK)

        val score = (1.0 / (1.0 + kotlin.math.exp(-logit.toDouble()))).toFloat()
        return AttributionResult(logit, score, topPositive, topNegative)
    }

    private fun buildFeatureNames(): List<String> {
        val names = ArrayList<String>()
        when (bytesMode) {
            "hist" -> {
                for (i in 0..256) names.add("byte_beg_hist[$i]")
                for (i in 0..256) names.add("byte_end_hist[$i]")
            }
            "raw" -> {
                for (i in 0 until 1024) names.add("byte_beg_raw[$i]")
                for (i in 0 until 1024) names.add("byte_end_raw[$i]")
            }
            else -> {
                // Fallback to avoid crash if bytesMode is unexpected.
                for (i in weights.indices) names.add("feature[$i]")
                return names
            }
        }
        names.addAll(structFeatureNames)
        if (names.size != weights.size) {
            // Safety fallback for mismatch.
            return List(weights.size) { idx -> "feature[$idx]" }
        }
        return names
    }

    private fun buildLog1pMask(): BooleanArray {
        val mask = BooleanArray(weights.size)
        val bytesDim = when (bytesMode) {
            "hist" -> 514
            "raw" -> 2048
            else -> 0
        }
        val logFields = setOf(
            "min_width",
            "min_height",
            "ifd_entry_max",
            "subifd_count_sum",
            "new_subfile_types_unique",
            "total_opcodes",
            "unknown_opcodes",
            "max_opcode_id",
            "opcode_list1_bytes",
            "opcode_list2_bytes",
            "opcode_list3_bytes",
        )
        for ((i, name) in structFeatureNames.withIndex()) {
            if (name in logFields) {
                val idx = bytesDim + i
                if (idx in mask.indices) {
                    mask[idx] = true
                }
            }
        }
        return mask
    }
}
