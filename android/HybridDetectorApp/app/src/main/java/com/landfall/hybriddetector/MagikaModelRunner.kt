package com.landfall.hybriddetector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.IntBuffer
import kotlin.math.min

enum class MagikaPredictionMode {
    BEST_GUESS,
    MEDIUM_CONFIDENCE,
    HIGH_CONFIDENCE
}

data class MagikaResult(
    val dlLabel: String,
    val outputLabel: String,
    val score: Float
)

private data class MagikaConfig(
    val begSize: Int,
    val midSize: Int,
    val endSize: Int,
    val blockSize: Int,
    val paddingToken: Int,
    val minFileSizeForDl: Int,
    val mediumConfidenceThreshold: Float,
    val targetLabels: List<String>,
    val thresholds: Map<String, Float>,
    val overwriteMap: Map<String, String>
)

class MagikaModelRunner private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val config: MagikaConfig,
    private val predictionMode: MagikaPredictionMode
) {
    companion object {
        private const val MODEL_ASSET = "magika/standard_v3_3/model.onnx"
        private const val CONFIG_ASSET = "magika/standard_v3_3/config.min.json"
        private const val MODEL_CACHE_NAME = "magika_standard_v3_3.onnx"
        private const val INPUT_NAME = "bytes"
        private const val OUTPUT_NAME = "target_label"
        private val WHITESPACE_BYTES = setOf(0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20)

        fun fromAssets(
            context: Context,
            predictionMode: MagikaPredictionMode = MagikaPredictionMode.BEST_GUESS
        ): MagikaModelRunner {
            val config = loadConfig(context)
            require(config.midSize == 0) { "Magika mid features not supported" }
            val env = OrtEnvironment.getEnvironment()
            val modelFile = copyAsset(context, MODEL_ASSET, MODEL_CACHE_NAME)
            val session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
            return MagikaModelRunner(env, session, config, predictionMode)
        }

        private fun loadConfig(context: Context): MagikaConfig {
            val json = JSONObject(loadAssetText(context, CONFIG_ASSET))
            val labelsArray = json.getJSONArray("target_labels_space")
            val labels = ArrayList<String>(labelsArray.length())
            for (i in 0 until labelsArray.length()) {
                labels.add(labelsArray.getString(i))
            }

            val thresholdsJson = json.getJSONObject("thresholds")
            val thresholds = mutableMapOf<String, Float>()
            for (key in thresholdsJson.keys()) {
                thresholds[key] = thresholdsJson.getDouble(key).toFloat()
            }

            val overwriteJson = json.getJSONObject("overwrite_map")
            val overwriteMap = mutableMapOf<String, String>()
            for (key in overwriteJson.keys()) {
                overwriteMap[key] = overwriteJson.getString(key)
            }

            return MagikaConfig(
                begSize = json.getInt("beg_size"),
                midSize = json.getInt("mid_size"),
                endSize = json.getInt("end_size"),
                blockSize = json.getInt("block_size"),
                paddingToken = json.getInt("padding_token"),
                minFileSizeForDl = json.getInt("min_file_size_for_dl"),
                mediumConfidenceThreshold = json.getDouble("medium_confidence_threshold").toFloat(),
                targetLabels = labels,
                thresholds = thresholds,
                overwriteMap = overwriteMap
            )
        }

        private fun loadAssetText(context: Context, assetPath: String): String {
            return context.assets.open(assetPath).bufferedReader().use { it.readText() }
        }

        private fun copyAsset(context: Context, assetPath: String, destName: String): File {
            val outFile = File(context.filesDir, destName)
            if (outFile.exists() && outFile.length() > 0L) {
                return outFile
            }
            context.assets.open(assetPath).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(8192)
                    while (true) {
                        val read = input.read(buffer)
                        if (read <= 0) break
                        output.write(buffer, 0, read)
                    }
                }
            }
            return outFile
        }
    }

    fun classify(path: String): MagikaResult? {
        val file = File(path)
        if (!file.exists() || !file.isFile) return null
        val size = file.length()
        if (size < config.minFileSizeForDl) {
            val label = labelFromFewBytes(file)
            return MagikaResult(dlLabel = label, outputLabel = label, score = 1.0f)
        }

        val bytesToRead = min(config.blockSize.toLong(), size).toInt()
        val begContent: ByteArray
        val endContent: ByteArray
        RandomAccessFile(file, "r").use { raf ->
            begContent = raf.readAt(0L, bytesToRead)
            endContent = raf.readAt(size - bytesToRead, bytesToRead)
        }

        val begStripped = lstripWhitespace(begContent)
        val endStripped = rstripWhitespace(endContent)
        val begInts = getBegInts(begStripped, config.begSize, config.paddingToken)
        val endInts = getEndInts(endStripped, config.endSize, config.paddingToken)
        val features = IntArray(config.begSize + config.endSize)
        System.arraycopy(begInts, 0, features, 0, begInts.size)
        System.arraycopy(endInts, 0, features, begInts.size, endInts.size)

        val inputTensor = OnnxTensor.createTensor(
            env,
            IntBuffer.wrap(features),
            longArrayOf(1L, features.size.toLong())
        )
        inputTensor.use { tensor ->
            val outputs = session.run(mapOf(INPUT_NAME to tensor))
            outputs.use { result ->
                val raw = result[0].value as Array<FloatArray>
                val scores = raw[0]
                var maxIdx = 0
                var maxScore = scores[0]
                for (i in 1 until scores.size) {
                    if (scores[i] > maxScore) {
                        maxScore = scores[i]
                        maxIdx = i
                    }
                }
                val dlLabel = config.targetLabels.getOrElse(maxIdx) { "unknown" }
                val outputLabel = applyOutputLabel(dlLabel, maxScore)
                return MagikaResult(dlLabel = dlLabel, outputLabel = outputLabel, score = maxScore)
            }
        }
    }

    private fun labelFromFewBytes(file: File): String {
        return try {
            val bytes = file.readBytes()
            bytes.toString(Charsets.UTF_8)
            "txt"
        } catch (_: Exception) {
            "unknown"
        }
    }

    private fun applyOutputLabel(dlLabel: String, score: Float): String {
        val overwritten = config.overwriteMap[dlLabel] ?: dlLabel
        return when (predictionMode) {
            MagikaPredictionMode.BEST_GUESS -> overwritten
            MagikaPredictionMode.MEDIUM_CONFIDENCE -> {
                if (score >= config.mediumConfidenceThreshold) overwritten else "unknown"
            }
            MagikaPredictionMode.HIGH_CONFIDENCE -> {
                val thr = config.thresholds[dlLabel] ?: config.mediumConfidenceThreshold
                if (score >= thr) overwritten else "unknown"
            }
        }
    }

    private fun lstripWhitespace(input: ByteArray): ByteArray {
        var idx = 0
        while (idx < input.size) {
            val v = input[idx].toInt() and 0xFF
            if (!WHITESPACE_BYTES.contains(v)) break
            idx++
        }
        return input.copyOfRange(idx, input.size)
    }

    private fun rstripWhitespace(input: ByteArray): ByteArray {
        var idx = input.size - 1
        while (idx >= 0) {
            val v = input[idx].toInt() and 0xFF
            if (!WHITESPACE_BYTES.contains(v)) break
            idx--
        }
        return if (idx < 0) ByteArray(0) else input.copyOfRange(0, idx + 1)
    }

    private fun getBegInts(content: ByteArray, begSize: Int, paddingToken: Int): IntArray {
        val truncated = if (content.size > begSize) content.copyOfRange(0, begSize) else content
        val ints = IntArray(begSize)
        var i = 0
        while (i < truncated.size) {
            ints[i] = truncated[i].toInt() and 0xFF
            i++
        }
        while (i < begSize) {
            ints[i] = paddingToken
            i++
        }
        return ints
    }

    private fun getEndInts(content: ByteArray, endSize: Int, paddingToken: Int): IntArray {
        val truncated = if (content.size > endSize) {
            content.copyOfRange(content.size - endSize, content.size)
        } else {
            content
        }
        val ints = IntArray(endSize)
        var padCount = endSize - truncated.size
        var i = 0
        while (i < padCount) {
            ints[i] = paddingToken
            i++
        }
        var j = 0
        while (j < truncated.size) {
            ints[padCount + j] = truncated[j].toInt() and 0xFF
            j++
        }
        return ints
    }

    private fun RandomAccessFile.readAt(offset: Long, size: Int): ByteArray {
        val out = ByteArray(size)
        seek(offset)
        val read = read(out, 0, size)
        return if (read == size) out else out.copyOf(read.coerceAtLeast(0))
    }
}
