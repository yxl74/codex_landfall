package com.landfall.hybriddetector

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream

class TagSeqModelRunner(private val interpreter: Interpreter, private val meta: TagSeqMeta) {
    companion object {
        private const val MODEL_NAME = "tagseq_gru_ae.tflite"
        private const val TAG = "TagSeqModel"

        fun fromAssets(context: Context, meta: TagSeqMeta): TagSeqModelRunner {
            val model = loadModel(context, MODEL_NAME)
            val options = Interpreter.Options().apply {
                setNumThreads(2)
                setUseXNNPACK(true)
            }
            return TagSeqModelRunner(Interpreter(model, options), meta)
        }

        private fun loadModel(context: Context, assetName: String): MappedByteBuffer {
            context.assets.openFd(assetName).use { fd ->
                FileInputStream(fd.fileDescriptor).use { fis ->
                    val channel = fis.channel
                    return channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        fd.startOffset,
                        fd.declaredLength
                    )
                }
            }
        }
    }

    fun predict(input: TagSeqInput): Float? {
        val length = input.length
        if (length <= 0) return null

        val featureDim = meta.featureDim
        val maxLen = meta.maxSeqLen

        val output = Array(1) { Array(maxLen) { FloatArray(featureDim) } }

        val inputs = arrayOfNulls<Any>(interpreter.inputTensorCount)
        val inputNameMap = mutableMapOf<String, Int>()
        for (i in 0 until interpreter.inputTensorCount) {
            val name = interpreter.getInputTensor(i).name()
            inputNameMap[name] = i
        }

        fun findIndex(key: String): Int? {
            return inputNameMap.entries.firstOrNull { it.key.contains(key) }?.value
        }

        val idxFeatures = findIndex("features") ?: 0
        val idxTagIds = findIndex("tag_ids") ?: 1
        val idxTypeIds = findIndex("type_ids") ?: 2
        val idxIfdKinds = findIndex("ifd_kinds") ?: 3

        inputs[idxFeatures] = arrayOf(input.features)
        inputs[idxTagIds] = arrayOf(input.tagIds)
        inputs[idxTypeIds] = arrayOf(input.typeIds)
        inputs[idxIfdKinds] = arrayOf(input.ifdKinds)

        try {
            val outputs = mutableMapOf<Int, Any>(0 to output)
            interpreter.runForMultipleInputsOutputs(inputs, outputs)
        } catch (e: Exception) {
            Log.e(TAG, "TagSeq inference failed", e)
            return null
        }

        var sum = 0.0f
        for (i in 0 until length) {
            var row = 0.0f
            val inRow = input.features[i]
            val outRow = output[0][i]
            for (j in 0 until featureDim) {
                val diff = inRow[j] - outRow[j]
                row += diff * diff
            }
            sum += row
        }
        return sum / length
    }
}
