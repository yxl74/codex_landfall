package com.landfall.hybriddetector

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream

class ModelRunner(private val interpreter: Interpreter) {
    companion object {
        fun fromAssets(context: Context): ModelRunner {
            val model = loadModel(context, "hybrid_model.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(2)
                setUseXNNPACK(true)
            }
            return ModelRunner(Interpreter(model, options))
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

    fun predict(features: FloatArray): Float {
        val input = arrayOf(features)
        val output = Array(1) { FloatArray(1) }
        interpreter.run(input, output)
        return output[0][0]
    }
}
