package com.landfall.hybriddetector

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class AnomalyModelRunner(private val interpreter: Interpreter, val threshold: Float) {
    companion object {
        fun fromAssets(context: Context, threshold: Float): AnomalyModelRunner {
            val model = loadModel(context, "anomaly_ae.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(2)
                setUseXNNPACK(true)
            }
            return AnomalyModelRunner(Interpreter(model, options), threshold)
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
