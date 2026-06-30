package dev.distraction.demo.ml.tflite

import android.content.Context
import android.util.Log
import dev.distraction.demo.ml.api.ModelInputs
import dev.distraction.demo.ml.api.ModelRunner
import dev.distraction.demo.ml.debug.ModelInputDumpManager
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class TFLiteMultiInputRunner(
    private val context: Context,
    private val assetPath: String
) : ModelRunner {

    private var isReady = false

    private val interpreter: Interpreter by lazy {
        Log.d("TFLite_INIT", "Initializing interpreter for $assetPath")
        
        val flexDelegate = org.tensorflow.lite.flex.FlexDelegate()
        val options = Interpreter.Options().apply {
            addDelegate(flexDelegate)
            setNumThreads(4)
        }
        
        try {
            val modelFile = loadModelFile(context, assetPath)
            val interp = Interpreter(modelFile, options)
            interp.allocateTensors()
            isReady = true
            interp
        } catch (e: Exception) {
            Log.e("TFLite_INIT", "Critical error creating Interpreter: ${e.message}", e)
            throw RuntimeException("Failed to load model $assetPath", e)
        }
    }

    override fun run(inputs: ModelInputs): Float {
        // Accessing the interpreter triggers the lazy initialization
        val interp = try {
            interpreter
        } catch (e: Exception) {
            Log.e("TFLite_RUN", "Interpreter initialization failed", e)
            return 0f
        }

        if (inputs.inputs.isEmpty()) {
            Log.e("TFLite_RUN", "No tensor inputs provided")
            return 0f
        }

        val sortedInputs = inputs.inputs.sortedBy { it.index }
        val inputArray = sortedInputs.map { it.data }.toTypedArray()
        val outputBuffer = Array(1) { FloatArray(1) }
        
        interp.runForMultipleInputsOutputs(inputArray, mapOf(0 to outputBuffer))
        
        val score = outputBuffer[0][0]
        ModelInputDumpManager.dumpRun(assetPath, inputs, score)
        return score
    }

    fun getInputNames(): List<String> {
        return (0 until interpreter.inputTensorCount).map {
            interpreter.getInputTensor(it).name()
        }
    }

    fun getInputShapes(): List<IntArray> {
        return (0 until interpreter.inputTensorCount).map {
            interpreter.getInputTensor(it).shape()
        }
    }

    override fun close() {
        interpreter.close()
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context, assetPath: String): File {
        val modelFile = File(context.filesDir, assetPath.substringAfterLast("/"))
        context.assets.open(assetPath).use { input ->
            FileOutputStream(modelFile).use { output -> 
                input.copyTo(output)
                output.fd.sync()
            }
        }
        return modelFile
    }
}
