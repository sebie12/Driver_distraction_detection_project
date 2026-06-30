package dev.distraction.demo.ml.models

import android.content.Context
import android.util.Log
import dev.distraction.demo.ml.api.*
import dev.distraction.demo.ml.core.BinaryThresholdPostProcessor
import dev.distraction.demo.ml.core.ModelMetadata
import dev.distraction.demo.ml.core.StandardInferencePipeline
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

abstract class TreeModelDefinition : InferenceModelDefinition {
    abstract val featureNames: List<String>
    abstract val windowDurationMillis: Long
    abstract val threshold: Float
    abstract override val metadataAssetPath: String
    abstract override val tfliteAssetPath: String

    override val category: ModelCategory = ModelCategory.TREE

    override fun createPipeline(context: Context, isTrainingEnabled: Boolean, strideMillis: Long): InferencePipeline {
        val finalStrideMillis = if (strideMillis > 0) strideMillis else windowDurationMillis

        return StandardInferencePipeline(
            metadata = ModelMetadata(
                id = id,
                displayName = displayName,
                version = version,
                description = "Tree-based model ($id) with window ${windowDurationMillis / 1000.0}s and stride ${finalStrideMillis / 1000.0}s",
                requiresSensors = true,
                requiresLocation = true,
                requiresDeviceState = true,
                requiresActivity = false,
                windowDurationMillis = windowDurationMillis,
                supportsRealtime = true,
                optimalThreshold = threshold
            ),
            rawDataProcessor = TreeRawDataProcessor(
                windowDurationMillis = windowDurationMillis,
                strideMillis = finalStrideMillis,
                featureNames = featureNames
            ),
            modelRunner = TreeModelRunner(context, tfliteAssetPath, featureNames),
            postProcessor = BinaryThresholdPostProcessor()
        )
    }
}

class TreeModelRunner(
    private val context: Context,
    private val assetPath: String,
    private val featureNames: List<String>
) : ModelRunner {
    private var isReady = false
    
    private val interpreter: Interpreter by lazy {
        Log.d("TreeModelRunner", "Initializing interpreter for $assetPath")
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        try {
            val modelFile = loadModelFile(context, assetPath)
            val interp = Interpreter(modelFile, options)
            interp.allocateTensors()
            isReady = true
            interp
        } catch (e: Exception) {
            Log.e("TreeModelRunner", "Critical error creating Interpreter: ${e.message}", e)
            throw RuntimeException("Failed to load tree model $assetPath", e)
        }
    }

    override fun run(inputs: ModelInputs): Float {
        if (!isReady) {
            Log.e("TreeModelRunner", "Interpreter not ready")
            return 0f
        }
        
        val mapInput = inputs.mapInput
        if (mapInput == null) {
            Log.e("TreeModelRunner", "No mapInput provided to tree model runner")
            return 0f
        }
        
        // Tree models expect a float array with features in exact order defined by featureNames
        val featureArray = FloatArray(featureNames.size)
        for (i in featureNames.indices) {
            featureArray[i] = mapInput.features[featureNames[i]] ?: 0f
        }
        
        // Model expects input shape [1, num_features]
        val inputArray = arrayOf(featureArray)
        
        // Model output shape [1, 1] for probability
        val outputBuffer = Array(1) { FloatArray(1) }
        
        interpreter.runForMultipleInputsOutputs(arrayOf(inputArray), mapOf(0 to outputBuffer))
        
        return outputBuffer[0][0]
    }

    override fun close() {
        if (isReady) {
            interpreter.close()
            isReady = false
        }
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
