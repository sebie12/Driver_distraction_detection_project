package dev.distraction.demo.ml.tflite

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class TensorInfoUi(
    val index: Int,
    val name: String,
    val shape: IntArray,
    val dataType: String
)

data class ModelInspectionResult(
    val assetPath: String,
    val inputTensors: List<TensorInfoUi>,
    val outputTensors: List<TensorInfoUi>
) {
    fun toPrettyString(): String {
        return buildString {
            appendLine("Modelo: $assetPath")
            appendLine()

            appendLine("Inputs (${inputTensors.size}):")
            if (inputTensors.isEmpty()) {
                appendLine("- nenhum -")
            } else {
                inputTensors.forEach { tensor ->
                    appendLine(
                        "[${tensor.index}] ${tensor.name} | shape=${tensor.shape.contentToString()} | type=${tensor.dataType}"
                    )
                }
            }

            appendLine()
            appendLine("Outputs (${outputTensors.size}):")
            if (outputTensors.isEmpty()) {
                appendLine("- nenhum -")
            } else {
                outputTensors.forEach { tensor ->
                    appendLine(
                        "[${tensor.index}] ${tensor.name} | shape=${tensor.shape.contentToString()} | type=${tensor.dataType}"
                    )
                }
            }
        }
    }
}

object ModelInspector {

    fun inspect(context: Context, assetPath: String): ModelInspectionResult {
        val options = Interpreter.Options()
        try {
            val flexDelegate = org.tensorflow.lite.flex.FlexDelegate()
            options.addDelegate(flexDelegate)
        } catch (_: Exception) {}

        val interpreter = try {
            Interpreter(loadModelFile(context, assetPath), options)
        } catch (e: Exception) {
            throw RuntimeException("Falha ao abrir modelo para inspeção: ${e.message}", e)
        }

        try {
            val inputTensors = (0 until interpreter.inputTensorCount).map { index ->
                interpreter.getInputTensor(index).toTensorInfoUi(index)
            }

            val outputTensors = (0 until interpreter.outputTensorCount).map { index ->
                interpreter.getOutputTensor(index).toTensorInfoUi(index)
            }

            return ModelInspectionResult(
                assetPath = assetPath,
                inputTensors = inputTensors,
                outputTensors = outputTensors
            )
        } finally {
            interpreter.close()
        }
    }

    private fun Tensor.toTensorInfoUi(index: Int): TensorInfoUi {
        return TensorInfoUi(
            index = index,
            name = name(),
            shape = shape(),
            dataType = dataType().toString()
        )
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context, assetPath: String): java.io.File {
        val modelFile = java.io.File(context.filesDir, assetPath.substringAfterLast("/"))
        
        // Copy the main .tflite model
        if (!modelFile.exists()) {
            context.assets.open(assetPath).use { inputStream ->
                java.io.FileOutputStream(modelFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        
        // Attempt to copy companion checkpoint files if they exist in assets
        val baseName = modelFile.name.removeSuffix(".tflite")
        val indexFile = java.io.File(context.filesDir, "$baseName.index")
        val dataFile = java.io.File(context.filesDir, "$baseName.data-00000-of-00001")
        
        val assetBaseName = assetPath.removeSuffix(".tflite")
        val assetIndex = "$assetBaseName.index"
        val assetData = "$assetBaseName.data-00000-of-00001"
        
        try {
            if (!indexFile.exists()) {
                context.assets.open(assetIndex).use { inputStream ->
                    java.io.FileOutputStream(indexFile).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            }
        } catch (e: Exception) {
            // Might not exist
        }
        
        try {
            if (!dataFile.exists()) {
                context.assets.open(assetData).use { inputStream ->
                    java.io.FileOutputStream(dataFile).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            }
        } catch (e: Exception) {
            // Might not exist
        }
        
        return modelFile
    }
}