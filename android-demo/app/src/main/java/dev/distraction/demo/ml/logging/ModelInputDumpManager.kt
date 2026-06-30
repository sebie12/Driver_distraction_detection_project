package dev.distraction.demo.ml.debug

import dev.distraction.demo.ml.api.ModelInputs
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger

object ModelInputDumpManager {

    @Volatile
    private var enabled: Boolean = false

    @Volatile
    private var sessionFolder: File? = null

    @Volatile
    private var modelId: String? = null

    private val windowCounter = AtomicInteger(0)

    fun configure(
        enabled: Boolean,
        sessionFolder: File?,
        modelId: String?
    ) {
        this.enabled = enabled
        this.sessionFolder = sessionFolder
        this.modelId = modelId
        windowCounter.set(0)

        if (enabled && sessionFolder != null) {
            File(sessionFolder, DUMPS_DIR_NAME).mkdirs()
        }
    }

    fun clear() {
        enabled = false
        sessionFolder = null
        modelId = null
        windowCounter.set(0)
    }

    fun dumpRun(
        assetPath: String,
        inputs: ModelInputs,
        outputScore: Float
    ) {
        if (!enabled) return

        val root = sessionFolder ?: return
        val dumpRoot = File(root, DUMPS_DIR_NAME).apply { mkdirs() }

        val windowIndex = windowCounter.incrementAndGet()
        val windowDir = File(dumpRoot, "window_${windowIndex.toString().padStart(6, '0')}")

        try {
            windowDir.mkdirs()

            writeMetadata(
                File(windowDir, "metadata.txt"),
                assetPath = assetPath,
                modelId = modelId,
                inputs = inputs,
                outputScore = outputScore
            )

            inputs.inputs
                .sortedBy { it.index }
                .forEach { tensor ->
                    val safeName = sanitizeFileName(tensor.name)
                    val outFile = File(
                        windowDir,
                        "input_${tensor.index}_${safeName}.csv"
                    )
                    writeTensorToFile(outFile, tensor.data)
                }

            val outputFile = File(windowDir, "output_0.csv")
            FileWriter(outputFile).use { fw ->
                fw.append(String.format(Locale.US, "%.8f\n", outputScore))
            }
        } catch (_: Exception) {
        }
    }

    private fun writeMetadata(
        file: File,
        assetPath: String,
        modelId: String?,
        inputs: ModelInputs,
        outputScore: Float
    ) {
        FileWriter(file).use { fw ->
            fw.appendLine("model_id=${modelId.orEmpty()}")
            fw.appendLine("asset_path=$assetPath")
            fw.appendLine("inputs_count=${inputs.inputs.size}")
            fw.appendLine("output_score=${String.format(Locale.US, "%.8f", outputScore)}")
            fw.appendLine()

            inputs.inputs.sortedBy { it.index }.forEach { tensor ->
                fw.appendLine("input_index=${tensor.index}")
                fw.appendLine("input_name=${tensor.name}")
                fw.appendLine("input_shape=${describeShape(tensor.data)}")
                fw.appendLine("input_type=${tensor.data::class.java.simpleName}")
                fw.appendLine()
            }
        }
    }

    private fun writeTensorToFile(file: File, tensor: Any) {
        when (tensor) {
            is Array<*> -> writeArrayTensor(file, tensor)
            is FloatArray -> writeFloatArray(file, tensor)
            else -> {
                FileWriter(file).use { fw ->
                    fw.append("unsupported_tensor_type,${tensor::class.java.name}\n")
                }
            }
        }
    }

    private fun writeArrayTensor(file: File, tensor: Array<*>) {
        if (tensor.isEmpty()) {
            FileWriter(file).use { fw ->
                fw.append("empty_tensor\n")
            }
            return
        }

        val first = tensor[0]

        when (first) {
            is Array<*> -> {
                if (first.isNotEmpty() && first[0] is FloatArray) {
                    @Suppress("UNCHECKED_CAST")
                    val rank3 = tensor as Array<Array<FloatArray>>
                    writeRank3FloatTensor(file, rank3)
                } else {
                    FileWriter(file).use { fw ->
                        fw.append("unsupported_nested_array\n")
                    }
                }
            }

            is FloatArray -> {
                @Suppress("UNCHECKED_CAST")
                val rank2 = tensor as Array<FloatArray>
                writeRank2FloatTensor(file, rank2)
            }

            else -> {
                FileWriter(file).use { fw ->
                    fw.append("unsupported_array_type\n")
                }
            }
        }
    }

    private fun writeRank3FloatTensor(file: File, tensor: Array<Array<FloatArray>>) {
        FileWriter(file).use { fw ->
            val batch = tensor.size
            val dim1 = if (tensor.isNotEmpty()) tensor[0].size else 0
            val dim2 = if (tensor.isNotEmpty() && tensor[0].isNotEmpty()) tensor[0][0].size else 0

            fw.appendLine("shape=[${batch},${dim1},${dim2}]")
            fw.appendLine("batch_index=0")
            fw.appendLine()

            val matrix = tensor[0]
            matrix.forEach { row ->
                fw.append(row.joinToString(",") { String.format(Locale.US, "%.8f", it) })
                fw.append('\n')
            }
        }
    }

    private fun writeRank2FloatTensor(file: File, tensor: Array<FloatArray>) {
        FileWriter(file).use { fw ->
            val dim1 = tensor.size
            val dim2 = if (tensor.isNotEmpty()) tensor[0].size else 0

            fw.appendLine("shape=[${dim1},${dim2}]")
            fw.appendLine()

            tensor.forEach { row ->
                fw.append(row.joinToString(",") { String.format(Locale.US, "%.8f", it) })
                fw.append('\n')
            }
        }
    }

    private fun writeFloatArray(file: File, tensor: FloatArray) {
        FileWriter(file).use { fw ->
            fw.appendLine("shape=[${tensor.size}]")
            fw.appendLine()
            fw.append(tensor.joinToString(",") { String.format(Locale.US, "%.8f", it) })
            fw.append('\n')
        }
    }

    private fun describeShape(tensor: Any): String {
        return try {
            when (tensor) {
                is Array<*> -> {
                    if (tensor.isEmpty()) return "[]"

                    val first = tensor[0]
                    when (first) {
                        is Array<*> -> {
                            if (first.isNotEmpty() && first[0] is FloatArray) {
                                val dim0 = tensor.size
                                val dim1 = first.size
                                val dim2 = (first[0] as FloatArray).size
                                "[$dim0,$dim1,$dim2]"
                            } else {
                                "[${tensor.size},?]"
                            }
                        }

                        is FloatArray -> {
                            "[${tensor.size},${first.size}]"
                        }

                        else -> "[${tensor.size}]"
                    }
                }

                is FloatArray -> "[${tensor.size}]"
                else -> tensor::class.java.simpleName
            }
        } catch (_: Exception) {
            "unknown"
        }
    }

    private fun sanitizeFileName(name: String): String {
        return name.replace(Regex("[^a-zA-Z0-9._-]"), "_")
    }

    private const val DUMPS_DIR_NAME = "model_input_dumps"
}