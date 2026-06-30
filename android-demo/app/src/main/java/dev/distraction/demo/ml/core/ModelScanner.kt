package dev.distraction.demo.ml.core

import android.content.Context
import android.util.Log
import dev.distraction.demo.ml.api.InferenceModelDefinition
import dev.distraction.demo.ml.api.ModelCategory
import dev.distraction.demo.ml.models.DynamicTfliteModelDefinition
import dev.distraction.demo.ml.models.FeatureDef
import dev.distraction.demo.ml.models.FeatureDefinitions
import dev.distraction.demo.ml.models.ModelInputConfig
import dev.distraction.demo.ml.models.TensorLayout
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream

object ModelScanner {
    fun scanModels(context: Context): List<InferenceModelDefinition> {
        val models = mutableListOf<InferenceModelDefinition>()
        val assetManager = context.assets

        // Scan denoised models. A folder qualifies as a model if it contains BOTH
        // metadata.json and model.tflite — the folder name is irrelevant. Anything
        // skipped or failed is logged so a present-but-broken model is visible in
        // logcat (tag: ModelScanner) instead of silently disappearing.
        try {
            val entries = assetManager.list("models_denoised") ?: emptyArray()
            for (dirName in entries) {
                val dirPath = "models_denoised/$dirName"
                val contents = try {
                    assetManager.list(dirPath) ?: emptyArray()
                } catch (e: Exception) {
                    emptyArray<String>()
                }

                val hasModel = contents.contains("metadata.json") && contents.contains("model.tflite")
                if (!hasModel) {
                    // Warn only for non-empty dirs (a stray file lists as empty) so we
                    // flag a real-but-incomplete model dir without spamming on junk.
                    if (contents.isNotEmpty()) {
                        Log.w("ModelScanner",
                            "Skipping '$dirPath': needs metadata.json + model.tflite (found: ${contents.joinToString()})")
                    }
                    continue
                }

                val metadataPath = "$dirPath/metadata.json"
                val tflitePath = "$dirPath/model.tflite"
                val modelDef = tryParseModelDef(context, metadataPath, tflitePath, ModelCategory.DENOISED, "denoised_$dirName")
                if (modelDef != null) {
                    models.add(modelDef)
                    Log.i("ModelScanner", "Loaded model '$dirName' -> ${modelDef.displayName}")
                } else {
                    Log.w("ModelScanner",
                        "Model files present in '$dirPath' but parsing failed — see the error logged above")
                }
            }
            Log.i("ModelScanner", "Scan complete: ${models.size} model(s) loaded from models_denoised")
        } catch (e: Exception) {
            Log.e("ModelScanner", "Error scanning denoised models", e)
        }

        return models
    }

    private fun tryParseModelDef(
        context: Context,
        metadataPath: String,
        tflitePath: String,
        category: ModelCategory,
        fallbackId: String
    ): DynamicTfliteModelDefinition? {
        return try {
            val jsonStr = context.assets.open(metadataPath).bufferedReader().use { it.readText() }
            val json = JSONObject(jsonStr)

            val id = json.optString("version", fallbackId)
            val windowSeconds = json.getInt("window_seconds")
            val threshold = json.getDouble("threshold").toFloat()
            val inputNamesArray = json.getJSONArray("input_names")
            val inputNames = List(inputNamesArray.length()) { inputNamesArray.getString(it) }

            // Inspect the TFLite model to get shapes
            val interpreter = getInterpreter(context, tflitePath)
            val tfliteInputCount = interpreter.inputTensorCount

            val inputConfigs = mutableListOf<ModelInputConfig>()
            for (i in 0 until tfliteInputCount) {
                val tensor = interpreter.getInputTensor(i)
                val tensorName = tensor.name()
                val shape = tensor.shape()

                // Find the logical input name by matching the tensor name against input_names
                val logicalName = inputNames.firstOrNull { tensorName.contains(it, ignoreCase = true) }
                    ?: inputNames.firstOrNull { tensorName.contains("sensor", ignoreCase = true) && it == "sensor_data" }
                    ?: inputNames.firstOrNull { tensorName.contains("device", ignoreCase = true) && it == "device_data" }
                    ?: inputNames.firstOrNull { tensorName.contains("gamerot", ignoreCase = true) && it == "gamerot_data" }
                    ?: inputNames.firstOrNull { tensorName.contains("speed", ignoreCase = true) && it == "speed_data" }
                    ?: inputNames.firstOrNull { tensorName.contains("accel", ignoreCase = true) && it == "accel_data" }
                    ?: continue

                val basePrefix = logicalName.substringBefore("_")

                // Determine which feature array keys belong to this logical input
                val candidateFeatureKeys = mutableListOf<String>()
                val featureKeys = listOf("sensor_features", "device_features", "position_features",
                    "speed_features", "accel_features", "gamerot_features")

                for (key in featureKeys) {
                    val prefix = key.substringBefore("_features")
                    val isMatch = if (basePrefix == "position") {
                        prefix == "position" || prefix == "speed" || prefix == "accel"
                    } else {
                        prefix == basePrefix
                    }
                    if (isMatch) candidateFeatureKeys.add(key)
                }

                val featureDefs = mutableListOf<FeatureDef>()
                for (key in candidateFeatureKeys) {
                    val array = json.optJSONArray(key)
                    if (array != null) {
                        for (j in 0 until array.length()) {
                            val fName = array.getString(j)
                            FeatureDefinitions.getFeatureDef(fName)?.let { featureDefs.add(it) }
                        }
                    }
                }

                if (featureDefs.isNotEmpty()) {
                    val tensorInfo = resolveTensorInfo(shape, featureDefs.size, tensorName, id)
                    inputConfigs.add(
                        ModelInputConfig(
                            inputName = tensorName,
                            featureDefs = featureDefs,
                            nTimesteps = tensorInfo.first,
                            layout = tensorInfo.second,
                            inputIndex = i
                        )
                    )
                }
            }
            interpreter.close()

            DynamicTfliteModelDefinition(
                id = "${category.name.lowercase()}_$id",
                displayName = "Model $id (${windowSeconds}s)",
                version = "1.0",
                category = category,
                tfliteAssetPath = tflitePath,
                metadataAssetPath = metadataPath,
                windowDurationMillis = windowSeconds * 1000L,
                threshold = threshold,
                inputConfigs = inputConfigs
            )
        } catch (e: Exception) {
            Log.e("ModelScanner", "Failed to parse $metadataPath: ${e.message}")
            null
        }
    }

    private fun getInterpreter(context: Context, assetPath: String): Interpreter {
        val file = File(context.cacheDir, "temp_model.tflite")
        context.assets.open(assetPath).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        return Interpreter(file)
    }

    private fun resolveTensorInfo(
        shape: IntArray,
        expectedFeatures: Int,
        label: String,
        modelId: String
    ): Pair<Int, TensorLayout> {
        require(shape.size >= 3) {
            "Input $label inválido no modelo $modelId: rank=${shape.size}"
        }

        val dim1 = shape[1]
        val dim2 = shape[2]

        return when {
            dim2 == expectedFeatures -> Pair(dim1, TensorLayout.TIMESTEPS_FEATURES)
            dim1 == expectedFeatures -> Pair(dim2, TensorLayout.FEATURES_TIMESTEPS)
            else -> throw IllegalArgumentException(
                "Input $label inválido no modelo $modelId: esperadas=$expectedFeatures features, shape=${shape.contentToString()}"
            )
        }
    }
}