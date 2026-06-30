package dev.distraction.demo.ml.models

import android.content.Context
import dev.distraction.demo.ml.api.ModelCategory
import dev.distraction.demo.ml.api.InferenceModelDefinition
import dev.distraction.demo.ml.api.InferencePipeline
import dev.distraction.demo.ml.core.BinaryThresholdPostProcessor
import dev.distraction.demo.ml.core.ModelMetadata
import dev.distraction.demo.ml.core.StandardInferencePipeline
import dev.distraction.demo.ml.tflite.TFLiteMultiInputRunner

data class ModelInputConfig(
    val inputName: String,
    val featureDefs: List<FeatureDef>,
    val nTimesteps: Int,
    val layout: TensorLayout,
    val inputIndex: Int
)

class DynamicTfliteModelDefinition(
    override val id: String,
    override val displayName: String,
    override val version: String,
    override val category: ModelCategory,
    override val tfliteAssetPath: String,
    override val metadataAssetPath: String,
    val windowDurationMillis: Long,
    val threshold: Float,
    private val inputConfigs: List<ModelInputConfig>
) : InferenceModelDefinition {

    override fun createPipeline(context: Context, isTrainingEnabled: Boolean, strideMillis: Long): InferencePipeline {
        val runner = TFLiteMultiInputRunner(context, tfliteAssetPath)
        val finalStrideMillis = windowDurationMillis / 2

        return StandardInferencePipeline(
            metadata = ModelMetadata(
                id = id,
                displayName = displayName,
                version = version,
                description = "Dynamic TFLite model ($id) window ${windowDurationMillis / 1000.0}s stride ${finalStrideMillis / 1000.0}s",
                requiresSensors = inputConfigs.any { it.featureDefs.any { f -> f.sourceType in listOf("LINACC", "GYRO", "ACC", "GAMEROT") } },
                requiresLocation = inputConfigs.any { it.featureDefs.any { f -> f.sourceType == "LOC" } },
                requiresDeviceState = inputConfigs.any { it.featureDefs.any { f -> f.sourceType == "DEVICE" } },
                requiresActivity = false,
                windowDurationMillis = windowDurationMillis,
                supportsRealtime = true,
                optimalThreshold = threshold
            ),
            rawDataProcessor = SlidingWindowRawDataProcessor(
                windowDurationMillis = windowDurationMillis,
                strideMillis = finalStrideMillis,
                inputConfigs = inputConfigs,
                isDenoised = category == ModelCategory.DENOISED
            ),
            modelRunner = runner,
            postProcessor = BinaryThresholdPostProcessor()
        )
    }
}