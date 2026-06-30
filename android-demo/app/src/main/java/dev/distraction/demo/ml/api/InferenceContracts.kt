package dev.distraction.demo.ml.api

import android.content.Context
import dev.distraction.demo.ml.core.ModelMetadata
import dev.distraction.demo.ml.types.RawEvent

interface InferenceModelDefinition {
    val id: String
    val displayName: String
    val version: String
    val category: ModelCategory
    val metadataAssetPath: String?
    val tfliteAssetPath: String?

    fun createPipeline(context: Context, isTrainingEnabled: Boolean = false, strideMillis: Long = -1L): InferencePipeline
}

interface InferencePipeline {
    val metadata: ModelMetadata

    fun onTripStarted(tripId: String?, startTimestampMillis: Long)
    fun onTripEnded(tripId: String?, endTimestampMillis: Long)

    fun onRawEvent(event: RawEvent)

    fun canInfer(): Boolean
    fun runInferenceIfReady(): InferenceResult?

    fun reset()
    fun train(inputs: ModelInputs, label: Float) {}
    fun save(checkpointPath: String) {}
    fun restore(checkpointPath: String) {}
}

interface RawDataProcessor {
    fun onTripStarted(tripId: String?, startTimestampMillis: Long)
    fun onTripEnded(tripId: String?, endTimestampMillis: Long)

    fun addEvent(event: RawEvent)

    fun isWindowReady(): Boolean
    fun buildInputs(): WindowedModelInputs?

    fun reset()
}

interface ModelRunner {
    fun run(inputs: ModelInputs): Float
    fun close()
}

interface PostProcessor {
    fun process(
        score: Float,
        windowStartMillis: Long,
        windowEndMillis: Long,
        metadata: ModelMetadata,
        sensorSamplesCount: Int,
        locationSamplesCount: Int,
        deviceSamplesCount: Int,
        modelInputs: ModelInputs? = null
    ): InferenceResult
}

data class WindowedModelInputs(
    val modelInputs: ModelInputs,
    val windowStartMillis: Long,
    val windowEndMillis: Long,
    val sensorSamplesCount: Int,
    val locationSamplesCount: Int,
    val deviceSamplesCount: Int
)