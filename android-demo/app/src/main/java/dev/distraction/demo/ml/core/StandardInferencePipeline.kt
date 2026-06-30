package dev.distraction.demo.ml.core

import dev.distraction.demo.ml.api.InferencePipeline
import dev.distraction.demo.ml.api.InferenceResult
import dev.distraction.demo.ml.api.ModelRunner
import dev.distraction.demo.ml.api.PostProcessor
import dev.distraction.demo.ml.api.RawDataProcessor
import dev.distraction.demo.ml.types.RawEvent

class StandardInferencePipeline(
    override val metadata: ModelMetadata,
    private val rawDataProcessor: RawDataProcessor,
    private val modelRunner: ModelRunner,
    private val postProcessor: PostProcessor
) : InferencePipeline {

    override fun onTripStarted(tripId: String?, startTimestampMillis: Long) {
        rawDataProcessor.onTripStarted(tripId, startTimestampMillis)
    }

    override fun onTripEnded(tripId: String?, endTimestampMillis: Long) {
        rawDataProcessor.onTripEnded(tripId, endTimestampMillis)
    }

    override fun onRawEvent(event: RawEvent) {
        rawDataProcessor.addEvent(event)
    }

    override fun canInfer(): Boolean {
        return rawDataProcessor.isWindowReady()
    }

    override fun runInferenceIfReady(): InferenceResult? {
        if (!canInfer()) return null

        val windowedInputs = rawDataProcessor.buildInputs() ?: return null
        val score = modelRunner.run(windowedInputs.modelInputs)

        return postProcessor.process(
            score = score,
            windowStartMillis = windowedInputs.windowStartMillis,
            windowEndMillis = windowedInputs.windowEndMillis,
            metadata = metadata,
            sensorSamplesCount = windowedInputs.sensorSamplesCount,
            locationSamplesCount = windowedInputs.locationSamplesCount,
            deviceSamplesCount = windowedInputs.deviceSamplesCount,
            modelInputs = windowedInputs.modelInputs
        )
    }

    override fun reset() {
        rawDataProcessor.reset()
    }
}