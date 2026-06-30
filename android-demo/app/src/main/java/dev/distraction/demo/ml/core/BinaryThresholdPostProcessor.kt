package dev.distraction.demo.ml.core

import dev.distraction.demo.ml.api.InferenceResult
import dev.distraction.demo.ml.api.PostProcessor

class BinaryThresholdPostProcessor : PostProcessor {

    override fun process(
        score: Float,
        windowStartMillis: Long,
        windowEndMillis: Long,
        metadata: ModelMetadata,
        sensorSamplesCount: Int,
        locationSamplesCount: Int,
        deviceSamplesCount: Int,
        modelInputs: dev.distraction.demo.ml.api.ModelInputs?
    ): InferenceResult {
        val threshold = metadata.optimalThreshold
        val predictedClass = if (score >= threshold) 1 else 0

        return InferenceResult(
            modelId = metadata.id,
            modelVersion = metadata.version,
            windowStartMillis = windowStartMillis,
            windowEndMillis = windowEndMillis,
            score = score,
            predictedClass = predictedClass,
            threshold = threshold,
            sensorSamplesCount = sensorSamplesCount,
            locationSamplesCount = locationSamplesCount,
            deviceSamplesCount = deviceSamplesCount,
            modelInputs = modelInputs
        )
    }
}