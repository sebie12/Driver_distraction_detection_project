package dev.distraction.demo.ml.api

data class InferenceResult(
    val modelId: String,
    val modelVersion: String,
    val windowStartMillis: Long,
    val windowEndMillis: Long,
    val score: Float,
    val predictedClass: Int,
    val threshold: Float? = null,
    val extras: Map<String, String> = emptyMap(),
    val sensorSamplesCount: Int = 0,
    val locationSamplesCount: Int = 0,
    val deviceSamplesCount: Int = 0,
    val modelInputs: ModelInputs? = null
)