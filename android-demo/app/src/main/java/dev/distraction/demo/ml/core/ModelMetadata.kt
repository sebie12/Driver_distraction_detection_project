package dev.distraction.demo.ml.core

data class ModelMetadata(
    val id: String,
    val displayName: String,
    val version: String,
    val description: String,
    val requiresSensors: Boolean,
    val requiresLocation: Boolean,
    val requiresDeviceState: Boolean,
    val requiresActivity: Boolean,
    val windowDurationMillis: Long,
    val supportsRealtime: Boolean,
    val optimalThreshold: Float = 0.5f
)