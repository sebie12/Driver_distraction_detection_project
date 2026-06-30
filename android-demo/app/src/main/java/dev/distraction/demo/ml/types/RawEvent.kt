package dev.distraction.demo.ml.types

sealed class RawEvent {
    abstract val timestampMillis: Long

    data class SensorEvent(
        override val timestampMillis: Long,
        val sensorType: String,
        val values: FloatArray
    ) : RawEvent()

    data class LocationEvent(
        override val timestampMillis: Long,
        val latitude: Double,
        val longitude: Double,
        val speedMps: Float?,
        val bearingDeg: Float?,
        val accuracyM: Float?,
        val altitudeM: Double? = null
    ) : RawEvent()

    data class DeviceStateEvent(
        override val timestampMillis: Long,
        val screenInteractive: Boolean?,
        val deviceLocked: Boolean?,
        val audioActive: Boolean?,
        val audioOutput: String? = null,
        val batteryPct: Int? = null,
        val charging: Boolean? = null
    ) : RawEvent()

    data class ActivityEvent(
        override val timestampMillis: Long,
        val activityType: String,
        val confidence: Int?
    ) : RawEvent()

    data class MarkerEvent(
        override val timestampMillis: Long,
        val name: String,
        val value: String? = null
    ) : RawEvent()
}