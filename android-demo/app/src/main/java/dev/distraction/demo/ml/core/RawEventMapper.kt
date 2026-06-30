package dev.distraction.demo.ml.core

import dev.distraction.demo.ml.types.RawEvent

object RawEventMapper {

    fun sensor(
        timestampMillis: Long,
        sensorType: String,
        values: FloatArray
    ): RawEvent {
        return RawEvent.SensorEvent(
            timestampMillis = timestampMillis,
            sensorType = sensorType,
            values = values.copyOf()
        )
    }

    fun location(
        timestampMillis: Long,
        latitude: Double,
        longitude: Double,
        speedMps: Float?,
        bearingDeg: Float?,
        accuracyM: Float?,
        altitudeM: Double? = null
    ): RawEvent {
        return RawEvent.LocationEvent(
            timestampMillis = timestampMillis,
            latitude = latitude,
            longitude = longitude,
            speedMps = speedMps,
            bearingDeg = bearingDeg,
            accuracyM = accuracyM,
            altitudeM = altitudeM
        )
    }

    fun deviceState(
        timestampMillis: Long,
        screenInteractive: Boolean?,
        deviceLocked: Boolean?,
        audioActive: Boolean?,
        audioOutput: String? = null,
        batteryPct: Int? = null,
        charging: Boolean? = null
    ): RawEvent {
        return RawEvent.DeviceStateEvent(
            timestampMillis = timestampMillis,
            screenInteractive = screenInteractive,
            deviceLocked = deviceLocked,
            audioActive = audioActive,
            audioOutput = audioOutput,
            batteryPct = batteryPct,
            charging = charging
        )
    }

    fun activity(
        timestampMillis: Long,
        activityType: String,
        confidence: Int?
    ): RawEvent {
        return RawEvent.ActivityEvent(
            timestampMillis = timestampMillis,
            activityType = activityType,
            confidence = confidence
        )
    }

    fun marker(
        timestampMillis: Long,
        name: String,
        value: String? = null
    ): RawEvent {
        return RawEvent.MarkerEvent(
            timestampMillis = timestampMillis,
            name = name,
            value = value
        )
    }
}