package dev.distraction.demo.ml.models

import dev.distraction.demo.ml.api.ModelInputs
import dev.distraction.demo.ml.api.RawDataProcessor
import dev.distraction.demo.ml.api.TensorInput
import dev.distraction.demo.ml.api.WindowedModelInputs
import dev.distraction.demo.ml.types.RawEvent
import java.util.ArrayDeque

class SlidingWindowRawDataProcessor(
    private val windowDurationMillis: Long,
    private val strideMillis: Long,
    private val inputConfigs: List<ModelInputConfig>,
    private val isDenoised: Boolean = false
) : RawDataProcessor {

    private val eventBuffer = ArrayDeque<RawEvent>()
    private val readyWindows = ArrayDeque<WindowedModelInputs>()

    private var currentWindowStartMillis: Long? = null

    override fun onTripStarted(tripId: String?, startTimestampMillis: Long) {
        reset()
        currentWindowStartMillis = startTimestampMillis
    }

    override fun onTripEnded(tripId: String?, endTimestampMillis: Long) = Unit

    override fun addEvent(event: RawEvent) {
        eventBuffer.addLast(event)

        if (currentWindowStartMillis == null) {
            currentWindowStartMillis = event.timestampMillis
        }

        createReadyWindowsIfPossible(event.timestampMillis)
        trimOldEvents()
    }

    override fun isWindowReady(): Boolean = readyWindows.isNotEmpty()

    override fun buildInputs(): WindowedModelInputs? {
        return if (readyWindows.isEmpty()) null else readyWindows.removeFirst()
    }

    override fun reset() {
        eventBuffer.clear()
        readyWindows.clear()
        currentWindowStartMillis = null
    }

    private fun createReadyWindowsIfPossible(latestTimestampMillis: Long) {
        var windowStart = currentWindowStartMillis ?: return

        while (windowStart + windowDurationMillis <= latestTimestampMillis) {
            val windowEnd = windowStart + windowDurationMillis

            val windowEvents = eventBuffer.filter {
                it.timestampMillis >= windowStart && it.timestampMillis < windowEnd
            }

            val sensorSamplesCount = windowEvents.count { it is RawEvent.SensorEvent }
            val locationSamplesCount = windowEvents.count { it is RawEvent.LocationEvent }
            val deviceSamplesCount = windowEvents.count { it is RawEvent.DeviceStateEvent }

            val tensorInputs = inputConfigs.map { config ->
                val tensor = SequenceExtractionUtils.buildTensor(
                    windowEvents = windowEvents,
                    featureList = config.featureDefs,
                    nTimesteps = config.nTimesteps,
                    layout = config.layout,
                    isDenoised = isDenoised
                )
                TensorInput(
                    index = config.inputIndex,
                    name = config.inputName,
                    data = tensor
                )
            }

            readyWindows.addLast(
                WindowedModelInputs(
                    modelInputs = ModelInputs(
                        inputs = tensorInputs
                    ),
                    windowStartMillis = windowStart,
                    windowEndMillis = windowEnd,
                    sensorSamplesCount = sensorSamplesCount,
                    locationSamplesCount = locationSamplesCount,
                    deviceSamplesCount = deviceSamplesCount
                )
            )

            windowStart += strideMillis
            currentWindowStartMillis = windowStart
        }
    }

    private fun trimOldEvents() {
        val minTsToKeep = (currentWindowStartMillis ?: return) - windowDurationMillis
        while (eventBuffer.isNotEmpty() && eventBuffer.first().timestampMillis < minTsToKeep) {
            eventBuffer.removeFirst()
        }
    }
}