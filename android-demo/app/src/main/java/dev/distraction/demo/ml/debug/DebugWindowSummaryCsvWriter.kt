package dev.distraction.demo.ml.debug

import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.util.Locale

class DebugWindowSummaryCsvWriter {

    private var csvFile: File? = null
    private var writer: FileWriter? = null

    fun open(sessionFolder: File) {
        close()

        if (!sessionFolder.exists()) {
            sessionFolder.mkdirs()
        }

        csvFile = File(sessionFolder, FILE_NAME)
        val exists = csvFile!!.exists()

        writer = FileWriter(csvFile, true)

        if (!exists || csvFile!!.length() == 0L) {
            writer?.append(
                "window_start_ms,window_end_ms,window_duration_ms,model_id,model_version,score,predicted_class,threshold,sensor_samples_count,location_samples_count,device_samples_count,auto_event_emitted\n"
            )
            writer?.flush()
        }
    }

    fun write(
        windowStartMillis: Long,
        windowEndMillis: Long,
        modelId: String,
        modelVersion: String,
        score: Float,
        predictedClass: Int,
        threshold: Float?,
        sensorSamplesCount: Int,
        locationSamplesCount: Int,
        deviceSamplesCount: Int,
        autoEventEmitted: Boolean
    ) {
        val safeWriter = writer ?: return

        val line = buildString {
            append(windowStartMillis)
            append(',')
            append(windowEndMillis)
            append(',')
            append(windowEndMillis - windowStartMillis)
            append(',')
            append(escape(modelId))
            append(',')
            append(escape(modelVersion))
            append(',')
            append(String.format(Locale.US, "%.6f", score))
            append(',')
            append(predictedClass)
            append(',')
            append(threshold?.let { String.format(Locale.US, "%.6f", it) } ?: "")
            append(',')
            append(sensorSamplesCount)
            append(',')
            append(locationSamplesCount)
            append(',')
            append(deviceSamplesCount)
            append(',')
            append(if (autoEventEmitted) 1 else 0)
            append('\n')
        }

        try {
            safeWriter.append(line)
            safeWriter.flush()
        } catch (_: IOException) {
        }
    }

    fun close() {
        try {
            writer?.flush()
        } catch (_: IOException) {
        }

        try {
            writer?.close()
        } catch (_: IOException) {
        }

        writer = null
        csvFile = null
    }

    private fun escape(value: String): String {
        val escaped = value.replace("\"", "\"\"")
        return "\"$escaped\""
    }

    companion object {
        private const val FILE_NAME = "debug_window_summary.csv"
    }
}