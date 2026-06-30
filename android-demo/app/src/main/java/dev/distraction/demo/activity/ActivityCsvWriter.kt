package dev.distraction.demo.activity

import java.io.BufferedWriter

class ActivityCsvWriter(
    private val writerProvider: () -> BufferedWriter?
) {
    fun writeHeader() {
        writeLine("epoch_s,kind,activity,confidence,transition,elapsed_realtime_ns")
    }

    fun writeCurrent(epochS: String, activity: String, confidence: Int, elapsedNs: Long) {
        writeLine(
            listOf(
                epochS,
                "CURR",
                activity,
                confidence,
                "",
                elapsedNs
            ).joinToString(",")
        )
    }

    fun writeTransition(epochS: String, activity: String, transition: String, elapsedNs: Long) {
        writeLine(
            listOf(
                epochS,
                "TRANS",
                activity,
                "",
                transition,
                elapsedNs
            ).joinToString(",")
        )
    }

    fun writeInfo(epochS: String, info: String, elapsedNs: Long) {
        writeLine("$epochS,INFO,$info,,,${elapsedNs}")
    }

    fun writeError(epochS: String, error: String, elapsedNs: Long) {
        writeLine("$epochS,ERR,$error,,,${elapsedNs}")
    }

    fun writeLine(line: String) {
        writerProvider()?.apply {
            write(line)
            newLine()
        }
    }
}