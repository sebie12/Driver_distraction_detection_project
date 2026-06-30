package dev.distraction.demo.events

import java.io.BufferedWriter

class EventCsvWriter(
    private val writerProvider: () -> BufferedWriter?
) {
    fun writeHeader() {
        writeLine("epoch_s,type")
    }

    fun writeEvent(epochS: String, type: String) {
        writeLine("$epochS,$type")
    }

    fun writeLine(line: String) {
        writerProvider()?.apply {
            write(line)
            newLine()
        }
    }
}