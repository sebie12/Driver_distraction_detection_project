package dev.distraction.demo.sensors

import android.hardware.Sensor
import android.hardware.SensorEvent
import java.io.BufferedWriter

class SensorCsvWriter(
    private val writerProvider: () -> BufferedWriter?
) {
    fun writeHeader() {
        writeLine("epoch_s,type,v1,v2,v3,v4")
    }

    fun writeSensor(epochS: String, event: SensorEvent) {
        val type = when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> "ACC"
            Sensor.TYPE_GYROSCOPE -> "GYRO"
            Sensor.TYPE_ROTATION_VECTOR -> "ROT"
            Sensor.TYPE_LINEAR_ACCELERATION -> "LINACC"
            Sensor.TYPE_GRAVITY -> "GRAV"
            Sensor.TYPE_MAGNETIC_FIELD -> "MAG"
            Sensor.TYPE_GAME_ROTATION_VECTOR -> "GAMEROT"
            else -> "SENS_${event.sensor.type}"
        }

        val v = event.values
        val v0 = v.getOrNull(0) ?: ""
        val v1 = v.getOrNull(1) ?: ""
        val v2 = v.getOrNull(2) ?: ""
        val v3 = v.getOrNull(3) ?: ""

        writeLine("$epochS,$type,$v0,$v1,$v2,$v3")
    }

    fun writeLine(line: String) {
        writerProvider()?.apply {
            write(line)
            newLine()
        }
    }
}