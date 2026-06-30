package dev.distraction.demo.location

import android.location.Location
import android.os.Build
import java.io.BufferedWriter

class LocationCsvWriter(
    private val writerProvider: () -> BufferedWriter?
) {
    fun writeHeader() {
        writeLine(
            "epoch_s,type,lat,lon,accuracy_m,alt_m,msl_alt_m," +
                    "speed_mps,speed_acc_mps,bearing_deg,bearing_acc_deg," +
                    "vertical_acc_m,provider,time_ms,elapsed_realtime_ns,is_mock"
        )
    }

    fun writeLocation(
        epochS: String,
        loc: Location
    ) {
        val accuracy = if (loc.hasAccuracy()) loc.accuracy else null
        val altitude = if (loc.hasAltitude()) loc.altitude else null

        val mslAltitude =
            if (Build.VERSION.SDK_INT >= 34 && loc.hasMslAltitude()) {
                loc.mslAltitudeMeters
            } else {
                null
            }

        val speed = if (loc.hasSpeed()) loc.speed else null
        val bearing = if (loc.hasBearing()) loc.bearing else null

        val speedAcc =
            if (Build.VERSION.SDK_INT >= 31 && loc.hasSpeedAccuracy()) {
                loc.speedAccuracyMetersPerSecond
            } else {
                null
            }

        val bearingAcc =
            if (Build.VERSION.SDK_INT >= 31 && loc.hasBearingAccuracy()) {
                loc.bearingAccuracyDegrees
            } else {
                null
            }

        val verticalAcc =
            if (Build.VERSION.SDK_INT >= 26 && loc.hasVerticalAccuracy()) {
                loc.verticalAccuracyMeters
            } else {
                null
            }

        writeLine(
            listOf(
                epochS,
                "LOC",
                loc.latitude,
                loc.longitude,
                csvOpt(accuracy),
                csvOpt(altitude),
                csvOpt(mslAltitude),
                csvOpt(speed),
                csvOpt(speedAcc),
                csvOpt(bearing),
                csvOpt(bearingAcc),
                csvOpt(verticalAcc),
                loc.provider ?: "",
                loc.time,
                loc.elapsedRealtimeNanos,
                loc.isFromMockProvider
            ).joinToString(",")
        )
    }

    fun writeError(epochS: String) {
        writeLine(
            listOf(
                epochS,
                "ERR",
                "", "", "", "", "",
                "", "", "", "",
                "", "", "", "",
                ""
            ).joinToString(",")
        )
    }

    fun writeLine(line: String) {
        writerProvider()?.apply {
            write(line)
            newLine()
        }
    }

    private fun csvOpt(value: Any?): String = value?.toString() ?: ""
}