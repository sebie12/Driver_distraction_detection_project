package dev.distraction.demo.common

import java.util.Locale

object TimeUtils {
    fun computeBaseEpochNs(): Long {
        val epochNowNs = System.currentTimeMillis() * 1_000_000L
        val elapsedNowNs = android.os.SystemClock.elapsedRealtimeNanos()
        return epochNowNs - elapsedNowNs
    }

    fun formatEpochSeconds(epochNs: Long): String {
        val seconds = epochNs / 1_000_000_000.0
        return String.format(Locale.US, "%.3f", seconds)
    }

    fun toEpochMillis(epochNs: Long): Long {
        return epochNs / 1_000_000L
    }
}