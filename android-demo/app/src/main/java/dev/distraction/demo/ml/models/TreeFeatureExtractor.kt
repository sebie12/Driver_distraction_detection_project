package dev.distraction.demo.ml.models

import dev.distraction.demo.ml.types.RawEvent
import kotlin.math.sqrt

object TreeFeatureExtractor {

    fun extractFeatures(
        windowEvents: List<RawEvent>,
        featureNames: List<String>
    ): Map<String, Float> {
        val features = mutableMapOf<String, Float>()

        val linaccEvents = windowEvents.filterIsInstance<RawEvent.SensorEvent>()
            .filter { it.sensorType == "LINACC" }
        val gyroEvents = windowEvents.filterIsInstance<RawEvent.SensorEvent>()
            .filter { it.sensorType == "GYRO" }
        val gameRotEvents = windowEvents.filterIsInstance<RawEvent.SensorEvent>()
            .filter { it.sensorType == "GAMEROT" }
        val locEvents = windowEvents.filterIsInstance<RawEvent.LocationEvent>()
        val deviceEvents = windowEvents.filterIsInstance<RawEvent.DeviceStateEvent>()

        // LINACC features
        extractSensorStats("LINACC", linaccEvents, features)
        
        // GYRO features
        extractSensorStats("GYRO", gyroEvents, features)

        // GAMEROT features
        if (gameRotEvents.isNotEmpty()) {
            val angularDistances = mutableListOf<Float>()
            for (i in 1 until gameRotEvents.size) {
                val q1 = gameRotEvents[i - 1].values
                val q2 = gameRotEvents[i].values
                angularDistances.add(calculateAngularDistance(q1, q2))
            }
            if (angularDistances.isNotEmpty()) {
                features["GAMEROT_angular_distance_mean"] = angularDistances.mean()
                features["GAMEROT_angular_distance_max"] = angularDistances.maxOrNull() ?: 0f
            }
        }

        // Location features
        if (locEvents.size >= 2) {
            val accelerations = mutableListOf<Float>()
            for (i in 1 until locEvents.size) {
                val prev = locEvents[i - 1]
                val curr = locEvents[i]
                val dt = (curr.timestampMillis - prev.timestampMillis) / 1000f
                val v1 = prev.speedMps
                val v2 = curr.speedMps
                if (v1 != null && v2 != null && dt > 0f) {
                    accelerations.add((v2 - v1) / dt)
                }
            }
            if (accelerations.isNotEmpty()) {
                features["loc_instant_acceleration_mean"] = accelerations.mean()
                features["loc_instant_acceleration_min"] = accelerations.minOrNull() ?: 0f
                features["loc_instant_acceleration_max"] = accelerations.maxOrNull() ?: 0f
            }
        }

        // Device features
        if (deviceEvents.isNotEmpty()) {
            val totalWindow = (windowEvents.lastOrNull()?.timestampMillis ?: 0L) - 
                              (windowEvents.firstOrNull()?.timestampMillis ?: 0L)
            if (totalWindow > 0) {
                var lockedTime = 0L
                var audioTime = 0L
                
                // Simplified fraction calculation based on samples
                val lockedFrac = deviceEvents.count { it.deviceLocked == true }.toFloat() / deviceEvents.size
                val audioFrac = deviceEvents.count { it.audioActive == true }.toFloat() / deviceEvents.size
                
                features["ds_device_locked_frac"] = lockedFrac
                features["ds_audio_active_frac"] = audioFrac
            }
        }

        // Ensure all requested features exist
        featureNames.forEach { name ->
            if (!features.containsKey(name)) {
                features[name] = 0f
            }
        }

        return features
    }

    private fun extractSensorStats(prefix: String, events: List<RawEvent.SensorEvent>, out: MutableMap<String, Float>) {
        if (events.isEmpty()) return

        val v1 = events.map { it.values.getOrElse(0) { 0f } }
        val v2 = events.map { it.values.getOrElse(1) { 0f } }
        val v3 = events.map { it.values.getOrElse(2) { 0f } }
        val mags = events.map { e ->
            val x = e.values.getOrElse(0) { 0f }
            val y = e.values.getOrElse(1) { 0f }
            val z = e.values.getOrElse(2) { 0f }
            sqrt(x * x + y * y + z * z)
        }

        out["${prefix}_v1_mean"] = v1.mean()
        out["${prefix}_v1_min"] = v1.minOrNull() ?: 0f
        out["${prefix}_v1_max"] = v1.maxOrNull() ?: 0f
        
        out["${prefix}_v2_mean"] = v2.mean()
        out["${prefix}_v2_min"] = v2.minOrNull() ?: 0f
        out["${prefix}_v2_max"] = v2.maxOrNull() ?: 0f
        
        out["${prefix}_v3_mean"] = v3.mean()
        out["${prefix}_v3_min"] = v3.minOrNull() ?: 0f
        out["${prefix}_v3_max"] = v3.maxOrNull() ?: 0f

        out["${prefix}_mag_mean"] = mags.mean()
        out["${prefix}_mag_min"] = mags.minOrNull() ?: 0f
        out["${prefix}_mag_max"] = mags.maxOrNull() ?: 0f

        // Plane Energy
        val sqsum_xy = events.map { e ->
            val x = e.values.getOrElse(0) { 0f }
            val y = e.values.getOrElse(1) { 0f }
            x * x + y * y
        }
        val sqsum_yz = events.map { e ->
            val y = e.values.getOrElse(1) { 0f }
            val z = e.values.getOrElse(2) { 0f }
            y * y + z * z
        }
        out["${prefix}_sqsum_xy_mean"] = sqsum_xy.mean()
        out["${prefix}_sqsum_yz_mean"] = sqsum_yz.mean()

        // Orientation Bias
        val xz_y = events.map { e ->
            val x = e.values.getOrElse(0) { 0f }
            val y = e.values.getOrElse(1) { 0f }
            val z = e.values.getOrElse(2) { 0f }
            (x + z) / 2.0f - y
        }
        out["${prefix}_xz_y_mean"] = xz_y.mean()

        // sumsq75
        val allValues = mutableListOf<Float>()
        events.forEach { e ->
            e.values.forEach { allValues.add(it) }
        }
        if (allValues.isNotEmpty()) {
            val sorted = allValues.sorted()
            val index75 = (sorted.size * 0.75).toInt().coerceAtMost(sorted.size - 1)
            val p75 = sorted[index75]
            out["${prefix}_sumsq75"] = allValues.filter { it <= p75 }.sumOf { (it * it).toDouble() }.toFloat()
        }
    }

    private fun List<Float>.mean(): Float = if (isEmpty()) 0f else sum() / size

    private fun calculateAngularDistance(q1: FloatArray, q2: FloatArray): Float {
        val q1_0 = q1.getOrElse(0){0f}
        val q1_1 = q1.getOrElse(1){0f}
        val q1_2 = q1.getOrElse(2){0f}
        val q1_3 = q1.getOrElse(3){1f}
        
        val q2_0 = q2.getOrElse(0){0f}
        val q2_1 = q2.getOrElse(1){0f}
        val q2_2 = q2.getOrElse(2){0f}
        val q2_3 = q2.getOrElse(3){1f}

        var norm1 = kotlin.math.sqrt(q1_0*q1_0 + q1_1*q1_1 + q1_2*q1_2 + q1_3*q1_3)
        if (norm1 < 1e-10f) norm1 = 1.0f
        
        var norm2 = kotlin.math.sqrt(q2_0*q2_0 + q2_1*q2_1 + q2_2*q2_2 + q2_3*q2_3)
        if (norm2 < 1e-10f) norm2 = 1.0f

        val dotProduct = ((q1_0/norm1) * (q2_0/norm2)) + 
                         ((q1_1/norm1) * (q2_1/norm2)) + 
                         ((q1_2/norm1) * (q2_2/norm2)) + 
                         ((q1_3/norm1) * (q2_3/norm2))
                         
        val absDot = kotlin.math.abs(dotProduct)
        val clampedDot = absDot.coerceIn(0.0f, 1.0f)
        return 2.0f * kotlin.math.acos(clampedDot)
    }
}
