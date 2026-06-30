package dev.distraction.demo.ml.models

import dev.distraction.demo.ml.types.RawEvent
import kotlin.math.sqrt
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sign
import kotlin.math.PI

object SequenceExtractionUtils {

    // Sensor sampling rate (Hz) used for spectral features; matches the Butterworth design.
    private const val SENSOR_FS = 16.67f

    fun buildTensor(
        windowEvents: List<RawEvent>,
        featureList: List<FeatureDef>,
        nTimesteps: Int,
        layout: TensorLayout,
        isDenoised: Boolean = false
    ): Any {
        val matrix = extractSequence(
            windowEvents = windowEvents,
            featureList = featureList,
            nTimesteps = nTimesteps,
            isDenoised = isDenoised
        )
        return wrapTensor(matrix, layout)
    }

    private fun wrapTensor(
        matrix: Array<FloatArray>,
        layout: TensorLayout
    ): Any {
        return when (layout) {
            TensorLayout.TIMESTEPS_FEATURES -> arrayOf(matrix)
            TensorLayout.FEATURES_TIMESTEPS -> arrayOf(transpose(matrix))
        }
    }

    private fun transpose(matrix: Array<FloatArray>): Array<FloatArray> {
        if (matrix.isEmpty()) return emptyArray()

        val rows = matrix.size
        val cols = matrix[0].size
        val out = Array(cols) { FloatArray(rows) }

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                out[c][r] = matrix[r][c]
            }
        }

        return out
    }

    private fun extractSequence(
        windowEvents: List<RawEvent>,
        featureList: List<FeatureDef>,
        nTimesteps: Int,
        isDenoised: Boolean
    ): Array<FloatArray> {
        val nFeatures = featureList.size
        val out = Array(nTimesteps) { FloatArray(nFeatures) }

        featureList.forEachIndexed { featureIndex, feature ->
            val colData = extractColumn(windowEvents, feature, isDenoised)

            if (colData.isEmpty()) return@forEachIndexed

            val resampled = if (colData.size == nTimesteps) {
                colData
            } else {
                interpolate(colData, nTimesteps)
            }

            for (t in 0 until nTimesteps) {
                out[t][featureIndex] = resampled[t]
            }
        }

        return out
    }

    private fun extractColumn(
        windowEvents: List<RawEvent>,
        feature: FeatureDef,
        isDenoised: Boolean
    ): FloatArray {
        return when (feature.sourceType) {
            "LOC" -> {
                val locs = windowEvents.asSequence()
                    .filterIsInstance<RawEvent.LocationEvent>()
                    .toList()
                    
                val instantAccels = if (locs.isNotEmpty()) {
                    val accels = FloatArray(locs.size)
                    accels[0] = 0f
                    for (i in 1 until locs.size) {
                        val prev = locs[i - 1]
                        val curr = locs[i]
                        val dt = (curr.timestampMillis - prev.timestampMillis) / 1000f
                        val v1 = prev.speedMps
                        val v2 = curr.speedMps
                        accels[i] = if (v1 != null && v2 != null && dt > 0f) (v2 - v1) / dt else 0f
                    }
                    accels
                } else {
                    FloatArray(0)
                }

                when (feature.column) {
                    "instant_acceleration" -> instantAccels
                    "peak_acceleration" -> {
                        val peak = if (instantAccels.isNotEmpty()) instantAccels.maxOfOrNull { abs(it) } ?: 0f else 0f
                        floatArrayOf(peak)
                    }
                    "hard_brake_flag" -> {
                        val hasHardBrake = instantAccels.any { it < -3.0f }
                        floatArrayOf(if (hasHardBrake) 1f else 0f)
                    }
                    "hard_accel_flag" -> {
                        val hasHardAccel = instantAccels.any { it > 3.0f }
                        floatArrayOf(if (hasHardAccel) 1f else 0f)
                    }
                    "speed_mps" -> locs.map { it.speedMps ?: 0f }.toFloatArray()
                    else -> floatArrayOf()
                }
            }

            "DEVICE" -> {
                windowEvents.asSequence()
                    .filterIsInstance<RawEvent.DeviceStateEvent>()
                    .map {
                        when (feature.column) {
                            "ds_screen_interactive" -> if (it.screenInteractive == true) 1f else 0f
                            "ds_device_locked" -> if (it.deviceLocked == true) 1f else 0f
                            "ds_audio_active" -> if (it.audioActive == true) 1f else 0f
                            "ds_handsfree" -> {
                                // 1.0 when audio is routed hands-free (Bluetooth / wired
                                // headset). Mirrors the Python ds_handsfree mapping.
                                val o = it.audioOutput?.uppercase() ?: ""
                                if (o == "BLUETOOTH" || o == "WIRED" || o.contains("HEADSET")) 1f else 0f
                            }
                            else -> 0f
                        }
                    }
                    .toList()
                    .toFloatArray()
            }

            else -> {
                val rows = windowEvents.asSequence()
                    .filterIsInstance<RawEvent.SensorEvent>()
                    .filter { it.sensorType == feature.sourceType }
                    .toList()

                if (rows.isEmpty()) {
                    floatArrayOf()
                } else {
                    // Pull one axis as an array, applying the low-pass filter when
                    // denoising. Matches the Python training pipeline, which filters
                    // each axis BEFORE computing magnitude.
                    fun axis(index: Int): FloatArray {
                        val raw = FloatArray(rows.size) { rows[it].values.getOrElse(index) { 0f } }
                        return if (isDenoised) ButterworthFilter.filter(raw) else raw
                    }

                    when (feature.column) {
                        "v1" -> axis(0)
                        "v2" -> axis(1)
                        "v3" -> axis(2)
                        "mag" -> {
                            val x = axis(0)
                            val y = axis(1)
                            val z = axis(2)
                            FloatArray(rows.size) {
                                sqrt(x[it] * x[it] + y[it] * y[it] + z[it] * z[it])
                            }
                        }
                        "angular_distance" -> {
                            // Derived from quaternion deltas; never denoised
                            // (the Python pipeline also skips the filter here).
                            val out = FloatArray(rows.size)
                            if (rows.isNotEmpty()) {
                                out[0] = 0f
                                for (i in 1 until rows.size) {
                                    val prev = rows[i - 1].values
                                    val curr = rows[i].values
                                    val q1 = FloatArray(4) { idx -> prev.getOrElse(idx) { 0f } }
                                    val q2 = FloatArray(4) { idx -> curr.getOrElse(idx) { 0f } }
                                    out[i] = calculateAngularDistance(q1, q2)
                                }
                            }
                            out
                        }
                        "reorientation" -> {
                            // Angle (rad) between consecutive RAW gravity vectors
                            // (never denoised). Mirrors calculate_gravity_reorientation.
                            val out = FloatArray(rows.size)
                            if (rows.isNotEmpty()) {
                                out[0] = 0f
                                for (i in 1 until rows.size) {
                                    out[i] = gravityReorientation(rows[i - 1].values, rows[i].values)
                                }
                            }
                            out
                        }
                        "hf_ratio", "zcr" -> {
                            // Per-window spectral texture of the RAW magnitude (denoise
                            // intentionally ignored — the high-freq content is the signal).
                            // Single scalar; interpolate() broadcasts it across the window.
                            val mag = FloatArray(rows.size) {
                                val v = rows[it].values
                                val x = v.getOrElse(0) { 0f }
                                val y = v.getOrElse(1) { 0f }
                                val z = v.getOrElse(2) { 0f }
                                sqrt(x * x + y * y + z * z)
                            }
                            val scalar = if (feature.column == "zcr") zeroCrossingRate(mag)
                                         else hfRatio(mag, SENSOR_FS, 6f)
                            floatArrayOf(scalar)
                        }
                        else -> floatArrayOf()
                    }
                }
            }
        }
    }

    private fun interpolate(source: FloatArray, targetSize: Int): FloatArray {
        if (source.isEmpty() || targetSize <= 0) return FloatArray(targetSize)
        if (source.size == 1) return FloatArray(targetSize) { source[0] }

        val out = FloatArray(targetSize)
        val maxSourceIndex = source.lastIndex.toFloat()
        val maxTargetIndex = (targetSize - 1).coerceAtLeast(1).toFloat()

        for (i in 0 until targetSize) {
            val srcPos = (i / maxTargetIndex) * maxSourceIndex
            val left = srcPos.toInt()
            val right = (left + 1).coerceAtMost(source.lastIndex)

            out[i] = if (left == right) {
                source[left]
            } else {
                val f = srcPos - left
                source[left] + (source[right] - source[left]) * f
            }
        }

        return out
    }

    private fun calculateAngularDistance(q1: FloatArray, q2: FloatArray): Float {
        // 0. Normalize quaternions
        var norm1 = sqrt(q1[0]*q1[0] + q1[1]*q1[1] + q1[2]*q1[2] + q1[3]*q1[3])
        if (norm1 < 1e-10f) norm1 = 1.0f
        val q1n = floatArrayOf(q1[0]/norm1, q1[1]/norm1, q1[2]/norm1, q1[3]/norm1)

        var norm2 = sqrt(q2[0]*q2[0] + q2[1]*q2[1] + q2[2]*q2[2] + q2[3]*q2[3])
        if (norm2 < 1e-10f) norm2 = 1.0f
        val q2n = floatArrayOf(q2[0]/norm2, q2[1]/norm2, q2[2]/norm2, q2[3]/norm2)

        // 1. Calculate the dot product (x1*x2 + y1*y2 + z1*z2 + w1*w2)
        val dotProduct = (q1n[0] * q2n[0]) + (q1n[1] * q2n[1]) + (q1n[2] * q2n[2]) + (q1n[3] * q2n[3])
        
        // 2. Take the absolute value for the shortest path
        val absDot = abs(dotProduct)
        
        // 3. Clamp the value between -1.0 and 1.0 to prevent rounding errors crashing acos
        val clampedDot = absDot.coerceIn(0.0f, 1.0f)
        
        // 4. Calculate delta theta in radians
        return 2.0f * acos(clampedDot)
    }

    private fun gravityReorientation(g1: FloatArray, g2: FloatArray): Float {
        // Angle between two gravity vectors. No abs() (gravity is a true direction,
        // not a double-covered quaternion), matching the Python implementation.
        val x1 = g1.getOrElse(0) { 0f }; val y1 = g1.getOrElse(1) { 0f }; val z1 = g1.getOrElse(2) { 0f }
        val x2 = g2.getOrElse(0) { 0f }; val y2 = g2.getOrElse(1) { 0f }; val z2 = g2.getOrElse(2) { 0f }
        var n1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1); if (n1 < 1e-10f) n1 = 1f
        var n2 = sqrt(x2 * x2 + y2 * y2 + z2 * z2); if (n2 < 1e-10f) n2 = 1f
        val dot = (x1 * x2 + y1 * y2 + z1 * z2) / (n1 * n2)
        return acos(dot.coerceIn(-1f, 1f))
    }

    private fun zeroCrossingRate(m: FloatArray): Float {
        val n = m.size
        if (n < 8) return 0f
        val mean = (m.sum() / n)
        var prev = sign(m[0] - mean)
        var crossings = 0
        for (t in 1 until n) {
            val s = sign(m[t] - mean)
            if (s != prev) crossings++
            prev = s
        }
        return crossings.toFloat() / (n - 1)
    }

    private fun hfRatio(m: FloatArray, fs: Float, cutoff: Float): Float {
        // Fraction of spectral power at/above `cutoff` Hz, via a direct real DFT on
        // the detrended signal. Runs once per window (not per timestep), so O(n^2)
        // over ~166 samples is fine.
        val n = m.size
        if (n < 8) return 0f
        val mean = m.sum() / n
        val x = FloatArray(n) { m[it] - mean }
        var total = 0.0
        var hf = 0.0
        val half = n / 2
        for (k in 0..half) {
            var re = 0.0
            var im = 0.0
            val w = -2.0 * PI * k / n
            for (t in 0 until n) {
                val a = w * t
                re += x[t] * cos(a)
                im += x[t] * sin(a)
            }
            val p = re * re + im * im
            total += p
            if (k.toFloat() * fs / n >= cutoff) hf += p
        }
        return if (total > 1e-12) (hf / total).toFloat() else 0f
    }
}