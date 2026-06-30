package dev.distraction.demo.ml.models

/**
 * A 2nd-order Butterworth low-pass filter (Direct Form I).
 * Coefficients correspond to order=2, cutoff=4.0Hz, fs=16.67Hz, matching the
 * Python training pipeline (scipy.signal.butter in build_dl_dataset.py, fs=16.67).
 * b = [0.27464080, 0.54928160, 0.27464080]
 * a = [1.0, -0.07397728, 0.17254048]
 */
object ButterworthFilter {
    private val b0 = 0.27464080f
    private val b1 = 0.54928160f
    private val b2 = 0.27464080f

    private val a1 = -0.07397728f
    private val a2 = 0.17254048f

    fun filter(input: FloatArray): FloatArray {
        if (input.isEmpty()) return FloatArray(0)
        
        val output = FloatArray(input.size)
        
        var x1 = 0f
        var x2 = 0f
        var y1 = 0f
        var y2 = 0f

        for (i in input.indices) {
            val x0 = input[i]
            val y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            
            output[i] = y0
            
            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0
        }
        
        return output
    }
}
