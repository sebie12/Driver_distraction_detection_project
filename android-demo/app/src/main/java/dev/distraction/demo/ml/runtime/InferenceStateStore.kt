package dev.distraction.demo.ml.runtime

import android.content.Context

class InferenceStateStore(context: Context) {

    private val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    fun saveLastInference(
        modelId: String,
        score: Float,
        predictedClass: Int,
        windowEndMillis: Long
    ) {
        prefs.edit()
            .putString(KEY_MODEL_ID, modelId)
            .putFloat(KEY_SCORE, score)
            .putInt(KEY_PREDICTED_CLASS, predictedClass)
            .putLong(KEY_WINDOW_END_MS, windowEndMillis)
            .remove(KEY_LAST_ERROR)
            .apply()
    }

    fun saveLastError(error: String) {
        prefs.edit().putString(KEY_LAST_ERROR, error).apply()
    }

    fun getLastError(): String? = prefs.getString(KEY_LAST_ERROR, null)

    fun clearLastError() {
        prefs.edit().remove(KEY_LAST_ERROR).apply()
    }

    fun clear() {
        prefs.edit().clear().apply()
    }

    fun getLastInference(): LastInferenceUi? {
        val modelId = prefs.getString(KEY_MODEL_ID, null) ?: return null
        val score = prefs.getFloat(KEY_SCORE, 0f)
        val predictedClass = prefs.getInt(KEY_PREDICTED_CLASS, 0)
        val windowEndMillis = prefs.getLong(KEY_WINDOW_END_MS, 0L)

        return LastInferenceUi(
            modelId = modelId,
            score = score,
            predictedClass = predictedClass,
            windowEndMillis = windowEndMillis
        )
    }

    data class LastInferenceUi(
        val modelId: String,
        val score: Float,
        val predictedClass: Int,
        val windowEndMillis: Long
    )

    companion object {
        private const val PREF_NAME = "ml_inference_state"
        private const val KEY_MODEL_ID = "model_id"
        private const val KEY_SCORE = "score"
        private const val KEY_PREDICTED_CLASS = "predicted_class"
        private const val KEY_WINDOW_END_MS = "window_end_ms"
        private const val KEY_LAST_ERROR = "last_error"
    }
}
