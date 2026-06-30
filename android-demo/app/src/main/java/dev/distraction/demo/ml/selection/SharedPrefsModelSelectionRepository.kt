package dev.distraction.demo.ml.selection

import android.content.Context
import android.content.SharedPreferences

class SharedPrefsModelSelectionRepository(
    context: Context
) : ModelSelectionRepository {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    override fun getSelectedModelId(): String? {
        return prefs.getString(KEY_SELECTED_MODEL_ID, null)
    }

    override fun setSelectedModelId(modelId: String) {
        prefs.edit().putString(KEY_SELECTED_MODEL_ID, modelId).apply()
    }

    companion object {
        private const val PREF_NAME = "ml_model_selection"
        private const val KEY_SELECTED_MODEL_ID = "selected_model_id"
    }
}