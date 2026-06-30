package dev.distraction.demo.ml.debug

import android.content.Context
import android.content.SharedPreferences

class DebugModeRepository(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    fun isDebugWindowSummaryEnabled(): Boolean {
        return prefs.getBoolean(KEY_DEBUG_WINDOW_SUMMARY_ENABLED, false)
    }

    fun setDebugWindowSummaryEnabled(enabled: Boolean) {
        prefs.edit()
            .putBoolean(KEY_DEBUG_WINDOW_SUMMARY_ENABLED, enabled)
            .apply()
    }

    companion object {
        private const val PREF_NAME = "ml_debug_settings"
        private const val KEY_DEBUG_WINDOW_SUMMARY_ENABLED = "debug_window_summary_enabled"
    }
}