package dev.distraction.demo.ml.selection

import android.content.Context
import android.content.SharedPreferences

class StrideSelectionRepository(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    fun getStrideMillis(): Long {
        return prefs.getLong(KEY_STRIDE, 2500L) // Default 2.5s
    }

    fun setStrideMillis(strideMillis: Long) {
        prefs.edit()
            .putLong(KEY_STRIDE, strideMillis)
            .apply()
    }

    companion object {
        private const val PREF_NAME = "ml_stride_settings"
        private const val KEY_STRIDE = "stride_millis"
    }
}
