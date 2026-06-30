package dev.distraction.demo.ml.selection

import android.content.Context
import android.content.SharedPreferences
import dev.distraction.demo.ml.api.ModelCategory

class ModelCategoryRepository(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    fun getSelectedCategory(): ModelCategory {
        val name = prefs.getString(KEY_CATEGORY, ModelCategory.NORMAL.name)
        return try {
            ModelCategory.valueOf(name!!)
        } catch (e: Exception) {
            ModelCategory.NORMAL
        }
    }

    fun setSelectedCategory(category: ModelCategory) {
        prefs.edit()
            .putString(KEY_CATEGORY, category.name)
            .apply()
    }

    companion object {
        private const val PREF_NAME = "ml_category_settings"
        private const val KEY_CATEGORY = "model_category"
    }
}
