package dev.distraction.demo.ml.core

import android.content.Context
import dev.distraction.demo.ml.api.InferenceModelDefinition

object ModelRegistry {

    private val models: MutableList<InferenceModelDefinition> = mutableListOf()
    private var isInitialized = false

    fun init(context: Context) {
        if (isInitialized) return
        
        models.clear()
        
        val scannedModels = ModelScanner.scanModels(context)
        models.addAll(scannedModels)
        
        isInitialized = true
    }

    fun getAll(): List<InferenceModelDefinition> = models

    fun getById(id: String?): InferenceModelDefinition? {
        if (id.isNullOrBlank()) return null
        return models.firstOrNull { it.id == id }
    }

    fun getDefault(): InferenceModelDefinition? = models.firstOrNull()
}