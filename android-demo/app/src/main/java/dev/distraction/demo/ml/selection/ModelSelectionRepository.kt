package dev.distraction.demo.ml.selection

interface ModelSelectionRepository {
    fun getSelectedModelId(): String?
    fun setSelectedModelId(modelId: String)
}