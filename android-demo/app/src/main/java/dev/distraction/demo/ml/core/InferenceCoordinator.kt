package dev.distraction.demo.ml.core

import android.content.Context
import dev.distraction.demo.ml.api.ModelCategory
import dev.distraction.demo.ml.selection.ModelCategoryRepository
import dev.distraction.demo.ml.api.InferencePipeline
import dev.distraction.demo.ml.api.InferenceResult
import android.util.Log
import dev.distraction.demo.ml.selection.ModelSelectionRepository
import dev.distraction.demo.ml.types.RawEvent
import java.io.File

class InferenceCoordinator(
    private val context: Context,
    private val modelSelectionRepository: ModelSelectionRepository,
    private val categoryRepository: ModelCategoryRepository
) {
    private var activePipeline: InferencePipeline? = null
    private var activeModelId: String? = null
    var isPaused: Boolean = false

    fun initialize() {
        val currentCategory = categoryRepository.getSelectedCategory()
        val isDenoised = currentCategory == ModelCategory.DENOISED

        val selectedModelId = modelSelectionRepository.getSelectedModelId()
        val modelDefinition = ModelRegistry.getById(selectedModelId)
            ?: ModelRegistry.getDefault()
            ?: return

        activePipeline = modelDefinition.createPipeline(context, isDenoised, -1L)
        activeModelId = modelDefinition.id
    }

    fun getActiveModelId(): String? = activeModelId

    fun getActiveMetadata(): ModelMetadata? = activePipeline?.metadata

    fun onTripStarted(tripId: String?, startTimestampMillis: Long) {
        ensureInitialized()
        activePipeline?.onTripStarted(tripId, startTimestampMillis)
    }

    fun onTripEnded(tripId: String?, endTimestampMillis: Long) {
        activePipeline?.onTripEnded(tripId, endTimestampMillis)
    }

    fun onRawEvent(event: RawEvent): InferenceResult? {
        ensureInitialized()

        if (isPaused) {
            return null
        }

        val pipeline = activePipeline ?: return null
        pipeline.onRawEvent(event)

        return if (pipeline.canInfer()) {
            pipeline.runInferenceIfReady()
        } else {
            null
        }
    }

    fun reloadSelectedModel() {
        activePipeline?.reset()
        activePipeline = null
        activeModelId = null
        initialize()
    }

    fun reset() {
        activePipeline?.reset()
    }

    private fun ensureInitialized() {
        if (activePipeline == null) {
            initialize()
        }
    }
}
