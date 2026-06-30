package dev.distraction.demo.ml.api

data class TensorInput(
    val index: Int,
    val name: String,
    val data: Any
)

data class MapInput(
    val features: Map<String, Float>
)

data class ModelInputs(
    val inputs: List<TensorInput> = emptyList(),
    val mapInput: MapInput? = null
)