package dev.distraction.demo.ml.models

enum class TensorLayout {
    TIMESTEPS_FEATURES,   // [1, timesteps, features]
    FEATURES_TIMESTEPS    // [1, features, timesteps]
}