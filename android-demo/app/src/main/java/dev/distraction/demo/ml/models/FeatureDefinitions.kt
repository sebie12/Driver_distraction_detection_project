package dev.distraction.demo.ml.models

data class FeatureDef(
    val sourceType: String,
    val column: String,
    val outputName: String
)

object FeatureDefinitions {

    val SENSOR_FEATURES = listOf(
        FeatureDef("LINACC", "mag", "LINACC_mag"),
        FeatureDef("GYRO", "mag", "GYRO_mag"),
        FeatureDef("GAMEROT", "angular_distance", "GAMEROT_angular_distance")
    )

    // Legacy features for backwards compatibility with v6
    val SENSOR_FEATURES_LEGACY = listOf(
        FeatureDef("LINACC", "v1", "LINACC_v1"),
        FeatureDef("LINACC", "v2", "LINACC_v2"),
        FeatureDef("LINACC", "v3", "LINACC_v3"),
        FeatureDef("GYRO", "v1", "GYRO_v1"),
        FeatureDef("GYRO", "v2", "GYRO_v2"),
        FeatureDef("GYRO", "v3", "GYRO_v3")
    )

    val GAMEROT_FEATURES = listOf(
        FeatureDef("GAMEROT", "angular_distance", "GAMEROT_angular_distance")
    )

    val SPEED_FEATURES = listOf(
        FeatureDef("LOC", "speed_mps", "speed_mps")
    )

    val ACCEL_FEATURES = listOf(
        FeatureDef("LOC", "instant_acceleration", "instant_acceleration"),
        FeatureDef("LOC", "peak_acceleration", "peak_acceleration"),
        FeatureDef("LOC", "hard_brake_flag", "hard_brake_flag"),
        FeatureDef("LOC", "hard_accel_flag", "hard_accel_flag")
    )

    val DEVICE_FEATURES = listOf(
        FeatureDef("DEVICE", "ds_device_locked", "ds_device_locked"),
        FeatureDef("DEVICE", "ds_audio_active", "ds_audio_active"),  // legacy: kept resolvable for v4–v14.4 models
        FeatureDef("DEVICE", "ds_handsfree", "ds_handsfree")
    )

    // Derived features used by suffix .4 (GRAV_reorientation) and .5 (spectral
    // texture: hf_ratio = power fraction >6 Hz, zcr = zero-crossing rate, both on
    // the RAW magnitude). NOTE: registering the name lets ModelScanner resolve the
    // metadata; the on-device value computation lives in SequenceExtractionUtils
    // and must be wired there (on the raw, pre-denoise signal) before deployment.
    val EXTRA_FEATURES = listOf(
        FeatureDef("GRAV", "reorientation", "GRAV_reorientation"),
        FeatureDef("GYRO", "hf_ratio", "GYRO_hf_ratio"),
        FeatureDef("GYRO", "zcr", "GYRO_zcr"),
        FeatureDef("LINACC", "hf_ratio", "LINACC_hf_ratio"),
        FeatureDef("LINACC", "zcr", "LINACC_zcr")
    )

    private val allFeatures: Map<String, FeatureDef> by lazy {
        (SENSOR_FEATURES + SENSOR_FEATURES_LEGACY + GAMEROT_FEATURES + SPEED_FEATURES + ACCEL_FEATURES + DEVICE_FEATURES + EXTRA_FEATURES)
            .associateBy { it.outputName }
    }

    fun getFeatureDef(name: String): FeatureDef? {
        return allFeatures[name]
    }
}