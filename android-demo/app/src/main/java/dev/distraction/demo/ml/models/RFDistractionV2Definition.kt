package dev.distraction.demo.ml.models

class RFDistractionV2Definition : TreeModelDefinition() {
    override val id: String = "RF_distraction_v2"
    override val displayName: String = "RF Distraction v2"
    override val version: String = "2.0"
    
    override val windowDurationMillis: Long = 10000L
    override val threshold: Float = 0.4599996507167816f
    override val metadataAssetPath: String = "models_trees/RF_distraction_v6/metadata.json"
    override val tfliteAssetPath: String = "models_trees/RF_distraction_v6/model.tflite"
    
    override val featureNames: List<String> = listOf(
        "LINACC_v1_mean", "LINACC_v1_min", "LINACC_v1_max",
        "LINACC_v2_mean", "LINACC_v2_min", "LINACC_v2_max",
        "LINACC_v3_mean", "LINACC_v3_min", "LINACC_v3_max",
        "LINACC_mag_mean", "LINACC_mag_min", "LINACC_mag_max",
        "LINACC_sqsum_xy_mean", "LINACC_sqsum_yz_mean",
        "LINACC_sumsq75", "LINACC_xz_y_mean",
        "GYRO_v1_mean", "GYRO_v1_min", "GYRO_v1_max",
        "GYRO_v2_mean", "GYRO_v2_min", "GYRO_v2_max",
        "GYRO_v3_mean", "GYRO_v3_min", "GYRO_v3_max",
        "GYRO_mag_mean", "GYRO_mag_min", "GYRO_mag_max",
        "GYRO_sqsum_xy_mean", "GYRO_sqsum_yz_mean",
        "GYRO_sumsq75", "GYRO_xz_y_mean",
        "GAMEROT_angular_distance_mean", "GAMEROT_angular_distance_max",
        "loc_instant_acceleration_mean", "loc_instant_acceleration_min", "loc_instant_acceleration_max",
        "ds_device_locked_frac", "ds_audio_active_frac"
    )
}
