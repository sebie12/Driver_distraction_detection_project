# Distraction Inference Demo (Android)

A standalone Android app that runs the **v14.5 distraction-detection model on-device, in real time** — the Edge AI half of the [showcase](../README.md). It reads the phone's inertial, GNSS and device-state streams, slides a 10-second window over them, runs the quantized TensorFlow Lite model, and surfaces a live distraction score.

This is the inference path extracted from the internship prototype, reduced to a single bundled model (`v14.5`) so it does one thing: **demonstrate the model running on a real device.**

> The bundled `app/src/main/assets/models_denoised/v14_5/model.tflite` is the **exact model that ran in the field**, not a re-export.

## What it does

1. **Collects** sensor data — gyroscope, linear accelerometer, gravity and game-rotation vectors (high sampling rate), GNSS speed, and device state (screen lock / hands-free audio).
2. **Processes** each stream into the model's feature windows — optional Butterworth denoise, the spectral-texture channels (`hf_ratio`, `zcr`), and a sliding window at 16.67 Hz. See `ml/models/SequenceExtractionUtils.kt` and `SlidingWindowRawDataProcessor.kt`.
3. **Infers** with a name-matched multi-input TFLite runner (`ml/tflite/TFLiteMultiInputRunner.kt`) and applies the metadata threshold (`ml/core/BinaryThresholdPostProcessor.kt`).
4. **Displays** the latest score / predicted class on the main screen, updated once per second.

The collection → processing → inference flow is orchestrated by a foreground `service/LoggingService.kt`; the inference subsystem itself lives entirely under `ml/` with no dependency on anything outside it (only `org.tensorflow.lite` and `org.json`).

## Architecture of the `ml/` package

```
ml/
├── api/        inference contracts (ModelRunner, RawDataProcessor, PostProcessor, ...)
├── core/       InferenceCoordinator, StandardInferencePipeline, ModelScanner/Registry
├── models/     feature definitions, sliding-window + sequence extraction, Butterworth filter
├── tflite/     multi-input TFLite runner (matches tensors BY NAME) + model inspector
├── runtime/    last-inference state store
└── selection/  which model / window stride is active
```

Models are discovered dynamically: any folder under `assets/models_denoised/` containing both `metadata.json` and `model.tflite` is picked up by `ModelScanner`. The window length, feature list, input tensor names and decision threshold all come from the model's `metadata.json` — no values are hardcoded.

## Build & run

Requirements: Android Studio (or the Android SDK + JDK 17–21), a device with API 28+.

```bash
# from this directory
echo "sdk.dir=$ANDROID_HOME" > local.properties
./gradlew assembleDebug
# APK: app/build/outputs/apk/debug/app-debug.apk
```

Then install on a phone, grant location + activity-recognition permissions, press **Iniciar**, and mount/drive (or hand-hold) to watch the score change. The model relies on TensorFlow Lite *Flex* ops (the LSTM), pulled in via the `tensorflow-lite-select-tf-ops` dependency already declared in `app/build.gradle.kts`.

## Provenance & licensing

This app is **derived from the internship prototype's inference code**, reduced to the v14.5 model for demonstration. The UI strings are in European Portuguese (as written during the internship). It is published as part of a portfolio/showcase under the repository's [CC BY-ND 4.0 license](../LICENSE); it is not intended to be forked into a product. No proprietary dataset or company telemetry is included.
