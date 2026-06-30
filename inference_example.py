"""
Minimal example: load the deployed distraction-detection model and run it.

This runs the full-precision Keras model, which is portable across platforms.
The quantized `model.tflite` is what actually ships on-device; because the
architecture contains an LSTM, the TFLite graph relies on TensorFlow "Flex" ops,
so on-device it is run through the LiteRT / Flex delegate (on Android, via the
`org.tensorflow:tensorflow-lite-select-tf-ops` dependency) — see the snippet at
the bottom of this file.

Inputs are matched *by name* (`sensor_data`, `speed_data`, `device_data`), never
by positional index, because conversion can reorder tensors. Here we feed random
data of the correct shape just to demonstrate the wiring; in production these
tensors are filled from a 10 s window of live phone telemetry sampled at 16.67 Hz.

    python inference_example.py
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).parent
META = json.loads((HERE / "model" / "metadata.json").read_text())

STREAM_FEATURES = {
    "sensor_data": META["sensor_features"],
    "speed_data": META["speed_features"],
    "device_data": META["device_features"],
}


def main() -> None:
    model = tf.keras.models.load_model(HERE / "model" / "keras_model.keras")

    print(f"window = {META['window_seconds']} s  |  "
          f"threshold = {META['threshold']:.3f}\n")

    # Build one random input window per named stream, matched by name.
    inputs = {}
    for tensor in model.inputs:
        name = tensor.name.split(":")[0]
        _, timesteps, n_features = tensor.shape
        inputs[name] = np.random.randn(1, timesteps, n_features).astype("float32")
        print(f"  {name:12s} shape=(1, {timesteps}, {n_features})  "
              f"features={STREAM_FEATURES.get(name, [])}")

    prob = float(model.predict(inputs, verbose=0).ravel()[0])

    print(f"\n  P(distraction) = {prob:.3f}")
    print(f"  decision       = "
          f"{'DISTRACTED' if prob >= META['threshold'] else 'attentive'}"
          "  (random input — illustrative only)")


if __name__ == "__main__":
    main()


# --- Running the quantized TFLite model (on-device path) -------------------
# Requires the Flex delegate because of the LSTM op. With full TensorFlow:
#
#   interp = tf.lite.Interpreter("model/model.tflite")   # needs select-tf-ops
#   interp.allocate_tensors()
#   by_name = {d["name"]: d for d in interp.get_input_details()}
#   for name, detail in by_name.items():
#       interp.set_tensor(detail["index"], window_for(name))   # match by NAME
#   interp.invoke()
#   prob = interp.get_tensor(interp.get_output_details()[0]["index"]).ravel()[0]
