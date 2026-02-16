import tensorflow as tf

# Paths (FIXED)
keras_model_path = r"D:\work\vidut_vega\mask_model\mask_detector_new.h5"
tflite_model_path = r"D:\work\vidut_vega\mask_model\mask_detector_tf216.tflite"

# Load Keras model
model = tf.keras.models.load_model(keras_model_path)

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Keep ops compatible with TF 2.16
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

# Float32 for stability
converter.optimizations = []

# Convert
tflite_model = converter.convert()

# Save
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as:", tflite_model_path)
