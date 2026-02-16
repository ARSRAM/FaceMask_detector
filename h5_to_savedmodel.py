import tensorflow as tf

h5_path = r"D:\work\vidut_vega\mask_model\mask_detector_new.h5"
savedmodel_path = r"D:\work\vidut_vega\mask_model\saved_model_tf216"

# Load WITHOUT compiling (important)
model = tf.keras.models.load_model(h5_path, compile=False)

# Save in TF 2.x format
model.save(savedmodel_path)

print("✅ SavedModel created at:", savedmodel_path)
