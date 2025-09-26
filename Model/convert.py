import os
import tensorflow as tf
import coremltools as ct
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# -----------------------------
# CONFIG
# -----------------------------
KERAS_H5_PATH = "/Users/t/Desktop/DermaLite/dermalite_mobilenet_model.h5" 
SAVED_MODEL_PATH = "dermalite_saved_model" 
MLMODEL_PATH = "dermalite_model.mlpackage" 
BACKBONE = "mobilenet"                          
NUM_CLASSES = 7                                 
INPUT_SHAPE = (224, 224, 3)                     
SCALE = 1/255.0                                 

# -----------------------------
# STEP 1: Rebuild model architecture
# -----------------------------
inputs = tf.keras.Input(shape=INPUT_SHAPE)
base_model = MobileNetV2(weights=None, include_top=False, input_tensor=inputs)

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs=inputs, outputs=outputs)

# -----------------------------
# STEP 2: Load weights
# -----------------------------
model.load_weights(KERAS_H5_PATH)
print("Weights loaded successfully.")

# -----------------------------
# STEP 3: Save TensorFlow SavedModel
# -----------------------------
if os.path.exists(SAVED_MODEL_PATH):
    import shutil
    shutil.rmtree(SAVED_MODEL_PATH)
model.save(SAVED_MODEL_PATH, save_format="tf") 
print("SavedModel created.")

# -----------------------------
# STEP 4: Convert to Core ML
# -----------------------------
mlmodel = ct.convert(
    SAVED_MODEL_PATH,
    inputs=[ct.ImageType(shape=(1, 224, 224, 3), scale=SCALE)]
)
mlmodel.save(MLMODEL_PATH) 
