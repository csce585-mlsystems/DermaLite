import os
import tensorflow as tf
import coremltools as ct
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Keras Conversion 

# # -----------------------------
# # CONFIG
# # -----------------------------
# KERAS_H5_PATH = "/Users/t/Desktop/DermaLite/dermalite_mobilenet_model.h5" 
# SAVED_MODEL_PATH = "dermalite_saved_model" 
# MLMODEL_PATH = "dermalite_model.mlpackage" 
# BACKBONE = "mobilenet"                          
# NUM_CLASSES = 7                                 
# INPUT_SHAPE = (224, 224, 3)                     
# SCALE = 1/255.0                                 

# # -----------------------------
# # STEP 1: Rebuild model architecture
# # -----------------------------
# inputs = tf.keras.Input(shape=INPUT_SHAPE)
# base_model = MobileNetV2(weights=None, include_top=False, input_tensor=inputs)

# x = layers.GlobalAveragePooling2D()(base_model.output)
# x = layers.Dropout(0.3)(x)
# x = layers.Dense(128, activation='relu')(x)
# outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
# model = models.Model(inputs=inputs, outputs=outputs)

# # -----------------------------
# # STEP 2: Load weights
# # -----------------------------
# model.load_weights(KERAS_H5_PATH)
# print("Weights loaded successfully.")

# # -----------------------------
# # STEP 3: Save TensorFlow SavedModel
# # -----------------------------
# if os.path.exists(SAVED_MODEL_PATH):
#     import shutil
#     shutil.rmtree(SAVED_MODEL_PATH)
# model.save(SAVED_MODEL_PATH, save_format="tf") 
# print("SavedModel created.")

# # -----------------------------
# # STEP 4: Convert to Core ML
# # -----------------------------
# mlmodel = ct.convert(
#     SAVED_MODEL_PATH,
#     inputs=[ct.ImageType(shape=(1, 224, 224, 3), scale=SCALE)]
# )
# mlmodel.save(MLMODEL_PATH) 

## Pytorch conversion

import torch
import coremltools as ct
from torchvision import models
import torch.nn as nn

### USER INPUT 
# Path to your PyTorch .pth file
PTH_PATH = "/Users/t/Desktop/DermaLite/mobilenetv2_ham10000.pth"
MLPACKAGE_PATH = "dermalite_mobilenetv2.mlpackage"

# Number of classes in your HAM10000 dataset
NUM_CLASSES = 7

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load MobileNetV2 architecture (same as training)
model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features, NUM_CLASSES)
)

# Load trained weights
model.load_state_dict(torch.load(PTH_PATH, map_location=device))
model.eval()  # set to evaluation mode

# 1. Create example input tensor
example_input = torch.rand(1, 3, 224, 224)

# 2. Trace the model
traced_model = torch.jit.trace(model, example_input)
print("PyTorch model successfully traced to TorchScript.")

# 3. Convert the TRACED model to Core ML
mlmodel = ct.convert(
    traced_model,  # Pass the traced_model instead of the raw model object
    inputs=[ct.ImageType(name="input_1", shape=example_input.shape, scale=1/255.0, bias=[0,0,0])],
    source='pytorch'
)

# Save as .mlpackage
mlmodel.save(MLPACKAGE_PATH)
print(f"Core ML .mlpackage saved at: {MLPACKAGE_PATH}")