import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
model = keras.saving.load_model("hf://hasibzunair/melanet")
