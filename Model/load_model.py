# --- load_model_eval.py ---
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split

# Load saved model
model = load_model("dermalite_model.h5")  

# Load metadata
df = pd.read_csv("/Users/t/Downloads/archive/HAM10000_metadata.csv")
df['image_id'] = df['image_id'] + '.jpg'

# Add full paths
part1 = "/Users/t/Downloads/archive/HAM10000_images_part_1"
part2 = "/Users/t/Downloads/archive/HAM10000_images_part_2"

def get_full_path(image_id):
    path1 = os.path.join(part1, image_id)
    path2 = os.path.join(part2, image_id)
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None

df['filepath'] = df['image_id'].apply(get_full_path)
df = df[df['filepath'].notnull()]

# Get validation split
_, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

# Create validation generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc:.4f}, Loss: {loss:.4f}")
