#region --- Imports ---
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


# --- Step 1: Load CSV ---
df = pd.read_csv("/Users/t/Downloads/archive/HAM10000_metadata.csv")

# Add '.jpg' extension if not included
df['image_id'] = df['image_id'] + '.jpg'

# Map labels
lesion_types = df['dx'].unique()
num_classes = len(lesion_types)
print("Classes:", lesion_types)

# --- Step 2: Add full path for images in two folders ---
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
        return None  # Will filter out later

df['filepath'] = df['image_id'].apply(get_full_path)
df = df[df['filepath'].notnull()]  # Remove images not found
print(f"Total valid images: {len(df)}")

# --- Step 3: Split into train/val ---
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['dx'],
    random_state=42
)

# --- Step 4: Create ImageDataGenerators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='dx',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# --- Step 5: Build ResNet50 model ---
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=output)

# Freeze base model initially
base_model.trainable = False

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Step 6: Train model ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5  # Start small; increase later
)

# --- Optional: Fine-tune last few layers ---
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)
#endregion 
model.save("dermalite_model.h5")