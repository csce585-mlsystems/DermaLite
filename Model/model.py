#region --- Imports ---
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.metrics import AUC, CategoricalAccuracy
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

#endregion

#region --- Parameters ---
DATA_CSV = "/Users/t/Downloads/archive/HAM10000_metadata.csv"
PART1 = "/Users/t/Downloads/archive/HAM10000_images_part_1"
PART2 = "/Users/t/Downloads/archive/HAM10000_images_part_2"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
L2_REG = 1e-4             # Regularization factor
EPOCHS_BASE = 15         
EPOCHS_FINE = 15          
#endregion

#region --- Step 1: Load CSV and image paths ---
df = pd.read_csv(DATA_CSV)
df['image_id'] = df['image_id'] + '.jpg'

# Map labels
lesion_types = df['dx'].unique()
num_classes = len(lesion_types)
print("Classes:", lesion_types)

# Define function to get full path
def get_full_path(image_id):
    path1 = os.path.join(PART1, image_id)
    path2 = os.path.join(PART2, image_id)
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    return None

df['filepath'] = df['image_id'].apply(get_full_path)
df = df[df['filepath'].notnull()]
print(f"Total valid images: {len(df)}")
#endregion

#region --- Step 2: Train/Validation Split ---
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['dx'],
    random_state=42
)
#endregion

#region --- Step 3: Image Generators (Robust Augmentation) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True 
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='dx',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    color_mode="rgb"
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='dx',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    color_mode="rgb"
)
#endregion

#region --- Step 4: Build MobileNetV2 model ---
# Explicitly define input tensor
inputs = Input(shape=IMG_SIZE + (3,))

# Base Model: MobileNetV2 for mobile deployment focus
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)

# Custom classification head with Regularization
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
# Added L2 regularization to the final classification layer
output = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x) 
model = models.Model(inputs=inputs, outputs=output)

# Freeze base model initially
base_model.trainable = False

# Added AUC to metrics list for better evaluation of imbalanced data
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[CategoricalAccuracy(name='accuracy'), AUC(name='auc')]
)
model.summary()
#endregion

#region --- Step 5: Compute Class Weights ---
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['dx']),
    y=train_df['dx']
)
class_weights = dict(enumerate(class_weights_array))
print("Class weights:", class_weights)
#endregion

#region --- Step 6: Callbacks ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), 
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.2) 
]
#endregion

#region --- Step 7: Base Training (Feature Extraction) ---
print("\n--- Starting Base Training (Feature Extraction) ---")
history_base = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_BASE,
    class_weight=class_weights, 
    callbacks=callbacks
)
#endregion

#region --- Step 8: Fine-Tuning ---
print("\n--- Starting Fine-Tuning ---")
base_model.trainable = True

# Aggressive Fine-Tuning: Unfreeze a larger section (MobileNetV2 is smaller, so unfreeze more)
# MobileNetV2 has about 155 layers; unfreezing the last 70 is a good start.
for layer in base_model.layers[:-70]:
    layer.trainable = False

# Compile again with a very low learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), 
    loss='categorical_crossentropy',
    metrics=[CategoricalAccuracy(name='accuracy'), AUC(name='auc')]
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE,
    class_weight=class_weights, 
    callbacks=callbacks
)
#endregion

#region --- Step 9: Save Model ---
model.save("dermalite_mobilenet_model.h5")
print("Saved dermalite_mobilenet_model.h5")
#endregion