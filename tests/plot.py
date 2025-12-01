import matplotlib.pyplot as plt
from pathlib import Path
from eval_stage1_to_2 import Stage1_2_Tester
import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
BINARY_MODEL = "/Users/t/Desktop/DermaLite/dermalite/DermaLite/DermaLite/Models/dermalite_binary_classifier.mlmodel"
MULTICLASS_MODEL = "/Users/t/Desktop/DermaLite/dermalite/DermaLite/DermaLite/Models/dermalite_mobilenetv2.mlpackage"
METADATA = "/Users/t/Downloads/archive/HAM10000_metadata.csv"
DATASET = "/Users/t/Downloads/archive/HAM10000_images_part_1"

# Optional: limit number of images for quick testing
MAX_SAMPLES = 100

# -------------------------------
# Initialize Tester
# -------------------------------
tester = Stage1_2_Tester(
    binary_model_path=BINARY_MODEL,
    multiclass_model_path=MULTICLASS_MODEL,
    debug=False,
    malignancy_threshold=0.35
)

# -------------------------------
# Evaluate
# -------------------------------
results = tester.evaluate_on_dataset(
    dataset_path=DATASET,
    metadata_file=METADATA,
    max_samples=MAX_SAMPLES
)

# -------------------------------
# Plot confusion matrix
# -------------------------------
import seaborn as sns
import numpy as np

cm = results['confusion_matrix']
labels = ['benign', 'malignant']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Binary Classification Confusion Matrix\nAccuracy: {results['accuracy']*100:.2f}%")
plt.show()

# -------------------------------
# Optional: Plot confidence histogram
# -------------------------------
plt.figure(figsize=(6, 4))
plt.hist(results['confidences'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("Prediction Confidence Distribution")
plt.show()
