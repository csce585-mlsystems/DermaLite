# Mole Detector - Skin vs Non-Skin Image Classifier

This model serves as the **first stage** in the DermaLite pipeline, filtering out non-skin images before they reach the binary and multiclass classifiers.

## Pipeline Position

```
Image → [Mole Detector] → Is it skin/mole?
           ├─→ NO: Return "Not a skin image"
           └─→ YES: → Binary Classifier → Multiclass Classifier
```

## Purpose

The mole detector prevents non-skin images from being processed by downstream models. This includes:
- Random objects (furniture, electronics, etc.)
- Food items
- Animals
- Clothing
- Text/documents
- Any non-dermatological images

## Dataset Requirements

This model requires two datasets:

### 1. Positive Class (Skin/Mole Images) - ISIC 2024
- **Location:** `../../ISIC2024_dataset/`
- **Source:** Already available in your project
- **Size:** ~400K images
- **Label:** `1` (skin)

### 2. Negative Class (Non-Skin Images) - ImageNet Subset
- **Location:** `../../nonmole_dataset/`
- **Source:** ImageNet random subset
- **Recommended Size:** 10K-50K images
- **Label:** `0` (non_skin)

## How to Download Non-Skin Images (ImageNet)

### Option 1: Using ImageNet-Datasets-Downloader (Recommended)

```bash
# Clone the downloader tool
git clone https://github.com/mf1024/ImageNet-Datasets-Downloader.git
cd ImageNet-Datasets-Downloader

# Install requirements
pip install -r requirements.txt

# Download random classes (adjust num_classes and images_per_class)
python downloader.py \
  -data_root ../../nonmole_dataset \
  -number_of_classes 50 \
  -images_per_class 1000

# This will download 50K images from 50 random ImageNet classes
```

### Option 2: Using Kaggle ImageNet

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (requires account + API token)
# Download from: https://www.kaggle.com/settings -> Create New API Token

# Download ImageNet Object Localization Challenge
kaggle competitions download -c imagenet-object-localization-challenge

# Extract a subset to nonmole_dataset/
# (You'll need to manually extract and organize)
```

### Option 3: Hugging Face Datasets

```python
from datasets import load_dataset
from PIL import Image
import os

# Load ImageNet-1K
dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)

# Download first 20K images
output_dir = "../../nonmole_dataset"
os.makedirs(output_dir, exist_ok=True)

for i, sample in enumerate(dataset):
    if i >= 20000:  # Download 20K images
        break
    img = sample['image']
    img.save(f"{output_dir}/imagenet_{i:06d}.jpg")
    if i % 1000 == 0:
        print(f"Downloaded {i} images...")
```

### Quick Test Option (For Development)

If you just want to test the pipeline without downloading ImageNet:

```bash
# Create a small test dataset with random images from the internet
mkdir -p ../../nonmole_dataset/test_images

# Manually add 100-1000 random non-skin images:
# - Screenshots from your computer
# - Photos of objects around you
# - Downloaded stock photos (chairs, tables, food, etc.)
```

## Training Configuration

Configuration is defined in `Model/shared/utils/config.py`:

```python
MOLE_DETECTOR_CONFIG = {
    'dataset': 'ISIC2024_and_ImageNet',
    'num_classes': 2,
    'class_names': ['non_skin', 'skin'],

    'training': {
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 1e-4,          # Higher than binary classifier
        'sample_fraction': 0.3,          # Use 30% of data for faster training
        'manual_class_weights': [1.0, 5.0],  # Heavily weight skin class
    }
}
```

### Class Weighting Strategy

- **Weights:** `[1.0, 5.0]` (non_skin, skin)
- **Rationale:** We strongly prefer **false positives** over **false negatives**
  - False Positive: Non-skin image classified as skin → processed downstream (safe, just wastes computation)
  - False Negative: Real skin image classified as non-skin → MISSED completely (dangerous!)

## Training the Model

```bash
# Navigate to mole_detector directory
cd Model/mole_detector

# Run training (requires datasets to be in place)
python train.py
```

### Expected Output

```
Training Mole Detector (Skin vs Non-Skin Detection)
Using ISIC 2024 + ImageNet datasets

======================================================================
Loading Mole Detector Dataset (ISIC 2024 + ImageNet)
======================================================================
Loading skin/mole images from ISIC 2024...
Loaded 400000 skin/mole images

Loading non-skin images from ../../nonmole_dataset...
Loaded 50000 non-skin images

Total dataset size: 450000 images
Class distribution:
  - Non-skin (0): 50000 (11.1%)
  - Skin (1): 400000 (88.9%)

⚡ BALANCED SAMPLING ENABLED: 67500 samples per class
   Strategy: Each class gets equal representation
   Original: 450000 → Sampled: 135000 images (balanced across 2 classes)
   Estimated training time: ~2.7-4.1 hours

Training with class weights: [1.0, 5.0] (manual)
...
```

## Model Output

After training, you'll get:
- **Checkpoint:** `Model/mole_detector/checkpoints/best_mole_detector_model.pth`
- **Architecture:** MobileNetV2 with 2-class output
- **Input:** 224×224 RGB images
- **Output:** Binary classification (0=non_skin, 1=skin)

## Next Steps

1. **Train the model** (after setting up datasets)
2. **Convert to Core ML** (see `Model/shared/scripts/convert.py`)
3. **Integrate into iOS** (update `MLService.swift`)

## Questions?

Think about these before asking:
1. How many non-skin images do you want to download? (More = better generalization, but longer download/training)
2. What ImageNet classes make sense for your use case? (furniture, electronics, animals, food, etc.)
3. Do you want to train on the full dataset or a subset first?
