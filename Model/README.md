# Skin Lesion Classification Models

Two-stage deep learning system for skin cancer detection and diagnosis.

## Models

### 1. Binary Classifier (Malignant Detection)
**Purpose:** First-stage screening - determines if a mole is malignant or benign

- **Architecture:** MobileNetV2 (transfer learning from ImageNet)
- **Dataset:** ISIC 2024 (~400K images)
- **Classes:** 2 (benign, malignant)
- **Input Size:** 224×224 RGB
- **Parameters:** ~3.5M
- **Inference Time:** ~15ms per image (MPS)
- **Target Metric:** Recall > 85% for malignant (minimize false negatives)

**Location:** `binary_classifier/`

---

### 2. Multi-Class Classifier (Diagnosis Classification)
**Purpose:** Second-stage diagnosis - classifies specific type of skin lesion

- **Architecture:** MobileNetV2 (transfer learning from ImageNet)
- **Dataset:** HAM10000 (~10K images)
- **Classes:** 7 lesion types
  - akiec: Actinic keratoses
  - bcc: Basal cell carcinoma
  - bkl: Benign keratosis
  - df: Dermatofibroma
  - mel: Melanoma
  - nv: Melanocytic nevi
  - vasc: Vascular lesions
- **Input Size:** 224×224 RGB
- **Parameters:** ~3.5M
- **Inference Time:** ~15ms per image (MPS)
- **Target Metric:** Balanced accuracy across all classes

**Location:** `multiclass_classifier/`

---

## Two-Stage Pipeline

```
Input Image → Binary Classifier → Benign? → Stop (no further action)
                                ↓
                            Malignant? → Multi-Class Classifier → Specific Diagnosis
```

1. **Stage 1 (Binary):** Screen for malignancy - optimized for high recall
2. **Stage 2 (Multi-Class):** Diagnose specific condition - only runs if malignant

---

## Model Specifications

### Training Configuration
- **Optimizer:** Adam (lr=5e-5 binary, 1e-4 multiclass)
- **Batch Size:** 32
- **Augmentation:** Horizontal/vertical flips, rotation (±20°), color jitter
- **Class Weighting:** 80× for malignant (binary), auto-balanced (multiclass)
- **Early Stopping:** 7 epochs patience
- **LR Scheduling:** ReduceLROnPlateau (factor=0.5, patience=3)

### Model Size
- **Checkpoint Size:** ~14 MB per model
- **Memory Usage:** ~500 MB GPU/MPS during inference
- **Total System:** ~28 MB for both models

---

## Quick Start

### Train Binary Classifier
```bash
# Option 1: Using shared scripts with task argument
cd shared/scripts
python train.py --task binary

# Option 2: Using convenience wrapper
cd binary_classifier
python train.py
```

### Train Multi-Class Classifier
```bash
# Option 1: Using shared scripts with task argument
cd shared/scripts
python train.py --task multiclass

# Option 2: Using convenience wrapper
cd multiclass_classifier
python train.py
```

### Benchmark Models
```bash
cd shared/scripts
python benchmark.py --checkpoint ../binary_classifier/checkpoints/best_binary_model.pth
python benchmark.py --checkpoint ../multiclass_classifier/checkpoints/best_multiclass_model.pth
```

---

## Directory Structure

```
Model/
├── shared/                          # Shared utilities and scripts (DRY principle)
│   ├── utils/
│   │   ├── dataset_loaders.py      # HAM10000 & ISIC2024 dataset loaders
│   │   ├── config.py               # Unified configuration system
│   │   └── __init__.py
│   ├── scripts/
│   │   ├── train.py                # Unified training script (supports both tasks)
│   │   ├── benchmark.py            # Model evaluation and benchmarking
│   │   ├── verify_dataset.py       # Dataset verification utility
│   │   └── __init__.py
│   └── README.md                   # Detailed documentation
│
├── binary_classifier/               # Binary classifier workspace
│   ├── train.py                    # Convenience wrapper
│   ├── checkpoints/                # Saved model weights
│   ├── results/                    # Benchmark outputs
│   ├── requirements.txt
│   └── README.md
│
└── multiclass_classifier/           # Multiclass classifier workspace
    ├── train.py                    # Convenience wrapper
    ├── checkpoints/                # Saved model weights
    ├── results/                    # Benchmark outputs
    ├── requirements.txt
    └── README.md
```

### Refactoring Benefits

**Before:** ~2000 lines of duplicated code across both classifiers
**After:** ~1000 lines of shared code with no duplication

- **Single Source of Truth:** All training logic in one place
- **Easy Maintenance:** Bug fixes applied to both models simultaneously
- **Consistent Behavior:** Identical training pipeline for both tasks
- **Flexible Configuration:** Switch between binary/multiclass with one argument
- **Cleaner Codebase:** Eliminated duplicate files (dataset_loaders.py, train.py, benchmark.py, verify_dataset.py)
