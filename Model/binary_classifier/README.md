# Binary Malignant Detection Model

First-stage screening classifier that determines if a mole is malignant or benign.

## Model Overview

- **Task:** Binary classification (malignant vs benign)
- **Dataset:** ISIC 2024 (~400K dermoscopic images)
- **Architecture:** MobileNetV2 + custom classifier head
- **Input:** 224×224 RGB images
- **Output:** 2 classes (benign: 0, malignant: 1)
- **Purpose:** High-sensitivity screening to catch potential malignancies

## Performance Targets

- **Malignant Recall:** > 85% (minimize false negatives - avoid missing cancer)
- **Malignant Precision:** > 40% (balance false positives with safety)
- **AUC:** > 0.90
- **Overall Accuracy:** ~85-90%

## Class Weighting

Uses aggressive 80× weighting for malignant class:
- **Benign:** 1.0× weight
- **Malignant:** 80.0× weight

This heavily penalizes false negatives (missing cancer cases).

## Training

```bash
cd scripts
python train.py
```

Training outputs:
- Best model → `checkpoints/best_binary_model.pth`
- Metrics displayed per epoch with malignant-specific stats

## Evaluation

```bash
cd scripts
python benchmark.py --checkpoint ../checkpoints/best_binary_model.pth
```

Outputs:
- `results/confusion_matrix.png` - Visual performance breakdown
- `results/roc_curve.png` - Diagnostic ability curve
- `results/benchmark_results.txt` - Detailed metrics

## Configuration

Edit `utils/config.py` to adjust:
- Data sampling fraction
- Class weights
- Training epochs
- Learning rate

## Usage in Pipeline

This model runs first in the two-stage system:
1. Input image → Binary classifier
2. If **benign** → Stop (no further action)
3. If **malignant** → Send to multi-class classifier for specific diagnosis
