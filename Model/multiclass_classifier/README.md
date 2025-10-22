# Multi-Class Diagnosis Classifier

Second-stage classifier that determines the specific type of skin lesion.

## Model Overview

- **Task:** 7-class classification of skin lesion types
- **Dataset:** HAM10000 (~10K dermoscopic images)
- **Architecture:** MobileNetV2 + custom classifier head
- **Input:** 224×224 RGB images
- **Output:** 7 lesion types
- **Purpose:** Detailed diagnosis for malignant cases

## Classes

1. **akiec** - Actinic keratoses and intraepithelial carcinoma
2. **bcc** - Basal cell carcinoma
3. **bkl** - Benign keratosis-like lesions
4. **df** - Dermatofibroma
5. **mel** - Melanoma
6. **nv** - Melanocytic nevi
7. **vasc** - Vascular lesions

## Performance Targets

- **Balanced Accuracy:** > 75% across all classes
- **Per-class F1:** > 0.65 for all classes
- **AUC:** > 0.85 (macro average)

## Class Weighting

Uses auto-computed balanced weights to handle HAM10000 class imbalance.

## Training

```bash
cd scripts
python train.py
```

Training outputs:
- Best model → `checkpoints/best_multiclass_model.pth`
- Per-epoch metrics for all 7 classes

## Evaluation

```bash
cd scripts
python benchmark.py --checkpoint ../checkpoints/best_multiclass_model.pth
```

Outputs:
- `results/confusion_matrix.png` - 7×7 confusion matrix
- `results/benchmark_results.txt` - Per-class precision/recall/F1
- Detailed classification report

## Configuration

Edit `utils/config.py` to adjust:
- Dataset path (HAM10000 location)
- Training epochs
- Learning rate
- Class weighting strategy

## Usage in Pipeline

This model runs second in the two-stage system:
1. Receives images flagged as "malignant" by binary classifier
2. Classifies into one of 7 specific lesion types
3. Provides detailed diagnosis for medical review
