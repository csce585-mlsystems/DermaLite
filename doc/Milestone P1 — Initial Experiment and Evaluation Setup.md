# Efficient Model Quantization and Deployment on Apple Silicon with MLX
**Theodore Villalva, DJ Ravenell, Ryan Caudill**  
Department of Computer Science  
University of South Carolina  

---

## Introduction

The goal of this milestone was to evaluate the initial performance of our MobileNetV2-based skin lesion classification model on the HAM10000 dataset. This baseline establishes a reference point for later optimization, quantization, and deployment using Apple’s MLX framework. We hypothesized that transfer learning with pre-trained ImageNet weights and moderate data augmentation would achieve competitive classification accuracy (>70%) with minimal overfitting during early epochs.

---

## System and Data Setup

### Environment

All experiments were conducted locally on macOS using an Apple M1 Pro (16GB unified memory) with full Metal acceleration support. The model was implemented using TensorFlow 2.20.0 and the Keras backend, running under Python 3.13.3 — the latest stable version in the current ML ecosystem. Random seeds and environment variables were fixed to ensure full reproducibility.

**Configuration Summary**
- **Platform:** macOS (local development)
- **Python:** 3.13.3
- **Framework:** TensorFlow 2.20.0 / Keras backend
- **Hardware:** Apple M1 Pro (16GB unified memory)
- **Reproducibility:** Fixed random seed (42) and deterministic dataset splits

**Dependencies**
- `tensorflow==2.20.0` — Core deep learning framework (Metal acceleration)
- `pandas>=2.2.3` — Dataset handling
- `scikit-learn>=1.6.0` — Metrics and class weighting
- `pillow>=11.0.0` — Image preprocessing
- `numpy>=1.26.0` — Numerical operations

This setup provided a stable, GPU-accelerated environment compatible with Apple’s Metal backend, ensuring efficient training performance while maintaining reproducibility.

---

### Dataset

**Dataset Name:** HAM10000 (Human Against Machine with 10,000 training images)  
**Size:** 10,015 dermatoscopic images across 7 diagnostic categories — *akiec, bcc, bkl, df, nv, mel, and vasc.*

Each image was resized to **224×224**, normalized to **[0, 1]**, and augmented to improve generalization. Augmentation included random rotations, width/height shifts, brightness variations, and horizontal/vertical flips.

**Data Split**
- **Training:** 80%
- **Validation:** 20%

**Rationale:** The HAM10000 dataset is a widely used benchmark in dermatological deep learning research and presents realistic challenges due to significant class imbalance. This makes it ideal for evaluating model robustness in medical image classification tasks.

---

## Baseline Implementation

### Model Architecture

The baseline model used **MobileNetV2**, pretrained on ImageNet, as a feature extractor fine-tuned for skin lesion classification.

**Architecture Summary**
- **Base:** `MobileNetV2(weights='imagenet', include_top=False)`
- **Head:**
  - `GlobalAveragePooling2D`
  - `Dropout(0.3)`
  - `Dense(128, activation='relu', kernel_regularizer=L2(1e-4))`
  - `Dense(num_classes, activation='softmax', kernel_regularizer=L2(1e-4))`

**Training Configuration**
- **Optimizer:** Adam (lr=1e-4 for base, 1e-5 for fine-tuning)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy, AUC
- **Batch Size:** 32
- **Epochs:** 15 (base) + 15 (fine-tuning)
- **Callbacks:**
  - `EarlyStopping(patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(factor=0.2, patience=5)`
- **Class Weights:** Computed via `sklearn.utils.compute_class_weight` to mitigate imbalance

This structure provides strong performance with efficient computation, making it a suitable starting point before quantization and model deployment.

---

## Preliminary Experiment Results

After 30 total epochs (15 base training + 15 fine-tuning), the model demonstrated stable learning and solid generalization on validation data.

**Performance Metrics**
- **Accuracy:** 66.5%
- **AUC:** 0.923
- **Loss:** 0.974

### Observations

1. **High AUC despite moderate accuracy** — The model ranks a randomly chosen positive example higher than a negative one 92.3% of the time, reflecting strong discriminative ability.  
2. **Class imbalance impact** — Lower overall accuracy is largely due to the uneven distribution of classes within HAM10000, which reduces per-class precision on rare categories.  
3. **Stable convergence** — Validation loss and accuracy show minimal overfitting, indicating a well-regularized model suitable for fine-tuning.  

---

## Next Steps

The next phase will focus on interpretability, deeper analysis, and architectural expansion:

1. Integrate **Grad-CAM** visualizations to interpret model focus regions.  
2. Evaluate **per-class metrics** including precision, recall, and F1-score to assess imbalance handling.  
3. Train an **EfficientNet-B0** baseline to compare efficiency and performance before quantization.  
4. Prepare for **quantization and MLX deployment**, where inference time, model size, and accuracy trade-offs will be systematically measured.  

---

## Conclusion

This milestone successfully established a reproducible baseline for skin lesion classification using MobileNetV2 on the HAM10000 dataset. With an AUC of 0.923, the model exhibits strong feature discrimination, though accuracy remains constrained by class imbalance. These results form the foundation for future work on interpretability, per-class evaluation, and performance optimization through quantization and MLX-based deployment on Apple Silicon hardware.
