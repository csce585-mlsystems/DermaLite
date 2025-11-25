# Robust Mole Detection Model Report (Final)

## 1. Model Overview
* **Architecture:** ResNet-18 (Trained from scratch with Robust pipeline)
* **Input Size:** 224x224 RGB
* **Model File Size:** 43.17 MB
* **Training Device:** Apple M2 (MPS Backend)

## 2. Dataset Information
* **Positive Class (Mole):** ISIC 2024 & HAM10000
* **Negative Class (Hard Negatives):**
    * **Oxford-IIIT Pet** (Biological features: fur, eyes, wet noses)
    * **DTD (Describable Textures Dataset)** (Abstract textures: blotchy, dotted, porous)
    * **Flowers102** (Organic shapes and colors)
    * *(Note: Caltech 101 was removed to eliminate easy shortcuts)*
* **Preprocessing (Anti-Artifact):** Aggressive RandomResizedCrop (0.4-1.0), GaussianBlur (p=0.5), GaussianNoise, RandomGrayscale (p=0.2), Heavy ColorJitter.

## 3. Performance Metrics
| Metric | Score |
| :--- | :--- |
| **Accuracy** | 99.39% |
| **Precision (Mole)** | 0.99 |
| **Recall (Mole)** | 1.00 |
| **F1-Score (Mole)** | 0.99 |

## 4. Confusion Matrix
* **True Negatives (Correctly identified 'Not Mole'):** 1268
* **False Positives (Non-moles thought to be Moles):** 12
* **False Negatives (Moles missed):** 4
* **True Positives (Correctly identified 'Mole'):** 1348