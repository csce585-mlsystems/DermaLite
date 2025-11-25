# Mole Detection Model Report (Refined)

## 1. Model Overview
* **Architecture:** ResNet-18 (Fine-Tuned Phase 2)
* **Input Size:** 224x224 RGB
* **Model File Size:** 42.92 MB (Expected to remain the same after fine-tuning)
* **Training Device:** Apple M2 (MPS Backend)

## 2. Dataset Information
* **Positive Class (Mole):** ISIC 2024 & HAM10000 (Subsampled for balance)
* **Negative Class (Other - Hard Negatives):** * **Caltech 101** (General objects)
    * **Oxford-IIIT Pet** (Biological features like fur, skin, eyes)
    * **DTD (Describable Textures Dataset)** (Abstract, mole-like textures e.g., blotchy, dotted)
* **Preprocessing:** Resize, RGB Conversion, Normalization (ImageNet stats)

## 3. Performance Metrics (After Fine-Tuning)
| Metric | Score |
| :--- | :--- |
| **Accuracy** | [INSERT ACCURACY]% |
| **Precision (Mole)** | [INSERT PRECISION] |
| **Recall (Mole)** | [INSERT RECALL] |
| **F1-Score (Mole)** | [INSERT F1] |

## 4. Confusion Matrix (After Fine-Tuning)
* **True Negatives (Correctly identified 'Not Mole'):** [INSERT TOP-LEFT NUMBER]
* **False Positives (Non-moles thought to be Moles):** [INSERT TOP-RIGHT NUMBER]
* **False Negatives (Moles missed):** [INSERT BOTTOM-LEFT NUMBER]
* **True Positives (Correctly identified 'Mole'):** [INSERT BOTTOM-RIGHT NUMBER]