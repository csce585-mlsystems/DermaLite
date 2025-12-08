# DermaLite
Lightweight, quantized deep learning system for mobile-friendly skin lesion classification with explainable visualizations

## Group Info  
- Theodore Villalva   
  - Email: villalvt@email.sc.edu  
- Ryan Caudill 
  - Email: rcaudill@email.sc.edu
- DJ Ravenell 
  - Email: devarr@email.sc.edu  

## Project Summary/Abstract  
### DermaLite is a lightweight AI system designed to bring skin cancer screening to mobile devices. Instead of focusing only on model accuracy, we engineered the pipeline around real-world usability: compact models that run fast on phones, clear visual explanations with Grad-CAM, and reproducible workflows with DVC. The result is a system that balances medical reliability with the efficiency required for everyday use.

## Problem Description  
- Traditional skin cancer detection models often demand high compute resources and lack transparency, which limits their use outside research labs. Our project addresses this by building an efficient, explainable model that can classify skin lesions on-device.  
- Motivation  
  - Expand access to early skin cancer screening through on-device AI.
  - Increase trust in AI diagnostics via visual explanations (Grad-CAM) that show why predictions are made.
  - Explore the trade-offs between accuracy, speed, and memory in medical AI.  
- Challenges  
  - Compressing models without losing diagnostic performance.
  - Designing a pipeline that works reproducibly across different datasets and devices.
  - Making explanations simple enough for users while still useful for clinicians. 

## Contribution  
- [`Extension of existing work`]  
- [`Novel contribution`]  
- We improve skin lesion classification for mobile devices as follows:

  - Contribution 1: We implement model quantization and pruning to reduce memory footprint and inference time, enabling fast, real-time on-device predictions without sacrificing clinically relevant accuracy.
  - Contribution 2: We integrate explainable visualizations via Grad-CAM directly into the mobile pipeline, allowing clinicians and users to see why the model makes each prediction.
  - Contribution 3: We implement on-device uncertainty estimation, giving users and clinicians confidence metrics alongside predictions to support better decision-making.

## Future Ideas  
- A pipeline that transforms and augments input images into a format that enables more accurate classification by the model.

- Rather than just sticking with one CNN (MobileNet), we can also experiment with EfficientNet and quantize the models made from that. This way, our experiment and project are more robust and insightful, rather than just analyzing the metrics after quantization and being able to deploy them on mobile.  


## References  
- Nirupama, and Virupakshappa. “Mobilenet-V2: An Enhanced Skin Disease Classification by Attention and Multi-Scale Features.” Journal of Imaging Informatics in Medicine, U.S. National Library of Medicine, June 2025, pmc.ncbi.nlm.nih.gov/articles/PMC12092329/.
- Himel, Galib Muhammad Shahriar, et al. “Skin Cancer Segmentation and Classification Using Vision Transformer for Automatic Analysis in Dermatoscopy-Based Noninvasive Digital System.” International Journal of Biomedical Imaging, U.S. National Library of Medicine, 3 Feb. 2024, pmc.ncbi.nlm.nih.gov/articles/PMC10858797/. 

## Reproducing Code for Milestone 1 

### 1. Download the Dataset
- Dataset: HAM10000 (10,015 dermatoscopic images across 7 categories: `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`)  
- Download link: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  

```bash
# Install Kaggle CLI if not already installed
pip install kaggle

# Download dataset
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Unzip to data folder
unzip skin-cancer-mnist-ham10000.zip -d data/
```

### 2. Install Dependencies
```bash
uv sync
```

### 2. Train the model
```bash
python trainp0.py
```
### What to Look for During Training

- **Training and validation loss** decreasing over epochs  
- **Training and validation accuracy** improving  
- **Early stopping messages** (if validation performance plateaus)  
- **Final metrics printed at the end**, including:
  - **Accuracy**: Overall classification correctness
  - **AUC**: Ability to rank positive cases higher than negative cases
  - **Loss**: Categorical crossentropy

  
# < The following is only applicable for the final project submission >  


## Dependencies  
### Include all dependencies required to run the project. Example:  
- Python 3.12+  
- Ubuntu 22.04 / macOS 13+
- TensorFlow >= 2.20.0
- Pandas >= 2.2.3
- NumPy >= 1.26.0
- scikit-learn >= 1.6.0
- Pillow >= 11.0.0

For Python users: Please use [uv](https://docs.astral.sh/uv/) as your package manager instead of `pip`. Your repo must include both the `uv.lock` and `pyproject.toml` files.  

## Directory Structure  
Example:  
```
├── apis
│   └── gradcam_api.py
├── best_base_weights.h5
├── confusion_matrix_epoch_10.png
├── confusion_matrix_epoch_2.png
├── confusion_matrix_epoch_4.png
├── confusion_matrix_epoch_6.png
├── confusion_matrix_epoch_8.png
├── dermalite
│   ├── bin
│   ├── DermaLite
│   ├── DermaLite.xcodeproj
│   ├── include
│   ├── lib
│   ├── pyvenv.cfg
│   └── share
├── dermalite_efficientnetb0_model.h5
├── DermaLite_Final_Report.docx
├── dermalite_mobilenet_model.h5
├── dermalite_mobilenetv2.mlpackage
│   ├── Data
│   └── Manifest.json
├── dermalite_model.h5
├── dermalite_model.mlpackage
│   ├── Data
│   └── Manifest.json
├── dermalite_saved_model
│   ├── assets
│   ├── fingerprint.pb
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
├── dermalite_test
│   ├── bin
│   ├── include
│   ├── lib
│   └── pyvenv.cfg
├── doc
│   ├── DermaLite_Final_Report.docx
│   ├── Milestone P0 — Project Proposal and Motivation.md
│   ├── Milestone P1 — Initial Experiment and Evaluation Setup.md
│   ├── Slides Milestone P0 — Project Proposal and Motivation.pdf
│   ├── Slides Milestone P1 — Initial Experiment and Evaluation Setup.pdf
│   └── Slides Milestone P2 - Final Presentation.pdf
├── mobilenetv2_ham10000_balanced.pth
├── mobilenetv2_ham10000.pth
├── Model
│   ├── binary_classifier
│   ├── modelv1.py
│   ├── mole_detector
│   ├── multiclass_classifier
│   ├── README.md
│   ├── requirements.txt
│   ├── sa.py
│   ├── shared
│   └── trainp1.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── test_binary_classifier.py
├── tests
│   ├── __init__.py
│   ├── __pycache__
│   ├── conftest.py
│   ├── README.md
│   ├── test_binary_classifier.py
│   ├── test_dermalite_binary.py
│   ├── test_mole_detector.py
│   └── test_multiclass_classifier.py
├── training_history.pn
└── uv.lock
```

## Demo  
[![DermaLite Demo](https://github.com/user-attachments/assets/33fe860a-8395-460a-8d9b-99c8ea23f55d)](https://github.com/csce585-mlsystems/DermaLite/wiki/Demo)
- This demo walks through the core functionality of DermaLite. It begins by showing the three primary screens: Scan, Library, and Insights. After that, it demonstrates selecting a mole image from the library, running the classifier, and viewing the prediction along with a Grad CAM heatmap that the user can adjust for transparency.

- The demo also highlights important model limitations. A non lesion image of a dog is not classified as a mole, which is expected. The camera workflow is then shown, capturing images in real time. The model fails to handle a darker skinned sample and does not return a correct result. A lighter skinned sample is processed but incorrectly labeled as malignant. These cases help illustrate the need for more diverse training data and additional calibration work.
---
