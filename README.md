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
|- data (mandatory)
|- src (mandatory)
|   |- model.py
|   |- example.py
|- train.py
|- run.py (mandatory)
|- result.py (mandatory)
```

⚠️ Notes:  
- All projects must include the `run.<ext>` script (extension depends on your programming language) at the project root directory. This is the script users will run to execute your project.  
- If your project computes/compares metrics such as accuracy, latency, or energy, you must include the `result.<ext>` script to plot the results.  
- Result files such as `.csv`, `.jpg`, or raw data must be saved in the `data` directory.  

## How to reproduce code 

## Demo  
- All projects must include video(s) demonstrating your project.  
- Please use annotations/explanations to clarify what is happening in the demo.  
---
