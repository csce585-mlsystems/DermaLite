#!/usr/bin/env python3
"""
Convenience wrapper for training the binary classifier
Automatically sets the config to 'binary' mode
"""

import sys
import os

# Add shared scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

# Import config and set to binary mode
from utils import config
config.set_config('binary')

# Import and run the training script
from scripts.train import train_model

if __name__ == "__main__":
    print("Training Binary Classifier (Malignant Detection)")
    print("Using ISIC 2024 dataset\n")
    train_model()
