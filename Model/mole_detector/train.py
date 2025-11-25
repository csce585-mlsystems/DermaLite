#!/usr/bin/env python3
"""
Convenience wrapper for training the mole detector
Automatically sets the config to 'mole_detector' mode
"""

import sys
import os

# Add shared scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

# Import config and set to mole_detector mode
from utils import config
config.set_config('mole_detector')

# Import and run the training script
from scripts.train import train_model

if __name__ == "__main__":
    print("Training Mole Detector (Skin vs Non-Skin Detection)")
    print("Using ISIC 2024 + ImageNet datasets\n")
    train_model()
