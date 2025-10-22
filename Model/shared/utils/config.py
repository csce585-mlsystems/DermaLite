"""
Unified configuration system for DermaLite models
Supports both binary (malignant detection) and multiclass (lesion diagnosis) tasks
"""

import os

# ==================== MODEL CONFIGURATION (SHARED) ====================

MODEL_CONFIG = {
    'architecture': 'mobilenet_v2',  # Options: 'mobilenet_v2', 'efficientnet_b0', 'resnet50'
    'pretrained': True,
    'dropout': 0.2,
    'input_size': 224
}

# ==================== DATA AUGMENTATION CONFIGURATION (SHARED) ====================

AUGMENTATION_CONFIG = {
    'horizontal_flip_prob': 0.5,
    'vertical_flip_prob': 0.5,
    'rotation_degrees': 20,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225]
}

# ==================== DEVICE CONFIGURATION (SHARED) ====================

DEVICE = 'auto'  # Options: 'auto', 'mps', 'cuda', 'cpu'

# ==================== DATASET-SPECIFIC CONFIGURATIONS ====================

# Binary Classifier: ISIC 2024 - Malignant vs Benign
BINARY_CONFIG = {
    'dataset': 'ISIC2024',
    'task': 'binary',
    'root_dir': '../../ISIC2024_dataset',
    'metadata_file': 'train-metadata.csv',
    'num_classes': 2,
    'class_names': ['benign', 'malignant'],

    'training': {
        'num_epochs': 30,
        'batch_size': 32,
        'learning_rate': 5e-5,
        'num_workers': 4,
        'test_size': 0.2,
        'random_state': 42,

        # Data sampling
        'use_sampling': True,
        'sample_fraction': 0.5,
        'sampling_strategy': 'balanced',

        # Learning rate scheduler
        'use_scheduler': True,
        'scheduler_patience': 3,
        'scheduler_factor': 0.5,

        # Early stopping
        'use_early_stopping': True,
        'early_stop_patience': 7,

        # Class weighting - CRITICAL for malignant detection
        'use_class_weights': True,
        'manual_class_weights': [1.0, 80.0],  # [benign, malignant]
        'use_manual_weights': True,  # Aggressive weighting for medical safety

        # Save directory
        'save_dir': '../checkpoints',
        'checkpoint_name': 'best_binary_model.pth'
    }
}

# Multiclass Classifier: HAM10000 - 7-Class Lesion Diagnosis
MULTICLASS_CONFIG = {
    'dataset': 'HAM10000',
    'task': 'multiclass',
    'root_dir': '../../HAM10000_dataset',
    'num_classes': 7,
    'class_names': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],

    'training': {
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_workers': 4,
        'test_size': 0.2,
        'random_state': 42,

        # Data sampling
        'use_sampling': False,
        'sample_fraction': 1.0,
        'sampling_strategy': 'balanced',

        # Learning rate scheduler
        'use_scheduler': True,
        'scheduler_patience': 3,
        'scheduler_factor': 0.5,

        # Early stopping
        'use_early_stopping': True,
        'early_stop_patience': 7,

        # Class weighting
        'use_class_weights': True,
        'use_manual_weights': False,  # Auto-compute balanced weights

        # Save directory
        'save_dir': '../checkpoints',
        'checkpoint_name': 'best_multiclass_model.pth'
    }
}

# ==================== ACTIVE CONFIGURATION ====================

# Default to binary classifier - can be overridden
_ACTIVE_CONFIG = BINARY_CONFIG
DATASET = _ACTIVE_CONFIG['dataset']
TRAINING_CONFIG = _ACTIVE_CONFIG['training']

def set_config(config_name):
    """
    Set the active configuration

    Args:
        config_name: 'binary' or 'multiclass'
    """
    global _ACTIVE_CONFIG, DATASET, TRAINING_CONFIG

    if config_name.lower() == 'binary':
        _ACTIVE_CONFIG = BINARY_CONFIG
    elif config_name.lower() == 'multiclass':
        _ACTIVE_CONFIG = MULTICLASS_CONFIG
    else:
        raise ValueError(f"Unknown config: {config_name}. Use 'binary' or 'multiclass'")

    DATASET = _ACTIVE_CONFIG['dataset']
    TRAINING_CONFIG = _ACTIVE_CONFIG['training']

def get_active_config():
    """Get the active dataset configuration"""
    return _ACTIVE_CONFIG

def print_config():
    """Print current configuration"""
    config = get_active_config()
    task_name = "BINARY CLASSIFIER" if config['task'] == 'binary' else "MULTI-CLASS CLASSIFIER"

    print("=" * 70)
    print(f"{task_name} CONFIGURATION")
    print("=" * 70)
    print(f"\nDataset: {config['dataset']}")
    print(f"Task: {config['task']}")
    print(f"Number of classes: {config['num_classes']}")
    print(f"Class names: {config['class_names']}")
    print(f"\nModel: {MODEL_CONFIG}")
    print(f"\nTraining: {config['training']}")
    print(f"\nAugmentation: {AUGMENTATION_CONFIG}")
    print(f"\nDevice: {DEVICE}")
    print("=" * 70)

if __name__ == "__main__":
    print("Testing Binary Config:")
    set_config('binary')
    print_config()

    print("\n\nTesting Multiclass Config:")
    set_config('multiclass')
    print_config()
