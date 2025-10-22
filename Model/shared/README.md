# DermaLite Shared Model Components

This directory contains shared utilities and scripts for both binary and multiclass skin lesion classification models.

## Structure

```
shared/
├── utils/
│   ├── dataset_loaders.py   # HAM10000 and ISIC2024 dataset loaders
│   ├── config.py             # Unified configuration system
│   └── __init__.py
└── scripts/
    ├── train.py              # Unified training script
    ├── benchmark.py          # Model evaluation and benchmarking
    ├── verify_dataset.py     # Dataset verification utility
    └── __init__.py
```

## Configuration System

The unified config system (`utils/config.py`) supports two tasks:

### Binary Classification (Malignant Detection)
- **Dataset**: ISIC 2024
- **Task**: Detect malignant vs benign lesions
- **Classes**: 2 (benign, malignant)
- **Use case**: First-stage screening

### Multiclass Classification (Lesion Diagnosis)
- **Dataset**: HAM10000
- **Task**: Classify specific lesion types
- **Classes**: 7 (akiec, bcc, bkl, df, mel, nv, vasc)
- **Use case**: Second-stage diagnosis

## Usage

### Training

Train a binary classifier:
```bash
cd shared/scripts
python train.py --task binary
```

Train a multiclass classifier:
```bash
cd shared/scripts
python train.py --task multiclass
```

### Benchmarking

```bash
cd shared/scripts
python benchmark.py --checkpoint path/to/model.pth --dataset ISIC2024
```

### Dataset Verification

```bash
cd shared/scripts
python verify_dataset.py
```

## Benefits of This Structure

1. **No Code Duplication**: Single source of truth for all shared functionality
2. **Easy Maintenance**: Bug fixes and improvements in one place
3. **Consistent Behavior**: Both models use the same training logic
4. **Flexible Configuration**: Switch between tasks with a single argument
5. **Cleaner Codebase**: Reduced from ~2000 lines of duplicated code to ~1000 lines of shared code

## Migration Notes

The previous structure had duplicate code in:
- `binary_classifier/utils/` → **Removed** (now uses `shared/utils/`)
- `binary_classifier/scripts/` → **Removed** (now uses `shared/scripts/`)
- `multiclass_classifier/utils/` → **Removed** (now uses `shared/utils/`)
- `multiclass_classifier/scripts/` → **Removed** (now uses `shared/scripts/`)

Checkpoints and results remain in their respective classifier directories for organization.
