# CoreML Model Tests

Basic tests for DermaLite CoreML models, plus accuracy benchmarking and quantization tools.

## Running Tests

```bash
# Install test dependencies
pip install pytest coremltools numpy pillow scikit-learn pandas tqdm

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_mole_detector.py

# Run with verbose output
pytest tests/ -v
```

## Test Files

- `test_mole_detector.py` - Tests for mole detection model
- `test_binary_classifier.py` - Tests for binary benign/malignant classifier
- `test_multiclass_classifier.py` - Tests for 7-class lesion type classifier
- `test_dermalite_binary.py` - Tests for dermalite binary classifier
- `test_pipeline_accuracy.py` - Tests full 3-stage pipeline accuracy

## Accuracy Testing

### Test Pipeline Accuracy

```bash
# Test pipeline on a dataset
python tests/test_pipeline_accuracy.py

# Or use the PipelineTester class programmatically
from tests.test_pipeline_accuracy import PipelineTester

tester = PipelineTester()
results = tester.evaluate_on_dataset(
    dataset_path="./data/HAM10000",
    metadata_file="./data/HAM10000_metadata.csv",
    max_samples=100
)
```

## Quantization

### Quantize a Single Model

```bash
# Quantize a model to 8-bit
python tests/utils/quantize_models.py \
    --model path/to/model.mlmodel \
    --nbits 8 \
    --mode linear

# Quantize to 16-bit (less compression, better accuracy)
python tests/utils/quantize_models.py \
    --model path/to/model.mlmodel \
    --nbits 16 \
    --mode linear
```

### Quantize All Models

```bash
# Quantize all models in a directory
python tests/utils/quantize_models.py \
    --models-dir dermalite/DermaLite/DermaLite/Models \
    --nbits 8 \
    --mode linear
```

## Benchmarking Before/After Quantization

### Compare Single Model

```bash
# Benchmark original vs quantized model
python tests/benchmark_quantization.py \
    --original path/to/original.mlmodel \
    --quantized path/to/quantized.mlmodel \
    --test-data path/to/test/images \
    --test-labels path/to/test_labels.csv \
    --max-samples 100
```

### Quantize and Benchmark

```bash
# Automatically quantize and compare
python tests/benchmark_quantization.py \
    --original path/to/original.mlmodel \
    --test-data path/to/test/images \
    --nbits 8 \
    --quantize-mode linear \
    --max-samples 100
```

### Benchmark Full Pipeline

```python
from tests.benchmark_quantization import benchmark_pipeline
from pathlib import Path

# Benchmark the full 3-stage pipeline
benchmark_pipeline(
    original_models_dir=Path("dermalite/DermaLite/DermaLite/Models"),
    quantized_models_dir=Path("dermalite/DermaLite/DermaLite/Models_quantized"),
    test_images=test_images,
    test_labels=test_labels
)
```

## Quantization Modes

- `linear` - Linear quantization (default, good balance)
- `linear_symmetric` - Symmetric linear quantization
- `kmeans_lut` - K-means clustering based quantization (better accuracy, slower)
- `linear_lut` - Linear lookup table quantization

## Expected Results

After quantization, you typically see:
- **Size reduction**: 50-75% for 8-bit, 25-50% for 16-bit
- **Speed improvement**: 1.5-3x faster inference
- **Accuracy drop**: Usually <1-2% for 8-bit, <0.5% for 16-bit

## Model Locations

Models are expected to be in:
- `dermalite/DermaLite/DermaLite/Models/` (primary location)
- Root directory for `.mlpackage` files


