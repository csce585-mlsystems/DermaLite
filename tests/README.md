# CoreML Model Tests

Basic tests for DermaLite CoreML models.

## Running Tests

```bash
# Install test dependencies
pip install pytest coremltools numpy pillow

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


