"""
Pytest configuration and shared fixtures for CoreML model tests.
"""
import pytest
import numpy as np
import coremltools as ct
from pathlib import Path
from PIL import Image

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "dermalite" / "DermaLite" / "DermaLite" / "Models"
ROOT_MODEL_DIR = PROJECT_ROOT


@pytest.fixture
def sample_image_array():
    """Create a sample image array (224, 224, 3) in RGB format, values 0-255."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor in normalized format (1, 3, 224, 224)."""
    # ImageNet normalized: values typically in [0, 1] range
    return np.random.rand(1, 3, 224, 224).astype(np.float32)


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL Image (224x224 RGB) for CoreML model predictions."""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture(scope="session")
def mole_detector_model():
    """Load mole detector CoreML model."""
    path = MODEL_DIR / "mole_detector.mlmodel"
    if not path.exists():
        pytest.skip(f"Mole detector model not found at {path}")
    try:
        return ct.models.MLModel(str(path))
    except Exception as e:
        pytest.skip(f"Failed to load mole detector model: {e}")


@pytest.fixture(scope="session")
def binary_classifier_model():
    """Load binary classifier CoreML model."""
    path = MODEL_DIR / "MalignancyResNet50Features.mlmodel"
    if not path.exists():
        pytest.skip(f"Binary classifier model not found at {path}")
    try:
        return ct.models.MLModel(str(path))
    except Exception as e:
        pytest.skip(f"Failed to load binary classifier model: {e}")


@pytest.fixture(scope="session")
def multiclass_classifier_model():
    """Load multiclass classifier CoreML model."""
    # Try .mlpackage first
    path = MODEL_DIR / "dermalite_mobilenetv2.mlpackage"
    if not path.exists():
        path = ROOT_MODEL_DIR / "dermalite_mobilenetv2.mlpackage"
    if not path.exists():
        pytest.skip(f"Multiclass classifier model not found")
    try:
        return ct.models.MLModel(str(path))
    except Exception as e:
        pytest.skip(f"Failed to load multiclass classifier model: {e}")


@pytest.fixture(scope="session")
def dermalite_binary_model():
    """Load dermalite binary classifier model."""
    path = MODEL_DIR / "dermalite_binary_classifier.mlmodel"
    if not path.exists():
        pytest.skip(f"Dermalite binary classifier model not found at {path}")
    try:
        return ct.models.MLModel(str(path))
    except Exception as e:
        pytest.skip(f"Failed to load dermalite binary classifier model: {e}")

