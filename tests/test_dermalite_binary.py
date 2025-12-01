"""
Basic tests for dermalite binary classifier CoreML model.
"""
import pytest
import numpy as np
import coremltools as ct


def test_model_loads(dermalite_binary_model):
    """Test that the dermalite binary classifier model loads successfully."""
    assert dermalite_binary_model is not None
    assert isinstance(dermalite_binary_model, ct.models.MLModel)


def test_model_has_input_description(dermalite_binary_model):
    """Test that the model has input description."""
    spec = dermalite_binary_model.get_spec()
    assert len(spec.description.input) > 0
    print(f"Dermalite binary classifier input: {spec.description.input[0].name}")


def test_model_has_output_description(dermalite_binary_model):
    """Test that the model has output description."""
    spec = dermalite_binary_model.get_spec()
    assert len(spec.description.output) > 0
    print(f"Dermalite binary classifier outputs: {[out.name for out in spec.description.output]}")


def test_model_prediction(dermalite_binary_model, sample_pil_image):
    """Test that the model can make a prediction."""
    spec = dermalite_binary_model.get_spec()
    input_name = spec.description.input[0].name
    
    input_dict = {input_name: sample_pil_image}
    prediction = dermalite_binary_model.predict(input_dict)
    
    assert prediction is not None
    assert isinstance(prediction, dict)
    
    print(f"Dermalite binary classifier prediction keys: {list(prediction.keys())}")

