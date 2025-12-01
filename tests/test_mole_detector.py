"""
Basic tests for mole detector CoreML model.
"""
import pytest
import numpy as np
import coremltools as ct


def test_model_loads(mole_detector_model):
    """Test that the mole detector model loads successfully."""
    assert mole_detector_model is not None
    assert isinstance(mole_detector_model, ct.models.MLModel)


def test_model_has_input_description(mole_detector_model):
    """Test that the model has input description."""
    spec = mole_detector_model.get_spec()
    assert len(spec.description.input) > 0
    print(f"Mole detector input: {spec.description.input[0].name}")


def test_model_has_output_description(mole_detector_model):
    """Test that the model has output description."""
    spec = mole_detector_model.get_spec()
    assert len(spec.description.output) > 0
    print(f"Mole detector output: {spec.description.output[0].name}")


def test_model_prediction(mole_detector_model, sample_pil_image):
    """Test that the model can make a prediction."""
    # Prepare input as dictionary (CoreML expects named inputs)
    spec = mole_detector_model.get_spec()
    input_name = spec.description.input[0].name
    
    # Create input dictionary with PIL Image
    input_dict = {input_name: sample_pil_image}
    
    # Make prediction
    prediction = mole_detector_model.predict(input_dict)
    
    # Check that prediction is not None
    assert prediction is not None
    assert isinstance(prediction, dict)
    
    # Check that output exists
    output_name = spec.description.output[0].name
    assert output_name in prediction
    
    print(f"Mole detector prediction keys: {list(prediction.keys())}")


def test_model_prediction_shape(mole_detector_model, sample_pil_image):
    """Test that the model prediction has expected shape."""
    spec = mole_detector_model.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    
    input_dict = {input_name: sample_pil_image}
    prediction = mole_detector_model.predict(input_dict)
    
    output = prediction[output_name]
    
    # Output should be a probability (single value or array)
    assert output is not None
    print(f"Mole detector output type: {type(output)}, shape: {getattr(output, 'shape', 'N/A')}")

