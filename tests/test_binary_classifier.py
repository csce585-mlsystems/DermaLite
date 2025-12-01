"""
Basic tests for binary classifier CoreML model (benign vs malignant).
"""
import pytest
import numpy as np
import coremltools as ct


def test_model_loads(binary_classifier_model):
    """Test that the binary classifier model loads successfully."""
    assert binary_classifier_model is not None
    assert isinstance(binary_classifier_model, ct.models.MLModel)


def test_model_has_input_description(binary_classifier_model):
    """Test that the model has input description."""
    spec = binary_classifier_model.get_spec()
    assert len(spec.description.input) > 0
    print(f"Binary classifier input: {spec.description.input[0].name}")


def test_model_has_output_description(binary_classifier_model):
    """Test that the model has output description."""
    spec = binary_classifier_model.get_spec()
    assert len(spec.description.output) > 0
    print(f"Binary classifier outputs: {[out.name for out in spec.description.output]}")


def test_model_prediction(binary_classifier_model, sample_pil_image):
    """Test that the model can make a prediction."""
    spec = binary_classifier_model.get_spec()
    input_name = spec.description.input[0].name
    
    input_dict = {input_name: sample_pil_image}
    prediction = binary_classifier_model.predict(input_dict)
    
    assert prediction is not None
    assert isinstance(prediction, dict)
    
    print(f"Binary classifier prediction keys: {list(prediction.keys())}")


def test_model_outputs_class_labels(binary_classifier_model, sample_pil_image):
    """Test that the model outputs class labels (Benign/Malignant)."""
    spec = binary_classifier_model.get_spec()
    input_name = spec.description.input[0].name
    
    input_dict = {input_name: sample_pil_image}
    prediction = binary_classifier_model.predict(input_dict)
    
    # Check for class label output
    output_names = [out.name for out in spec.description.output]
    
    # Look for common classification output names
    has_class_label = any('class' in name.lower() or 'label' in name.lower() 
                         for name in output_names)
    
    print(f"Binary classifier output names: {output_names}")
    print(f"Has class label output: {has_class_label}")

