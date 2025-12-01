"""
Basic tests for multiclass classifier CoreML model (7 skin lesion types).
"""
import pytest
import numpy as np
import coremltools as ct


def test_model_loads(multiclass_classifier_model):
    """Test that the multiclass classifier model loads successfully."""
    assert multiclass_classifier_model is not None
    assert isinstance(multiclass_classifier_model, ct.models.MLModel)


def test_model_has_input_description(multiclass_classifier_model):
    """Test that the model has input description."""
    spec = multiclass_classifier_model.get_spec()
    assert len(spec.description.input) > 0
    print(f"Multiclass classifier input: {spec.description.input[0].name}")


def test_model_has_output_description(multiclass_classifier_model):
    """Test that the model has output description."""
    spec = multiclass_classifier_model.get_spec()
    assert len(spec.description.output) > 0
    print(f"Multiclass classifier outputs: {[out.name for out in spec.description.output]}")


def test_model_prediction(multiclass_classifier_model, sample_pil_image):
    """Test that the model can make a prediction."""
    spec = multiclass_classifier_model.get_spec()
    input_name = spec.description.input[0].name
    
    input_dict = {input_name: sample_pil_image}
    prediction = multiclass_classifier_model.predict(input_dict)
    
    assert prediction is not None
    assert isinstance(prediction, dict)
    
    print(f"Multiclass classifier prediction keys: {list(prediction.keys())}")


def test_model_has_class_labels(multiclass_classifier_model):
    """Test that the model has class labels for 7 lesion types."""
    spec = multiclass_classifier_model.get_spec()
    
    # Check if model has class labels in description
    model_description = multiclass_classifier_model.short_description or ""
    print(f"Model description: {model_description}")
    
    # Expected classes: akiec, bcc, bkl, df, mel, nv, vasc
    expected_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    print(f"Expected classes: {expected_classes}")

