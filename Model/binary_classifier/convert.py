import torch
import torch.nn as nn
from torchvision import models
import coremltools as ct
import joblib
import numpy as np
import os
import sys

# Explicit imports to fix "name '_tree' is not defined" error in coremltools
import sklearn
from sklearn.ensemble import RandomForestClassifier
import sklearn.tree 
# Sometimes coremltools relies on this internal module being available in the namespace
try:
    from sklearn.tree import _tree
except ImportError:
    pass

# --- CONFIGURATION ---
# UPDATED: Using the final model saved by the Autopilot Trainer
CHECKPOINT_PATH = "./checkpoints_malignancy/malignancy_resnet50_focal.pth" 
RF_PATH = "./hybrid_model/rf_head.pkl" 
SCALER_PATH = "./hybrid_model/scaler.pkl"
FINAL_MLPACKAGE_PATH = "MalignancyHybridRF.mlmodel"

# ImageNet Normalization
INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]

# Device
device = torch.device("cpu") 

# --- 1. PREPARE THE BACKBONE (ResNet-50 + Scaler) ---
class BackboneWithScaler(nn.Module):
    """
    Wraps ResNet AND the StandardScaler logic into one PyTorch module.
    Output: Normalized feature vector ready for the Random Forest.
    """
    def __init__(self, checkpoint_path, scaler_path):
        super().__init__()
        # A. Load ResNet
        self.model = models.resnet50()
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )
        
        print(f"Loading ResNet weights from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # The final model save (malignancy_resnet50_focal.pth) is a state_dict directly
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading PyTorch checkpoint: {e}")
            sys.exit(1)
        
        # Remove Head
        self.model.fc = nn.Identity()
        self.model.eval()

        # B. Load Scaler Stats
        print(f"Loading Scaler stats from {scaler_path}...")
        try:
            scaler = joblib.load(scaler_path)
            # StandardScaler logic: z = (x - mean) / scale
            # We register these as buffers so they are saved with the model but not trained
            self.register_buffer('scaler_mean', torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer('scaler_scale', torch.tensor(scaler.scale_, dtype=torch.float32))
        except Exception as e:
            print(f"Error loading Scaler: {e}")
            sys.exit(1)

    def forward(self, x):
        # 1. Extract Features
        features = self.model(x)
        features = torch.flatten(features, 1) # [1, 2048]

        # 2. Apply Scaling (Math: (x - u) / s)
        # Broadcasting will handle the shapes
        features_norm = (features - self.scaler_mean) / self.scaler_scale

        # Return with shape [1, 2048] for CoreML (batch dimension required)
        return features_norm

def convert_backbone():
    print("--- 🏗️ Converting Backbone (ResNet + Scaler) ---")
    pt_model = BackboneWithScaler(CHECKPOINT_PATH, SCALER_PATH)
    
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(pt_model, example_input)
    
    image_input = ct.ImageType(
        name="input_image",
        shape=(1, 3, 224, 224),
        scale=1.0 / (255.0 * INPUT_STD[0]), 
        bias=[-INPUT_MEAN[0] / INPUT_STD[0], 
              -INPUT_MEAN[1] / INPUT_STD[1], 
              -INPUT_MEAN[2] / INPUT_STD[2]],
        channel_first=True,
        color_layout='RGB'
    )

    # Convert to Legacy 'neuralnetwork'
    mlmodel_backbone = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="neuralnetwork"
    )

    # Fix the output shape in the spec
    spec = mlmodel_backbone.get_spec()
    # The output should be [1, 2048] not [] or [2048]
    output = spec.description.output[0]
    output.type.multiArrayType.shape.extend([1, 2048])
    print(f"   (Fixed backbone output shape to: {list(output.type.multiArrayType.shape)})")

    mlmodel_backbone = ct.models.MLModel(spec)
    return mlmodel_backbone

# --- 2. PREPARE THE CLASSIFIER (RF Only) ---
def convert_classifier():
    print("--- 🌲 Converting Random Forest ---")
    
    try:
        rf = joblib.load(RF_PATH)
        print(f"   Loaded object type: {type(rf)}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find RF model at {RF_PATH}")
        sys.exit(1)

    # Convert with explicit input name="embedding" to match backbone output
    # Shape must be (1, 2048) to match backbone
    print("Converting RF with input='embedding' shape (1, 2048) to match backbone output...")
    import coremltools.converters.sklearn
    mlmodel_classifier = coremltools.converters.sklearn.convert(
        rf,
        input_features=[("embedding", ct.models.datatypes.Array(1, 2048))]
    )

    # --- MANUAL PROTOBUF SURGERY ---
    spec = mlmodel_classifier.get_spec()
    print(f"   (Classifier input confirmed as '{spec.description.input[0].name}')")

    # Replace class labels from [0, 1] to ["Benign", "Malignant"]
    # The sklearn converter uses int64 labels, we need string labels
    print(f"   (Updating class labels to ['Benign', 'Malignant'])")
    if spec.WhichOneof('Type') == 'pipelineClassifier':
        # Update pipeline classifier labels
        spec.pipelineClassifier.pipeline.models[-1].treeEnsembleClassifier.stringClassLabels.vector.extend(
            ["Benign", "Malignant"]
        )
    elif spec.WhichOneof('Type') == 'treeEnsembleClassifier':
        # Clear int64 labels and set string labels
        spec.treeEnsembleClassifier.ClearField('int64ClassLabels')
        spec.treeEnsembleClassifier.stringClassLabels.vector.extend(["Benign", "Malignant"])
        # Update output descriptions
        spec.description.output[0].type.stringType.CopyFrom(
            ct.proto.FeatureTypes_pb2.StringFeatureType()
        )

    mlmodel_classifier = ct.models.MLModel(spec)
    return mlmodel_classifier

# --- 3. GLUE THEM TOGETHER ---
def create_pipeline(backbone, classifier):
    print("--- 🔗 Creating Combined Pipeline ---")

    # Build pipeline spec manually to avoid protobuf serialization issues
    from coremltools.proto import Model_pb2

    # Create a pipeline spec
    pipeline_spec = Model_pb2.Model()
    pipeline_spec.specificationVersion = 5

    # Copy input from backbone
    pipeline_spec.description.input.extend(backbone.get_spec().description.input)

    # Copy output from classifier
    pipeline_spec.description.output.extend(classifier.get_spec().description.output)

    # Add models to pipeline
    pipeline_spec.pipeline.models.add().CopyFrom(backbone.get_spec())
    pipeline_spec.pipeline.models.add().CopyFrom(classifier.get_spec())

    # Convert to MLModel
    final_model = ct.models.MLModel(pipeline_spec)

    # Add metadata
    final_model.short_description = "Hybrid ResNet50 + Random Forest for Malignancy Detection"
    final_model.author = "DermaLite Team"

    final_model.save(FINAL_MLPACKAGE_PATH)
    print(f"✅ Hybrid Model Saved: {FINAL_MLPACKAGE_PATH}")

if __name__ == "__main__":
    backbone_ml = convert_backbone()
    classifier_ml = convert_classifier()
    create_pipeline(backbone_ml, classifier_ml)
    
    print("\n--- 📱 Integration Notes ---")
    print(f"1. Drag {FINAL_MLPACKAGE_PATH} into Xcode.")
    print("2. Input: 'input_image' (CVPixelBuffer).")
    print("3. Output: 'classLabel' and 'classProbability'.")