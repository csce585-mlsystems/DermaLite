import torch
import torch.nn as nn
from torchvision import models
import coremltools as ct
import os

# --- CONFIGURATION ---
CHECKPOINT_PATH = "./checkpoints_malignancy/malignancy_resnet50_focal.pth"
# RF_PATH and SCALER_PATH are now used externally, NOT in the CoreML model
OUTPUT_MODEL = "MalignancyResNet50Features.mlmodel"

# ImageNet Normalization
INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]

device = torch.device("cpu")

# --- ResNet Feature Extractor Class ---
class FeatureExtractor(nn.Module):
    """
    A standalone model for extracting features from ResNet50.
    This replaces the IntegratedHybridModel to ensure traceability.
    """
    def __init__(self, checkpoint_path):
        super().__init__()

        # Load ResNet backbone
        self.resnet = models.resnet50()
        
        # NOTE: Must re-create the classifier head structure to match the checkpoint,
        # but then set it to Identity to get the features BEFORE the final head.
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.resnet.load_state_dict(state_dict)
        
        # Set the final layer to Identity to get the features
        self.resnet.fc = nn.Identity()
        self.resnet.eval()

    def forward(self, x):
        features = self.resnet(x)
        # Flatten the features to get the vector required by the Scaler/RF
        features = torch.flatten(features, 1)  # [batch, 2048]
        return features


# Create and convert model
print("Creating ResNet Feature Extractor model...")
# The RF/Scaler logic is removed from this PyTorch model.
model = FeatureExtractor(CHECKPOINT_PATH)
model.eval()

# Trace the model
example_input = torch.rand(1, 3, 224, 224)
with torch.no_grad():
    # Only trace the feature extraction, which is now purely convolutional/linear
    traced_model = torch.jit.trace(model, example_input)

# Test it works
test_out = traced_model(example_input)
print(f"Test output shape: {test_out.shape}")
print(f"Test output (first 5 features): {test_out.detach().numpy()[:, :5]}")

# Convert to CoreML
print("\nConverting to CoreML...")

# Define the input type, including normalization parameters
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

# Define the output features tensor name
output_name = "extracted_features"

mlmodel = ct.convert(
    traced_model,
    inputs=[image_input],
    outputs=[ct.TensorType(name=output_name)],
    convert_to="neuralnetwork",
    # Classifier config removed as the model now only extracts features
)

mlmodel.save(OUTPUT_MODEL)
print(f"\n✅ Feature Extractor Model saved: {OUTPUT_MODEL}")
print("NOTE: The Scaler and RandomForest steps must be re-implemented in the application code after feature extraction.")