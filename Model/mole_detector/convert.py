import torch
import coremltools as ct
from torchvision import models
import torch.nn as nn
import sys

# --- CONFIGURATION ---
# Path to your PyTorch .pth file (mole_detector_robust.pth)
PTH_PATH = "mole_detector_robust.pth"

# CHANGED: Switched to .mlmodel (Legacy Format) to fix Python 3.13 FileNotFoundError
MLMODEL_PATH = "mole_detector.mlmodel" 

# ImageNet Normalization values used in training (CRITICAL for matching accuracy)
INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]
IMAGE_SHAPE = [1, 3, 224, 224] # Batch size 1, 3 channels, 224x224
NUM_CLASSES = 2 

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- TRACING HELPER ---
class CoreMLWrapper(torch.nn.Module):
    """
    Wraps the ResNet model to ensure the output is a 0-1 probability 
    and simplifies the graph for Core ML tracing.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # We explicitly apply Sigmoid to the model's raw logit output
        # The output shape will be [1, 1], which is fine for binary classification
        return torch.sigmoid(self.model(x))

def build_and_load_model(pth_path, device):
    """
    Builds the ResNet18 architecture and loads the robust weights.
    """
    print("Building ResNet-18 architecture...")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    
    # CRITICAL: The classifier head must match the one defined in train_robust.py 
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256), 
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1) # Output is 1 logit for binary classification
    )
    
    print(f"Loading weights from {pth_path}...")
    try:
        model.load_state_dict(torch.load(pth_path, map_location=device))
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("Ensure 'mole_detector_robust.pth' is in the same directory.")
        sys.exit(1)

    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def convert_to_coreml(model):
    """
    Converts the PyTorch model to Core ML, applying normalization parameters.
    """
    # 1. Create example input tensor
    # Ensure the input tensor is moved to CPU for stable tracing
    example_input = torch.rand(IMAGE_SHAPE).cpu() 

    # 2. Wrap the model and trace it to include the Sigmoid operation
    # This CoreMLWrapper (renamed from SigmoidWrapper) provides a cleaner graph.
    wrapped_model = CoreMLWrapper(model.cpu())
    wrapped_model.eval() # Ensure dropout is disabled for tracing
    
    # Tracing must happen on the CPU
    traced_model = torch.jit.trace(wrapped_model, example_input)
    print("PyTorch model successfully traced to TorchScript.")
    
    # 3. Define the Core ML input specification with normalization
    # Note: Core ML expects the input to be in the range [0, 255] if getting CVPixelBuffer
    # But ImageType logic handles the math: (Input * Scale) + Bias
    image_input = ct.ImageType(
        name="input_image",
        shape=example_input.shape,
        # Normalization parameters: Core ML will perform Z = (Input_0_255 * Scale) + Bias
        # We want Z = (Input_0_1 - Mean) / Std
        # Since input from Vision/iOS comes as 0-255 usually, we rely on the standard ImageType scaling
        scale=1.0 / (255.0 * INPUT_STD[0]), 
        bias=[-INPUT_MEAN[0] / INPUT_STD[0], 
              -INPUT_MEAN[1] / INPUT_STD[1], 
              -INPUT_MEAN[2] / INPUT_STD[2]],
        channel_first=True, # PyTorch standard
        color_layout='RGB'
    )
    
    print("Starting Core ML conversion (this may take a minute)...")
    
    # 4. Convert the TRACED model to Core ML
    # CHANGED: 'neuralnetwork' is much more stable on Python 3.13 than 'mlprogram'
    mlmodel = ct.convert(
        traced_model,  
        inputs=[image_input],
        compute_units=ct.ComputeUnit.ALL,
        convert_to="neuralnetwork" 
    )

    # 5. Add custom metadata
    mlmodel.short_description = "Detects skin moles vs other objects."
    mlmodel.user_defined_metadata["model_type"] = "Dermatology Binary Classifier"
    mlmodel.user_defined_metadata["labels"] = "Not Mole, Mole"
    
    try:
        mlmodel.save(MLMODEL_PATH)
        print(f"✅ Core ML model saved at: {MLMODEL_PATH}")
        print_integration_plan()
    except Exception as e:
        print(f"❌ Failed to save model: {e}")

def print_integration_plan():
    print("\n" + "="*50)
    print("       📱 SWIFT INTEGRATION PLAN       ")
    print("="*50)
    print("The Gaussian Noise and Random Crop layers were removed from the")
    print("Core ML model to ensure stability. You must recreate them in Swift.")
    print("\n1. RESIZING & CROPPING:")
    print("   - Use Vision Framework (VNImageRequestHandler).")
    print("   - Set 'centerCrop' to true or manually crop to 224x224.")
    print("   - This removes the 'black corners' and rulers from the image.")
    print("\n2. GAUSSIAN NOISE (The Anti-Cheat):")
    print("   - The model expects 'noisy/grainy' images.")
    print("   - If the user's camera is too clean, the model might fail.")
    print("   - SWIFT ACTION: Add random noise to the CVPixelBuffer or")
    print("     MLMultiArray before prediction.")
    print("   - Logic: input[i] = input[i] + Random(-0.05, 0.05)")
    print("="*50 + "\n")

if __name__ == "__main__":
    
    # Load the PyTorch model
    pt_model = build_and_load_model(PTH_PATH, device)
    
    # Run the conversion
    convert_to_coreml(pt_model)