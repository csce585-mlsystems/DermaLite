# export_coreml_with_features.py
import torch
import torch.nn as nn
from torchvision import models
import coremltools as ct
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pth", required=True, help="Path to mobilenetv2_ham10000_balanced.pth")
parser.add_argument("--out", default="MobileNetV2_CAM.mlpackage", help="Output Core ML package")
parser.add_argument("--weights-out", default="classifier_weights.npy", help="Classifier weights .npy")
parser.add_argument("--input-shape", default="1,3,224,224")
args = parser.parse_args()

pth_path = Path(args.pth)
assert pth_path.exists(), "Provide correct path to .pth file"

# 1) Build MobileNetV2 architecture and replace classifier to match training
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
num_classes = 7  # adjust if different
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))

# 2) Load weights (state_dict)
state = torch.load(str(pth_path), map_location="cpu")
# state could be a state_dict or a saved model object. Try state_dict first:
if isinstance(state, dict) and any(k.startswith("classifier") or k.startswith("features") for k in state.keys()):
    model.load_state_dict(state)
else:
    # if someone saved the whole model object
    try:
        model = state
    except Exception as e:
        raise RuntimeError("Couldn't load .pth as state_dict or model object: " + str(e))

model.eval()

# 3) Make a wrapper module that returns (logits, last_conv_features)
class WrapperModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        # For MobileNetV2 the last conv block is typically model.features[18]
        # We will capture the activation after the final feature layer.
        # If you want a different layer, change the index accordingly.
        self.target_idx = "18"

    def forward(self, x):
        # run through features but capture last conv activation
        activations = None
        out = x
        # model.features is an nn.Sequential of layers
        for name, layer in self.m.features.named_children():
            out = layer(out)
            if name == self.target_idx:
                activations = out  # shape [B, C, H, W]
        # global pooling & classifier (mirror torchvision mobilenet forward)
        out = out.mean([2,3])  # global avg pool (B,C)
        out = self.m.classifier(out)  # logits (B, num_classes)
        return out, activations

wrapped = WrapperModel(model)

# 4) Save classifier weights for CAM (num_classes x channels)
# classifier linear layer is model.classifier[1]; extract its weight
classifier_weight = model.classifier[1].weight.detach().cpu().numpy()  # shape [num_classes, channels]
np.save(args.weights_out, classifier_weight)
print("Saved classifier weights to", args.weights_out, classifier_weight.shape)

# 5) Trace / convert to Core ML using coremltools (mlprogram is preferred)
# Create an example input; coremltools likes ImageType or TensorType
example_input = torch.randn(tuple(int(x) for x in args.input_shape.split(",")))  # e.g. 1,3,224,224
traced = torch.jit.trace(wrapped.eval(), example_input)

# Convert to Core ML (mlprogram)
mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[ct.TensorType(name="input", shape=example_input.shape, dtype=np.float32)]
)

# The converted model will have two outputs: the logits and the activations.
# Save as .mlpackage (contains metadata and is Xcode-ready)
mlmodel.save(args.out)
print("Saved Core ML model to", args.out)
