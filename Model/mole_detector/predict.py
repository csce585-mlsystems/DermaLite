import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import sys

# Load Architecture
def load_model(path):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )
    
    # Load weights map to CPU to ensure compatibility if moving files around
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        img_t = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_t)
            prob = torch.sigmoid(output).item()
            
        return prob
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    model = load_model("mole_detector_robust.pth")
    
    probability = predict(img_path, model)
    
    if probability is not None:
        print(f"\nAnalysis for: {img_path}")
        print(f"Mole Probability: {probability:.4f}")
        if probability > 0.5:
            print(">>> RESULT: MOLE DETECTED")
        else:
            print(">>> RESULT: NOT A MOLE")