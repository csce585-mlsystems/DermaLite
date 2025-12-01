import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib
import sys
import os

# --- CONFIGURATION ---
# Paths
CHECKPOINT_PATH = "./checkpoints_malignancy/malignancy_resnet50_focal.pth"
SVM_PATH = "./hybrid_model/svm_head.pkl"
SCALER_PATH = "./hybrid_model/scaler.pkl"
IMG_SIZE = 224

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_backbone():
    """Loads the frozen ResNet-50 feature extractor"""
    print(f"Loading Backbone from {CHECKPOINT_PATH}...")
    model = models.resnet50()
    # Recreate the head structure just to load weights
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1)
    )
    
    # Load weights
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)
        
    # Lobotomy: Remove the head to get raw features
    model.fc = nn.Identity()
    
    model.to(device)
    model.eval()
    return model

def load_ml_components():
    """Loads the SVM and Scaler"""
    print("Loading SVM and Scaler...")
    try:
        svm = joblib.load(SVM_PATH)
        scaler = joblib.load(SCALER_PATH)
        return svm, scaler
    except FileNotFoundError:
        print(f"❌ Error: SVM/Scaler not found in ./hybrid_model/")
        print("   Did you run 'train_svm_hybrid.py'?")
        sys.exit(1)

def predict(image_path, backbone, svm, scaler):
    # Validation Transforms (No augmentation)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # 1. Image -> Tensor
        image = Image.open(image_path).convert("RGB")
        img_t = transform(image).unsqueeze(0).to(device)
        
        # 2. Tensor -> Features (ResNet)
        with torch.no_grad():
            features = backbone(img_t).cpu().numpy()
            
        # 3. Features -> Normalized Features (Scaler)
        features_norm = scaler.transform(features)
        
        # 4. Features -> Prediction (SVM)
        # SVM classes are [0, 1] usually. 1 is Malignant.
        # probability=True in training allows predict_proba
        probs = svm.predict_proba(features_norm)[0]
        malignancy_prob = probs[1] # Probability of Class 1
        
        return malignancy_prob

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_hybrid.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    
    # Load components once
    backbone = get_backbone()
    svm, scaler = load_ml_components()
    
    print(f"\nAnalyzing: {img_path} ...")
    risk = predict(img_path, backbone, svm, scaler)
    
    if risk is not None:
        print("\n" + "="*40)
        print(f"🔬 MALIGNANCY RISK: {risk*100:.2f}%")
        print("="*40)
        
        if risk > 0.5:
            print("🚨 RESULT: MALIGNANT (High Risk)")
            print("   Action: Recommended to see a dermatologist.")
        elif risk > 0.2:
            print("⚠️ RESULT: SUSPICIOUS (Moderate Risk)")
            print("   Action: Monitor for changes (ABCD).")
        else:
            print("✅ RESULT: BENIGN (Low Risk)")