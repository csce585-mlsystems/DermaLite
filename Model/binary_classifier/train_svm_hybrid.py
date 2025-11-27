import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import os

# --- CONFIGURATION ---
DATASET_PATH = "./data/malignancy_dataset"
# We use the checkpoint that actually saw features, even if unstable
CHECKPOINT_PATH = "./checkpoints_malignancy/malignancy_resnet50_focal.pth"
OUTPUT_DIR = "./hybrid_model"

BATCH_SIZE = 64
IMG_SIZE = 224

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- TRANSFORMS ---
# We use Val transforms because we want the "True" look of the image, 
# not a distorted/augmented version. The SVM needs consistent features.
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model_backbone():
    print(f"--- 🧠 Loading Backbone from {CHECKPOINT_PATH} ---")
    model = models.resnet50()
    # Recreate the head to load weights correctly
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1)
    )
    
    # Load weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    # CRITICAL: Remove the head (The "Brain")
    # We want the raw features (2048 numbers) coming out of the Average Pooling layer
    # ResNet structure: ... -> avgpool -> fc. We replace fc with Identity.
    model.fc = nn.Identity()
    
    model.to(device)
    model.eval() # Freeze completely
    return model

def extract_features(model, dataloader):
    features_list = []
    labels_list = []
    
    print("--- 🧬 Extracting Features (Embeddings) ---")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            # Forward pass gives us (Batch, 2048) vectors
            outputs = model(inputs)
            # Move to CPU numpy
            features_list.append(outputs.cpu().numpy())
            labels_list.append(labels.numpy())
            
    # Flatten
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Prepare Data
    print("Loading Data...")
    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=data_transforms)
    
    # We manually handle the split to ensure we train/test on different data
    # (Scikit-learn has its own splitters, but we can reuse indices if needed)
    # For simplicity, we'll extract ALL data, then use sklearn's train_test_split
    # so we can rapidly iterate on the ratio.
    
    loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Extract Features using ResNet
    backbone = get_model_backbone()
    X, y = extract_features(backbone, loader)
    
    print(f"Feature Shape: {X.shape} (Images x Features)")
    print(f"Labels Shape: {y.shape}")
    
    # 3. Standardize Features (Crucial for SVM)
    print("Normalizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 4. Split Data (Using sklearn is faster/easier here)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Set: {X_train.shape[0]} samples")
    print(f"Testing Set:  {X_test.shape[0]} samples")
    
    # --- MODEL 1: SVM (The "Geometric" Solver) ---
    print("\n--- 🤖 Training SVM Classifier ---")
    # class_weight='balanced' automatically handles the 10:1 imbalance
    # probability=True allows us to get confidence scores later
    #svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42)
    #svm.fit(X_train, y_train)
    
    #print("Evaluating SVM...")
    #y_pred = svm.predict(X_test)
    #print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
    #print("Confusion Matrix:")
    #print(confusion_matrix(y_test, y_pred))
    
    # --- MODEL 2: Random Forest (The "Decision Tree" Solver) ---
    print("\n--- 🌲 Training Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    print("Evaluating Random Forest...")
    y_pred_rf = rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf, target_names=["Benign", "Malignant"]))
    
    # 5. Save the best one
    # You can pick based on the output. Usually SVM has better Recall.
    #joblib.dump(svm, os.path.join(OUTPUT_DIR, "svm_head.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(rf, os.path.join(OUTPUT_DIR, 'rf_head.pkl'))
    print(f"\n✅ Saved SVM model and Scaler to {OUTPUT_DIR}")
    print("   (To use this: Load ResNet -> Extract Features -> Pass to SVM)")

if __name__ == "__main__":
    main()