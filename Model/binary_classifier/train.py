import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from PIL import Image
from tqdm import tqdm
import numpy as np
import sys
import random

# --- CONFIGURATION ---
DATASET_PATH = "./data/malignancy_dataset"
SAVE_MODEL_PATH = "malignancy_detector_resnet50.pth"
CHECKPOINT_DIR = "./checkpoints_malignancy" # NEW: Directory for checkpoints

BATCH_SIZE = 64 
IMG_SIZE = 224
EPOCHS = 15 
LEARNING_RATE = 0.0001 

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- ROBUST TRANSFORMS ---
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Transforms remain the same (anti-cheat pipeline)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.6, 1.0)), 
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.05),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- DATASET MANAGEMENT ---

def get_data_loaders():
    print("--- 🔄 Loading and Balancing Data (Oversampling) ---")
    
    # Load all data using a temporary transform to find indices
    base_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=None) 
    
    # Split indices first
    dataset_size = len(base_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # 1. VAL Data (Simple Subset with Val Transforms)
    val_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=DATASET_PATH, transform=val_transforms), val_indices)
    
    # 2. TRAIN Data (Oversample Malignant class)
    train_targets = [base_dataset.targets[i] for i in train_indices]
    malignant_indices = [i for i in train_indices if base_dataset.targets[i] == 1]
    benign_indices = [i for i in train_indices if base_dataset.targets[i] == 0]
    
    # Determine Oversampling Factor (e.g., duplicate malignant 3x)
    oversample_factor = 3
    
    # Oversampling: Repeat malignant indices multiple times
    oversampled_malignant_indices = malignant_indices * oversample_factor
    
    # Combine original benign indices with oversampled malignant indices
    final_train_indices = benign_indices + oversampled_malignant_indices
    random.shuffle(final_train_indices)
    
    # Create training dataset subset, ensuring to use the *training* transforms
    train_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=DATASET_PATH, transform=train_transforms), final_train_indices)

    print(f"   - Benign Samples in Training: {len(benign_indices)}")
    print(f"   - Malignant Samples (Oversampled {oversample_factor}x): {len(oversampled_malignant_indices)}")
    print(f"   - Total Training Images (Oversampled): {len(train_dataset)}")
    
    # CRITICAL FIX: Since we are oversampling data, we use standard DataLoader settings
    # and remove the WeightedRandomSampler
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, None # Loss weights are now handled by data

def train():
    
    # Ensure Checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    train_loader, val_loader, _ = get_data_loaders()

    print("--- 🧠 Initializing ResNet-50 ---")
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace Head
    num_ftrs = model.fc.in_features 
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1)
    )

    model = model.to(device)
    
    # CRITICAL FIX: Use simple BCEWithLogitsLoss (no pos_weight) since data is balanced by oversampling
    criterion = nn.BCEWithLogitsLoss() 
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    print(f"--- 🚀 Starting Malignancy Training for {EPOCHS} epochs ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # Validation
        val_loss, val_acc, sensitivity = validate(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1} Train Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | 🚨 Sensitivity (Recall): {sensitivity:.2f}%")

        # --- CHECKPOINT SAVING ---
        checkpoint_name = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'sensitivity': sensitivity
        }, checkpoint_name)
        print(f"Checkpoint saved to {checkpoint_name}")


    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"✅ Final Malignancy model saved to {SAVE_MODEL_PATH}")

def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    tp = 0 
    fn = 0 
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Sensitivity Tracking (Recall for Malignant class 1)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    
    sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    val_acc = 100 * correct / total
    return val_loss, val_acc, sensitivity

if __name__ == "__main__":
    # Ensure to restart the training run with this updated script.
    train()