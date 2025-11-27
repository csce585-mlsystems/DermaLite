import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import copy

# --- CONFIGURATION ---
DATASET_PATH = "./data/malignancy_dataset"
LOAD_CHECKPOINT = "./checkpoints_malignancy/checkpoint_epoch_5.pth" # Starting point
SAVE_PATH = "malignancy_resnet50_focal.pth"

BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS = 30 # Long run allowed
START_LR = 0.0001
PATIENCE = 3 # Epochs to wait before lowering LR

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- FOCAL LOSS CLASS ---
class FocalLoss(nn.Module):
    """
    Focal Loss focuses on hard examples.
    alpha: Balance factor (0.25 means we down-weight the majority class)
    gamma: Focusing parameter (2.0 is standard)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # Prevents nans
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# --- TRANSFORMS (Same Robust Pipeline) ---
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

train_transforms = transforms.Compose([
    # UPDATED: Lowered min scale to 0.3. 
    # This forces the model to look at "zoomed in" texture patches (simulating large moles),
    # preventing it from relying solely on borders which might be out of frame.
    transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.3, 1.0)), 
    
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

def get_loaders():
    print("--- 🔄 Preparing Autopilot Data ---")
    base_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=None)
    
    # Split
    dataset_size = len(base_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # VAL Data
    val_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=DATASET_PATH, transform=val_transforms), val_indices)
    
    # TRAIN Data (Oversampling Malignant)
    train_targets = [base_dataset.targets[i] for i in train_indices]
    malignant_indices = [i for i in train_indices if base_dataset.targets[i] == 1]
    benign_indices = [i for i in train_indices if base_dataset.targets[i] == 0]
    
    # 3x Oversample
    oversampled_mal = malignant_indices * 3
    final_train_indices = benign_indices + oversampled_mal
    random.shuffle(final_train_indices)
    
    train_dataset = torch.utils.data.Subset(datasets.ImageFolder(root=DATASET_PATH, transform=train_transforms), final_train_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def train_autopilot():
    train_loader, val_loader = get_loaders()
    
    print(f"--- ✈️  Engaging Autopilot: Loading {LOAD_CHECKPOINT} ---")
    
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1)
    )
    
    # Load Weights
    checkpoint = torch.load(LOAD_CHECKPOINT, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    
    # USE FOCAL LOSS
    criterion = FocalLoss(alpha=0.75, gamma=2.0) # alpha > 0.5 weights positive class higher
    
    optimizer = optim.AdamW(model.parameters(), lr=START_LR, weight_decay=1e-3)
    
    # Automated LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=PATIENCE
    )
    
    best_sensitivity = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"--- 🚀 Starting {EPOCHS} Epoch Run (Focal Loss) ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
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
        val_acc, sensitivity = validate(model, val_loader)
        
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | 🚨 Recall: {sensitivity:.2f}%")
        
        # Autopilot Logic: Optimize for SENSITIVITY
        scheduler.step(sensitivity)
        
        if sensitivity > best_sensitivity:
            best_sensitivity = sensitivity
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ New Best Model Saved (Recall: {sensitivity:.2f}%)")
            
    print("--- 🏁 Training Complete ---")

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    tp = 0
    fn = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            
    sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    acc = 100 * correct / total
    return acc, sensitivity

if __name__ == "__main__":
    train_autopilot()