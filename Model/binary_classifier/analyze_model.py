import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import shutil
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATASET_PATH = "./data/malignancy_dataset"
CHECKPOINT_PATH = "./checkpoints_malignancy/checkpoint_epoch_5.pth"
OUTPUT_DEBUG_DIR = "./debug_analysis"
IMG_SIZE = 224
BATCH_SIZE = 32

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- TRANSFORMS (Must match validation transforms) ---
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def unnormalize(tensor):
    """Helper to convert tensor back to image for saving"""
    tensor = tensor.clone().detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

def load_checkpoint():
    print(f"--- 📂 Loading Checkpoint: {CHECKPOINT_PATH} ---")
    model = models.resnet50()
    num_ftrs = model.fc.in_features 
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 1)
    )
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    # Handle dictionary vs direct state_dict saving
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def analyze():
    # 1. Load Data
    full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=val_transforms)
    
    # We want to analyze the VALIDATION set, so we replicate the split logic
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    # Note: To exactly match training, we need the seed. 
    # If not possible, analyzing the whole dataset is safer for debugging.
    # Let's analyze the WHOLE dataset to catch all edge cases.
    print(f"--- 🔍 Analyzing {len(full_dataset)} images ---")
    
    loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model = load_checkpoint()
    
    all_probs = []
    all_labels = []
    all_images = [] # Be careful with memory here, storing tensors
    
    # 2. Run Inference
    print("Running Inference...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            # We won't store images in RAM to prevent crash, we will re-iterate for saving
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 3. Threshold Optimization
    print("\n--- 🎚️ Threshold Optimization Report ---")
    print(f"{'Threshold':<10} | {'Recall (Sensitivity)':<20} | {'Specificity':<15} | {'Accuracy':<10}")
    print("-" * 65)
    
    best_thresh = 0.5
    best_f1 = 0
    target_thresh = 0.5 # The one closest to 85% recall
    min_dist_to_85 = 1.0
    
    for thresh in np.arange(0.05, 0.96, 0.05):
        preds = (all_probs > thresh).astype(int)
        
        tp = np.sum((preds == 1) & (all_labels == 1))
        fn = np.sum((preds == 0) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fp = np.sum((preds == 1) & (all_labels == 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tp + tn) / len(all_labels)
        
        # Identify "Scenario B" Candidate (approx 85% recall)
        dist = abs(recall - 0.85)
        if dist < min_dist_to_85:
            min_dist_to_85 = dist
            target_thresh = thresh
            
        print(f"{thresh:.2f}       | {recall*100:.2f}%               | {specificity*100:.2f}%          | {acc*100:.2f}%")

    print("-" * 65)
    print(f"👉 Recommended Threshold for Scenario B (~85% Recall): {target_thresh:.2f}")
    
    # 4. Save Missed Cancers (Visual Debugging)
    print(f"\n--- 📸 Saving False Negatives (Missed Cancers) using Threshold {target_thresh:.2f} ---")
    save_dir = os.path.join(OUTPUT_DEBUG_DIR, "false_negatives")
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    # Re-run loader to save images (memory efficient)
    cnt = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            labels = labels.cpu().numpy()
            
            for i in range(len(labels)):
                # If it IS Malignant (1) but we predicted Benign (prob < thresh)
                if labels[i] == 1 and probs[i] < target_thresh:
                    img = unnormalize(inputs[i])
                    conf_str = f"{probs[i]*100:.1f}%"
                    img.save(os.path.join(save_dir, f"missed_{cnt}_conf_{conf_str}.jpg"))
                    cnt += 1
                    if cnt >= 100: break # Limit to 100 images to save space
            if cnt >= 100: break
            
    print(f"✅ Saved {cnt} examples of missed cancers to {save_dir}")
    print("   Open these images to see WHAT the model is missing.")

if __name__ == "__main__":
    analyze()