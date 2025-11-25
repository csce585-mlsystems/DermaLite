import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm
import sys

# --- CONFIGURATION ---
MODEL_PATH = "mole_detector_robust.pth"
ISIC_PATH = "./data/ISIC2024_dataset"  
HAM_PATH = "./data/HAM10000_dataset"    

# Updated Negative Paths
NEG_PETS_PATH = "./data/oxford_pets" 
NEG_DTD_PATH = "./data/dtd"     
NEG_FLOWERS_PATH = "./data/flowers102" 

BATCH_SIZE = 64
IMG_SIZE = 224
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- HELPER FUNCTIONS ---
def robust_load(img_path):
    try:
        with Image.open(img_path) as img:
            return img.convert("RGB")
    except Exception:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE))

class MoleDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = robust_load(img_path)
        if self.transform:
            image = self.transform(image)
        return image, 1

class NegativeWrapper(Dataset):
    def __init__(self, original_dataset, transform=None):
        self.dataset = original_dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data[0] if isinstance(data, tuple) else data
        if hasattr(img, 'convert') and img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0

# --- EVALUATION LOGIC ---
def get_test_loader():
    print("--- 🔍 Loading Evaluation Data ---")
    
    # 1. POSITIVE
    isic_files = glob.glob(os.path.join(ISIC_PATH, "*", "*.jpg")) + \
                 glob.glob(os.path.join(ISIC_PATH, "*", "*.png"))
    ham_files = glob.glob(os.path.join(HAM_PATH, "*", "*.jpg")) + \
                glob.glob(os.path.join(HAM_PATH, "*", "*.png"))
    all_mole_files = isic_files + ham_files

    # 2. NEGATIVE (Hard Mining Sets)
    # Note: We use None for transform here, passing it to wrapper later
    pets = datasets.OxfordIIITPet(root=NEG_PETS_PATH, split='trainval', download=True, transform=None)
    dtd = datasets.DTD(root=NEG_DTD_PATH, split='train', download=True, transform=None)
    flowers = datasets.Flowers102(root=NEG_FLOWERS_PATH, split='train', download=True, transform=None)

    # 3. TRANSFORMS (Simple Validation Transform)
    eval_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    neg_pets = NegativeWrapper(pets, transform=eval_transforms)
    neg_dtd = NegativeWrapper(dtd, transform=eval_transforms)
    neg_flowers = NegativeWrapper(flowers, transform=eval_transforms)
    combined_negatives = ConcatDataset([neg_pets, neg_dtd, neg_flowers])
    
    pos_dataset = MoleDataset(all_mole_files, transform=eval_transforms)

    # 4. BALANCE & SPLIT
    n_pos = len(pos_dataset)
    n_neg = len(combined_negatives)
    target_size = min(n_pos, n_neg)
    
    indices_pos = torch.randperm(n_pos)[:target_size]
    indices_neg = torch.randperm(n_neg)[:target_size]
    
    full_dataset = ConcatDataset([
        torch.utils.data.Subset(pos_dataset, indices_pos),
        torch.utils.data.Subset(combined_negatives, indices_neg)
    ])
    
    # Re-create the 80/20 split and grab the 20% validation set
    torch.manual_seed(42) # Must match train seed logic implies consistent splitting behavior
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"📉 Evaluation Set Size: {len(val_dataset)} images")
    return DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def load_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    
    # UPDATE: 256 Neurons
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def evaluate():
    loader = get_test_loader()
    model = load_model()
    
    y_true = []
    y_pred = []
    
    print("--- Running Inference ---")
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Not Mole", "Mole"], labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) 
    size_mb = get_model_size(model)
    
    print("\n" + "="*30)
    print("       MODEL REPORT       ")
    print("="*30)
    print(f"Model Size: {size_mb:.2f} MB")
    print(f"Accuracy:   {acc*100:.2f}%")
    print("\n--- Classification Report ---")
    print(report)
    print("\n--- Confusion Matrix ---")
    print(cm)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Mole", "Mole"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Mole Detector Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("\n✅ Confusion matrix saved to 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate()