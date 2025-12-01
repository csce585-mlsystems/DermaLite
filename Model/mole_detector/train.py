import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm
import random

# --- CONFIGURATION ---
ISIC_PATH = "./data/ISIC2024_dataset"  
HAM_PATH = "./data/HAM10000_dataset"    

# Negative Datasets (Caltech Removed - Too easy/irrelevant)
NEG_PETS_PATH = "./data/oxford_pets" 
NEG_DTD_PATH = "./data/dtd"     
NEG_FLOWERS_PATH = "./data/flowers102" 

SAVE_MODEL_PATH = "mole_detector_robust.pth"

BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS = 8 
LEARNING_RATE = 0.001

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- HELPER: ROBUST IMAGE LOADING ---
def robust_load(img_path):
    """Loads image and forces RGB. Handles corrupt files."""
    try:
        with Image.open(img_path) as img:
            return img.convert("RGB")
    except Exception:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE))

# --- CUSTOM TRANSFORMS ---
class AddGaussianNoise(object):
    """Adds random noise to the image tensor to simulate phone camera grain."""
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# --- DATASET CLASSES ---
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
        return image, 1  # Label 1 for Mole

class NegativeWrapper(Dataset):
    """
    Generic Wrapper to force Label 0 (Not Mole).
    Now handles transforms internally to ensure RGB conversion happens BEFORE Tensor conversion.
    """
    def __init__(self, original_dataset, transform=None):
        self.dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Get raw data (Tuple of image, label)
        data = self.dataset[idx]
        
        if isinstance(data, tuple):
            img = data[0]
        else:
            img = data

        # 2. Force RGB (Must happen while it is still a PIL Image)
        if hasattr(img, 'convert'):
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
        # 3. Apply Transforms (RandomCrop, Blur, ToTensor, etc.)
        if self.transform:
            img = self.transform(img)
            
        return img, 0 # Label 0 for Not-Mole

# --- TRANSFORMS: THE "BRUTAL" ANTI-CHEAT PIPELINE ---
train_transforms = transforms.Compose([
    # 1. RandomResizedCrop: cuts off black corners/rulers
    transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.4, 1.0)), 
    
    # 2. GaussianBlur: Increased probability to 0.5
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
    
    # 3. RandomGrayscale: Forces model to learn structure, not just "Pink = Mole"
    transforms.RandomGrayscale(p=0.2),
    
    # 4. ColorJitter: Aggressive color changes (Hue/Sat bumped significantly)
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), 
    transforms.RandomRotation(90), 
    
    transforms.ToTensor(),
    
    # 5. Add Noise: Simulates sensor grain
    AddGaussianNoise(0., 0.05),
    
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_data():
    print("--- 🔄 Preparing Robust Data ---")
    
    # 1. POSITIVE: MOLES (ISIC + HAM)
    print("🔍 Scanning for Mole images...")
    isic_files = glob.glob(os.path.join(ISIC_PATH, "*", "*.jpg")) + \
                 glob.glob(os.path.join(ISIC_PATH, "*", "*.png"))
    ham_files = glob.glob(os.path.join(HAM_PATH, "*", "*.jpg")) + \
                glob.glob(os.path.join(HAM_PATH, "*", "*.png"))
    
    all_mole_files = isic_files + ham_files
    print(f"✅ Found {len(all_mole_files)} Mole images.")

    # 2. NEGATIVE: PETS, TEXTURES, FLOWERS (Removed Caltech)
    print("📦 Loading Negatives (Hard Mining Only)...")
    
    # Initialize datasets with transform=None so we get raw PIL images first
    pets = datasets.OxfordIIITPet(root=NEG_PETS_PATH, split='trainval', download=True, transform=None)
    dtd = datasets.DTD(root=NEG_DTD_PATH, split='train', download=True, transform=None)
    flowers = datasets.Flowers102(root=NEG_FLOWERS_PATH, split='train', download=True, transform=None)

    # Pass transforms to the Wrapper instead
    neg_pets = NegativeWrapper(pets, transform=train_transforms)
    neg_dtd = NegativeWrapper(dtd, transform=train_transforms)
    neg_flowers = NegativeWrapper(flowers, transform=train_transforms)
    
    combined_negatives = ConcatDataset([neg_pets, neg_dtd, neg_flowers])
    print(f"✅ Total 'Not Mole' Images: {len(combined_negatives)}")
    print(f"   (This roughly matches the positive count, allowing for excellent balance without junk data)")

    # 3. CREATE & BALANCE
    pos_dataset = MoleDataset(all_mole_files, transform=train_transforms)

    n_pos = len(pos_dataset)
    n_neg = len(combined_negatives)
    
    # Strict 50/50 balance
    target_size = min(n_pos, n_neg)
    print(f"⚖️  Balancing dataset to {target_size} images per class...")
    
    indices_pos = torch.randperm(n_pos)[:target_size]
    pos_subset = torch.utils.data.Subset(pos_dataset, indices_pos)
    
    indices_neg = torch.randperm(n_neg)[:target_size]
    neg_subset = torch.utils.data.Subset(combined_negatives, indices_neg)
    
    full_dataset = ConcatDataset([pos_subset, neg_subset])
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    return (DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
            DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2))

def train():
    train_loader, val_loader = get_data()

    print("--- 🧠 Initializing ResNet18 (Fresh Start) ---")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256), 
        nn.ReLU(),
        nn.Dropout(0.5), 
        nn.Linear(256, 1)
    )

    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"--- 🚀 Starting Robust Training for {EPOCHS} epochs ---")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1} Train Acc: {train_acc:.2f}%")

        validate(model, val_loader, criterion)

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"✅ Robust model saved to {SAVE_MODEL_PATH}")

def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    print(f"Validation Acc: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()