import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Custom dataset class with class balancing
class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform=None, oversample_minority=True):
        self.df = dataframe.dropna(subset=["image_path"])
        self.transform = transform
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df["dx"].unique()))}
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        
        # Oversample minority classes to balance dataset
        if oversample_minority:
            self.df = self._balance_classes()
    
    def _balance_classes(self):
        """Oversample minority classes to match the majority class"""
        class_counts = self.df["dx"].value_counts()
        max_count = class_counts.max()
        
        balanced_dfs = []
        for class_name in class_counts.index:
            class_df = self.df[self.df["dx"] == class_name]
            # Oversample to match max class
            if len(class_df) < max_count:
                oversampled = class_df.sample(n=max_count, replace=True, random_state=42)
                balanced_dfs.append(oversampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = self.label2idx[row["dx"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def calculate_class_weights(df, label2idx):
    """Calculate inverse frequency weights for each class"""
    class_counts = Counter(df["dx"])
    total_samples = len(df)
    num_classes = len(label2idx)
    
    weights = []
    for i in range(num_classes):
        label = [k for k, v in label2idx.items() if v == i][0]
        count = class_counts[label]
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def plot_confusion_matrix(y_true, y_pred, class_names, epoch):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_epoch_{epoch}.png')
    plt.close()


if __name__ == "__main__":
    
    ## Dataset preprocessing 
    
    # Paths
    root_dir = "/Users/t/Downloads/archive"
    part1 = os.path.join(root_dir, "HAM10000_images_part_1")
    part2 = os.path.join(root_dir, "HAM10000_images_part_2")
    meta_path = os.path.join(root_dir, "HAM10000_metadata.csv")
    
    # Read metadata
    df = pd.read_csv(meta_path)
    
    # Merge both image folders
    all_image_paths = {os.path.basename(x): os.path.join(part1, x)
                      for x in os.listdir(part1)}
    all_image_paths.update({os.path.basename(x): os.path.join(part2, x)
                            for x in os.listdir(part2)})
    
    # Add full path column
    df["image_path"] = df["image_id"].map(lambda x: all_image_paths.get(f"{x}.jpg"))
    
    # Print class distribution
    print("\n=== Original Class Distribution ===")
    print(df["dx"].value_counts())
    print()
    
    ## Data transformation and loaders
    
    # Split into sets
    train_df, val_df = train_test_split(df, stratify=df["dx"], test_size=0.2, random_state=42)
    
    # HEAVY augmentation for training to create more diverse samples
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])
    
    # Clean validation transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])
    
    # Training dataset WITH oversampling
    train_ds = HAM10000Dataset(train_df, transform=train_transform, oversample_minority=True)
    # Validation dataset WITHOUT oversampling
    val_ds = HAM10000Dataset(val_df, transform=val_transform, oversample_minority=False)
    
    train_labels = train_ds.df["dx"]
    print("=== Balanced Training Set Distribution ===")
    print(train_labels.value_counts())
    
    # DataLoaders for batch loading
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    ## MobileNetV2 Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    num_classes = len(train_ds.label2idx)
    
    # Load pre-trained model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)
    
    # Calculate class weights based on ORIGINAL (unbalanced) distribution
    class_weights = calculate_class_weights(train_df, train_ds.label2idx).to(device)
    print("=== Class Weights ===")
    for i, weight in enumerate(class_weights):
        print(f"{train_ds.idx2label[i]}: {weight:.3f}")
    print()
    
    # Use weighted loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=2)
    
    # Training loop
    num_epochs = 10
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        
        train_loss = running_loss / len(train_ds)
        history['train_loss'].append(train_loss)
        
        # --- Validation ---
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Acc = {val_acc:.4f}")
        
        # Detailed metrics every 2 epochs
        if (epoch + 1) % 2 == 0:
            class_names = [train_ds.idx2label[i] for i in range(num_classes)]
            print("\n=== Classification Report ===")
            print(classification_report(all_labels, all_preds, 
                                       target_names=class_names, 
                                       zero_division=0))
            plot_confusion_matrix(all_labels, all_preds, class_names, epoch+1)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "mobilenetv2_ham10000_balanced.pth")
            print(f"✓ Saved best model with accuracy: {best_acc:.4f}")
        
        # Adjust learning rate
        scheduler.step(val_acc)
        print()
    
    print(f"\n=== Training Complete ===")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training plots saved!")