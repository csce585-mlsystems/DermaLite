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
import cv2
# custom dataset class 

class HAM10000Dataset(Dataset):
  def __init__(self,dataframe, transform= None):
    self.df = dataframe.dropna(subset=["image_path"])
    self.transform = transform
    self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df["dx"].unique()))}
    self.idx2label = {v: k for k, v in self.label2idx.items()}

  def __len__(self):
    return len(self.df)
    
  def __getitem__(self,idx):
    row = self.df.iloc[idx]
    img = Image.open(row["image_path"]).convert("RGB")
    label = self.label2idx[row["dx"]]
    if self.transform:
      img = self.transform(img)
    return img, label



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
  print(df.head())
    
  ## data transformation and loaders

  # split into sets
  train_df, val_df = train_test_split(df, stratify= df["dx"], test_size = 0.2, random_state = 42)

  # transformations that generate slightly different version of the same image
  # Goal: to make the model generalize better and prevent overfitting
  train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # conversion to PIL (Python Image Library) to PyTorch tensor 
    transforms.Normalize(mean= [.485, .456, .406], std = [.229, .224, .225]) # ImageNet normalizaton 
  ])

  # evaluate model accurately without randomness 
  val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485,.456,.406], std = [.229,.224, .225])
  ])

  # Training / Validating Datasets
  train_ds = HAM10000Dataset(train_df, transform=train_transform)
  val_ds = HAM10000Dataset(val_df, transform=val_transform)

  #DataLoaders for batch loading 
  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    
  ## MobileNetV2 Model 
  # uses MPS (Metal Performance Shaders) Backend for GPU training acceleration 
  # if not compatible, falls back on standard cpu usage 
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

  # should be 7 classes for the skin lesions
  num_classes = len(train_ds.label2idx)

  # loading pre-trained model for transfer learning
  # not sensible to rebuild from scratch because of a small dataset 
  # If built from scratch: overfitting of the data 
  model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

  # replaces original classifier from mobilenet
  in_features = model.classifier[1].in_features
  model.classifier = nn.Sequential(nn.Dropout(0.2),
      nn.Linear(in_features, num_classes)
  )
  model = model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-4)

  # training loop 
  num_epochs = 5
  best_acc = 0.0

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
      
      # --- Validation ---
      model.eval()
      correct = 0
      total = 0
      with torch.no_grad():
          for imgs, labels in val_loader:
              imgs, labels = imgs.to(device), labels.to(device)
              outputs = model(imgs)
              _, preds = torch.max(outputs, 1)
              correct += (preds == labels).sum().item()
              total += labels.size(0)
      val_acc = correct / total
      
      print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Acc = {val_acc:.4f}")
      
      # Save best model
      if val_acc > best_acc:
          best_acc = val_acc
          torch.save(model.state_dict(), "mobilenetv2_ham10000.pth")
    

  print(f"Training complete. Best Val Accuracy: {best_acc:.4f}")

