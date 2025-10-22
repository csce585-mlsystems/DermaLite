"""
Unified training script with support for both HAM10000 and ISIC 2024 datasets
Supports both binary (malignant detection) and multiclass (lesion diagnosis) tasks
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms, models
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from utils.dataset_loaders import HAM10000Dataset, ISIC2024Dataset, load_ham10000_data, load_isic2024_data
from utils import config

def get_transforms():
    """Create train and validation transforms based on config"""
    train_transform = transforms.Compose([
        transforms.Resize((config.MODEL_CONFIG['input_size'], config.MODEL_CONFIG['input_size'])),
        transforms.RandomHorizontalFlip(p=config.AUGMENTATION_CONFIG['horizontal_flip_prob']),
        transforms.RandomVerticalFlip(p=config.AUGMENTATION_CONFIG['vertical_flip_prob']),
        transforms.RandomRotation(config.AUGMENTATION_CONFIG['rotation_degrees']),
        transforms.ColorJitter(
            brightness=config.AUGMENTATION_CONFIG['color_jitter']['brightness'],
            contrast=config.AUGMENTATION_CONFIG['color_jitter']['contrast'],
            saturation=config.AUGMENTATION_CONFIG['color_jitter']['saturation'],
            hue=config.AUGMENTATION_CONFIG['color_jitter']['hue']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.AUGMENTATION_CONFIG['normalize_mean'],
            std=config.AUGMENTATION_CONFIG['normalize_std']
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.MODEL_CONFIG['input_size'], config.MODEL_CONFIG['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.AUGMENTATION_CONFIG['normalize_mean'],
            std=config.AUGMENTATION_CONFIG['normalize_std']
        )
    ])

    return train_transform, val_transform


def load_dataset():
    """Load dataset based on configuration"""
    dataset_config = config.get_active_config()

    if config.DATASET == 'HAM10000':
        print(f"\n{'='*70}")
        print("Loading HAM10000 Dataset")
        print(f"{'='*70}")

        # Load data
        df = load_ham10000_data(dataset_config['root_dir'])

        # Apply data sampling if enabled (for quick testing)
        if config.TRAINING_CONFIG['use_sampling']:
            sample_frac = config.TRAINING_CONFIG['sample_fraction']
            sampling_strategy = config.TRAINING_CONFIG['sampling_strategy']
            original_size = len(df)

            if sampling_strategy == 'balanced':
                # Balanced sampling: equal samples per class
                total_target_samples = int(original_size * sample_frac)
                num_classes = df['dx'].nunique()
                samples_per_class = total_target_samples // num_classes

                df = df.groupby('dx', group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), samples_per_class), random_state=config.TRAINING_CONFIG['random_state'])
                ).reset_index(drop=True)

                print(f"\n⚡ BALANCED SAMPLING ENABLED: {samples_per_class} samples per class")
                print(f"   Strategy: Each class gets equal representation")
                print(f"   Original: {original_size} → Sampled: {len(df)} images (balanced across {num_classes} classes)")
                print(f"   Estimated training time: ~{len(df)*0.0001*config.TRAINING_CONFIG['num_epochs']:.1f}-{len(df)*0.00015*config.TRAINING_CONFIG['num_epochs']:.1f} hours\n")

            elif sampling_strategy == 'proportional':
                # Proportional sampling: maintain original class distribution
                df = df.groupby('dx', group_keys=False).apply(
                    lambda x: x.sample(frac=sample_frac, random_state=config.TRAINING_CONFIG['random_state'])
                ).reset_index(drop=True)

                print(f"\n⚡ PROPORTIONAL SAMPLING ENABLED: Using {sample_frac*100:.1f}% of data")
                print(f"   Strategy: Maintains original class distribution")
                print(f"   Original: {original_size} → Sampled: {len(df)} images")
                print(f"   Estimated training time: ~{len(df)*0.0001*config.TRAINING_CONFIG['num_epochs']:.1f}-{len(df)*0.00015*config.TRAINING_CONFIG['num_epochs']:.1f} hours\n")

            else:
                raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}. Use 'balanced' or 'proportional'")

        # Split dataset
        train_df, val_df = train_test_split(
            df,
            stratify=df["dx"],
            test_size=config.TRAINING_CONFIG['test_size'],
            random_state=config.TRAINING_CONFIG['random_state']
        )

        # Get transforms
        train_transform, val_transform = get_transforms()

        # Create datasets
        train_ds = HAM10000Dataset(train_df, transform=train_transform)
        val_ds = HAM10000Dataset(val_df, transform=val_transform)

        # Get number of classes
        num_classes = len(train_ds.label2idx)
        class_names = [train_ds.idx2label[i] for i in range(num_classes)]

        return train_ds, val_ds, num_classes, class_names, train_df

    elif config.DATASET == 'ISIC2024':
        print(f"\n{'='*70}")
        print("Loading ISIC 2024 Dataset")
        print(f"{'='*70}")

        # Load data
        df, image_dir = load_isic2024_data(
            dataset_config['root_dir'],
            dataset_config.get('metadata_file', 'train-metadata.csv')
        )

        # Determine stratification column
        if dataset_config['task'] == 'binary':
            stratify_col = 'target'
        else:
            stratify_col = 'diagnosis'

        # Apply data sampling if enabled (for quick testing)
        if config.TRAINING_CONFIG['use_sampling']:
            sample_frac = config.TRAINING_CONFIG['sample_fraction']
            sampling_strategy = config.TRAINING_CONFIG['sampling_strategy']
            original_size = len(df)

            if sampling_strategy == 'balanced':
                # Balanced sampling: equal samples per class
                total_target_samples = int(original_size * sample_frac)
                num_classes = df[stratify_col].nunique()
                samples_per_class = total_target_samples // num_classes

                df = df.groupby(stratify_col, group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), samples_per_class), random_state=config.TRAINING_CONFIG['random_state'])
                ).reset_index(drop=True)

                print(f"\n⚡ BALANCED SAMPLING ENABLED: {samples_per_class} samples per class")
                print(f"   Strategy: Each class gets equal representation")
                print(f"   Original: {original_size} → Sampled: {len(df)} images (balanced across {num_classes} classes)")
                print(f"   Estimated training time: ~{len(df)*0.0001*config.TRAINING_CONFIG['num_epochs']:.1f}-{len(df)*0.00015*config.TRAINING_CONFIG['num_epochs']:.1f} hours\n")

            elif sampling_strategy == 'proportional':
                # Proportional sampling: maintain original class distribution
                df = df.groupby(stratify_col, group_keys=False).apply(
                    lambda x: x.sample(frac=sample_frac, random_state=config.TRAINING_CONFIG['random_state'])
                ).reset_index(drop=True)

                print(f"\n⚡ PROPORTIONAL SAMPLING ENABLED: Using {sample_frac*100:.1f}% of data")
                print(f"   Strategy: Maintains original class distribution")
                print(f"   Original: {original_size} → Sampled: {len(df)} images")
                print(f"   Estimated training time: ~{len(df)*0.0001*config.TRAINING_CONFIG['num_epochs']:.1f}-{len(df)*0.00015*config.TRAINING_CONFIG['num_epochs']:.1f} hours\n")

            else:
                raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}. Use 'balanced' or 'proportional'")

        # Split dataset
        train_df, val_df = train_test_split(
            df,
            stratify=df[stratify_col],
            test_size=config.TRAINING_CONFIG['test_size'],
            random_state=config.TRAINING_CONFIG['random_state']
        )

        # Get transforms
        train_transform, val_transform = get_transforms()

        # Create datasets
        train_ds = ISIC2024Dataset(
            train_df,
            image_dir,
            transform=train_transform,
            task=dataset_config['task']
        )
        val_ds = ISIC2024Dataset(
            val_df,
            image_dir,
            transform=val_transform,
            task=dataset_config['task']
        )

        # Get number of classes
        num_classes = len(train_ds.label2idx)
        class_names = [train_ds.idx2label[i] for i in range(num_classes)]

        return train_ds, val_ds, num_classes, class_names, train_df

    else:
        raise ValueError(f"Unknown dataset: {config.DATASET}")


def compute_class_weights_from_dataset(train_df, train_ds, dataset_name):
    """Compute class weights for imbalanced datasets"""
    if dataset_name == 'HAM10000':
        label_col = 'dx'
    elif config.get_active_config()['task'] == 'binary':
        label_col = 'target'
    else:
        label_col = 'diagnosis'

    # Get labels
    labels = train_df[label_col].map(train_ds.label2idx).values

    # Compute balanced weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(range(len(train_ds.label2idx))),
        y=labels
    )

    return torch.FloatTensor(class_weights)


def get_model(num_classes, device):
    """Create model based on configuration"""
    arch = config.MODEL_CONFIG['architecture']

    if arch == 'mobilenet_v2':
        if config.MODEL_CONFIG['pretrained']:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            model = models.mobilenet_v2(weights=None)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(config.MODEL_CONFIG['dropout']),
            nn.Linear(in_features, num_classes)
        )

    elif arch == 'efficientnet_b0':
        if config.MODEL_CONFIG['pretrained']:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(config.MODEL_CONFIG['dropout']),
            nn.Linear(in_features, num_classes)
        )

    elif arch == 'resnet50':
        if config.MODEL_CONFIG['pretrained']:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(config.MODEL_CONFIG['dropout']),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model.to(device)


def train_model():
    """Main training function"""

    # Print configuration
    config.print_config()

    # Determine device
    if config.DEVICE == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.DEVICE)

    print(f"\nUsing device: {device}\n")

    # Load dataset
    train_ds, val_ds, num_classes, class_names, train_df = load_dataset()

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=config.TRAINING_CONFIG['num_workers']
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config.TRAINING_CONFIG['num_workers']
    )

    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}\n")

    # Compute class weights
    if config.TRAINING_CONFIG['use_class_weights']:
        if config.TRAINING_CONFIG.get('use_manual_weights', False):
            # Use manual class weights for aggressive weighting
            manual_weights = config.TRAINING_CONFIG['manual_class_weights']
            if len(manual_weights) != num_classes:
                raise ValueError(f"Manual weights length {len(manual_weights)} doesn't match num_classes {num_classes}")
            class_weights = torch.FloatTensor(manual_weights)
            print("Using MANUAL class weights (aggressive for medical diagnosis)...")
            print(f"Class weights: {class_weights}")
            for i, (name, weight) in enumerate(zip(class_names, manual_weights)):
                print(f"  {name}: {weight}x weight")
            print("(Malignant class heavily weighted to avoid false negatives)\n")
        else:
            # Auto-compute balanced weights
            print("Computing class weights for imbalanced dataset...")
            class_weights = compute_class_weights_from_dataset(train_df, train_ds, config.DATASET)
            print(f"Class weights: {class_weights}")
            print("(Higher weights = rarer classes get more importance)\n")
    else:
        class_weights = None

    # Create model
    print(f"Creating {config.MODEL_CONFIG['architecture']} model...")
    model = get_model(num_classes, device)
    print(f"Model created successfully!\n")

    # Loss function
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.TRAINING_CONFIG['learning_rate']
    )

    # Learning rate scheduler
    if config.TRAINING_CONFIG['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.TRAINING_CONFIG['scheduler_factor'],
            patience=config.TRAINING_CONFIG['scheduler_patience']
        )

    # Training loop
    num_epochs = config.TRAINING_CONFIG['num_epochs']
    best_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"{'='*70}")
    print(f"Starting Training - {num_epochs} epochs")
    print(f"{'='*70}\n")

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
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * imgs.size(0)

                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Store for detailed metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_acc = correct / total
        val_loss = val_running_loss / len(val_ds)

        # Compute AUC score
        try:
            if num_classes == 2:
                # Binary classification: use probability of positive class
                val_auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
            else:
                # Multi-class: use one-vs-rest
                val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except:
            val_auc = 0.0

        # Calculate per-class metrics for binary classification (important for medical diagnosis)
        if num_classes == 2:
            from sklearn.metrics import precision_score, recall_score, f1_score
            all_preds_np = np.array(all_preds)
            all_labels_np = np.array(all_labels)

            # Metrics for malignant class (class 1) - most important!
            malignant_precision = precision_score(all_labels_np, all_preds_np, pos_label=1, zero_division=0)
            malignant_recall = recall_score(all_labels_np, all_preds_np, pos_label=1, zero_division=0)
            malignant_f1 = f1_score(all_labels_np, all_preds_np, pos_label=1, zero_division=0)

            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.4f} | Val AUC = {val_auc:.4f}")
            print(f"  → Malignant: Precision={malignant_precision:.4f} | Recall={malignant_recall:.4f} | F1={malignant_f1:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.4f} | Val AUC = {val_auc:.4f}")

        # Step scheduler
        if config.TRAINING_CONFIG['use_scheduler']:
            scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'class_names': class_names,
                'num_classes': num_classes,
                'config': {
                    'dataset': config.DATASET,
                    'architecture': config.MODEL_CONFIG['architecture']
                }
            }

            save_path = os.path.join(
                config.TRAINING_CONFIG['save_dir'],
                config.TRAINING_CONFIG['checkpoint_name']
            )
            torch.save(checkpoint, save_path)
            print(f"  → Best model saved! (Val Loss improved to {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement for {patience_counter} epoch(s)")

        # Early stopping
        if config.TRAINING_CONFIG['use_early_stopping']:
            if patience_counter >= config.TRAINING_CONFIG['early_stop_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}\n")

    # Final evaluation
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(
        config.TRAINING_CONFIG['save_dir'],
        config.TRAINING_CONFIG['checkpoint_name']
    ))
    model.load_state_dict(checkpoint['model_state_dict'])

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

    # Print classification report
    print(f"\n{'='*70}")
    print("Final Classification Report")
    print(f"{'='*70}")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    ))

    # Print confusion matrix
    print(f"\n{'='*70}")
    print("Confusion Matrix")
    print(f"{'='*70}")
    cm = confusion_matrix(all_labels, all_preds)
    print("Rows=Actual, Columns=Predicted")
    print(cm)
    print(f"\nClass labels: {class_names}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DermaLite skin lesion classifier')
    parser.add_argument(
        '--task',
        type=str,
        choices=['binary', 'multiclass'],
        default='binary',
        help='Task type: binary (malignant detection) or multiclass (lesion diagnosis)'
    )
    args = parser.parse_args()

    # Set configuration based on task
    config.set_config(args.task)

    # Run training
    train_model()
