"""
Model Benchmark Script
Tests a trained model and provides comprehensive performance metrics
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from torchvision import transforms, models
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from utils.dataset_loaders import HAM10000Dataset, ISIC2024Dataset, load_ham10000_data, load_isic2024_data
from utils import config


class ModelBenchmark:
    """Comprehensive model benchmarking class"""

    def __init__(self, checkpoint_path, device='auto'):
        """
        Initialize benchmark with trained model checkpoint

        Args:
            checkpoint_path: Path to saved model checkpoint (.pth file)
            device: Device to run on ('auto', 'mps', 'cuda', 'cpu')
        """
        # Determine device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load checkpoint
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract model info
        self.num_classes = self.checkpoint.get('num_classes', 2)
        self.class_names = self.checkpoint.get('class_names', [f'class_{i}' for i in range(self.num_classes)])
        self.config_info = self.checkpoint.get('config', {})

        print(f"Model info:")
        print(f"  Classes: {self.num_classes}")
        print(f"  Class names: {self.class_names}")
        print(f"  Architecture: {self.config_info.get('architecture', 'unknown')}")
        print(f"  Dataset: {self.config_info.get('dataset', 'unknown')}")

        # Build and load model
        self.model = self._build_model()

        # Storage for results
        self.results = {}

    def _build_model(self):
        """Rebuild model architecture and load weights"""
        arch = self.config_info.get('architecture', 'mobilenet_v2')

        # Build model based on architecture
        if arch == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, self.num_classes)
            )
        elif arch == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, self.num_classes)
            )
        elif arch == 'resnet50':
            model = models.resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, self.num_classes)
            )
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Load weights
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print("✓ Model loaded successfully\n")
        return model

    def load_test_data(self, dataset_name=None, data_fraction=1.0):
        """
        Load test dataset

        Args:
            dataset_name: 'HAM10000' or 'ISIC2024' (auto-detected from checkpoint if None)
            data_fraction: Fraction of data to use (1.0 = all, 0.1 = 10%)
        """
        if dataset_name is None:
            dataset_name = self.config_info.get('dataset', config.DATASET)

        print(f"Loading {dataset_name} test data...")

        # Get transforms (no augmentation for testing)
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load dataset
        if dataset_name == 'HAM10000':
            # Use multiclass config for HAM10000
            config.set_config('multiclass')
            dataset_config = config.get_active_config()
            df = load_ham10000_data(dataset_config['root_dir'])

            # Sample if requested
            if data_fraction < 1.0:
                df = df.groupby('dx', group_keys=False).apply(
                    lambda x: x.sample(frac=data_fraction, random_state=42)
                ).reset_index(drop=True)

            # Use test split
            _, test_df = train_test_split(
                df, stratify=df["dx"], test_size=0.2, random_state=42
            )

            test_ds = HAM10000Dataset(test_df, transform=test_transform)

        elif dataset_name == 'ISIC2024':
            # Use binary config for ISIC2024
            config.set_config('binary')
            dataset_config = config.get_active_config()
            df, image_dir = load_isic2024_data(
                dataset_config['root_dir'],
                dataset_config.get('metadata_file', 'train-metadata.csv')
            )

            stratify_col = 'target' if dataset_config['task'] == 'binary' else 'diagnosis'

            # Sample if requested
            if data_fraction < 1.0:
                df = df.groupby(stratify_col, group_keys=False).apply(
                    lambda x: x.sample(frac=data_fraction, random_state=42)
                ).reset_index(drop=True)

            # Use test split
            _, test_df = train_test_split(
                df, stratify=df[stratify_col], test_size=0.2, random_state=42
            )

            test_ds = ISIC2024Dataset(
                test_df, image_dir, transform=test_transform,
                task=dataset_config['task']
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Create data loader
        self.test_loader = DataLoader(
            test_ds, batch_size=32, shuffle=False, num_workers=4
        )

        self.test_dataset = test_ds

        print(f"✓ Loaded {len(test_ds)} test samples\n")

    def run_inference(self):
        """Run model inference on test data"""
        print("Running inference on test set...")

        all_preds = []
        all_labels = []
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(self.test_loader, desc="Testing"):
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        self.all_preds = np.array(all_preds)
        self.all_labels = np.array(all_labels)
        self.all_probs = np.array(all_probs)

        print("✓ Inference complete\n")

    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        print("="*70)
        print("BENCHMARK RESULTS")
        print("="*70)

        # Overall metrics
        accuracy = accuracy_score(self.all_labels, self.all_preds)

        # Multi-class metrics
        precision_macro = precision_score(self.all_labels, self.all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(self.all_labels, self.all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(self.all_labels, self.all_preds, average='macro', zero_division=0)

        # Weighted metrics (account for class imbalance)
        precision_weighted = precision_score(self.all_labels, self.all_preds, average='weighted', zero_division=0)
        recall_weighted = recall_score(self.all_labels, self.all_preds, average='weighted', zero_division=0)
        f1_weighted = f1_score(self.all_labels, self.all_preds, average='weighted', zero_division=0)

        # AUC
        try:
            if self.num_classes == 2:
                auc = roc_auc_score(self.all_labels, self.all_probs[:, 1])
            else:
                auc = roc_auc_score(self.all_labels, self.all_probs, multi_class='ovr', average='macro')
        except:
            auc = 0.0

        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'auc': auc
        }

        # Print overall metrics
        print("\n📊 OVERALL METRICS:")
        print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  AUC Score:          {auc:.4f}")
        print(f"\n  Macro Avg:")
        print(f"    Precision:        {precision_macro:.4f}")
        print(f"    Recall:           {recall_macro:.4f}")
        print(f"    F1-Score:         {f1_macro:.4f}")
        print(f"\n  Weighted Avg:")
        print(f"    Precision:        {precision_weighted:.4f}")
        print(f"    Recall:           {recall_weighted:.4f}")
        print(f"    F1-Score:         {f1_weighted:.4f}")

        # Per-class metrics
        print(f"\n📋 PER-CLASS METRICS:")
        for i, class_name in enumerate(self.class_names):
            class_mask = self.all_labels == i
            class_precision = precision_score(self.all_labels, self.all_preds, labels=[i], average=None, zero_division=0)[0]
            class_recall = recall_score(self.all_labels, self.all_preds, labels=[i], average=None, zero_division=0)[0]
            class_f1 = f1_score(self.all_labels, self.all_preds, labels=[i], average=None, zero_division=0)[0]
            class_support = class_mask.sum()

            print(f"\n  {class_name}:")
            print(f"    Precision:        {class_precision:.4f}")
            print(f"    Recall:           {class_recall:.4f}")
            print(f"    F1-Score:         {class_f1:.4f}")
            print(f"    Support:          {class_support} samples")

        # Classification report
        print(f"\n📄 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(
            self.all_labels, self.all_preds,
            target_names=self.class_names,
            digits=4
        ))

        return self.results

    def plot_confusion_matrix(self, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(self.all_labels, self.all_preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: {save_path}")
        plt.close()

    def plot_roc_curve(self, save_path='roc_curve.png'):
        """Plot ROC curve (for binary classification)"""
        if self.num_classes != 2:
            print("⚠ ROC curve plotting only supported for binary classification")
            return

        fpr, tpr, thresholds = roc_curve(self.all_labels, self.all_probs[:, 1])
        auc = roc_auc_score(self.all_labels, self.all_probs[:, 1])

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate (Recall)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {save_path}")
        plt.close()

    def save_results(self, save_path='benchmark_results.txt'):
        """Save benchmark results to text file"""
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL BENCHMARK RESULTS\n")
            f.write("="*70 + "\n\n")

            f.write(f"Model: {self.config_info.get('architecture', 'unknown')}\n")
            f.write(f"Dataset: {self.config_info.get('dataset', 'unknown')}\n")
            f.write(f"Classes: {self.class_names}\n")
            f.write(f"Test samples: {len(self.all_labels)}\n\n")

            f.write("OVERALL METRICS:\n")
            f.write(f"  Accuracy:     {self.results['accuracy']:.4f}\n")
            f.write(f"  AUC:          {self.results['auc']:.4f}\n")
            f.write(f"  Precision:    {self.results['precision_macro']:.4f} (macro)\n")
            f.write(f"  Recall:       {self.results['recall_macro']:.4f} (macro)\n")
            f.write(f"  F1-Score:     {self.results['f1_macro']:.4f} (macro)\n\n")

            f.write("CLASSIFICATION REPORT:\n")
            f.write(classification_report(
                self.all_labels, self.all_preds,
                target_names=self.class_names,
                digits=4
            ))

        print(f"✓ Results saved to: {save_path}")


def main():
    """Main benchmark function"""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark a trained skin lesion classification model')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Path to model checkpoint file')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to test on (HAM10000 or ISIC2024, auto-detected if not specified)')
    parser.add_argument('--fraction', type=float, default=1.0,
                        help='Fraction of data to use (1.0 = all, 0.1 = 10%%)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, mps, cuda, cpu)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize benchmark
    benchmark = ModelBenchmark(args.checkpoint, device=args.device)

    # Load test data
    benchmark.load_test_data(dataset_name=args.dataset, data_fraction=args.fraction)

    # Run inference
    benchmark.run_inference()

    # Calculate metrics
    benchmark.calculate_metrics()

    # Generate visualizations
    benchmark.plot_confusion_matrix(
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    if benchmark.num_classes == 2:
        benchmark.plot_roc_curve(
            save_path=os.path.join(args.output_dir, 'roc_curve.png')
        )

    # Save results
    benchmark.save_results(
        save_path=os.path.join(args.output_dir, 'benchmark_results.txt')
    )

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
