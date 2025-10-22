"""
Dataset loaders for skin lesion classification
Supports: HAM10000 and ISIC 2024 datasets
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class HAM10000Dataset(Dataset):
    """
    HAM10000 Skin Lesion Dataset Loader
    - 10,015 dermoscopic images
    - 7 classes of skin lesions
    - Highly imbalanced dataset
    """
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.dropna(subset=["image_path"])
        self.transform = transform
        # Create label mappings (sorted for consistency)
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df["dx"].unique()))}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = self.label2idx[row["dx"]]
        if self.transform:
            img = self.transform(img)
        return img, label


class ISIC2024Dataset(Dataset):
    """
    ISIC 2024 (SLICE-3D) Dataset Loader
    - 400,000+ skin lesion image crops from 3D TBP
    - Binary classification: malignant vs benign
    - More balanced than HAM10000
    - Can also support multi-class if diagnosis column is used
    """
    def __init__(self, dataframe, image_dir, transform=None, task='binary'):
        """
        Args:
            dataframe: pandas DataFrame with metadata
            image_dir: directory containing image files
            transform: torchvision transforms
            task: 'binary' for malignant/benign or 'multiclass' for specific diagnoses
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.task = task

        # Determine label column and create mappings
        if task == 'binary':
            # Binary classification: malignant (1) vs benign (0)
            # Assumes 'target' column exists with 0/1 labels
            if 'target' in self.df.columns:
                self.label_col = 'target'
            else:
                raise ValueError("Binary task requires 'target' column in metadata")

            self.label2idx = {0: 0, 1: 1}
            self.idx2label = {0: 'benign', 1: 'malignant'}

        elif task == 'multiclass':
            # Multi-class classification using diagnosis labels
            # Uses diagnosis or mel_class column if available
            if 'diagnosis' in self.df.columns:
                self.label_col = 'diagnosis'
                unique_labels = sorted(self.df[self.label_col].dropna().unique())
                self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
                self.idx2label = {v: k for k, v in self.label2idx.items()}
            else:
                raise ValueError("Multiclass task requires 'diagnosis' column in metadata")

        # Drop rows with missing labels
        self.df = self.df.dropna(subset=[self.label_col])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct image path
        # ISIC 2024 images are typically named: {isic_id}.jpg or stored in subdirs
        if 'isic_id' in row:
            img_filename = f"{row['isic_id']}.jpg"
        elif 'filename' in row:
            img_filename = row['filename']
        else:
            # Fallback: assume first column is image ID
            img_filename = f"{row.iloc[0]}.jpg"

        img_path = os.path.join(self.image_dir, img_filename)

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Get label
        if self.task == 'binary':
            label = int(row[self.label_col])
        else:
            label = self.label2idx[row[self.label_col]]

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label


def load_ham10000_data(root_dir):
    """
    Load HAM10000 dataset metadata and create image path mappings

    Args:
        root_dir: Path to HAM10000 dataset directory containing:
            - HAM10000_images_part_1/
            - HAM10000_images_part_2/
            - HAM10000_metadata.csv

    Returns:
        pandas DataFrame with image paths
    """
    part1 = os.path.join(root_dir, "HAM10000_images_part_1")
    part2 = os.path.join(root_dir, "HAM10000_images_part_2")
    meta_path = os.path.join(root_dir, "HAM10000_metadata.csv")

    # Verify paths exist
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # Read metadata
    df = pd.read_csv(meta_path)

    # Merge both image folders
    all_image_paths = {}
    if os.path.exists(part1):
        all_image_paths.update({os.path.basename(x): os.path.join(part1, x)
                                for x in os.listdir(part1) if x.endswith('.jpg')})
    if os.path.exists(part2):
        all_image_paths.update({os.path.basename(x): os.path.join(part2, x)
                                for x in os.listdir(part2) if x.endswith('.jpg')})

    # Add full path column
    df["image_path"] = df["image_id"].map(lambda x: all_image_paths.get(f"{x}.jpg"))

    print(f"Loaded HAM10000: {len(df)} samples")
    print(f"Class distribution:\n{df['dx'].value_counts()}\n")

    return df


def load_isic2024_data(root_dir, metadata_file='train-metadata.csv'):
    """
    Load ISIC 2024 dataset metadata

    Args:
        root_dir: Path to ISIC2024 dataset directory containing:
            - train-image/ or images/
            - train-metadata.csv
        metadata_file: Name of the metadata CSV file

    Returns:
        tuple: (pandas DataFrame with metadata, image directory path)
    """
    meta_path = os.path.join(root_dir, metadata_file)

    # Find image directory (could be train-image/, images/, or train/)
    possible_img_dirs = ['train-image', 'images', 'train', 'train-images']
    image_dir = None
    for dirname in possible_img_dirs:
        potential_path = os.path.join(root_dir, dirname)
        if os.path.exists(potential_path):
            image_dir = potential_path
            break

    if image_dir is None:
        raise FileNotFoundError(f"Image directory not found in {root_dir}. Looked for: {possible_img_dirs}")

    # Check if images are in a nested 'image' subdirectory
    # Some ISIC datasets have structure: train-image/image/*.jpg instead of train-image/*.jpg
    if os.path.exists(image_dir):
        # Check if current directory has .jpg files
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # If no images found, check for nested 'image' subdirectory
        if len(image_files) == 0:
            nested_image_dir = os.path.join(image_dir, 'image')
            if os.path.exists(nested_image_dir):
                nested_files = [f for f in os.listdir(nested_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if len(nested_files) > 0:
                    print(f"Found images in nested directory: {nested_image_dir}")
                    image_dir = nested_image_dir

    # Verify metadata exists
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # Read metadata
    df = pd.read_csv(meta_path)

    print(f"Loaded ISIC 2024: {len(df)} samples")
    print(f"Metadata columns: {list(df.columns)}")

    # Print class distribution if target column exists
    if 'target' in df.columns:
        print(f"Binary class distribution:\n{df['target'].value_counts()}\n")
    elif 'diagnosis' in df.columns:
        print(f"Diagnosis distribution:\n{df['diagnosis'].value_counts()}\n")

    return df, image_dir


# Example usage demonstration
if __name__ == "__main__":
    print("Dataset Loaders Module")
    print("=" * 50)
    print("\nSupported datasets:")
    print("1. HAM10000Dataset - 7-class skin lesion classification")
    print("2. ISIC2024Dataset - Binary or multi-class classification")
    print("\nUse this module by importing in your training script:")
    print("  from dataset_loaders import HAM10000Dataset, ISIC2024Dataset")
