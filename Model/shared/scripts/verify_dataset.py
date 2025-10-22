"""
Dataset Verification Script
Verifies that your dataset is properly set up before training
"""

import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config
from utils.dataset_loaders import load_ham10000_data, load_isic2024_data, HAM10000Dataset, ISIC2024Dataset


def verify_ham10000():
    """Verify HAM10000 dataset setup"""
    print("\n" + "="*70)
    print("VERIFYING HAM10000 DATASET")
    print("="*70 + "\n")

    dataset_config = config.HAM10000_CONFIG

    # Check if root directory exists
    if not os.path.exists(dataset_config['root_dir']):
        print(f"❌ ERROR: Root directory not found: {dataset_config['root_dir']}")
        print("\nPlease update config.py with the correct path to your HAM10000 dataset")
        return False

    try:
        # Load data
        df = load_ham10000_data(dataset_config['root_dir'])

        # Check for missing images
        missing_images = df['image_path'].isna().sum()
        print(f"\n✓ Total samples: {len(df)}")
        print(f"✓ Missing image paths: {missing_images}")

        if missing_images > 0:
            print(f"\n⚠ WARNING: {missing_images} images have missing paths")
            return False

        # Try loading a sample image
        sample_path = df.iloc[0]['image_path']
        print(f"\n✓ Testing image load: {os.path.basename(sample_path)}")
        img = Image.open(sample_path)
        print(f"✓ Image size: {img.size}")
        print(f"✓ Image mode: {img.mode}")

        # Create a test dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        test_ds = HAM10000Dataset(df.head(10), transform=transform)

        print(f"\n✓ Dataset class initialized successfully")
        print(f"✓ Number of classes: {len(test_ds.label2idx)}")
        print(f"✓ Class mapping: {test_ds.label2idx}")

        # Try getting a sample
        sample_img, sample_label = test_ds[0]
        print(f"\n✓ Sample tensor shape: {sample_img.shape}")
        print(f"✓ Sample label: {sample_label} ({test_ds.idx2label[sample_label]})")

        print("\n" + "="*70)
        print("✅ HAM10000 DATASET VERIFICATION PASSED")
        print("="*70 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nDataset verification failed!")
        import traceback
        traceback.print_exc()
        return False


def verify_isic2024():
    """Verify ISIC 2024 dataset setup"""
    print("\n" + "="*70)
    print("VERIFYING ISIC 2024 DATASET")
    print("="*70 + "\n")

    dataset_config = config.ISIC2024_CONFIG

    # Check if root directory exists
    if not os.path.exists(dataset_config['root_dir']):
        print(f"❌ ERROR: Root directory not found: {dataset_config['root_dir']}")
        print("\nPlease run download_isic2024.sh first to download the dataset")
        print("Or update config.py with the correct path")
        return False

    try:
        # Load data
        df, image_dir = load_isic2024_data(
            dataset_config['root_dir'],
            dataset_config['metadata_file']
        )

        print(f"\n✓ Metadata file loaded successfully")
        print(f"✓ Total samples: {len(df)}")
        print(f"✓ Image directory: {image_dir}")

        # Check if image directory exists and has images
        if not os.path.exists(image_dir):
            print(f"❌ ERROR: Image directory not found: {image_dir}")
            return False

        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        print(f"✓ Number of image files: {len(image_files)}")

        if len(image_files) == 0:
            print(f"❌ ERROR: No image files found in {image_dir}")
            return False

        # Check metadata columns
        print(f"\n✓ Metadata columns: {list(df.columns)}")

        # Verify required columns based on task
        if dataset_config['task'] == 'binary':
            if 'target' not in df.columns:
                print("❌ ERROR: 'target' column not found for binary classification")
                return False
            print(f"✓ Binary target column found")
            print(f"✓ Class distribution:\n{df['target'].value_counts()}")
        else:
            if 'diagnosis' not in df.columns:
                print("❌ ERROR: 'diagnosis' column not found for multiclass classification")
                return False
            print(f"✓ Diagnosis column found")
            print(f"✓ Class distribution:\n{df['diagnosis'].value_counts()}")

        # Create a test dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        test_ds = ISIC2024Dataset(
            df.head(10),
            image_dir,
            transform=transform,
            task=dataset_config['task']
        )

        print(f"\n✓ Dataset class initialized successfully")
        print(f"✓ Number of classes: {len(test_ds.label2idx)}")
        print(f"✓ Class mapping: {test_ds.label2idx}")

        # Try getting a sample
        try:
            sample_img, sample_label = test_ds[0]
            print(f"\n✓ Sample tensor shape: {sample_img.shape}")
            print(f"✓ Sample label: {sample_label} ({test_ds.idx2label[sample_label]})")
        except FileNotFoundError as e:
            print(f"\n⚠ WARNING: Could not load sample image")
            print(f"Error: {str(e)}")
            print("\nThis might be due to image filename mismatch.")
            print("Please check that image filenames in the metadata match actual files.")
            return False

        print("\n" + "="*70)
        print("✅ ISIC 2024 DATASET VERIFICATION PASSED")
        print("="*70 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nDataset verification failed!")
        import traceback
        traceback.print_exc()
        return False


def visualize_samples(num_samples=5):
    """Visualize sample images from the active dataset"""
    print("\n" + "="*70)
    print(f"VISUALIZING {num_samples} SAMPLE IMAGES FROM {config.DATASET}")
    print("="*70 + "\n")

    try:
        if config.DATASET == 'HAM10000':
            df = load_ham10000_data(config.HAM10000_CONFIG['root_dir'])
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            dataset = HAM10000Dataset(df, transform=transform)

        elif config.DATASET == 'ISIC2024':
            df, image_dir = load_isic2024_data(
                config.ISIC2024_CONFIG['root_dir'],
                config.ISIC2024_CONFIG['metadata_file']
            )
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            dataset = ISIC2024Dataset(
                df,
                image_dir,
                transform=transform,
                task=config.ISIC2024_CONFIG['task']
            )

        # Create visualization
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

        for i in range(num_samples):
            img, label = dataset[i]
            # Convert tensor to numpy for display
            img_np = img.permute(1, 2, 0).numpy()

            axes[i].imshow(img_np)
            axes[i].set_title(f"Label: {dataset.idx2label[label]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig('dataset_samples.png')
        print("✓ Sample visualization saved to: dataset_samples.png")
        plt.close()

    except Exception as e:
        print(f"❌ ERROR: Could not visualize samples")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main verification function"""
    print("\n" + "="*70)
    print("DATASET VERIFICATION TOOL")
    print("="*70)

    # Print current config
    config.print_config()

    # Verify based on active dataset
    if config.DATASET == 'HAM10000':
        success = verify_ham10000()
    elif config.DATASET == 'ISIC2024':
        success = verify_isic2024()
    else:
        print(f"\n❌ ERROR: Unknown dataset: {config.DATASET}")
        print("Please set DATASET to 'HAM10000' or 'ISIC2024' in config.py")
        success = False

    if success:
        # Optionally visualize samples
        try:
            visualize = input("\nWould you like to visualize sample images? (y/n): ")
            if visualize.lower() == 'y':
                visualize_samples()
        except:
            pass

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
