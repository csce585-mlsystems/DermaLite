"""
Example script showing how to benchmark CoreML model accuracy
before and after quantization.
"""
from pathlib import Path
from PIL import Image
import pandas as pd
from test_pipeline_accuracy import PipelineTester
from utils.quantize_models import quantize_model
from benchmark_quantization import compare_models, benchmark_pipeline

# Configuration
MODELS_DIR = Path("dermalite/DermaLite/DermaLite/Models")
TEST_DATA_DIR = Path("./data/HAM10000")  # Adjust to your test data path
METADATA_FILE = Path("./data/HAM10000_metadata.csv")  # Adjust if needed
MAX_SAMPLES = 100  # Limit for quick testing


def load_test_images(data_dir, metadata_file=None, max_samples=None):
    """Load test images from dataset"""
    images = []
    labels = []
    
    if metadata_file and metadata_file.exists():
        # Load from metadata
        df = pd.read_csv(metadata_file)
        if max_samples:
            df = df.head(max_samples)
        
        for _, row in df.iterrows():
            if 'image_path' in row:
                img_path = Path(row['image_path'])
            elif 'image_id' in row:
                img_path = data_dir / f"{row['image_id']}.jpg"
            else:
                continue
            
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB").resize((224, 224))
                    images.append(img)
                    
                    # Get label
                    if 'dx' in row:
                        labels.append(row['dx'])
                    elif 'target' in row:
                        labels.append('malignant' if row['target'] == 1 else 'benign')
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    else:
        # Load all images from directory
        image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
        if max_samples:
            image_files = image_files[:max_samples]
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images, labels if labels else None


def example_1_quantize_single_model():
    """Example: Quantize a single model"""
    print("="*70)
    print("EXAMPLE 1: Quantize Single Model")
    print("="*70)
    
    model_path = MODELS_DIR / "mole_detector.mlmodel"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    # Quantize to 8-bit
    quantized_path = quantize_model(
        model_path,
        nbits=8,
        quantization_mode="linear"
    )
    
    print(f"\nQuantized model saved to: {quantized_path}")


def example_2_benchmark_single_model():
    """Example: Benchmark a single model before/after quantization"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Benchmark Single Model")
    print("="*70)
    
    # Load test data
    test_images, test_labels = load_test_images(TEST_DATA_DIR, METADATA_FILE, MAX_SAMPLES)
    
    if not test_images:
        print("No test images found. Please adjust TEST_DATA_DIR and METADATA_FILE paths.")
        return
    
    original_model = MODELS_DIR / "mole_detector.mlmodel"
    quantized_model = MODELS_DIR / "mole_detector_quantized.mlmodel"
    
    # Create quantized model if it doesn't exist
    if not quantized_model.exists():
        print("Creating quantized model...")
        quantize_model(original_model, nbits=8)
    
    # Compare
    comparison = compare_models(
        original_model,
        quantized_model,
        test_images,
        test_labels
    )
    
    print(f"\nComparison complete!")
    print(f"Size reduction: {comparison['size_reduction_percent']:.1f}%")
    print(f"Speedup: {comparison['speedup']:.2f}x")
    if 'accuracy_drop' in comparison:
        print(f"Accuracy drop: {comparison['accuracy_drop']:.4f}")


def example_3_test_pipeline_accuracy():
    """Example: Test full pipeline accuracy"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Test Pipeline Accuracy")
    print("="*70)
    
    tester = PipelineTester()
    
    # Test on dataset
    if TEST_DATA_DIR.exists():
        results = tester.evaluate_on_dataset(
            dataset_path=str(TEST_DATA_DIR),
            metadata_file=str(METADATA_FILE) if METADATA_FILE.exists() else None,
            max_samples=MAX_SAMPLES
        )
        
        if 'accuracy' in results:
            print(f"\nPipeline Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    else:
        print(f"Test data directory not found: {TEST_DATA_DIR}")
        print("Please adjust TEST_DATA_DIR path or create test images.")


def example_4_benchmark_pipeline():
    """Example: Benchmark full pipeline before/after quantization"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Benchmark Full Pipeline")
    print("="*70)
    
    # First, quantize all models
    print("Quantizing models...")
    models_to_quantize = [
        MODELS_DIR / "mole_detector.mlmodel",
        MODELS_DIR / "MalignancyResNet50Features.mlmodel",
    ]
    
    # Note: .mlpackage quantization may need special handling
    multiclass_path = MODELS_DIR / "dermalite_mobilenetv2.mlpackage"
    if not multiclass_path.exists():
        multiclass_path = Path("dermalite_mobilenetv2.mlpackage")
    
    for model_path in models_to_quantize:
        if model_path.exists():
            quantized_path = model_path.parent / f"{model_path.stem}_quantized{model_path.suffix}"
            if not quantized_path.exists():
                quantize_model(model_path, nbits=8)
    
    # Load test data
    test_images, test_labels = load_test_images(TEST_DATA_DIR, METADATA_FILE, MAX_SAMPLES)
    
    if not test_images:
        print("No test images found. Please adjust TEST_DATA_DIR and METADATA_FILE paths.")
        return
    
    # Benchmark pipeline
    benchmark_pipeline(
        MODELS_DIR,
        MODELS_DIR,  # Assuming quantized models are in same dir with _quantized suffix
        test_images,
        test_labels
    )


if __name__ == "__main__":
    print("CoreML Model Accuracy Benchmarking Examples")
    print("="*70)
    print("\nChoose an example to run:")
    print("1. Quantize single model")
    print("2. Benchmark single model")
    print("3. Test pipeline accuracy")
    print("4. Benchmark full pipeline")
    print("\nOr run all examples sequentially...")
    
    # Run examples
    try:
        example_1_quantize_single_model()
        example_2_benchmark_single_model()
        example_3_test_pipeline_accuracy()
        # example_4_benchmark_pipeline()  # Uncomment if you have all models quantized
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure to adjust paths in the script to match your setup.")


