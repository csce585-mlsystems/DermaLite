"""
Benchmark CoreML models before and after quantization.
Compares accuracy, size, and inference speed.
"""
import argparse
import time
from pathlib import Path
import coremltools as ct
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_pipeline_accuracy import PipelineTester
from utils.quantize_models import quantize_model


def benchmark_model_accuracy(model_path, test_images, test_labels=None):
    """
    Benchmark model accuracy on test images.
    
    Args:
        model_path: Path to CoreML model
        test_images: List of PIL Images
        test_labels: Optional list of true labels
    
    Returns:
        dict with accuracy metrics and predictions
    """
    print(f"Loading model: {model_path}")
    model = ct.models.MLModel(str(model_path))
    
    spec = model.get_spec()
    input_name = spec.description.input[0].name
    
    predictions = []
    inference_times = []
    
    print("Running inference...")
    for img in tqdm(test_images):
        start_time = time.time()
        
        prediction = model.predict({input_name: img})
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Extract prediction (simplified - adjust based on your model)
        output_name = spec.description.output[0].name
        output = prediction[output_name]
        
        if isinstance(output, dict):
            # Classification output
            pred = max(output.items(), key=lambda x: x[1])[0]
        elif hasattr(output, '__iter__') and not isinstance(output, str):
            # Array output - get index of max
            pred = int(np.argmax(output))
        else:
            pred = output
        
        predictions.append(pred)
    
    results = {
        'predictions': predictions,
        'avg_inference_time': np.mean(inference_times),
        'total_inference_time': np.sum(inference_times)
    }
    
    # Calculate accuracy if labels provided
    if test_labels:
        # Simple accuracy calculation (adjust based on label format)
        correct = sum(1 for p, l in zip(predictions, test_labels) if str(p).lower() == str(l).lower())
        accuracy = correct / len(test_labels)
        results['accuracy'] = accuracy
    
    return results


def compare_models(original_model_path, quantized_model_path, test_images, test_labels=None):
    """
    Compare original and quantized models.
    
    Args:
        original_model_path: Path to original model
        quantized_model_path: Path to quantized model
        test_images: List of PIL Images for testing
        test_labels: Optional list of true labels
    
    Returns:
        dict with comparison results
    """
    print("="*70)
    print("BENCHMARKING ORIGINAL MODEL")
    print("="*70)
    original_results = benchmark_model_accuracy(original_model_path, test_images, test_labels)
    
    print("\n" + "="*70)
    print("BENCHMARKING QUANTIZED MODEL")
    print("="*70)
    quantized_results = benchmark_model_accuracy(quantized_model_path, test_images, test_labels)
    
    # Compare model sizes
    original_size = Path(original_model_path).stat().st_size / (1024*1024)
    quantized_size = Path(quantized_model_path).stat().st_size / (1024*1024)
    size_reduction = (1 - quantized_size / original_size) * 100
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    comparison = {
        'original': {
            'size_mb': original_size,
            'avg_inference_time': original_results['avg_inference_time'],
            'total_inference_time': original_results['total_inference_time']
        },
        'quantized': {
            'size_mb': quantized_size,
            'avg_inference_time': quantized_results['avg_inference_time'],
            'total_inference_time': quantized_results['total_inference_time']
        },
        'size_reduction_percent': size_reduction,
        'speedup': original_results['avg_inference_time'] / quantized_results['avg_inference_time']
    }
    
    if 'accuracy' in original_results:
        comparison['original']['accuracy'] = original_results['accuracy']
        comparison['quantized']['accuracy'] = quantized_results['accuracy']
        comparison['accuracy_drop'] = original_results['accuracy'] - quantized_results['accuracy']
    
    # Print results
    print(f"\nModel Size:")
    print(f"  Original:  {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Reduction: {size_reduction:.1f}%")
    
    print(f"\nInference Speed:")
    print(f"  Original avg:  {original_results['avg_inference_time']*1000:.2f} ms")
    print(f"  Quantized avg: {quantized_results['avg_inference_time']*1000:.2f} ms")
    print(f"  Speedup:       {comparison['speedup']:.2f}x")
    
    if 'accuracy' in original_results:
        print(f"\nAccuracy:")
        print(f"  Original:  {original_results['accuracy']:.4f} ({original_results['accuracy']*100:.2f}%)")
        print(f"  Quantized: {quantized_results['accuracy']:.4f} ({quantized_results['accuracy']*100:.2f}%)")
        print(f"  Drop:      {comparison['accuracy_drop']:.4f} ({comparison['accuracy_drop']*100:.2f}%)")
    
    return comparison


def benchmark_pipeline(original_models_dir, quantized_models_dir, test_images, test_labels=None):
    """
    Benchmark the full 3-stage pipeline before and after quantization.
    """
    print("="*70)
    print("BENCHMARKING FULL PIPELINE")
    print("="*70)
    
    # Test original pipeline
    print("\n--- Original Pipeline ---")
    original_tester = PipelineTester()
    original_predictions = []
    original_times = []
    
    for img in tqdm(test_images):
        start_time = time.time()
        pred, conf = original_tester.predict_pipeline(img)
        inference_time = time.time() - start_time
        
        original_predictions.append(pred)
        original_times.append(inference_time)
    
    # Test quantized pipeline
    print("\n--- Quantized Pipeline ---")
    quantized_tester = PipelineTester(
        mole_model_path=quantized_models_dir / "mole_detector_quantized.mlmodel",
        binary_model_path=quantized_models_dir / "MalignancyResNet50Features_quantized.mlmodel",
        multiclass_model_path=quantized_models_dir / "dermalite_mobilenetv2_quantized.mlpackage"
    )
    quantized_predictions = []
    quantized_times = []
    
    for img in tqdm(test_images):
        start_time = time.time()
        pred, conf = quantized_tester.predict_pipeline(img)
        inference_time = time.time() - start_time
        
        quantized_predictions.append(pred)
        quantized_times.append(inference_time)
    
    # Compare
    print("\n" + "="*70)
    print("PIPELINE COMPARISON")
    print("="*70)
    
    original_avg_time = np.mean(original_times)
    quantized_avg_time = np.mean(quantized_times)
    
    print(f"\nInference Speed:")
    print(f"  Original avg:  {original_avg_time*1000:.2f} ms")
    print(f"  Quantized avg: {quantized_avg_time*1000:.2f} ms")
    print(f"  Speedup:       {original_avg_time/quantized_avg_time:.2f}x")
    
    if test_labels:
        # Calculate accuracy
        original_correct = sum(1 for p, l in zip(original_predictions, test_labels) 
                              if str(p).lower() == str(l).lower())
        quantized_correct = sum(1 for p, l in zip(quantized_predictions, test_labels) 
                               if str(p).lower() == str(l).lower())
        
        original_acc = original_correct / len(test_labels)
        quantized_acc = quantized_correct / len(test_labels)
        
        print(f"\nAccuracy:")
        print(f"  Original:  {original_acc:.4f} ({original_acc*100:.2f}%)")
        print(f"  Quantized: {quantized_acc:.4f} ({quantized_acc*100:.2f}%)")
        print(f"  Drop:      {original_acc - quantized_acc:.4f} ({(original_acc - quantized_acc)*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CoreML models before and after quantization")
    parser.add_argument("--original", type=str, required=True, help="Path to original model")
    parser.add_argument("--quantized", type=str, help="Path to quantized model (will create if not provided)")
    parser.add_argument("--test-data", type=str, help="Path to test images directory")
    parser.add_argument("--test-labels", type=str, help="Path to test labels CSV")
    parser.add_argument("--nbits", type=int, default=8, help="Quantization bits")
    parser.add_argument("--quantize-mode", type=str, default="linear", help="Quantization mode")
    parser.add_argument("--max-samples", type=int, help="Maximum number of test samples")
    
    args = parser.parse_args()
    
    # Load test images
    test_images = []
    test_labels = None
    
    if args.test_data:
        test_dir = Path(args.test_data)
        image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if args.max_samples:
            image_files = image_files[:args.max_samples]
        
        for img_path in image_files:
            test_images.append(Image.open(img_path).convert("RGB").resize((224, 224)))
    
    if args.test_labels:
        df = pd.read_csv(args.test_labels)
        test_labels = df['label'].tolist() if 'label' in df.columns else None
    
    # Create quantized model if needed
    if not args.quantized:
        print("Creating quantized model...")
        args.quantized = quantize_model(
            args.original,
            nbits=args.nbits,
            quantization_mode=args.quantize_mode
        )
    
    # Compare models
    comparison = compare_models(
        args.original,
        args.quantized,
        test_images,
        test_labels
    )
    
    # Save results
    results_file = Path("quantization_benchmark_results.txt")
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QUANTIZATION BENCHMARK RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Original model: {args.original}\n")
        f.write(f"Quantized model: {args.quantized}\n\n")
        f.write(f"Size reduction: {comparison['size_reduction_percent']:.1f}%\n")
        f.write(f"Speedup: {comparison['speedup']:.2f}x\n")
        if 'accuracy_drop' in comparison:
            f.write(f"Accuracy drop: {comparison['accuracy_drop']:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")

