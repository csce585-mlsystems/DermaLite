"""
Utility to quantize CoreML models for size reduction and performance.
Supports 8-bit, 16-bit, and other quantization modes.
"""
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
from pathlib import Path
import argparse


def quantize_model(
    model_path,
    output_path=None,
    nbits=8,
    quantization_mode="linear",
    sample_data=None
):
    """
    Quantize a CoreML model to reduce size and improve performance.
    
    Args:
        model_path: Path to input CoreML model (.mlmodel or .mlpackage)
        output_path: Path to save quantized model (default: adds _quantized suffix)
        nbits: Number of bits for quantization (8, 16, etc.)
        quantization_mode: Quantization mode ("linear", "linear_symmetric", "kmeans_lut", etc.)
        sample_data: Optional sample data for calibration (list of dicts or image directory path)
    
    Returns:
        Path to quantized model
    """
    print(f"Loading model from: {model_path}")
    model = ct.models.MLModel(str(model_path))
    
    # Get model info
    spec = model.get_spec()
    print(f"Model type: {spec.WhichOneof('Type')}")
    print(f"Model size (before): {Path(model_path).stat().st_size / (1024*1024):.2f} MB")
    
    # Determine output path
    if output_path is None:
        model_path_obj = Path(model_path)
        if model_path_obj.suffix == '.mlpackage':
            output_path = model_path_obj.parent / f"{model_path_obj.stem}_quantized.mlpackage"
        else:
            output_path = model_path_obj.parent / f"{model_path_obj.stem}_quantized.mlmodel"
    
    print(f"\nQuantizing model...")
    print(f"  Bits: {nbits}")
    print(f"  Mode: {quantization_mode}")
    
    # Quantize the model
    quantized_model = quantization_utils.quantize_weights(
        model,
        nbits=nbits,
        quantization_mode=quantization_mode,
        sample_data=sample_data
    )
    
    # Save quantized model
    print(f"\nSaving quantized model to: {output_path}")
    quantized_model.save(str(output_path))
    
    # Compare sizes
    original_size = Path(model_path).stat().st_size / (1024*1024)
    quantized_size = Path(output_path).stat().st_size / (1024*1024)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"\n{'='*50}")
    print("QUANTIZATION COMPLETE")
    print(f"{'='*50}")
    print(f"Original size:  {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"{'='*50}")
    
    return str(output_path)


def quantize_all_models(models_dir, nbits=8, quantization_mode="linear"):
    """
    Quantize all CoreML models in a directory.
    
    Args:
        models_dir: Directory containing CoreML models
        nbits: Number of bits for quantization
        quantization_mode: Quantization mode
    """
    models_dir = Path(models_dir)
    
    # Find all .mlmodel and .mlpackage files
    model_files = list(models_dir.glob("*.mlmodel"))
    model_files.extend(models_dir.glob("*.mlpackage"))
    model_files.extend(models_dir.glob("**/*.mlmodel"))
    model_files.extend(models_dir.glob("**/*.mlpackage"))
    
    print(f"Found {len(model_files)} models to quantize")
    
    for model_path in model_files:
        if "_quantized" in str(model_path):
            print(f"Skipping already quantized model: {model_path}")
            continue
        
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {model_path.name}")
            print(f"{'='*50}")
            quantize_model(model_path, nbits=nbits, quantization_mode=quantization_mode)
        except Exception as e:
            print(f"Error quantizing {model_path}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("ALL MODELS QUANTIZED")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize CoreML models")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--models-dir", type=str, help="Directory containing models")
    parser.add_argument("--output", type=str, help="Output path for quantized model")
    parser.add_argument("--nbits", type=int, default=8, help="Number of bits (8, 16, etc.)")
    parser.add_argument("--mode", type=str, default="linear", 
                       choices=["linear", "linear_symmetric", "kmeans_lut", "linear_lut"],
                       help="Quantization mode")
    parser.add_argument("--sample-data", type=str, help="Path to sample data directory or file")
    
    args = parser.parse_args()
    
    if args.models_dir:
        quantize_all_models(args.models_dir, nbits=args.nbits, quantization_mode=args.mode)
    elif args.model:
        sample_data = args.sample_data if args.sample_data else None
        quantize_model(
            args.model,
            output_path=args.output,
            nbits=args.nbits,
            quantization_mode=args.mode,
            sample_data=sample_data
        )
    else:
        parser.print_help()


