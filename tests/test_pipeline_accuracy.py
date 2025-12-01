"""
Evaluate 3-stage CoreML pipeline (Mole Detector → Binary Classifier → Multiclass Classifier)
on HAM10000 dataset, matching the exact logic used in the iOS app (MLService.swift).

Pipeline Flow (matching app):
1. Stage 0 (Mole Detector): Add Gaussian noise → detect mole → if no mole, return "No Mole Detected"
2. Stage 1 (Binary Classifier): If mole detected → classify benign/malignant → if benign, return "Benign"
3. Stage 2 (Multiclass Classifier): If malignant → classify specific type → filter to only malignant types
"""

import coremltools as ct
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import argparse
import random

# ---------------------------
# Constants (matching app)
# ---------------------------
MALIGNANT_TYPES = {"akiec", "bcc", "mel"}  # Only these are considered malignant
LESION_TYPES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}
CLASS_LABELS_FALLBACK = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Binary label mapping for evaluation
CANCER_LABELS = ['mel', 'bcc', 'akiec']  # malignant
NON_CANCER_LABELS = ['nv', 'bkl', 'df', 'vasc']  # benign

# ---------------------------
# Helper Functions
# ---------------------------
def add_gaussian_noise(image):
    """
    Add Gaussian noise to image matching Swift implementation.
    Swift adds uniform random noise in [-12.75, 12.75] to each RGB channel.
    This simulates phone camera grain for mole detector robustness.
    """
    img_array = np.array(image, dtype=np.float32)
    noise = np.random.uniform(-12.75, 12.75, img_array.shape)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def softmax(logits):
    """Compute softmax probabilities from logits."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

# ---------------------------
# PipelineTester class (matching app logic)
# ---------------------------
class PipelineTester:
    """3-stage CoreML pipeline tester matching iOS app behavior"""
    
    def __init__(self, mole_model_path=None, binary_model_path=None, multiclass_model_path=None, debug=False):
        self.mole_model = None
        self.binary_model = None
        self.multiclass_model = None
        self.debug = debug
        
        # Try to find models in standard locations
        MODEL_DIR = Path(__file__).parent.parent / "dermalite" / "DermaLite" / "DermaLite" / "Models"
        if not MODEL_DIR.exists():
            MODEL_DIR = Path(__file__).parent / "dermalite" / "DermaLite" / "DermaLite" / "Models"
        
        # Load models
        if mole_model_path:
            self.mole_model = ct.models.MLModel(str(mole_model_path))
        else:
            path = MODEL_DIR / "mole_detector.mlmodel"
            if path.exists():
                self.mole_model = ct.models.MLModel(str(path))
                print(f"✓ Loaded mole detector: {path}")
            else:
                print(f"⚠ Mole detector not found at {path}")

        if binary_model_path:
            binary_path = Path(binary_model_path)
            if not binary_path.exists():
                print(f"⚠ Warning: Binary model not found at {binary_path}")
                binary_path = None
            else:
                # Check if it's the correct model
                if "dermalite_binary_classifier" in binary_path.name:
                    print(f"⚠ Warning: Using {binary_path.name} instead of MalignancyResNet50Features.mlmodel")
                    print(f"   The iOS app uses MalignancyResNet50Features.mlmodel - output format may differ!")
                self.binary_model = ct.models.MLModel(str(binary_path))
                print(f"✓ Loaded binary classifier: {binary_path}")
        else:
            # Try to find the correct model (preferred by iOS app)
            path = MODEL_DIR / "MalignancyResNet50Features.mlmodel"
            if not path.exists():
                path = MODEL_DIR / "MalignancyResNet50Features.mlpackage"
            if not path.exists():
                # Fallback to alternative model
                path = MODEL_DIR / "dermalite_binary_classifier.mlmodel"
                if path.exists():
                    print(f"⚠ Warning: Using fallback model {path.name} instead of MalignancyResNet50Features.mlmodel")
            
            if path.exists():
                self.binary_model = ct.models.MLModel(str(path))
                print(f"✓ Loaded binary classifier: {path}")
            else:
                print(f"⚠ Binary classifier not found in {MODEL_DIR}")
                print(f"   Expected: MalignancyResNet50Features.mlmodel or dermalite_binary_classifier.mlmodel")

        if multiclass_model_path:
            self.multiclass_model = ct.models.MLModel(str(multiclass_model_path))
        else:
            path = MODEL_DIR / "dermalite_mobilenetv2.mlpackage"
            if not path.exists():
                path = Path(__file__).parent.parent / "dermalite_mobilenetv2.mlpackage"
            if path.exists():
                self.multiclass_model = ct.models.MLModel(str(path))
                print(f"✓ Loaded multiclass classifier: {path}")
            else:
                print(f"⚠ Multiclass classifier not found at {path}")
        
        # Extract class labels from multiclass model
        self.class_labels = None
        if self.multiclass_model:
            try:
                spec = self.multiclass_model.get_spec()
                if hasattr(spec.description, 'output') and len(spec.description.output) > 0:
                    # Try to get class labels from model description
                    # This may vary by model format
                    self.class_labels = CLASS_LABELS_FALLBACK
            except:
                self.class_labels = CLASS_LABELS_FALLBACK
        else:
            self.class_labels = CLASS_LABELS_FALLBACK
        
        # Validate binary model output format if debug mode
        if self.debug and self.binary_model:
            self._validate_binary_model_output()
    
    def _validate_binary_model_output(self):
        """Test the binary model with a sample image to understand its output format"""
        print("\n🔍 Validating Binary Model Output Format...")
        try:
            test_img = Image.new('RGB', (224, 224), color='red')
            spec = self.binary_model.get_spec()
            input_name = spec.description.input[0].name
            prediction = self.binary_model.predict({input_name: test_img})
            
            print(f"  Model outputs: {list(prediction.keys())}")
            for key, value in prediction.items():
                if isinstance(value, dict):
                    print(f"    {key}: {value}")
                elif isinstance(value, (list, np.ndarray)):
                    arr = np.asarray(value)
                    print(f"    {key}: array shape {arr.shape}, values: {arr.flatten()[:5]}")
                else:
                    print(f"    {key}: {value} (type: {type(value)})")
            
            # Check output names from spec
            output_names = [out.name for out in spec.description.output]
            print(f"  Spec output names: {output_names}")
            
        except Exception as e:
            print(f"  ⚠ Could not validate model: {e}")

    def predict_stage0(self, image):
        """
        Stage 0: Mole Detection (matching app logic)
        - Add Gaussian noise (critical for robustness)
        - Return (is_mole: bool, confidence: float)
        """
        if not self.mole_model:
            return True, 1.0  # Skip if model not available
        
        # Ensure image is RGB and properly sized
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        image = image.convert("RGB").resize((224, 224))
        
        # Add Gaussian noise (matching Swift implementation)
        noisy_image = add_gaussian_noise(image)
        
        try:
            spec = self.mole_model.get_spec()
            input_name = spec.description.input[0].name
            output_name = spec.description.output[0].name
            
            # CoreML accepts PIL Images directly
            pred = self.mole_model.predict({input_name: noisy_image})
            
            # Handle different output formats
            output = pred[output_name]
            if isinstance(output, dict):
                prob = float(output.get('Mole', output.get('mole', 0.5)))
            elif isinstance(output, (list, np.ndarray)):
                # Fix NumPy deprecation warning - extract single element properly
                arr = np.asarray(output)
                if arr.size > 0:
                    prob = float(arr.flat[0])
                else:
                    prob = 0.5
            else:
                prob = float(output)
            
            is_mole = prob >= 0.5
            return is_mole, prob
        except Exception as e:
            print(f"Error in Stage 0: {e}")
            return True, 1.0  # Default to mole present on error

    def predict_stage1(self, image):
        """
        Stage 1: Binary Classification (matching app logic)
        - Returns (is_malignant: bool, confidence: float)
        - Outputs "Benign" or "Malignant" as strings
        """
        if not self.binary_model:
            return False, 0.5
        
        # Ensure image is RGB and properly sized
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        image = image.convert("RGB").resize((224, 224))
        
        try:
            spec = self.binary_model.get_spec()
            input_name = spec.description.input[0].name
            prediction = self.binary_model.predict({input_name: image})
            
            # Debug: print model outputs if enabled
            if self.debug:
                print(f"\nDEBUG Stage 1 - Model outputs:")
                for key, value in prediction.items():
                    if isinstance(value, dict):
                        print(f"  {key}: {value}")
                    elif isinstance(value, (list, np.ndarray)):
                        arr = np.asarray(value)
                        print(f"  {key}: array shape {arr.shape}, sample: {arr.flatten()[:min(5, arr.size)]}")
                    else:
                        print(f"  {key}: {value} (type: {type(value)})")
            
            # Try to find classification output
            output_names = [out.name for out in spec.description.output]
            if self.debug:
                print(f"DEBUG Stage 1 - Output names from spec: {output_names}")
                print(f"DEBUG Stage 1 - Prediction keys: {list(prediction.keys())}")
            
            is_malignant = False
            confidence = 0.5
            
            # First, check all outputs to understand the format
            malignant_prob = 0.0
            benign_prob = 0.0
            
            # Strategy 1: Look for label/class output
            for name in output_names:
                if 'class' in name.lower() or 'label' in name.lower():
                    label = prediction.get(name, 'Benign')
                    # Handle both string and dict outputs
                    if isinstance(label, dict):
                        # If dict, find the key with highest value
                        label = max(label.items(), key=lambda x: x[1])[0]
                    
                    label_str = str(label).lower()
                    is_malignant = 'malignant' in label_str
                    
                    # Try to get probability
                    prob_name = name.replace('Label', 'Probability').replace('label', 'probability')
                    if prob_name in prediction:
                        probs = prediction[prob_name]
                        if isinstance(probs, dict):
                            malignant_prob = float(probs.get('Malignant', probs.get('malignant', 0.0)))
                            benign_prob = float(probs.get('Benign', probs.get('benign', 0.0)))
                            confidence = max(malignant_prob, benign_prob)
                            if malignant_prob > benign_prob:
                                is_malignant = True
                    break
            
            # Strategy 2: Check all outputs for probability dictionaries
            if confidence == 0.5 or (malignant_prob == 0.0 and benign_prob == 0.0):
                for name in output_names:
                    output = prediction.get(name)
                    if isinstance(output, dict):
                        # Check if this is a probability dictionary
                        for key, value in output.items():
                            key_lower = str(key).lower()
                            try:
                                val = float(value)
                                if 'malignant' in key_lower or 'cancer' in key_lower or 'positive' in key_lower:
                                    malignant_prob = max(malignant_prob, val)
                                elif 'benign' in key_lower or 'negative' in key_lower:
                                    benign_prob = max(benign_prob, val)
                            except (ValueError, TypeError):
                                pass
                        
                        if malignant_prob > 0 or benign_prob > 0:
                            is_malignant = malignant_prob > benign_prob
                            confidence = max(malignant_prob, benign_prob)
                            break
            
            # Strategy 3: Check for multi-array outputs (logits or probabilities)
            if confidence == 0.5 or (malignant_prob == 0.0 and benign_prob == 0.0):
                for name in output_names:
                    output = prediction.get(name)
                    if hasattr(output, 'shape') or isinstance(output, (list, np.ndarray)):
                        # Convert to numpy array
                        arr = np.asarray(output)
                        if arr.size >= 2:
                            # Assume binary classification: [benign_prob, malignant_prob] or [malignant_prob, benign_prob]
                            # Try both orderings
                            probs = arr.flatten()
                            if len(probs) >= 2:
                                # Try first ordering: [benign, malignant]
                                benign_candidate = float(probs[0])
                                malignant_candidate = float(probs[1])
                                
                                # If values don't sum to ~1, might be logits - apply softmax
                                if abs(benign_candidate + malignant_candidate - 1.0) > 0.1:
                                    probs = softmax(probs[:2])
                                    benign_candidate = float(probs[0])
                                    malignant_candidate = float(probs[1])
                                
                                # Check if this makes sense (both should be between 0 and 1)
                                if 0 <= benign_candidate <= 1 and 0 <= malignant_candidate <= 1:
                                    benign_prob = benign_candidate
                                    malignant_prob = malignant_candidate
                                    is_malignant = malignant_prob > benign_prob
                                    confidence = max(malignant_prob, benign_prob)
                                    break
                                
                                # Try reverse ordering: [malignant, benign]
                                benign_candidate = float(probs[1])
                                malignant_candidate = float(probs[0])
                                if 0 <= benign_candidate <= 1 and 0 <= malignant_candidate <= 1:
                                    benign_prob = benign_candidate
                                    malignant_prob = malignant_candidate
                                    is_malignant = malignant_prob > benign_prob
                                    confidence = max(malignant_prob, benign_prob)
                                    break
            
            # Strategy 4: Single value output (probability of malignant)
            if confidence == 0.5:
                for name in output_names:
                    output = prediction.get(name)
                    if not isinstance(output, dict) and not isinstance(output, (list, np.ndarray)):
                        try:
                            val = float(output)
                            if 0 <= val <= 1:
                                # Could be probability of malignant
                                malignant_prob = val
                                benign_prob = 1.0 - val
                                is_malignant = malignant_prob > 0.5
                                confidence = max(malignant_prob, benign_prob)
                                break
                        except (ValueError, TypeError):
                            pass
            
            # Final fallback: if we have probabilities, use them
            if malignant_prob > 0 or benign_prob > 0:
                is_malignant = malignant_prob > benign_prob
                confidence = max(malignant_prob, benign_prob)
            
            if self.debug:
                print(f"DEBUG Stage 1 - Final: is_malignant={is_malignant}, confidence={confidence}, malignant_prob={malignant_prob}, benign_prob={benign_prob}")
            
            # Warning if we couldn't parse properly
            if confidence == 0.5 and malignant_prob == 0.0 and benign_prob == 0.0:
                print(f"⚠ Warning: Could not parse binary classifier output. Using default (benign).")
                if not self.debug:
                    print(f"   Run with --debug to see model outputs and diagnose the issue.")
            
            return is_malignant, confidence
        except Exception as e:
            print(f"❌ Error in Stage 1: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False, 0.5

    def predict_stage2(self, image):
        """
        Stage 2: Multiclass Classification (matching app logic)
        - Only considers malignant types: akiec, bcc, mel
        - Returns (diagnosis: str, confidence: float)
        - If conflict (binary says malignant but multiclass returns benign), 
          returns "Malignant (Requires Medical Evaluation)"
        """
        if not self.multiclass_model:
            return "Malignant (Requires Medical Evaluation)", 0.5
        
        # Ensure image is RGB and properly sized
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        image = image.convert("RGB").resize((224, 224))
        
        try:
            spec = self.multiclass_model.get_spec()
            input_name = spec.description.input[0].name
            prediction = self.multiclass_model.predict({input_name: image})
            
            output_names = [out.name for out in spec.description.output]
            best_label = None
            best_prob = 0.0
            
            # Try standard classification output
            for name in output_names:
                if 'class' in name.lower() or 'label' in name.lower():
                    label = prediction.get(name, 'nv')
                    if isinstance(label, dict):
                        label = max(label.items(), key=lambda x: x[1])[0]
                    
                    label_str = str(label).lower()
                    
                    # Filter to only malignant types (matching app logic)
                    if label_str in MALIGNANT_TYPES:
                        # Get probability
                        prob_name = name.replace('Label', 'Probability').replace('label', 'probability')
                        if prob_name in prediction:
                            probs = prediction[prob_name]
                            if isinstance(probs, dict):
                                prob = float(probs.get(label_str, 0.0))
                                if prob > best_prob:
                                    best_prob = prob
                                    best_label = label_str
                    break
            
            # Fallback: check probability dictionary directly
            if best_label is None:
                for name in output_names:
                    if 'prob' in name.lower():
                        probs = prediction.get(name, {})
                        if isinstance(probs, dict):
                            # Find best malignant type
                            for label, prob in probs.items():
                                label_str = str(label).lower()
                                if label_str in MALIGNANT_TYPES:
                                    prob_val = float(prob)
                                    if prob_val > best_prob:
                                        best_prob = prob_val
                                        best_label = label_str
                        break
            
            # Fallback: try multi-array format
            if best_label is None:
                for name in output_names:
                    output = prediction.get(name)
                    if hasattr(output, 'shape') or isinstance(output, (list, np.ndarray)):
                        # Convert to numpy array
                        if not isinstance(output, np.ndarray):
                            output = np.array(output)
                        
                        # Apply softmax if needed
                        if output.ndim == 1:
                            probs = softmax(output.flatten())
                        else:
                            probs = softmax(output.flatten())
                        
                        # Find best malignant class
                        for idx, prob in enumerate(probs):
                            if idx < len(self.class_labels):
                                label = self.class_labels[idx].lower()
                                if label in MALIGNANT_TYPES and prob > best_prob:
                                    best_prob = float(prob)
                                    best_label = label
            
            if best_label:
                diagnosis = LESION_TYPES.get(best_label, best_label)
                return diagnosis, best_prob
            else:
                # Conflict: binary said malignant but multiclass returned benign
                return "Malignant (Requires Medical Evaluation)", 0.5
        except Exception as e:
            print(f"Error in Stage 2: {e}")
            return "Malignant (Requires Medical Evaluation)", 0.5

    def predict_pipeline(self, image):
        """
        Full 3-stage pipeline (matching app logic exactly):
        1. Stage 0: Mole detector → if no mole, return "No Mole Detected"
        2. Stage 1: Binary classifier → if benign, return "Benign"
        3. Stage 2: Multiclass classifier → return specific diagnosis
        """
        # Stage 0: Mole Detection
        is_mole, mole_confidence = self.predict_stage0(image)
        if not is_mole:
            # Invert confidence for display (matching app)
            not_mole_confidence = 1.0 - mole_confidence
            return "No Mole Detected", not_mole_confidence
        
        # Stage 1: Binary Classification
        is_malignant, binary_confidence = self.predict_stage1(image)
        if not is_malignant:
            return "Benign", binary_confidence
        
        # Stage 2: Multiclass Classification
        diagnosis, multiclass_confidence = self.predict_stage2(image)
        return diagnosis, multiclass_confidence

    def evaluate_on_dataset(self, dataset_path, metadata_file, max_samples=None):
        """
        Evaluate pipeline on a dataset.
        
        Args:
            dataset_path: Path to directory containing images or list of image paths
            metadata_file: Path to CSV file with image_id and dx columns
            max_samples: Maximum number of samples to evaluate (None for all)
        
        Returns:
            Dictionary with accuracy metrics and detailed results
        """
        # Load metadata
        df = pd.read_csv(metadata_file)
        image_id_to_label = {row['image_id']: row['dx'] for idx, row in df.iterrows()}
        
        # Get image paths
        if isinstance(dataset_path, (list, tuple)):
            image_paths = [Path(p) for p in dataset_path]
        else:
            dataset_path = Path(dataset_path)
            if dataset_path.is_file():
                image_paths = [dataset_path]
            else:
                # Look for images in common formats
                image_paths = (
                    list(dataset_path.glob("*.jpg")) +
                    list(dataset_path.glob("*.jpeg")) +
                    list(dataset_path.glob("*.png")) +
                    list(dataset_path.glob("*.JPG")) +
                    list(dataset_path.glob("*.JPEG")) +
                    list(dataset_path.glob("*.PNG"))
                )
        
        if max_samples:
            image_paths = image_paths[:max_samples]
        
        print(f"Evaluating on {len(image_paths)} images...")
        
        predictions = []
        labels = []
        predictions_detailed = []
        labels_detailed = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                image_id = img_path.stem
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                
                # Run pipeline
                pred, confidence = self.predict_pipeline(img)
                
                # Get ground truth label
                true_label = image_id_to_label.get(image_id, '').lower()
                
                # Map to binary for evaluation
                pred_bin = 'cancer' if any(c in pred.lower() for c in CANCER_LABELS) else 'non-cancer'
                if pred == "No Mole Detected":
                    pred_bin = 'non-cancer'
                elif pred == "Benign":
                    pred_bin = 'non-cancer'
                
                label_bin = 'cancer' if true_label in CANCER_LABELS else 'non-cancer'
                
                predictions.append(pred_bin)
                labels.append(label_bin)
                predictions_detailed.append(pred)
                labels_detailed.append(true_label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate metrics
        acc = accuracy_score(labels, predictions)
        
        # Use zero_division parameter to avoid warnings
        results = {
            'accuracy': acc,
            'classification_report': classification_report(labels, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions),
            'predictions': predictions,
            'labels': labels,
            'predictions_detailed': predictions_detailed,
            'labels_detailed': labels_detailed
        }
        
        return results


# ---------------------------
# Main execution
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Test 3-stage pipeline accuracy')
    parser.add_argument('--dataset', type=str, 
                       default="/Users/t/Downloads/archive/HAM10000_images_part_1",
                       help='Path to dataset directory or image list')
    parser.add_argument('--dataset2', type=str,
                       default="/Users/t/Downloads/archive/HAM10000_images_part_2",
                       help='Path to second dataset directory (optional)')
    parser.add_argument('--metadata', type=str,
                       default="/Users/t/Downloads/archive/HAM10000_metadata.csv",
                       help='Path to metadata CSV file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--mole-model', type=str, default=None,
                       help='Path to mole detector model')
    parser.add_argument('--binary-model', type=str, default=None,
                       help='Path to binary classifier model')
    parser.add_argument('--multiclass-model', type=str, default=None,
                       help='Path to multiclass classifier model')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output to see model predictions')
    parser.add_argument('--validate-models', action='store_true',
                       help='Validate model output formats on startup')
    
    args = parser.parse_args()
    
    # Print model information
    print("\n" + "="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    if args.binary_model:
        binary_name = Path(args.binary_model).name
        if "dermalite_binary_classifier" in binary_name:
            print(f"⚠ Using: {binary_name}")
            print(f"  Note: iOS app uses 'MalignancyResNet50Features.mlmodel'")
            print(f"  Output format may differ - use --validate-models to check")
        else:
            print(f"✓ Using: {binary_name}")
    print("="*60 + "\n")
    
    # Initialize tester
    tester = PipelineTester(
        mole_model_path=args.mole_model,
        binary_model_path=args.binary_model,
        multiclass_model_path=args.multiclass_model,
        debug=args.debug or args.validate_models
    )
    
    # Validate models if requested
    if args.validate_models:
        print("\n" + "="*60)
        print("MODEL VALIDATION")
        print("="*60)
        if tester.binary_model:
            tester._validate_binary_model_output()
        print("="*60 + "\n")
    
    # Collect image paths
    image_paths = []
    dataset1 = Path(args.dataset)
    if dataset1.exists():
        if dataset1.is_file():
            image_paths.append(dataset1)
        else:
            image_paths.extend(list(dataset1.glob("*.jpg")) + list(dataset1.glob("*.jpeg")))
    
    if args.dataset2:
        dataset2 = Path(args.dataset2)
        if dataset2.exists():
            image_paths.extend(list(dataset2.glob("*.jpg")) + list(dataset2.glob("*.jpeg")))
    
    if not image_paths:
        print("No images found! Please check dataset paths.")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Evaluate
    results = tester.evaluate_on_dataset(
        dataset_path=image_paths,
        metadata_file=args.metadata,
        max_samples=args.max_samples
    )
    
    # Print results
    print("\n" + "="*60)
    print("PIPELINE ACCURACY RESULTS")
    print("="*60)
    print(f"\nBinary Accuracy (non-cancer vs cancer): {results['accuracy']*100:.2f}%\n")
    print("Classification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
