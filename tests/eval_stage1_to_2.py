"""
Evaluate 2-stage CoreML pipeline (Binary Classifier → Multiclass Classifier)
on HAM10000 dataset, ASSUMING all images are moles (skipping Stage 0).

Pipeline Flow:
1. Stage 1 (Binary Classifier): Classify benign/malignant → if benign, return "Benign"
2. Stage 2 (Multiclass Classifier): If malignant → classify specific type
"""

import coremltools as ct
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import argparse

# ---------------------------
# Constants 
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
def softmax(logits):
    """Compute softmax probabilities from logits."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

# ---------------------------
# Stage1-2 Pipeline Tester
# ---------------------------
class Stage1_2_Tester:
    """2-stage CoreML pipeline tester (Binary → Multiclass)"""
    
    def __init__(self, binary_model_path=None, multiclass_model_path=None, debug=False, malignancy_threshold=0.5):
        self.binary_model = None
        self.multiclass_model = None
        self.debug = debug
        self.malignancy_threshold = malignancy_threshold  # Threshold for classifying as malignant
        
        # Try to find models in standard locations
        MODEL_DIR = Path(__file__).parent.parent / "dermalite" / "DermaLite" / "DermaLite" / "Models"
        if not MODEL_DIR.exists():
            MODEL_DIR = Path(__file__).parent / "dermalite" / "DermaLite" / "DermaLite" / "Models"
        
        # Load binary model
        if binary_model_path:
            binary_path = Path(binary_model_path)
            if not binary_path.exists():
                print(f"⚠ Warning: Binary model not found at {binary_path}")
            else:
                if "dermalite_binary_classifier" in binary_path.name:
                    print(f"⚠ Warning: Using {binary_path.name} instead of MalignancyResNet50Features.mlmodel")
                self.binary_model = ct.models.MLModel(str(binary_path))
                print(f"✓ Loaded binary classifier: {binary_path}")
        else:
            path = MODEL_DIR / "MalignancyResNet50Features.mlmodel"
            if not path.exists():
                path = MODEL_DIR / "MalignancyResNet50Features.mlpackage"
            if not path.exists():
                path = MODEL_DIR / "dermalite_binary_classifier.mlmodel"
                if path.exists():
                    print(f"⚠ Warning: Using fallback model {path.name}")
            
            if path.exists():
                self.binary_model = ct.models.MLModel(str(path))
                print(f"✓ Loaded binary classifier: {path}")
            else:
                print(f"❌ Binary classifier not found in {MODEL_DIR}")
                raise FileNotFoundError("Binary classifier model required")

        # Load multiclass model
        if multiclass_model_path:
            self.multiclass_model = ct.models.MLModel(str(multiclass_model_path))
            print(f"✓ Loaded multiclass classifier: {multiclass_model_path}")
        else:
            path = MODEL_DIR / "dermalite_mobilenetv2.mlpackage"
            if not path.exists():
                path = Path(__file__).parent.parent / "dermalite_mobilenetv2.mlpackage"
            if path.exists():
                self.multiclass_model = ct.models.MLModel(str(path))
                print(f"✓ Loaded multiclass classifier: {path}")
            else:
                print(f"❌ Multiclass classifier not found")
                raise FileNotFoundError("Multiclass classifier model required")
        
        # Extract class labels from multiclass model
        self.class_labels = CLASS_LABELS_FALLBACK
        
        if self.debug:
            self._validate_models()
    
    def _validate_models(self):
        """Test models with sample images to understand output formats"""
        print("\n🔍 Validating Model Output Formats...")
        test_img = Image.new('RGB', (224, 224), color='red')
        
        # Test binary model
        if self.binary_model:
            print("\n--- Binary Model ---")
            try:
                spec = self.binary_model.get_spec()
                input_name = spec.description.input[0].name
                prediction = self.binary_model.predict({input_name: test_img})
                
                print(f"Outputs: {list(prediction.keys())}")
                for key, value in prediction.items():
                    if isinstance(value, dict):
                        print(f"  {key}: {value}")
                    elif isinstance(value, (list, np.ndarray)):
                        arr = np.asarray(value)
                        print(f"  {key}: shape {arr.shape}, sample: {arr.flatten()[:5]}")
                    else:
                        print(f"  {key}: {value} (type: {type(value)})")
            except Exception as e:
                print(f"  ⚠ Error: {e}")
        
        # Test multiclass model
        if self.multiclass_model:
            print("\n--- Multiclass Model ---")
            try:
                spec = self.multiclass_model.get_spec()
                input_name = spec.description.input[0].name
                prediction = self.multiclass_model.predict({input_name: test_img})
                
                print(f"Outputs: {list(prediction.keys())}")
                for key, value in prediction.items():
                    if isinstance(value, dict):
                        print(f"  {key}: {dict(list(value.items())[:3])}...")
                    elif isinstance(value, (list, np.ndarray)):
                        arr = np.asarray(value)
                        print(f"  {key}: shape {arr.shape}, sample: {arr.flatten()[:5]}")
                    else:
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"  ⚠ Error: {e}")
        print()

    def predict_stage1(self, image):
        """Stage 1: Binary Classification → (is_malignant, confidence)"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        image = image.convert("RGB").resize((224, 224))
        
        try:
            spec = self.binary_model.get_spec()
            input_name = spec.description.input[0].name
            prediction = self.binary_model.predict({input_name: image})
            
            if self.debug:
                print(f"\n[Stage 1] Model outputs: {list(prediction.keys())}")
            
            output_names = [out.name for out in spec.description.output]
            is_malignant = False
            confidence = 0.5
            malignant_prob = 0.0
            benign_prob = 0.0
            
            # Strategy 1: Look for label/class output
            for name in output_names:
                if 'class' in name.lower() or 'label' in name.lower():
                    # Get probability first (more important than label)
                    prob_name = name.replace('Label', 'Probability').replace('label', 'probability')
                    if prob_name in prediction:
                        probs = prediction[prob_name]
                        if isinstance(probs, dict):
                            malignant_prob = float(probs.get('Malignant', probs.get('malignant', 0.0)))
                            benign_prob = float(probs.get('Benign', probs.get('benign', 0.0)))
                            # Apply threshold here
                            is_malignant = malignant_prob > self.malignancy_threshold
                            confidence = max(malignant_prob, benign_prob)
                    else:
                        # Fallback to label if no probabilities
                        label = prediction.get(name, 'Benign')
                        if isinstance(label, dict):
                            label = max(label.items(), key=lambda x: x[1])[0]
                        label_str = str(label).lower()
                        is_malignant = 'malignant' in label_str
                    break
            
            # Strategy 2: Check for probability dictionaries
            if confidence == 0.5:
                for name in output_names:
                    output = prediction.get(name)
                    if isinstance(output, dict):
                        for key, value in output.items():
                            key_lower = str(key).lower()
                            try:
                                val = float(value)
                                if 'malignant' in key_lower:
                                    malignant_prob = max(malignant_prob, val)
                                elif 'benign' in key_lower:
                                    benign_prob = max(benign_prob, val)
                            except (ValueError, TypeError):
                                pass
                        
                        if malignant_prob > 0 or benign_prob > 0:
                            # Apply threshold
                            is_malignant = malignant_prob > self.malignancy_threshold
                            confidence = max(malignant_prob, benign_prob)
                            break
            
            # Strategy 3: Array outputs
            if confidence == 0.5:
                for name in output_names:
                    output = prediction.get(name)
                    if hasattr(output, 'shape') or isinstance(output, (list, np.ndarray)):
                        arr = np.asarray(output)
                        if arr.size >= 2:
                            probs = arr.flatten()
                            if len(probs) >= 2:
                                # Try [benign, malignant] ordering
                                if abs(probs[0] + probs[1] - 1.0) > 0.1:
                                    probs = softmax(probs[:2])
                                
                                benign_prob = float(probs[0])
                                malignant_prob = float(probs[1])
                                # Apply threshold
                                is_malignant = malignant_prob > self.malignancy_threshold
                                confidence = max(malignant_prob, benign_prob)
                                break
            
            if self.debug:
                print(f"[Stage 1] Result: malignant={is_malignant}, conf={confidence:.3f} (M:{malignant_prob:.3f}, B:{benign_prob:.3f}), threshold={self.malignancy_threshold}")
            
            return is_malignant, confidence
        except Exception as e:
            print(f"❌ Error in Stage 1: {e}")
            return False, 0.5

    def predict_stage2(self, image):
        """Stage 2: Multiclass Classification → (diagnosis, confidence)"""
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
                    
                    # Filter to only malignant types
                    if label_str in MALIGNANT_TYPES:
                        prob_name = name.replace('Label', 'Probability').replace('label', 'probability')
                        if prob_name in prediction:
                            probs = prediction[prob_name]
                            if isinstance(probs, dict):
                                prob = float(probs.get(label_str, 0.0))
                                if prob > best_prob:
                                    best_prob = prob
                                    best_label = label_str
                    break
            
            # Fallback: check probability dictionary
            if best_label is None:
                for name in output_names:
                    if 'prob' in name.lower():
                        probs = prediction.get(name, {})
                        if isinstance(probs, dict):
                            for label, prob in probs.items():
                                label_str = str(label).lower()
                                if label_str in MALIGNANT_TYPES:
                                    prob_val = float(prob)
                                    if prob_val > best_prob:
                                        best_prob = prob_val
                                        best_label = label_str
                        break
            
            # Fallback: array format
            if best_label is None:
                for name in output_names:
                    output = prediction.get(name)
                    if hasattr(output, 'shape') or isinstance(output, (list, np.ndarray)):
                        arr = np.asarray(output)
                        probs = softmax(arr.flatten())
                        
                        for idx, prob in enumerate(probs):
                            if idx < len(self.class_labels):
                                label = self.class_labels[idx].lower()
                                if label in MALIGNANT_TYPES and prob > best_prob:
                                    best_prob = float(prob)
                                    best_label = label
            
            if best_label:
                diagnosis = LESION_TYPES.get(best_label, best_label)
                if self.debug:
                    print(f"[Stage 2] Result: {diagnosis} (conf={best_prob:.3f})")
                return diagnosis, best_prob
            else:
                if self.debug:
                    print(f"[Stage 2] No malignant type found, returning generic")
                return "Malignant (Requires Medical Evaluation)", 0.5
        except Exception as e:
            print(f"❌ Error in Stage 2: {e}")
            return "Malignant (Requires Medical Evaluation)", 0.5

    def predict_pipeline(self, image):
        """
        2-stage pipeline (assuming mole detected):
        1. Stage 1: Binary classifier → if benign, return "Benign"
        2. Stage 2: Multiclass classifier → return specific diagnosis
        """
        # Stage 1: Binary Classification
        is_malignant, binary_confidence = self.predict_stage1(image)
        if not is_malignant:
            return "Benign", binary_confidence
        
        # Stage 2: Multiclass Classification
        diagnosis, multiclass_confidence = self.predict_stage2(image)
        return diagnosis, multiclass_confidence

    def evaluate_on_dataset(self, dataset_path, metadata_file, max_samples=None):
        """Evaluate pipeline on dataset"""
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
                image_paths = (
                    list(dataset_path.glob("*.jpg")) +
                    list(dataset_path.glob("*.jpeg")) +
                    list(dataset_path.glob("*.png"))
                )
        
        if max_samples:
            image_paths = image_paths[:max_samples]
        
        print(f"\nEvaluating on {len(image_paths)} images...")
        
        predictions = []
        labels = []
        predictions_detailed = []
        labels_detailed = []
        confidences = []
        
        for img_path in tqdm(image_paths, desc="Processing"):
            try:
                image_id = img_path.stem
                img = Image.open(img_path).convert("RGB").resize((224, 224))
                
                # Run pipeline
                pred, confidence = self.predict_pipeline(img)
                
                # Get ground truth
                true_label = image_id_to_label.get(image_id, '').lower()
                
                # Map to binary
                pred_bin = 'malignant' if pred != "Benign" else 'benign'
                label_bin = 'malignant' if true_label in CANCER_LABELS else 'benign'
                
                predictions.append(pred_bin)
                labels.append(label_bin)
                predictions_detailed.append(pred)
                labels_detailed.append(true_label)
                confidences.append(confidence)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate metrics
        acc = accuracy_score(labels, predictions)
        
        results = {
            'accuracy': acc,
            'classification_report': classification_report(labels, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, predictions),
            'predictions': predictions,
            'labels': labels,
            'predictions_detailed': predictions_detailed,
            'labels_detailed': labels_detailed,
            'confidences': confidences,
            'avg_confidence': np.mean(confidences)
        }
        
        return results


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Test Stage 1-2 pipeline (assuming mole detected)')
    parser.add_argument('--dataset', type=str, 
                       default="/Users/t/Downloads/archive/HAM10000_images_part_1",
                       help='Path to dataset directory')
    parser.add_argument('--dataset2', type=str,
                       default="/Users/t/Downloads/archive/HAM10000_images_part_2",
                       help='Path to second dataset directory')
    parser.add_argument('--metadata', type=str,
                       default="/Users/t/Downloads/archive/HAM10000_metadata.csv",
                       help='Path to metadata CSV')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples to evaluate')
    parser.add_argument('--binary-model', type=str, default=None,
                       help='Path to binary classifier model')
    parser.add_argument('--multiclass-model', type=str, default=None,
                       help='Path to multiclass classifier model')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--validate', action='store_true',
                       help='Validate models on startup')
    parser.add_argument('--malignancy-threshold', type=float, default=0.5,
                       help='Threshold for classifying as malignant (default: 0.5, lower = more sensitive)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("STAGE 1-2 PIPELINE TESTER (Assuming Mole Detected)")
    print("="*60)
    print(f"Malignancy Threshold: {args.malignancy_threshold} (lower = more sensitive to cancer)")
    print("="*60)
    
    # Initialize tester
    tester = Stage1_2_Tester(
        binary_model_path=args.binary_model,
        multiclass_model_path=args.multiclass_model,
        debug=args.debug or args.validate,
        malignancy_threshold=args.malignancy_threshold
    )
    
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
        print("❌ No images found! Check dataset paths.")
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
    print("RESULTS")
    print("="*60)
    print(f"\nBinary Accuracy (Benign vs Malignant): {results['accuracy']*100:.2f}%")
    print(f"Average Confidence: {results['avg_confidence']*100:.2f}%\n")
    print("Classification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("="*60)


if __name__ == "__main__":
    main()