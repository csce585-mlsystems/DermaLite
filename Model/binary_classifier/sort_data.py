import pandas as pd
import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = "./data/malignancy_dataset"

# 1. HAM10000 CONFIG
HAM_BASE_DIR = "./data/HAM10000_dataset"
HAM_PART_1 = os.path.join(HAM_BASE_DIR, "HAM10000_images_part_1")
HAM_PART_2 = os.path.join(HAM_BASE_DIR, "HAM10000_images_part_2")
HAM_METADATA = os.path.join(HAM_BASE_DIR, "HAM10000_metadata.csv")

# 2. PAD-UFES-20 CONFIG
# Expects a folder named 'PAD-UFES-20' inside 'data'
PAD_BASE_DIR = "./data/PAD-UFES-20" 
PAD_IMAGES = os.path.join(PAD_BASE_DIR, "images") # Often images are in a subfolder or directly in root
PAD_METADATA = os.path.join(PAD_BASE_DIR, "metadata.csv")

def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "benign"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "malignant"), exist_ok=True)

def sort_ham10000():
    print("\n--- 🗂️ Sorting HAM10000 (Dermoscopy) ---")
    if not os.path.exists(HAM_METADATA):
        print(f"⚠️  Skipping HAM10000: Metadata not found at {HAM_METADATA}")
        return

    df = pd.read_csv(HAM_METADATA)
    
    # HAM Mapping
    malignant_codes = ['mel', 'bcc', 'akiec'] # Melanoma, Basal Cell, Actinic/SCC
    benign_codes = ['nv', 'bkl', 'vasc', 'df'] # Nevi, Keratosis, Vascular, Dermatofibroma

    count = 0
    skipped_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row['image_id']
        dx = row['dx']
        
        # Determine Label
        if dx in malignant_codes:
            label = "malignant"
        elif dx in benign_codes:
            label = "benign"
        else:
            continue

        filename = f"{image_id}.jpg"
        
        # Find Source
        src_path = os.path.join(HAM_PART_1, filename)
        if not os.path.exists(src_path):
            src_path = os.path.join(HAM_PART_2, filename)
            if not os.path.exists(src_path):
                # Try PNG
                src_path = os.path.join(HAM_PART_1, f"{image_id}.png")
                if not os.path.exists(src_path):
                     src_path = os.path.join(HAM_PART_2, f"{image_id}.png")
                     if not os.path.exists(src_path): continue

        # Copy with Prefix to avoid collisions
        dest_path = os.path.join(OUTPUT_DIR, label, f"HAM_{filename}")
        
        # Elegant Repetition Check: Skip if already exists
        if os.path.exists(dest_path):
            skipped_count += 1
            continue

        shutil.copy2(src_path, dest_path)
        count += 1
    print(f"✅ HAM10000: Sorted {count} new images (Skipped {skipped_count} existing).")

def sort_pad_ufes():
    print("\n--- 🗂️ Sorting PAD-UFES-20 (Smartphone) ---")
    if not os.path.exists(PAD_METADATA):
        print(f"⚠️  Skipping PAD-UFES-20: Metadata not found at {PAD_METADATA}")
        print("    (Ensure your folder is named 'PAD-UFES-20' and has 'metadata.csv')")
        return

    df = pd.read_csv(PAD_METADATA)
    
    # PAD-UFES-20 Mapping
    # BCC: Basal Cell Carcinoma (Malignant)
    # SCC: Squamous Cell Carcinoma (Malignant)
    # ACK: Actinic Keratosis (Pre-Malignant/Malignant)
    # SEK: Seborrheic Keratosis (Benign)
    # BOD: Bowen's Disease (Malignant SCC in situ)
    # MEL: Melanoma (Malignant)
    # NEV: Nevus (Benign)
    
    malignant_codes = ['BCC', 'SCC', 'ACK', 'BOD', 'MEL']
    benign_codes = ['SEK', 'NEV']

    count = 0
    skipped_count = 0
    # Determine image folder location (sometimes images are flat, sometimes in subfolders)
    # We assume standard structure or search recursively if needed
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['img_id']
        diagnostic = row['diagnostic']
        
        # Determine Label
        if diagnostic in malignant_codes:
            label = "malignant"
        elif diagnostic in benign_codes:
            label = "benign"
        else:
            continue
            
        # PAD filenames often look like 'PAT_1516_1765_539.png'
        # Adjust if your CSV has different ID format
        filename = img_id 
        
        # Search for file
        src_path = os.path.join(PAD_IMAGES, filename)
        if not os.path.exists(src_path):
            # Try recursive search if flat path fails
            # (Some datasets group by patient)
            continue 

        # Copy with Prefix
        dest_path = os.path.join(OUTPUT_DIR, label, f"PAD_{filename}")
        
        # Elegant Repetition Check: Skip if already exists
        if os.path.exists(dest_path):
            skipped_count += 1
            continue

        shutil.copy2(src_path, dest_path)
        count += 1
        
    print(f"✅ PAD-UFES-20: Sorted {count} new images (Skipped {skipped_count} existing).")

if __name__ == "__main__":
    ensure_dirs()
    # sort_ham10000() # Commented out as requested since HAM is already processed
    sort_pad_ufes()