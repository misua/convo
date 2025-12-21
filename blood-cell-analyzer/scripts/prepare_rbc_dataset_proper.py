#!/usr/bin/env python3
"""
Proper RBC Dataset Preparation for CNN Training
NO SHORTCUTS - Uses only verified labeled data from trusted sources

Dataset sources:
1. NIH Malaria Dataset (peer-reviewed, expert labeled)
   - Uninfected cells → "normal" class
   - Parasitized cells → "infected" class
   
2. Kaggle Sickle Cell Dataset
   - Positive/Labelled → "sickle" class
   - Requires manual QA for quality verification

Output: 3-class CNN training dataset (normal, infected, sickle)
Format: data/rbc_extended_3class/{train,val,test}/{normal,infected,sickle}/
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json
import random
import shutil
from collections import Counter
from typing import List, Tuple, Dict
import hashlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Constants
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "rbc_extended_3class"
TRAIN_DIR = OUTPUT_ROOT / "train"
VAL_DIR = OUTPUT_ROOT / "val"
TEST_DIR = OUTPUT_ROOT / "test"

# Target image size for ResNet34
TARGET_SIZE = (224, 224)

# Class names
CLASSES = ["normal", "infected", "sickle"]

# Dataset balancing config
MAX_SAMPLES_PER_CLASS = 3000  # Downsample large classes
MIN_SAMPLES_FOR_TRAINING = 100  # Minimum to be viable

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def setup_directories():
    """Create clean directory structure."""
    print("\n" + "="*70)
    print("PHASE 2: RBC DATASET PREPARATION (PROPER METHODOLOGY)")
    print("="*70)
    
    # Remove old data if exists
    if OUTPUT_ROOT.exists():
        print(f"\nRemoving existing output directory: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    
    # Create fresh directories
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for cls in CLASSES:
            (split_dir / cls).mkdir(exist_ok=True)
    
    print(f"✓ Created clean directory structure at {OUTPUT_ROOT}")


def load_and_validate_image(img_path: Path) -> Tuple[bool, np.ndarray]:
    """
    Load and validate image quality.
    
    Returns:
        (is_valid, image_array)
    """
    try:
        # Load image
        img = cv2.imread(str(img_path))
        
        if img is None:
            return False, None
        
        # Check minimum size
        if img.shape[0] < 50 or img.shape[1] < 50:
            return False, None
        
        # Check if not completely black/white
        mean_intensity = np.mean(img)
        if mean_intensity < 10 or mean_intensity > 245:
            return False, None
        
        # Check if has color variance
        std_intensity = np.std(img)
        if std_intensity < 5:
            return False, None
        
        return True, img
        
    except Exception as e:
        print(f"  ✗ Error loading {img_path.name}: {e}")
        return False, None


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size maintaining aspect ratio with padding.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def compute_image_hash(img: np.ndarray) -> str:
    """Compute hash of image to detect duplicates."""
    return hashlib.md5(img.tobytes()).hexdigest()


def load_class_images(source_dir: Path, class_name: str, 
                      max_samples: int = None) -> List[Tuple[np.ndarray, str]]:
    """
    Load and validate images from a directory.
    
    Args:
        source_dir: Directory containing images
        class_name: Class label for these images
        max_samples: Maximum number of samples to load (for balancing)
    
    Returns:
        List of (image_array, filename) tuples
    """
    print(f"\n--- Loading {class_name} class ---")
    print(f"Source: {source_dir}")
    
    if not source_dir.exists():
        print(f"✗ Source directory not found!")
        return []
    
    # Find all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG']:
        image_files.extend(list(source_dir.glob(ext)))
    
    print(f"Found {len(image_files)} image files")
    
    # Shuffle for random sampling
    random.shuffle(image_files)
    
    # Load and validate
    valid_images = []
    invalid_count = 0
    seen_hashes = set()  # Detect duplicates
    
    for img_path in image_files:
        # Check if we have enough samples
        if max_samples and len(valid_images) >= max_samples:
            print(f"Reached maximum of {max_samples} samples")
            break
        
        # Load and validate
        is_valid, img = load_and_validate_image(img_path)
        
        if not is_valid:
            invalid_count += 1
            continue
        
        # Check for duplicates
        img_hash = compute_image_hash(img)
        if img_hash in seen_hashes:
            continue
        seen_hashes.add(img_hash)
        
        # Resize to target size
        img_resized = resize_image(img, TARGET_SIZE)
        
        valid_images.append((img_resized, img_path.stem))
        
        # Progress update
        if len(valid_images) % 500 == 0:
            print(f"  Loaded {len(valid_images)} valid images...")
    
    print(f"✓ Loaded {len(valid_images)} valid images")
    print(f"  Invalid/corrupted: {invalid_count}")
    print(f"  Duplicates removed: {len(image_files) - invalid_count - len(valid_images)}")
    
    return valid_images


def split_and_save_class(images: List[Tuple[np.ndarray, str]], class_name: str):
    """
    Split class data into train/val/test and save.
    
    Args:
        images: List of (image_array, filename) tuples
        class_name: Class label
    """
    if len(images) < MIN_SAMPLES_FOR_TRAINING:
        print(f"✗ WARNING: {class_name} has only {len(images)} samples (minimum {MIN_SAMPLES_FOR_TRAINING})")
        print(f"  This class may not train well!")
    
    # Shuffle
    random.shuffle(images)
    
    # Calculate split indices
    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]
    
    print(f"\n{class_name} split:")
    print(f"  Train: {len(train_images)}")
    print(f"  Val:   {len(val_images)}")
    print(f"  Test:  {len(test_images)}")
    
    # Save images
    def save_images(image_list, split_dir):
        saved = 0
        for idx, (img, orig_filename) in enumerate(image_list):
            filename = f"{class_name}_{orig_filename}_{idx:05d}.jpg"
            save_path = split_dir / class_name / filename
            try:
                cv2.imwrite(str(save_path), img)
                saved += 1
            except Exception as e:
                print(f"  ✗ Failed to save {filename}: {e}")
        return saved
    
    n_train_saved = save_images(train_images, TRAIN_DIR)
    n_val_saved = save_images(val_images, VAL_DIR)
    n_test_saved = save_images(test_images, TEST_DIR)
    
    print(f"✓ Saved: {n_train_saved} train, {n_val_saved} val, {n_test_saved} test")


def manual_qa_samples(class_name: str, sample_dir: Path, n_samples: int = 5):
    """
    Display sample images for manual quality check.
    
    Args:
        class_name: Class being checked
        sample_dir: Directory containing class images
        n_samples: Number of samples to show
    """
    print(f"\n--- Manual QA for {class_name} ---")
    
    sample_files = list(sample_dir.glob("*.jpg"))[:n_samples]
    
    if not sample_files:
        print(f"✗ No samples found for QA")
        return
    
    print(f"Sample image paths (inspect manually):")
    for idx, img_path in enumerate(sample_files, 1):
        print(f"  {idx}. {img_path}")
    
    print(f"\nPlease manually inspect these {n_samples} sample images to verify:")
    print(f"  - Images are actually {class_name} cells")
    print(f"  - Image quality is acceptable")
    print(f"  - No obvious mislabeling or corruption")


def generate_dataset_manifest():
    """Generate comprehensive dataset manifest."""
    print("\n" + "="*70)
    print("DATASET MANIFEST")
    print("="*70)
    
    manifest = {
        "dataset_name": "RBC 3-Class CNN Training Dataset",
        "classes": CLASSES,
        "image_size": TARGET_SIZE,
        "splits": {}
    }
    
    # Collect statistics
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_ROOT / split
        manifest["splits"][split] = {}
        
        for cls in CLASSES:
            cls_dir = split_dir / cls
            count = len(list(cls_dir.glob("*.jpg")))
            manifest["splits"][split][cls] = count
    
    # Calculate totals
    manifest["totals"] = {}
    for cls in CLASSES:
        total = sum(manifest["splits"][split][cls] for split in ["train", "val", "test"])
        manifest["totals"][cls] = total
    
    # Print table
    print(f"\n{'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    for cls in CLASSES:
        train_c = manifest["splits"]["train"][cls]
        val_c = manifest["splits"]["val"][cls]
        test_c = manifest["splits"]["test"][cls]
        total_c = manifest["totals"][cls]
        print(f"{cls:<15} {train_c:<10} {val_c:<10} {test_c:<10} {total_c:<10}")
    
    # Print totals row
    train_total = sum(manifest["splits"]["train"][cls] for cls in CLASSES)
    val_total = sum(manifest["splits"]["val"][cls] for cls in CLASSES)
    test_total = sum(manifest["splits"]["test"][cls] for cls in CLASSES)
    grand_total = sum(manifest["totals"].values())
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {train_total:<10} {val_total:<10} {test_total:<10} {grand_total:<10}")
    
    # Check class balance
    print("\n--- Class Balance Analysis ---")
    max_count = max(manifest["totals"].values())
    min_count = min(manifest["totals"].values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Largest class: {max_count} samples")
    print(f"Smallest class: {min_count} samples")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("⚠️  WARNING: Significant class imbalance detected!")
        print("   Consider using class weights during training.")
    else:
        print("✓ Class balance is reasonable")
    
    # Save manifest
    manifest_file = OUTPUT_ROOT / "dataset_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest saved to {manifest_file}")
    
    # Data sources documentation
    print("\n--- Data Sources ---")
    print("1. Normal class:")
    print("   Source: NIH Malaria Dataset - Uninfected cells")
    print("   URL: https://lhncbc.nlm.nih.gov/publication/pub9932")
    print("   Quality: ✓ Peer-reviewed, expert labeled")
    
    print("\n2. Infected class:")
    print("   Source: NIH Malaria Dataset - Parasitized cells")
    print("   URL: https://lhncbc.nlm.nih.gov/publication/pub9932")
    print("   Quality: ✓ Peer-reviewed, expert labeled")
    
    print("\n3. Sickle class:")
    print("   Source: Kaggle Sickle Cell Disease Dataset")
    print("   URL: https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset")
    print("   Quality: ⚠️  Requires manual verification")
    
    return manifest


def main():
    """Main dataset preparation workflow."""
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Step 1: Setup
    setup_directories()
    
    # Step 2: Load normal class (NIH Malaria Uninfected)
    normal_dir = DATA_ROOT / "malaria_dataset" / "cell_images" / "Uninfected"
    normal_images = load_class_images(normal_dir, "normal", max_samples=MAX_SAMPLES_PER_CLASS)
    
    # Step 3: Load infected class (NIH Malaria Parasitized)
    infected_dir = DATA_ROOT / "malaria_dataset" / "cell_images" / "Parasitized"
    infected_images = load_class_images(infected_dir, "infected", max_samples=MAX_SAMPLES_PER_CLASS)
    
    # Step 4: Load sickle class (Kaggle Sickle Cell)
    sickle_dir = DATA_ROOT / "sickle_cell_dataset" / "Positive" / "Labelled"
    sickle_images = load_class_images(sickle_dir, "sickle", max_samples=MAX_SAMPLES_PER_CLASS)
    
    # Check if we have minimum data
    if len(normal_images) < MIN_SAMPLES_FOR_TRAINING:
        print(f"\n✗ ERROR: Insufficient normal samples ({len(normal_images)} < {MIN_SAMPLES_FOR_TRAINING})")
        return
    
    if len(infected_images) < MIN_SAMPLES_FOR_TRAINING:
        print(f"\n✗ ERROR: Insufficient infected samples ({len(infected_images)} < {MIN_SAMPLES_FOR_TRAINING})")
        return
    
    if len(sickle_images) < MIN_SAMPLES_FOR_TRAINING:
        print(f"\n✗ ERROR: Insufficient sickle samples ({len(sickle_images)} < {MIN_SAMPLES_FOR_TRAINING})")
        print(f"   Consider data augmentation or finding additional sickle cell dataset")
        return
    
    # Step 5: Split and save
    print("\n" + "="*70)
    print("SPLITTING AND SAVING DATASET")
    print("="*70)
    
    split_and_save_class(normal_images, "normal")
    split_and_save_class(infected_images, "infected")
    split_and_save_class(sickle_images, "sickle")
    
    # Step 6: Generate manifest and statistics
    manifest = generate_dataset_manifest()
    
    # Step 7: Manual QA prompt
    print("\n" + "="*70)
    print("MANUAL QUALITY ASSURANCE REQUIRED")
    print("="*70)
    
    manual_qa_samples("normal", TRAIN_DIR / "normal")
    manual_qa_samples("infected", TRAIN_DIR / "infected")
    manual_qa_samples("sickle", TRAIN_DIR / "sickle")
    
    print("\n" + "="*70)
    print("✓ DATASET PREPARATION COMPLETE")
    print("="*70)
    
    print(f"\nDataset location: {OUTPUT_ROOT}")
    print(f"\nNext steps:")
    print(f"1. Manually inspect sample images from each class")
    print(f"2. Verify data quality and labeling accuracy")
    print(f"3. Proceed to Phase 3: Create training script")
    print(f"4. Then Phase 4: Train ResNet34 model")


if __name__ == "__main__":
    main()
