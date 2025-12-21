#!/usr/bin/env python3
"""
Prepare RBC Extended Dataset for Training
Extracts RBC crops from BCCD and organizes with malaria/sickle datasets into 9 classes
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml
import shutil
from collections import Counter
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.models.shape_classifier import RBCShapeClassifier, RBCShape

# Constants
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = DATA_ROOT / "rbc_extended"
TRAIN_DIR = OUTPUT_ROOT / "train"
VAL_DIR = OUTPUT_ROOT / "val"
TEST_DIR = OUTPUT_ROOT / "test"

# Class folders
CLASSES = ["normal", "microcyte", "target", "teardrop", "spherocyte", "irregular", "infected", "sickle"]

def setup_directories():
    """Create directory structure for organized dataset."""
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for cls in CLASSES:
            (split_dir / cls).mkdir(exist_ok=True)
    print(f"✓ Created directory structure at {OUTPUT_ROOT}")


def extract_rbc_crops_from_bccd():
    """Extract RBC crops from BCCD dataset using YOLOv11-seg."""
    print("\n=== Extracting RBC crops from BCCD ===")
    
    # Load segmentation model
    model_path = PROJECT_ROOT / "yolo11n-seg.pt"
    model = YOLO(str(model_path))
    
    # Load rule-based classifier for categorization
    classifier = RBCShapeClassifier(use_deep_learning=False)
    
    # Get BCCD images
    bccd_images_dir = DATA_ROOT / "bccd_yolo" / "train" / "images"
    if not bccd_images_dir.exists():
        print(f"✗ BCCD images not found at {bccd_images_dir}")
        return {}
    
    image_files = list(bccd_images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} BCCD images")
    
    class_counts = Counter()
    crops_saved = []
    
    for idx, img_path in enumerate(image_files):
        if idx % 10 == 0:
            print(f"Processing image {idx+1}/{len(image_files)}...")
        
        # Run detection
        results = model(str(img_path), conf=0.3, verbose=False)
        
        if len(results) == 0 or results[0].boxes is None:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Process each detected cell
        for detection in results[0].boxes:
            cls_id = int(detection.cls[0])
            
            # Only process RBCs (class 0)
            if cls_id != 0:
                continue
            
            # Get bounding box
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            
            # Crop RBC with padding
            padding = 10
            y1 = max(0, y1 - padding)
            y2 = min(img.shape[0], y2 + padding)
            x1 = max(0, x1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            
            # Calculate metrics and classify shape
            metrics = classifier.calculate_metrics(crop)
            shape, confidence = classifier.classify(metrics, crop)
            
            # Map to simplified classes
            if shape == RBCShape.NORMAL:
                class_name = "normal"
            elif shape == RBCShape.MICROCYTE:
                class_name = "microcyte"
            elif shape == RBCShape.TARGET:
                class_name = "target"
            elif shape == RBCShape.TEARDROP:
                class_name = "teardrop"
            elif shape == RBCShape.SPHEROCYTE:
                class_name = "spherocyte"
            else:
                class_name = "irregular"
            
            class_counts[class_name] += 1
            crops_saved.append((crop, class_name, img_path.stem))
    
    print(f"\n✓ Extracted {len(crops_saved)} RBC crops from BCCD")
    print(f"Distribution: {dict(class_counts)}")
    
    return crops_saved


def organize_malaria_dataset():
    """Organize NIH malaria dataset images."""
    print("\n=== Organizing Malaria Dataset ===")
    
    malaria_root = DATA_ROOT / "malaria_dataset" / "cell_images"
    parasitized_dir = malaria_root / "Parasitized"
    
    if not parasitized_dir.exists():
        print(f"✗ Malaria parasitized dir not found at {parasitized_dir}")
        return []
    
    # Get all parasitized images
    parasitized_files = list(parasitized_dir.glob("*.png"))
    print(f"Found {len(parasitized_files)} infected malaria images")
    
    # Load and prepare images
    crops_saved = []
    for img_path in parasitized_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            crops_saved.append((img, "infected", img_path.stem))
    
    print(f"✓ Loaded {len(crops_saved)} malaria infected cells")
    return crops_saved


def organize_sickle_dataset():
    """Organize sickle cell dataset images."""
    print("\n=== Organizing Sickle Cell Dataset ===")
    
    sickle_dir = DATA_ROOT / "sickle_cell_dataset" / "Positive" / "Labelled"
    
    if not sickle_dir.exists():
        print(f"✗ Sickle cell dir not found at {sickle_dir}")
        return []
    
    # Get all sickle cell images
    sickle_files = list(sickle_dir.glob("*.jpg")) + list(sickle_dir.glob("*.png"))
    print(f"Found {len(sickle_files)} sickle cell images")
    
    crops_saved = []
    for img_path in sickle_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            crops_saved.append((img, "sickle", img_path.stem))
    
    print(f"✓ Loaded {len(crops_saved)} sickle cells")
    return crops_saved


def split_and_save_dataset(all_crops):
    """Split dataset into train/val/test and save images."""
    print("\n=== Splitting and Saving Dataset ===")
    
    # Group by class
    class_crops = {cls: [] for cls in CLASSES}
    for crop, class_name, filename in all_crops:
        class_crops[class_name].append((crop, filename))
    
    # Print distribution
    print("\nClass distribution:")
    for cls, crops in class_crops.items():
        print(f"  {cls}: {len(crops)}")
    
    # Split each class: 80% train, 10% val, 10% test
    for cls, crops in class_crops.items():
        random.shuffle(crops)
        n = len(crops)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        train_crops = crops[:n_train]
        val_crops = crops[n_train:n_train + n_val]
        test_crops = crops[n_train + n_val:]
        
        # Save train
        for idx, (crop, filename) in enumerate(train_crops):
            save_path = TRAIN_DIR / cls / f"{cls}_{filename}_{idx:04d}.jpg"
            cv2.imwrite(str(save_path), crop)
        
        # Save val
        for idx, (crop, filename) in enumerate(val_crops):
            save_path = VAL_DIR / cls / f"{cls}_{filename}_{idx:04d}.jpg"
            cv2.imwrite(str(save_path), crop)
        
        # Save test
        for idx, (crop, filename) in enumerate(test_crops):
            save_path = TEST_DIR / cls / f"{cls}_{filename}_{idx:04d}.jpg"
            cv2.imwrite(str(save_path), crop)
        
        print(f"  {cls}: {len(train_crops)} train, {len(val_crops)} val, {len(test_crops)} test")
    
    print(f"\n✓ Dataset saved to {OUTPUT_ROOT}")


def generate_dataset_stats():
    """Generate and save dataset statistics."""
    print("\n=== Dataset Statistics ===")
    
    stats = {
        "train": {},
        "val": {},
        "test": {},
        "total": {}
    }
    
    for split in ["train", "val", "test"]:
        split_dir = OUTPUT_ROOT / split
        for cls in CLASSES:
            cls_dir = split_dir / cls
            count = len(list(cls_dir.glob("*.jpg")))
            stats[split][cls] = count
            stats["total"][cls] = stats["total"].get(cls, 0) + count
    
    # Print table
    print(f"\n{'Class':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 55)
    for cls in CLASSES:
        train_count = stats["train"][cls]
        val_count = stats["val"][cls]
        test_count = stats["test"][cls]
        total_count = stats["total"][cls]
        print(f"{cls:<15} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")
    
    # Save to file
    stats_file = OUTPUT_ROOT / "dataset_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"{'Class':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}\n")
        f.write("-" * 55 + "\n")
        for cls in CLASSES:
            f.write(f"{cls:<15} {stats['train'][cls]:<8} {stats['val'][cls]:<8} {stats['test'][cls]:<8} {stats['total'][cls]:<8}\n")
    
    print(f"\n✓ Statistics saved to {stats_file}")
    return stats


def main():
    print("=" * 60)
    print("RBC Extended Dataset Preparation")
    print("=" * 60)
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Extract from BCCD
    bccd_crops = extract_rbc_crops_from_bccd()
    
    # Step 3: Organize malaria
    malaria_crops = organize_malaria_dataset()
    
    # Step 4: Organize sickle
    sickle_crops = organize_sickle_dataset()
    
    # Step 5: Combine all
    all_crops = bccd_crops + malaria_crops + sickle_crops
    print(f"\n✓ Total crops collected: {len(all_crops)}")
    
    # Step 6: Split and save
    split_and_save_dataset(all_crops)
    
    # Step 7: Generate stats
    generate_dataset_stats()
    
    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
