#!/usr/bin/env python3
"""
Convert BBBC041 Parsed Annotations to YOLO Format

Converts parsed BBBC041 annotations to YOLO training format:
- Creates train/val split (90/10 from training set, test set separate)
- Converts bounding boxes from [x, y, w, h] to YOLO format [x_center, y_center, w, h] normalized
- Creates YOLO label files (.txt) with format: class_id x_center y_center width height
- Organizes images and labels into train/val/test directories
- Creates data.yaml configuration file

YOLO Classes (7 total):
0: red blood cell (uninfected)
1: leukocyte (white blood cell)
2: ring (early malaria stage)
3: trophozoite (feeding stage)
4: schizont (reproducing stage)
5: gametocyte (sexual stage)
6: difficult (unclear/ambiguous)
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List
import random


# Class mapping: BBBC041 category name -> YOLO class ID
CLASS_MAPPING = {
    'red blood cell': 0,
    'leukocyte': 1,
    'ring': 2,
    'trophozoite': 3,
    'schizont': 4,
    'gametocyte': 5,
    'difficult': 6
}


def convert_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert COCO bbox [x, y, w, h] to YOLO format [x_center, y_center, w, h] normalized.
    
    Args:
        bbox: [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [x_center, y_center, width, height] normalized to [0, 1]
    """
    x, y, w, h = bbox
    
    # Calculate center coordinates
    x_center = x + w / 2
    y_center = y + h / 2
    
    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center_norm, y_center_norm, w_norm, h_norm]


def create_yolo_label(annotation: Dict, output_path: Path):
    """
    Create YOLO label file for a single image.
    
    Label format (one line per cell):
    class_id x_center y_center width height
    """
    img_width = annotation['width']
    img_height = annotation['height']
    
    lines = []
    for cell in annotation['cells']:
        category = cell['category']
        class_id = CLASS_MAPPING.get(category, 6)  # Default to 'difficult' if unknown
        
        # Convert bbox to YOLO format
        bbox_coco = cell['bbox']
        bbox_yolo = convert_bbox_to_yolo(bbox_coco, img_width, img_height)
        
        # Format: class x_center y_center width height
        line = f"{class_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}"
        lines.append(line)
    
    # Write label file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def copy_image(src_img_dir: Path, img_filename: str, dest_img_dir: Path):
    """Copy image file to destination directory."""
    src_path = src_img_dir / img_filename
    
    # Handle both .png and .jpg files
    if not src_path.exists():
        # Try alternative extension
        alt_ext = '.png' if src_path.suffix == '.jpg' else '.jpg'
        alt_path = src_path.with_suffix(alt_ext)
        if alt_path.exists():
            src_path = alt_path
        else:
            print(f"⚠️  Image not found: {src_path}")
            return False
    
    dest_path = dest_img_dir / src_path.name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    return True


def create_data_yaml(output_dir: Path, class_names: List[str]):
    """
    Create YOLO data.yaml configuration file.
    
    Format:
    path: /absolute/path/to/dataset
    train: train/images
    val: val/images
    test: test/images
    
    nc: 7
    names: ['red blood cell', 'leukocyte', ...]
    """
    yaml_content = f"""# BBBC041 Malaria Dataset - YOLO Format
# Generated from Broad Bioimage Benchmark Collection

path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(class_names)}
names: {class_names}

# Class distribution (see dataset_stats.json for details)
# Total cells: 86,035
# - Red blood cells: 96.51% (uninfected)
# - Parasites: 2.85% (ring: 0.61%, trophozoite: 1.84%, schizont: 0.22%, gametocyte: 0.18%)
# - Leukocytes: 0.12%
# - Difficult: 0.52%
"""
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"💾 Created data.yaml at {yaml_path}")


def main():
    """Convert BBBC041 parsed annotations to YOLO format."""
    
    # Paths
    parsed_dir = Path("data/bbbc041_malaria/parsed")
    images_dir = Path("data/bbbc041_malaria/malaria/images")
    output_dir = Path("data/bbbc041_yolo")
    
    train_parsed = parsed_dir / "training_parsed.json"
    test_parsed = parsed_dir / "test_parsed.json"
    
    # Check input files
    if not train_parsed.exists():
        print(f"❌ Training annotations not found: {train_parsed}")
        return
    if not test_parsed.exists():
        print(f"❌ Test annotations not found: {test_parsed}")
        return
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    print("=" * 70)
    print("BBBC041 TO YOLO CONVERTER")
    print("=" * 70)
    
    # Load parsed annotations
    print("\n📖 Loading parsed annotations...")
    with open(train_parsed, 'r') as f:
        train_data = json.load(f)
    with open(test_parsed, 'r') as f:
        test_data = json.load(f)
    
    print(f"   Training samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    
    # Create train/val split (90/10)
    random.seed(42)  # Reproducible split
    random.shuffle(train_data)
    
    split_idx = int(len(train_data) * 0.9)
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]
    
    print(f"\n✂️  Split training set:")
    print(f"   Train: {len(train_split)} images (90%)")
    print(f"   Val: {len(val_split)} images (10%)")
    
    # Process each split
    splits = {
        'train': train_split,
        'val': val_split,
        'test': test_data
    }
    
    stats = {}
    
    for split_name, split_data in splits.items():
        print(f"\n🔄 Processing {split_name} split...")
        
        img_dir = output_dir / split_name / "images"
        label_dir = output_dir / split_name / "labels"
        
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        cell_count = 0
        
        for annotation in split_data:
            filename = annotation['filename']
            
            # Copy image
            success = copy_image(images_dir, filename, img_dir)
            if not success:
                continue
            
            # Create YOLO label
            label_filename = Path(filename).stem + '.txt'
            label_path = label_dir / label_filename
            create_yolo_label(annotation, label_path)
            
            copied_count += 1
            cell_count += len(annotation['cells'])
        
        stats[split_name] = {
            'images': copied_count,
            'cells': cell_count
        }
        
        print(f"   ✅ {copied_count} images, {cell_count} cells")
    
    # Create data.yaml
    print("\n📝 Creating data.yaml configuration...")
    class_names = [name for name, _ in sorted(CLASS_MAPPING.items(), key=lambda x: x[1])]
    create_data_yaml(output_dir, class_names)
    
    # Summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    
    total_images = sum(s['images'] for s in stats.values())
    total_cells = sum(s['cells'] for s in stats.values())
    
    print(f"\n✅ Successfully converted:")
    print(f"   Total images: {total_images}")
    print(f"   Total cells: {total_cells}")
    print(f"\n   Train: {stats['train']['images']} images, {stats['train']['cells']} cells")
    print(f"   Val:   {stats['val']['images']} images, {stats['val']['cells']} cells")
    print(f"   Test:  {stats['test']['images']} images, {stats['test']['cells']} cells")
    
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print(f"\n🎯 Ready for YOLOv11 training!")


if __name__ == "__main__":
    main()
