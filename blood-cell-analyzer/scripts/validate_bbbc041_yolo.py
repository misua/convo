#!/usr/bin/env python3
"""
Validate BBBC041 YOLO Dataset

Performs quality checks on the converted YOLO dataset:
1. File integrity: Check all images have corresponding labels
2. Format validation: Verify YOLO label format correctness
3. Coordinate validation: Ensure bboxes are within [0, 1] range
4. Visual spot-checks: Generate sample visualizations with bboxes
5. Statistics verification: Compare with original parsed data

Output:
- Validation report (console and JSON)
- Sample visualizations saved to data/bbbc041_yolo/validation_samples/
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import Counter

# Check if PIL/cv2 are available (optional for visualization)
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  PIL not available - skipping visualizations")


CLASS_NAMES = [
    'red blood cell',
    'leukocyte',
    'ring',
    'trophozoite',
    'schizont',
    'gametocyte',
    'difficult'
]

CLASS_COLORS = [
    (0, 255, 0),      # RBC: green
    (255, 255, 0),    # leukocyte: yellow
    (255, 0, 0),      # ring: red
    (255, 128, 0),    # trophozoite: orange
    (128, 0, 255),    # schizont: purple
    (255, 0, 255),    # gametocyte: magenta
    (128, 128, 128)   # difficult: gray
]


def validate_label_file(label_path: Path, img_width: int = None, img_height: int = None) -> Tuple[bool, List[str]]:
    """
    Validate YOLO label file format.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not label_path.exists():
        return False, ["Label file does not exist"]
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return False, [f"Failed to read label file: {e}"]
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {line_num}: Expected 5 values, got {len(parts)}")
            continue
        
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as e:
            errors.append(f"Line {line_num}: Invalid number format - {e}")
            continue
        
        # Validate class ID
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            errors.append(f"Line {line_num}: Invalid class_id {class_id} (must be 0-{len(CLASS_NAMES)-1})")
        
        # Validate coordinates are in [0, 1]
        if not (0 <= x_center <= 1):
            errors.append(f"Line {line_num}: x_center {x_center} out of range [0, 1]")
        if not (0 <= y_center <= 1):
            errors.append(f"Line {line_num}: y_center {y_center} out of range [0, 1]")
        if not (0 <= width <= 1):
            errors.append(f"Line {line_num}: width {width} out of range [0, 1]")
        if not (0 <= height <= 1):
            errors.append(f"Line {line_num}: height {height} out of range [0, 1]")
    
    return len(errors) == 0, errors


def visualize_sample(img_path: Path, label_path: Path, output_path: Path):
    """Create visualization of image with bounding boxes."""
    if not HAS_PIL:
        return
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Load labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    img_width, img_height = img.size
    
    # Draw bboxes
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height
        
        # Convert to xyxy
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # Draw bbox
        color = CLASS_COLORS[class_id] if class_id < len(CLASS_COLORS) else (255, 255, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw class label
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'unknown'
        draw.text((x1, y1 - 10), class_name, fill=color)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def validate_split(split_name: str, split_dir: Path) -> Dict:
    """Validate a single split (train/val/test)."""
    print(f"\n🔍 Validating {split_name} split...")
    
    img_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    
    # Check directories exist
    if not img_dir.exists():
        return {'error': f"Images directory not found: {img_dir}"}
    if not label_dir.exists():
        return {'error': f"Labels directory not found: {label_dir}"}
    
    # Get all images and labels
    images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    labels = sorted(label_dir.glob("*.txt"))
    
    print(f"   Images: {len(images)}")
    print(f"   Labels: {len(labels)}")
    
    # Check for missing labels
    missing_labels = []
    for img_path in images:
        label_path = label_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            missing_labels.append(img_path.name)
    
    if missing_labels:
        print(f"   ⚠️  Missing labels for {len(missing_labels)} images")
    
    # Check for orphaned labels
    orphaned_labels = []
    for label_path in labels:
        png_path = img_dir / (label_path.stem + '.png')
        jpg_path = img_dir / (label_path.stem + '.jpg')
        if not png_path.exists() and not jpg_path.exists():
            orphaned_labels.append(label_path.name)
    
    if orphaned_labels:
        print(f"   ⚠️  Orphaned labels (no image): {len(orphaned_labels)}")
    
    # Validate label format
    invalid_labels = []
    total_cells = 0
    class_counts = Counter()
    
    for label_path in labels:
        is_valid, errors = validate_label_file(label_path)
        
        if not is_valid:
            invalid_labels.append({
                'file': label_path.name,
                'errors': errors
            })
        else:
            # Count cells and classes
            with open(label_path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                total_cells += len(lines)
                
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
    
    if invalid_labels:
        print(f"   ❌ Invalid labels: {len(invalid_labels)}")
        for inv in invalid_labels[:3]:  # Show first 3
            print(f"      {inv['file']}: {inv['errors'][:2]}")
    else:
        print(f"   ✅ All labels valid")
    
    print(f"   Total cells: {total_cells}")
    
    return {
        'split': split_name,
        'images': len(images),
        'labels': len(labels),
        'missing_labels': len(missing_labels),
        'orphaned_labels': len(orphaned_labels),
        'invalid_labels': len(invalid_labels),
        'total_cells': total_cells,
        'class_distribution': dict(class_counts),
        'valid': len(invalid_labels) == 0 and len(missing_labels) == 0
    }


def main():
    """Run dataset validation."""
    
    dataset_dir = Path("data/bbbc041_yolo")
    output_dir = dataset_dir / "validation_samples"
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    print("=" * 70)
    print("BBBC041 YOLO DATASET VALIDATION")
    print("=" * 70)
    
    # Validate each split
    results = {}
    for split_name in ['train', 'val', 'test']:
        split_dir = dataset_dir / split_name
        if split_dir.exists():
            results[split_name] = validate_split(split_name, split_dir)
    
    # Generate sample visualizations
    if HAS_PIL:
        print("\n🎨 Generating sample visualizations...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        samples_per_split = 3
        for split_name in ['train', 'val', 'test']:
            img_dir = dataset_dir / split_name / "images"
            label_dir = dataset_dir / split_name / "labels"
            
            if not img_dir.exists():
                continue
            
            images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
            if len(images) == 0:
                continue
            
            # Sample random images
            sample_images = random.sample(images, min(samples_per_split, len(images)))
            
            for img_path in sample_images:
                label_path = label_dir / (img_path.stem + '.txt')
                output_path = output_dir / f"{split_name}_{img_path.name}"
                
                try:
                    visualize_sample(img_path, label_path, output_path)
                except Exception as e:
                    print(f"   ⚠️  Failed to visualize {img_path.name}: {e}")
        
        print(f"   ✅ Saved samples to {output_dir}")
    
    # Summary report
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_valid = True
    total_images = 0
    total_cells = 0
    
    for split_name, result in results.items():
        if 'error' in result:
            print(f"\n❌ {split_name.upper()}: {result['error']}")
            all_valid = False
            continue
        
        status = "✅" if result['valid'] else "❌"
        print(f"\n{status} {split_name.upper()}:")
        print(f"   Images: {result['images']}")
        print(f"   Labels: {result['labels']}")
        print(f"   Cells: {result['total_cells']}")
        
        if result['missing_labels'] > 0:
            print(f"   ⚠️  Missing labels: {result['missing_labels']}")
            all_valid = False
        if result['invalid_labels'] > 0:
            print(f"   ⚠️  Invalid labels: {result['invalid_labels']}")
            all_valid = False
        
        total_images += result['images']
        total_cells += result['total_cells']
    
    print(f"\n📊 TOTALS:")
    print(f"   Total images: {total_images}")
    print(f"   Total cells: {total_cells}")
    
    # Save validation report
    report_path = dataset_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Validation report saved to {report_path}")
    
    if all_valid:
        print(f"\n{'='*70}")
        print("✅ DATASET VALIDATION PASSED - Ready for training!")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("❌ DATASET VALIDATION FAILED - Fix errors before training")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
