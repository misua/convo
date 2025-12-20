#!/usr/bin/env python3
"""
Convert BCCD binary mask dataset to YOLO instance segmentation format.

Input: BCCD Dataset with mask (binary masks where cells are white, background black)
Output: YOLO segmentation format with polygon annotations

YOLO segmentation label format:
class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import List, Tuple
import random


def find_cell_contours(mask: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
    """
    Find cell contours from a binary mask.
    
    Args:
        mask: Binary mask (white cells on black background)
        min_area: Minimum contour area to filter noise
        
    Returns:
        List of contours (each contour is Nx1x2 array of points)
    """
    # Convert to grayscale if needed
    if len(mask.shape) == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask
    
    # Threshold to ensure binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    return filtered


def simplify_contour(contour: np.ndarray, epsilon_factor: float = 0.01) -> np.ndarray:
    """
    Simplify contour using Douglas-Peucker algorithm.
    
    Args:
        contour: Input contour
        epsilon_factor: Simplification factor (smaller = more points)
        
    Returns:
        Simplified contour
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    return simplified


def contour_to_yolo_polygon(contour: np.ndarray, img_width: int, img_height: int) -> str:
    """
    Convert OpenCV contour to YOLO polygon format.
    
    Args:
        contour: OpenCV contour (Nx1x2)
        img_width: Image width for normalization
        img_height: Image height for normalization
        
    Returns:
        String of normalized x,y coordinates
    """
    points = contour.reshape(-1, 2)
    normalized = []
    for x, y in points:
        nx = x / img_width
        ny = y / img_height
        # Clamp to [0, 1]
        nx = max(0, min(1, nx))
        ny = max(0, min(1, ny))
        normalized.extend([f"{nx:.6f}", f"{ny:.6f}"])
    return " ".join(normalized)


def classify_cell_by_size(contour: np.ndarray, img_area: int) -> int:
    """
    Classify cell type based on size (heuristic for RBC vs WBC vs Platelet).
    
    Args:
        contour: Cell contour
        img_area: Total image area
        
    Returns:
        Class ID: 0=RBC, 1=WBC, 2=Platelet
    """
    area = cv2.contourArea(contour)
    relative_area = area / img_area
    
    # WBCs are typically larger (nucleus + cytoplasm)
    # RBCs are medium sized
    # Platelets are smallest
    
    if relative_area > 0.015:  # Large cell = WBC
        return 1
    elif relative_area > 0.001:  # Medium cell = RBC
        return 0
    else:  # Small = Platelet
        return 2


def process_image(
    img_path: Path,
    mask_path: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    min_points: int = 6
) -> Tuple[int, int, int]:
    """
    Process a single image and its mask.
    
    Returns:
        Tuple of (rbc_count, wbc_count, platelet_count)
    """
    # Read image and mask
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path))
    
    if img is None or mask is None:
        print(f"  ⚠️ Could not read {img_path.name}")
        return 0, 0, 0
    
    height, width = img.shape[:2]
    img_area = height * width
    
    # Find cell contours
    contours = find_cell_contours(mask, min_area=50)
    
    if not contours:
        print(f"  ⚠️ No contours found in {img_path.name}")
        return 0, 0, 0
    
    # Process each contour
    labels = []
    counts = {0: 0, 1: 0, 2: 0}  # RBC, WBC, Platelet
    
    for contour in contours:
        # Simplify contour
        simplified = simplify_contour(contour, epsilon_factor=0.005)
        
        # Need at least 3 points for a polygon
        if len(simplified) < 3:
            continue
            
        # Classify by size
        class_id = classify_cell_by_size(contour, img_area)
        counts[class_id] += 1
        
        # Convert to YOLO format
        polygon_str = contour_to_yolo_polygon(simplified, width, height)
        labels.append(f"{class_id} {polygon_str}")
    
    if labels:
        # Copy image
        shutil.copy(img_path, output_img_dir / img_path.name)
        
        # Write label file
        label_file = output_label_dir / f"{img_path.stem}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(labels))
    
    return counts[0], counts[1], counts[2]


def main():
    # Paths
    data_root = Path("data/bccd_seg/BCCD Dataset with mask")
    output_root = Path("data/bccd_yolo_seg")
    
    # Create output directories
    for split in ['train', 'val']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BCCD Binary Mask → YOLO Segmentation Converter")
    print("=" * 60)
    
    # Gather all images
    all_images = []
    for split in ['train', 'test']:
        img_dir = data_root / split / 'original'
        mask_dir = data_root / split / 'mask'
        
        if not img_dir.exists():
            print(f"⚠️ Directory not found: {img_dir}")
            continue
            
        for img_path in sorted(img_dir.glob('*.png')):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                all_images.append((img_path, mask_path))
    
    print(f"\n📊 Found {len(all_images)} image-mask pairs")
    
    # Shuffle and split 80/20
    random.seed(42)
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"   Train: {len(train_images)}")
    print(f"   Val: {len(val_images)}")
    
    # Process images
    total_rbc, total_wbc, total_plt = 0, 0, 0
    
    print("\n🔄 Processing training images...")
    for img_path, mask_path in train_images:
        rbc, wbc, plt = process_image(
            img_path, mask_path,
            output_root / 'train' / 'images',
            output_root / 'train' / 'labels'
        )
        total_rbc += rbc
        total_wbc += wbc
        total_plt += plt
    
    print(f"\n🔄 Processing validation images...")
    for img_path, mask_path in val_images:
        rbc, wbc, plt = process_image(
            img_path, mask_path,
            output_root / 'val' / 'images',
            output_root / 'val' / 'labels'
        )
        total_rbc += rbc
        total_wbc += wbc
        total_plt += plt
    
    print(f"\n✅ Conversion complete!")
    print(f"   Total RBCs: {total_rbc}")
    print(f"   Total WBCs: {total_wbc}")
    print(f"   Total Platelets: {total_plt}")
    
    # Create data.yaml
    data_yaml = output_root / 'data.yaml'
    yaml_content = f"""# BCCD Instance Segmentation Dataset
# Auto-generated from binary masks

path: {output_root.absolute()}
train: train/images
val: val/images

# Classes
names:
  0: RBC
  1: WBC
  2: Platelets

# Dataset info
nc: 3  # number of classes
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n📄 Created {data_yaml}")
    print(f"\n🚀 Ready to train with:")
    print(f"   yolo segment train data={data_yaml} model=yolo11n-seg.pt epochs=50")


if __name__ == "__main__":
    main()
