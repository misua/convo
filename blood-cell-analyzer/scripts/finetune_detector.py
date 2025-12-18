#!/usr/bin/env python3
"""
Fine-tune YOLOv8 detector on custom blood smear images.
Combines BCCD pre-trained weights with your microscope images.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import random
import yaml

PROJECT_ROOT = Path(__file__).parent.parent


def prepare_custom_dataset():
    """Prepare custom dataset with train/val split."""
    
    custom_dir = PROJECT_ROOT / "data/custom"
    images_dir = custom_dir / "images"
    labels_dir = custom_dir / "labels"
    
    # Check for images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if len(image_files) == 0:
        print("❌ No images found in data/custom/images/")
        print("   Please add your blood smear images first.")
        return None
    
    print(f"📂 Found {len(image_files)} images")
    
    # Check for labels
    labeled_count = 0
    for img in image_files:
        label_file = labels_dir / f"{img.stem}.txt"
        if label_file.exists():
            labeled_count += 1
    
    if labeled_count == 0:
        print("❌ No label files found in data/custom/labels/")
        print("   Please label your images first using LabelImg or Roboflow.")
        return None
    
    print(f"🏷️  Found {labeled_count}/{len(image_files)} labeled images")
    
    if labeled_count < 20:
        print(f"⚠️  Warning: Only {labeled_count} labeled images. Recommend at least 50.")
    
    # Create train/val split
    output_dir = PROJECT_ROOT / "data/custom_yolo"
    train_img_dir = output_dir / "images/train"
    val_img_dir = output_dir / "images/val"
    train_lbl_dir = output_dir / "labels/train"
    val_lbl_dir = output_dir / "labels/val"
    
    # Clean and create directories
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get labeled images only
    labeled_images = []
    for img in image_files:
        label_file = labels_dir / f"{img.stem}.txt"
        if label_file.exists():
            labeled_images.append(img)
    
    # Shuffle and split (80% train, 20% val)
    random.shuffle(labeled_images)
    split_idx = int(len(labeled_images) * 0.8)
    train_images = labeled_images[:split_idx]
    val_images = labeled_images[split_idx:]
    
    # Copy files
    for img in train_images:
        shutil.copy(img, train_img_dir / img.name)
        shutil.copy(labels_dir / f"{img.stem}.txt", train_lbl_dir / f"{img.stem}.txt")
    
    for img in val_images:
        shutil.copy(img, val_img_dir / img.name)
        shutil.copy(labels_dir / f"{img.stem}.txt", val_lbl_dir / f"{img.stem}.txt")
    
    print(f"✅ Split: {len(train_images)} train, {len(val_images)} val")
    
    # Create data.yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "RBC",
            1: "WBC", 
            2: "Platelets"
        }
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"📄 Created {yaml_path}")
    
    return yaml_path


def finetune_model(data_yaml: Path, epochs: int = 30):
    """Fine-tune the pre-trained BCCD model on custom data."""
    
    # Start from our pre-trained BCCD model
    base_model = PROJECT_ROOT / "models/detection/bccd_v1/weights/best.pt"
    
    if not base_model.exists():
        print("❌ Base model not found. Using fresh YOLOv8n instead.")
        base_model = "yolov8n.pt"
    else:
        print(f"📦 Starting from: {base_model}")
    
    # Load model
    model = YOLO(str(base_model))
    
    # Fine-tune with lower learning rate (transfer learning)
    print(f"\n🏋️ Fine-tuning for {epochs} epochs...")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=8,  # Smaller batch for fine-tuning
        project=str(PROJECT_ROOT / "models/detection"),
        name="custom_v1",
        exist_ok=True,
        device=0,
        patience=10,
        lr0=0.001,  # Lower learning rate for fine-tuning
        lrf=0.01,
        warmup_epochs=3,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,
    )
    
    # Copy best model
    best_model = PROJECT_ROOT / "models/detection/custom_v1/weights/best.pt"
    latest_link = PROJECT_ROOT / "models/detection/detector_latest.pt"
    
    if best_model.exists():
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(best_model)
        print(f"\n✅ Model saved to: {best_model}")
        print(f"   Symlink: {latest_link}")
    
    return results


def main():
    """Main fine-tuning workflow."""
    print("=" * 60)
    print("FINE-TUNE DETECTOR ON CUSTOM DATA")
    print("=" * 60)
    
    # Prepare dataset
    data_yaml = prepare_custom_dataset()
    
    if data_yaml is None:
        print("\n" + "=" * 60)
        print("HOW TO ADD YOUR DATA:")
        print("=" * 60)
        print("""
1. Copy your blood smear images to:
   data/custom/images/

2. Label them using LabelImg:
   source venv/bin/activate
   pip install labelImg
   labelImg data/custom/images/ data/custom/labels/ data/custom/classes.txt

3. Run this script again:
   python scripts/finetune_detector.py

Classes to label:
   0 = RBC (Red Blood Cell)
   1 = WBC (White Blood Cell)
   2 = Platelets
        """)
        return
    
    # Fine-tune
    epochs = 30
    print(f"\n🚀 Starting fine-tuning ({epochs} epochs)...")
    
    results = finetune_model(data_yaml, epochs=epochs)
    
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE!")
    print("=" * 60)
    print("""
Next steps:
1. Test on your images:
   python scripts/analyze_blood_smear.py data/samples/your_image.jpg

2. Or launch the dashboard:
   python app.py
    """)


if __name__ == "__main__":
    main()
