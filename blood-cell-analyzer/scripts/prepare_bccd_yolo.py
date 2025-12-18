"""
Convert BCCD dataset from VOC XML format to YOLO format for training.
"""
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random

# Paths
BCCD_ROOT = Path("data/bccd/BCCD/BCCD")
IMAGES_DIR = BCCD_ROOT / "JPEGImages"
ANNOT_DIR = BCCD_ROOT / "Annotations"
OUTPUT_DIR = Path("data/bccd_yolo")

# Class mapping
CLASSES = ["RBC", "WBC", "Platelets"]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

def convert_bbox(size, box):
    """Convert VOC bbox to YOLO format (normalized xywh)"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(xml_path, output_path):
    """Convert single XML annotation to YOLO txt format"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    
    with open(output_path, "w") as f:
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in CLASS_TO_ID:
                continue
            
            cls_id = CLASS_TO_ID[cls_name]
            bbox = obj.find("bndbox")
            b = (
                float(bbox.find("xmin").text),
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text),
            )
            yolo_bbox = convert_bbox((w, h), b)
            f.write(f"{cls_id} {' '.join(f'{x:.6f}' for x in yolo_bbox)}\n")

def main():
    print("Converting BCCD to YOLO format...")
    
    # Create output directories
    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Get all images
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(images)} images")
    
    # Split: 80% train, 20% val
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Process each split
    for split, split_images in [("train", train_images), ("val", val_images)]:
        for img_path in split_images:
            # Copy image
            dst_img = OUTPUT_DIR / split / "images" / img_path.name
            shutil.copy(img_path, dst_img)
            
            # Convert annotation
            xml_path = ANNOT_DIR / (img_path.stem + ".xml")
            if xml_path.exists():
                txt_path = OUTPUT_DIR / split / "labels" / (img_path.stem + ".txt")
                convert_annotation(xml_path, txt_path)
    
    # Create data.yaml for YOLOv8
    yaml_content = f"""# BCCD Dataset for YOLOv8
path: {OUTPUT_DIR.absolute()}
train: train/images
val: val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"\n✅ Done! Created YOLO dataset at {OUTPUT_DIR}")
    print(f"   data.yaml: {yaml_path}")

if __name__ == "__main__":
    main()
