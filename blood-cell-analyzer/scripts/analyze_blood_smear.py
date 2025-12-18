#!/usr/bin/env python3
"""
Blood Smear Analysis Pipeline
Combines YOLOv8 detection + ResNet34 classification for full-field analysis.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from fastai.vision.all import load_learner, PILImage
from collections import Counter
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class BloodSmearAnalyzer:
    """Full pipeline for blood smear analysis."""
    
    def __init__(
        self,
        detection_model_path: str = None,
        classification_model_path: str = None,
        detection_conf: float = 0.5,
        min_crop_size: int = 50
    ):
        """
        Initialize the analyzer with detection and classification models.
        
        Args:
            detection_model_path: Path to YOLOv8 detection model
            classification_model_path: Path to FastAI classification model
            detection_conf: Confidence threshold for detection
            min_crop_size: Minimum size for WBC crops
        """
        self.detection_conf = detection_conf
        self.min_crop_size = min_crop_size
        
        # Default model paths
        if detection_model_path is None:
            detection_model_path = PROJECT_ROOT / "models/detection/bccd_v1/weights/best.pt"
        if classification_model_path is None:
            classification_model_path = PROJECT_ROOT / "models/classification/wbc_latest.pkl"
        
        print("=" * 60)
        print("BLOOD SMEAR ANALYZER")
        print("=" * 60)
        
        # Load detection model
        print(f"\n📦 Loading detection model: {detection_model_path}")
        self.detector = YOLO(str(detection_model_path))
        print("   ✅ Detection model loaded")
        
        # Load classification model
        print(f"\n📦 Loading classification model: {classification_model_path}")
        self.classifier = load_learner(str(classification_model_path))
        self.wbc_classes = self.classifier.dls.vocab
        print(f"   ✅ Classification model loaded")
        print(f"   Classes: {list(self.wbc_classes)}")
        
        # Detection class names (from BCCD)
        self.detection_classes = {0: "RBC", 1: "WBC", 2: "Platelets"}
        
    def detect_cells(self, image_path: str, use_sliding_window: bool = True) -> dict:
        """
        Run YOLOv8 detection on the image.
        
        Args:
            image_path: Path to input image
            use_sliding_window: If True, use sliding window for large images
            
        Returns:
            dict with detections organized by cell type
        """
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        
        detections = {"RBC": [], "WBC": [], "Platelets": []}
        
        # Use sliding window for large images (>640 in any dimension)
        if use_sliding_window and (h > 640 or w > 640):
            tile_size = 640
            overlap = 100  # Overlap to catch cells at tile boundaries
            stride = tile_size - overlap
            
            all_boxes = []
            
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    # Extract tile
                    x2 = min(x + tile_size, w)
                    y2 = min(y + tile_size, h)
                    tile = image[y:y2, x:x2]
                    
                    # Run detection on tile
                    results = self.detector(tile, conf=self.detection_conf, verbose=False)[0]
                    
                    # Adjust coordinates to full image
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        # Offset by tile position
                        xyxy[0] += x
                        xyxy[1] += y
                        xyxy[2] += x
                        xyxy[3] += y
                        
                        all_boxes.append({
                            "cls_id": cls_id,
                            "conf": conf,
                            "bbox": xyxy.tolist()
                        })
            
            # Apply NMS to remove duplicate detections at tile boundaries
            all_boxes = self._nms_across_tiles(all_boxes)
            
            # Organize by class
            for box in all_boxes:
                cls_name = self.detection_classes.get(box["cls_id"], "Unknown")
                detections[cls_name].append({
                    "bbox": box["bbox"],
                    "confidence": box["conf"]
                })
        else:
            # Standard detection for smaller images
            results = self.detector(image_path, conf=self.detection_conf)[0]
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.detection_classes.get(cls_id, "Unknown")
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                
                detections[cls_name].append({
                    "bbox": xyxy,
                    "confidence": conf
                })
            
        return detections
    
    def _nms_across_tiles(self, boxes: list, iou_threshold: float = 0.5) -> list:
        """Apply Non-Maximum Suppression across tile boundaries."""
        if not boxes:
            return []
        
        # Group by class
        from collections import defaultdict
        by_class = defaultdict(list)
        for box in boxes:
            by_class[box["cls_id"]].append(box)
        
        result = []
        for cls_id, class_boxes in by_class.items():
            # Sort by confidence
            class_boxes.sort(key=lambda x: x["conf"], reverse=True)
            
            kept = []
            while class_boxes:
                best = class_boxes.pop(0)
                kept.append(best)
                
                # Remove overlapping boxes
                remaining = []
                for box in class_boxes:
                    iou = self._compute_iou(best["bbox"], box["bbox"])
                    if iou < iou_threshold:
                        remaining.append(box)
                class_boxes = remaining
            
            result.extend(kept)
        
        return result
    
    def _compute_iou(self, box1: list, box2: list) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def crop_wbc(self, image: np.ndarray, bbox: list, padding: int = 10) -> np.ndarray:
        """
        Crop a WBC from the image with padding.
        
        Args:
            image: Input image as numpy array
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Pixels to add around the crop
            
        Returns:
            Cropped image as numpy array
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        crop = image[y1:y2, x1:x2]
        return crop
    
    def classify_wbc(self, crop: np.ndarray) -> tuple:
        """
        Classify a WBC crop.
        
        Args:
            crop: WBC crop as numpy array (BGR)
            
        Returns:
            Tuple of (class_name, confidence, all_probs)
        """
        # Convert BGR to RGB for fastai
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Create PIL image
        pil_img = PILImage.create(crop_rgb)
        
        # Predict
        pred_class, pred_idx, probs = self.classifier.predict(pil_img)
        
        # Get probabilities as dict
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(self.wbc_classes)}
        
        return str(pred_class), float(probs[pred_idx]), prob_dict
    
    def analyze(self, image_path: str, save_visualization: bool = True) -> dict:
        """
        Full analysis pipeline.
        
        Args:
            image_path: Path to input blood smear image
            save_visualization: Whether to save annotated image
            
        Returns:
            Analysis results dict
        """
        image_path = Path(image_path)
        print(f"\n🔬 Analyzing: {image_path.name}")
        print("-" * 60)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = image.shape[:2]
        print(f"   Image size: {w} x {h}")
        
        # Step 1: Detect cells
        print("\n📍 Step 1: Detecting cells...")
        detections = self.detect_cells(str(image_path))
        
        rbc_count = len(detections["RBC"])
        wbc_count = len(detections["WBC"])
        platelet_count = len(detections["Platelets"])
        
        print(f"   Found: {rbc_count} RBC, {wbc_count} WBC, {platelet_count} Platelets")
        
        # Step 2: Classify WBCs
        print("\n🔍 Step 2: Classifying WBC subtypes...")
        wbc_classifications = []
        
        for i, wbc in enumerate(detections["WBC"]):
            bbox = wbc["bbox"]
            crop = self.crop_wbc(image, bbox)
            
            # Skip if crop is too small
            if crop.shape[0] < self.min_crop_size or crop.shape[1] < self.min_crop_size:
                print(f"   WBC {i+1}: Skipped (too small)")
                continue
            
            subtype, conf, probs = self.classify_wbc(crop)
            
            wbc_classifications.append({
                "id": i + 1,
                "bbox": bbox,
                "detection_conf": wbc["confidence"],
                "subtype": subtype,
                "classification_conf": conf,
                "probabilities": probs
            })
            
            print(f"   WBC {i+1}: {subtype} ({conf*100:.1f}%)")
        
        # Step 3: Generate statistics
        print("\n📊 Step 3: Generating statistics...")
        subtype_counts = Counter([w["subtype"] for w in wbc_classifications])
        
        for subtype in ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]:
            count = subtype_counts.get(subtype, 0)
            pct = (count / len(wbc_classifications) * 100) if wbc_classifications else 0
            print(f"   {subtype}: {count} ({pct:.1f}%)")
        
        # Step 4: Create visualization
        if save_visualization:
            print("\n🎨 Step 4: Creating visualization...")
            vis_image = self.create_visualization(
                image, detections, wbc_classifications
            )
            
            output_path = image_path.parent / f"{image_path.stem}_analyzed.jpg"
            cv2.imwrite(str(output_path), vis_image)
            print(f"   Saved: {output_path}")
        
        # Compile results
        results = {
            "image_path": str(image_path),
            "image_size": {"width": w, "height": h},
            "timestamp": datetime.now().isoformat(),
            "cell_counts": {
                "RBC": rbc_count,
                "WBC": wbc_count,
                "Platelets": platelet_count
            },
            "wbc_subtypes": dict(subtype_counts),
            "wbc_classifications": wbc_classifications,
            "summary": {
                "total_cells": rbc_count + wbc_count + platelet_count,
                "wbc_classified": len(wbc_classifications)
            }
        }
        
        # Calculate differential
        if wbc_classifications:
            results["differential"] = {
                subtype: {
                    "count": subtype_counts.get(subtype, 0),
                    "percentage": subtype_counts.get(subtype, 0) / len(wbc_classifications) * 100
                }
                for subtype in ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
            }
        
        return results
    
    def create_visualization(
        self,
        image: np.ndarray,
        detections: dict,
        wbc_classifications: list
    ) -> np.ndarray:
        """Create annotated visualization."""
        vis = image.copy()
        
        # Color scheme
        colors = {
            "RBC": (0, 0, 255),       # Red
            "Platelets": (255, 0, 0),  # Blue
            "EOSINOPHIL": (0, 165, 255),   # Orange
            "LYMPHOCYTE": (0, 255, 0),     # Green
            "MONOCYTE": (255, 255, 0),     # Cyan
            "NEUTROPHIL": (255, 0, 255)    # Magenta
        }
        
        # Draw RBCs (thin circles)
        for rbc in detections["RBC"]:
            x1, y1, x2, y2 = map(int, rbc["bbox"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = max((x2 - x1), (y2 - y1)) // 2
            cv2.circle(vis, (cx, cy), r, colors["RBC"], 1)
        
        # Draw Platelets (small dots)
        for plt in detections["Platelets"]:
            x1, y1, x2, y2 = map(int, plt["bbox"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(vis, (cx, cy), 3, colors["Platelets"], -1)
        
        # Draw classified WBCs
        for wbc in wbc_classifications:
            x1, y1, x2, y2 = map(int, wbc["bbox"])
            subtype = wbc["subtype"]
            conf = wbc["classification_conf"]
            
            color = colors.get(subtype, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{subtype[:3]} {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(vis, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add legend
        y_offset = 30
        for label, color in [("RBC", colors["RBC"]), 
                            ("Platelets", colors["Platelets"]),
                            ("Eosinophil", colors["EOSINOPHIL"]),
                            ("Lymphocyte", colors["LYMPHOCYTE"]),
                            ("Monocyte", colors["MONOCYTE"]),
                            ("Neutrophil", colors["NEUTROPHIL"])]:
            cv2.rectangle(vis, (10, y_offset-15), (25, y_offset), color, -1)
            cv2.putText(vis, label, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return vis


def main():
    """Run analysis on sample image."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze blood smear image")
    parser.add_argument("image", nargs="?", help="Path to blood smear image")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--output", help="Path to save JSON results")
    args = parser.parse_args()
    
    # Use sample image if none provided
    if args.image is None:
        sample_path = PROJECT_ROOT / "data/samples/1000026798.jpg"
        if sample_path.exists():
            args.image = str(sample_path)
        else:
            print("❌ No image provided and sample image not found")
            return
    
    # Initialize analyzer
    analyzer = BloodSmearAnalyzer(detection_conf=args.conf)
    
    # Run analysis
    results = analyzer.analyze(args.image)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n📊 Cell Counts:")
    print(f"   • RBC: {results['cell_counts']['RBC']}")
    print(f"   • WBC: {results['cell_counts']['WBC']}")
    print(f"   • Platelets: {results['cell_counts']['Platelets']}")
    
    if "differential" in results:
        print(f"\n🔬 WBC Differential:")
        for subtype, data in results["differential"].items():
            print(f"   • {subtype}: {data['count']} ({data['percentage']:.1f}%)")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.image).parent / f"{Path(args.image).stem}_results.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")
    
    print("\n✅ Analysis complete!")
    
    return results


if __name__ == "__main__":
    main()
