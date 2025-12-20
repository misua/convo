#!/usr/bin/env python3
"""
Blood Smear Analysis Pipeline
Combines YOLOv11/v8 detection + ResNet34 classification + Shape Analysis for full-field analysis.
Enhanced with RBC shape classification for thalassemia screening.
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
import yaml
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shape classifier
from src.models.shape_classifier import (
    RBCShapeClassifier, ShapeMetricsCalculator, 
    ShapeClassification, ShapePopulationStats, RBCShape
)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = PROJECT_ROOT / "configs/config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


class BloodSmearAnalyzer:
    """Full pipeline for blood smear analysis with shape classification."""
    
    def __init__(
        self,
        detection_model_path: str = None,
        classification_model_path: str = None,
        detection_conf: float = 0.5,
        min_crop_size: int = 50,
        enable_shape_analysis: bool = True,
        config: dict = None
    ):
        """
        Initialize the analyzer with detection, classification, and shape models.
        
        Args:
            detection_model_path: Path to YOLO detection model
            classification_model_path: Path to FastAI classification model
            detection_conf: Confidence threshold for detection
            min_crop_size: Minimum size for WBC crops
            enable_shape_analysis: Whether to run RBC shape analysis
            config: Configuration dict (loads from file if None)
        """
        self.detection_conf = detection_conf
        self.min_crop_size = min_crop_size
        self.enable_shape_analysis = enable_shape_analysis
        
        # Load config
        self.config = config or load_config()
        
        # Get calibration
        morph_config = self.config.get('morphology', {})
        self.um_per_pixel = morph_config.get('um_per_pixel', 0.25)
        self.rbc_normal_range = morph_config.get('rbc_normal_diameter_um', [6.2, 8.2])
        self.central_pallor_threshold = morph_config.get('central_pallor_ratio', 0.33)
        
        # Get model paths from config or use defaults
        detection_config = self.config.get('detection', {})
        segmentation_config = self.config.get('segmentation', {})
        
        if detection_model_path is None:
            model_name = detection_config.get('model', 'bccd_v1/weights/best.pt')
            detection_model_path = PROJECT_ROOT / "models/detection" / model_name
            # Fallback if primary model doesn't exist
            if not detection_model_path.exists():
                fallback = detection_config.get('fallback_model', 'bccd_v1/weights/best.pt')
                detection_model_path = PROJECT_ROOT / "models/detection" / fallback
        if classification_model_path is None:
            # Try 5-class model first, fallback to 4-class
            model_5class = PROJECT_ROOT / "models/classification/wbc_5class_latest.pkl"
            model_4class = PROJECT_ROOT / "models/classification/wbc_latest.pkl"
            classification_model_path = model_5class if model_5class.exists() else model_4class
        
        # Check for segmentation model
        self.segmentor = None
        self.use_segmentation = segmentation_config.get('enabled', False)
        if self.use_segmentation:
            seg_model_name = segmentation_config.get('model', 'bccd_seg_v1/weights/best.pt')
            seg_model_path = PROJECT_ROOT / "models/segmentation" / seg_model_name
            if seg_model_path.exists():
                print(f"\n📦 Loading segmentation model: {seg_model_path}")
                self.segmentor = YOLO(str(seg_model_path))
                print(f"   ✅ Segmentation model loaded (mask-based shape analysis)")
            elif segmentation_config.get('fallback_to_detection', True):
                print(f"\n⚠️ Segmentation model not found, using detection model")
                self.use_segmentation = False
        
        print("=" * 60)
        print("BLOOD SMEAR ANALYZER (v2 - Shape Analysis Enabled)")
        print("=" * 60)
        
        # Load detection model
        print(f"\n📦 Loading detection model: {detection_model_path}")
        self.detector = YOLO(str(detection_model_path))
        self.has_segmentation = '-seg' in str(detection_model_path) or 'seg' in str(detection_model_path)
        print(f"   ✅ Detection model loaded (segmentation: {self.has_segmentation})")
        
        # Load classification model
        print(f"\n📦 Loading classification model: {classification_model_path}")
        self.classifier = load_learner(str(classification_model_path))
        self.wbc_classes = self.classifier.dls.vocab
        print(f"   ✅ Classification model loaded")
        print(f"   Classes: {list(self.wbc_classes)}")
        
        # Initialize shape classifier
        if enable_shape_analysis:
            print(f"\n📦 Initializing shape classifier...")
            self.shape_classifier = RBCShapeClassifier(
                um_per_pixel=self.um_per_pixel,
                use_deep_learning=False,  # Use rule-based for now
                config=self.config
            )
            print(f"   ✅ Shape classifier ready")
            print(f"   Calibration: {self.um_per_pixel} μm/pixel")
        else:
            self.shape_classifier = None
        
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
    
    def segment_cells(self, image_path: str) -> dict:
        """
        Run segmentation model to get cell masks.
        
        Args:
            image_path: Path to input image
            
        Returns:
            dict with segmentation results including masks
        """
        if self.segmentor is None:
            return None
        
        results = self.segmentor(image_path, conf=self.detection_conf, verbose=False)[0]
        
        segmentations = {"RBC": [], "WBC": [], "Platelets": []}
        
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()  # (N, H, W)
            boxes = results.boxes
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                cls_id = int(box.cls[0])
                cls_name = self.detection_classes.get(cls_id, "Unknown")
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                
                # Resize mask to original image size
                mask_resized = mask  # Already in original size for YOLOv11
                
                segmentations[cls_name].append({
                    "bbox": xyxy,
                    "confidence": conf,
                    "mask": mask_resized  # Binary mask for this cell
                })
        
        return segmentations
    
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
        Full analysis pipeline with shape analysis.
        
        Args:
            image_path: Path to input blood smear image
            save_visualization: Whether to save annotated image
            
        Returns:
            Analysis results dict including shape analysis
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
        
        # Step 1b: Run segmentation if enabled (for RBC shape analysis)
        segmentations = None
        if self.use_segmentation and self.segmentor:
            print("\n🎭 Step 1b: Running instance segmentation...")
            segmentations = self.segment_cells(str(image_path))
            if segmentations:
                seg_rbc_count = len(segmentations.get("RBC", []))
                print(f"   Segmented: {seg_rbc_count} RBC with pixel masks")
        
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
        
        # Step 3: Analyze RBC shapes (with masks if available)
        shape_stats = None
        shape_classifications = []
        
        if self.enable_shape_analysis and self.shape_classifier and rbc_count > 0:
            print("\n🔬 Step 3: Analyzing RBC shapes...")
            
            # Use segmentation masks if available for better accuracy
            if segmentations and len(segmentations.get("RBC", [])) > 0:
                print(f"   Using instance segmentation masks (pixel-accurate)")
                shape_classifications, shape_stats = self.analyze_rbc_shapes_with_masks(
                    image, segmentations["RBC"]
                )
            else:
                print(f"   Using bounding box approximation")
                shape_classifications, shape_stats = self.analyze_rbc_shapes(
                    image, detections["RBC"]
                )
            
            if shape_stats:
                print(f"   Shape distribution:")
                for shape, pct in shape_stats.shape_percentages.items():
                    if pct > 0:
                        print(f"      {shape}: {pct:.1f}%")
                print(f"   Thalassemia indicators: {shape_stats.thalassemia_indicator_pct:.1f}%")
                print(f"   Abnormality index: {shape_stats.abnormality_index:.2f}")
        
        # Step 4: Generate statistics
        print("\n📊 Step 4: Generating statistics...")
        # Normalize subtype names to uppercase for consistent counting
        subtype_counts = Counter([w["subtype"].upper() for w in wbc_classifications])
        
        for subtype in ["BASOPHIL", "EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]:
            count = subtype_counts.get(subtype, 0)
            pct = (count / len(wbc_classifications) * 100) if wbc_classifications else 0
            print(f"   {subtype}: {count} ({pct:.1f}%)")
        
        # Step 5: Calculate morphology metrics
        morphology = self._calculate_morphology(detections["RBC"], shape_stats, image)
        
        # Step 6: Calculate risk score
        risk_score, risk_level = self._calculate_thalassemia_risk(morphology, shape_stats)
        print(f"\n⚠️ Thalassemia Risk: {risk_level.upper()} ({risk_score*100:.0f}%)")
        
        # Step 7: Create visualization
        if save_visualization:
            print("\n🎨 Step 5: Creating visualization...")
            vis_image = self.create_visualization(
                image, detections, wbc_classifications, shape_classifications
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
            },
            "morphology": morphology,
            "risk_assessment": {
                "score": risk_score,
                "level": risk_level,
                "interpretation": self._get_risk_interpretation(risk_level)
            }
        }
        
        # Add shape analysis results
        if shape_stats:
            used_masks = segmentations and len(segmentations.get("RBC", [])) > 0
            results["shape_analysis"] = {
                "enabled": True,
                "method": "instance_segmentation" if used_masks else "bbox_approximation",
                "cells_analyzed": shape_stats.total_cells,
                "shape_distribution": shape_stats.shape_percentages,
                "shape_counts": shape_stats.shape_counts,
                "thalassemia_indicators": shape_stats.thalassemia_indicators,
                "thalassemia_indicator_pct": shape_stats.thalassemia_indicator_pct,
                "abnormality_index": shape_stats.abnormality_index,
                "mean_circularity": shape_stats.mean_circularity,
                "mean_elongation": shape_stats.mean_elongation,
                "size_cv": shape_stats.size_cv,
                "flagged_cells": [
                    {
                        "shape": fc.shape.value,
                        "confidence": fc.confidence,
                        "bbox": fc.bbox,
                        "is_thalassemia_indicator": fc.is_thalassemia_indicator,
                        "metrics": {
                            "circularity": fc.metrics.circularity,
                            "elongation": fc.metrics.elongation,
                            "diameter_um": fc.metrics.diameter_um
                        }
                    }
                    for fc in shape_stats.flagged_cells[:10]  # Top 10
                ]
            }
        else:
            results["shape_analysis"] = {"enabled": False}
        
        # Calculate differential (5 WBC classes)
        if wbc_classifications:
            results["differential"] = {
                subtype: {
                    "count": subtype_counts.get(subtype, 0),
                    "percentage": subtype_counts.get(subtype, 0) / len(wbc_classifications) * 100
                }
                for subtype in ["BASOPHIL", "EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
            }
        
        return results
    
    def analyze_rbc_shapes(
        self,
        image: np.ndarray,
        rbc_detections: list
    ) -> tuple:
        """
        Analyze shapes of all detected RBCs.
        
        Args:
            image: Full image as numpy array
            rbc_detections: List of RBC detection dicts with bbox
            
        Returns:
            Tuple of (list of ShapeClassification, ShapePopulationStats)
        """
        if not self.shape_classifier or not rbc_detections:
            return [], None
        
        classifications = []
        
        for rbc in rbc_detections:
            bbox = rbc["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid bbox
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop cell
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Classify shape (no mask available with standard YOLO)
            clf = self.shape_classifier.classify_cell(
                mask=None,
                image_crop=crop,
                bbox=bbox
            )
            
            classifications.append(clf)
        
        # Calculate population statistics
        shape_weights = self.config.get('risk_scoring', {}).get('shape_weights', None)
        stats = self.shape_classifier.analyze_population(classifications, shape_weights)
        
        return classifications, stats
    
    def analyze_rbc_shapes_with_masks(
        self,
        image: np.ndarray,
        rbc_segmentations: list
    ) -> tuple:
        """
        Analyze shapes of RBCs using instance segmentation masks.
        
        This provides much more accurate shape metrics than bounding box
        approximation because it uses actual cell contours.
        
        Args:
            image: Full image as numpy array
            rbc_segmentations: List of RBC segmentation dicts with mask
            
        Returns:
            Tuple of (list of ShapeClassification, ShapePopulationStats)
        """
        if not self.shape_classifier or not rbc_segmentations:
            return [], None
        
        classifications = []
        h, w = image.shape[:2]
        
        for rbc in rbc_segmentations:
            bbox = rbc["bbox"]
            mask = rbc.get("mask")
            
            if mask is None:
                continue
            
            # Resize mask if needed (YOLO may return different size)
            mask_h, mask_w = mask.shape
            if (mask_h, mask_w) != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Convert to binary uint8
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Extract bbox region
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop image and mask
            crop = image[y1:y2, x1:x2]
            mask_crop = mask[y1:y2, x1:x2]
            
            if crop.size == 0 or mask_crop.size == 0:
                continue
            
            # Classify shape using the actual mask contour
            clf = self.shape_classifier.classify_cell(
                mask=mask_crop,
                image_crop=crop,
                bbox=bbox
            )
            
            classifications.append(clf)
        
        # Calculate population statistics
        shape_weights = self.config.get('risk_scoring', {}).get('shape_weights', None)
        stats = self.shape_classifier.analyze_population(classifications, shape_weights)
        
        return classifications, stats
    
    def _calculate_morphology(
        self,
        rbc_detections: list,
        shape_stats: ShapePopulationStats,
        image: np.ndarray
    ) -> dict:
        """Calculate RBC morphology metrics."""
        morphology = {
            "microcyte_percentage": 0.0,
            "macrocyte_percentage": 0.0,
            "hypochromic_percentage": 0.0,
            "rdw_proxy": 12.0,  # Default normal
            "mean_diameter_um": 7.2  # Default normal
        }
        
        if not rbc_detections:
            return morphology
        
        # Calculate diameters
        diameters = []
        for rbc in rbc_detections:
            x1, y1, x2, y2 = rbc["bbox"]
            width = x2 - x1
            height = y2 - y1
            diameter_px = (width + height) / 2
            diameter_um = diameter_px * self.um_per_pixel
            diameters.append(diameter_um)
        
        if diameters:
            # Mean diameter
            mean_diameter = sum(diameters) / len(diameters)
            morphology["mean_diameter_um"] = mean_diameter
            
            # Microcyte/macrocyte percentage
            microcytes = sum(1 for d in diameters if d < self.rbc_normal_range[0])
            macrocytes = sum(1 for d in diameters if d > self.rbc_normal_range[1])
            morphology["microcyte_percentage"] = (microcytes / len(diameters)) * 100
            morphology["macrocyte_percentage"] = (macrocytes / len(diameters)) * 100
            
            # RDW proxy (CV of diameters)
            if len(diameters) >= 2 and mean_diameter > 0:
                variance = sum((d - mean_diameter) ** 2 for d in diameters) / len(diameters)
                std_dev = math.sqrt(variance)
                morphology["rdw_proxy"] = (std_dev / mean_diameter) * 100
        
        # Use shape stats for more accurate metrics if available
        if shape_stats:
            if shape_stats.size_cv > 0:
                morphology["rdw_proxy"] = shape_stats.size_cv
            # Get microcyte percentage from shape classification
            morphology["microcyte_percentage"] = shape_stats.shape_percentages.get("microcyte", 0)
        
        # Hypochromia detection (simplified - based on central pallor)
        # Would need actual intensity analysis for accuracy
        morphology["hypochromic_percentage"] = 0.0
        
        return morphology
    
    def _calculate_thalassemia_risk(
        self,
        morphology: dict,
        shape_stats: ShapePopulationStats
    ) -> tuple:
        """
        Calculate thalassemia risk score.
        
        Returns:
            Tuple of (risk_score 0-1, risk_level string)
        """
        risk_config = self.config.get('risk_scoring', {})
        
        # Weights
        shape_weight = risk_config.get('shape_abnormality_weight', 0.35)
        micro_weight = risk_config.get('microcytic_weight', 0.30)
        hypo_weight = risk_config.get('hypochromic_weight', 0.25)
        rdw_weight = risk_config.get('rdw_weight', 0.10)
        
        # Calculate component scores (normalized 0-1)
        micro_score = min(1.0, morphology.get("microcyte_percentage", 0) / 50.0)
        hypo_score = min(1.0, morphology.get("hypochromic_percentage", 0) / 50.0)
        
        rdw = morphology.get("rdw_proxy", 12.0)
        if rdw <= 14.5:
            rdw_score = 0.0
        else:
            rdw_score = min(1.0, (rdw - 14.5) / 5.0)
        
        shape_score = 0.0
        if shape_stats:
            shape_score = shape_stats.abnormality_index
        
        # Composite score
        risk_score = (
            shape_score * shape_weight +
            micro_score * micro_weight +
            hypo_score * hypo_weight +
            rdw_score * rdw_weight
        )
        
        risk_score = min(1.0, risk_score)
        
        # Determine level
        thresholds = risk_config.get('thresholds', {})
        low_threshold = thresholds.get('low', 0.3)
        medium_threshold = thresholds.get('medium', 0.6)
        
        if risk_score <= low_threshold:
            risk_level = "low"
        elif risk_score <= medium_threshold:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return risk_score, risk_level
    
    def _get_risk_interpretation(self, risk_level: str) -> str:
        """Get plain-language interpretation of risk level."""
        interpretations = {
            "low": "Findings are within normal ranges. No immediate follow-up typically needed.",
            "medium": "Some findings suggest follow-up with a healthcare provider may be beneficial. Consider Hemoglobin Electrophoresis testing.",
            "high": "Significant findings detected. Medical consultation recommended to discuss further testing options."
        }
        return interpretations.get(risk_level, "Unable to determine risk level.")
    
    def create_visualization(
        self,
        image: np.ndarray,
        detections: dict,
        wbc_classifications: list,
        shape_classifications: list = None
    ) -> np.ndarray:
        """Create annotated visualization with shape highlighting."""
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
        
        # Shape colors for flagged cells
        shape_colors = {
            "normal": (0, 0, 255),      # Red (standard RBC)
            "microcyte": (0, 200, 255),  # Orange-yellow
            "target": (0, 255, 255),     # Yellow - thalassemia indicator
            "teardrop": (0, 255, 200),   # Yellow-green - thalassemia indicator
            "spherocyte": (255, 200, 0), # Light blue
            "irregular": (100, 100, 255) # Light red
        }
        
        # Build set of flagged RBC indices for shape highlighting
        flagged_rbcs = set()
        rbc_shapes = {}
        if shape_classifications:
            for i, clf in enumerate(shape_classifications):
                if clf.is_thalassemia_indicator or clf.shape.value != "normal":
                    flagged_rbcs.add(i)
                    rbc_shapes[i] = clf.shape.value
        
        # Draw RBCs (with shape highlighting)
        for i, rbc in enumerate(detections["RBC"]):
            x1, y1, x2, y2 = map(int, rbc["bbox"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = max((x2 - x1), (y2 - y1)) // 2
            
            # Check if this RBC has a notable shape
            if i in flagged_rbcs:
                shape = rbc_shapes.get(i, "normal")
                color = shape_colors.get(shape, colors["RBC"])
                thickness = 2  # Thicker for flagged cells
                # Draw with highlight
                cv2.circle(vis, (cx, cy), r, color, thickness)
                # Add small label for thalassemia indicators
                if shape in ["target", "teardrop"]:
                    label = "T" if shape == "target" else "D"
                    cv2.putText(vis, label, (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
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
        
        # Add shape legend if shape analysis was done
        if shape_classifications:
            y_offset += 10
            cv2.putText(vis, "Shapes:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            for label, color in [("Target (T)", shape_colors["target"]),
                                ("Teardrop (D)", shape_colors["teardrop"]),
                                ("Microcyte", shape_colors["microcyte"])]:
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
    parser.add_argument("--no-shape", action="store_true", help="Disable shape analysis")
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
    analyzer = BloodSmearAnalyzer(
        detection_conf=args.conf,
        enable_shape_analysis=not args.no_shape
    )
    
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
    
    # Print shape analysis results
    if results.get("shape_analysis", {}).get("enabled"):
        print(f"\n🔬 Shape Analysis:")
        shape_dist = results["shape_analysis"]["shape_distribution"]
        for shape, pct in shape_dist.items():
            if pct > 0:
                print(f"   • {shape}: {pct:.1f}%")
        print(f"   • Thalassemia indicators: {results['shape_analysis']['thalassemia_indicator_pct']:.1f}%")
    
    # Print risk assessment
    if "risk_assessment" in results:
        risk = results["risk_assessment"]
        print(f"\n⚠️ Risk Assessment:")
        print(f"   • Level: {risk['level'].upper()}")
        print(f"   • Score: {risk['score']*100:.0f}%")
        print(f"   • {risk['interpretation']}")
    
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
