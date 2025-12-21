#!/usr/bin/env python3
"""
RBC Shape Classifier Module

Classifies red blood cell morphology into 9 categories for multi-disorder detection.
Detects thalassemia indicators, malaria parasites, and sickle cells.
Uses shape metrics (circularity, elongation, solidity) and optional deep learning.
"""

import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class RBCShape(Enum):
    """RBC shape classification categories for multi-disorder detection."""
    # Thalassemia-related shapes
    NORMAL = "normal"
    MICROCYTE = "microcyte"    # Small cell - thalassemia/iron deficiency indicator
    TARGET = "target"          # Codocyte - thalassemia indicator
    TEARDROP = "teardrop"      # Dacrocyte - thalassemia indicator
    SPHEROCYTE = "spherocyte"  # Hereditary spherocytosis
    IRREGULAR = "irregular"
    
    # Malaria parasite stages
    RING = "ring"              # Ring stage - early malaria infection
    TROPHOZOITE = "trophozoite"  # Mature malaria parasite
    
    # Sickle cell disease
    SICKLE = "sickle"          # Crescent/banana shaped cell


@dataclass
class ShapeMetrics:
    """Geometric shape metrics for a single cell."""
    circularity: float = 0.0    # 4π × area / perimeter² (1.0 = perfect circle)
    elongation: float = 1.0      # major_axis / minor_axis (1.0 = symmetric)
    solidity: float = 1.0        # area / convex_hull_area
    area: float = 0.0            # Cell area in pixels
    perimeter: float = 0.0       # Cell perimeter in pixels
    diameter_px: float = 0.0     # Equivalent diameter in pixels
    diameter_um: float = 0.0     # Diameter in micrometers (if calibrated)
    central_pallor_ratio: float = 0.0  # Center intensity / edge intensity
    has_bullseye: bool = False   # Target cell pattern detected


@dataclass 
class ShapeClassification:
    """Classification result for a single RBC."""
    shape: RBCShape
    confidence: float
    metrics: ShapeMetrics
    is_thalassemia_indicator: bool = False
    is_infected: bool = False  # True if malaria parasite detected
    parasite_stage: Optional[str] = None  # ring, trophozoite, etc.
    bbox: List[float] = field(default_factory=list)
    mask: Optional[np.ndarray] = None
    crop: Optional[np.ndarray] = None


@dataclass
class ShapePopulationStats:
    """Aggregate statistics for all RBCs in an image."""
    total_cells: int = 0
    shape_counts: Dict[str, int] = field(default_factory=dict)
    shape_percentages: Dict[str, float] = field(default_factory=dict)
    abnormality_index: float = 0.0
    thalassemia_indicators: int = 0
    thalassemia_indicator_pct: float = 0.0
    malaria_infected: int = 0  # Total infected RBCs (RING + TROPHOZOITE)
    malaria_infected_pct: float = 0.0
    sickle_cells: int = 0  # Total sickle cells
    sickle_cell_pct: float = 0.0
    parasite_stages: Dict[str, int] = field(default_factory=dict)  # {"ring": 5, "trophozoite": 2}
    flagged_cells: List[ShapeClassification] = field(default_factory=list)
    mean_circularity: float = 0.0
    mean_elongation: float = 0.0
    size_cv: float = 0.0  # Coefficient of variation (RDW proxy)


class ShapeMetricsCalculator:
    """Calculate shape metrics from cell masks."""
    
    def __init__(
        self,
        um_per_pixel: float = 0.25,
        circularity_normal_min: float = 0.85,
        elongation_normal_max: float = 1.3,
        solidity_normal_min: float = 0.95,
        rbc_normal_diameter_um: Tuple[float, float] = (6.2, 8.2)
    ):
        self.um_per_pixel = um_per_pixel
        self.circularity_normal_min = circularity_normal_min
        self.elongation_normal_max = elongation_normal_max
        self.solidity_normal_min = solidity_normal_min
        self.rbc_min_diameter = rbc_normal_diameter_um[0]
        self.rbc_max_diameter = rbc_normal_diameter_um[1]
    
    def calculate_from_mask(
        self,
        mask: np.ndarray,
        image_crop: Optional[np.ndarray] = None
    ) -> ShapeMetrics:
        """
        Calculate shape metrics from a binary mask.
        
        Args:
            mask: Binary mask of the cell (255 = cell, 0 = background)
            image_crop: Optional grayscale image for pallor analysis
            
        Returns:
            ShapeMetrics with all calculated values
        """
        metrics = ShapeMetrics()
        
        # Ensure binary mask
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return metrics
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        metrics.area = cv2.contourArea(contour)
        metrics.perimeter = cv2.arcLength(contour, True)
        
        if metrics.area < 10:  # Too small
            return metrics
        
        # Circularity: 4π × area / perimeter²
        if metrics.perimeter > 0:
            metrics.circularity = (4 * math.pi * metrics.area) / (metrics.perimeter ** 2)
            metrics.circularity = min(1.0, metrics.circularity)  # Cap at 1.0
        
        # Equivalent diameter
        metrics.diameter_px = 2 * math.sqrt(metrics.area / math.pi)
        metrics.diameter_um = metrics.diameter_px * self.um_per_pixel
        
        # Elongation: major_axis / minor_axis
        if len(contour) >= 5:  # Need at least 5 points for ellipse
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, (minor_axis, major_axis), _) = ellipse
                if minor_axis > 0:
                    metrics.elongation = major_axis / minor_axis
            except cv2.error:
                pass
        
        # Solidity: area / convex_hull_area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            metrics.solidity = metrics.area / hull_area
        
        # Central pallor analysis (for target cell detection)
        if image_crop is not None:
            metrics.central_pallor_ratio, metrics.has_bullseye = \
                self._analyze_central_pallor(image_crop, mask, contour)
        
        return metrics
    
    def _analyze_central_pallor(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        contour: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Analyze central pallor pattern for target cell detection.
        
        Returns:
            Tuple of (pallor_ratio, is_bullseye)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Get centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0, False
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        h, w = gray.shape[:2]
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return 0.0, False
        
        # Calculate mean intensity in center vs edge
        # Center: inner 1/3 radius
        # Edge: outer 1/3 radius
        
        # Create distance transform
        dist = cv2.distanceTransform(mask * 255 if mask.max() <= 1 else mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        
        if max_dist < 3:  # Cell too small
            return 0.0, False
        
        # Inner region (center)
        inner_mask = dist > (max_dist * 0.66)
        # Outer region (edge)
        outer_mask = (dist > 0) & (dist < (max_dist * 0.33))
        
        # Get intensities
        if inner_mask.sum() > 0 and outer_mask.sum() > 0:
            center_intensity = gray[inner_mask].mean()
            edge_intensity = gray[outer_mask].mean()
            
            if edge_intensity > 0:
                pallor_ratio = center_intensity / edge_intensity
                
                # Target cell pattern: bright center AND bright ring with darker middle
                # Check for bullseye by looking at middle region
                middle_mask = (dist >= max_dist * 0.33) & (dist <= max_dist * 0.66)
                if middle_mask.sum() > 0:
                    middle_intensity = gray[middle_mask].mean()
                    
                    # Bullseye: center bright, middle dark, edge bright
                    is_bullseye = (
                        center_intensity > middle_intensity * 1.1 and
                        edge_intensity > middle_intensity * 1.05 and
                        pallor_ratio > 0.9
                    )
                    return pallor_ratio, is_bullseye
                
                return pallor_ratio, False
        
        return 0.0, False
    
    def calculate_from_bbox(
        self,
        image: np.ndarray,
        bbox: List[float]
    ) -> ShapeMetrics:
        """
        Calculate approximate shape metrics from bounding box.
        Used when segmentation masks are not available.
        
        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            ShapeMetrics (approximate, from bbox)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        metrics = ShapeMetrics()
        
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return metrics
        
        # Approximate diameter as average of width/height
        metrics.diameter_px = (width + height) / 2
        metrics.diameter_um = metrics.diameter_px * self.um_per_pixel
        
        # Approximate area (assume ellipse)
        metrics.area = math.pi * (width / 2) * (height / 2)
        
        # Approximate elongation
        metrics.elongation = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
        
        # Cannot calculate circularity or solidity without mask
        metrics.circularity = 0.0
        metrics.solidity = 0.0
        
        # Analyze pallor if we have the image crop
        if image is not None:
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                # Create approximate circular mask
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.ellipse(mask, (width//2, height//2), (width//2, height//2), 0, 0, 360, 255, -1)
                metrics.central_pallor_ratio, metrics.has_bullseye = \
                    self._analyze_central_pallor(crop, mask, None) if mask.sum() > 0 else (0.0, False)
        
        return metrics


class RBCShapeClassifier:
    """
    Classify RBC shapes using metrics and optional deep learning.
    
    Uses rule-based classification on shape metrics as primary method,
    with optional CNN classifier for malaria/sickle detection.
    """
    
    # CNN class mapping (must match training vocab order)
    CNN_CLASSES = ['infected', 'normal', 'sickle']
    
    def __init__(
        self,
        um_per_pixel: float = 0.25,
        use_deep_learning: bool = False,
        model_path: Optional[str] = None,
        config: Optional[dict] = None
    ):
        """
        Initialize the shape classifier.
        
        Args:
            um_per_pixel: Microscope calibration
            use_deep_learning: Whether to use CNN classifier
            model_path: Path to trained CNN model (.pkl file)
            config: Configuration dict from config.yaml
        """
        self.um_per_pixel = um_per_pixel
        self.use_deep_learning = use_deep_learning
        
        # Load config values
        if config:
            shape_config = config.get('shape_classification', {})
            metrics_config = shape_config.get('metrics', {})
            self.circularity_normal_min = metrics_config.get('circularity_normal_min', 0.85)
            self.circularity_abnormal = metrics_config.get('circularity_abnormal_threshold', 0.70)
            self.elongation_normal_max = metrics_config.get('elongation_normal_max', 1.3)
            self.elongation_abnormal = metrics_config.get('elongation_abnormal_threshold', 1.5)
            self.solidity_normal_min = metrics_config.get('solidity_normal_min', 0.95)
            
            morph_config = config.get('morphology', {})
            rbc_range = morph_config.get('rbc_normal_diameter_um', [6.2, 8.2])
            self.rbc_min_diameter = rbc_range[0]
            self.rbc_max_diameter = rbc_range[1]
        else:
            self.circularity_normal_min = 0.85
            self.circularity_abnormal = 0.70
            self.elongation_normal_max = 1.3
            self.elongation_abnormal = 1.5
            self.solidity_normal_min = 0.95
            self.rbc_min_diameter = 6.2
            self.rbc_max_diameter = 8.2
        
        # Initialize metrics calculator
        self.metrics_calculator = ShapeMetricsCalculator(
            um_per_pixel=um_per_pixel,
            circularity_normal_min=self.circularity_normal_min,
            elongation_normal_max=self.elongation_normal_max,
            solidity_normal_min=self.solidity_normal_min,
            rbc_normal_diameter_um=(self.rbc_min_diameter, self.rbc_max_diameter)
        )
        
        # CNN model for malaria/sickle detection
        self.cnn_model = None
        self.cnn_device = None
        self.cnn_transform = None
        
        if use_deep_learning and model_path:
            self._load_cnn_model(model_path)
    
    def _load_cnn_model(self, model_path: str):
        """Load trained CNN RBC classifier (ResNet34 for malaria/sickle detection)."""
        try:
            import torch
            from torchvision import transforms
            from pathlib import Path
            
            model_file = Path(model_path)
            if not model_file.exists():
                # Try symlink path
                alt_path = model_file.parent / "rbc_3class_latest.pkl"
                if alt_path.exists():
                    model_file = alt_path
                else:
                    print(f"   ⚠️ CNN model not found: {model_path}")
                    return
            
            # Load fastai exported model
            from fastai.learner import load_learner
            learner = load_learner(model_file)
            
            # Extract raw PyTorch model (bypass fastai's buggy predict())
            self.cnn_model = learner.model
            self.cnn_model.eval()
            
            # Set device
            self.cnn_device = next(self.cnn_model.parameters()).device
            
            # ImageNet normalization (ResNet34 pretrained)
            self.cnn_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print(f"   ✅ RBC CNN classifier loaded: {model_file}")
            print(f"      Classes: {self.CNN_CLASSES}")
            print(f"      Device: {self.cnn_device}")
            
        except ImportError as e:
            print(f"   ⚠️ Failed to load CNN: {e}")
            self.cnn_model = None
        except Exception as e:
            print(f"   ⚠️ CNN model load error: {e}")
            self.cnn_model = None
    
    def _cnn_classify(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Classify cell crop using CNN (malaria/sickle detection).
        
        Uses raw PyTorch inference to bypass fastai bugs.
        
        Args:
            crop: BGR image crop of the cell
            
        Returns:
            Tuple of (class_name, confidence)
        """
        import torch
        
        if self.cnn_model is None or crop is None:
            return None, 0.0
        
        try:
            # Convert BGR to RGB
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                rgb_crop = crop
            
            # Apply transforms
            tensor = self.cnn_transform(rgb_crop).unsqueeze(0).to(self.cnn_device)
            
            # Inference
            with torch.no_grad():
                logits = self.cnn_model(tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()
            
            class_name = self.CNN_CLASSES[pred_idx]
            return class_name, confidence
            
        except Exception as e:
            print(f"   ⚠️ CNN inference error: {e}")
            return None, 0.0
    
    def classify(
        self,
        metrics: ShapeMetrics,
        crop: Optional[np.ndarray] = None
    ) -> Tuple[RBCShape, float]:
        """
        Classify RBC shape based on metrics and optional CNN.
        
        Classification priority:
        1. CNN for malaria/sickle detection (if available and confident)
        2. Rule-based for thalassemia indicators (target, teardrop, microcyte)
        3. Rule-based for other abnormalities
        
        Args:
            metrics: Calculated shape metrics
            crop: Optional cell image crop for CNN classification
            
        Returns:
            Tuple of (RBCShape, confidence)
        """
        # 1. Try CNN classification for malaria/sickle (most critical)
        if self.cnn_model is not None and crop is not None:
            cnn_class, cnn_conf = self._cnn_classify(crop)
            
            if cnn_class is not None and cnn_conf >= 0.7:  # High confidence threshold
                if cnn_class == 'infected':
                    # Malaria detected - return as RING (early stage most common)
                    # Could enhance to detect specific stage with more training data
                    return RBCShape.RING, cnn_conf
                elif cnn_class == 'sickle':
                    return RBCShape.SICKLE, cnn_conf
                # 'normal' from CNN - continue with rule-based for thalassemia check
        
        # 2. Rule-based sickle cell detection (fallback if no CNN)
        if metrics.circularity < 0.5 and metrics.elongation > 2.0:
            return RBCShape.SICKLE, 0.80
        
        # 3. Target cell (bullseye pattern - thalassemia indicator)
        if metrics.has_bullseye:
            return RBCShape.TARGET, 0.85
        
        # 4. High central pallor ratio with good circularity = target
        if metrics.central_pallor_ratio > 0.5 and metrics.circularity > 0.80:
            return RBCShape.TARGET, 0.75
        
        # 5. Teardrop (high elongation - thalassemia indicator)
        if metrics.elongation > self.elongation_abnormal:
            return RBCShape.TEARDROP, 0.70
        
        # 6. Spherocyte (very high circularity, small, dense)
        if metrics.circularity > 0.95 and metrics.solidity > 0.98:
            if metrics.diameter_um < self.rbc_min_diameter:
                return RBCShape.SPHEROCYTE, 0.75
        
        # 7. Microcyte (small but otherwise normal - thalassemia indicator)
        if metrics.diameter_um < self.rbc_min_diameter:
            if metrics.circularity >= self.circularity_normal_min:
                return RBCShape.MICROCYTE, 0.85
            else:
                return RBCShape.IRREGULAR, 0.60
        
        # 8. Irregular (low circularity or solidity)
        if metrics.circularity < self.circularity_abnormal:
            return RBCShape.IRREGULAR, 0.65
        if metrics.solidity < 0.85:
            return RBCShape.IRREGULAR, 0.60
        
        # 9. Normal
        if (metrics.circularity >= self.circularity_normal_min and
            metrics.elongation <= self.elongation_normal_max and
            metrics.solidity >= self.solidity_normal_min and
            self.rbc_min_diameter <= metrics.diameter_um <= self.rbc_max_diameter):
            return RBCShape.NORMAL, 0.90
        
        # Default to normal with lower confidence if close to normal
        if metrics.circularity >= 0.75:
            return RBCShape.NORMAL, 0.70
        
        return RBCShape.IRREGULAR, 0.50
    
    def classify_cell(
        self,
        mask: Optional[np.ndarray],
        image_crop: np.ndarray,
        bbox: List[float]
    ) -> ShapeClassification:
        """
        Full classification pipeline for a single cell.
        
        Args:
            mask: Binary segmentation mask (or None)
            image_crop: Cell image crop
            bbox: Bounding box in original image
            
        Returns:
            ShapeClassification with all results
        """
        # Calculate metrics
        if mask is not None:
            metrics = self.metrics_calculator.calculate_from_mask(mask, image_crop)
        else:
            metrics = self.metrics_calculator.calculate_from_bbox(image_crop, bbox)
        
        # Classify
        shape, confidence = self.classify(metrics, image_crop)
        
        # Check if thalassemia indicator
        is_thal_indicator = shape in [RBCShape.TARGET, RBCShape.TEARDROP]
        
        # Check if malaria infected
        is_infected = shape in [RBCShape.RING, RBCShape.TROPHOZOITE]
        parasite_stage = shape.value if is_infected else None
        
        return ShapeClassification(
            shape=shape,
            confidence=confidence,
            metrics=metrics,
            is_thalassemia_indicator=is_thal_indicator,
            is_infected=is_infected,
            parasite_stage=parasite_stage,
            bbox=bbox,
            mask=mask,
            crop=image_crop
        )
    
    def analyze_population(
        self,
        classifications: List[ShapeClassification],
        shape_weights: Optional[Dict[str, float]] = None
    ) -> ShapePopulationStats:
        """
        Aggregate shape classifications for population statistics.
        
        Args:
            classifications: List of individual cell classifications
            shape_weights: Weights for abnormality index calculation
            
        Returns:
            ShapePopulationStats with aggregate statistics
        """
        if not classifications:
            return ShapePopulationStats()
        
        # Default weights for abnormality index
        if shape_weights is None:
            shape_weights = {
                "target": 3.0,
                "teardrop": 2.5,
                "microcyte": 1.5,
                "spherocyte": 1.0,
                "irregular": 1.0,
                "sickle": 2.0,  # Sickle cells are significant
                "ring": 0.0,  # Malaria tracked separately
                "trophozoite": 0.0,
                "normal": 0.0
            }
        
        stats = ShapePopulationStats()
        stats.total_cells = len(classifications)
        
        # Count shapes
        for shape in RBCShape:
            stats.shape_counts[shape.value] = 0
        
        for clf in classifications:
            stats.shape_counts[clf.shape.value] += 1
            if clf.is_thalassemia_indicator:
                stats.thalassemia_indicators += 1
                stats.flagged_cells.append(clf)
            if clf.is_infected:
                stats.malaria_infected += 1
                stage = clf.parasite_stage
                if stage:
                    stats.parasite_stages[stage] = stats.parasite_stages.get(stage, 0) + 1
            if clf.shape == RBCShape.SICKLE:
                stats.sickle_cells += 1
        
        # Calculate percentages
        for shape, count in stats.shape_counts.items():
            stats.shape_percentages[shape] = (count / stats.total_cells) * 100
        
        # Thalassemia indicator percentage
        stats.thalassemia_indicator_pct = (stats.thalassemia_indicators / stats.total_cells) * 100
        
        # Malaria and sickle cell percentages
        stats.malaria_infected_pct = (stats.malaria_infected / stats.total_cells) * 100
        stats.sickle_cell_pct = (stats.sickle_cells / stats.total_cells) * 100
        
        # Malaria and sickle cell percentages
        stats.malaria_infected_pct = (stats.malaria_infected / stats.total_cells) * 100
        stats.sickle_cell_pct = (stats.sickle_cells / stats.total_cells) * 100
        
        # Calculate abnormality index (weighted sum)
        abnormal_score = 0.0
        for shape, pct in stats.shape_percentages.items():
            weight = shape_weights.get(shape, 0.0)
            abnormal_score += (pct / 100) * weight
        stats.abnormality_index = min(1.0, abnormal_score)  # Cap at 1.0
        
        # Mean metrics
        circularities = [c.metrics.circularity for c in classifications if c.metrics.circularity > 0]
        elongations = [c.metrics.elongation for c in classifications if c.metrics.elongation > 0]
        diameters = [c.metrics.diameter_um for c in classifications if c.metrics.diameter_um > 0]
        
        if circularities:
            stats.mean_circularity = sum(circularities) / len(circularities)
        if elongations:
            stats.mean_elongation = sum(elongations) / len(elongations)
        
        # Size CV (RDW proxy)
        if len(diameters) >= 2:
            mean_diameter = sum(diameters) / len(diameters)
            if mean_diameter > 0:
                variance = sum((d - mean_diameter) ** 2 for d in diameters) / len(diameters)
                std_dev = math.sqrt(variance)
                stats.size_cv = (std_dev / mean_diameter) * 100  # As percentage
        
        # Sort flagged cells by confidence
        stats.flagged_cells.sort(key=lambda x: x.confidence, reverse=True)
        
        return stats
