#!/usr/bin/env python3
"""
Grad-CAM Heatmap Generator for Blood Cell Analysis

Provides visual explanations for model predictions by highlighting
regions that contributed most to classification decisions.
"""

import hashlib
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM, EigenCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed. Grad-CAM features disabled.")


class GradCAMGenerator:
    """
    Generate Grad-CAM heatmaps for model predictions.
    
    Supports:
    - EfficientNet classification models (WBC/shape)
    - YOLO detection models (via EigenCAM)
    
    Example:
        >>> generator = GradCAMGenerator(model, "features.8")
        >>> heatmap = generator.generate_heatmap(image, class_idx=2)
        >>> overlay = generator.generate_overlay(image, class_idx=2)
    """
    
    SUPPORTED_COLORMAPS = ["jet", "turbo", "viridis", "inferno"]
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Union[str, torch.nn.Module],
        device: str = None,
        cache_enabled: bool = True,
        model_type: str = "classifier"
    ):
        """
        Initialize Grad-CAM generator.
        
        Args:
            model: PyTorch model
            target_layer: Layer name (str) or module to extract activations from
            device: Device to run on (auto-detect if None)
            cache_enabled: Whether to cache heatmaps for repeated analysis
            model_type: "classifier" or "detector" (affects CAM method used)
        """
        if not GRADCAM_AVAILABLE:
            raise ImportError(
                "pytorch-grad-cam is required for Grad-CAM features. "
                "Install with: pip install pytorch-grad-cam>=1.4.8"
            )
        
        self.model = model
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, np.ndarray] = {}
        
        # Get target layer
        if isinstance(target_layer, str):
            self.target_layer = self._get_layer_by_name(model, target_layer)
        else:
            self.target_layer = target_layer
            
        # Initialize CAM method
        # Use EigenCAM for YOLO (doesn't require gradients through detection head)
        # Use GradCAM for classifiers
        if model_type == "detector":
            self.cam = EigenCAM(
                model=model,
                target_layers=[self.target_layer],
            )
        else:
            self.cam = GradCAM(
                model=model,
                target_layers=[self.target_layer],
            )
        
        # Put model in eval mode
        self.model.eval()
    
    def _get_layer_by_name(self, model: torch.nn.Module, name: str) -> torch.nn.Module:
        """Get a layer from model by its name (e.g., 'features.8')."""
        parts = name.split(".")
        layer = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer
    
    def _get_cache_key(self, image: np.ndarray, class_idx: int) -> str:
        """Generate cache key from image and class."""
        img_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        return f"{img_hash}_{class_idx}"
    
    def generate_heatmap(
        self,
        image: np.ndarray,
        class_idx: int = None,
        resize_to: Tuple[int, int] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate grayscale Grad-CAM heatmap for given class.
        
        Args:
            image: Input image (H, W, 3) RGB uint8 or float32 [0-1]
            class_idx: Target class index (None for top predicted class)
            resize_to: Resize heatmap to (H, W) if provided
            use_cache: Whether to use cached result if available
            
        Returns:
            Grayscale heatmap (H, W) with values in [0, 1]
        """
        # Check cache
        cache_key = self._get_cache_key(image, class_idx or -1)
        if use_cache and self.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            if resize_to and cached.shape[:2] != resize_to:
                return cv2.resize(cached, (resize_to[1], resize_to[0]))
            return cached
        
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        # Set target
        targets = None
        if class_idx is not None and self.model_type == "classifier":
            targets = [ClassifierOutputTarget(class_idx)]
        
        # Generate CAM
        with torch.no_grad():
            grayscale_cam = self.cam(
                input_tensor=input_tensor,
                targets=targets
            )
        
        # Get first (and only) batch item
        heatmap = grayscale_cam[0]
        
        # Resize to original image size if needed
        original_h, original_w = image.shape[:2]
        if heatmap.shape != (original_h, original_w):
            heatmap = cv2.resize(heatmap, (original_w, original_h))
        
        # Resize to target size if specified
        if resize_to:
            heatmap = cv2.resize(heatmap, (resize_to[1], resize_to[0]))
        
        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = heatmap
        
        return heatmap
    
    def generate_overlay(
        self,
        image: np.ndarray,
        class_idx: int = None,
        colormap: str = "jet",
        alpha: float = 0.4,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate RGB image with heatmap overlay.
        
        Args:
            image: Input image (H, W, 3) RGB uint8
            class_idx: Target class index (None for top predicted class)
            colormap: Colormap name (jet, turbo, viridis, inferno)
            alpha: Overlay transparency (0=original, 1=heatmap only)
            use_cache: Whether to use cached heatmap
            
        Returns:
            RGB image (H, W, 3) with heatmap overlay, uint8
        """
        if colormap not in self.SUPPORTED_COLORMAPS:
            raise ValueError(
                f"Invalid colormap '{colormap}'. "
                f"Supported: {self.SUPPORTED_COLORMAPS}"
            )
        
        # Generate heatmap
        heatmap = self.generate_heatmap(image, class_idx, use_cache=use_cache)
        
        # Normalize image to [0, 1] for overlay
        if image.dtype == np.uint8:
            img_normalized = image.astype(np.float32) / 255.0
        else:
            img_normalized = image.astype(np.float32)
        
        # Apply colormap overlay
        overlay = show_cam_on_image(
            img_normalized,
            heatmap,
            use_rgb=True,
            colormap=self._get_cv2_colormap(colormap),
            image_weight=1 - alpha
        )
        
        return overlay
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to model input tensor."""
        # Ensure float32 [0, 1]
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        
        # Convert HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, 0)
        
        # Convert to tensor
        tensor = torch.from_numpy(img).to(self.device)
        
        return tensor
    
    def _get_cv2_colormap(self, name: str) -> int:
        """Convert colormap name to OpenCV constant."""
        colormaps = {
            "jet": cv2.COLORMAP_JET,
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "inferno": cv2.COLORMAP_INFERNO,
        }
        return colormaps.get(name, cv2.COLORMAP_JET)
    
    def clear_cache(self):
        """Clear the heatmap cache."""
        self._cache.clear()
    
    def validate_attention(
        self,
        heatmap: np.ndarray,
        bbox: List[int],
        threshold: float = 0.3
    ) -> Tuple[float, bool]:
        """
        Validate that model attention is focused on the cell.
        
        Args:
            heatmap: Grayscale heatmap (H, W)
            bbox: Bounding box [x1, y1, x2, y2]
            threshold: Minimum fraction of attention inside bbox
            
        Returns:
            (attention_ratio, is_valid) - fraction inside bbox and validity
        """
        h, w = heatmap.shape
        x1, y1, x2, y2 = bbox
        
        # Clip to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        # Create mask for bbox region
        mask = np.zeros_like(heatmap)
        mask[y1:y2, x1:x2] = 1
        
        # Calculate attention ratio
        total_attention = heatmap.sum()
        if total_attention == 0:
            return 0.0, False
        
        attention_inside = (heatmap * mask).sum()
        attention_ratio = attention_inside / total_attention
        
        return float(attention_ratio), attention_ratio >= threshold


class ClassificationGradCAM(GradCAMGenerator):
    """Specialized Grad-CAM for classification models (EfficientNet, ResNet)."""
    
    # Default target layers for common architectures
    # FastAI wraps models in Sequential, so use index-based paths
    DEFAULT_LAYERS = {
        "efficientnet_b0": "features.8",
        "efficientnet_v2_s": "features.6",
        "resnet34": "layer4",
        "resnet50": "layer4",
        "fastai_resnet34": "0.7",  # FastAI Sequential wrapper: model[0][7]
        "fastai_resnet50": "0.7",
    }
    
    @classmethod
    def from_config(
        cls,
        model: torch.nn.Module,
        model_name: str,
        config: dict,
        device: str = None
    ) -> "ClassificationGradCAM":
        """
        Create GradCAM from config.yaml settings.
        
        Args:
            model: Classification model
            model_name: Model architecture name
            config: gradcam config dict
            device: Device to run on
        """
        # Get target layer from config or use default
        target_layers = config.get("target_layers", {})
        layer_name = target_layers.get(
            "classification",
            cls.DEFAULT_LAYERS.get(model_name, "0.7")  # Default to FastAI ResNet path
        )
        
        return cls(
            model=model,
            target_layer=layer_name,
            device=device,
            cache_enabled=True,
            model_type="classifier"
        )


class DetectionGradCAM(GradCAMGenerator):
    """Specialized Grad-CAM for YOLO detection models."""
    
    # Default target layers for YOLO versions
    DEFAULT_LAYERS = {
        "yolov8": "model.model.22",
        "yolo11": "model.model.22",
        "yolov5": "model.model.24",
    }
    
    def __init__(
        self,
        model,
        target_layer: str = None,
        device: str = None,
        cache_enabled: bool = True
    ):
        """
        Initialize YOLO Grad-CAM.
        
        Uses EigenCAM which works better with detection models.
        """
        # Default to YOLO11 layer if not specified
        if target_layer is None:
            target_layer = self.DEFAULT_LAYERS["yolo11"]
        
        super().__init__(
            model=model,
            target_layer=target_layer,
            device=device,
            cache_enabled=cache_enabled,
            model_type="detector"
        )
    
    @classmethod
    def from_yolo(
        cls,
        yolo_model,
        config: dict = None,
        device: str = None
    ) -> "DetectionGradCAM":
        """
        Create GradCAM from YOLO model.
        
        Args:
            yolo_model: Ultralytics YOLO model
            config: gradcam config dict (optional)
            device: Device to run on
        """
        # Get the underlying PyTorch model
        model = yolo_model.model
        
        # Get target layer from config or use default
        layer_name = cls.DEFAULT_LAYERS["yolo11"]
        if config:
            target_layers = config.get("target_layers", {})
            layer_name = target_layers.get("detection", layer_name)
        
        return cls(
            model=model,
            target_layer=layer_name,
            device=device,
            cache_enabled=True
        )
