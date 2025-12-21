#!/usr/bin/env python3
"""
Visualization utilities for Grad-CAM heatmaps.

Provides functions for creating heatmap overlays, comparison grids,
and colorbar legends.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Colormap options
COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
}


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create heatmap overlay on original image.
    
    Args:
        image: Original RGB image (H, W, 3) uint8
        heatmap: Grayscale heatmap (H, W) float [0, 1]
        colormap: Colormap name (jet, turbo, viridis, inferno)
        alpha: Overlay transparency (0=original, 1=heatmap only)
        
    Returns:
        RGB overlay image (H, W, 3) uint8
    """
    # Validate colormap
    if colormap not in COLORMAPS:
        raise ValueError(f"Invalid colormap '{colormap}'. Supported: {list(COLORMAPS.keys())}")
    
    # Ensure heatmap matches image size
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap to [0, 255]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap (result is BGR)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, COLORMAPS[colormap])
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend with original
    overlay = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)
    
    return overlay


def create_comparison_grid(
    original: np.ndarray,
    overlay: np.ndarray,
    heatmap: np.ndarray = None,
    label: str = None,
    include_heatmap: bool = False,
    spacing: int = 4,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create side-by-side comparison image.
    
    Args:
        original: Original RGB image (H, W, 3)
        overlay: Overlay RGB image (H, W, 3)
        heatmap: Optional grayscale heatmap to include
        label: Optional label to add at top
        include_heatmap: Whether to include raw heatmap in grid
        spacing: Pixels between images
        background_color: Background color (RGB)
        
    Returns:
        Comparison grid image (H, W*N, 3) where N is number of images
    """
    images = [original, overlay]
    
    # Add colorized heatmap if requested
    if include_heatmap and heatmap is not None:
        heatmap_colored = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        images.append(heatmap_colored)
    
    # Ensure all images have same height
    h = max(img.shape[0] for img in images)
    resized_images = []
    for img in images:
        if img.shape[0] != h:
            scale = h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, h))
        resized_images.append(img)
    
    # Calculate total width
    total_w = sum(img.shape[1] for img in resized_images) + spacing * (len(resized_images) - 1)
    
    # Add space for label if provided
    label_height = 30 if label else 0
    
    # Create canvas
    canvas = np.full((h + label_height, total_w, 3), background_color, dtype=np.uint8)
    
    # Place images
    x = 0
    for i, img in enumerate(resized_images):
        canvas[label_height:label_height + img.shape[0], x:x + img.shape[1]] = img
        x += img.shape[1] + spacing
    
    # Add label if provided
    if label:
        canvas = add_text_label(canvas, label, position=(total_w // 2, 5), center=True)
    
    return canvas


def add_colorbar(
    image: np.ndarray,
    colormap: str = "jet",
    width: int = 30,
    padding: int = 10,
    label_low: str = "Low",
    label_high: str = "High"
) -> np.ndarray:
    """
    Add attention intensity colorbar to image.
    
    Args:
        image: RGB image (H, W, 3)
        colormap: Colormap name
        width: Width of colorbar in pixels
        padding: Padding between image and colorbar
        label_low: Label for low intensity
        label_high: Label for high intensity
        
    Returns:
        Image with colorbar added on right side
    """
    h, w = image.shape[:2]
    
    # Create colorbar gradient
    gradient = np.linspace(0, 255, h).astype(np.uint8)
    gradient = gradient[::-1]  # Flip so high is at top
    gradient = np.tile(gradient.reshape(-1, 1), (1, width))
    
    # Apply colormap
    colorbar = cv2.applyColorMap(gradient, COLORMAPS.get(colormap, cv2.COLORMAP_JET))
    colorbar = cv2.cvtColor(colorbar, cv2.COLOR_BGR2RGB)
    
    # Create extended canvas
    label_width = 50
    new_w = w + padding + width + label_width
    canvas = np.ones((h, new_w, 3), dtype=np.uint8) * 255
    
    # Place original image
    canvas[:, :w] = image
    
    # Place colorbar
    canvas[:, w + padding:w + padding + width] = colorbar
    
    # Add labels using PIL for better text rendering
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Position labels
    label_x = w + padding + width + 5
    draw.text((label_x, 5), label_high, fill=(0, 0, 0), font=font)
    draw.text((label_x, h - 20), label_low, fill=(0, 0, 0), font=font)
    
    return np.array(pil_img)


def add_text_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 14,
    color: Tuple[int, int, int] = (0, 0, 0),
    center: bool = False
) -> np.ndarray:
    """
    Add text label to image.
    
    Args:
        image: RGB image
        text: Text to add
        position: (x, y) position
        font_size: Font size
        color: Text color (RGB)
        center: Whether to center text at position
        
    Returns:
        Image with text added
    """
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    if center:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        position = (position[0] - text_w // 2, position[1])
    
    draw.text(position, text, fill=color, font=font)
    
    return np.array(pil_img)


def create_cell_heatmap_gallery(
    cells: List[dict],
    cols: int = 4,
    cell_size: Tuple[int, int] = (100, 100),
    spacing: int = 5
) -> np.ndarray:
    """
    Create gallery grid of cell images with heatmap overlays.
    
    Args:
        cells: List of dicts with 'original', 'overlay', and optional 'label'
        cols: Number of columns in grid
        cell_size: Size to resize each cell image
        spacing: Spacing between cells
        
    Returns:
        Gallery image
    """
    if not cells:
        # Return empty placeholder
        return np.ones((100, 400, 3), dtype=np.uint8) * 200
    
    rows = (len(cells) + cols - 1) // cols
    
    # Calculate canvas size
    canvas_w = cols * cell_size[0] + (cols - 1) * spacing
    canvas_h = rows * cell_size[1] + (rows - 1) * spacing
    
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    for i, cell in enumerate(cells):
        row = i // cols
        col = i % cols
        
        x = col * (cell_size[0] + spacing)
        y = row * (cell_size[1] + spacing)
        
        # Use overlay if available, else original
        img = cell.get('overlay', cell.get('original'))
        if img is None:
            continue
        
        # Resize to cell size
        img = cv2.resize(img, cell_size)
        
        # Place in canvas
        canvas[y:y + cell_size[1], x:x + cell_size[0]] = img
    
    return canvas


def generate_gradcam_legend() -> np.ndarray:
    """
    Generate a legend image explaining Grad-CAM colors.
    
    Returns:
        RGB legend image
    """
    width = 400
    height = 80
    
    # Create canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Create gradient bar
    gradient = np.linspace(0, 255, width - 100).astype(np.uint8)
    gradient = np.tile(gradient.reshape(1, -1), (20, 1))
    gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    gradient_colored = cv2.cvtColor(gradient_colored, cv2.COLOR_BGR2RGB)
    
    # Place gradient
    canvas[15:35, 50:width-50] = gradient_colored
    
    # Add labels using PIL
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_title = font
    
    # Title
    draw.text((width // 2 - 80, 2), "Model Attention Intensity", fill=(50, 50, 50), font=font_title)
    
    # Labels
    draw.text((30, 40), "Low", fill=(0, 0, 200), font=font)
    draw.text((width - 70, 40), "High", fill=(200, 0, 0), font=font)
    
    # Description
    draw.text((50, 55), "Blue = low attention | Red/Yellow = high attention", fill=(100, 100, 100), font=font)
    
    return np.array(pil_img)
