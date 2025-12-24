"""
Nucleus Extractor
Extracts nucleus region from cell masks using image processing techniques.
This is a bridge module until nucleus segmentation model is trained.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NucleusExtractionResult:
    """Result of nucleus extraction."""
    nucleus_mask: np.ndarray  # Binary mask of nucleus region
    nucleus_area: float
    cell_area: float
    extraction_quality: str  # 'good', 'fair', 'poor'
    method_used: str  # Extraction method name


class NucleusExtractor:
    """
    Extracts nucleus region from whole cell masks using image processing.
    
    Strategy:
    1. Extract cell region using provided mask
    2. Convert to grayscale if needed
    3. Apply intensity-based thresholding (nuclei are darker in blood smears)
    4. Use morphological operations to clean up
    5. Return nucleus mask
    
    Note: This is a fallback approach until nucleus segmentation model is trained.
    Blood cell nuclei are stained purple/blue (darker) compared to cytoplasm.
    """
    
    def __init__(
        self,
        method: str = 'consensus',
        min_nucleus_ratio: float = 0.15,
        max_nucleus_ratio: float = 0.85,
        morph_kernel_size: int = 5,
        erosion_iterations: int = 2
    ):
        """
        Initialize nucleus extractor.
        
        Args:
            method: Extraction method ('otsu', 'adaptive', 'color_based', 'consensus')
            min_nucleus_ratio: Minimum nucleus/cell area ratio (filters noise)
            max_nucleus_ratio: Maximum nucleus/cell area ratio (filters bad extractions)
            morph_kernel_size: Kernel size for morphological operations
            erosion_iterations: Number of erosion iterations to shrink nucleus (conservative)
        """
        self.method = method
        self.min_nucleus_ratio = min_nucleus_ratio
        self.max_nucleus_ratio = max_nucleus_ratio
        self.erosion_iterations = erosion_iterations
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size)
        )
        self.small_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
    
    def extract(
        self,
        image: np.ndarray,
        cell_mask: np.ndarray
    ) -> NucleusExtractionResult:
        """
        Extract nucleus from cell region.
        
        Args:
            image: Original RGB image
            cell_mask: Binary mask of cell region (from YOLO segmentation)
        
        Returns:
            NucleusExtractionResult with nucleus mask and metrics
        """
        # Validate inputs
        if image.shape[:2] != cell_mask.shape:
            raise ValueError("Image and mask must have same dimensions")
        
        # Calculate cell area
        cell_area = np.sum(cell_mask > 0)
        
        if cell_area == 0:
            return self._empty_result()
        
        # Extract cell region
        cell_region = cv2.bitwise_and(image, image, mask=cell_mask.astype(np.uint8))
        
        # Apply extraction method
        if self.method == 'consensus':
            nucleus_mask = self._extract_consensus(cell_region, cell_mask)
        elif self.method == 'otsu':
            nucleus_mask = self._extract_otsu(cell_region, cell_mask)
        elif self.method == 'adaptive':
            nucleus_mask = self._extract_adaptive(cell_region, cell_mask)
        elif self.method == 'color_based':
            nucleus_mask = self._extract_color_based(cell_region, cell_mask)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Clean up nucleus mask
        nucleus_mask = self._cleanup_mask(nucleus_mask)
        
        # Apply erosion to be more conservative (shrink nucleus boundaries)
        if self.erosion_iterations > 0:
            nucleus_mask = cv2.erode(nucleus_mask, self.small_kernel, iterations=self.erosion_iterations)
        
        # Calculate nucleus area
        nucleus_area = np.sum(nucleus_mask > 0)
        
        # Check quality
        quality = self._assess_quality(nucleus_area, cell_area)
        
        return NucleusExtractionResult(
            nucleus_mask=nucleus_mask,
            nucleus_area=nucleus_area,
            cell_area=cell_area,
            extraction_quality=quality,
            method_used=self.method
        )
    
    def _extract_otsu(
        self,
        cell_region: np.ndarray,
        cell_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract nucleus using Otsu's thresholding.
        Works well when nucleus is significantly darker than cytoplasm.
        Uses conservative thresholding to avoid false positives.
        """
        # Convert to grayscale
        if len(cell_region.shape) == 3:
            gray = cv2.cvtColor(cell_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_region
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding only within cell mask
        masked_gray = np.where(cell_mask > 0, gray, 255)
        
        # Get Otsu threshold
        otsu_thresh, _ = cv2.threshold(
            masked_gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Be more conservative - use threshold lower than Otsu suggests
        # This ensures only the darkest regions (true nucleus) are captured
        conservative_thresh = int(otsu_thresh * 0.85)  # 15% more conservative
        
        _, nucleus_mask = cv2.threshold(
            masked_gray,
            conservative_thresh,
            255,
            cv2.THRESH_BINARY_INV
        )
        
        # Constrain to cell mask
        nucleus_mask = cv2.bitwise_and(nucleus_mask, nucleus_mask, mask=cell_mask.astype(np.uint8))
        
        return nucleus_mask
    
    def _extract_adaptive(
        self,
        cell_region: np.ndarray,
        cell_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract nucleus using adaptive thresholding.
        Better for varying illumination.
        """
        # Convert to grayscale
        if len(cell_region.shape) == 3:
            gray = cv2.cvtColor(cell_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_region
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold (inverted)
        nucleus_mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=5
        )
        
        # Constrain to cell mask
        nucleus_mask = cv2.bitwise_and(nucleus_mask, nucleus_mask, mask=cell_mask.astype(np.uint8))
        
        return nucleus_mask
    
    def _extract_color_based(
        self,
        cell_region: np.ndarray,
        cell_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract nucleus using color-based segmentation.
        Blood cell nuclei are stained purple/blue (high blue channel).
        """
        if len(cell_region.shape) != 3:
            raise ValueError("Color-based extraction requires RGB image")
        
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(cell_region, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Nucleus is typically darker (low L) and purple/blue (high A)
        # Create mask based on L and A channels
        _, l_mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, a_mask = cv2.threshold(a_channel, 128, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        nucleus_mask = cv2.bitwise_and(l_mask, a_mask)
        
        # Constrain to cell mask
        nucleus_mask = cv2.bitwise_and(nucleus_mask, nucleus_mask, mask=cell_mask.astype(np.uint8))
        
        return nucleus_mask
    
    def _extract_consensus(
        self,
        cell_region: np.ndarray,
        cell_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract nucleus using consensus of multiple methods.
        Most conservative approach - only marks as nucleus what all methods agree on.
        """
        # Get masks from all three methods
        mask_otsu = self._extract_otsu(cell_region, cell_mask)
        mask_adaptive = self._extract_adaptive(cell_region, cell_mask)
        mask_color = self._extract_color_based(cell_region, cell_mask)
        
        # Consensus: intersection of all three methods (very conservative)
        consensus_mask = cv2.bitwise_and(mask_otsu, mask_adaptive)
        consensus_mask = cv2.bitwise_and(consensus_mask, mask_color)
        
        return consensus_mask
    
    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up nucleus mask using morphological operations.
        """
        if mask is None or mask.sum() == 0:
            return mask
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.small_kernel)
        
        # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.small_kernel)
        
        # Keep only largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        
        return mask
    
    def _assess_quality(self, nucleus_area: float, cell_area: float) -> str:
        """
        Assess extraction quality based on nucleus/cell ratio.
        Uses clinically-informed thresholds.
        
        Clinical reference ranges:
        - Normal mature cells: 0.25-0.50
        - Lymphocytes: 0.65-0.80 (high but normal)
        - Immature/blast cells: >0.80
        
        Returns:
            'good', 'fair', or 'poor'
        """
        if cell_area == 0 or nucleus_area == 0:
            return 'poor'
        
        ratio = nucleus_area / cell_area
        
        # Reject clearly invalid ratios
        if ratio < self.min_nucleus_ratio or ratio > self.max_nucleus_ratio:
            return 'poor'
        
        # Good: typical range for normal cells or lymphocytes
        if 0.20 <= ratio <= 0.70:
            return 'good'
        
        # Fair: borderline cases (need review)
        elif 0.15 <= ratio < 0.20 or 0.70 < ratio <= 0.82:
            return 'fair'
        
        # Poor: extreme values indicate extraction failure
        else:
            return 'poor'
    
    def _empty_result(self) -> NucleusExtractionResult:
        """Return empty result for invalid inputs."""
        return NucleusExtractionResult(
            nucleus_mask=np.zeros((1, 1), dtype=np.uint8),
            nucleus_area=0,
            cell_area=0,
            extraction_quality='poor',
            method_used=self.method
        )
    
    def visualize(
        self,
        image: np.ndarray,
        cell_mask: np.ndarray,
        nucleus_mask: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Visualize cell and nucleus masks overlaid on image.
        
        Args:
            image: Original RGB image
            cell_mask: Binary cell mask
            nucleus_mask: Binary nucleus mask
            alpha: Transparency for overlays
        
        Returns:
            Visualization image
        """
        vis = image.copy()
        
        # Create colored overlays
        cell_overlay = np.zeros_like(vis)
        cell_overlay[cell_mask > 0] = [0, 255, 0]  # Green for cell
        
        nucleus_overlay = np.zeros_like(vis)
        nucleus_overlay[nucleus_mask > 0] = [255, 0, 0]  # Red for nucleus
        
        # Blend overlays
        vis = cv2.addWeighted(vis, 1 - alpha, cell_overlay, alpha, 0)
        vis = cv2.addWeighted(vis, 1, nucleus_overlay, alpha * 1.5, 0)
        
        # Draw contours
        cell_contours, _ = cv2.findContours(
            cell_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, cell_contours, -1, (0, 255, 0), 2)
        
        nucleus_contours, _ = cv2.findContours(
            nucleus_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, nucleus_contours, -1, (255, 0, 0), 2)
        
        return vis


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing NucleusExtractor with synthetic data...")
    
    # Create synthetic cell image
    img = np.ones((200, 200, 3), dtype=np.uint8) * 200  # Light background
    
    # Draw cell (lighter cytoplasm)
    cv2.circle(img, (100, 100), 60, (180, 150, 150), -1)
    
    # Draw nucleus (darker)
    cv2.circle(img, (100, 100), 30, (100, 80, 100), -1)
    
    # Create cell mask
    cell_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(cell_mask, (100, 100), 60, 255, -1)
    
    # Test each method
    for method in ['otsu', 'adaptive', 'color_based']:
        print(f"\nTesting {method} method:")
        extractor = NucleusExtractor(method=method)
        result = extractor.extract(img, cell_mask)
        
        print(f"  Cell area: {result.cell_area:.0f} pixels")
        print(f"  Nucleus area: {result.nucleus_area:.0f} pixels")
        print(f"  N/C ratio: {result.nucleus_area / result.cell_area:.3f}")
        print(f"  Quality: {result.extraction_quality}")
