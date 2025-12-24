#!/usr/bin/env python3
"""
Nucleus-to-Cytoplasm (N/C) Ratio Analyzer for Leukemia Detection
Calculates N/C ratio from WBC segmentation masks as a biomarker for blast cells
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NCRatioResult:
    """Results of N/C ratio analysis for a single cell."""
    nucleus_area: int
    cytoplasm_area: int
    total_cell_area: int
    nc_ratio: float
    is_suspicious: bool
    blast_confidence: float
    interpretation: str


class NCRatioAnalyzer:
    """
    Analyzes Nucleus-to-Cytoplasm ratio to detect blast cells (leukemia indicator).
    
    Clinical Thresholds:
    - Normal mature WBCs: N/C < 0.5 (except lymphocytes ~0.7)
    - Blast cells (cancer): N/C > 0.8
    - Suspicious: N/C 0.75-0.85
    """
    
    # Clinical thresholds
    VERY_HIGH_THRESHOLD = 0.85  # N/C > 0.85 very strong blast indicator
    BLAST_THRESHOLD = 0.80  # N/C > 0.80 strongly suggests blast cell
    SUSPICIOUS_THRESHOLD = 0.75  # N/C > 0.75 warrants investigation
    
    # Normal N/C ranges for mature WBCs
    NORMAL_RANGES = {
        'Neutrophil': (0.25, 0.45),
        'Monocyte': (0.35, 0.50),
        'Eosinophil': (0.30, 0.45),
        'Basophil': (0.35, 0.50),
        'Lymphocyte': (0.65, 0.80),  # Lymphocytes naturally have higher N/C
    }
    
    def __init__(self):
        """Initialize N/C ratio analyzer."""
        pass
    
    def analyze_cell(
        self,
        nucleus_mask: np.ndarray,
        cell_mask: np.ndarray,
        cell_class: str = "Unknown"
    ) -> NCRatioResult:
        """
        Calculate N/C ratio from segmentation masks.
        
        Args:
            nucleus_mask: Binary mask of nucleus (255 = nucleus, 0 = background)
            cell_mask: Binary mask of entire cell (nucleus + cytoplasm)
            cell_class: WBC classification (e.g., "Neutrophil")
        
        Returns:
            NCRatioResult with N/C ratio and interpretation
        """
        
        # Calculate areas
        nucleus_area = np.sum(nucleus_mask > 0)
        total_cell_area = np.sum(cell_mask > 0)
        cytoplasm_area = total_cell_area - nucleus_area
        
        # Calculate N/C ratio
        if total_cell_area == 0:
            return NCRatioResult(
                nucleus_area=0,
                cytoplasm_area=0,
                total_cell_area=0,
                nc_ratio=0.0,
                is_suspicious=False,
                blast_confidence=0.0,
                interpretation="Invalid cell mask (zero area)"
            )
        
        nc_ratio = nucleus_area / total_cell_area
        
        # Determine if suspicious based on N/C ratio AND cell type
        # Lymphocytes naturally have high N/C (0.65-0.80), so adjust threshold
        if cell_class.lower() == "lymphocyte" and nc_ratio < 0.82:
            # Lymphocytes with N/C < 0.82 are normal
            is_suspicious = False
            blast_confidence = 0.0
        elif nc_ratio >= self.VERY_HIGH_THRESHOLD:
            # Very high: definitely suspicious regardless of cell type
            is_suspicious = True
            blast_confidence = min(100.0, 75.0 + (nc_ratio - 0.85) * 500)
        elif nc_ratio >= self.BLAST_THRESHOLD:
            # High: suspicious unless confirmed lymphocyte
            is_suspicious = True
            blast_confidence = 50.0 + (nc_ratio - 0.80) * 500
        elif nc_ratio >= self.SUSPICIOUS_THRESHOLD:
            # Borderline: suspicious
            is_suspicious = True
            blast_confidence = 25.0 + (nc_ratio - 0.75) * 500
        else:
            # Normal range
            is_suspicious = False
            blast_confidence = max(0.0, (nc_ratio - 0.5) * 50)
        
        # Generate interpretation
        interpretation = self._interpret_nc_ratio(nc_ratio, cell_class)
        
        return NCRatioResult(
            nucleus_area=int(nucleus_area),
            cytoplasm_area=int(cytoplasm_area),
            total_cell_area=int(total_cell_area),
            nc_ratio=round(nc_ratio, 3),
            is_suspicious=is_suspicious,
            blast_confidence=round(blast_confidence, 1),
            interpretation=interpretation
        )
    
    def _interpret_nc_ratio(self, nc_ratio: float, cell_class: str) -> str:
        """Generate clinical interpretation of N/C ratio."""
        
        # Check if within normal range for this cell type
        if cell_class in self.NORMAL_RANGES:
            normal_low, normal_high = self.NORMAL_RANGES[cell_class]
            
            if normal_low <= nc_ratio <= normal_high:
                return f"Normal N/C ratio for {cell_class}"
            elif nc_ratio < normal_low:
                return f"Low N/C ratio for {cell_class} (may indicate mature/hypersegmented cell)"
        
        # Abnormal N/C ratio interpretations
        if nc_ratio >= 0.85:
            return "⚠️ BLAST CELL SUSPECTED - Very high N/C ratio (>0.85). Urgent hematology review required."
        elif nc_ratio >= self.BLAST_THRESHOLD:
            return "⚠️ SUSPICIOUS FOR BLAST - High N/C ratio (0.80-0.85). Blast cells or immature forms possible."
        elif nc_ratio >= self.SUSPICIOUS_THRESHOLD:
            return "○ BORDERLINE - Moderately high N/C ratio (0.75-0.80). Consider immature forms or lymphocyte."
        elif nc_ratio >= 0.65:
            if cell_class == "Lymphocyte":
                return "✓ Normal for lymphocyte (N/C 0.65-0.80 expected)"
            else:
                return "○ Elevated N/C ratio - May indicate immature cell or lymphocyte misclassification"
        else:
            return "✓ Normal N/C ratio for mature white blood cell"
    
    def analyze_batch(
        self,
        cells_data: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Analyze N/C ratio for multiple cells and generate summary statistics.
        
        Args:
            cells_data: List of dicts with keys:
                - 'nucleus_mask': Binary mask of nucleus
                - 'cell_mask': Binary mask of entire cell
                - 'class': WBC classification
                - 'confidence': Classification confidence
        
        Returns:
            Summary statistics and blast cell screening results
        """
        
        results = []
        suspicious_cells = []
        blast_suspects = []
        
        for i, cell_data in enumerate(cells_data):
            result = self.analyze_cell(
                nucleus_mask=cell_data['nucleus_mask'],
                cell_mask=cell_data['cell_mask'],
                cell_class=cell_data.get('class', 'Unknown')
            )
            
            results.append(result)
            
            if result.blast_confidence >= 50:
                blast_suspects.append({
                    'cell_id': i,
                    'class': cell_data.get('class', 'Unknown'),
                    'nc_ratio': result.nc_ratio,
                    'blast_confidence': result.blast_confidence,
                    'interpretation': result.interpretation
                })
            elif result.is_suspicious:
                suspicious_cells.append({
                    'cell_id': i,
                    'class': cell_data.get('class', 'Unknown'),
                    'nc_ratio': result.nc_ratio,
                    'blast_confidence': result.blast_confidence
                })
        
        # Calculate summary statistics
        nc_ratios = [r.nc_ratio for r in results]
        avg_nc_ratio = np.mean(nc_ratios) if nc_ratios else 0.0
        max_nc_ratio = np.max(nc_ratios) if nc_ratios else 0.0
        
        blast_cell_percentage = (len(blast_suspects) / len(results) * 100) if results else 0.0
        suspicious_percentage = (len(suspicious_cells) / len(results) * 100) if results else 0.0
        
        # Generate clinical recommendation
        if len(blast_suspects) > 0:
            if blast_cell_percentage > 20:
                clinical_action = "URGENT: >20% blast-like cells detected. Immediate hematology referral for bone marrow biopsy."
                risk_level = "CRITICAL"
            elif len(blast_suspects) >= 3:
                clinical_action = "HIGH PRIORITY: Multiple blast-like cells detected. Urgent peripheral smear review by hematopathologist."
                risk_level = "HIGH"
            else:
                clinical_action = "MODERATE: Blast-like cell(s) detected. Manual microscopy review recommended."
                risk_level = "MODERATE"
        elif len(suspicious_cells) > 0:
            clinical_action = "LOW: Cells with elevated N/C ratio detected. Consider follow-up if clinical suspicion exists."
            risk_level = "LOW"
        else:
            clinical_action = "NEGATIVE: No blast-like cells detected based on N/C ratio analysis."
            risk_level = "NEGATIVE"
        
        return {
            'total_cells_analyzed': len(results),
            'blast_suspects_count': len(blast_suspects),
            'suspicious_cells_count': len(suspicious_cells),
            'blast_cell_percentage': round(blast_cell_percentage, 2),
            'suspicious_percentage': round(suspicious_percentage, 2),
            'average_nc_ratio': round(avg_nc_ratio, 3),
            'max_nc_ratio': round(max_nc_ratio, 3),
            'blast_suspects': blast_suspects,
            'suspicious_cells': suspicious_cells,
            'risk_level': risk_level,
            'clinical_action': clinical_action,
            'individual_results': results
        }
    
    def visualize_nc_ratio(
        self,
        image: np.ndarray,
        nucleus_mask: np.ndarray,
        cell_mask: np.ndarray,
        result: NCRatioResult
    ) -> np.ndarray:
        """
        Create visualization showing nucleus, cytoplasm, and N/C ratio.
        
        Args:
            image: Original cell image (RGB)
            nucleus_mask: Binary nucleus mask
            cell_mask: Binary cell mask
            result: NCRatioResult from analysis
        
        Returns:
            Annotated image with color-coded masks and N/C ratio
        """
        
        vis = image.copy()
        
        # Color code based on blast suspicion
        if result.blast_confidence >= 50:
            nucleus_color = (255, 0, 0)  # Red for blast suspects
            text_color = (255, 0, 0)
        elif result.is_suspicious:
            nucleus_color = (255, 165, 0)  # Orange for suspicious
            text_color = (255, 165, 0)
        else:
            nucleus_color = (0, 255, 0)  # Green for normal
            text_color = (0, 255, 0)
        
        # Overlay nucleus (solid color)
        nucleus_overlay = np.zeros_like(vis)
        nucleus_overlay[nucleus_mask > 0] = nucleus_color
        vis = cv2.addWeighted(vis, 0.7, nucleus_overlay, 0.3, 0)
        
        # Outline cytoplasm
        cytoplasm_mask = (cell_mask > 0) & (nucleus_mask == 0)
        contours, _ = cv2.findContours(
            cytoplasm_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
        
        # Add N/C ratio text
        text = f"N/C: {result.nc_ratio:.2f}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Add blast confidence if suspicious
        if result.blast_confidence > 25:
            conf_text = f"Blast: {result.blast_confidence:.0f}%"
            cv2.putText(vis, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        return vis


def calculate_nc_ratio_from_contours(
    nucleus_contour: np.ndarray,
    cell_contour: np.ndarray,
    image_shape: Tuple[int, int]
) -> float:
    """
    Helper function to calculate N/C ratio from contours.
    
    Args:
        nucleus_contour: Contour points for nucleus
        cell_contour: Contour points for entire cell
        image_shape: (height, width) of image
    
    Returns:
        N/C ratio
    """
    
    # Create masks from contours
    nucleus_mask = np.zeros(image_shape, dtype=np.uint8)
    cell_mask = np.zeros(image_shape, dtype=np.uint8)
    
    cv2.drawContours(nucleus_mask, [nucleus_contour], -1, 255, -1)
    cv2.drawContours(cell_mask, [cell_contour], -1, 255, -1)
    
    # Calculate areas
    nucleus_area = np.sum(nucleus_mask > 0)
    total_area = np.sum(cell_mask > 0)
    
    if total_area == 0:
        return 0.0
    
    return nucleus_area / total_area


if __name__ == "__main__":
    # Test with synthetic data
    print("🧪 Testing N/C Ratio Analyzer\n")
    
    analyzer = NCRatioAnalyzer()
    
    # Test 1: Normal neutrophil (N/C ~ 0.35)
    nucleus_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(nucleus_mask, (50, 50), 20, 255, -1)  # Nucleus radius 20
    
    cell_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(cell_mask, (50, 50), 35, 255, -1)  # Cell radius 35
    
    result = analyzer.analyze_cell(nucleus_mask, cell_mask, "Neutrophil")
    print(f"Test 1 - Normal Neutrophil:")
    print(f"  N/C Ratio: {result.nc_ratio}")
    print(f"  Suspicious: {result.is_suspicious}")
    print(f"  Interpretation: {result.interpretation}\n")
    
    # Test 2: Blast cell (N/C ~ 0.88)
    nucleus_mask_blast = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(nucleus_mask_blast, (50, 50), 40, 255, -1)  # Large nucleus
    
    cell_mask_blast = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(cell_mask_blast, (50, 50), 43, 255, -1)  # Minimal cytoplasm
    
    result_blast = analyzer.analyze_cell(nucleus_mask_blast, cell_mask_blast, "Unknown")
    print(f"Test 2 - Blast Cell:")
    print(f"  N/C Ratio: {result_blast.nc_ratio}")
    print(f"  Suspicious: {result_blast.is_suspicious}")
    print(f"  Blast Confidence: {result_blast.blast_confidence}%")
    print(f"  Interpretation: {result_blast.interpretation}\n")
    
    # Test 3: Batch analysis
    batch_data = [
        {'nucleus_mask': nucleus_mask, 'cell_mask': cell_mask, 'class': 'Neutrophil'},
        {'nucleus_mask': nucleus_mask_blast, 'cell_mask': cell_mask_blast, 'class': 'Unknown'},
        {'nucleus_mask': nucleus_mask, 'cell_mask': cell_mask, 'class': 'Neutrophil'},
    ]
    
    summary = analyzer.analyze_batch(batch_data)
    print(f"Test 3 - Batch Analysis:")
    print(f"  Total cells: {summary['total_cells_analyzed']}")
    print(f"  Blast suspects: {summary['blast_suspects_count']} ({summary['blast_cell_percentage']}%)")
    print(f"  Risk Level: {summary['risk_level']}")
    print(f"  Clinical Action: {summary['clinical_action']}")
