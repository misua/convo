"""
CVAT Annotation Parser
Parses CVAT XML format annotations to extract cell polygons and labels.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class CellAnnotation:
    """Represents a single cell annotation."""
    image_path: str
    image_width: int
    image_height: int
    cell_polygon: np.ndarray  # Shape: (N, 2) - array of (x, y) coordinates
    cell_class: str
    cell_id: int = 0
    occluded: bool = False


class CVATAnnotationParser:
    """
    Parses CVAT XML annotations for WBC instance segmentation.
    
    Example usage:
        parser = CVATAnnotationParser("annotations.xml", "BLAST CELLS")
        annotations = parser.parse()
    """
    
    # CVAT label mapping to standardized class names
    LABEL_MAP = {
        'blast cell': 'blast_cell',
        'neutrophil': 'neutrophil',
        'lymphocyte': 'lymphocyte',
        'monocyte': 'monocyte',
        'eosinophil': 'eosinophil',
        'basophil': 'basophil',
        'band cell': 'band_cell',
        'promyelocyte': 'promyelocyte',
        'myelocyte': 'myelocyte',
        'metamyelocyte': 'metamyelocyte',
        'bg': 'background'  # Background class
    }
    
    def __init__(self, xml_path: str, cell_type_folder: str):
        """
        Initialize parser.
        
        Args:
            xml_path: Path to CVAT annotations.xml file
            cell_type_folder: Name of the cell type folder (e.g., "BLAST CELLS")
        """
        self.xml_path = Path(xml_path)
        self.cell_type_folder = cell_type_folder
        self.annotations: List[CellAnnotation] = []
        
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    def parse(self) -> List[CellAnnotation]:
        """
        Parse the XML file and extract all cell annotations.
        
        Returns:
            List of CellAnnotation objects
        """
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        self.annotations = []
        
        # Parse each image in the XML
        for image_elem in root.findall('.//image'):
            image_id = int(image_elem.get('id'))
            image_name = image_elem.get('name')
            image_width = int(image_elem.get('width'))
            image_height = int(image_elem.get('height'))
            
            # Get image path relative to annotations.xml
            image_dir = self.xml_path.parent
            image_path = str(image_dir / image_name)
            
            # Parse all polygons in this image
            for poly_elem in image_elem.findall('.//polygon'):
                label = poly_elem.get('label')
                occluded = poly_elem.get('occluded', '0') == '1'
                points_str = poly_elem.get('points')
                
                # Skip background labels
                if label == 'bg':
                    continue
                
                # Map label to standardized class name
                if label not in self.LABEL_MAP:
                    print(f"Warning: Unknown label '{label}' in {image_name}, skipping")
                    continue
                
                cell_class = self.LABEL_MAP[label]
                
                # Parse polygon points
                polygon = self._parse_polygon_points(points_str)
                
                if polygon is None or len(polygon) < 3:
                    print(f"Warning: Invalid polygon in {image_name}, skipping")
                    continue
                
                # Create annotation
                annotation = CellAnnotation(
                    image_path=image_path,
                    image_width=image_width,
                    image_height=image_height,
                    cell_polygon=polygon,
                    cell_class=cell_class,
                    cell_id=len(self.annotations),
                    occluded=occluded
                )
                
                self.annotations.append(annotation)
        
        print(f"Parsed {len(self.annotations)} cell annotations from {self.xml_path}")
        return self.annotations
    
    def _parse_polygon_points(self, points_str: str) -> Optional[np.ndarray]:
        """
        Parse polygon points string to numpy array.
        
        Args:
            points_str: String like "1356.45,462.73;1411.10,426.29;..."
        
        Returns:
            Numpy array of shape (N, 2) with (x, y) coordinates
        """
        try:
            # Split by semicolon to get individual points
            point_pairs = points_str.strip().split(';')
            
            points = []
            for pair in point_pairs:
                if not pair.strip():
                    continue
                x, y = pair.split(',')
                points.append([float(x), float(y)])
            
            return np.array(points, dtype=np.float32)
        
        except Exception as e:
            print(f"Error parsing polygon points: {e}")
            return None
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get count of annotations per class.
        
        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {}
        for ann in self.annotations:
            distribution[ann.cell_class] = distribution.get(ann.cell_class, 0) + 1
        return distribution
    
    def filter_by_class(self, class_name: str) -> List[CellAnnotation]:
        """
        Filter annotations by cell class.
        
        Args:
            class_name: Class name to filter (e.g., 'blast_cell')
        
        Returns:
            Filtered list of annotations
        """
        return [ann for ann in self.annotations if ann.cell_class == class_name]
    
    def get_images_with_annotations(self) -> List[str]:
        """
        Get list of unique image paths that have annotations.
        
        Returns:
            List of image paths
        """
        return list(set(ann.image_path for ann in self.annotations))


def parse_all_cvat_folders(base_path: str) -> Dict[str, List[CellAnnotation]]:
    """
    Parse all CVAT annotation folders in the WBC dataset.
    
    Args:
        base_path: Path to wbc_instance_seg directory
    
    Returns:
        Dictionary mapping folder names to annotation lists
    """
    base_path = Path(base_path)
    all_annotations = {}
    
    # Cell type folders to parse
    folders = [
        'BLAST CELLS',
        'NEUTROPHILS',
        'LYMPHOCYTES',
        'MONOCYTES',
        'EOSINOPHILS',
        'BASOPHILS',
        'BAND CELLS',
        'METAMYELOCYTES',
        'MYELOCYTE',
        'PROMYELOCYTES'
    ]
    
    for folder in folders:
        xml_path = base_path / folder / 'annotations.xml'
        
        if not xml_path.exists():
            print(f"Warning: No annotations.xml found in {folder}")
            continue
        
        try:
            parser = CVATAnnotationParser(str(xml_path), folder)
            annotations = parser.parse()
            all_annotations[folder] = annotations
            
            # Print class distribution
            dist = parser.get_class_distribution()
            print(f"{folder}: {dist}")
        
        except Exception as e:
            print(f"Error parsing {folder}: {e}")
    
    return all_annotations


if __name__ == "__main__":
    # Test parser on BLAST CELLS folder
    import sys
    
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "data/wbc_instance_seg"
    
    print(f"Parsing CVAT annotations from: {base_path}")
    print("=" * 80)
    
    all_annotations = parse_all_cvat_folders(base_path)
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    
    total = sum(len(anns) for anns in all_annotations.values())
    print(f"Total annotations: {total}")
    
    for folder, anns in all_annotations.items():
        print(f"{folder}: {len(anns)} annotations")
