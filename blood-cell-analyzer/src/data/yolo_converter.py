"""
YOLO Segmentation Converter
Converts CVAT polygon annotations to YOLO instance segmentation format.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from dataclasses import dataclass

from cvat_parser import CellAnnotation


@dataclass
class YOLOSegmentation:
    """Represents YOLO format segmentation annotation."""
    class_id: int
    normalized_polygon: np.ndarray  # Shape: (N, 2), values in [0, 1]


class YOLOSegmentationConverter:
    """
    Converts cell annotations to YOLO segmentation format.
    
    YOLO segmentation format:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    
    Where all coordinates are normalized to [0, 1] relative to image dimensions.
    """
    
    # Class name to YOLO class ID mapping
    CLASS_MAP = {
        'blast_cell': 0,
        'neutrophil': 1,
        'lymphocyte': 2,
        'monocyte': 3,
        'eosinophil': 4,
        'basophil': 5,
        'band_cell': 6,
        'promyelocyte': 7,
        'myelocyte': 8,
        'metamyelocyte': 9,
    }
    
    # Reverse mapping
    ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
    
    @classmethod
    def get_class_names(cls) -> List[str]:
        """Get ordered list of class names for data.yaml."""
        return [cls.ID_TO_CLASS[i] for i in range(len(cls.CLASS_MAP))]
    
    def __init__(self, simplify_tolerance: float = 2.0):
        """
        Initialize converter.
        
        Args:
            simplify_tolerance: Tolerance for polygon simplification (Douglas-Peucker algorithm).
                               Higher values = more simplification. Set to 0 to disable.
        """
        self.simplify_tolerance = simplify_tolerance
    
    def convert_annotation(self, annotation: CellAnnotation) -> YOLOSegmentation:
        """
        Convert a single CellAnnotation to YOLO format.
        
        Args:
            annotation: CellAnnotation object
        
        Returns:
            YOLOSegmentation object
        """
        # Get class ID
        class_id = self.CLASS_MAP.get(annotation.cell_class)
        if class_id is None:
            raise ValueError(f"Unknown class: {annotation.cell_class}")
        
        # Normalize polygon coordinates to [0, 1]
        polygon = annotation.cell_polygon.copy()
        polygon[:, 0] /= annotation.image_width
        polygon[:, 1] /= annotation.image_height
        
        # Clip to [0, 1] range (handle any out-of-bounds points)
        polygon = np.clip(polygon, 0.0, 1.0)
        
        # Simplify polygon if requested
        if self.simplify_tolerance > 0:
            polygon = self._simplify_polygon(
                polygon,
                annotation.image_width,
                annotation.image_height
            )
        
        return YOLOSegmentation(
            class_id=class_id,
            normalized_polygon=polygon
        )
    
    def _simplify_polygon(
        self,
        normalized_polygon: np.ndarray,
        img_width: int,
        img_height: int
    ) -> np.ndarray:
        """
        Simplify polygon using Douglas-Peucker algorithm.
        
        Args:
            normalized_polygon: Normalized polygon coordinates
            img_width: Image width for denormalization
            img_height: Image height for denormalization
        
        Returns:
            Simplified normalized polygon
        """
        # Denormalize for simplification
        polygon = normalized_polygon.copy()
        polygon[:, 0] *= img_width
        polygon[:, 1] *= img_height
        
        # Convert to integer coordinates for cv2
        polygon = polygon.astype(np.int32)
        
        # Simplify using cv2.approxPolyDP
        epsilon = self.simplify_tolerance
        simplified = cv2.approxPolyDP(polygon, epsilon, closed=True)
        
        # Reshape if needed
        if len(simplified.shape) == 3:
            simplified = simplified.squeeze(1)
        
        # Normalize back
        simplified = simplified.astype(np.float32)
        simplified[:, 0] /= img_width
        simplified[:, 1] /= img_height
        
        return simplified
    
    def save_yolo_annotation(
        self,
        yolo_annotations: List[YOLOSegmentation],
        output_path: str
    ):
        """
        Save YOLO annotations to a text file.
        
        Args:
            yolo_annotations: List of YOLOSegmentation objects for one image
            output_path: Output .txt file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for yolo_ann in yolo_annotations:
                # Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
                class_id = yolo_ann.class_id
                coords = yolo_ann.normalized_polygon.flatten()
                
                # Write line
                line = f"{class_id}"
                for coord in coords:
                    line += f" {coord:.6f}"
                f.write(line + "\n")
    
    def convert_batch(
        self,
        annotations: List[CellAnnotation]
    ) -> Dict[str, List[YOLOSegmentation]]:
        """
        Convert a batch of annotations, grouped by image.
        
        Args:
            annotations: List of CellAnnotation objects
        
        Returns:
            Dictionary mapping image paths to lists of YOLOSegmentation objects
        """
        # Group annotations by image
        image_groups: Dict[str, List[CellAnnotation]] = {}
        for ann in annotations:
            if ann.image_path not in image_groups:
                image_groups[ann.image_path] = []
            image_groups[ann.image_path].append(ann)
        
        # Convert each group
        yolo_groups: Dict[str, List[YOLOSegmentation]] = {}
        for image_path, anns in image_groups.items():
            yolo_groups[image_path] = [
                self.convert_annotation(ann) for ann in anns
            ]
        
        return yolo_groups
    
    def visualize_annotation(
        self,
        image_path: str,
        yolo_annotations: List[YOLOSegmentation],
        output_path: str = None
    ) -> np.ndarray:
        """
        Visualize YOLO annotations on image for verification.
        
        Args:
            image_path: Path to image file
            yolo_annotations: List of YOLOSegmentation objects
            output_path: Optional path to save visualization
        
        Returns:
            Annotated image as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        h, w = img.shape[:2]
        
        # Define colors for each class (BGR format)
        colors = [
            (0, 0, 255),      # blast_cell - red
            (255, 0, 0),      # neutrophil - blue
            (0, 255, 0),      # lymphocyte - green
            (255, 255, 0),    # monocyte - cyan
            (255, 0, 255),    # eosinophil - magenta
            (0, 255, 255),    # basophil - yellow
            (128, 0, 128),    # band_cell - purple
            (255, 128, 0),    # promyelocyte - orange
            (0, 128, 255),    # myelocyte - light blue
            (128, 255, 0),    # metamyelocyte - lime
        ]
        
        # Draw each annotation
        for yolo_ann in yolo_annotations:
            # Denormalize coordinates
            polygon = yolo_ann.normalized_polygon.copy()
            polygon[:, 0] *= w
            polygon[:, 1] *= h
            polygon = polygon.astype(np.int32)
            
            # Get color
            color = colors[yolo_ann.class_id % len(colors)]
            
            # Draw polygon
            cv2.polylines(img, [polygon], True, color, 2)
            
            # Draw class label
            class_name = self.ID_TO_CLASS[yolo_ann.class_id]
            text_pos = tuple(polygon[0])
            cv2.putText(
                img,
                class_name,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), img)
        
        return img


def create_data_yaml(
    output_path: str,
    train_path: str,
    val_path: str,
    test_path: str = None
):
    """
    Create data.yaml file for YOLO training.
    
    Args:
        output_path: Path to save data.yaml
        train_path: Path to training images directory
        val_path: Path to validation images directory
        test_path: Optional path to test images directory
    """
    class_names = YOLOSegmentationConverter.get_class_names()
    
    yaml_content = f"""# WBC Instance Segmentation Dataset
# Generated for YOLO11-seg training

# Dataset paths
train: {train_path}
val: {val_path}
"""
    
    if test_path:
        yaml_content += f"test: {test_path}\n"
    
    yaml_content += f"""
# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created data.yaml at {output_path}")


if __name__ == "__main__":
    # Test converter
    from cvat_parser import CVATAnnotationParser
    
    # Parse BLAST CELLS annotations
    parser = CVATAnnotationParser(
        "data/wbc_instance_seg/BLAST CELLS/annotations.xml",
        "BLAST CELLS"
    )
    annotations = parser.parse()
    
    print(f"Converting {len(annotations)} annotations to YOLO format...")
    
    # Convert to YOLO format
    converter = YOLOSegmentationConverter(simplify_tolerance=2.0)
    yolo_groups = converter.convert_batch(annotations)
    
    print(f"Converted {len(yolo_groups)} images")
    
    # Test saving one annotation
    if yolo_groups:
        first_image = list(yolo_groups.keys())[0]
        first_annotations = yolo_groups[first_image]
        
        output_file = "/tmp/test_annotation.txt"
        converter.save_yolo_annotation(first_annotations, output_file)
        print(f"\nSaved test annotation to {output_file}")
        
        # Print content
        with open(output_file, 'r') as f:
            print(f.read())
