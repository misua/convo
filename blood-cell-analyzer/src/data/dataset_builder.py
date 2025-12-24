"""
WBC Dataset Builder
Builds YOLO-format dataset from CVAT annotations with train/val/test splits.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
import json
from collections import defaultdict
import random
import numpy as np

from cvat_parser import CellAnnotation, parse_all_cvat_folders
from yolo_converter import YOLOSegmentationConverter, create_data_yaml


class WBCDatasetBuilder:
    """
    Builds YOLO dataset structure from CVAT annotations.
    
    Creates directory structure:
    output_dir/
        images/
            train/ - training images
            val/   - validation images
            test/  - test images
        labels/
            train/ - training annotations
            val/   - validation annotations
            test/  - test annotations
        data.yaml - dataset configuration
    """
    
    def __init__(
        self,
        cvat_base_path: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        min_blast_cells_per_split: int = 10,
        random_seed: int = 42
    ):
        """
        Initialize dataset builder.
        
        Args:
            cvat_base_path: Path to wbc_instance_seg directory with CVAT annotations
            output_dir: Output directory for YOLO dataset
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            min_blast_cells_per_split: Minimum blast cell samples per split
            random_seed: Random seed for reproducibility
        """
        self.cvat_base_path = Path(cvat_base_path)
        self.output_dir = Path(output_dir)
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_blast_cells = min_blast_cells_per_split
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.converter = YOLOSegmentationConverter(simplify_tolerance=2.0)
        self.annotations: Dict[str, List[CellAnnotation]] = {}
    
    def build(self):
        """
        Build complete YOLO dataset from CVAT annotations.
        
        Steps:
        1. Parse all CVAT annotations
        2. Split into train/val/test
        3. Convert to YOLO format
        4. Copy images and create label files
        5. Generate data.yaml
        """
        print("=" * 80)
        print("Building WBC YOLO Dataset")
        print("=" * 80)
        
        # Step 1: Parse annotations
        print("\n[1/5] Parsing CVAT annotations...")
        self.annotations = parse_all_cvat_folders(str(self.cvat_base_path))
        
        # Flatten all annotations
        all_annotations = []
        for folder_anns in self.annotations.values():
            all_annotations.extend(folder_anns)
        
        print(f"Total annotations: {len(all_annotations)}")
        
        # Step 2: Split dataset
        print("\n[2/5] Splitting dataset...")
        splits = self._split_dataset(all_annotations)
        
        print(f"Train: {len(splits['train'])} annotations")
        print(f"Val: {len(splits['val'])} annotations")
        print(f"Test: {len(splits['test'])} annotations")
        
        # Verify blast cell distribution
        for split_name, anns in splits.items():
            blast_count = sum(1 for a in anns if a.cell_class == 'blast_cell')
            print(f"  {split_name} blast cells: {blast_count}")
        
        # Step 3: Convert to YOLO format
        print("\n[3/5] Converting to YOLO format...")
        yolo_splits = {}
        for split_name, anns in splits.items():
            yolo_splits[split_name] = self.converter.convert_batch(anns)
            print(f"  {split_name}: {len(yolo_splits[split_name])} images")
        
        # Step 4: Create dataset structure
        print("\n[4/5] Creating dataset structure...")
        self._create_dataset_structure(splits, yolo_splits)
        
        # Step 5: Generate data.yaml
        print("\n[5/5] Generating data.yaml...")
        self._create_data_yaml()
        
        # Generate summary report
        self._generate_summary_report(splits)
        
        print("\n" + "=" * 80)
        print(f"Dataset created successfully at: {self.output_dir}")
        print("=" * 80)
    
    def _split_dataset(
        self,
        annotations: List[CellAnnotation]
    ) -> Dict[str, List[CellAnnotation]]:
        """
        Split annotations into train/val/test ensuring blast cells are distributed.
        
        Strategy:
        1. Group annotations by image
        2. Separate images with blast cells from others
        3. Split blast cell images proportionally
        4. Split remaining images proportionally
        
        Args:
            annotations: List of all annotations
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        # Group annotations by image
        image_groups: Dict[str, List[CellAnnotation]] = defaultdict(list)
        for ann in annotations:
            image_groups[ann.image_path].append(ann)
        
        # Separate images with/without blast cells
        blast_images = []
        other_images = []
        
        for img_path, anns in image_groups.items():
            has_blast = any(a.cell_class == 'blast_cell' for a in anns)
            if has_blast:
                blast_images.append((img_path, anns))
            else:
                other_images.append((img_path, anns))
        
        print(f"Images with blast cells: {len(blast_images)}")
        print(f"Images without blast cells: {len(other_images)}")
        
        # Shuffle
        random.shuffle(blast_images)
        random.shuffle(other_images)
        
        # Calculate split indices for blast images
        n_blast = len(blast_images)
        n_blast_train = max(
            int(n_blast * self.train_ratio),
            self.min_blast_cells
        )
        n_blast_val = max(
            int(n_blast * self.val_ratio),
            self.min_blast_cells
        )
        
        # Ensure we have enough blast images
        if n_blast < n_blast_train + n_blast_val + self.min_blast_cells:
            print(f"Warning: Only {n_blast} blast images, adjusting split ratios")
            n_blast_train = int(n_blast * self.train_ratio)
            n_blast_val = int(n_blast * self.val_ratio)
        
        # Split blast images
        blast_train = blast_images[:n_blast_train]
        blast_val = blast_images[n_blast_train:n_blast_train + n_blast_val]
        blast_test = blast_images[n_blast_train + n_blast_val:]
        
        # Split other images
        n_other = len(other_images)
        n_other_train = int(n_other * self.train_ratio)
        n_other_val = int(n_other * self.val_ratio)
        
        other_train = other_images[:n_other_train]
        other_val = other_images[n_other_train:n_other_train + n_other_val]
        other_test = other_images[n_other_train + n_other_val:]
        
        # Combine and flatten
        splits = {
            'train': self._flatten_image_groups(blast_train + other_train),
            'val': self._flatten_image_groups(blast_val + other_val),
            'test': self._flatten_image_groups(blast_test + other_test)
        }
        
        return splits
    
    def _flatten_image_groups(
        self,
        image_groups: List[Tuple[str, List[CellAnnotation]]]
    ) -> List[CellAnnotation]:
        """Flatten list of (image_path, annotations) tuples."""
        annotations = []
        for _, anns in image_groups:
            annotations.extend(anns)
        return annotations
    
    def _create_dataset_structure(
        self,
        splits: Dict[str, List[CellAnnotation]],
        yolo_splits: Dict[str, Dict[str, List]]
    ):
        """
        Create YOLO dataset directory structure and copy files.
        
        Args:
            splits: Original annotation splits
            yolo_splits: YOLO-format annotation splits
        """
        # Create directories
        for split_name in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split_name).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split_name).mkdir(parents=True, exist_ok=True)
        
        # Process each split
        for split_name, yolo_groups in yolo_splits.items():
            images_dir = self.output_dir / 'images' / split_name
            labels_dir = self.output_dir / 'labels' / split_name
            
            for img_path, yolo_anns in yolo_groups.items():
                img_path = Path(img_path)
                
                # Copy image
                dest_img = images_dir / img_path.name
                if not dest_img.exists():
                    shutil.copy2(img_path, dest_img)
                
                # Create label file
                label_name = img_path.stem + '.txt'
                label_path = labels_dir / label_name
                self.converter.save_yolo_annotation(yolo_anns, str(label_path))
            
            print(f"  Processed {len(yolo_groups)} images for {split_name}")
    
    def _create_data_yaml(self):
        """Create data.yaml configuration file."""
        train_path = str(self.output_dir / 'images' / 'train')
        val_path = str(self.output_dir / 'images' / 'val')
        test_path = str(self.output_dir / 'images' / 'test')
        
        yaml_path = self.output_dir / 'data.yaml'
        create_data_yaml(str(yaml_path), train_path, val_path, test_path)
    
    def _generate_summary_report(self, splits: Dict[str, List[CellAnnotation]]):
        """Generate summary report with dataset statistics."""
        report = {
            'dataset_path': str(self.output_dir),
            'cvat_source': str(self.cvat_base_path),
            'splits': {}
        }
        
        for split_name, anns in splits.items():
            # Count by class
            class_counts = defaultdict(int)
            for ann in anns:
                class_counts[ann.cell_class] += 1
            
            # Count images
            image_paths = set(ann.image_path for ann in anns)
            
            report['splits'][split_name] = {
                'num_images': len(image_paths),
                'num_annotations': len(anns),
                'class_distribution': dict(class_counts)
            }
        
        # Save report
        report_path = self.output_dir / 'dataset_summary.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nSummary report saved to: {report_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Dataset Summary")
        print("=" * 80)
        for split_name, stats in report['splits'].items():
            print(f"\n{split_name.upper()}:")
            print(f"  Images: {stats['num_images']}")
            print(f"  Annotations: {stats['num_annotations']}")
            print("  Class distribution:")
            for class_name, count in sorted(stats['class_distribution'].items()):
                print(f"    {class_name}: {count}")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 2:
        cvat_base = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        cvat_base = "data/wbc_instance_seg"
        output_dir = "data/wbc_yolo_seg"
    
    print(f"CVAT base path: {cvat_base}")
    print(f"Output directory: {output_dir}")
    
    # Build dataset
    builder = WBCDatasetBuilder(
        cvat_base_path=cvat_base,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        min_blast_cells_per_split=10
    )
    
    builder.build()
