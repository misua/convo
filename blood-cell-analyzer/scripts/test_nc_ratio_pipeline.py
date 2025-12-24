"""
Test N/C Ratio Pipeline
Demonstrates complete workflow: WBC Segmentation → Nucleus Extraction → N/C Ratio Analysis
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.inference.nucleus_extractor import NucleusExtractor
from src.inference.nc_ratio_analyzer import NCRatioAnalyzer


def test_nc_ratio_pipeline(
    image_path: str,
    model_path: str = 'models/segmentation/wbc_nc_ratio/train/weights/best.pt',
    output_dir: str = 'data/nc_ratio_test_results'
):
    """
    Test the complete N/C ratio pipeline on a single image.
    
    Args:
        image_path: Path to blood smear image
        model_path: Path to trained YOLO11-seg model
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Testing N/C Ratio Pipeline")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"\n[1/5] Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Load YOLO model
    print(f"\n[2/5] Loading YOLO11-seg model: {model_path}")
    model = YOLO(model_path)
    
    # Run cell segmentation
    print(f"\n[3/5] Running cell segmentation...")
    results = model.predict(image, conf=0.25, iou=0.45, verbose=False)
    
    if len(results) == 0 or results[0].masks is None:
        print("No cells detected!")
        return
    
    result = results[0]
    num_cells = len(result.masks)
    print(f"Detected {num_cells} cells")
    
    # Initialize extractors
    print(f"\n[4/5] Extracting nuclei and calculating N/C ratios...")
    # Use improved Otsu method with conservative erosion
    nucleus_extractor = NucleusExtractor(
        method='otsu',
        erosion_iterations=2  # Conservative erosion shrinks nucleus boundaries
    )
    nc_analyzer = NCRatioAnalyzer()
    
    # Process each detected cell
    blast_candidates = []
    all_results = []
    
    for i, (mask, box, cls) in enumerate(zip(
        result.masks.data.cpu().numpy(),
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy()
    )):
        # Resize mask to original image size
        mask_resized = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convert to uint8
        cell_mask = (mask_resized * 255).astype(np.uint8)
        
        # Extract nucleus
        nucleus_result = nucleus_extractor.extract(image, cell_mask)
        
        # Get class name
        class_id = int(cls)
        class_names = model.names
        cell_type = class_names[class_id]
        
        # Calculate N/C ratio with cell type for proper interpretation
        nc_result = nc_analyzer.analyze_cell(
            nucleus_mask=nucleus_result.nucleus_mask,
            cell_mask=cell_mask,
            cell_class=cell_type
        )
        
        # Store result
        cell_info = {
            'id': i + 1,
            'type': cell_type,
            'nc_ratio': nc_result.nc_ratio,
            'blast_confidence': nc_result.blast_confidence,
            'interpretation': nc_result.interpretation,
            'quality': nucleus_result.extraction_quality,
            'box': box
        }
        all_results.append(cell_info)
        
        # Check if blast candidate
        if nc_result.blast_confidence > 0.5 or cell_type == 'blast_cell':
            blast_candidates.append(cell_info)
        
        print(f"  Cell {i+1}: {cell_type:15s} | N/C: {nc_result.nc_ratio:.3f} | "
              f"Blast conf: {nc_result.blast_confidence:.1%} | {nc_result.interpretation}")
    
    # Print summary
    print(f"\n[5/5] Summary:")
    print(f"  Total cells analyzed: {num_cells}")
    print(f"  Blast cell candidates: {len(blast_candidates)}")
    
    if blast_candidates:
        print(f"\n  ⚠️  BLAST CELL ALERTS:")
        for bc in blast_candidates:
            print(f"    Cell #{bc['id']}: {bc['type']} - N/C ratio {bc['nc_ratio']:.3f} "
                  f"({bc['interpretation']})")
    
    # Create visualization
    print(f"\nCreating visualizations...")
    
    # Draw all detections
    vis_image = image.copy()
    for info in all_results:
        box = info['box'].astype(int)
        color = (0, 0, 255) if info['blast_confidence'] > 0.5 else (0, 255, 0)
        
        # Draw box
        cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw label
        label = f"#{info['id']} {info['type']}"
        nc_label = f"N/C:{info['nc_ratio']:.2f}"
        
        cv2.putText(vis_image, label, (box[0], box[1] - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(vis_image, nc_label, (box[0], box[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save visualization
    output_path = output_dir / f"nc_ratio_analysis_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(output_path), vis_image)
    print(f"Saved visualization: {output_path}")
    
    # Save detailed results for blast candidates
    if blast_candidates:
        for bc in blast_candidates:
            # Find the cell in original detection
            idx = bc['id'] - 1
            mask = result.masks.data[idx].cpu().numpy()
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            cell_mask = (mask_resized * 255).astype(np.uint8)
            
            # Re-extract nucleus for visualization
            nucleus_result = nucleus_extractor.extract(image, cell_mask)
            
            # Visualize
            cell_vis = nucleus_extractor.visualize(
                image, cell_mask, nucleus_result.nucleus_mask, alpha=0.3
            )
            
            # Add info text
            info_text = [
                f"Cell #{bc['id']}: {bc['type']}",
                f"N/C Ratio: {bc['nc_ratio']:.3f}",
                f"Blast Confidence: {bc['blast_confidence']:.1%}",
                f"{bc['interpretation']}",
                f"Quality: {bc['quality']}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(cell_vis, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(cell_vis, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                y_offset += 30
            
            # Save
            blast_output = output_dir / f"blast_candidate_{bc['id']}.jpg"
            cv2.imwrite(str(blast_output), cell_vis)
            print(f"Saved blast candidate: {blast_output}")
    
    print("\n" + "=" * 80)
    print("Pipeline test complete!")
    print("=" * 80)
    
    return all_results, blast_candidates


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test N/C ratio pipeline')
    parser.add_argument('image', help='Path to blood smear image')
    parser.add_argument('--model', default='models/segmentation/wbc_nc_ratio/train/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--output', default='data/nc_ratio_test_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    test_nc_ratio_pipeline(args.image, args.model, args.output)
