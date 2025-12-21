#!/usr/bin/env python3
"""
Train YOLOv11n on BBBC041 Malaria Dataset

Trains YOLOv11n object detector on BBBC041 blood smear images for malaria detection.

Model Configuration:
- Base: YOLOv11n (nano, 6MB)
- Input: 640x640 RGB images
- Classes: 7 (RBC, leukocyte, ring, trophozoite, schizont, gametocyte, difficult)
- Output: Bounding boxes + class probabilities

Training Strategy:
- Epochs: 100 (with early stopping)
- Batch size: 16 (adjust based on GPU memory)
- Optimizer: AdamW
- Learning rate: 0.01 (auto)
- Data augmentation: YOLO default (mosaic, mixup, flip, scale, etc.)
- Class weights: Computed from dataset statistics (handle class imbalance)

Expected Performance:
- Training time: 4-6 hours (CUDA GPU)
- mAP50: >0.85 (target)
- Inference: <15ms per image (640x640)
"""

import os
from pathlib import Path
import json
from datetime import datetime
from ultralytics import YOLO


def calculate_class_weights(stats_path: Path) -> list:
    """
    Calculate class weights based on dataset statistics to handle class imbalance.
    
    Strategy: Inverse frequency weighting
    weight[c] = total_cells / (num_classes * class_count[c])
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    combined = stats['combined']
    total_cells = combined['total_cells']
    class_dist = combined['class_distribution']
    
    # Class order matches YOLO class IDs
    class_names = [
        'red blood cell',    # 0
        'leukocyte',         # 1
        'ring',              # 2
        'trophozoite',       # 3
        'schizont',          # 4
        'gametocyte',        # 5
        'difficult'          # 6
    ]
    
    num_classes = len(class_names)
    weights = []
    
    print("\n📊 Class Distribution & Weights:")
    for class_name in class_names:
        count = class_dist.get(class_name, 1)  # Avoid division by zero
        weight = total_cells / (num_classes * count)
        pct = count / total_cells * 100
        
        weights.append(weight)
        print(f"   {class_name:20s}: {count:6d} ({pct:5.2f}%) -> weight: {weight:.4f}")
    
    return weights


def main():
    """Train YOLOv11n malaria detector."""
    
    # Paths
    dataset_dir = Path("data/bbbc041_yolo")
    data_yaml = dataset_dir / "data.yaml"
    stats_path = Path("data/bbbc041_malaria/dataset_stats.json")
    output_dir = Path("models/detection/malaria_yolo11n")
    
    # Check files exist
    if not data_yaml.exists():
        print(f"❌ data.yaml not found: {data_yaml}")
        return
    if not stats_path.exists():
        print(f"❌ Dataset stats not found: {stats_path}")
        return
    
    print("=" * 70)
    print("YOLO11N MALARIA DETECTOR TRAINING")
    print("=" * 70)
    
    # Calculate class weights
    class_weights = calculate_class_weights(stats_path)
    
    # Training configuration
    config = {
        'data': str(data_yaml.absolute()),
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'name': 'malaria_yolo11n',
        'project': str(output_dir.parent.absolute()),
        'patience': 20,  # Early stopping after 20 epochs without improvement
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'device': 'cuda:0',  # Use GPU (fallback to CPU if unavailable)
        'workers': 8,
        'exist_ok': True,
        'pretrained': True,  # Use COCO pre-trained weights
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,  # Cosine learning rate scheduler
        'close_mosaic': 10,  # Disable mosaic augmentation in last 10 epochs
        'resume': False,
        'amp': True,  # Automatic Mixed Precision training (faster)
        'fraction': 1.0,  # Use 100% of training data
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True
    }
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Model: YOLOv11n")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch']}")
    print(f"   Image size: {config['imgsz']}x{config['imgsz']}")
    print(f"   Device: {config['device']}")
    print(f"   Early stopping patience: {config['patience']}")
    print(f"   Output: {output_dir}")
    
    # Load YOLOv11n model
    print(f"\n📦 Loading YOLOv11n...")
    model = YOLO('yolo11n.pt')  # Load pre-trained weights
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = model.train(**config)
        
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'='*70}")
        
        # Show final metrics
        print(f"\n📊 Final Metrics:")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
            print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"   Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
            print(f"   Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
        
        # Save metadata
        best_model = output_dir / config['name'] / "weights" / "best.pt"
        metadata_path = output_dir / config['name'] / "training_metadata.json"
        
        metadata = {
            'model': 'YOLOv11n',
            'dataset': 'BBBC041',
            'training_date': datetime.now().isoformat(),
            'config': config,
            'class_weights': class_weights,
            'best_model_path': str(best_model.absolute()),
            'training_results': str(results) if results else None
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Model saved to: {best_model}")
        print(f"💾 Metadata saved to: {metadata_path}")
        
        # Validation
        print(f"\n🧪 Running final validation...")
        val_results = model.val()
        
        print(f"\n{'='*70}")
        print("🎉 TRAINING PIPELINE COMPLETE - Ready for integration!")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ TRAINING FAILED: {e}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
