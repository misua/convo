"""
WBC Segmentation Training Script
Trains YOLO11-seg model for WBC instance segmentation to enable N/C ratio analysis.
"""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch


def train_wbc_segmentation(
    data_yaml: str,
    model_size: str = 'n',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = None,
    output_dir: str = 'models/segmentation/wbc_nc_ratio',
    pretrained: bool = True,
    patience: int = 20,
    save_period: int = 10
):
    """
    Train YOLO11-seg model for WBC segmentation.
    
    Args:
        data_yaml: Path to data.yaml configuration file
        model_size: YOLO11 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: Device to use (None = auto-detect)
        output_dir: Directory to save trained model
        pretrained: Use pretrained weights
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
    
    Returns:
        Training results
    """
    print("=" * 80)
    print("WBC Segmentation Training - YOLO11-seg")
    print("=" * 80)
    
    # Validate data.yaml exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    
    # Load data.yaml to check configuration
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nDataset Configuration:")
    print(f"  Train path: {data_config.get('train')}")
    print(f"  Val path: {data_config.get('val')}")
    print(f"  Test path: {data_config.get('test', 'Not specified')}")
    print(f"  Number of classes: {data_config.get('nc')}")
    print(f"  Classes: {data_config.get('names')}")
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nTraining Configuration:")
    print(f"  Model: YOLO11{model_size}-seg")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Output directory: {output_dir}")
    
    # Initialize model
    model_name = f'yolo11{model_size}-seg.pt' if pretrained else f'yolo11{model_size}-seg.yaml'
    print(f"\nInitializing model: {model_name}")
    model = YOLO(model_name)
    
    # Train model
    print("\nStarting training...\n")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=output_dir,
        name='train',
        exist_ok=True,
        pretrained=pretrained,
        patience=patience,
        save_period=save_period,
        verbose=True,
        # Augmentation settings
        hsv_h=0.015,      # HSV-Hue augmentation
        hsv_s=0.7,        # HSV-Saturation augmentation
        hsv_v=0.4,        # HSV-Value augmentation
        degrees=10.0,     # Rotation
        translate=0.1,    # Translation
        scale=0.5,        # Scaling
        shear=0.0,        # Shear
        perspective=0.0,  # Perspective
        flipud=0.0,       # Flip up-down
        fliplr=0.5,       # Flip left-right
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.0,        # Mixup augmentation
        # Optimization
        optimizer='AdamW',
        lr0=0.001,        # Initial learning rate
        lrf=0.01,         # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Loss weights
        box=7.5,          # Box loss weight
        cls=0.5,          # Class loss weight
        dfl=1.5,          # DFL loss weight
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    # Print best metrics
    print(f"\nBest Model Metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(M)', 'N/A')}")
    
    # Model paths
    best_model_path = Path(output_dir) / 'train' / 'weights' / 'best.pt'
    last_model_path = Path(output_dir) / 'train' / 'weights' / 'last.pt'
    
    print(f"\nSaved Models:")
    print(f"  Best: {best_model_path}")
    print(f"  Last: {last_model_path}")
    
    # Validate on test set if available
    if data_config.get('test'):
        print("\n" + "=" * 80)
        print("Validating on Test Set")
        print("=" * 80)
        
        test_results = model.val(
            data=str(data_path),
            split='test',
            device=device
        )
        
        print(f"\nTest Set Metrics:")
        print(f"  mAP50: {test_results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
        print(f"  mAP50-95: {test_results.results_dict.get('metrics/mAP50-95(M)', 'N/A')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO11-seg model for WBC segmentation'
    )
    
    parser.add_argument(
        'data',
        type=str,
        help='Path to data.yaml configuration file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLO11 model size (default: n)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/segmentation/wbc_nc_ratio',
        help='Output directory (default: models/segmentation/wbc_nc_ratio)'
    )
    
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Train from scratch without pretrained weights'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20)'
    )
    
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Train model
    train_wbc_segmentation(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_dir=args.output,
        pretrained=not args.no_pretrained,
        patience=args.patience,
        save_period=args.save_period
    )


if __name__ == "__main__":
    main()
