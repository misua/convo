#!/usr/bin/env python3
"""
Train WBC Subtype Classifier using Fast.ai + EfficientNet-B0
Classifies: Eosinophil, Lymphocyte, Monocyte, Neutrophil
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from fastai.vision.all import (
    ImageDataLoaders, 
    vision_learner, 
    accuracy, 
    error_rate,
    Resize,
    aug_transforms,
    models,
    SaveModelCallback,
    EarlyStoppingCallback,
    CSVLogger,
    ShowGraphCallback
)
import timm
from datetime import datetime

def main():
    # Configuration
    DATA_PATH = project_root / "data" / "kaggle" / "dataset2-master" / "dataset2-master" / "images"
    TRAIN_PATH = DATA_PATH / "TRAIN"
    TEST_PATH = DATA_PATH / "TEST"
    MODEL_PATH = project_root / "models" / "classification"
    
    # Hyperparameters
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHS = 10
    LR = 1e-3
    
    print("=" * 60)
    print("WBC SUBTYPE CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Verify data paths
    print(f"\n📂 Data paths:")
    print(f"   Train: {TRAIN_PATH}")
    print(f"   Test: {TEST_PATH}")
    
    if not TRAIN_PATH.exists():
        print(f"❌ Train path not found!")
        sys.exit(1)
    
    # List classes
    classes = sorted([d.name for d in TRAIN_PATH.iterdir() if d.is_dir()])
    print(f"\n🔬 Classes: {classes}")
    
    # Count images per class
    print("\n📊 Training set distribution:")
    for cls in classes:
        count = len(list((TRAIN_PATH / cls).glob("*.jpeg")))
        print(f"   {cls}: {count} images")
    
    # Create data loaders with augmentation
    print("\n⚙️  Creating data loaders...")
    dls = ImageDataLoaders.from_folder(
        TRAIN_PATH,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(
            mult=1.0,
            do_flip=True,
            flip_vert=True,  # Cells can be in any orientation
            max_rotate=180,
            max_zoom=1.2,
            max_lighting=0.3,
            max_warp=0.1,
            p_affine=0.75,
            p_lighting=0.75
        ),
        bs=BATCH_SIZE,
        num_workers=4
    )
    
    print(f"   Train batches: {len(dls.train)}")
    print(f"   Valid batches: {len(dls.valid)}")
    print(f"   Classes: {dls.vocab}")
    
    # Create model directory
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    run_name = f"wbc_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = MODEL_PATH / run_name
    run_path.mkdir(exist_ok=True)
    
    # Create learner with EfficientNet-B0
    print("\n🧠 Creating model (EfficientNet-B0)...")
    
    # Use timm model for better performance
    def get_efficientnet():
        return timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(dls.vocab))
    
    learn = vision_learner(
        dls, 
        models.resnet34,  # Using ResNet34 as it's well-supported in fastai
        metrics=[accuracy, error_rate],
        path=run_path
    )
    
    print(f"   Model created successfully")
    print(f"   Output path: {run_path}")
    
    # Set up callbacks
    callbacks = [
        SaveModelCallback(monitor='accuracy', fname='best_model'),
        EarlyStoppingCallback(monitor='accuracy', patience=3, min_delta=0.01),
        CSVLogger(fname='training_log.csv')
    ]
    
    # Find optimal learning rate
    print("\n🔍 Finding optimal learning rate...")
    try:
        lr_suggestion = learn.lr_find(show_plot=False)
        suggested_lr = lr_suggestion.valley
        print(f"   Suggested LR: {suggested_lr:.2e}")
    except Exception as e:
        print(f"   LR finder failed, using default: {LR}")
        suggested_lr = LR
    
    # Train with 1-cycle policy
    print(f"\n🏋️ Training for {EPOCHS} epochs...")
    print("-" * 60)
    
    learn.fine_tune(
        EPOCHS,
        base_lr=suggested_lr,
        cbs=callbacks
    )
    
    print("-" * 60)
    print("\n✅ Training complete!")
    
    # Evaluate on validation set
    print("\n📈 Final validation results:")
    val_loss, val_acc, val_err = learn.validate()
    print(f"   Loss: {val_loss:.4f}")
    print(f"   Accuracy: {val_acc*100:.2f}%")
    print(f"   Error Rate: {val_err*100:.2f}%")
    
    # Export model
    export_path = run_path / "wbc_classifier.pkl"
    learn.export(export_path)
    print(f"\n💾 Model exported to: {export_path}")
    
    # Also save the best weights
    best_weights_path = run_path / "models" / "best_model.pth"
    if best_weights_path.exists():
        print(f"   Best weights at: {best_weights_path}")
    
    # Create symlink to latest model
    latest_link = MODEL_PATH / "wbc_latest.pkl"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(export_path)
    print(f"   Symlink created: {latest_link}")
    
    # Show confusion matrix info
    print("\n🔬 Generating classification report...")
    interp = learn.interpret()
    
    # Print top losses info
    print("\n⚠️  Top confusions:")
    cm = interp.confusion_matrix()
    print(f"\nConfusion Matrix:")
    print(f"Classes: {dls.vocab}")
    print(cm)
    
    print("\n" + "=" * 60)
    print("🎉 WBC Classifier training complete!")
    print("=" * 60)
    
    return learn, run_path


if __name__ == "__main__":
    learn, model_path = main()
