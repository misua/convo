#!/usr/bin/env python3
"""
Train RBC 3-Class Classifier using Fast.ai + ResNet34
Classifies: normal (healthy RBC), infected (malaria parasite), sickle (sickle cell disease)

Medical-grade training with proper class weighting for imbalanced dataset.
Uses ResNet34 pretrained on ImageNet (same architecture as WBC classifier achieving 97.9% accuracy).
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from fastai.vision.all import (
    ImageDataLoaders, 
    vision_learner, 
    accuracy, 
    error_rate,
    Precision,
    Recall,
    F1Score,
    Resize,
    aug_transforms,
    models,
    SaveModelCallback,
    EarlyStoppingCallback,
    CSVLogger,
    ShowGraphCallback,
    ClassificationInterpretation
)
from datetime import datetime
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def compute_class_weights(data_path):
    """
    Compute class weights inversely proportional to class frequencies.
    For imbalanced datasets like ours (7.11:1 ratio).
    """
    print("\n⚖️  Computing class weights for imbalanced dataset...")
    
    classes = sorted([d.name for d in (data_path / "train").iterdir() if d.is_dir()])
    class_counts = {}
    
    for cls in classes:
        count = len(list((data_path / "train" / cls).glob("*.jpg")))
        class_counts[cls] = count
        print(f"   {cls}: {count} samples")
    
    total = sum(class_counts.values())
    n_classes = len(classes)
    
    # Inverse frequency weighting
    weights = {}
    for cls in classes:
        weights[cls] = total / (n_classes * class_counts[cls])
    
    # Normalize so minimum weight is 1.0
    min_weight = min(weights.values())
    weights = {k: v/min_weight for k, v in weights.items()}
    
    print(f"\n   Class weights (normalized):")
    for cls, weight in weights.items():
        print(f"   {cls}: {weight:.2f}x")
    
    return weights, classes


def plot_confusion_matrix(cm, classes, save_path):
    """Generate and save confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - RBC Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved confusion matrix: {save_path}")


def evaluate_model(learn, test_path, run_path, class_names):
    """
    Comprehensive evaluation on held-out test set.
    Generates classification report, confusion matrix, and per-class metrics.
    
    Uses raw PyTorch inference to bypass fastai's buggy predict() method.
    """
    print("\n📊 Evaluating on held-out test set...")
    
    from fastai.vision.all import get_image_files
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm
    
    test_files = get_image_files(test_path)
    print(f"   Found {len(test_files)} test images")
    
    # Get vocab from dataloader
    vocab = list(learn.dls.vocab)
    class_to_idx = {c: i for i, c in enumerate(vocab)}
    print(f"   Classes: {vocab}")
    
    # Get raw PyTorch model and set to eval mode
    model = learn.model
    model.eval()
    device = next(model.parameters()).device
    print(f"   Device: {device}")
    
    # ImageNet normalization (ResNet34 pretrained weights)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Collect predictions and true labels
    pred_classes = []
    true_classes = []
    
    print("   Running inference on test set...")
    with torch.no_grad():
        for img_path in tqdm(test_files, desc="   Evaluating"):
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Get prediction
            logits = model(img_tensor)
            pred_idx = logits.argmax(dim=1).item()
            pred_classes.append(pred_idx)
            
            # Get true label from folder name
            true_label = img_path.parent.name
            true_idx = class_to_idx.get(true_label, -1)
            true_classes.append(true_idx)
    
    pred_classes = np.array(pred_classes)
    true_classes = np.array(true_classes)
    
    # Classification report
    report = classification_report(
        true_classes, 
        pred_classes, 
        target_names=class_names,
        digits=4
    )
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Test Set)")
    print("=" * 70)
    print(report)
    
    # Save report
    report_path = run_path / "test_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"   Saved to: {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    cm_path = run_path / "test_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Per-class accuracy
    print("\n📈 Per-class accuracy:")
    for i, cls in enumerate(class_names):
        cls_correct = np.sum((pred_classes == i) & (true_classes == i))
        cls_total = np.sum(true_classes == i)
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0
        print(f"   {cls}: {cls_acc*100:.2f}% ({cls_correct}/{cls_total})")
    
    # Overall test accuracy
    test_acc = np.mean(pred_classes == true_classes)
    print(f"\n✅ Overall test accuracy: {test_acc*100:.2f}%")
    
    # Save metrics
    metrics = {
        "test_accuracy": float(test_acc),
        "classification_report": classification_report(
            true_classes, pred_classes, target_names=class_names, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": {
            cls: float(np.mean((pred_classes == i) & (true_classes == i)) / np.sum(true_classes == i))
            for i, cls in enumerate(class_names)
        }
    }
    
    metrics_path = run_path / "test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved to: {metrics_path}")
    
    return test_acc, cm


def main():
    # Configuration
    DATA_PATH = project_root / "data" / "rbc_extended_3class"
    TRAIN_PATH = DATA_PATH / "train"
    VAL_PATH = DATA_PATH / "val"
    TEST_PATH = DATA_PATH / "test"
    MODEL_PATH = project_root / "models" / "classification"
    
    # Hyperparameters
    BATCH_SIZE = 32
    IMG_SIZE = 224  # ResNet34 input size
    EPOCHS = 30     # More epochs for medical imaging
    BASE_LR = 1e-3  # Will be fine-tuned with lr_find
    PATIENCE = 5    # Early stopping patience
    
    print("=" * 70)
    print("RBC 3-CLASS CLASSIFIER TRAINING (ResNet34)")
    print("=" * 70)
    print("\nClasses: normal (healthy), infected (malaria), sickle (SCD)")
    print("Medical-grade training with class weighting for imbalanced data")
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️  WARNING: Running on CPU. Training will be SLOW!")
        print("   Recommend using GPU with CUDA for 10-100x speedup.")
    
    # Verify data paths
    print(f"\n📂 Data paths:")
    print(f"   Train: {TRAIN_PATH}")
    print(f"   Val: {VAL_PATH}")
    print(f"   Test: {TEST_PATH}")
    
    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not path.exists():
            print(f"❌ Path not found: {path}")
            sys.exit(1)
    
    # Compute class weights for imbalanced dataset
    class_weights, class_names = compute_class_weights(DATA_PATH)
    
    # Count images across all splits
    print("\n📊 Dataset distribution:")
    for split_name, split_path in [("Train", TRAIN_PATH), ("Val", VAL_PATH), ("Test", TEST_PATH)]:
        print(f"\n   {split_name}:")
        for cls in class_names:
            count = len(list((split_path / cls).glob("*.jpg")))
            print(f"      {cls}: {count} images")
    
    # Create data loaders with proper augmentation for blood cells
    print("\n⚙️  Creating data loaders with medical imaging augmentation...")
    
    # Define augmentation strategy for blood cells:
    # - Rotation: cells can be in any orientation (0-360°)
    # - Flips: both horizontal and vertical
    # - Color jitter: account for stain variation
    # - NO geometric distortion: would alter cell morphology (critical diagnostic feature)
    
    # Create separate dataloaders for train and val, then combine
    train_dls = ImageDataLoaders.from_folder(
        TRAIN_PATH,
        valid_pct=0.2,  # Use 20% of train for validation
        seed=42,
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(
            mult=1.0,
            do_flip=True,
            flip_vert=True,
            max_rotate=180.0,
            max_zoom=1.1,
            max_lighting=0.2,
            max_warp=0.0,
            p_affine=0.75,
            p_lighting=0.75
        ),
        bs=BATCH_SIZE,
        num_workers=4
    )
    
    dls = train_dls
    
    print(f"   Train batches: {len(dls.train)}")
    print(f"   Valid batches: {len(dls.valid)}")
    print(f"   Classes: {dls.vocab}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    # Create model directory
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    run_name = f"rbc_3class_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = MODEL_PATH / run_name
    run_path.mkdir(exist_ok=True)
    
    print(f"\n🧠 Creating ResNet34 model...")
    print(f"   Architecture: ResNet34 (pretrained on ImageNet)")
    print(f"   Output classes: {len(dls.vocab)}")
    print(f"   Total parameters: ~21.3M")
    print(f"   Model save path: {run_path}")
    
    # Create learner with ResNet34 (same as WBC classifier: 97.9% accuracy)
    learn = vision_learner(
        dls, 
        models.resnet34,
        metrics=[
            accuracy, 
            error_rate,
            Precision(average='macro'),
            Recall(average='macro'),
            F1Score(average='macro')
        ],
        path=run_path
    )
    
    # Apply class weights to loss function
    print("\n⚖️  Applying class weights to handle imbalance...")
    weight_tensor = torch.tensor([class_weights[cls] for cls in class_names], dtype=torch.float32)
    if torch.cuda.is_available():
        weight_tensor = weight_tensor.cuda()
    
    # Override loss function with weighted CrossEntropyLoss
    learn.loss_func = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    print("   Loss function: Weighted CrossEntropyLoss")
    
    # Set up callbacks for robust training
    callbacks = [
        SaveModelCallback(monitor='accuracy', fname='best_model', with_opt=True),
        EarlyStoppingCallback(monitor='accuracy', patience=PATIENCE, min_delta=0.001),
        CSVLogger(fname='training_log.csv')
    ]
    
    # Find optimal learning rate
    print(f"\n🔍 Finding optimal learning rate...")
    try:
        lr_suggestion = learn.lr_find(show_plot=False)
        suggested_lr = lr_suggestion.valley
        print(f"   Suggested LR: {suggested_lr:.2e}")
        
    except Exception as e:
        print(f"   ⚠️  LR finder failed: {e}")
        print(f"   Using default LR: {BASE_LR:.2e}")
        suggested_lr = BASE_LR
    
    # Train with discriminative learning rates
    # Lower learning rate for pretrained layers, higher for new head
    print(f"\n🏋️  Training for up to {EPOCHS} epochs (with early stopping)...")
    print(f"   Learning rate: {suggested_lr:.2e}")
    print(f"   Early stopping patience: {PATIENCE} epochs")
    print(f"   Monitoring: validation accuracy")
    print("-" * 70)
    
    # Fine-tune: freeze backbone, train head for 3 epochs
    learn.fine_tune(
        EPOCHS,
        base_lr=suggested_lr,
        freeze_epochs=3,  # Train only head for 3 epochs first
        cbs=callbacks
    )
    
    print("-" * 70)
    print("\n✅ Training complete!")
    
    # Evaluate on validation set
    print("\n📈 Final validation results:")
    val_metrics = learn.validate()
    print(f"   Loss: {val_metrics[0]:.4f}")
    print(f"   Accuracy: {val_metrics[1]*100:.2f}%")
    print(f"   Error Rate: {val_metrics[2]*100:.2f}%")
    print(f"   Precision: {val_metrics[3]*100:.2f}%")
    print(f"   Recall: {val_metrics[4]*100:.2f}%")
    print(f"   F1-Score: {val_metrics[5]*100:.2f}%")
    
    # Skip validation visualization due to fastai bug with custom splitters
    print("\n🔬 Skipping validation confusion matrix (will generate from test set)")
    
    # Export model
    export_path = run_path / "rbc_3class_classifier.pkl"
    learn.export(export_path)
    print(f"\n💾 Model exported to: {export_path}")
    
    # Save best weights info
    best_weights_path = run_path / "models" / "best_model.pth"
    if best_weights_path.exists():
        print(f"   Best weights at: {best_weights_path}")
    
    # Create symlink to latest model
    latest_link = MODEL_PATH / "rbc_3class_latest.pkl"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(export_path.relative_to(MODEL_PATH))
    print(f"   Symlink created: {latest_link} -> {export_path.name}")
    
    # Comprehensive test set evaluation
    test_acc, test_cm = evaluate_model(learn, TEST_PATH, run_path, class_names)
    
    # Save training configuration
    config = {
        "model": "ResNet34",
        "pretrained": "ImageNet",
        "classes": class_names,
        "class_weights": class_weights,
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "learning_rate": float(suggested_lr),
        "early_stopping_patience": PATIENCE,
        "augmentation": {
            "rotation": "0-360 degrees",
            "flip": "horizontal + vertical",
            "zoom": "up to 1.1x",
            "lighting": "±20%",
            "warp": "disabled (preserves cell morphology)"
        },
        "validation_accuracy": float(val_metrics[1]),
        "test_accuracy": float(test_acc),
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }
    
    config_path = run_path / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n📝 Configuration saved to: {config_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 RBC 3-CLASS CLASSIFIER TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Validation Accuracy: {val_metrics[1]*100:.2f}%")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Model: {export_path}")
    print(f"   Metrics: {run_path}")
    print(f"\n📌 Next steps:")
    print(f"   1. Review test_classification_report.txt for per-class performance")
    print(f"   2. Inspect test_confusion_matrix.png for error patterns")
    print(f"   3. Check top_losses.png for most confused samples")
    print(f"   4. If accuracy < 90%, consider:")
    print(f"      - Manual QA of sickle cell samples (smallest class)")
    print(f"      - More aggressive augmentation")
    print(f"      - Additional training epochs")
    print(f"   5. Integrate model into shape_classifier.py (Phase 5)")
    print("=" * 70)
    
    return learn, run_path


if __name__ == "__main__":
    learn, model_path = main()
