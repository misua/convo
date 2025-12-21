# Design: Thalassemia Shape Diagnosis System

## Context

Enhancing the blood cell analyzer with RBC shape analysis for better thalassemia detection. The current system detects cells with YOLOv8 but doesn't analyze cell shapes. Thalassemia causes characteristic morphological changes that a shape-aware system can detect.

### Current State
```
Image → YOLOv8 Detect → Count cells → Size/Color analysis → Risk score
                              ↓
                    (Shape NOT analyzed)
```

### Proposed State
```
Image → YOLOv11-seg → Instance masks → Shape classifier → Population stats → Risk score
                           ↓                  ↓                                    ↓
                     Precise boundaries   6-class shape          Plain-language diagnosis
```

### Stakeholders
- **Non-medical users** - Need understandable diagnosis reports
- **Clinicians** - Need detailed technical findings for review
- **Patients** - Need clear next-step guidance

## Goals / Non-Goals

### Goals
- Detect thalassemia-associated RBC shapes (target cells, tear drops)
- Provide diagnosis explanations a layperson can understand
- Work with user's existing sample images without retraining
- Maintain sub-2-second inference time
- Display visual shape gallery in dashboard

### Non-Goals
- Replace clinical diagnosis
- Detect sickle cell disease
- Provide treatment recommendations
- Support DICOM/medical imaging formats

## Technical Decisions

### Decision 1: Segmentation Model - YOLOv11-seg

**Choice:** Ultralytics YOLO11n-seg for instance segmentation

**Rationale:**
- Provides precise cell boundaries needed for shape metrics
- YOLOv11 is 22% smaller than v8 with better accuracy
- Single model handles detection + segmentation
- Drop-in replacement for existing YOLOv8

**Shape Metrics from Masks:**
```python
# From instance mask, calculate:
circularity = 4 * π * area / perimeter²  # 1.0 = perfect circle
elongation = major_axis / minor_axis     # 1.0 = symmetric
solidity = area / convex_hull_area       # detects concavities
central_pallor = center_intensity / edge_intensity
```

**Alternatives Considered:**
| Option | Pros | Cons |
|--------|------|------|
| Mask R-CNN | Higher accuracy | 3x slower inference |
| SAM (Segment Anything) | Zero-shot | Too slow for real-time |
| YOLOv8-seg | Stable | YOLOv11 is better |

### Decision 2: Shape Classifier - EfficientNet-V2-S

**Choice:** EfficientNet-V2-Small for RBC shape classification

**Rationale:**
- Modern architecture with strong small-object performance
- 22M params (fits in VRAM alongside detection model)
- Pre-trained on ImageNet, fine-tune on RBC shapes
- 3x faster training than ViT with comparable accuracy

**Shape Classes:**
```python
RBC_SHAPES = [
    "normal",       # Biconcave disc
    "microcyte",    # Small but normal shape
    "target",       # Bull's-eye pattern (codocyte) ⭐ THALASSEMIA
    "teardrop",     # Dacrocyte ⭐ THALASSEMIA  
    "spherocyte",   # Sphere (hereditary spherocytosis)
    "irregular",    # Catch-all for other abnormalities
]
```

### Decision 3: Plain-Language Generator

**Choice:** Template-based explanation system with severity mapping

**Rationale:**
- Deterministic (no hallucinations)
- Fast (no LLM inference needed)
- Easy to review/update by medical advisors
- Consistent phrasing

**Architecture:**
```python
class DiagnosisExplainer:
    def explain(self, findings: dict) -> PlainReport:
        """Convert technical findings to plain language."""
        
        # Map technical terms to plain language
        explanations = {
            "target_cells": {
                "name": "Bull's-eye shaped cells",
                "meaning": "Some of your red blood cells have an unusual pattern...",
                "significance": "This is commonly seen in thalassemia trait..."
            },
            ...
        }
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THALASSEMIA SHAPE DIAGNOSIS SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐       │
│  │   Image      │    │  YOLOv11-seg      │    │  Shape Metrics     │       │
│  │   Input      │───▶│  Detection +      │───▶│  Calculator        │       │
│  │              │    │  Segmentation     │    │  (circularity,etc) │       │
│  └──────────────┘    └───────────────────┘    └────────────────────┘       │
│                              │                          │                   │
│                              ▼                          ▼                   │
│                      ┌───────────────┐         ┌────────────────┐          │
│                      │  Cell Crops   │         │  Shape Stats   │          │
│                      │  + Masks      │         │  per Cell      │          │
│                      └───────────────┘         └────────────────┘          │
│                              │                          │                   │
│                              ▼                          │                   │
│                      ┌───────────────────┐              │                   │
│                      │  EfficientNet-V2  │              │                   │
│                      │  Shape Classifier │              │                   │
│                      │  (6 classes)      │              │                   │
│                      └───────────────────┘              │                   │
│                              │                          │                   │
│                              ▼                          ▼                   │
│                      ┌─────────────────────────────────────┐               │
│                      │      Population Analyzer            │               │
│                      │  - % target cells                   │               │
│                      │  - % teardrops                      │               │
│                      │  - RDW-proxy                        │               │
│                      │  - Microcyte ratio                  │               │
│                      └─────────────────────────────────────┘               │
│                                       │                                     │
│                                       ▼                                     │
│                      ┌─────────────────────────────────────┐               │
│                      │      Risk Score Engine              │               │
│                      │  Weighted combination:              │               │
│                      │  - Shape abnormalities (0.35)       │               │
│                      │  - Microcytosis (0.30)              │               │
│                      │  - Hypochromia (0.25)               │               │
│                      │  - RDW elevation (0.10)             │               │
│                      └─────────────────────────────────────┘               │
│                                       │                                     │
│                    ┌──────────────────┴──────────────────┐                 │
│                    ▼                                      ▼                 │
│         ┌─────────────────────┐              ┌─────────────────────┐       │
│         │  Technical Report   │              │  Plain-Language     │       │
│         │  (for clinicians)   │              │  Diagnosis Panel    │       │
│         │  - All metrics      │              │  - Traffic lights   │       │
│         │  - Confidence       │              │  - Simple terms     │       │
│         │  - Cell gallery     │              │  - Next steps       │       │
│         └─────────────────────┘              └─────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## UI Design: Plain-Language Diagnosis Panel

### Traffic Light System
```
┌─────────────────────────────────────────────────────────────────┐
│  🩸 Your Blood Analysis Results                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Overall Assessment:                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  🟡 MODERATE FINDINGS - Follow-up Recommended           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  What We Found:                                                 │
│  ───────────────                                                │
│  📊 We analyzed 274 red blood cells from your sample.          │
│                                                                 │
│  ✅ Cell count: Normal                                          │
│  ⚠️ Cell size: 18% of cells are smaller than typical           │
│  ⚠️ Cell shape: 12% have an unusual "bull's-eye" pattern       │
│  ✅ Cell color: Normal hemoglobin distribution                  │
│                                                                 │
│  What This Might Mean:                                          │
│  ─────────────────────                                          │
│  The patterns we found are sometimes seen in people with        │
│  thalassemia trait - a common inherited blood condition.        │
│  Many people have this trait and live completely normal lives.  │
│                                                                 │
│  ⚠️ Important: This is a screening tool, NOT a diagnosis.      │
│                                                                 │
│  Recommended Next Steps:                                        │
│  ───────────────────────                                        │
│  1. 🏥 Visit your doctor to discuss these findings             │
│  2. 🧪 Ask about a Hemoglobin Electrophoresis test             │
│  3. 📋 Bring this report to your appointment                   │
│                                                                 │
│  [📥 Download Report]  [📧 Email to Doctor]  [❓ Learn More]   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Shape Gallery Component
```
┌─────────────────────────────────────────────────────────────────┐
│  🔬 Cell Shape Gallery                          [Filter: All ▼] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Cells of Interest (flagged for review):                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │  🎯    │ │  🎯    │ │  💧    │ │  🎯    │ │  ⚪    │        │
│  │ Target │ │ Target │ │Teardrop│ │ Target │ │Sphero- │        │
│  │  92%   │ │  87%   │ │  84%   │ │  81%   │ │ cyte   │        │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│                                                                 │
│  Shape Distribution:                                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Normal    ████████████████████████████████████ 78%     │    │
│  │ Target    ████████                             12%     │    │
│  │ Microcyte ██████                                8%     │    │
│  │ Teardrop  ██                                    2%     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Pipeline (Offline)
```
1. Annotate RBC shapes in existing dataset
   └── Use LabelImg or Roboflow for segmentation masks
   
2. Train YOLOv11-seg on BCCD + shape annotations
   └── python scripts/train_shape_segmenter.py
   
3. Train EfficientNet-V2 shape classifier on cell crops
   └── python scripts/train_shape_classifier.py
   
4. Validate on held-out test set
   └── python scripts/evaluate_shapes.py
```

### Inference Pipeline (Runtime)
```
User uploads image
    │
    ▼
YOLOv11-seg detects cells + generates masks
    │
    ▼
For each RBC:
    ├── Calculate shape metrics from mask
    ├── Crop cell image
    └── Classify shape with EfficientNet
    │
    ▼
Aggregate population statistics
    │
    ▼
Calculate thalassemia risk score
    │
    ▼
Generate reports (technical + plain-language)
    │
    ▼
Display in Gradio dashboard
```

## Model Training Configuration

### YOLOv11-seg
```yaml
model: yolo11n-seg.pt
epochs: 100
imgsz: 640
batch: 16
data: data/bccd_yolo/data_seg.yaml
```

### EfficientNet-V2-S Shape Classifier
```yaml
model: efficientnet_v2_s
pretrained: imagenet
num_classes: 6
epochs: 30
batch_size: 32
learning_rate: 0.0001
augmentation:
  - rotation: [-15, 15]
  - scale: [0.9, 1.1]
  - color_jitter: 0.2
  - horizontal_flip: true
```

## Backward Compatibility

| Existing Component | Impact |
|-------------------|--------|
| YOLOv8 detection | Replaced by YOLOv11-seg (drop-in) |
| ResNet34 classifier | Unchanged (WBC classification) |
| `analyze_blood_smear.py` | Extended with shape analysis |
| `app.py` Gradio dashboard | New tab added |
| Sample images | Fully compatible |
| Config.yaml | New `shape_analysis` section |
