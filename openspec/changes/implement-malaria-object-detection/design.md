# Technical Design: Malaria Object Detection with BBBC041

## Overview

Replace cell-level classification (ResNet34 CNN) with object-level detection (YOLOv11) to identify malaria parasites in blood smears. This addresses the root cause of false positives: domain shift between training data (isolated cells) and inference data (full blood smears with artifacts).

## Architecture Decision

### Why YOLO over Other Approaches?

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Faster R-CNN** | High accuracy | Slow (2-stage), complex | ❌ Too slow |
| **SSD** | Fast, simpler | Lower accuracy on small objects | ❌ Parasites are small |
| **YOLO** | Fast, accurate, single-stage | Requires more data | ✅ **Selected** |
| **EfficientDet** | Excellent accuracy | Complex, slower than YOLO | ❌ Overkill |
| **CenterNet** | Anchor-free, simple | Less ecosystem support | ❌ Harder to maintain |

**YOLO Selection Rationale:**
- Already in use (YOLOv11-seg for cell segmentation)
- Proven in medical imaging (mAP ~0.85 on microscopy)
- Fast inference (~10-20ms on CPU)
- Strong community support (Ultralytics)
- Anchor-free YOLOv11 handles small objects well

### Model Variant: YOLOv11n (Nano)

| Variant | Params | Size | Inference (CPU) | mAP@0.5 | Decision |
|---------|--------|------|----------------|---------|----------|
| **YOLOv11n** | 2.6M | 6MB | 10-15ms | ~0.80 | ✅ **Primary** |
| YOLOv11s | 9.4M | 22MB | 20-30ms | ~0.85 | 🟡 Fallback if accuracy insufficient |
| YOLOv11m | 20.1M | 50MB | 50-80ms | ~0.87 | ❌ Too slow |

**Justification**: Start with YOLOv11n for speed. Medical context allows 85% precision threshold. If validation shows <80% mAP, upgrade to YOLOv11s.

## Data Pipeline

### BBBC041 Dataset Structure

```
bbbc041_malaria/
├── images/
│   ├── brazil/           # Stefanie Lopes dataset
│   ├── southeast_asia/   # Benoit Malleret dataset  
│   └── timecourse/       # Gabriel Rangel dataset
└── annotations/
    ├── brazil_annotations.json
    ├── southeast_asia_annotations.json
    └── timecourse_annotations.json
```

**Annotation Format (Original):**
```json
{
  "images": [{"id": 1, "file_name": "C1_thinblood_IMG_...", "width": 1600, "height": 1200}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 3,  # 1=uninfected, 2=leukocyte, 3=ring, 4=trophozoite, 5=schizont, 6=gametocyte
      "bbox": [x, y, width, height],  # COCO format: top-left xy + size
      "area": 2025,
      "iscrowd": 0,
      "difficult": 0  # Some cells marked as difficult
    }
  ],
  "categories": [
    {"id": 1, "name": "uninfected_rbc"},
    {"id": 2, "name": "leukocyte"},
    {"id": 3, "name": "ring"},
    {"id": 4, "name": "trophozoite"},
    {"id": 5, "name": "schizont"},
    {"id": 6, "name": "gametocyte"}
  ]
}
```

### Conversion to YOLO Format

**YOLO Format (per image):**
```
# Format: class x_center y_center width height (all normalized 0-1)
2 0.456 0.678 0.023 0.028  # ring at center (0.456, 0.678), size (0.023, 0.028)
3 0.789 0.234 0.031 0.029  # trophozoite
```

**Conversion Logic:**
```python
# COCO bbox: [x_min, y_min, width, height] (pixels)
# YOLO bbox: [x_center, y_center, width, height] (normalized 0-1)

x_center = (x_min + width/2) / image_width
y_center = (y_min + height/2) / image_height
w_norm = width / image_width
h_norm = height / image_height
yolo_class = coco_category_id - 1  # YOLO uses 0-indexed classes
```

### Train/Val/Test Split

**Strategy: Stratified by Source**
```
Total: 1,364 images
├── Train: 1,091 images (80%)
│   ├── Brazil: ~450
│   ├── Southeast Asia: ~450
│   └── Timecourse: ~190
├── Val: 136 images (10%)
└── Test: 137 images (10%)
```

**Rationale**: Maintain representation from all three sources to ensure model generalizes across different microscopy setups, staining protocols, and geographic parasite variations.

### Data Augmentation

**Training Time Augmentation (Ultralytics built-in):**
```yaml
augmentation:
  hsv_h: 0.015      # Hue variation (staining differences)
  hsv_s: 0.7        # Saturation (Giemsa stain intensity)
  hsv_v: 0.4        # Brightness (microscope illumination)
  degrees: 15       # Rotation (field orientation)
  translate: 0.1    # Translation (field shifting)
  scale: 0.5        # Zoom (different magnifications)
  mosaic: 1.0       # Mosaic (combine 4 images)
  mixup: 0.1        # MixUp (blend images)
```

**Validation/Inference**: No augmentation (raw images only)

## Model Architecture

### YOLOv11n Specifics

**Backbone**: CSPDarknet (modified for v11)
- Efficient cross-stage partial connections
- Reduces parameters while maintaining accuracy

**Neck**: PAN (Path Aggregation Network)
- Multi-scale feature fusion
- Critical for detecting small parasites (5-20 pixels)

**Head**: Decoupled detection head
- Separate branches for classification and localization
- Anchor-free (reduces hyperparameters)

**Input**: 640x640 (resized with letterboxing)
**Output**: Predictions at 3 scales (80x80, 40x40, 20x20)

### Loss Function

**YOLOv11 Loss = Classification Loss + Localization Loss + Distribution Loss**

```python
# Ultralytics implementation
loss = (
    λ_cls * BCE(predicted_class, true_class) +      # Classification
    λ_box * CIoU(predicted_bbox, true_bbox) +       # Bounding box
    λ_dfl * DFL(predicted_distribution, true_bbox)  # Distribution focal loss
)

# λ_cls=0.5, λ_box=7.5, λ_dfl=1.5 (YOLOv11 defaults)
```

**Why CIoU?** Complete IoU accounts for overlap, center distance, and aspect ratio - crucial for small, circular parasites.

## Training Strategy

### Transfer Learning Approach

**Step 1: Start with COCO Pre-trained Weights**
```python
model = YOLO('yolov11n.pt')  # COCO pre-trained
```

**Rationale**: COCO contains "person" and "cell phone" classes with similar appearance patterns (small, circular objects). Transfer learning reduces training time from days to hours.

**Step 2: Fine-tune on BBBC041**
```python
results = model.train(
    data='data/bbbc041_malaria/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU (or 'cpu' if unavailable)
)
```

### Hyperparameter Selection

**Learning Rate Schedule: Cosine Annealing**
```python
lr0: 0.001       # Initial learning rate
lrf: 0.01        # Final learning rate (1% of initial)
warmup_epochs: 3 # Warm-up period
```

**Why Cosine?** Smooth decay prevents oscillation in final epochs, critical for small dataset fine-tuning.

**Batch Size: 16**
- Fits in 8GB GPU memory with 640x640 images
- Larger batches (32) improve stability but risk overfitting on 1,364 images

**Early Stopping:**
```python
patience: 20  # Stop if no improvement for 20 epochs
```

### Class Imbalance Handling

**BBBC041 Class Distribution:**
```
Uninfected RBCs: ~70,000 (87.5%)  # Majority class
Leukocytes:      ~5,000  (6.25%)
Ring:            ~3,500  (4.4%)   # Most common parasite
Trophozoite:     ~1,200  (1.5%)
Schizont:        ~200    (0.25%)  # Rare
Gametocyte:      ~100    (0.13%)  # Rarest
```

**Mitigation Strategies:**

1. **Weighted Loss** (optional, if needed):
```python
# Inverse frequency weighting
class_weights = {
    0: 1.0,    # uninfected_rbc (baseline)
    1: 14.0,   # leukocyte
    2: 20.0,   # ring
    3: 58.0,   # trophozoite
    4: 350.0,  # schizont
    5: 700.0   # gametocyte
}
```

2. **Oversampling Rare Classes**:
- Duplicate images with schizonts/gametocytes 2-3x

3. **Evaluation Focus**:
- Monitor per-class mAP, not just overall
- Prioritize ring detection (most clinically relevant)

## Inference Pipeline

### Integration with Existing System

**Current Pipeline:**
```python
# analyzer_blood_smear.py
def analyze(image_path):
    # 1. Segment cells with YOLOv11-seg
    cells = segment_cells(image)
    
    # 2. Classify RBCs with ResNet34
    for cell in rbc_cells:
        if cnn_classify(cell) == 'infected':  # ❌ High false positive
            infected_count += 1
    
    return {"malaria_infected": infected_count}
```

**New Pipeline:**
```python
def analyze(image_path):
    # 1. Segment cells with YOLOv11-seg (unchanged)
    cells = segment_cells(image)
    
    # 2. Detect parasites with YOLOv11 malaria detector
    parasites = malaria_detector.detect(image)  # ✅ Direct detection
    
    # 3. Calculate metrics
    parasite_density = len(parasites) / wbc_count * 100  # WHO standard
    
    return {
        "parasites_detected": parasites,
        "parasite_density": parasite_density,
        "summary": count_by_stage(parasites)
    }
```

### Post-Processing

**1. Non-Maximum Suppression (NMS)**
```python
conf_threshold: 0.5   # Minimum confidence
iou_threshold: 0.4    # Overlap threshold for duplicate removal
```

**Rationale**: IoU=0.4 allows slight overlap (parasites can be close together) but removes clear duplicates.

**2. Size Filtering**
```python
min_bbox_area: 25 pixels²   # Filter noise
max_bbox_area: 2500 pixels² # Filter false positives (too large = not parasite)
```

**3. Confidence Calibration**
```python
# Optional: Adjust confidence scores based on validation
calibrated_conf = 1 / (1 + exp(-α * (raw_conf - β)))
# α=2.0, β=0.5 (tuned on validation set)
```

## Visualization

### Bounding Box Overlay

**Color Coding by Class:**
```python
COLORS = {
    'ring':        (0, 255, 0),     # Green (early stage - good prognosis)
    'trophozoite': (255, 165, 0),   # Orange (mature - treatment monitoring)
    'schizont':    (255, 0, 0),     # Red (dividing - high parasitemia)
    'gametocyte':  (255, 0, 255),   # Magenta (sexual stage - transmission)
    'leukocyte':   (0, 255, 255),   # Cyan (reference for density calculation)
    'uninfected':  (128, 128, 128)  # Gray (background - usually not shown)
}
```

**Label Format:**
```
"Ring 92%"  # Class + confidence
```

### Dashboard Integration

**New Section: "Malaria Detection"**
```
┌─────────────────────────────────────────┐
│ Malaria Screening Results               │
├─────────────────────────────────────────┤
│ Status: POSITIVE - Parasites Detected   │
│                                         │
│ Stage Distribution:                     │
│   Ring Forms:      ███████ 3 (75%)     │
│   Trophozoites:    ██ 1 (25%)          │
│   Schizonts:       0                    │
│   Gametocytes:     0                    │
│                                         │
│ Parasite Density: 4.2 per 100 WBCs     │
│ (WHO standard metric)                   │
│                                         │
│ [View Annotated Image]                  │
└─────────────────────────────────────────┘
```

## Performance Optimization

### Inference Speed Targets

| Component | Target | Strategy |
|-----------|--------|----------|
| **YOLOv11n forward pass** | <15ms | Use TorchScript JIT compilation |
| **NMS post-processing** | <2ms | Ultralytics optimized C++ |
| **Visualization** | <5ms | OpenCV hardware acceleration |
| **Total end-to-end** | <25ms | Pipeline optimization |

### Memory Optimization

**Model Size:**
- YOLOv11n: 6 MB (fits in CPU cache)
- Compare: ResNet34 malaria CNN: 83 MB

**Batch Inference** (optional for multiple images):
```python
# Process 4 images at once (GPU)
results = model(images, batch=4)  # ~30ms total = 7.5ms per image
```

## Testing Strategy

### Unit Tests

**Test Coverage:**
```python
# test_malaria_detector.py
def test_load_model():
    """Verify model loads without errors."""

def test_detect_on_known_positive():
    """Test on image with labeled parasites."""
    
def test_detect_on_known_negative():
    """Ensure 0 detections on confirmed negative."""
    
def test_bbox_format():
    """Verify bounding box coordinates are valid."""
    
def test_confidence_thresholds():
    """Test varying confidence levels."""
```

### Integration Tests

```python
# test_integration.py
def test_full_pipeline():
    """End-to-end test: image → results."""
    
def test_api_compatibility():
    """Ensure output format matches expected schema."""
    
def test_performance_benchmark():
    """Verify inference time < 25ms."""
```

### Validation Tests

**Hospital Validation Set:**
```python
# test_clinical_validation.py
def test_hospital_negative_cases():
    """All hospital negatives should show 0 parasites."""
    assert all(len(detect(img)) == 0 for img in negative_cases)
    
def test_hospital_positive_cases():
    """Hospital positives should detect >0 parasites."""
    assert all(len(detect(img)) > 0 for img in positive_cases)
```

## Deployment Considerations

### Feature Flag Architecture

**Gradual Rollout:**
```python
# config.yaml
malaria_detection:
  mode: "yolo"  # Options: "yolo", "cnn", "hybrid"
  fallback: true  # Fall back to CNN if YOLO fails
```

**Hybrid Mode** (optional safety net):
```python
if config.mode == "hybrid":
    yolo_result = yolo_detector.detect(image)
    cnn_result = cnn_classifier.classify(crops)
    
    # Both must agree for positive result (high specificity)
    if yolo_result and cnn_result:
        return "POSITIVE"
```

### Monitoring & Alerts

**Metrics to Track:**
```python
# Performance monitoring
- inference_time_ms
- detections_per_image
- confidence_distribution
- false_positive_rate (from feedback)
- false_negative_rate (from feedback)
```

**Alerting Thresholds:**
- If FPR > 5%: Alert for model review
- If inference_time > 50ms: Performance degradation
- If confidence < 0.3 for all detections: Potential data shift

## Rollback Plan

**If YOLO Performs Worse:**
1. Revert to CNN classifier via feature flag
2. Investigate failure modes (precision vs recall)
3. Retrain with adjusted hyperparameters or YOLOv11s
4. Consider ensemble approach

**Rollback Criteria:**
- FPR > 10% on validation set
- FNR > 30% on known positives
- Inference time > 100ms
- User complaints > 3 in first week

## Future Enhancements

### Phase 2 Improvements (Post-MVP)

1. **Multi-Species Detection**
   - Distinguish P. falciparum vs P. vivax
   - Requires species-labeled dataset

2. **Parasitemia Quantification**
   - Automated cell counting
   - WHO-compliant reporting

3. **Model Ensemble**
   - Combine YOLOv11n + YOLOv11s for higher confidence
   - Reduce false negatives on edge cases

4. **Active Learning**
   - Collect user corrections
   - Retrain on edge cases

## References

- YOLOv11 Architecture: [Ultralytics Documentation](https://docs.ultralytics.com/)
- BBBC041 Dataset Paper: [Nature Methods, 2012](https://doi.org/10.1038/nmeth.2083)
- WHO Malaria Guidelines: [Microscopy Quality Assurance](https://www.who.int/publications/i/item/9789241549394)
- Medical YOLO Applications: [arXiv:2104.11892](https://arxiv.org/abs/2104.11892)
