# Change: Implement Malaria Object Detection with BBBC041

## Why

The current malaria detection system uses **cell-level classification** with a ResNet34 CNN trained on the NIH malaria dataset (isolated cell crops). This approach has a **critical flaw**: it produces **high false positive rates** (16.82% malaria detection on hospital-confirmed negative cases) because:

1. **Domain Shift**: Trained on clean, isolated cells but applied to full blood smears with artifacts, overlaps, and staining variations
2. **Wrong Detection Target**: Classifies general cell morphology instead of detecting specific parasite structures
3. **Clinical Mismatch**: Microscopists identify specific parasite forms (rings, trophozoites), not just "infected cells"

**Real-World Failure:**
- Hospital diagnosis: "No malarial parasite seen" (negative)
- Our system: "37 infected cells (16.82%)" 
- Root cause: Model misinterprets artifacts, cell overlap, and staining variations as infection

**Solution:** Switch from **classification** to **object detection** using YOLO trained on BBBC041, which provides:
- Bounding box annotations on full blood smears (not isolated cells)
- Detection of specific parasite structures (rings, trophozoites, schizonts, gametocytes)
- Spatial localization matching clinical workflow

## What Changes

### 1. Replace CNN Classifier with YOLO Detector

**Current Architecture:**
```
Blood Smear → YOLOv11-seg → RBC crops → ResNet34 → "infected"/"normal"
                                       ↓
                            High False Positives (artifacts → "infected")
```

**New Architecture:**
```
Blood Smear → YOLOv11 Malaria Detector → Bounding boxes on parasites
                                        ↓
                           "Ring form at (x,y), confidence 0.92"
```

### 2. BBBC041 Dataset Integration

**Dataset Specifications:**
- **Source**: Broad Bioimage Benchmark Collection (BBBC041)
- **Size**: 1,364 images, ~80,000 cells
- **Annotations**: Bounding box coordinates + class labels
- **Classes**: 6 total
  - Uninfected RBCs
  - Leukocytes (WBCs)
  - Ring forms (early malaria)
  - Trophozoites (mature parasite)
  - Schizonts (dividing stage)
  - Gametocytes (sexual stage)
- **Expert-Labeled**: Annotated by malaria researcher from hospital
- **Real Blood Smears**: Giemsa-stained microscopy (matches clinical samples)
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 3.0

### 3. Detection Pipeline Changes

| Component | Current | New |
|-----------|---------|-----|
| **Detection Method** | Classification (infected/normal) | Object detection (parasite localization) |
| **Training Data** | NIH isolated cells (27,558) | BBBC041 full smears (1,364 images) |
| **Output** | Cell label ("infected") | Bounding boxes + parasite stage |
| **Confidence Threshold** | 0.7 (70%) | 0.5 (50% - YOLO standard) |
| **False Positive Rate** | ~17% on negatives | Expected <1% |
| **Clinical Alignment** | Low (general morphology) | High (specific structures) |

### 4. New Capabilities

**Parasite Stage Detection:**
- Identify specific life cycle stages (ring/trophozoite/schizont/gametocyte)
- Enable treatment monitoring (stages indicate drug efficacy)
- Support species identification (P. falciparum vs P. vivax patterns)

**Spatial Localization:**
- Display bounding boxes on detected parasites
- Show exact coordinates for manual verification
- Calculate parasite density (parasites per field)

**Improved Reporting:**
```
BEFORE: "16.82% cells infected" (vague, wrong)
AFTER:  "3 ring forms detected at coordinates [(234,567), (445,123), (789,234)]
         1 trophozoite detected at (345,678)
         Parasite density: 4 parasites per 100 WBCs"
```

## Architecture Changes

### File Structure
```
blood-cell-analyzer/
├── data/
│   └── bbbc041_malaria/              # NEW: BBBC041 dataset
│       ├── images/
│       ├── annotations/
│       └── data.yaml                 # YOLO format config
├── models/
│   └── detection/
│       └── malaria_yolo11n/          # NEW: Trained YOLO model
│           ├── weights/
│           │   └── best.pt
│           └── data.yaml
├── scripts/
│   ├── prepare_bbbc041_dataset.py    # NEW: Convert BBBC041 to YOLO
│   └── train_malaria_detector.py     # NEW: Train YOLO on BBBC041
└── src/
    └── models/
        ├── malaria_detector.py       # NEW: YOLO inference wrapper
        └── shape_classifier.py       # MODIFIED: Remove CNN malaria detection
```

### Code Changes

**Remove from `shape_classifier.py`:**
- `self.cnn_model` initialization for malaria
- `_cnn_classify()` method for malaria detection
- `RBCShape.RING`, `RBCShape.TROPHOZOITE` (now in separate detector)
- CNN confidence threshold logic

**Add `malaria_detector.py`:**
```python
class MalariaDetector:
    """YOLO-based malaria parasite detector."""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.classes = ['uninfected_rbc', 'leukocyte', 
                       'ring', 'trophozoite', 'schizont', 'gametocyte']
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect parasites in full blood smear image."""
        results = self.model(image, conf=0.5)
        return self._parse_detections(results)
```

**Modify `analyze_blood_smear.py`:**
- Replace CNN malaria detection with YOLO detector
- Update result structure to include parasite coordinates
- Calculate parasite density (WHO standard: parasites per 100 WBCs)

## Dataset Preparation

### BBBC041 Download & Conversion Pipeline

**Step 1: Download**
```bash
wget https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip
unzip malaria.zip -d data/bbbc041_malaria
```

**Step 2: Parse Annotations**
- BBBC041 provides JSON/CSV annotations with bounding boxes
- Format: `{image_id, x, y, width, height, class}`

**Step 3: Convert to YOLO Format**
- YOLO expects: `class x_center y_center width height` (normalized 0-1)
- Create train/val/test splits (80/10/10)
- Generate `data.yaml` with class names and paths

**Step 4: Data Augmentation**
- Horizontal/vertical flips
- Rotation (±15°)
- Brightness/contrast adjustments
- Mosaic augmentation (YOLO built-in)

## Training Specifications

### Model Selection: YOLOv11n (Nano)
**Rationale:**
- Already used in project (`yolo11n.pt`, `yolo11n-seg.pt`)
- Fast inference (~10-15ms per image)
- Small model size (~6MB)
- Good accuracy for medical imaging (mAP ~0.85 expected)

### Training Configuration
```yaml
# Training hyperparameters
epochs: 100
batch_size: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
weight_decay: 0.0005

# Data augmentation
hsv_h: 0.015  # Hue augmentation
hsv_s: 0.7    # Saturation augmentation
hsv_v: 0.4    # Value augmentation
degrees: 15   # Rotation
mosaic: 1.0   # Mosaic augmentation
```

### Success Metrics
| Metric | Target | Rationale |
|--------|--------|-----------|
| **mAP@0.5** | >0.80 | Overall detection accuracy |
| **Precision** | >0.85 | Minimize false positives (critical for negatives) |
| **Recall** | >0.75 | Catch most parasites |
| **Ring F1** | >0.80 | Most common stage, critical for early detection |
| **Inference Time** | <20ms | Real-time analysis |

### Validation Strategy
- **Cross-validation**: 5-fold on BBBC041
- **Hospital validation**: Test on user's negative cases (should show 0 detections)
- **Known positives**: Validate on confirmed malaria-positive samples
- **Stage distribution**: Ensure balanced performance across all stages

## Impact

### Performance Improvements
| Metric | Current (CNN) | New (YOLO) | Change |
|--------|--------------|------------|--------|
| **False Positive Rate** | ~17% | <1% | 📉 94% reduction |
| **Clinical Specificity** | Low | High | ✅ Matches microscopists |
| **Parasite Localization** | None | Exact coordinates | ✅ New capability |
| **Stage Identification** | Generic "infected" | Specific stages | ✅ New capability |
| **Inference Time** | ~50ms | ~15ms | 📉 70% faster |

### API Changes

**BloodSmearAnalyzer.analyze() Output:**
```python
# BEFORE
{
    "shape_analysis": {
        "malaria_infected": 37,
        "malaria_infected_pct": 16.82
    }
}

# AFTER
{
    "malaria_analysis": {
        "parasites_detected": [
            {"class": "ring", "bbox": [234, 567, 45, 45], "confidence": 0.92},
            {"class": "trophozoite", "bbox": [345, 678, 52, 48], "confidence": 0.87}
        ],
        "parasite_density": 4.0,  # per 100 WBCs (WHO standard)
        "summary": {
            "ring": 3,
            "trophozoite": 1,
            "schizont": 0,
            "gametocyte": 0
        },
        "total_parasites": 4
    }
}
```

### User-Facing Changes

**Dashboard:**
- New "Malaria Detection" section showing parasite detections
- Visual overlay of bounding boxes on blood smear image
- Stage breakdown chart (ring/trophozoite/schizont/gametocyte)
- Parasite density calculation (WHO standard)

**PDF Report:**
- "Malaria Screening" section with detection results
- Annotated image showing detected parasites
- Stage-specific counts and interpretation
- Clinical recommendations based on findings

**CLI Output:**
```
Malaria Analysis:
  Status: POSITIVE - Parasites Detected
  Ring Forms: 3 detected
  Trophozoites: 1 detected
  Parasite Density: 4 per 100 WBCs
  Recommendation: Immediate laboratory confirmation required
```

## Risks & Mitigations

### Risk 1: Insufficient Training Data
**Risk**: 1,364 images might not cover all variations
**Mitigation**:
- Use data augmentation (flips, rotations, color shifts)
- Transfer learning from COCO-pretrained YOLOv11
- Evaluate on hospital validation set, add samples if needed

### Risk 2: Performance Degradation
**Risk**: YOLO might be slower than CNN
**Mitigation**:
- Use YOLOv11n (nano) - optimized for speed
- Benchmark shows ~15ms inference (acceptable)
- Option to use YOLOv11s if accuracy needs improvement

### Risk 3: Integration Complexity
**Risk**: Replacing CNN might break existing workflows
**Mitigation**:
- Maintain backward compatibility in API
- Feature flag for CNN vs YOLO (allow gradual rollout)
- Comprehensive integration tests

### Risk 4: False Negatives on Low Parasitemia
**Risk**: Missing rare parasites in negative-appearing samples
**Mitigation**:
- Low confidence threshold (0.3) for screening mode
- Manual review workflow for borderline cases
- Disclaimer: "Screening tool, not diagnostic replacement"

## Dependencies

### New Python Packages
```
ultralytics>=8.1.0      # YOLO training & inference
albumentations>=1.3.1   # Data augmentation
pycocotools>=2.0.7      # Annotation format conversion
```

### External Resources
- BBBC041 dataset (2.26 GB download)
- Pre-trained YOLOv11n weights (6 MB)

### Compute Requirements
- **Training**: GPU with 8GB+ VRAM (Google Colab Free tier sufficient)
- **Inference**: CPU-only capable (15-20ms per image)
- **Training Time**: 4-6 hours on T4 GPU (100 epochs)

## Success Criteria

### Quantitative Metrics
- [ ] mAP@0.5 > 0.80 on BBBC041 test set
- [ ] Precision > 0.85 (minimize false positives)
- [ ] Recall > 0.75 (catch most parasites)
- [ ] False positive rate < 1% on confirmed negative samples
- [ ] Inference time < 20ms per image

### Qualitative Metrics
- [ ] Hospital negative case shows 0 detections (not 16.82%)
- [ ] Detected parasites align with microscopist annotations
- [ ] Stage classifications match expert labels
- [ ] Bounding boxes accurately localize parasites

### Clinical Validation
- [ ] Test on 10+ confirmed negative samples → 0 false positives
- [ ] Test on 10+ confirmed positive samples → >90% detection rate
- [ ] Expert review: "Would trust this for screening"

## Rollout Strategy

### Phase 1: Development & Training (Week 1)
- Download BBBC041 dataset
- Implement conversion script
- Train YOLOv11n model
- Validate on test set

### Phase 2: Integration (Week 1-2)
- Create `MalariaDetector` class
- Update `BloodSmearAnalyzer` pipeline
- Modify dashboard and reports
- Add visualization overlays

### Phase 3: Validation (Week 2)
- Test on hospital negative cases
- Benchmark performance metrics
- User acceptance testing
- Documentation updates

### Phase 4: Deployment (Week 3)
- Feature flag rollout (optional CNN fallback)
- Monitor false positive/negative rates
- Collect feedback from medical professionals
- Iterate based on real-world performance

## Alternatives Considered

### Alternative 1: Keep CNN, Increase Threshold
**Pros**: Minimal code changes, quick fix
**Cons**: Doesn't address root cause (domain shift), still misses spatial info
**Decision**: Rejected - band-aid solution

### Alternative 2: Use Pre-trained Model (e.g., CenterNet)
**Pros**: No training required, faster deployment
**Cons**: Unknown dataset quality, hard to customize, potential licensing issues
**Decision**: Rejected - BBBC041 is gold standard, worth training

### Alternative 3: Hybrid CNN + YOLO
**Pros**: Belt-and-suspenders approach
**Cons**: Slower inference, complex logic, redundant
**Decision**: Rejected - YOLO alone is sufficient and cleaner

### Alternative 4: Wait for Better Dataset
**Pros**: Might find larger dataset in future
**Cons**: BBBC041 is already excellent (expert-labeled, 80K cells), unnecessary delay
**Decision**: Rejected - proceed with proven dataset

## Related Changes

- **add-multi-disorder-detection**: This change replaces the malaria detection component planned in that change
- **extend-rbc-classification**: Independent - RBC shape analysis remains separate from malaria detection

## References

- [BBBC041 Dataset](https://data.broadinstitute.org/bbbc/BBBC041/)
- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [WHO Malaria Microscopy Guidelines](https://www.who.int/publications/i/item/9789241549394)
- GitHub: [malaria_centernet](https://github.com/hnmspirit/malaria_centernet) - Similar approach validation
