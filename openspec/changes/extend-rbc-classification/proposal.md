# Change: Extend RBC Classification for Multi-Disorder Detection

## Why

The current RBC shape classifier has 6 classes focused on thalassemia indicators. The existing architecture already:
- Segments individual RBCs via YOLOv11-seg
- Classifies RBC morphology using rule-based metrics + optional EfficientNet-V2-S CNN
- Crops and analyzes each RBC independently

We can **extend the same classifier** to detect malaria-infected RBCs and sickle cells without adding new models. The classifier already receives cropped RBC images and can identify visual patterns.

## What Changes

### Extend RBCShape Enum

**Current (6 classes):**
- Normal, Microcyte, Target, Teardrop, Spherocyte, Irregular

**Extended (9-10 classes):**
- Normal, Microcyte, Target, Teardrop, Spherocyte, Irregular
- **RING** (malaria ring stage - purple dot/ring inside RBC)
- **TROPHOZOITE** (malaria mature form - larger purple structure)
- **SICKLE** (sickle cell - crescent/elongated shape)
- _(Optional)_ **SCHIZONT** (late malaria - multiple nuclei)

### Classification Approach

**Rule-based detection (current method):**
- Sickle: circularity < 0.5, elongation > 2.0
- Malaria parasites: Not detectable by shape metrics alone

**CNN-based detection (when model is trained):**
- Train ResNet34 on extended 9-class dataset (same architecture as WBC classifier)
- Model sees cropped RBC image, identifies both shape AND internal parasites
- Can detect subtle features (purple rings, hemozoin pigment)

### Key Insight

The RBC classifier already gets perfect crops from YOLO segmentation. Adding malaria detection requires training the CNN on infected RBC images, not building a separate parasite detector.

## Architecture - No Changes Required

```
Current (works today):
  Image → YOLOv11-seg → RBC crops → Rule-based classifier → 6 classes

With Extended Classes (after CNN training):
  Image → YOLOv11-seg → RBC crops → ResNet34 → 9 classes
                                    ↓
                            Both shape + infection status
```

## Dataset Requirements

| Class | Images Needed | Source |
|-------|--------------|---------|
| Normal | 3,000 | Current BCCD + augmentation |
| Microcyte | 500 | Current data |
| Target | 500 | Current data |
| Teardrop | 300 | Current data |
| Spherocyte | 300 | Current data |
| Irregular | 500 | Current data |
| **Ring** | 2,000 | NIH Malaria Dataset (27K images) |
| **Trophozoite** | 1,000 | NIH Malaria Dataset |
| **Sickle** | 1,500 | erythrocytesIDB, SCD datasets |
| **Schizont** | 500 | NIH Malaria Dataset (optional) |

**Total: ~10K labeled RBC crops**

## Implementation Plan

### Phase 1: Update Enum & Rule-Based Logic (30 min)
- Add RING, TROPHOZOITE, SICKLE to RBCShape enum
- Add sickle detection rules (circularity < 0.5, elongation > 2.0)
- Malaria classes default to "requires CNN" (fall back to irregular)

### Phase 2: Dataset Curation (1-2 days)
- Download NIH Malaria Dataset
- Extract infected RBC crops using existing segmentation
- Download sickle cell datasets
- Organize into train/val splits

### Phase 3: Train Extended Classifier (4-8 hours)
- Modify training script to use 9 classes
- Train ResNet34 with augmentation (same as WBC classifier)
- Validate on holdout set
- Target: >90% accuracy on normal/sickle/malaria

### Phase 4: Integration (2-3 hours)
- Update shape_classifier.py to use trained model
- Update risk assessment logic
- Update PDF report to show malaria/sickle findings
- Update dashboard visualization

## Impact

| Component | Change |
|-----------|--------|
| RBCShape enum | 6 → 9 classes |
| shape_classifier.py | +50 lines (new class handling) |
| Training script | Modify for 9-class dataset |
| Risk assessment | Add malaria/sickle thresholds |
| PDF report | Add new findings sections |
| **New models** | **None** - extends existing |
| **Latency** | **No change** - same inference path |

## Success Criteria

1. ✅ Sickle cell detection: >85% sensitivity on test set
2. ✅ Malaria ring detection: >80% sensitivity (critical for diagnosis)
3. ✅ False positive rate: <10% for malaria (avoid alarm fatigue)
4. ✅ Inference time: unchanged (<1.5s total)
5. ✅ Model size: <50MB (EfficientNet-V2-S baseline)

## Why This Approach Works

1. **Leverage existing pipeline** - no architectural changes
2. **YOLO already crops RBCs perfectly** - ideal input for classifier
4. **ResNet34 sees full RBC image** - can identify internal parasites (same model as WBC classifier)
4. **Rule-based fallback** - works for sickle cells today
5. **Single model** - simpler than multi-model approach

## Out of Scope

- Parasite species identification (P. falciparum vs P. vivax)
- Parasite load quantification (requires specialized counting)
- WBC-based disorders (leukemia) - different pipeline
- Multiple infections per cell (edge case)

## Validation Notes

The existing WBC classifier already uses ResNet34 successfully (97.9% accuracy on Raabin-WBC). We'll use the same proven architecture for RBC classification:
```python
# Similar to train_wbc_classifier.py
learn = vision_learner(
    dls, 
    models.resnet34,  # Same as WBC classifier
    metrics=[accuracy, error_rate]
)
```

This confirms the architecture is proven - we just need to:
1. Update the enum
2. Train ResNet34 on extended RBC dataset
3. Load trained weights in shape_classifier.py
