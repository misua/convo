# Change: Add Thalassemia Shape Diagnosis with Plain-Language Reporting

## Why

The current blood cell analyzer detects cells and classifies WBCs but lacks:
1. **RBC Shape Analysis** - Thalassemia causes characteristic shape abnormalities (target cells, tear drops) that are not currently detected
2. **User-Friendly Diagnosis** - Current reports are technical; non-medical users cannot understand the clinical significance
3. **Sample Image Support** - User's trained sample images (e.g., `1000026798.jpg`) should work seamlessly with enhanced analysis

Better thalassemia detection requires analyzing RBC morphology beyond just size (microcytosis) and color (hypochromia). Shape abnormalities like **target cells (codocytes)** and **tear drop cells (dacrocytes)** are strong indicators that the current system misses.

## What Changes

### 1. RBC Shape Classification Model
- Add instance segmentation (YOLOv11-seg) to extract precise cell boundaries
- Train shape classifier to identify 6 RBC morphology classes:
  - Normal, Microcyte, Target Cell, Tear Drop, Spherocyte, Irregular
- Calculate shape metrics: circularity, elongation, central pallor pattern

### 2. Enhanced Thalassemia Risk Scoring
- Integrate shape findings into risk calculation
- Weight target cells and tear drops heavily (strong thalassemia indicators)
- Provide confidence intervals on risk scores

### 3. Plain-Language Diagnosis Interface
- Add "Patient-Friendly Report" tab in Gradio dashboard
- Translate technical findings to understandable language
- Visual indicators (traffic light system) for risk levels
- Explain what each finding means and recommended next steps

### 4. Sample Image Compatibility
- Ensure existing trained models work with `data/samples/*.jpg`
- Add calibration presets for user's microscope setup
- Store analysis history for comparison

## Impact

| Area | Changes |
|------|---------|
| **Affected specs** | `morphology-analysis` (MODIFIED), `clinical-dashboard` (MODIFIED), new `shape-classification` |
| **Affected code** | `scripts/analyze_blood_smear.py`, `app.py`, new `src/models/shape_classifier.py` |
| **New models** | YOLOv11-seg for segmentation, EfficientNet-V2-S for shape classification |
| **Hardware** | RTX 3060 compatible; ~50ms added latency per image |
| **Data needs** | Annotated RBC shape dataset (~2000 cells minimum) |

## Success Criteria

1. Shape classification accuracy > 85% on test set
2. Target cell detection recall > 80% (critical for thalassemia)
3. Non-medical users rate diagnosis clarity ≥ 4/5 in usability test
4. Existing sample image `1000026798.jpg` produces valid shape analysis
5. End-to-end latency remains < 2 seconds on RTX 3060

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Insufficient shape training data | Use synthetic augmentation; partner with pathology lab |
| Shape classifier overfits | Cross-validation; extensive augmentation |
| User misinterprets "High Risk" | Clear disclaimers; emphasize "screening, not diagnosis" |

## Out of Scope

- Automated treatment recommendations
- Integration with electronic health records
- Multi-language support (English only for MVP)
- Sickle cell detection (different morphology patterns)
