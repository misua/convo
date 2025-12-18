# Change: Add Blood Cell Analyzer for Thalassemia Screening

## Why
Manual blood cell analysis for thalassemia screening is time-consuming and requires expert pathologists. An AI-powered system can automate RBC counting, morphology analysis, and provide clinicians with actionable insights, reducing diagnostic time and improving accessibility in resource-limited settings.

## What Changes
- Add image classification model for blood cell types (RBC, WBC subtypes, platelets)
- Implement RBC morphology analysis (microcytosis, hypochromia detection)
- Add cell counting and measurement pipeline
- Build clinician review dashboard with confidence scores
- Create automated RBC indices calculation

## Impact
- Affected specs: `blood-cell-classification`, `morphology-analysis`, `clinical-dashboard`
- Affected code: New project - training pipeline, inference API, dashboard UI
- Hardware requirements: RTX 3060 (12GB VRAM) for local training
- Data dependencies: BCCD Dataset or Kaggle Blood Cells dataset

## Feasibility Assessment

### Hardware Capability (RTX 3060)
| Aspect | Assessment |
|--------|------------|
| VRAM (12GB) | ✅ Sufficient for ResNet/EfficientNet with batch size 16-32 |
| Training | ✅ Can train custom models; ~2-4 hours for full dataset |
| Inference | ✅ Real-time inference (>30 FPS) achievable |
| Limitation | ⚠️ Large models (ViT-Large) may require gradient checkpointing |

### Dataset Options

**Option A: BCCD Dataset (Recommended for MVP)**
- Source: https://github.com/Shenggan/BCCD_Dataset
- Size: ~364 images with YOLO/VOC annotations
- Classes: RBC, WBC, Platelets
- Pros: Includes bounding boxes for detection, well-annotated
- Cons: Smaller dataset size

**Option B: Kaggle Blood Cell Images**
- Source: https://www.kaggle.com/datasets/paultimothymooney/blood-cells
- Size: ~12,500 augmented images
- Classes: Eosinophil, Lymphocyte, Monocyte, Neutrophil (WBC subtypes)
- Pros: Larger dataset, good for classification
- Cons: Focused on WBC subtypes, no RBC morphology labels

**Recommendation:** Two-stage pipeline using BOTH datasets:
1. **BCCD** → Train YOLOv8 to detect/locate cells in full field-of-view images
2. **Kaggle** → Train classifier for WBC subtypes on cropped cells

User's images are full microscope fields (1152x1703), requiring detection before classification.

## Success Criteria
1. Cell detection mAP > 0.80 on BCCD test set
2. WBC subtype classification accuracy > 90% on Kaggle test set
3. End-to-end inference < 500ms per full image on RTX 3060
4. Works with user's existing blood smear images (verified: 1152x1703 JPG)

## Out of Scope (Phase 1)
- FDA/CE certification (research prototype only)
- Integration with LIMS systems
- Multi-institution validation
- Mobile app deployment
- Complex microscope calibration (use sensible defaults)
