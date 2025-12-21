# Tasks: Extend RBC Classification

## Current System State

**✅ Working Now (Rule-Based):**
- Sickle cell detection (geometric shape analysis)
- Thalassemia detection (microcytes, targets, teardrops)
- UI/PDF infrastructure ready for all 9 classes

**❌ NOT Working (Requires CNN Training):**
- Malaria detection (RING, TROPHOZOITE) - cannot detect internal parasites with geometry alone

**Note:** Phases 2-4 are REQUIRED to enable malaria detection. Without CNN training, malaria_infected will always be 0.

---

## Phase 1: Enum & Structure Updates (1 hour) ✅ COMPLETE

- [x] Update RBCShape enum in shape_classifier.py to include RING, TROPHOZOITE, SICKLE, SCHIZONT
- [x] Add sickle cell detection rules in classify() method (circularity < 0.5, elongation > 2.0)
- [x] Update ShapeClassification dataclass with new fields (is_infected, parasite_stage)
- [x] Update ShapePopulationStats to track malaria and sickle cell counts
- [x] Run unit tests to ensure enum changes don't break existing code

## Phase 2: Dataset Preparation (2-3 days) ⚠️ REQUIRED FOR MALARIA DETECTION

**Downloads Required (~550 MB):**
- NIH Malaria Dataset: ~400 MB (27,558 images)
- Sickle Cell Dataset: ~150 MB (erythrocytesIDB)

**Tasks:**
- [ ] Download NIH Malaria Dataset (27,558 images) from https://lhncbc.nlm.nih.gov/publication/pub9932
- [ ] Write script to extract infected RBC crops using YOLOv11-seg
- [ ] Manual QA: verify parasite annotations are accurate
- [ ] Download sickle cell dataset (erythrocytesIDB or SCD-Dataset) from https://github.com/MahmoudYidi/erythrocytesIDB
- [ ] Create unified dataset structure: `data/rbc_extended/train/` with 9 class folders
- [ ] Split data: 80% train, 10% val, 10% test
- [ ] Document dataset statistics (class distribution, image counts)

## Phase 3: Training Script Updates (2 hours) ⚠️ REQUIRED FOR MALARIA DETECTION - ✅ COMPLETE

- [x] Create train_rbc_classifier.py based on train_wbc_classifier.py
- [x] Configure for 3-class classification (normal, infected, sickle) with ResNet34
- [x] Add data augmentation (rotations, flips, color jitter for stain variation)
- [x] Add weighted loss to handle class imbalance (7.11:1 ratio)
- [x] Set up early stopping and model checkpointing
- [x] Add validation metrics (per-class accuracy, confusion matrix)
- [x] Test set evaluation with comprehensive classification report

## Phase 4: Model Training (4-8 hours GPU time) ⚠️ REQUIRED FOR MALARIA DETECTION - ✅ COMPLETE

**What gets trained:**
- Download ImageNet pretrained ResNet34 weights (~85 MB) - automatic via PyTorch
- Train new 3-class RBC classifier on datasets from Phase 2
- Output: Trained model file (~85 MB) saved as .pkl export

**Results:**
- Validation Accuracy: 96.79%
- Test Accuracy: 96.27%
- Per-class: infected 95.33%, normal 96.67%, sickle 100%
- Model: models/classification/rbc_3class_v1_20251221_022903/rbc_3class_classifier.pkl

**Tasks:**
- [x] Train on full dataset with learning rate finder
- [x] Fine-tune with discriminative learning rates
- [x] Evaluate on test set - achieved 96.27% overall accuracy
- [x] Analyze confusion matrix (ensure malaria not confused with artifacts)
- [x] Generate classification report with precision/recall per class
- [x] Save best model weights to models/classification/rbc_3class_v1/

## Phase 5: Integration (3 hours) - ✅ COMPLETE

- [x] Update shape_classifier.py to load trained CNN weights
- [x] Add _cnn_classify() method using raw PyTorch (bypasses fastai bugs)
- [x] Modify classify() to use CNN when available (70% confidence threshold)
- [x] Update analyze_blood_smear.py to auto-detect and load RBC CNN
- [x] Test CNN loading and inference
- [x] Hybrid approach: CNN for malaria/sickle, rule-based for thalassemia

## Phase 6: Risk Assessment Updates (2 hours) ✅ COMPLETE

- [x] Add malaria risk calculation in scripts/analyze_blood_smear.py
- [x] Add sickle cell risk calculation
- [x] Define urgency levels: malaria=urgent, sickle=referral, thalassemia=monitor
- [x] Update results JSON schema to include new risk fields
- [x] Add clinical interpretation text for each disorder

## Phase 7: Dashboard Updates (2 hours) ✅ COMPLETE

- [x] Update app.py to display extended shape distribution (9 bars instead of 6)
- [x] Add color coding: malaria=red alert, sickle=orange warning
- [x] Add new metrics cards: "Malaria Infected", "Sickle Cells"
- [x] Update sample images section to show malaria/sickle examples
- [x] Test with real malaria/sickle images

## Phase 8: PDF Report Updates (1 hour) ✅ COMPLETE

- [x] Update pdf_generator.py shape_analysis section for 9 classes
- [x] Add malaria findings subsection with parasite stage breakdown
- [x] Add sickle cell findings with percentage and risk level
- [x] Update clinical notes to include malaria/sickle interpretations
- [x] Add disclaimer about malaria requiring lab confirmation
- [x] Generate test PDFs with sample data

## Phase 9: Testing & Validation (2 hours)

- [ ] Create test dataset with known malaria/sickle/normal images
- [ ] Run inference on test set, measure accuracy
- [ ] Verify false positive rate < 10% for malaria
- [ ] Test edge cases: low parasitemia, mixed infections
- [ ] Performance benchmark: ensure latency unchanged
- [ ] Document known limitations

## Phase 10: Documentation (1 hour)

- [ ] Update README.md with extended classification capabilities
- [ ] Document new RBCShape classes in docstrings
- [ ] Add training instructions for future model updates
- [ ] Update config.yaml with new classification settings
- [ ] Add example images for each new class

## Dependencies

- Phase 1 → Phases 2-10 (enum must be updated first)
- Phase 2 → Phase 3 (need dataset before training script)
- Phase 3 → Phase 4 (need script before training)
- Phase 4 → Phases 5-8 (need trained model before integration)
- Phases 5-8 can be parallelized (different files)
- Phase 9 depends on Phases 5-8 (need full integration)

## Success Metrics

- [ ] Model achieves >90% accuracy on 9-class test set
- [ ] Malaria sensitivity >80%, specificity >90%
- [ ] Sickle cell sensitivity >85%
- [ ] End-to-end inference time <2s
- [ ] PDF report correctly displays all new findings
- [ ] Dashboard visualizations are clear and actionable
