# Implementation Tasks: Malaria Object Detection

## Phase 1: Dataset Preparation (Day 1-2)

### Task 1.1: Download BBBC041 Dataset
- [ ] Create download script `scripts/download_bbbc041.sh`
- [ ] Download malaria.zip (2.26 GB) from Broad Institute
- [ ] Extract to `data/bbbc041_malaria/`
- [ ] Verify file integrity (checksums if available)
- [ ] Document dataset structure in `data/bbbc041_malaria/README.md`

**Validation**: Directory contains 1,364 images across 3 subdirectories (brazil, southeast_asia, timecourse)

**Dependencies**: None (start immediately)

---

### Task 1.2: Parse BBBC041 Annotations
- [ ] Create `scripts/parse_bbbc041_annotations.py`
- [ ] Parse JSON/CSV annotations from BBBC041
- [ ] Extract bounding boxes, class labels, image metadata
- [ ] Handle "difficult" flags (exclude or include based on testing)
- [ ] Generate statistics: total cells, class distribution, bbox sizes

**Validation**: Script outputs summary showing ~80,000 cells with class breakdown

**Dependencies**: Task 1.1 (requires downloaded dataset)

---

### Task 1.3: Convert to YOLO Format
- [ ] Create `scripts/prepare_bbbc041_dataset.py`
- [ ] Convert COCO bbox format to YOLO format (normalized coordinates)
- [ ] Create train/val/test split (80/10/10) stratified by source
- [ ] Generate YOLO label files (.txt) for each image
- [ ] Create `data/bbbc041_malaria/data.yaml` config
- [ ] Verify conversions with visual spot-checks

**Validation**: 
- Train: 1,091 images with .txt labels
- Val: 136 images with .txt labels  
- Test: 137 images with .txt labels
- data.yaml contains correct paths and 6 class names

**Dependencies**: Task 1.2 (requires parsed annotations)

---

### Task 1.4: Data Quality Validation
- [ ] Create `scripts/validate_dataset.py`
- [ ] Check for missing label files
- [ ] Verify bbox coordinates are within [0, 1] range
- [ ] Ensure no empty label files
- [ ] Visualize 10 random samples with bboxes overlaid
- [ ] Check class distribution per split

**Validation**: All 1,364 images have valid labels, no out-of-bounds bboxes

**Dependencies**: Task 1.3 (requires YOLO format data)

---

## Phase 2: Model Training (Day 3-4)

### Task 2.1: Create Training Script
- [ ] Create `scripts/train_malaria_detector.py`
- [ ] Configure YOLOv11n base model
- [ ] Set hyperparameters: epochs=100, batch=16, imgsz=640
- [ ] Configure data augmentation (hsv, rotation, mosaic)
- [ ] Add early stopping (patience=20)
- [ ] Enable checkpointing (save best/last weights)
- [ ] Add logging (TensorBoard or Ultralytics default)

**Validation**: Script runs without errors on 1 epoch test

**Dependencies**: Task 1.4 (requires validated dataset)

---

### Task 2.2: Train YOLOv11n Model
- [ ] Run full training (100 epochs, ~4-6 hours on GPU)
- [ ] Monitor training metrics (loss, mAP, precision, recall)
- [ ] Save training logs to `models/detection/malaria_yolo11n/training.log`
- [ ] Save best weights to `models/detection/malaria_yolo11n/weights/best.pt`
- [ ] Generate training curves (loss, mAP over epochs)

**Validation**: 
- Training completes successfully
- mAP@0.5 > 0.75 on validation set
- Precision > 0.80, Recall > 0.70

**Dependencies**: Task 2.1 (requires training script)

**Parallelizable**: Can run overnight/background

---

### Task 2.3: Evaluate Model Performance
- [ ] Create `scripts/evaluate_malaria_detector.py`
- [ ] Run inference on test set (137 images)
- [ ] Calculate metrics: mAP@0.5, mAP@0.5-0.95, precision, recall, F1
- [ ] Generate per-class metrics (ring, trophozoite, schizont, gametocyte)
- [ ] Create confusion matrix
- [ ] Visualize predictions on 20 random test images
- [ ] Document results in `models/detection/malaria_yolo11n/evaluation_report.md`

**Validation**: 
- mAP@0.5 > 0.80 (target met)
- Precision > 0.85 (minimize false positives)
- Recall > 0.75 (catch most parasites)

**Dependencies**: Task 2.2 (requires trained model)

---

### Task 2.4: Hospital Validation Test
- [ ] Test on hospital negative case (should show 0 detections)
- [ ] Document false positive rate on negatives
- [ ] If FPR > 1%, adjust confidence threshold or retrain
- [ ] Test on any available positive cases
- [ ] Compare results with CNN classifier (baseline)

**Validation**: Hospital negative case shows 0 detections (not 16.82%)

**Dependencies**: Task 2.3 (requires evaluated model)

---

## Phase 3: Code Integration (Day 5-6)

### Task 3.1: Create MalariaDetector Class
- [ ] Create `src/models/malaria_detector.py`
- [ ] Implement `MalariaDetector` class with YOLO model loading
- [ ] Add `detect()` method: image → list of detections
- [ ] Implement `Detection` dataclass: class, bbox, confidence
- [ ] Add confidence thresholding (default=0.5)
- [ ] Add NMS configuration (iou_threshold=0.4)
- [ ] Write unit tests in `tests/test_malaria_detector.py`

**Validation**: Unit tests pass, detector loads model and runs inference

**Dependencies**: Task 2.2 (requires trained model weights)

**Parallelizable**: Can start while training runs (use placeholder weights)

---

### Task 3.2: Remove CNN Malaria Detection
- [ ] Modify `src/models/shape_classifier.py`
- [ ] Remove `self.cnn_model` initialization for malaria
- [ ] Remove `_cnn_classify()` method
- [ ] Remove `RBCShape.RING`, `RBCShape.TROPHOZOITE` enums
- [ ] Update `classify()` method to remove CNN malaria logic
- [ ] Update docstrings to reflect changes
- [ ] Add migration notes in code comments

**Validation**: RBC shape classifier still works for thalassemia detection

**Dependencies**: Task 3.1 (ensures replacement is ready)

---

### Task 3.3: Integrate into BloodSmearAnalyzer
- [ ] Modify `scripts/analyze_blood_smear.py`
- [ ] Add `MalariaDetector` initialization in `__init__`
- [ ] Replace CNN malaria detection with YOLO detector call
- [ ] Calculate parasite density (parasites per 100 WBCs)
- [ ] Update result structure with new malaria fields
- [ ] Handle detector failures gracefully (try/except)
- [ ] Add logging for detection events

**Validation**: Full pipeline runs without errors, produces expected output

**Dependencies**: Tasks 3.1, 3.2 (requires new detector and cleaned classifier)

---

### Task 3.4: Update Result Schema
- [ ] Modify result dictionary structure in `analyze_blood_smear.py`
- [ ] Add `malaria_analysis` section with:
  - `parasites_detected` (list of detections)
  - `parasite_density` (float)
  - `summary` (dict of counts by stage)
  - `total_parasites` (int)
- [ ] Remove old `malaria_infected` and `malaria_infected_pct` fields
- [ ] Update JSON serialization to handle new types
- [ ] Document schema in `docs/api_schema.md`

**Validation**: Output JSON validates against new schema

**Dependencies**: Task 3.3 (requires integrated pipeline)

---

## Phase 4: UI Updates (Day 7)

### Task 4.1: Update Dashboard (Gradio)
- [ ] Modify `app.py` dashboard
- [ ] Add "Malaria Detection" section after "Shape Analysis"
- [ ] Display parasite counts by stage (ring/trophozoite/schizont/gametocyte)
- [ ] Show parasite density with WHO interpretation
- [ ] Add annotated image with bounding boxes overlay
- [ ] Add color-coded legend for parasite stages
- [ ] Handle case with 0 detections (show "No parasites detected")

**Validation**: Dashboard displays malaria results correctly for positive and negative cases

**Dependencies**: Task 3.4 (requires updated result schema)

**Parallelizable**: Can work on UI while backend integrates

---

### Task 4.2: Update Visualization
- [ ] Create `src/inference/malaria_visualization.py`
- [ ] Implement `draw_parasite_bboxes()` function
- [ ] Use color coding: green=ring, orange=trophozoite, red=schizont, magenta=gametocyte
- [ ] Add confidence labels to bboxes
- [ ] Create legend showing color meanings
- [ ] Ensure overlays are visible on microscopy images
- [ ] Add option to toggle bbox visibility

**Validation**: Annotated images clearly show detected parasites with correct colors

**Dependencies**: Task 3.1 (requires detection output format)

**Parallelizable**: Can develop independently using mock detections

---

### Task 4.3: Update PDF Reports
- [ ] Modify `src/reports/pdf_generator.py`
- [ ] Add "Malaria Screening" section
- [ ] Include annotated image with detections
- [ ] Show stage-specific counts table
- [ ] Add parasite density calculation
- [ ] Include clinical interpretation text
- [ ] Add disclaimer: "Screening tool, not diagnostic"

**Validation**: Generated PDF contains malaria section with correct data

**Dependencies**: Tasks 3.4, 4.2 (requires results + visualization)

---

### Task 4.4: Update CLI Output
- [ ] Modify CLI output in `scripts/analyze_blood_smear.py`
- [ ] Add malaria section to terminal output
- [ ] Format detection results as table
- [ ] Show parasite coordinates (optional verbose mode)
- [ ] Add color-coded status (GREEN=negative, RED=positive)
- [ ] Include interpretation text

**Validation**: CLI output is readable and informative

**Dependencies**: Task 3.3 (requires integrated pipeline)

---

## Phase 5: Testing & Validation (Day 8-9)

### Task 5.1: Unit Tests
- [ ] Write `tests/test_malaria_detector.py`
  - Test model loading
  - Test detection on known positive
  - Test detection on known negative
  - Test bbox format validation
  - Test confidence thresholding
- [ ] Write `tests/test_dataset_preparation.py`
  - Test COCO to YOLO conversion
  - Test train/val/test splitting
  - Test data.yaml generation
- [ ] Ensure all tests pass with >90% coverage

**Validation**: All unit tests pass, coverage >90% on new code

**Dependencies**: Tasks 3.1-3.4 (requires implemented code)

---

### Task 5.2: Integration Tests
- [ ] Write `tests/test_integration.py`
  - Test full pipeline: image → results
  - Test API schema compatibility
  - Test performance benchmark (<25ms)
  - Test error handling (missing model, corrupt image)
- [ ] Run integration tests on multiple sample images
- [ ] Document edge cases and failure modes

**Validation**: All integration tests pass

**Dependencies**: Task 3.3 (requires integrated pipeline)

---

### Task 5.3: Clinical Validation
- [ ] Create `tests/test_clinical_validation.py`
- [ ] Test on 10+ hospital negative cases (should show 0 detections)
- [ ] Test on any available positive cases
- [ ] Calculate false positive rate on negatives
- [ ] Calculate false negative rate on positives
- [ ] Document results in `validation_report.md`

**Validation**: 
- FPR < 1% on confirmed negatives
- FNR < 10% on confirmed positives (if available)

**Dependencies**: Task 2.4 (requires validated model)

---

### Task 5.4: Performance Benchmarking
- [ ] Create `scripts/benchmark_performance.py`
- [ ] Measure inference time on 100 images
- [ ] Profile memory usage
- [ ] Test on CPU and GPU (if available)
- [ ] Compare with CNN baseline
- [ ] Document results in `performance_report.md`

**Validation**: 
- Inference time < 25ms per image (average)
- Memory usage < 500MB

**Dependencies**: Task 3.3 (requires integrated pipeline)

**Parallelizable**: Can run independently

---

## Phase 6: Documentation & Deployment (Day 10)

### Task 6.1: Update Documentation
- [ ] Update `README.md` with malaria detection info
- [ ] Create `docs/malaria_detection.md` user guide
- [ ] Document BBBC041 dataset usage and citation
- [ ] Add training instructions in `docs/training.md`
- [ ] Update API documentation with new schema
- [ ] Add troubleshooting section

**Validation**: Documentation is clear and complete

**Dependencies**: All previous tasks (requires complete system)

---

### Task 6.2: Create Deployment Guide
- [ ] Write `DEPLOYMENT.md`
- [ ] Document model download/setup
- [ ] List system requirements (GPU optional)
- [ ] Add feature flag configuration
- [ ] Include rollback procedures
- [ ] Document monitoring metrics

**Validation**: Guide is sufficient for deployment without additional help

**Dependencies**: Task 6.1 (builds on documentation)

---

### Task 6.3: Package Model Artifacts
- [ ] Create `models/detection/malaria_yolo11n/README.md`
- [ ] Document training hyperparameters
- [ ] Include evaluation metrics
- [ ] Add model provenance (BBBC041 dataset info)
- [ ] Zip trained weights with metadata
- [ ] Upload to storage/release (if applicable)

**Validation**: Model package is self-documenting and reproducible

**Dependencies**: Tasks 2.2, 2.3 (requires trained and evaluated model)

---

### Task 6.4: Final System Test
- [ ] Run full end-to-end test on fresh environment
- [ ] Test with hospital negative case (baseline validation)
- [ ] Test with various blood smear images
- [ ] Verify all features work: CLI, dashboard, PDF reports
- [ ] Check error handling and edge cases
- [ ] Sign-off on system readiness

**Validation**: System works correctly on all test cases

**Dependencies**: All previous tasks (final integration check)

---

## Success Criteria Checklist

### Functional Requirements
- [ ] YOLO detector successfully replaces CNN classifier
- [ ] BBBC041 dataset properly integrated and formatted
- [ ] Detection accuracy meets targets (mAP@0.5 > 0.80)
- [ ] Hospital negative case shows 0 detections (not 16.82%)
- [ ] Dashboard displays malaria results with visualizations
- [ ] PDF reports include malaria screening section
- [ ] CLI output shows detection results

### Performance Requirements
- [ ] Inference time < 25ms per image
- [ ] False positive rate < 1% on confirmed negatives
- [ ] Precision > 0.85 (minimize false positives)
- [ ] Recall > 0.75 (catch most parasites)

### Quality Requirements
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] Clinical validation successful
- [ ] Documentation complete and clear
- [ ] Code reviewed and approved

---

## Risk Mitigation Checklist

- [ ] Backup CNN classifier code before removal (git branch)
- [ ] Feature flag implemented for gradual rollout
- [ ] Rollback procedure documented and tested
- [ ] Performance metrics logged for monitoring
- [ ] User feedback mechanism in place

---

## Estimated Timeline

| Phase | Duration | Effort |
|-------|----------|--------|
| Phase 1: Dataset Preparation | 1-2 days | 8-12 hours |
| Phase 2: Model Training | 2-3 days | 12-16 hours (includes GPU time) |
| Phase 3: Code Integration | 2-3 days | 10-14 hours |
| Phase 4: UI Updates | 1 day | 6-8 hours |
| Phase 5: Testing & Validation | 1-2 days | 8-12 hours |
| Phase 6: Documentation & Deployment | 1 day | 4-6 hours |
| **Total** | **8-12 days** | **48-68 hours** |

**Note**: Training can run in background (Phase 2.2), and some tasks are parallelizable (marked above). With parallel work, wall-clock time can be reduced to 6-8 days.
