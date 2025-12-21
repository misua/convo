# Tasks: Thalassemia Shape Diagnosis

Implementation checklist for adding RBC shape analysis and plain-language diagnosis.

## Phase 1: Model Infrastructure

### 1.1 Detection Upgrade
- [x] Update config.yaml to use YOLOv11-seg model
- [ ] Download/initialize `yolo11n-seg.pt` weights
- [x] Update `BloodSmearAnalyzer` to extract instance masks
- [x] Add mask-based shape metric calculations (circularity, elongation, solidity)
- [x] Verify existing sample image `1000026798.jpg` works with new detector

### 1.2 Shape Classification Model
- [x] Create `src/models/shape_classifier.py` module
- [x] Define 6-class RBC shape taxonomy in config
- [x] Implement rule-based shape classifier (MVP - CNN deferred)
- [ ] Add training script `scripts/train_shape_classifier.py`
- [ ] Create data augmentation pipeline for shape training

### 1.3 Shape Dataset Preparation
- [ ] Create annotation guidelines document for RBC shapes
- [ ] Set up annotation workspace (LabelImg or Roboflow)
- [ ] Annotate minimum 500 cells across shape classes (bootstrap set)
- [ ] Create train/val/test split for shape dataset
- [ ] Validate annotations with inter-rater agreement check

## Phase 2: Analysis Pipeline

### 2.1 Shape Analysis Integration
- [x] Add `analyze_rbc_shapes()` method to `BloodSmearAnalyzer`
- [x] Implement population statistics aggregator (% per shape class)
- [x] Calculate shape-based abnormality scores
- [x] Add shape findings to results JSON output

### 2.2 Enhanced Risk Scoring
- [x] Update risk scoring with shape weights
- [x] Implement target cell detection bonus (strong indicator)
- [x] Add teardrop cell detection bonus (strong indicator)
- [ ] Calculate confidence intervals for risk scores
- [ ] Validate risk scores against known thalassemia samples (if available)

### 2.3 Results Schema Update
- [x] Extend results JSON schema with shape analysis fields
- [x] Add `shape_distribution` object to results
- [x] Add `flagged_cells` array with shape classifications
- [x] Include shape metrics per cell in detailed output
- [ ] Update `1000026798_results.json` format as reference

## Phase 3: User Interface

### 3.1 Plain-Language Diagnosis Panel
- [x] Create `DiagnosisExplainer` class in `src/inference/explainer.py`
- [x] Define plain-language templates for all findings
- [x] Implement traffic light risk visualization (green/yellow/red)
- [x] Write "What This Means" explanations for each abnormality
- [x] Create "Recommended Next Steps" generator
- [x] Add medical disclaimer component

### 3.2 Gradio Dashboard Updates
- [x] Add "Patient-Friendly Report" tab to `app.py`
- [ ] Create shape gallery component with cell thumbnails
- [ ] Add shape distribution bar chart visualization
- [ ] Implement cell highlighting on click (show shape class)
- [ ] Add "Download Report" button (PDF export)
- [ ] Style traffic light indicators with Gradio CSS

### 3.3 Technical Report Enhancements
- [x] Add shape analysis section to technical report
- [x] Include shape metrics table (circularity, elongation per class)
- [x] Add confidence scores for shape classifications
- [ ] Create flagged cell gallery in technical view
- [ ] Add export to CSV with shape data

## Phase 4: Testing & Validation

### 4.1 Unit Tests
- [ ] Test shape metric calculations with synthetic masks
- [ ] Test risk score calculation edge cases
- [ ] Test plain-language generator outputs
- [ ] Test JSON schema validation

### 4.2 Integration Tests
- [ ] End-to-end test with `1000026798.jpg` sample
- [ ] Test batch processing of multiple images
- [ ] Test error handling for corrupted/invalid images
- [ ] Test memory usage on RTX 3060 (both models loaded)

### 4.3 User Acceptance
- [ ] Conduct usability test with 3+ non-medical users
- [ ] Verify plain-language report clarity (target: 4/5 rating)
- [ ] Validate clinical accuracy with pathologist review (if available)
- [ ] Document feedback and iterate on explanations

## Phase 5: Documentation & Deployment

### 5.1 Documentation
- [ ] Update README.md with shape analysis features
- [ ] Document new configuration options in config.yaml
- [ ] Create user guide for plain-language report interpretation
- [ ] Add troubleshooting section for shape detection issues

### 5.2 Model Artifacts
- [ ] Save trained YOLOv11-seg weights to `models/detection/`
- [ ] Save trained shape classifier to `models/classification/`
- [ ] Create model card documenting training data and performance
- [ ] Add model version tracking to config

---

## Dependencies

```
Phase 1.1 → Phase 2.1 (need detection before analysis)
Phase 1.2 + 1.3 → Phase 2.1 (need classifier and data)
Phase 2.1 + 2.2 → Phase 3.1 (need analysis before UI)
Phase 3.* → Phase 4.* (need UI before testing)
Phase 4.* → Phase 5.* (need tests before docs)
```

## Parallelizable Work

- **Phase 1.2** (classifier code) can run parallel with **Phase 1.3** (annotations)
- **Phase 3.1** (explainer) can run parallel with **Phase 3.2** (Gradio)
- **Phase 4.1** (unit tests) can run parallel with **Phase 4.2** (integration)
