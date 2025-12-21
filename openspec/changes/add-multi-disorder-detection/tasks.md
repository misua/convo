# Tasks: Multi-Disorder Blood Cell Detection

change-id: add-multi-disorder-detection

## Phase 1: Dataset Acquisition & Preparation

### 1.1 Sickle Cell Dataset
- [ ] Download erythrocytesIDB from https://github.com/MahmoudYidi/erythrocytesIDB
- [ ] Download SCD-Dataset from Mendeley
- [ ] Combine and deduplicate images
- [ ] Convert to standard format (YOLO labels if needed)
- [ ] Apply augmentation to reach 2000+ images
- [ ] Create train/val/test splits (70/15/15)

### 1.2 Malaria Dataset
- [ ] Download NIH Malaria Dataset (27,558 images)
- [ ] Verify parasitized/uninfected balance
- [ ] Resize to standard input size (224x224)
- [ ] Create train/val/test splits
- [ ] Document class distribution

### 1.3 Leukemia/Blast Cell Dataset
- [ ] Audit existing RV-PBS blast cells (`data/wbc_instance_seg/BLAST CELLS/` - 69 images)
- [ ] Download ALL-IDB dataset
- [ ] Download ISBI 2019 C-NMC dataset
- [ ] Combine with existing WBC data
- [ ] Balance classes with augmentation (target: 500+ per class)
- [ ] Verify basophil count in existing datasets

## Phase 2: Extend RBC Shape Classifier (Sickle Cell)

### 2.1 Code Changes
- [ ] Add `SICKLE = "sickle"` to `RBCShape` enum in `shape_classifier.py`
- [ ] Implement `is_sickle_cell()` detection function
  ```python
  def is_sickle_cell(metrics: ShapeMetrics) -> bool:
      return (
          metrics.circularity < 0.50 and
          metrics.elongation > 2.0 and
          metrics.solidity < 0.85
      )
  ```
- [ ] Update `classify_shape()` to include sickle detection
- [ ] Add sickle cell to thalassemia indicator check (both are hemoglobinopathies)
- [ ] Update `ShapePopulationStats` with sickle cell count

### 2.2 Model Training
- [ ] Create training script `scripts/train_shape_classifier.py`
- [ ] Add class weighting for imbalanced sickle cells
- [ ] Train EfficientNet-V2-S on 7 classes
- [ ] Evaluate: target sensitivity >85% for sickle cells
- [ ] Save model to `models/classification/shape_v2/`

### 2.3 Testing
- [ ] Unit tests for sickle cell thresholds
- [ ] Integration test with sample sickle cell images
- [ ] Verify no regression on thalassemia detection

## Phase 3: Extend WBC Classifier (Leukemia)

### 3.1 Code Changes  
- [ ] Update `train_wbc_classifier.py` for 6 classes
- [ ] Add class names: `["eosinophil", "lymphocyte", "monocyte", "neutrophil", "basophil", "blast"]`
- [ ] Implement weighted cross-entropy loss for class imbalance
- [ ] Add blast cell alert flag in results

### 3.2 Model Training
- [ ] Prepare combined dataset (RV-PBS + ALL-IDB + existing)
- [ ] Apply heavy augmentation to blast/basophil classes
- [ ] Train ResNet34 with class weights
- [ ] Evaluate confusion matrix (focus on blast recall)
- [ ] Target: >80% blast cell sensitivity, >95% specificity
- [ ] Save model to `models/classification/wbc_v2/`

### 3.3 Clinical Integration
- [ ] Add blast detection to `BloodSmearAnalyzer.analyze()`
- [ ] Implement urgent alert when blast cells detected
- [ ] Add recommendation: "URGENT: Blast cells detected - hematology consult recommended"

## Phase 4: Add Malaria Detection

### 4.1 New Module
- [ ] Create `src/models/malaria_detector.py`
  ```python
  class MalariaDetector:
      def __init__(self, model_path: str):
          self.model = load_learner(model_path)
      
      def detect(self, rbc_crop: np.ndarray) -> dict:
          """Return {infected: bool, confidence: float, stage: str}"""
  ```
- [ ] Implement binary classification (infected/uninfected)
- [ ] Optional: Add parasite stage classification

### 4.2 Model Training
- [ ] Create `scripts/train_malaria_detector.py`
- [ ] Train on NIH dataset
- [ ] Use ResNet18 (smaller model for binary task)
- [ ] Target: >95% sensitivity, >90% specificity
- [ ] Save to `models/classification/malaria_v1/`

### 4.3 Pipeline Integration
- [ ] Add malaria scan to `BloodSmearAnalyzer`
- [ ] Run malaria detection on all RBC crops
- [ ] Add parasitemia calculation (% infected RBCs)
- [ ] Add urgent alert for any parasites detected

## Phase 5: Multi-Disorder Risk Engine

### 5.1 New Module
- [ ] Create `src/inference/multi_disorder.py`
- [ ] Implement `MultiDisorderReport` dataclass
  ```python
  @dataclass
  class MultiDisorderReport:
      thalassemia_risk: RiskLevel
      sickle_cell_risk: RiskLevel
      leukemia_risk: RiskLevel
      malaria_risk: RiskLevel
      urgent_findings: List[str]
      recommendations: List[str]
  ```
- [ ] Implement priority ordering (urgent → elevated → low)
- [ ] Add combined risk score calculation

### 5.2 Risk Calculation
- [ ] Thalassemia: existing logic (target + teardrop %)
- [ ] Sickle cell: sickle cell % with severity thresholds
- [ ] Leukemia: any blast cell = urgent
- [ ] Malaria: any parasite = urgent

## Phase 6: Dashboard Updates

### 6.1 UI Changes
- [ ] Add "Multi-Disorder Analysis" tab in `app.py`
- [ ] Create disorder-specific result panels
- [ ] Add traffic light indicators (🔴🟡🟢) for each disorder
- [ ] Add urgent finding banner at top
- [ ] Update cell gallery to show disorder-relevant cells

### 6.2 Visualization
- [ ] Color-code cells by detected abnormality
- [ ] Add legend for disorder colors
- [ ] Show parasites highlighted in malaria-positive cells
- [ ] Add shape distribution histogram

## Phase 7: PDF Report Updates

### 7.1 Report Structure
- [ ] Update `pdf_generator.py` with multi-disorder sections
- [ ] Order by priority (urgent first)
- [ ] Add executive summary with all findings
- [ ] Include cell galleries for each abnormality type

### 7.2 Clinical Recommendations
- [ ] Add disorder-specific follow-up recommendations
- [ ] Include reference ranges for each metric
- [ ] Add disclaimer about screening vs diagnosis

## Phase 8: Configuration & Deployment

### 8.1 Config Updates
- [ ] Update `configs/config.yaml` with new model paths
  ```yaml
  models:
    shape_classifier: "models/classification/shape_v2/best.pkl"
    wbc_classifier: "models/classification/wbc_v2/best.pkl"
    malaria_detector: "models/classification/malaria_v1/best.pkl"
  
  disorders:
    enabled:
      - thalassemia
      - sickle_cell
      - leukemia
      - malaria
  ```
- [ ] Add feature flags for each disorder
- [ ] Configure alert thresholds

### 8.2 Documentation
- [ ] Update README.md with multi-disorder capabilities
- [ ] Document dataset sources and licenses
- [ ] Add clinical validation notes
- [ ] Create user guide for interpreting results

## Phase 9: Testing & Validation

### 9.1 Unit Tests
- [ ] Test sickle cell detection thresholds
- [ ] Test blast cell classification
- [ ] Test malaria detection
- [ ] Test multi-disorder report generation

### 9.2 Integration Tests
- [ ] End-to-end test with mixed abnormalities
- [ ] Test urgent alert system
- [ ] Test PDF report generation
- [ ] Test dashboard updates

### 9.3 Clinical Validation
- [ ] Review with hematology expert
- [ ] Validate on external dataset
- [ ] Document performance metrics
- [ ] Create validation report

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Dataset Prep | 1 week | None |
| 2. Sickle Cell | 2 weeks | Phase 1 |
| 3. Leukemia | 3 weeks | Phase 1 |
| 4. Malaria | 2 weeks | Phase 1 |
| 5. Risk Engine | 1 week | Phases 2-4 |
| 6. Dashboard | 1 week | Phase 5 |
| 7. PDF Report | 1 week | Phase 5 |
| 8. Config | 2 days | Phases 6-7 |
| 9. Testing | 1 week | All |

**Total: ~10-12 weeks**

---

## Validation Checklist

- [ ] All datasets downloaded and prepared
- [ ] Sickle cell sensitivity > 85%
- [ ] Blast cell sensitivity > 80%
- [ ] Malaria sensitivity > 95%
- [ ] Combined inference < 2 seconds
- [ ] Dashboard displays all disorders
- [ ] PDF report includes all sections
- [ ] `openspec validate add-multi-disorder-detection --strict` passes
