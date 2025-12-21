# Change: Add Multi-Disorder Blood Cell Detection

## Why

The current blood cell analyzer only screens for thalassemia through RBC shape analysis. Expanding to detect multiple blood disorders (sickle cell, malaria, leukemia) would significantly increase clinical utility. The existing architecture (ResNet + YOLOv11-seg) is well-suited for extension with additional classification heads.

**Key Insight:** WBC-based disorders (leukemia) require significantly more training images than RBC disorders because:
- WBCs are ~1000x rarer than RBCs in blood smears
- Need 2,000-5,000+ labeled WBC images for reliable detection
- RBC disorders can work with ~500-1000 images due to abundance

## What Changes

### 1. Extend RBC Shape Classifier for Sickle Cell
- Add **sickle cell** class to existing 6-class RBC shape model
- New RBC classes: Normal, Microcyte, Target, Teardrop, Spherocyte, **Sickle**, Irregular
- Sickle cells have distinctive crescent/elongated shape (circularity < 0.5, elongation > 2.0)

### 2. Add Malaria Parasite Detection
- Train dedicated YOLOv11 model to detect intra-cellular parasites
- Classes: Ring form, Trophozoite, Schizont, Gametocyte
- Requires separate dataset of infected blood smears

### 3. Add Leukemia Blast Cell Detection  
- Extend WBC classifier to detect abnormal blast cells
- Use existing RV-PBS dataset (69 blast cell images as starting point)
- New WBC classes: Eosinophil, Lymphocyte, Monocyte, Neutrophil, Basophil, **Blast**
- Blast cells indicate acute leukemia (require urgent follow-up)

### 4. Multi-Disorder Risk Report
- Unified dashboard showing risk indicators for all disorders
- Priority flagging (leukemia/malaria = urgent; thalassemia/sickle = referral)
- Differential diagnosis suggestions based on combined findings

## Architecture Extension

```
Current Pipeline:
  Image → YOLOv11-seg → Cell crops → WBC Classifier (4 classes)
                                   → RBC Shape (6 classes)
                                   → Risk: Thalassemia only

Extended Pipeline:
  Image → YOLOv11-seg → Cell crops ─┬─→ WBC Classifier (6 classes) → Leukemia risk
                                    ├─→ RBC Shape (7 classes) ────→ Thalassemia + Sickle Cell
                                    └─→ Malaria Detector ─────────→ Malaria risk
                                   
                       Combined → Multi-Disorder Risk Report
```

## Dataset Requirements

### Minimum Viable Datasets

| Disorder | Cell Type | Min Images | Public Sources |
|----------|-----------|------------|----------------|
| **Sickle Cell** | RBC | 2,000 | NIH Sickle Cell Dataset, erythrocytesIDB |
| **Malaria** | RBC | 5,000 | NIH Malaria Dataset (27,558 images) |
| **Leukemia** | WBC | 2,000 | RV-PBS (69 blast), ALL-IDB, ISBI 2019 |
| **Basophil** | WBC | 500 | RV-PBS dataset, existing wbc_5class |

### Why More Images for WBC?
- WBC:RBC ratio is ~1:600-1000 in normal blood
- Single smear image may contain 200+ RBCs but only 0-3 WBCs
- Need diverse blast cell morphology (lymphoblasts vs myeloblasts)
- Class imbalance is severe - basophils are <1% of WBCs

## Impact

| Area | Changes |
|------|---------|
| **Models** | Extended shape_classifier.py, new malaria_detector.py |
| **Classes** | RBC: 6→7, WBC: 4→6 |
| **Dashboard** | New "Multi-Disorder Analysis" tab |
| **Reports** | PDF includes all disorder risk sections |
| **Latency** | +100-150ms for malaria scan |

## Implementation Phases

### Phase 1: Sickle Cell (Easiest - extends existing RBC model)
- Add sickle class to RBCShape enum
- Update shape classifier thresholds
- Acquire/annotate ~500 sickle cell images
- Re-train with augmentation

### Phase 2: Leukemia (Uses existing WBC pipeline)  
- Add Basophil, Blast classes to WBC classifier
- Curate dataset from RV-PBS + public sources
- Train multi-class classifier (6 classes)
- Add blast cell alert system

### Phase 3: Malaria (New detection model)
- Download NIH Malaria Dataset
- Train YOLOv11 for parasite detection
- Add parasite overlay visualization
- Integrate with risk report

## Success Criteria

1. Sickle cell detection sensitivity > 85%
2. Blast cell detection sensitivity > 80% (critical for leukemia)
3. Malaria parasite detection mAP > 0.75
4. False positive rate < 10% for urgent findings
5. Combined inference < 2s on RTX 3060

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Insufficient sickle cell data | Model underperforms | Partner with sickle cell centers; heavy augmentation |
| Blast cell confusion with reactive lymphocytes | False positives | Include reactive cells in training; confidence thresholds |
| Malaria parasite size variation | Missed detections | Multi-scale detection; augmentation |
| Class imbalance (rare disorders) | Biased predictions | Weighted loss; oversampling minority classes |

## Out of Scope

- Automated treatment recommendations
- Malaria species identification (P. falciparum vs P. vivax)
- Leukemia subtype classification (ALL vs AML detailed)
- Beta-thalassemia vs alpha-thalassemia differentiation
- Quantitative parasite load estimation

## Public Dataset Sources

### Sickle Cell
- erythrocytesIDB: https://github.com/MahmoudYidi/erythrocytesIDB
- SCD-Dataset: https://data.mendeley.com/datasets/snkg9rcf3t

### Malaria
- NIH Malaria Dataset: https://lhncbc.nlm.nih.gov/publication/pub9932 (27,558 images)
- Plasmodium Dataset: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

### Leukemia  
- ALL-IDB: https://homes.di.unimi.it/scotti/all/
- ISBI 2019 C-NMC: https://wiki.cancerimagingarchive.net/
- RV-PBS (already in repo): 69 blast cell images

