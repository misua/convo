# Design: Multi-Disorder Blood Cell Detection System

## Context

Extending the blood cell analyzer from thalassemia-only to multi-disorder detection. The current architecture is well-suited for extension.

### Current State
```
Image → YOLOv11-seg → RBC (6 shapes) → Thalassemia risk
                    → WBC (4 types)  → Differential count
```

### Proposed State  
```
Image → YOLOv11-seg → RBC (7 shapes) → Thalassemia + Sickle risk
                    → WBC (6 types)  → Differential + Leukemia alert
                    → Malaria scan   → Infection risk
                    
       → Combined Multi-Disorder Report
```

## Technical Decisions

### Decision 1: Extend Existing Models vs New Models

**Choice:** Extend existing classification heads; add new malaria detector

**Rationale:**
- RBC shape classifier already handles 6 classes → easy to add 7th (sickle)
- WBC classifier is ResNet34 → proven architecture, just expand output
- Malaria requires different approach (intra-cellular parasites) → new model

```python
# Current shape_classifier.py
class RBCShape(Enum):
    NORMAL = "normal"
    MICROCYTE = "microcyte"
    TARGET = "target"
    TEARDROP = "teardrop"
    SPHEROCYTE = "spherocyte"
    IRREGULAR = "irregular"

# Extended
class RBCShape(Enum):
    NORMAL = "normal"
    MICROCYTE = "microcyte"
    TARGET = "target"
    TEARDROP = "teardrop"
    SPHEROCYTE = "spherocyte"
    SICKLE = "sickle"       # NEW
    IRREGULAR = "irregular"
```

### Decision 2: Sickle Cell Detection Thresholds

**Choice:** Shape-based detection using circularity and elongation

**Sickle Cell Signature:**
```python
def is_sickle_cell(metrics: ShapeMetrics) -> bool:
    """
    Sickle cells have distinctive crescent shape:
    - Very low circularity (< 0.5)
    - High elongation (> 2.0)
    - Often pointed ends (low solidity)
    """
    return (
        metrics.circularity < 0.50 and
        metrics.elongation > 2.0 and
        metrics.solidity < 0.85
    )
```

**Comparison with other abnormal shapes:**
| Shape | Circularity | Elongation | Solidity |
|-------|-------------|------------|----------|
| Normal | 0.85-1.0 | 1.0-1.2 | >0.95 |
| Target | 0.80-0.95 | 1.0-1.3 | >0.90 |
| Teardrop | 0.60-0.80 | 1.5-2.0 | 0.80-0.90 |
| **Sickle** | **0.30-0.50** | **2.0-4.0** | **0.60-0.85** |
| Spherocyte | 0.90-1.0 | 1.0-1.1 | >0.95 |

### Decision 3: WBC Classifier Extension

**Choice:** Add Basophil and Blast classes to existing ResNet34

**Current:** 4 classes (Eosinophil, Lymphocyte, Monocyte, Neutrophil)
**Extended:** 6 classes (+Basophil, +Blast)

```python
WBC_CLASSES = [
    "eosinophil",
    "lymphocyte", 
    "monocyte",
    "neutrophil",
    "basophil",    # NEW - rare but important
    "blast"        # NEW - leukemia indicator
]
```

**Blast Cell Characteristics:**
- Large nucleus-to-cytoplasm ratio
- Fine chromatin pattern
- Prominent nucleoli
- Indicates acute leukemia (urgent finding)

### Decision 4: Malaria Detection Architecture

**Choice:** Dedicated YOLOv11n trained on parasitized RBCs

**Why separate model:**
- Parasites are inside RBCs, not separate cells
- Different visual features (ring forms, chromatin dots)
- NIH dataset is pre-segmented (cell-level, not smear-level)

**Architecture:**
```
RBC crop → YOLOv11n-cls → Binary: Parasitized / Uninfected
                        → If parasitized: Ring / Trophozoite / Schizont / Gametocyte
```

**Simplified MVP:** Binary classification (infected/uninfected) first

### Decision 5: Multi-Disorder Risk Report

**Choice:** Unified report with priority flagging

```python
class DisorderRisk:
    URGENT = "urgent"      # Requires immediate attention
    ELEVATED = "elevated"  # Needs follow-up
    LOW = "low"           # Normal finding

DISORDER_PRIORITY = {
    "leukemia": DisorderRisk.URGENT,     # Blast cells detected
    "malaria": DisorderRisk.URGENT,       # Active infection
    "sickle_cell": DisorderRisk.ELEVATED, # Chronic condition
    "thalassemia": DisorderRisk.ELEVATED, # Chronic condition
}
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-DISORDER DETECTION SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌───────────────────┐                                 │
│  │   Image      │    │  YOLOv11-seg      │                                 │
│  │   Input      │───▶│  Cell Detection   │                                 │
│  │              │    │  + Segmentation   │                                 │
│  └──────────────┘    └────────┬──────────┘                                 │
│                               │                                             │
│              ┌────────────────┼────────────────┐                           │
│              ▼                ▼                ▼                           │
│      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                   │
│      │  RBC Crops   │ │  WBC Crops   │ │  All Cells   │                   │
│      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                   │
│             │                │                │                            │
│             ▼                ▼                ▼                            │
│      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                   │
│      │ Shape Class. │ │ WBC Class.   │ │ Malaria Det. │                   │
│      │ (7 classes)  │ │ (6 classes)  │ │ (binary/4)   │                   │
│      │              │ │              │ │              │                   │
│      │ - Normal     │ │ - Eosinophil │ │ - Uninfected │                   │
│      │ - Microcyte  │ │ - Lymphocyte │ │ - Ring       │                   │
│      │ - Target     │ │ - Monocyte   │ │ - Trophozoite│                   │
│      │ - Teardrop   │ │ - Neutrophil │ │ - Schizont   │                   │
│      │ - Spherocyte │ │ - Basophil   │ │ - Gametocyte │                   │
│      │ - SICKLE ⭐  │ │ - BLAST ⭐   │ │              │                   │
│      │ - Irregular  │ │              │ │              │                   │
│      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                   │
│             │                │                │                            │
│             └────────────────┴────────────────┘                            │
│                              │                                              │
│                              ▼                                              │
│             ┌─────────────────────────────────────┐                        │
│             │      Multi-Disorder Risk Engine     │                        │
│             │                                     │                        │
│             │  Thalassemia: (target + teardrop)   │                        │
│             │  Sickle Cell: (sickle cells %)      │                        │
│             │  Leukemia:    (blast cells present) │                        │
│             │  Malaria:     (parasites detected)  │                        │
│             └─────────────────────────────────────┘                        │
│                              │                                              │
│                              ▼                                              │
│             ┌─────────────────────────────────────┐                        │
│             │      Unified Risk Report            │                        │
│             │                                     │                        │
│             │  🔴 URGENT: Malaria (parasites)     │                        │
│             │  🔴 URGENT: Leukemia (3 blasts)     │                        │
│             │  🟡 ELEVATED: Thalassemia (15%)     │                        │
│             │  🟢 LOW: Sickle Cell (0%)           │                        │
│             └─────────────────────────────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Requirements Deep Dive

### Why WBC Needs More Data

```
Typical Blood Smear Statistics:
┌─────────────────────────────────────────────────────────────┐
│ Cell Type    │ Count per Smear │ % of Total │ Images Needed │
├─────────────────────────────────────────────────────────────┤
│ RBC          │ 4-6 million/μL  │ ~99.9%     │ 500-1000      │
│ Platelets    │ 150-400k/μL     │ ~0.08%     │ 500-1000      │
│ WBC (total)  │ 4-11k/μL        │ ~0.001%    │ 2000+         │
│  - Neutrophil│ 2-7k/μL         │ 60%        │ Available     │
│  - Lymphocyte│ 1-4k/μL         │ 30%        │ Available     │
│  - Monocyte  │ 0.2-1k/μL       │ 5%         │ Available     │
│  - Eosinophil│ 0.04-0.4k/μL    │ 3%         │ Available     │
│  - Basophil  │ 0.01-0.1k/μL    │ <1%        │ ~200 in repo  │
│  - BLAST     │ 0 (abnormal)    │ ~0%        │ ~69 in repo   │
└─────────────────────────────────────────────────────────────┘
```

**Current Dataset Inventory:**

| Source | Location | Blast Cells | Basophils | Usable? |
|--------|----------|-------------|-----------|---------|
| RV-PBS | `data/wbc_instance_seg/` | 69 | ~50 | ✅ |
| wbc_5class | `data/wbc_5class/` | 0 | 0 (5 class) | ⚠️ |
| Kaggle | `data/kaggle/` | 0 | 0 | ⚠️ |

**Gap Analysis:**
- Need ~2000 blast cells → have 69 → **gap: ~1931 images**
- Need ~500 basophils → have ~50 → **gap: ~450 images**
- Need ~2000 sickle cells → have 0 → **gap: ~2000 images**
- Need ~5000 malaria → have 0 → **gap: ~5000 images**

## File Changes

### Modified Files

```python
# src/models/shape_classifier.py
# - Add SICKLE to RBCShape enum
# - Update classify_shape() thresholds
# - Add is_sickle_cell() helper

# scripts/train_wbc_classifier.py  
# - Update WBC_CLASSES to 6
# - Add class weighting for imbalanced data
# - Add blast cell augmentation

# app.py
# - Add "Multi-Disorder Analysis" tab
# - Update results display for all disorders
# - Add urgent finding alerts

# src/reports/pdf_generator.py
# - Add disorder-specific sections
# - Priority-based ordering (urgent first)
```

### New Files

```python
# src/models/malaria_detector.py
class MalariaDetector:
    """Detect malaria parasites in RBC crops."""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.classes = ["uninfected", "parasitized"]
    
    def detect(self, rbc_crop: np.ndarray) -> dict:
        """Classify RBC as infected/uninfected."""
        pass

# src/inference/multi_disorder.py
class MultiDisorderAnalyzer:
    """Unified analysis for all blood disorders."""
    
    def analyze(self, image_path: str) -> MultiDisorderReport:
        """Run all detection pipelines."""
        pass
    
    def generate_report(self, results: dict) -> dict:
        """Create priority-ordered report."""
        pass

# scripts/train_malaria_detector.py
# scripts/download_datasets.py  # Automated dataset download
```

## Training Strategy

### Phase 1: Sickle Cell (2 weeks)
1. Download erythrocytesIDB dataset
2. Augment to 2000+ images
3. Add SICKLE class to shape classifier
4. Re-train with class weights

### Phase 2: Leukemia (3 weeks)
1. Combine RV-PBS blast cells + ALL-IDB
2. Balance dataset with augmentation
3. Fine-tune WBC classifier (6 classes)
4. Add blast alert system

### Phase 3: Malaria (2 weeks)
1. Download NIH Malaria Dataset (27k images)
2. Train binary classifier first
3. Optionally add stage classification
4. Integrate with pipeline

## Performance Targets

| Model | Inference Time | Target Accuracy |
|-------|---------------|-----------------|
| RBC Shape (7 class) | 15ms/cell | >85% |
| WBC Type (6 class) | 20ms/cell | >90% |
| Malaria Binary | 10ms/cell | >95% |
| **Total Pipeline** | **<2 seconds** | - |

## Validation Strategy

### Critical Validation (Before Deployment)
1. **Blast cell sensitivity >80%** - Missing leukemia is dangerous
2. **Malaria sensitivity >90%** - Active infection must be caught
3. **Sickle cell specificity >90%** - Avoid false positives

### Test Datasets
- Reserve 20% of each dataset for testing
- External validation with different microscope images
- Clinician review of flagged cases
