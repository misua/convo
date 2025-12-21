# Design: Extend RBC Classification

## Problem Statement

The current RBC shape classifier identifies 6 morphological categories for thalassemia screening. Users want to detect additional blood disorders (malaria, sickle cell disease) without adding model complexity or latency.

## Current Architecture

```python
# shape_classifier.py (existing)
class RBCShape(Enum):
    NORMAL = "normal"
    MICROCYTE = "microcyte"
    TARGET = "target"       # Thalassemia indicator
    TEARDROP = "teardrop"   # Thalassemia indicator
    SPHEROCYTE = "spherocyte"
    IRREGULAR = "irregular"

class RBCShapeClassifier:
    def __init__(self, use_deep_learning=False, model_path=None):
        # Rule-based metrics (always available)
        self.metrics_calculator = ShapeMetricsCalculator()
        
        # Optional CNN classifier (ResNet34 - same as WBC classifier)
        if use_deep_learning and model_path:
            self.cnn_model = models.resnet34(pretrained=False)
            self.cnn_model.fc = nn.Linear(512, len(RBCShape))  # Currently 6
    
    def classify(self, metrics, crop=None):
        # Rule-based classification using shape metrics
        if metrics.has_bullseye:
            return RBCShape.TARGET, 0.85
        if metrics.circularity < 0.70:
            return RBCShape.TEARDROP, 0.75
        # ... etc
```

**Key Insight:** The classifier already has a CNN placeholder (ResNet34, same as WBC classifier) that's not yet trained. We can train this CNN on extended classes.

## Design Decision: Extend Existing Classifier vs New Model

### Option A: Extend Existing Classifier ✅ CHOSEN
**Pros:**
- No architectural changes
- Same inference path (already cropping RBCs)
- Single model handles all RBC analysis- Use proven ResNet34 (same as WBC classifier - 97.9% accuracy)- `num_classes` automatically adjusts when enum is extended

**Cons:**
- Need to retrain CNN from scratch with all classes
- Rule-based fallback won't detect malaria (only sickle)

### Option B: Separate Malaria Detector
**Pros:**
- Can optimize malaria model independently
- No need to retrain shape classifier

**Cons:**
- Doubles inference time (2 models per RBC)
- More complex pipeline
- Harder to maintain

**Decision:** Option A. The ResNet34 CNN (same as WBC classifier) can learn to identify both RBC shape abnormalities AND internal parasites from the same crop.

## Extended RBCShape Enum

```python
class RBCShape(Enum):
    # Existing (thalassemia-related)
    NORMAL = "normal"
    MICROCYTE = "microcyte"
    TARGET = "target"
    TEARDROP = "teardrop"
    SPHEROCYTE = "spherocyte"
    IRREGULAR = "irregular"
    
    # New: Malaria parasites
    RING = "ring"                    # Ring stage (most common)
    TROPHOZOITE = "trophozoite"      # Mature form
    SCHIZONT = "schizont"            # Late stage (optional)
    
    # New: Sickle cell disease
    SICKLE = "sickle"                # Crescent/banana shaped
```

**Visual Distinction:**
- **Shape abnormalities** (microcyte, target, teardrop, sickle): Geometric features
- **Malaria infection** (ring, trophozoite, schizont): Color features (purple dots/rings due to Giemsa stain)

The ResNet34 CNN can learn both types of features.

## Classification Logic Updates

### Rule-Based (Immediate, No Training)

```python
def classify(self, metrics, crop=None):
    # Sickle cell detection (rule-based)
    if metrics.circularity < 0.5 and metrics.elongation > 2.0:
        return RBCShape.SICKLE, 0.80
    
    # Existing rules...
    if metrics.has_bullseye:
        return RBCShape.TARGET, 0.85
    
    # CNN-based (if model is trained)
    if self.cnn_model and crop is not None:
        return self._cnn_classify(crop)
    
    # Fallback to rule-based
    # ... existing geometric rules
```

### CNN-Based (After Training)

```python
def _cnn_classify(self, crop):
    # Preprocess crop (resize to 224x224, normalize)
    input_tensor = self._preprocess(crop)
    
    # Inference
    with torch.no_grad():
        logits = self.cnn_model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        class_idx = probabilities.argmax().item()
        confidence = probabilities[0, class_idx].item()
    
    # Map to RBCShape
    shape = list(RBCShape)[class_idx]
    return shape, confidence
```

## Training Strategy

### Dataset Composition

**Goal:** Balanced dataset with ~1,000 images per class (10K total)

| Class | Source | Expected Count |
|-------|--------|----------------|
| Normal | BCCD + augmentation | 3,000 |
| Microcyte | Current + synthetic | 500 |
| Target | Current + synthetic | 500 |
| Teardrop | Current + synthetic | 300 |
| Spherocyte | Current + synthetic | 300 |
| Irregular | Current + BCCD | 500 |
| Ring | NIH Malaria Dataset | 2,000 |
| Trophozoite | NIH Malaria Dataset | 1,000 |
| Sickle | erythrocytesIDB | 1,500 |
| Schizont | NIH Malaria Dataset (optional) | 500 |

### Data Augmentation

```python
train_transforms = A.Compose([
    A.Rotate(limit=180),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Stain variation
    A.GaussNoise(var_limit=(10, 30), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Training Hyperparameters

- **Model:** ResNet34 (pretrained on ImageNet) - same as WBC classifier
- **Input size:** 224×224
- **Batch size:** 32
- **Learning rate:** 1e-4 (body), 1e-3 (head)
- **Loss:** CrossEntropyLoss with class weights (handle imbalance)
- **Optimizer:** AdamW
- **Epochs:** 30 with early stopping (patience=5)
- **Validation:** Stratified 10% holdout

## Risk Assessment Updates

### New Metrics

```python
@dataclass
class ShapePopulationStats:
    # Existing
    thalassemia_indicators: int
    thalassemia_indicator_pct: float
    abnormality_index: float
    
    # New
    malaria_infected: int           # Ring + Trophozoite + Schizont
    malaria_infected_pct: float
    sickle_cells: int
    sickle_cell_pct: float
    parasite_stages: Dict[str, int] # {"ring": 5, "trophozoite": 2}
```

### Risk Levels

```python
def assess_risk(stats):
    risks = {}
    
    # Malaria (URGENT if detected)
    if stats.malaria_infected_pct > 0.5:
        risks['malaria'] = {
            'level': 'urgent',
            'score': min(1.0, stats.malaria_infected_pct / 10),
            'interpretation': f'{stats.malaria_infected} infected RBCs detected. Immediate lab confirmation required.'
        }
    
    # Sickle cell (REFERRAL if >5%)
    if stats.sickle_cell_pct > 5:
        risks['sickle_cell'] = {
            'level': 'referral',
            'score': min(1.0, stats.sickle_cell_pct / 40),
            'interpretation': f'{stats.sickle_cell_pct:.1f}% sickle cells. Consider sickle cell disease workup.'
        }
    
    # Thalassemia (existing logic)
    if stats.thalassemia_indicator_pct > 30:
        risks['thalassemia'] = {
            'level': 'monitor',
            'score': stats.abnormality_index,
            'interpretation': 'Significant morphological changes consistent with thalassemia.'
        }
    
    return risks
```

## Integration Points

### 1. Analyzer Script (analyze_blood_smear.py)

```python
# Enable CNN classifier when model is available
shape_classifier = RBCShapeClassifier(
    use_deep_learning=True,
    model_path='models/classification/rbc_extended_v1/best.pkl'  # fastai export
)

# Results now include malaria and sickle findings
results = {
    'shape_analysis': {
        'shape_distribution': {
            'normal': 45.0,
            'microcyte': 15.0,
            'target': 12.0,
            'ring': 8.0,      # Malaria
            'sickle': 10.0,   # Sickle cell
            # ...
        },
        'malaria_infected_pct': 8.0,
        'sickle_cell_pct': 10.0,
    }
}
```

### 2. Dashboard (app.py)

- Update shape distribution chart to show 9 classes (was 6)
- Add alert banner if malaria detected: "⚠️ URGENT: Malaria parasites detected"
- Add metric card: "Sickle Cells: 10.5%"

### 3. PDF Report (pdf_generator.py)

- Extend shape analysis table to 9 rows
- Add "Malaria Status" section with parasite stage breakdown
- Add urgent banner if infected cells found

## Performance Considerations

**Latency Impact:** ✅ None
- Same number of RBC crops processed
- CNN inference: ~5ms per crop (batched)
- Total inference time: unchanged (<1.5s for 200 RBCs)

**Model Size:** ✅ Acceptable
- ResNet34: ~85MB (similar to WBC classifier)
- 9-class output head: negligible

**Memory:** ✅ No change
- Still processing one crop at a time
- GPU memory: <2GB

## Validation Criteria

### Model Performance
- [ ] Overall accuracy >90% on balanced test set
- [ ] Malaria sensitivity >80% (critical - don't miss infections)
- [ ] Malaria specificity >90% (avoid false alarms)
- [ ] Sickle cell sensitivity >85%
- [ ] Thalassemia indicators maintain >85% accuracy (no regression)

### Integration Testing
- [ ] End-to-end inference completes in <2s
- [ ] PDF report generates without errors
- [ ] Dashboard displays all 9 classes correctly
- [ ] Rule-based fallback works when CNN unavailable

### Clinical Validation (Future)
- [ ] Expert hematologist review of 100 random predictions
- [ ] Inter-rater agreement >85%
- [ ] False positive rate acceptable for screening tool

## Future Extensions

1. **Parasite quantification:** Count parasites per 1000 RBCs (WHO standard)
2. **Species identification:** Distinguish P. falciparum vs P. vivax (requires more data)
3. **Mixed infections:** Detect cells with multiple parasites
4. **Confidence calibration:** Ensure predicted probabilities match true accuracy

## References

- NIH Malaria Dataset: https://lhncbc.nlm.nih.gov/publication/pub9932
- Sickle Cell Datasets: https://github.com/MahmoudYidi/erythrocytesIDB
- ResNet paper: https://arxiv.org/abs/1512.03385
- Fastai library: https://docs.fast.ai/
