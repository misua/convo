# Design: Blood Cell Analyzer Architecture

## Context
Building an AI-powered blood cell analysis system for thalassemia screening. Target users are clinicians who need rapid, interpretable results from peripheral blood smear images. System runs locally on RTX 3060 GPU.

### Stakeholders
- Clinicians (end users) - need clear, actionable outputs
- Lab technicians - capture blood smear images
- ML engineers - maintain and improve models

### Constraints
- Hardware: RTX 3060 (12GB VRAM, ~12 TFLOPS FP32)
- Privacy: Medical images must stay local (no cloud inference)
- Latency: Results needed in <5 seconds per image
- Interpretability: Must show which cells triggered alerts

## Goals / Non-Goals

### Goals
- Train accurate WBC subtype classifier on Kaggle dataset using local GPU
- Provide real-time inference with visual explanations
- Analyze user's existing blood smear images
- Create intuitive review dashboard
- Export results in standard formats (PDF, CSV)

### Non-Goals
- Cloud deployment (local-first MVP)
- Mobile app (desktop/web only)
- DICOM integration (raw image input for now)
- Complex microscope calibration (use defaults)

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     BLOOD CELL ANALYZER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Image      │    │   Detection  │    │  Morphology  │      │
│  │   Loader     │───▶│   Model      │───▶│  Analyzer    │      │
│  │              │    │  (YOLOv8)    │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │                   ▼                   ▼               │
│         │            ┌──────────────┐    ┌──────────────┐      │
│         │            │    Cell      │    │    RBC       │      │
│         │            │   Counter    │    │   Indices    │      │
│         │            └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │                   └─────────┬─────────┘               │
│         │                             ▼                         │
│         │                   ┌──────────────┐                   │
│         │                   │   Results    │                   │
│         └──────────────────▶│  Aggregator  │                   │
│                             └──────────────┘                   │
│                                    │                            │
│                                    ▼                            │
│                          ┌──────────────────┐                  │
│                          │    Dashboard     │                  │
│                          │  (Streamlit/     │                  │
│                          │   Gradio)        │                  │
│                          └──────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Technical Decisions

### Decision 1: Detection Framework - YOLOv8
**Choice:** Ultralytics YOLOv8 for cell detection

**Rationale:**
- Native PyTorch, easy to train on custom data
- Excellent inference speed on consumer GPUs
- Built-in support for BCCD-style annotations
- Single-stage detector = simpler pipeline

**Alternatives Considered:**
| Option | Pros | Cons |
|--------|------|------|
| Faster R-CNN | Higher accuracy on small objects | Slower inference, complex setup |
| YOLOv5 | Stable, well-documented | YOLOv8 has better performance |
| DETR | State-of-art architecture | Needs more training data/compute |

### Decision 2: Framework - PyTorch + Fast.ai
**Choice:** PyTorch with Fast.ai for classification tasks

**Rationale:**
- Fast.ai provides excellent data augmentation out-of-box
- Transfer learning from ImageNet works well for medical images
- PyTorch native = easy to customize
- Large community, good debugging support

### Decision 3: Dashboard - Gradio
**Choice:** Gradio for clinician interface

**Rationale:**
- Rapid prototyping (MVP in <100 lines)
- Built-in image annotation components
- Shareable via local network
- Works with Jupyter notebooks during development

**Alternatives:**
- Streamlit: Good but less image-focused
- Custom Flask: More work, less rapid iteration
- Panel: More complex for MVP needs

### Decision 4: Morphology Analysis Approach
**Choice:** Two-stage approach

**Stage 1 (MVP):** Rule-based metrics from detections
- Cell diameter measurement from bounding boxes
- Color analysis (mean intensity in HSV space)
- Shape regularity (circularity index)

**Stage 2 (Future):** Trained classifier for specific morphologies
- Microcytosis detection model
- Hypochromia detection model
- Target cell identification

### Decision 5: RBC Indices Calculation
**Choice:** Automated measurement from microscopy calibration

**Method:**
1. User provides microscope calibration (μm per pixel)
2. System measures cell diameters from detections
3. Calculate MCV proxy: mean cell area
4. Calculate RDW: coefficient of variation of diameters
5. Color analysis for MCHC proxy

**Note:** These are screening proxies, not replacements for CBC.

## Data Pipeline

```
Raw Image (1920x1080 or higher)
    │
    ▼
Preprocessing
    ├── Resize to model input (640x640 for YOLOv8)
    ├── Normalize (ImageNet stats or custom)
    └── Optional: White balance correction
    │
    ▼
Cell Detection (YOLOv8)
    ├── Output: Bounding boxes + confidence
    └── Classes: RBC, WBC, Platelet
    │
    ▼
Per-Cell Analysis
    ├── Crop each detected cell
    ├── Measure diameter (pixels → μm)
    ├── Analyze color (pale center detection)
    └── Shape metrics (circularity)
    │
    ▼
Aggregation
    ├── Count cells by type
    ├── Calculate size distribution
    ├── Flag abnormal percentages
    └── Generate confidence scores
    │
    ▼
Report Generation
    ├── Visual overlay on original image
    ├── Statistical summary
    └── Clinical alerts
```

## Project Structure

```
blood-cell-analyzer/
├── data/
│   ├── raw/                # Original dataset (BCCD/Kaggle)
│   ├── processed/          # Training-ready format
│   └── samples/            # Demo images
├── models/
│   ├── detection/          # YOLOv8 weights
│   └── classification/     # WBC subtype classifier
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py       # Dataset loading
│   │   └── augmentation.py # Custom augmentations
│   ├── models/
│   │   ├── detector.py     # YOLOv8 wrapper
│   │   └── analyzer.py     # Morphology analysis
│   ├── inference/
│   │   ├── pipeline.py     # Full inference pipeline
│   │   └── metrics.py      # RBC indices calculation
│   └── dashboard/
│       └── app.py          # Gradio interface
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_detector.ipynb
│   ├── 03_train_classifier.ipynb
│   └── 04_evaluation.ipynb
├── configs/
│   └── model_config.yaml   # Hyperparameters
├── requirements.txt
└── README.md
```

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| BCCD dataset too small | Model overfits | Use heavy augmentation, transfer learning |
| No morphology labels | Can't train direct classifier | Start with rule-based, collect labels later |
| GPU memory limits | Can't train large models | Use EfficientNet-B0/B2, gradient accumulation |
| Image quality variance | Poor detection on bad images | Add quality scoring, reject low-quality inputs |
| Clinical validation gap | Results may not match lab standards | Clearly label as "screening tool", not diagnostic |

## Migration Plan
N/A - New project, no migration needed.

## Open Questions

1. ~~**Calibration method:**~~ → RESOLVED: Use default values, user has existing images to analyze

2. **Dataset licensing:** Kaggle Blood Cell Images - verify license for derivative works before distribution

3. **Validation strategy:** How many images needed to validate clinical utility? Suggest collaborating with pathology lab.

4. ~~**WBC subtype priority:**~~ → RESOLVED: Include all 4 subtypes (Eosinophil, Lymphocyte, Monocyte, Neutrophil)
