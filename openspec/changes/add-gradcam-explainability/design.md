# Design: Grad-CAM Heatmap Visualization System

## Context

Adding model explainability via Grad-CAM to visualize which image regions contribute to classification decisions. This enables clinical trust verification and debugging.

### Current State
```
Image → Model → Prediction + Confidence
                     ↓
            (No explanation WHY)
```

### Proposed State
```
Image → Model → Prediction + Confidence + Grad-CAM Heatmap
                     ↓                            ↓
              Decision made              Visual explanation of focus regions
```

### Stakeholders
- **Clinicians** - Need to verify AI is looking at correct cell features
- **Lab technicians** - Learn diagnostic features from attention patterns
- **Developers** - Debug model behavior when predictions are wrong
- **Patients** - See visual evidence of AI analysis (builds trust)

## Goals / Non-Goals

### Goals
- Generate Grad-CAM heatmaps for all model predictions
- Support YOLO detection/segmentation and CNN classification models
- Provide intuitive visualization with overlays
- Minimal performance impact (<100ms per image)
- Seamless dashboard integration with toggle
- Include in PDF reports

### Non-Goals
- Other XAI methods (SHAP, LIME, attention rollout) - Grad-CAM only for MVP
- Real-time video analysis with heatmaps
- Training-time saliency maps
- Heatmaps for platelet detection (focus on WBC/RBC)

## Technical Decisions

### Decision 1: Use pytorch-grad-cam Library

**Choice:** `pytorch-grad-cam>=1.4.8` for Grad-CAM implementation

**Rationale:**
- Battle-tested library with 7k+ GitHub stars
- Supports diverse architectures (CNN, ViT, YOLO-compatible)
- Handles gradient computation correctly
- Multiple CAM variants available (GradCAM, GradCAM++, EigenCAM)

**Integration:**
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# For CNN classifier
cam = GradCAM(model=classifier, target_layers=[model.layer4])
grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(class_idx)])
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
```

**Alternatives Considered:**
| Option | Pros | Cons |
|--------|------|------|
| Custom implementation | Full control | Error-prone; maintenance burden |
| Captum (Facebook) | More methods | Heavier dependency; overkill |
| tf-explain | TensorFlow native | We use PyTorch |

### Decision 2: Target Layer Selection Strategy

**Choice:** Last convolutional layer before classification head

**Rationale:**
- Last conv layer contains most spatially-resolved semantic information
- Well-established in Grad-CAM literature
- Works across architectures with predictable results

**Layer Mapping:**
```yaml
gradcam:
  target_layers:
    yolo11: "model.model.22"           # Last C2f block before detect head
    yolo8: "model.model.22"            # Same structure
    efficientnet_b0: "features.8"      # Last MBConv block
    resnet34: "layer4"                 # Last residual block
    efficientnet_v2_s: "features.6"    # Last fused-MBConv
```

### Decision 3: Heatmap Visualization Parameters

**Choice:** Configurable colormap with default "jet" and alpha=0.4

**Rationale:**
- "jet" colormap (blue→green→yellow→red) is most intuitive for "attention intensity"
- Alpha=0.4 balances visibility of heatmap and original cell structure
- Configurable for user preference

**Config Structure:**
```yaml
gradcam:
  enabled: false  # Off by default (performance)
  colormap: "jet"  # Options: jet, turbo, viridis, inferno
  alpha: 0.4       # Overlay transparency [0.0-1.0]
  quality_threshold: 0.3  # Warn if attention outside bbox > threshold
  save_individual: false  # Save per-cell heatmaps to disk
```

### Decision 4: Performance Optimization

**Choice:** Lazy computation with optional caching

**Rationale:**
- Grad-CAM adds ~50-80ms per forward pass
- Users may not always want heatmaps
- Cache heatmaps when same image analyzed multiple times

**Implementation:**
```python
class GradCAMGenerator:
    def __init__(self, model, target_layer, cache_enabled=True):
        self.cache = {}  # {image_hash: {class_idx: heatmap}}
        
    def generate(self, image, class_idx, use_cache=True):
        cache_key = (hash(image.tobytes()), class_idx)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        # ... generate heatmap
        self.cache[cache_key] = heatmap
        return heatmap
```

## Architecture

### Module Structure
```
blood-cell-analyzer/
├── src/
│   └── inference/
│       ├── gradcam.py           # NEW: Core Grad-CAM generator
│       └── visualization.py     # NEW: Heatmap overlay utilities
├── scripts/
│   └── analyze_blood_smear.py   # MODIFY: Add --gradcam flag
├── app.py                       # MODIFY: Add heatmap toggle
└── configs/
    └── config.yaml              # MODIFY: Add gradcam section
```

### Class Design

```python
# gradcam.py
class GradCAMGenerator:
    """Generate Grad-CAM heatmaps for model predictions."""
    
    def __init__(self, model, target_layer: str, device: str = "cuda"):
        """Initialize with model and target layer name."""
        
    def generate_heatmap(
        self,
        image: np.ndarray,
        class_idx: int,
        resize_to: Tuple[int, int] = None
    ) -> np.ndarray:
        """Generate grayscale heatmap [0-1] for given class."""
        
    def generate_overlay(
        self,
        image: np.ndarray,
        class_idx: int,
        colormap: str = "jet",
        alpha: float = 0.4
    ) -> np.ndarray:
        """Generate RGB image with heatmap overlay."""


# visualization.py
def create_comparison_grid(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    label: str
) -> np.ndarray:
    """Create side-by-side comparison image."""

def add_colorbar(
    image: np.ndarray,
    colormap: str = "jet"
) -> np.ndarray:
    """Add attention intensity colorbar to image."""
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Analysis Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Image ──► Detection ──► Classification ──► Results            │
│                │                 │                               │
│                ▼                 ▼                               │
│         [If gradcam enabled]  [If gradcam enabled]              │
│                │                 │                               │
│                ▼                 ▼                               │
│          GradCAM for       GradCAM for                          │
│          detection         classification                       │
│                │                 │                               │
│                └────────┬────────┘                              │
│                         ▼                                        │
│                  Overlay Generation                              │
│                         │                                        │
│                         ▼                                        │
│               ┌─────────┴──────────┐                            │
│               ▼                    ▼                             │
│          Dashboard              PDF Report                       │
│          (interactive)          (static)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Points

### Dashboard (app.py)
```python
# Sidebar toggle
show_gradcam = st.sidebar.checkbox("Show Grad-CAM Heatmaps", value=False)

# When displaying results
if show_gradcam:
    gradcam_gen = GradCAMGenerator(model, target_layer)
    heatmap = gradcam_gen.generate_overlay(cell_crop, predicted_class)
    st.image(heatmap, caption=f"Attention: {class_name}")
```

### CLI (analyze_blood_smear.py)
```bash
python scripts/analyze_blood_smear.py \
    --image sample.jpg \
    --gradcam \
    --save-heatmaps output/heatmaps/
```

### PDF Report (pdf_generator.py)
```python
if gradcam_enabled:
    report.add_section("Model Attention Analysis")
    report.add_image_grid([original, overlay], 
                          captions=["Original", "Grad-CAM Attention"])
    report.add_note("Red regions indicate high model attention")
```

## Trade-offs

| Trade-off | Decision | Rationale |
|-----------|----------|-----------|
| Performance vs Always-on | Disabled by default | Most users don't need heatmaps every time |
| Accuracy vs Speed | Use Grad-CAM (not Grad-CAM++) | Sufficient quality; faster computation |
| Visual clarity vs Information | Fixed alpha=0.4 | Tested balance; configurable if needed |
| Per-cell vs Global | Per-cell heatmaps | More useful for diagnosis verification |

## Quality Assurance

### Heatmap Quality Check
```python
def validate_heatmap(heatmap: np.ndarray, bbox: List[int]) -> float:
    """Return fraction of attention inside bounding box."""
    mask = np.zeros_like(heatmap)
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    attention_inside = (heatmap * mask).sum() / heatmap.sum()
    return attention_inside  # Should be > 0.7 for good localization
```

If attention_inside < 0.3, log warning that model may be focusing on wrong regions.

## Testing Strategy

1. **Unit tests**: Verify heatmap dimensions match input image
2. **Visual tests**: Generate heatmaps for known cells; verify focus on cell body
3. **Performance tests**: Measure latency with/without Grad-CAM
4. **Integration tests**: End-to-end with dashboard toggle
5. **Regression tests**: Ensure predictions unchanged when Grad-CAM enabled
