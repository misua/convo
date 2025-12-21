# Change: Add Grad-CAM Heatmap Visualization for Model Explainability

## Why

The blood cell analyzer provides predictions but lacks visual explanations for **why** the model made specific decisions. This creates a trust gap:

1. **Clinical Trust** - Doctors need to see which cell regions the AI focused on before trusting automated classifications
2. **Quality Control** - Without attention maps, impossible to catch if model focuses on artifacts/background instead of cells
3. **Debugging** - When classification is wrong, no way to understand what the model "saw"
4. **Training Value** - Lab technicians can learn diagnostic features by seeing what the AI considers important

Grad-CAM (Gradient-weighted Class Activation Mapping) provides visual explanations by highlighting regions that contributed most to the prediction.

## What Changes

### 1. Grad-CAM Core Module
- New `src/inference/gradcam.py` module with `GradCAM` class
- Support for YOLO (detection/segmentation) and CNN classifiers (WBC/shape)
- Generate attention heatmaps showing model focus regions
- Configurable target layers per model architecture

### 2. Heatmap Overlay Visualization
- Overlay heatmaps on original cell images (customizable colormap)
- Side-by-side view: Original | Heatmap | Overlay
- Per-cell heatmaps in WBC gallery and shape analysis
- Color intensity scale (blue=low attention → red=high attention)

### 3. Dashboard Integration
- New sidebar toggle: "Show Grad-CAM Heatmaps"
- Heatmap overlay on detected cells when enabled
- Per-cell popup showing attention visualization
- Batch mode support for all detected cells

### 4. PDF Report Enhancement
- "Model Attention Analysis" section in generated reports
- Side-by-side Original vs Grad-CAM for key findings
- Include heatmaps for any cells with low confidence (debugging aid)

### 5. CLI Support
- `--gradcam` flag for `analyze_blood_smear.py`
- `--save-heatmaps <dir>` to export heatmap images

## Impact

| Area | Changes |
|------|---------|
| **Affected specs** | New `gradcam-visualization` capability |
| **Affected code** | `app.py`, `analyze_blood_smear.py`, `pdf_generator.py`, new `gradcam.py` |
| **New dependencies** | `pytorch-grad-cam>=1.4.8` |
| **Performance** | +50-80ms latency per image when enabled |
| **Hardware** | No additional VRAM; runs on CPU/GPU |

## Success Criteria

1. Grad-CAM generates valid heatmaps for all supported model types (YOLO, EfficientNet)
2. Heatmaps correctly highlight cell regions (not background) for >90% of predictions
3. Dashboard toggle enables/disables heatmaps without breaking existing workflow
4. PDF reports include heatmap section when enabled
5. Latency increase < 100ms per image on RTX 3060
6. Heatmaps render correctly on cells of all sizes (WBC, RBC, platelets)

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Heatmaps may highlight irrelevant regions | Add quality check; warn if attention outside bbox |
| Performance overhead too high | Cache heatmaps; make optional toggle |
| YOLO attention layer not straightforward | Use established pytorch-grad-cam library |
| Colormap confuses non-technical users | Add clear legend; explain "red = high attention" |
