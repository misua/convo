# Tasks: Grad-CAM Heatmap Visualization

change-id: add-gradcam-explainability

## Phase 1: Core Module (Foundation)

- [x] Create `src/inference/gradcam.py` with `GradCAMGenerator` class
  - [x] Support YOLO detection/segmentation models
  - [x] Support EfficientNet classification model
  - [x] Implement target layer auto-detection from config
  - [x] Add caching mechanism for repeated analysis
  
- [x] Create `src/inference/visualization.py` with overlay utilities
  - [x] `create_heatmap_overlay()` - Apply colormap and blend
  - [x] `create_comparison_grid()` - Side-by-side original/overlay
  - [x] `add_colorbar()` - Add intensity legend to image

- [x] Add `pytorch-grad-cam>=1.4.8` to requirements.txt

## Phase 2: Configuration

- [x] Update `configs/config.yaml`
  ```yaml
  gradcam:
    enabled: false
    colormap: "jet"
    alpha: 0.4
    target_layers:
      detection: "model.model.22"
      classification: "features.8"
    save_individual: false
  ```

- [x] Add config validation for gradcam section
- [x] Add config loading in BloodCellAnalyzer init

## Phase 3: Pipeline Integration

- [x] Integrate GradCAM into analysis pipeline
  - [x] Hook into `BloodCellAnalyzer.analyze()` method
  - [x] Generate heatmaps for detected cells when enabled
  - [x] Add heatmaps to results dictionary
  
- [x] Update `scripts/analyze_blood_smear.py`
  - [x] Add `--gradcam` / `--no-gradcam` CLI flag
  - [x] Add `--save-heatmaps <dir>` option
  - [x] Display heatmap summary in console output

## Phase 4: Dashboard Integration

- [x] Update `app.py` with Grad-CAM UI elements
  - [x] Add "Show Grad-CAM Heatmaps" toggle in sidebar
  - [x] Display heatmap overlays in results gallery
  - [x] Add tooltip explaining what heatmaps show
  - [x] Add colorbar legend in sidebar

- [x] Ensure toggle state persists in session

## Phase 5: PDF Report Integration

- [x] Update `src/reports/pdf_generator.py`
  - [x] Add "Model Attention Analysis" section
  - [x] Include heatmap grid (original + overlay pairs)
  - [x] Add explanatory note about red = high attention
  - [x] Conditional: only include if gradcam was enabled

## Phase 6: Testing

- [ ] Unit tests (`tests/test_gradcam.py`)
  - [ ] Test heatmap shape matches input image dimensions
  - [ ] Test overlay has correct colormap applied
  - [ ] Test caching works correctly
  - [ ] Test graceful handling of unsupported models

- [ ] Integration tests
  - [ ] End-to-end test with sample image
  - [ ] Dashboard toggle functionality
  - [ ] PDF generation with heatmaps
  
- [ ] Performance benchmarks
  - [ ] Measure baseline analysis time (no Grad-CAM)
  - [ ] Measure analysis time with Grad-CAM
  - [ ] Document expected overhead

## Phase 7: Documentation

- [ ] Update README.md with Grad-CAM section
  - [ ] Explain what Grad-CAM shows
  - [ ] CLI usage examples
  - [ ] Dashboard screenshot with heatmaps
  - [ ] Performance considerations

- [ ] Add docstrings to all new functions
- [ ] Add inline comments for complex gradient operations

---

## Validation Checklist

- [ ] `openspec validate add-gradcam-explainability --strict` passes
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing on sample images
- [ ] Performance acceptable (< 100ms overhead)
- [ ] Documentation complete
