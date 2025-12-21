# Spec: Grad-CAM Visualization

## ADDED Requirements

### Requirement: GradCAMGenerator Class
The system SHALL provide a `GradCAMGenerator` class that accepts a PyTorch model and target layer specification.

#### Scenario: GradCAMGenerator Initialization
- **WHEN** a PyTorch model and target layer name are provided
- **THEN** GradCAMGenerator is instantiated successfully
- **AND** the generator is ready to produce heatmaps

#### Scenario: Heatmap Generation
- **WHEN** generate_heatmap() is called with an image and class index
- **THEN** a grayscale numpy array is returned
- **AND** all values are in range [0.0, 1.0]
- **AND** array dimensions match input image dimensions

### Requirement: Multi-Architecture Support
The system SHALL support Grad-CAM heatmap generation for YOLO detection/segmentation models and EfficientNet classification models.

#### Scenario: YOLO Model Support
- **WHEN** a YOLO11n detection model is loaded
- **AND** target layer "model.model.22" is specified
- **THEN** heatmap generation succeeds without errors

#### Scenario: EfficientNet Model Support
- **WHEN** an EfficientNet-B0 classification model is loaded
- **AND** target layer "features.8" is specified
- **THEN** heatmap generation succeeds without errors

### Requirement: Heatmap Overlay Visualization
The system SHALL provide a create_heatmap_overlay() function that blends heatmap with original image.

#### Scenario: Create Overlay
- **WHEN** create_heatmap_overlay() is called with image and heatmap
- **THEN** returned image shows heatmap blended with original
- **AND** configurable colormap (jet, turbo, viridis, inferno) is applied
- **AND** configurable alpha transparency (default 0.4) is applied

#### Scenario: Comparison Grid
- **WHEN** create_comparison_grid() is called with original and overlay
- **THEN** side-by-side visualization is returned
- **AND** left half shows original image
- **AND** right half shows heatmap overlay

### Requirement: Grad-CAM Configuration
The config.yaml SHALL include a gradcam section with enabled flag, colormap, alpha, and target layers.

#### Scenario: Config Structure
- **WHEN** config.yaml is parsed
- **THEN** gradcam.enabled is a boolean (default false)
- **AND** gradcam.colormap is a string (default "jet")
- **AND** gradcam.alpha is a float between 0.0 and 1.0 (default 0.4)
- **AND** gradcam.target_layers contains detection and classification keys

#### Scenario: Config Validation
- **WHEN** gradcam.colormap is set to an invalid value
- **THEN** ConfigError is raised on startup
- **AND** error message lists supported colormaps

### Requirement: CLI Integration
The analyze_blood_smear.py script SHALL accept --gradcam flag and --save-heatmaps option.

#### Scenario: Enable via CLI Flag
- **WHEN** script is run with --gradcam flag
- **THEN** heatmap generation is enabled
- **AND** heatmaps are generated for detected cells

#### Scenario: Save Heatmaps
- **WHEN** script is run with --save-heatmaps /output/
- **THEN** heatmap PNG files are saved to specified directory
- **AND** console displays "Grad-CAM heatmaps saved to: /output/"

### Requirement: Dashboard Toggle
The Gradio dashboard SHALL include a "Show Grad-CAM Heatmaps" toggle checkbox.

#### Scenario: Toggle Visible
- **WHEN** user views the dashboard
- **THEN** "Show Grad-CAM Heatmaps" checkbox is visible in sidebar

#### Scenario: Toggle Enables Heatmaps
- **WHEN** user enables the toggle after analyzing an image
- **THEN** heatmap overlays appear for each detected cell
- **AND** colorbar legend is displayed explaining intensity scale

### Requirement: PDF Report Integration
When Grad-CAM is enabled, PDF reports SHALL include a "Model Attention Analysis" section.

#### Scenario: PDF Section Added
- **WHEN** analysis is run with gradcam enabled
- **AND** PDF report is generated
- **THEN** report contains "Model Attention Analysis" section
- **AND** section shows original/overlay image pairs
- **AND** section includes explanatory text about attention colors

### Requirement: Performance Requirements
Grad-CAM heatmap generation SHALL complete within 100ms per cell on GPU with caching support.

#### Scenario: Performance Target
- **WHEN** generate_heatmap() is called on GPU
- **THEN** execution completes in less than 100ms per cell

#### Scenario: Caching
- **WHEN** same image is analyzed twice for same class
- **THEN** cached result is returned on second call
- **AND** no gradient computation occurs

#### Scenario: No Overhead When Disabled
- **WHEN** gradcam.enabled is false
- **THEN** no gradcam-related computation occurs
- **AND** analysis time equals baseline
