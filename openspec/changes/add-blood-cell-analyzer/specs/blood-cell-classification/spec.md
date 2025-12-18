## ADDED Requirements

### Requirement: Cell Detection Model
The system SHALL detect and localize blood cells (RBC, WBC, Platelets) in peripheral blood smear images using a trained object detection model.

#### Scenario: Detect cells in standard blood smear
- **WHEN** a microscopy image of a blood smear is provided
- **THEN** the system returns bounding boxes for each detected cell
- **AND** each detection includes a class label (RBC, WBC, Platelet)
- **AND** each detection includes a confidence score between 0 and 1

#### Scenario: Handle empty field of view
- **WHEN** an image with no cells is provided
- **THEN** the system returns an empty detection list
- **AND** logs a warning about potential image quality issue

#### Scenario: High-density cell regions
- **WHEN** an image contains overlapping cells
- **THEN** the system applies non-maximum suppression
- **AND** returns only distinct cell detections with IoU threshold ≤ 0.5

### Requirement: Model Training Pipeline
The system SHALL provide a training pipeline for fine-tuning detection models on custom blood cell datasets.

#### Scenario: Train on BCCD dataset
- **WHEN** BCCD dataset is provided in YOLO format
- **THEN** the system trains a YOLOv8 model
- **AND** saves checkpoints every epoch
- **AND** logs training metrics (loss, mAP) to console

#### Scenario: Resume training from checkpoint
- **WHEN** a previous checkpoint path is provided
- **THEN** training resumes from that checkpoint
- **AND** preserves optimizer state and epoch count

#### Scenario: GPU memory management
- **WHEN** training on RTX 3060 (12GB VRAM)
- **THEN** the system uses batch size ≤ 16 by default
- **AND** enables automatic mixed precision (AMP) for memory efficiency

### Requirement: WBC Subtype Classification
The system SHALL classify detected WBC cells into subtypes (Eosinophil, Lymphocyte, Monocyte, Neutrophil).

#### Scenario: Classify cropped WBC
- **WHEN** a cropped WBC image is provided
- **THEN** the system returns the most likely subtype
- **AND** provides confidence scores for all subtypes

#### Scenario: Low confidence classification
- **WHEN** the highest confidence score is below 0.7
- **THEN** the system flags the cell as "uncertain"
- **AND** includes it in the manual review queue

### Requirement: Transfer Learning Support
The system SHALL support transfer learning from ImageNet-pretrained models for classification tasks.

#### Scenario: Initialize from pretrained weights
- **WHEN** training a new classifier
- **THEN** the system loads ImageNet-pretrained weights by default
- **AND** freezes backbone layers for initial epochs

#### Scenario: Fine-tune all layers
- **WHEN** unfreeze flag is set after initial training
- **THEN** all model layers become trainable
- **AND** learning rate is reduced for pretrained layers
