# Spec: RBC Shape Classification

## ADDED Requirements

### Requirement: Instance Segmentation for RBCs
The system SHALL use instance segmentation to extract precise boundaries of each red blood cell.

#### Scenario: Generate cell masks
- **WHEN** an image is processed by the detection model
- **THEN** the system generates a binary mask for each detected RBC
- **AND** masks are stored with their bounding box coordinates
- **AND** mask resolution matches the crop region

#### Scenario: Handle overlapping cells
- **WHEN** two or more RBCs overlap in the image
- **THEN** the system generates separate masks for each cell where distinguishable
- **AND** heavily overlapped cells are flagged as "uncertain"
- **AND** uncertain cells are excluded from shape statistics

### Requirement: Shape Metric Calculation
The system SHALL calculate geometric shape metrics from cell masks.

#### Scenario: Calculate circularity
- **WHEN** an RBC mask is available
- **THEN** the system calculates circularity = 4π × area / perimeter²
- **AND** normal RBCs have circularity ≥ 0.85
- **AND** circularity < 0.70 indicates significant shape abnormality

#### Scenario: Calculate elongation
- **WHEN** an RBC mask is available
- **THEN** the system calculates elongation = major_axis / minor_axis
- **AND** normal RBCs have elongation < 1.3
- **AND** elongation > 1.5 suggests elliptocyte or teardrop morphology

#### Scenario: Calculate solidity
- **WHEN** an RBC mask is available
- **THEN** the system calculates solidity = area / convex_hull_area
- **AND** normal RBCs have solidity > 0.95
- **AND** solidity < 0.90 suggests irregular or spiked morphology

#### Scenario: Detect central pallor pattern
- **WHEN** an RBC mask and image crop are available
- **THEN** the system analyzes intensity distribution
- **AND** identifies "bull's-eye" pattern characteristic of target cells
- **AND** central_pallor_ratio > 0.5 suggests target cell morphology

### Requirement: Shape Classification Model
The system SHALL classify each RBC into one of six morphology categories.

#### Scenario: Classify normal RBC
- **WHEN** shape metrics are within normal ranges
- **AND** no central pallor abnormality detected
- **THEN** the cell is classified as "normal"
- **AND** confidence score is returned

#### Scenario: Classify target cell (codocyte)
- **WHEN** central pallor shows bull's-eye pattern
- **AND** circularity is relatively preserved (> 0.80)
- **THEN** the cell is classified as "target"
- **AND** flagged as thalassemia indicator

#### Scenario: Classify teardrop cell (dacrocyte)
- **WHEN** elongation > 1.5
- **AND** shape shows asymmetric tapering
- **THEN** the cell is classified as "teardrop"
- **AND** flagged as thalassemia indicator

#### Scenario: Classify microcyte
- **WHEN** cell diameter < 6.2 μm
- **AND** shape metrics are otherwise normal
- **THEN** the cell is classified as "microcyte"
- **AND** flagged for size abnormality

#### Scenario: Classify spherocyte
- **WHEN** circularity > 0.95
- **AND** cell appears smaller and denser than normal
- **AND** no central pallor visible
- **THEN** the cell is classified as "spherocyte"

#### Scenario: Classify irregular
- **WHEN** shape does not fit other categories
- **AND** circularity < 0.70 OR solidity < 0.85
- **THEN** the cell is classified as "irregular"
- **AND** shape metrics are included for manual review

### Requirement: Shape Population Statistics
The system SHALL aggregate shape classifications across all cells in an image.

#### Scenario: Calculate shape distribution
- **WHEN** all RBCs in an image are classified
- **THEN** the system reports count and percentage per shape class
- **AND** distribution is displayed as bar chart
- **AND** abnormal classes are highlighted

#### Scenario: Calculate shape abnormality index
- **WHEN** shape distribution is calculated
- **THEN** the system computes abnormality index:
  - target_cells × 3.0 (high weight - thalassemia specific)
  - teardrop_cells × 2.5 (high weight - thalassemia specific)
  - microcytes × 1.5
  - spherocytes × 1.0
  - irregular × 1.0
- **AND** index > 15% triggers elevated risk flag

#### Scenario: Identify cells of interest
- **WHEN** analysis is complete
- **THEN** the system creates a ranked list of "cells of interest"
- **AND** cells are sorted by abnormality confidence (highest first)
- **AND** top 10 cells are displayed in review gallery
