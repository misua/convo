## ADDED Requirements

### Requirement: RBC Diameter Measurement
The system SHALL measure the diameter of detected red blood cells in calibrated units (micrometers).

#### Scenario: Measure with known calibration
- **WHEN** microscope calibration (μm/pixel) is provided
- **THEN** the system calculates diameter from bounding box dimensions
- **AND** returns measurements in micrometers
- **AND** normal RBC diameter reference is 6.2-8.2 μm

#### Scenario: Flag microcytic cells
- **WHEN** an RBC has diameter < 6.2 μm
- **THEN** the system flags the cell as "microcytic"
- **AND** includes it in the morphology alert summary

#### Scenario: Flag macrocytic cells
- **WHEN** an RBC has diameter > 8.2 μm
- **THEN** the system flags the cell as "macrocytic"
- **AND** includes it in the morphology alert summary

### Requirement: Hypochromia Detection
The system SHALL detect hypochromic (pale-centered) red blood cells indicative of hemoglobin deficiency.

#### Scenario: Analyze central pallor
- **WHEN** an RBC image is analyzed
- **THEN** the system measures the central pallor ratio
- **AND** central pallor > 1/3 of cell diameter indicates hypochromia

#### Scenario: Calculate hypochromia percentage
- **WHEN** all RBCs in an image are analyzed
- **THEN** the system reports percentage of hypochromic cells
- **AND** threshold > 20% triggers a clinical alert

### Requirement: RBC Distribution Width Proxy
The system SHALL calculate a size distribution metric analogous to RDW from cell measurements.

#### Scenario: Calculate size variation
- **WHEN** ≥ 100 RBCs are measured
- **THEN** the system calculates coefficient of variation of diameters
- **AND** reports as "RDW-proxy" percentage
- **AND** normal range reference is 11.5-14.5%

#### Scenario: Insufficient cells for statistics
- **WHEN** fewer than 100 RBCs are detected
- **THEN** the system reports RDW-proxy with a warning
- **AND** recommends analyzing additional fields of view

### Requirement: Shape Analysis
The system SHALL assess RBC shape regularity to detect abnormal morphologies.

#### Scenario: Calculate circularity index
- **WHEN** an RBC is detected
- **THEN** the system calculates circularity (4π × area / perimeter²)
- **AND** normal RBCs have circularity > 0.9

#### Scenario: Flag irregular shapes
- **WHEN** circularity < 0.8
- **THEN** the system flags potential abnormal morphology
- **AND** categorizes as "target cell", "sickle", or "irregular" based on shape features

### Requirement: Thalassemia Risk Scoring
The system SHALL provide an aggregate risk score for thalassemia trait based on morphology findings.

#### Scenario: Calculate composite score
- **WHEN** morphology analysis is complete for an image
- **THEN** the system combines:
  - Percentage of microcytic cells (weight: 0.4)
  - Percentage of hypochromic cells (weight: 0.4)  
  - RDW-proxy abnormality (weight: 0.2)
- **AND** outputs a risk category: Low (<30%), Medium (30-60%), High (>60%)

#### Scenario: Generate clinical recommendation
- **WHEN** risk score is Medium or High
- **THEN** the system recommends "Hb Electrophoresis for confirmation"
- **AND** displays which specific findings contributed to the score
