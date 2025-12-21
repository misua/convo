# Spec: Morphology Analysis (Delta)

## MODIFIED Requirements

### Requirement: Thalassemia Risk Scoring (MODIFIED)
The system SHALL provide an aggregate risk score for thalassemia trait based on morphology findings **including shape analysis**.

#### Scenario: Calculate composite score (MODIFIED)
- **WHEN** morphology analysis is complete for an image
- **THEN** the system combines:
  - Shape abnormality index (weight: 0.35) **[NEW]**
  - Percentage of microcytic cells (weight: 0.30) **[was 0.40]**
  - Percentage of hypochromic cells (weight: 0.25) **[was 0.40]**
  - RDW-proxy abnormality (weight: 0.10) **[was 0.20]**
- **AND** outputs a risk category: Low (<30%), Medium (30-60%), High (>60%)
- **AND** includes confidence interval based on cell count

#### Scenario: Generate clinical recommendation (MODIFIED)
- **WHEN** risk score is Medium or High
- **THEN** the system recommends "Hemoglobin Electrophoresis for confirmation"
- **AND** displays which specific findings contributed to the score
- **AND** lists shape abnormalities with counts **[NEW]**

## ADDED Requirements

### Requirement: Shape-Integrated Analysis
The system SHALL integrate shape classification results into morphology analysis.

#### Scenario: Correlate size and shape
- **WHEN** an RBC is both microcytic AND a target cell
- **THEN** the system flags as "strongly suggestive of thalassemia"
- **AND** increases individual cell risk weight by 50%

#### Scenario: Detect mixed population
- **WHEN** significant variation exists between cell shapes
- **THEN** the system reports "dimorphic population detected"
- **AND** suggests possible iron deficiency + thalassemia trait

### Requirement: Calibration Support
The system SHALL support microscope calibration for accurate measurements.

#### Scenario: Use default calibration
- **WHEN** no calibration is specified
- **THEN** the system uses default of 0.25 μm/pixel (40x objective)
- **AND** displays warning that measurements may be approximate

#### Scenario: User provides calibration
- **WHEN** user specifies μm/pixel ratio
- **THEN** all size measurements use the provided calibration
- **AND** calibration is stored for future analyses

#### Scenario: Auto-detect from metadata
- **WHEN** image contains EXIF metadata with magnification info
- **THEN** the system attempts to calculate calibration automatically
- **AND** prompts user to confirm or override
