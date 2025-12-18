## ADDED Requirements

### Requirement: Image Upload Interface
The system SHALL provide an interface for uploading blood smear images for analysis.

#### Scenario: Upload single image
- **WHEN** a user uploads a JPG/PNG image
- **THEN** the system displays the image preview
- **AND** enables the "Analyze" button

#### Scenario: Batch upload
- **WHEN** a user uploads multiple images
- **THEN** the system queues them for sequential analysis
- **AND** displays progress indicator

#### Scenario: Invalid file type
- **WHEN** a user uploads a non-image file
- **THEN** the system displays an error message
- **AND** requests a valid image format

### Requirement: Cell Detection Visualization
The system SHALL display detected cells with bounding boxes overlaid on the original image.

#### Scenario: Show detection overlay
- **WHEN** cell detection completes
- **THEN** the system displays bounding boxes around each cell
- **AND** boxes are color-coded by cell type (RBC: red, WBC: blue, Platelet: green)
- **AND** confidence scores are shown on hover

#### Scenario: Filter by cell type
- **WHEN** user selects a cell type filter
- **THEN** only boxes of that type are displayed
- **AND** count updates to show filtered total

### Requirement: Suspicious Cell Highlighting
The system SHALL highlight cells flagged as abnormal for clinician review.

#### Scenario: Highlight abnormal RBCs
- **WHEN** morphology analysis identifies abnormal cells
- **THEN** those cells are highlighted with distinct markers (yellow border)
- **AND** clicking a cell shows its specific abnormality

#### Scenario: Generate review queue
- **WHEN** analysis completes with abnormal findings
- **THEN** the dashboard creates a scrollable gallery of flagged cells
- **AND** cells are sorted by abnormality confidence (highest first)

### Requirement: Confidence Score Display
The system SHALL display confidence scores for all AI predictions.

#### Scenario: Show overall confidence
- **WHEN** analysis completes
- **THEN** the dashboard displays:
  - Detection confidence (average across all cells)
  - Classification confidence (per WBC subtype)
  - Morphology confidence (per flagged abnormality)

#### Scenario: Low confidence warning
- **WHEN** any key metric has confidence < 0.7
- **THEN** a warning banner is displayed
- **AND** suggests reviewing the image quality

### Requirement: RBC Indices Summary Panel
The system SHALL display calculated RBC indices in a dedicated panel.

#### Scenario: Display indices
- **WHEN** RBC analysis completes
- **THEN** the panel shows:
  - Total RBC count
  - Mean cell diameter (μm)
  - RDW-proxy (%)
  - Microcytic cell percentage
  - Hypochromic cell percentage

#### Scenario: Flag abnormal values
- **WHEN** any index is outside reference range
- **THEN** that value is highlighted in red
- **AND** reference range is shown for comparison

### Requirement: Cell Count Summary
The system SHALL display total cell counts by type.

#### Scenario: Display counts
- **WHEN** detection completes
- **THEN** the dashboard shows:
  - Total cells detected
  - RBC count
  - WBC count (with subtype breakdown)
  - Platelet count

#### Scenario: Export counts
- **WHEN** user clicks "Export CSV"
- **THEN** cell counts are downloaded as CSV file
- **AND** includes timestamp and image filename

### Requirement: Thalassemia Risk Alert
The system SHALL prominently display thalassemia risk assessment results.

#### Scenario: Display risk category
- **WHEN** risk scoring completes
- **THEN** the dashboard shows risk category with color coding:
  - Low: green
  - Medium: yellow  
  - High: red
- **AND** displays the recommendation text

#### Scenario: Show contributing factors
- **WHEN** user clicks on risk score
- **THEN** a breakdown panel shows:
  - Each morphology metric's contribution
  - Which cells triggered the highest concern
  - Links to relevant cells in the image

### Requirement: Report Export
The system SHALL allow exporting analysis results as a clinical report.

#### Scenario: Generate PDF report
- **WHEN** user clicks "Export PDF"
- **THEN** the system generates a report containing:
  - Original image with detection overlay
  - Cell counts table
  - RBC indices summary
  - Risk assessment with recommendation
  - Timestamp and disclaimer

#### Scenario: Include annotated images
- **WHEN** exporting with "Include annotations" checked
- **THEN** the report includes the gallery of flagged cells
- **AND** each cell image has its abnormality label

### Requirement: Microscope Calibration Input
The system SHALL allow users to input microscope calibration for accurate measurements.

#### Scenario: Manual calibration entry
- **WHEN** user enters μm/pixel value in settings
- **THEN** all subsequent measurements use this calibration
- **AND** calibration persists across sessions

#### Scenario: Default calibration
- **WHEN** no calibration is provided
- **THEN** the system uses default (0.25 μm/pixel for 40x objective)
- **AND** displays warning that measurements may be approximate
