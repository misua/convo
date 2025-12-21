# Spec: Clinical Dashboard (Delta)

## ADDED Requirements

### Requirement: Plain-Language Diagnosis Panel
The system SHALL provide a patient-friendly explanation of analysis results for non-medical users.

#### Scenario: Display overall assessment
- **WHEN** analysis is complete
- **THEN** the dashboard shows a traffic light indicator:
  - 🟢 GREEN: "Normal Findings - No follow-up needed"
  - 🟡 YELLOW: "Moderate Findings - Follow-up recommended"
  - 🔴 RED: "Significant Findings - Medical consultation advised"
- **AND** the color corresponds to the risk score category

#### Scenario: Explain findings in plain language
- **WHEN** shape abnormalities are detected
- **THEN** the panel explains each finding without medical jargon:
  - "target cells" → "Bull's-eye shaped cells"
  - "microcytes" → "Cells smaller than typical"
  - "hypochromia" → "Cells that appear paler than normal"
  - "teardrop cells" → "Cells with a teardrop or pear shape"
- **AND** includes brief explanation of what each finding means

#### Scenario: Provide "What This Means" section
- **WHEN** findings suggest thalassemia risk
- **THEN** the panel explains:
  - What thalassemia trait is (in simple terms)
  - That many people live normal lives with the trait
  - That this is a screening tool, not a diagnosis
- **AND** uses reassuring, non-alarming language

#### Scenario: Show recommended next steps
- **WHEN** risk is Medium or High
- **THEN** the panel displays actionable steps:
  1. "Visit your doctor to discuss these findings"
  2. "Ask about a Hemoglobin Electrophoresis test"
  3. "Bring this report to your appointment"
- **AND** provides option to download/email report

### Requirement: Shape Gallery Component
The system SHALL display a visual gallery of analyzed cells with their classifications.

#### Scenario: Show cells of interest
- **WHEN** abnormal cells are detected
- **THEN** the gallery displays thumbnails of flagged cells
- **AND** each thumbnail shows shape class label
- **AND** thumbnails are sorted by abnormality confidence

#### Scenario: Interactive cell inspection
- **WHEN** user clicks a cell thumbnail
- **THEN** an expanded view shows:
  - Larger cell image with mask overlay
  - Shape classification and confidence
  - Calculated metrics (circularity, elongation)
  - Plain-language explanation of abnormality

#### Scenario: Filter gallery by shape class
- **WHEN** user selects a shape class filter
- **THEN** only cells of that class are displayed
- **AND** count updates to show filtered total

### Requirement: Shape Distribution Visualization
The system SHALL display shape distribution as an interactive chart.

#### Scenario: Display bar chart
- **WHEN** shape analysis is complete
- **THEN** a horizontal bar chart shows:
  - Percentage of each shape class
  - Color-coded bars (green for normal, yellow/red for abnormal)
  - Hover text with exact counts

#### Scenario: Highlight abnormal distribution
- **WHEN** abnormal shapes exceed 10% of total
- **THEN** those bars are highlighted with border/glow
- **AND** a summary note explains the significance

### Requirement: Report Export
The system SHALL allow users to export analysis results.

#### Scenario: Download PDF report
- **WHEN** user clicks "Download Report"
- **THEN** a PDF is generated containing:
  - Patient-friendly summary
  - Technical details (optional section)
  - Cell gallery thumbnails
  - Risk score and recommendations
  - Medical disclaimer

#### Scenario: Email to healthcare provider
- **WHEN** user clicks "Email to Doctor"
- **THEN** a shareable link or attachment is prepared
- **AND** user is prompted to enter recipient email
- **AND** clear disclaimer is included about tool limitations

### Requirement: Medical Disclaimer Display
The system SHALL prominently display medical disclaimer.

#### Scenario: Show persistent disclaimer
- **WHEN** analysis results are displayed
- **THEN** a disclaimer banner states:
  - "This is an AI-assisted screening tool, NOT a medical diagnosis"
  - "Always consult a qualified healthcare professional"
  - "Results should be confirmed with laboratory testing"
- **AND** disclaimer cannot be dismissed or hidden

## MODIFIED Requirements

### Requirement: Thalassemia Risk Alert (MODIFIED)
The system SHALL prominently display thalassemia risk assessment results **with shape-based findings**.

#### Scenario: Display risk category (MODIFIED)
- **WHEN** risk scoring completes
- **THEN** the dashboard displays:
  - Risk level with traffic light color
  - Primary contributing factors listed
  - **Shape abnormality summary (target cells, teardrops)** [NEW]
  - Confidence score for the assessment
- **AND** explanation of what the risk level means

#### Scenario: Show contributing factors (MODIFIED)
- **WHEN** user expands risk details
- **THEN** breakdown shows:
  - Microcyte percentage with reference range
  - Hypochromia percentage with reference range
  - RDW-proxy with reference range
  - **Target cell count and percentage** [NEW]
  - **Teardrop cell count and percentage** [NEW]
  - Weight each factor contributed to final score
