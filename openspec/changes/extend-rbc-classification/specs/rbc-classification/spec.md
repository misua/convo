# Spec: RBC Classification

## ADDED Requirements

### Requirement: Extended RBC Shape Categories

The system SHALL classify red blood cells into 9 distinct morphological categories to detect thalassemia, malaria, and sickle cell disease.

**Categories:**
- Normal (healthy RBC)
- Microcyte (small cell - thalassemia/iron deficiency)
- Target cell (codocyte - thalassemia indicator)
- Teardrop (dacrocyte - thalassemia indicator)
- Spherocyte (hereditary spherocytosis)
- Irregular (various abnormalities)
- Ring stage (malaria parasite - early infection)
- Trophozoite (malaria parasite - mature stage)
- Sickle cell (crescent-shaped - sickle cell disease)

#### Scenario: Malaria parasite detection

**Given** a blood smear image with RBCs containing ring-stage malaria parasites  
**When** the analyzer processes the image  
**Then** infected RBCs SHALL be classified as "ring" with >80% sensitivity  
**And** the report SHALL show "Malaria Infected: X cells (Y%)"  
**And** the risk level SHALL be marked as "URGENT"

#### Scenario: Sickle cell detection

**Given** a blood smear with crescent-shaped sickle cells (circularity < 0.5, elongation > 2.0)  
**When** the shape classifier analyzes the cells  
**Then** sickle cells SHALL be identified with >85% sensitivity  
**And** if sickle cells exceed 5% of total RBCs  
**Then** the risk level SHALL be "REFERRAL" with interpretation text

#### Scenario: Mixed morphology detection

**Given** a blood smear with multiple abnormalities (target cells + microcytes + sickle cells)  
**When** the analyzer processes all RBCs  
**Then** each cell SHALL be classified into exactly one category  
**And** the shape distribution report SHALL show percentages for all detected categories  
**And** risk assessments SHALL be generated for each detected disorder

### Requirement: CNN-Based RBC Classification

The system SHALL support deep learning classification using ResNet34 (same architecture as WBC classifier) to detect both shape abnormalities and internal parasites from cropped RBC images.

#### Scenario: CNN classifier with trained model

**Given** a trained ResNet34 model exists at the configured path  
**When** the shape classifier is initialized with `use_deep_learning=True`  
**Then** the CNN model SHALL be loaded successfully  
**And** classification SHALL use CNN predictions with confidence scores  
**And** inference time SHALL remain under 2 seconds for 200 RBCs

#### Scenario: Fallback to rule-based classification

**Given** no trained CNN model is available  
**When** the shape classifier processes RBCs  
**Then** the system SHALL use rule-based geometric metrics (circularity, elongation, solidity)  
**And** SHALL detect sickle cells using circularity < 0.5 and elongation > 2.0  
**And** SHALL classify other shapes using existing threshold logic

### Requirement: Multi-Disorder Risk Assessment

The system SHALL calculate separate risk assessments for thalassemia, malaria, and sickle cell disease based on RBC morphology distribution.

#### Scenario: Malaria risk assessment

**Given** 8% of RBCs are classified as ring or trophozoite stages  
**When** the risk assessment is calculated  
**Then** malaria risk level SHALL be "urgent"  
**And** malaria score SHALL be proportional to infected percentage  
**And** interpretation SHALL state "X infected RBCs detected. Immediate lab confirmation required."

#### Scenario: Sickle cell risk assessment

**Given** 15% of RBCs are classified as sickle cells  
**When** the risk assessment is calculated  
**Then** sickle cell risk level SHALL be "referral"  
**And** interpretation SHALL recommend sickle cell disease workup  
**And** risk score SHALL be min(1.0, sickle_pct / 40)

#### Scenario: Thalassemia risk unchanged

**Given** 35% of RBCs are microcytes, targets, or teardrops  
**When** thalassemia risk is calculated  
**Then** the existing thalassemia logic SHALL remain unchanged  
**And** malaria and sickle cells SHALL NOT count as thalassemia indicators

## MODIFIED Requirements

### Requirement: RBCShape Enumeration

The `RBCShape` enum SHALL be extended from 6 to 9 classes to support malaria and sickle cell detection.

**Previous:** 6 classes (NORMAL, MICROCYTE, TARGET, TEARDROP, SPHEROCYTE, IRREGULAR)  
**Updated:** 9 classes (add RING, TROPHOZOITE, SICKLE)

#### Scenario: Enum compatibility

**Given** existing code references RBCShape enum values  
**When** the enum is extended with new classes  
**Then** all existing enum references SHALL continue to work  
**And** `len(RBCShape)` SHALL return 9  
**And** the CNN model SHALL automatically support 9 output classes

### Requirement: Shape Population Statistics

The `ShapePopulationStats` dataclass SHALL include malaria and sickle cell metrics.

**New fields:**
- `malaria_infected: int` - total infected RBCs
- `malaria_infected_pct: float` - percentage of infected RBCs
- `sickle_cells: int` - total sickle cells
- `sickle_cell_pct: float` - percentage of sickle cells
- `parasite_stages: Dict[str, int]` - breakdown by parasite stage

#### Scenario: Extended statistics calculation

**Given** 200 RBCs analyzed with 16 ring stage, 4 trophozoite, 20 sickle cells  
**When** population statistics are calculated  
**Then** `malaria_infected` SHALL equal 20 (ring + trophozoite)  
**And** `malaria_infected_pct` SHALL equal 10.0  
**And** `sickle_cells` SHALL equal 20  
**And** `sickle_cell_pct` SHALL equal 10.0  
**And** `parasite_stages` SHALL be {"ring": 16, "trophozoite": 4}

### Requirement: PDF Report Shape Analysis

The PDF report shape analysis section SHALL display all 9 RBC categories with clinical significance.

#### Scenario: Extended shape table

**Given** a blood smear analysis with 9 shape categories detected  
**When** the PDF report is generated  
**Then** the shape distribution table SHALL include 9 rows  
**And** malaria stages SHALL show clinical significance "Malaria parasite detected - urgent lab confirmation"  
**And** sickle cells SHALL show "May indicate sickle cell disease"

### Requirement: Dashboard Shape Distribution

The Gradio dashboard shape distribution chart SHALL display all 9 RBC classes with appropriate color coding.

#### Scenario: Urgent alert banner

**Given** malaria parasites are detected in the analysis  
**When** the dashboard renders results  
**Then** an alert banner SHALL display "⚠️ URGENT: Malaria parasites detected"  
**And** the banner SHALL be styled with red/urgent colors  
**And** malaria bars in the chart SHALL be colored red

#### Scenario: Extended bar chart

**Given** shape distribution results with 9 categories  
**When** the dashboard displays the shape distribution chart  
**Then** the chart SHALL show 9 bars (was 6)  
**And** each bar SHALL be labeled with the shape name and percentage  
**And** color coding SHALL be: normal=green, thalassemia=orange, malaria=red, sickle=yellow
