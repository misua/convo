# Malaria Detection Capability

## ADDED Requirements

### Requirement: Object Detection for Malaria Parasites

The system SHALL use YOLO-based object detection to identify malaria parasites in blood smear images.

#### Scenario: Detect parasites in positive sample

**Given** a blood smear image containing malaria parasites  
**When** the malaria detector analyzes the image  
**Then** it SHALL return bounding boxes for all detected parasites  
**And** each detection SHALL include class label (ring/trophozoite/schizont/gametocyte)  
**And** each detection SHALL include confidence score ≥ 0.5  
**And** each detection SHALL include normalized bbox coordinates [x_center, y_center, width, height]

#### Scenario: Report negative on clean sample

**Given** a blood smear image with no malaria parasites (hospital-confirmed negative)  
**When** the malaria detector analyzes the image  
**Then** it SHALL return zero detections  
**And** the false positive rate SHALL be < 1% across negative samples

#### Scenario: Identify parasite life cycle stages

**Given** a blood smear with mixed parasite stages  
**When** the detector identifies parasites  
**Then** it SHALL classify ring forms (early stage)  
**And** it SHALL classify trophozoites (mature stage)  
**And** it SHALL classify schizonts (dividing stage)  
**And** it SHALL classify gametocytes (sexual stage)  
**And** stage classification accuracy SHALL be > 75% per class

---

### Requirement: BBBC041 Dataset Integration

The system SHALL use the BBBC041 Broad Bioimage Benchmark Collection dataset for training and validation.

#### Scenario: Download and prepare dataset

**Given** the BBBC041 dataset source URL  
**When** the preparation script runs  
**Then** it SHALL download 1,364 blood smear images  
**And** it SHALL parse bounding box annotations for ~80,000 cells  
**And** it SHALL convert annotations from COCO format to YOLO format  
**And** it SHALL create train/val/test splits (80%/10%/10%)  
**And** it SHALL generate data.yaml with 6 class labels

#### Scenario: Validate dataset quality

**Given** the prepared YOLO-format dataset  
**When** validation runs  
**Then** all 1,364 images SHALL have corresponding label files  
**And** all bounding box coordinates SHALL be normalized [0, 1]  
**And** no label files SHALL be empty  
**And** class distribution SHALL match expected ratios (±5%)

---

### Requirement: High Detection Accuracy

The trained YOLO model SHALL meet minimum performance thresholds on the BBBC041 test set.

#### Scenario: Achieve target mAP on test set

**Given** the BBBC041 test set (137 images)  
**When** the trained YOLOv11n model is evaluated  
**Then** mAP@0.5 SHALL be ≥ 0.80  
**And** mAP@0.5-0.95 SHALL be ≥ 0.50  
**And** per-class mAP for "ring" SHALL be ≥ 0.75 (most clinically relevant)

#### Scenario: Minimize false positives

**Given** the BBBC041 test set  
**When** detections are evaluated  
**Then** precision SHALL be ≥ 0.85  
**And** false positive rate on uninfected RBCs SHALL be < 2%

#### Scenario: Catch most parasites

**Given** the BBBC041 test set with labeled parasites  
**When** detections are evaluated  
**Then** recall SHALL be ≥ 0.75  
**And** false negative rate SHALL be < 25%

---

### Requirement: Fast Inference Performance

The malaria detector SHALL perform real-time inference suitable for clinical workflow.

#### Scenario: Process images quickly

**Given** a standard blood smear image (1600x1200 pixels)  
**When** the detector runs on CPU  
**Then** inference time SHALL be < 25ms per image  
**And** NMS post-processing SHALL add < 5ms  
**And** total end-to-end detection SHALL be < 30ms

#### Scenario: Efficient memory usage

**Given** the YOLOv11n model loaded in memory  
**When** processing a single image  
**Then** peak memory usage SHALL be < 500MB  
**And** model size on disk SHALL be < 10MB

---

### Requirement: Parasite Density Calculation

The system SHALL calculate parasite density using WHO standard methodology.

#### Scenario: Calculate parasites per 100 WBCs

**Given** a blood smear with detected parasites and WBCs  
**When** analyzing the image  
**Then** it SHALL count total parasites detected  
**And** it SHALL count total WBCs (leukocytes) in the field  
**And** it SHALL calculate density = (parasites / WBCs) × 100  
**And** it SHALL report density in standard WHO format

#### Scenario: Interpret parasitemia levels

**Given** calculated parasite density  
**When** generating results  
**Then** it SHALL classify as:
- Low parasitemia (< 1 per 100 WBCs)
- Moderate parasitemia (1-10 per 100 WBCs)
- High parasitemia (> 10 per 100 WBCs)

---

### Requirement: Visual Detection Overlay

The system SHALL provide annotated images showing detected parasites with bounding boxes.

#### Scenario: Draw bounding boxes on detections

**Given** an image with detected parasites  
**When** generating visualization  
**Then** it SHALL draw bounding boxes around each parasite  
**And** boxes SHALL use color coding: green=ring, orange=trophozoite, red=schizont, magenta=gametocyte  
**And** each box SHALL display class label and confidence percentage  
**And** overlays SHALL be clearly visible on microscopy background

#### Scenario: Generate detection legend

**Given** an annotated image with detections  
**When** displaying results  
**Then** it SHALL include a legend showing color meanings  
**And** legend SHALL show detection counts per stage  
**And** legend SHALL include confidence threshold used

---

### Requirement: Clinical Reporting Integration

The malaria detection results SHALL integrate into existing blood analysis reports.

#### Scenario: Add malaria section to JSON output

**Given** blood smear analysis results  
**When** malaria detection completes  
**Then** output SHALL include "malaria_analysis" section  
**And** section SHALL contain:
- `parasites_detected` (list with bbox, class, confidence)
- `parasite_density` (float, per 100 WBCs)
- `summary` (dict with counts by stage)
- `total_parasites` (integer)
- `status` (POSITIVE/NEGATIVE)

#### Scenario: Display results in dashboard

**Given** malaria analysis results  
**When** user views dashboard  
**Then** it SHALL show "Malaria Detection" section  
**And** section SHALL display parasite counts by stage  
**And** section SHALL show annotated image with bounding boxes  
**And** section SHALL include parasite density with interpretation  
**And** negative results SHALL clearly state "No parasites detected"

#### Scenario: Include in PDF report

**Given** malaria analysis results  
**When** generating PDF report  
**Then** report SHALL include "Malaria Screening" section  
**And** section SHALL show annotated image with detections  
**And** section SHALL include stage-specific counts table  
**And** section SHALL calculate and display parasite density  
**And** section SHALL include clinical interpretation text  
**And** section SHALL include disclaimer: "Screening tool, requires laboratory confirmation"

---

## REMOVED Requirements

### Requirement: CNN-based Malaria Classification

The system SHALL NO LONGER use ResNet34 CNN for cell-level malaria classification.

#### Scenario: Remove CNN malaria detection

**Given** the existing RBC shape classifier with CNN  
**When** updating to object detection  
**Then** CNN malaria classification logic SHALL be removed  
**And** `RBCShape.RING` and `RBCShape.TROPHOZOITE` enums SHALL be removed  
**And** `_cnn_classify()` method for malaria SHALL be removed  
**And** CNN model initialization for malaria SHALL be removed

#### Scenario: Discontinue cell-level infection status

**Given** blood smear analysis output  
**When** transitioning to object detection  
**Then** `malaria_infected` field SHALL be removed from results  
**And** `malaria_infected_pct` field SHALL be removed from results  
**And** cell crops SHALL NOT be classified as "infected"/"normal"

---

## MODIFIED Requirements

### Requirement: Blood Smear Analysis Pipeline

The blood smear analysis pipeline SHALL use object detection instead of classification for malaria detection.

#### Scenario: Analyze blood smear with new detector

**Given** a blood smear image  
**When** running full analysis pipeline  
**Then** it SHALL perform cell segmentation with YOLOv11-seg (unchanged)  
**And** it SHALL detect parasites with YOLOv11 malaria detector (NEW)  
**And** it SHALL classify RBC shapes for thalassemia (unchanged)  
**And** it SHALL classify WBC types (unchanged)  
**And** it SHALL calculate parasite density (NEW)  
**And** results SHALL combine all analyses into unified report

#### Scenario: Backward compatibility in output structure

**Given** existing code consuming analysis results  
**When** new malaria detection is integrated  
**Then** existing fields (WBC, RBC, platelet counts) SHALL remain unchanged  
**And** existing shape analysis fields SHALL remain unchanged  
**And** new `malaria_analysis` section SHALL be added (not replace existing)  
**And** API version SHALL be incremented to indicate change
