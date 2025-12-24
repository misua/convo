# N/C Ratio Integration Guide

## Overview
The Nucleus-to-Cytoplasm (N/C) ratio analyzer enables blast cell detection for improved leukemia screening.

## Clinical Significance

**Normal Cells:**
- Neutrophils: N/C = 0.25-0.45 (small nucleus, abundant cytoplasm)
- Lymphocytes: N/C = 0.65-0.80 (larger nucleus, thin cytoplasm rim)
- Monocytes: N/C = 0.35-0.50

**Abnormal Cells:**
- Blast cells (leukemia): N/C > 0.80 (very large nucleus, minimal cytoplasm)
- Immature forms: N/C = 0.75-0.85

## Quick Start

### 1. Test the N/C Analyzer
```bash
python src/inference/nc_ratio_analyzer.py
```

### 2. Integration Example
```python
from src.inference.nc_ratio_analyzer import NCRatioAnalyzer

analyzer = NCRatioAnalyzer()

# For single cell
result = analyzer.analyze_cell(
    nucleus_mask=nucleus_mask,  # Binary mask (255=nucleus)
    cell_mask=cell_mask,        # Binary mask (255=cell)
    cell_class="Neutrophil"
)

print(f"N/C Ratio: {result.nc_ratio}")
print(f"Blast Confidence: {result.blast_confidence}%")
print(f"Interpretation: {result.interpretation}")

# For batch analysis
cells_data = [
    {'nucleus_mask': mask1, 'cell_mask': cell1, 'class': 'Neutrophil'},
    {'nucleus_mask': mask2, 'cell_mask': cell2, 'class': 'Unknown'},
]

summary = analyzer.analyze_batch(cells_data)
print(f"Blast suspects: {summary['blast_suspects_count']}")
print(f"Risk level: {summary['risk_level']}")
print(f"Clinical action: {summary['clinical_action']}")
```

### 3. Add to Blood Smear Analysis

To integrate with `scripts/analyze_blood_smear.py`:

1. **Train WBC segmentation model:**
   ```bash
   # Use data/wbc_instance_seg/ dataset
   python scripts/train_wbc_segmentation.py
   ```

2. **Add N/C calculation after WBC classification:**
   ```python
   # In analyze_blood_smear.py
   from src.inference.nc_ratio_analyzer import NCRatioAnalyzer
   
   nc_analyzer = NCRatioAnalyzer()
   
   # After WBC classification loop
   if wbc_segmentation_available:
       nc_results = []
       for wbc_box, wbc_class in zip(wbc_boxes, wbc_classes):
           # Extract nucleus and cell masks from segmentation
           nucleus_mask = extract_nucleus_mask(wbc_box)
           cell_mask = extract_cell_mask(wbc_box)
           
           result = nc_analyzer.analyze_cell(
               nucleus_mask, cell_mask, wbc_class
           )
           nc_results.append(result)
       
       # Add to results dict
       summary = nc_analyzer.analyze_batch(nc_results)
       results['nc_ratio_analysis'] = summary
   ```

3. **The diagnostic summary PDF will automatically show N/C ratio data!**

## Dataset: WBC Instance Segmentation

**Location:** `data/wbc_instance_seg/`
**Size:** 720 MB
**Format:** CVAT XML annotations

**Classes:**
- BLAST CELLS (68 images) ← **Key for leukemia detection!**
- NEUTROPHILS
- LYMPHOCYTES
- MONOCYTES
- EOSINOPHILS
- BASOPHILS
- BAND CELLS (immature neutrophils)
- METAMYELOCYTES
- MYELOCYTE
- PROMYELOCYTES

**Annotations:**
Each cell has:
- Nucleus polygon
- Cytoplasm polygon
- Cell type label

## Training a WBC Segmentation Model

You'll need to:
1. Convert CVAT XML to YOLO segmentation format
2. Train YOLO11-seg model
3. Extract nucleus and cytoplasm masks separately

Example training script structure:
```python
# scripts/train_wbc_segmentation.py
from ultralytics import YOLO

# Load model
model = YOLO('yolo11n-seg.pt')

# Train on WBC dataset
results = model.train(
    data='data/wbc_instance_seg_yolo/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='wbc_seg_nc_ratio'
)

# Model will predict:
# - nucleus mask
# - cytoplasm mask
# - cell class
```

## Clinical Impact

**Before (count-based only):**
- "Lymphocytes 60% → Maybe ALL?"
- Many false positives

**After (N/C ratio + counts):**
- "3 cells with N/C > 0.85 detected (blast suspects)"
- "Lymphocytes 60%, but N/C ratios normal → Reactive lymphocytosis, not leukemia"
- Much more specific screening

## Testing

Test the analyzer:
```bash
cd /home/hn/Desktop/convo/blood-cell-analyzer
python src/inference/nc_ratio_analyzer.py
```

Expected output:
```
🧪 Testing N/C Ratio Analyzer

Test 1 - Normal Neutrophil:
  N/C Ratio: 0.326
  Suspicious: False
  Interpretation: Normal N/C ratio for Neutrophil

Test 2 - Blast Cell:
  N/C Ratio: 0.868
  Suspicious: True
  Blast Confidence: 67.0%
  Interpretation: ⚠️ BLAST CELL SUSPECTED
```

## Next Steps

1. ✅ **Done:** N/C ratio analyzer created
2. ✅ **Done:** Diagnostic summary updated to show N/C ratio data
3. 🔄 **TODO:** Train WBC segmentation model on `wbc_instance_seg/` dataset
4. 🔄 **TODO:** Integrate N/C calculation into main analysis pipeline
5. 🔄 **TODO:** Add visualization showing nucleus vs cytoplasm overlay

**Would you like me to:**
- Create the WBC segmentation training script?
- Show how to parse the CVAT XML annotations?
- Create visualization showing blast cells vs normal cells?
