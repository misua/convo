# WBC Segmentation Training Pipeline

This directory contains a modular training pipeline for WBC (White Blood Cell) instance segmentation to enable N/C (Nucleus-to-Cytoplasm) ratio analysis for leukemia detection.

## 🎯 Purpose

Train a YOLO11-seg model that segments individual WBC cells, then use image processing to extract nucleus regions for N/C ratio calculation. This enables automated blast cell detection (N/C ratio >0.8) without manual nucleus annotation.

## 📁 Module Structure

```
src/data/
├── cvat_parser.py          # Parse CVAT XML annotations
├── yolo_converter.py       # Convert to YOLO segmentation format
└── dataset_builder.py      # Build train/val/test splits

scripts/
└── train_wbc_segmentation.py  # Training script

src/inference/
└── nucleus_extractor.py    # Extract nucleus from cell mask (image processing bridge)
```

## 🔧 Key Insight

The CVAT annotations **only contain whole cell boundaries**, not separate nucleus polygons. To enable N/C ratio analysis without weeks of manual annotation:

1. **Train cell segmentation** using existing annotations (10 WBC classes)
2. **Extract nucleus** using image processing (Otsu/adaptive thresholding)
3. **Calculate N/C ratio** using existing `NCRatioAnalyzer`

This gets the feature working immediately, with option to train nucleus model later.

## 🚀 Quick Start

### Step 1: Build YOLO Dataset

```bash
# From blood-cell-analyzer directory
cd src/data
python dataset_builder.py ../../data/wbc_instance_seg ../../data/wbc_yolo_seg
```

This will:
- Parse all CVAT XML files (11 cell types including blast cells)
- Convert polygons to YOLO segmentation format
- Split into train/val/test (70/20/10) ensuring blast cells distributed
- Create `data.yaml` configuration
- Generate dataset summary report

Expected output structure:
```
data/wbc_yolo_seg/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
└── dataset_summary.json
```

### Step 2: Train YOLO11-seg Model

```bash
# Small model (fastest)
python scripts/train_wbc_segmentation.py data/wbc_yolo_seg/data.yaml \
    --model n \
    --epochs 100 \
    --batch 16 \
    --imgsz 640

# Medium model (better accuracy)
python scripts/train_wbc_segmentation.py data/wbc_yolo_seg/data.yaml \
    --model m \
    --epochs 150 \
    --batch 8 \
    --imgsz 800
```

**Training Parameters:**
- `--model`: YOLO11 size (n/s/m/l/x) - 'n' for testing, 'm' for production
- `--epochs`: Training epochs (100-200 recommended)
- `--batch`: Batch size (adjust for GPU memory)
- `--imgsz`: Input image size (640/800/1024)
- `--patience`: Early stopping patience (default: 20)
- `--device`: cuda/cpu (auto-detected)

### Step 3: Test Nucleus Extraction

```bash
cd src/inference
python nucleus_extractor.py
```

This tests the nucleus extraction on synthetic data. Methods available:
- **otsu**: Otsu's thresholding (default, works well for stained cells)
- **adaptive**: Adaptive thresholding (better for varying illumination)
- **color_based**: LAB color space segmentation (best for purple/blue nuclei)

## 📊 Dataset Overview

**Total annotations**: ~720MB across 10+ cell type folders

**Classes (10)**:
1. **blast_cell** (68 samples) - Critical for leukemia detection
2. neutrophil
3. lymphocyte
4. monocyte
5. eosinophil
6. basophil
7. band_cell
8. promyelocyte
9. myelocyte
10. metamyelocyte

**CVAT Structure**: Each folder contains `annotations.xml` with polygon coordinates for whole cells (NOT nucleus).

## 🔬 Nucleus Extraction Strategy

Since CVAT annotations lack nucleus polygons, `NucleusExtractor` uses image processing:

1. **Otsu thresholding** (default): Automatically finds threshold separating dark nucleus from lighter cytoplasm
2. **Morphological cleanup**: Removes noise, fills holes
3. **Largest component**: Keeps only the largest connected region
4. **Quality assessment**: Validates nucleus/cell ratio is reasonable (0.1-0.95)

**Why this works**: Blood cell nuclei are stained with Romanowsky dyes (Wright-Giemsa), appearing purple/blue and darker than cytoplasm.

## 🧪 Module Testing

Each module can be tested independently:

```bash
# Test CVAT parser
cd src/data
python cvat_parser.py ../../data/wbc_instance_seg

# Test YOLO converter
python yolo_converter.py

# Test nucleus extractor
cd ../inference
python nucleus_extractor.py
```

## 📈 Integration with N/C Ratio Analyzer

After training, integrate with existing pipeline:

```python
from ultralytics import YOLO
from src.inference.nucleus_extractor import NucleusExtractor
from src.inference.nc_ratio_analyzer import NCRatioAnalyzer

# Load trained model
model = YOLO('models/segmentation/wbc_nc_ratio/train/weights/best.pt')

# Initialize extractors
nucleus_extractor = NucleusExtractor(method='otsu')
nc_analyzer = NCRatioAnalyzer()

# Run inference
results = model.predict(image_path)
for result in results:
    for mask in result.masks.data:
        # Extract nucleus from cell mask
        nucleus_result = nucleus_extractor.extract(image, mask.cpu().numpy())
        
        # Calculate N/C ratio
        nc_result = nc_analyzer.analyze_cell(
            nucleus_mask=nucleus_result.nucleus_mask,
            cell_mask=mask.cpu().numpy()
        )
        
        if nc_result.blast_confidence > 0.5:
            print(f"Blast cell detected! N/C ratio: {nc_result.nc_ratio:.3f}")
```

## 🎓 Design Principles

**Modularity**: Each component is independent and can be replaced
- Swap `NucleusExtractor` with trained nucleus model later
- Replace CVAT parser if annotation format changes
- Use different YOLO versions (YOLOv8, YOLOv11, etc.)

**Testability**: Every module has `if __name__ == "__main__"` test code

**Documentation**: Clear docstrings and type hints throughout

**Flexibility**: Easy to adjust thresholds, parameters, and methods

## 🚧 Future Enhancements

1. **Train nucleus segmentation model**: If N/C ratio proves valuable, invest in nucleus annotation
2. **Multi-scale training**: Use larger image sizes (1024px) for better small cell detection
3. **Ensemble models**: Combine multiple model sizes for robust predictions
4. **Active learning**: Prioritize annotation of uncertain cases

## 📝 Dataset Statistics

After running `dataset_builder.py`, check `data/wbc_yolo_seg/dataset_summary.json` for:
- Images per split
- Annotations per class
- Blast cell distribution verification

**Critical requirement**: Minimum 10 blast cell images per split to enable leukemia detection.

## ⚠️ Known Limitations

1. **Nucleus extraction quality**: Image processing is ~80% solution, not as accurate as trained model
2. **Staining variability**: Different lab staining protocols may affect nucleus detection
3. **Overlapping cells**: YOLO may struggle with heavily overlapped cells
4. **Image quality**: Poor focus or lighting degrades nucleus extraction

## 🔗 Related Documentation

- [N/C Ratio Integration Guide](../../docs/nc_ratio_integration.md)
- [Malaria Detection Implementation](../../docs/malaria_implementation_report.md)
- [NCRatioAnalyzer API](../inference/nc_ratio_analyzer.py)

## 📞 Troubleshooting

**Issue**: CVAT parser fails
- Check XML file exists: `data/wbc_instance_seg/<FOLDER>/annotations.xml`
- Verify folder names match (e.g., "BLAST CELLS" with space)

**Issue**: Low mAP during training
- Increase `--epochs` to 150-200
- Use larger model size (--model m or l)
- Check data.yaml paths are correct

**Issue**: Poor nucleus extraction
- Try different method: `NucleusExtractor(method='adaptive')`
- Adjust thresholds: `min_nucleus_ratio=0.15, max_nucleus_ratio=0.9`
- Inspect image quality and staining

---

**Last Updated**: 2024
**Author**: Blood Cell Analyzer Team
**License**: Project License
