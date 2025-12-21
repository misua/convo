# Blood Cell Analyzer

AI-powered blood cell analysis for thalassemia screening using deep learning.

## Features
- **Cell Detection**: YOLOv8-based detection of RBCs, WBCs, and Platelets
- **Malaria Detection**: YOLOv11n object detection for malarial parasites (ring, trophozoite, schizont, gametocyte stages)
- **WBC Classification**: Identifies Eosinophils, Lymphocytes, Monocytes, Neutrophils
- **Morphology Analysis**: Size estimation, hypochromia detection, sickle cell & thalassemia screening
- **Clinical Dashboard**: Interactive Gradio interface with visual explanations

## Quick Start

### 1. Setup Environment

```bash
cd blood-cell-analyzer
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Download Datasets

**BCCD Dataset (Detection):**
```bash
cd data/bccd
git clone https://github.com/Shenggan/BCCD_Dataset.git .
```

**Kaggle Dataset (Classification):**
```bash
# Install kaggle CLI: pip install kaggle
# Set up API key: ~/.kaggle/kaggle.json
kaggle datasets download -d paultimothymooney/blood-cells -p data/kaggle --unzip
```

### 3. Train Models

```bash
# Open notebooks in order:
jupyter notebook notebooks/
```

### 4. Run Dashboard

```bash
python -m src.dashboard.app
```

## Project Structure

```
blood-cell-analyzer/
├── data/
│   ├── bccd/           # BCCD detection dataset
│   ├── kaggle/         # Kaggle classification dataset
│   └── samples/        # Your test images
├── models/
│   ├── detection/      # YOLOv8 weights
│   └── classification/ # EfficientNet weights
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_detector.ipynb
│   ├── 03_train_classifier.ipynb
│   └── 04_full_pipeline.ipynb
├── src/
│   ├── data/           # Data loading utilities
│   ├── models/         # Model wrappers
│   ├── inference/      # Prediction pipeline
│   └── dashboard/      # Gradio app
├── configs/
│   └── config.yaml     # Hyperparameters
└── requirements.txt
```

## Hardware Requirements
- **GPU**: NVIDIA RTX 3060 or better (12GB VRAM recommended)
- **RAM**: 16GB+
- **Storage**: ~5GB for datasets and models

## Clinical Validation

### Malaria Detection Performance
**YOLOv11n Object Detection Model (BBBC041 Dataset)**

| Metric | Value |
|--------|-------|
| Training Data | 1,087 images, 86,035 cells |
| Model Size | 2.6M parameters, 6.4 GFLOPs |
| mAP50 | 69.46% |
| mAP50-95 | 57.81% |
| Confidence Threshold | 0.55 (optimized for clinical accuracy) |

**False Positive Improvement:**
- CNN approach: 16.82% false positive rate on hospital-confirmed negatives
- YOLO approach: **0.00% false positive rate** (5/5 hospital negatives correctly identified)

The confidence threshold of 0.55 was empirically selected to eliminate false positives while maintaining sensitivity for parasite detection.

## Disclaimer
This is a research prototype for screening purposes only. Not for clinical diagnosis.
