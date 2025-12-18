# Blood Cell Analyzer

AI-powered blood cell analysis for thalassemia screening using deep learning.

## Features
- **Cell Detection**: YOLOv8-based detection of RBCs, WBCs, and Platelets
- **WBC Classification**: Identifies Eosinophils, Lymphocytes, Monocytes, Neutrophils
- **Morphology Analysis**: Size estimation, hypochromia detection
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

## Disclaimer
This is a research prototype for screening purposes only. Not for clinical diagnosis.
