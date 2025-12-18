# Custom Dataset for Fine-tuning

## Steps to Add Your Data:

### 1. Add Images
Put 50-100 blood smear images from your microscope into:
```
data/custom/images/
```

### 2. Label Images

**Option A: LabelImg (Recommended)**
```bash
pip install labelImg
labelImg data/custom/images/ data/custom/labels/ data/custom/classes.txt
```

**Option B: Roboflow (Web-based)**
1. Go to https://roboflow.com
2. Create free account
3. Upload images
4. Draw bounding boxes
5. Export as "YOLO v8" format
6. Download and extract to this folder

### 3. Label Format (YOLO)
Each image needs a `.txt` file with same name in `labels/`:
```
# class_id center_x center_y width height (normalized 0-1)
0 0.5 0.5 0.1 0.1   # RBC
1 0.3 0.7 0.15 0.15 # WBC  
2 0.8 0.2 0.05 0.05 # Platelets
```

### 4. Classes
- 0: RBC (Red Blood Cell)
- 1: WBC (White Blood Cell)  
- 2: Platelets

### 5. Run Fine-tuning
```bash
cd /home/hn/Desktop/convo/blood-cell-analyzer
source venv/bin/activate
python scripts/finetune_detector.py
```

## Tips for Labeling:
- Label ALL visible cells (don't skip any)
- Draw tight bounding boxes
- Include partially visible cells at edges
- Be consistent with box sizes
- WBCs are larger and have visible nucleus
- RBCs are smaller, round, pinkish
- Platelets are tiny dots
