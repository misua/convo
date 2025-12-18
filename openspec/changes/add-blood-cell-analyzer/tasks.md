# Tasks: Blood Cell Analyzer Implementation

## Phase 1: Environment & Data Setup

- [ ] 1.1 **Create project structure** - Initialize Python project with `src/`, `notebooks/`, `data/`, `models/` directories
- [ ] 1.2 **Setup Python environment** - Create `requirements.txt` with PyTorch, Ultralytics, Fast.ai, Gradio, OpenCV
- [ ] 1.3 **Download BCCD dataset** - Clone from GitHub (cell detection annotations)
- [ ] 1.4 **Download Kaggle dataset** - Get Blood Cell Images (WBC subtype classification)
- [ ] 1.5 **Create data exploration notebook** - Visualize both datasets, verify annotations

## Phase 2: Cell Detection (REQUIRED - User has full field images)

- [ ] 2.1 **Convert BCCD to YOLO format** - Prepare annotations for YOLOv8
- [ ] 2.2 **Train YOLOv8 detector** - Fine-tune YOLOv8n on BCCD (~50 epochs)
- [ ] 2.3 **Evaluate detection** - Calculate mAP, visualize predictions on test images
- [ ] 2.4 **Test on user's image** - Run detector on 1000026798.jpg to verify it works
- [ ] 2.5 **Export detection model** - Save best checkpoint

## Phase 3: WBC Subtype Classification

- [ ] 3.1 **Prepare Kaggle dataset** - Organize train/valid/test splits
- [ ] 3.2 **Train classifier** - Fine-tune EfficientNet-B0 with Fast.ai (~20 epochs)
- [ ] 3.3 **Evaluate classifier** - Accuracy, confusion matrix for 4 subtypes
- [ ] 3.4 **Export classifier model** - Save best checkpoint

## Phase 4: Integration & Morphology

- [ ] 4.1 **Build end-to-end pipeline** - Detect → Crop → Classify
- [ ] 4.2 **Add RBC morphology analysis** - Size estimation, color analysis
- [ ] 4.3 **Build risk scoring** - Combine metrics for thalassemia indicators
- [ ] 4.4 **Test full pipeline on user's image** - Verify complete workflow

## Phase 5: Clinical Dashboard

- [ ] 5.1 **Create Gradio app skeleton** - Basic image upload and display
- [ ] 5.2 **Add detection visualization** - Overlay bounding boxes on uploaded images
- [ ] 5.3 **Build cell gallery component** - Show flagged cells with abnormality labels
- [ ] 5.4 **Implement RBC indices panel** - Display measurements with reference ranges
- [ ] 5.5 **Add risk score display** - Show thalassemia risk with contributing factors
- [ ] 5.6 **Implement calibration settings** - Allow μm/pixel input
- [ ] 5.7 **Add export functionality** - PDF report and CSV export

## Phase 6: Testing & Validation

- [ ] 6.1 **Create test image set** - Curate diverse test images (good/poor quality)
- [ ] 6.2 **Write unit tests** - Test measurement functions, risk calculation
- [ ] 6.3 **Benchmark inference speed** - Verify <100ms per image on 3060
- [ ] 6.4 **User acceptance testing** - Demo with sample clinical workflow
- [ ] 6.5 **Document limitations** - Add disclaimers about screening vs diagnostic use

## Dependencies

```
Phase 1 ──▶ Phase 2 (Detection) ──▶ Phase 4 (Integration)
                │                         │
                │                         ▼
                └──▶ Phase 3 (Classify) ──▶ Phase 5 (Dashboard)
                                                │
                                                ▼
                                          Phase 6 (Test)
```

## Parallelizable Work
- Phase 2 (detection) and Phase 3 (classification) can train in parallel after Phase 1
- Tasks 5.6-5.7 can be done independently once basic dashboard works

## Estimated Timeline (RTX 3060)
| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| Phase 1 | 1-2 hours | Setup and data download |
| Phase 2 | 2-3 hours | YOLOv8 detection training |
| Phase 3 | 2-3 hours | Fast.ai classification |
| Phase 4 | 2-3 hours | Pipeline integration |
| Phase 5 | 3-4 hours | Dashboard development |
| Phase 6 | 1-2 hours | Testing and docs |
| **Total** | **11-17 hours** | Can parallelize Phase 2+3 |

## Quick Start Checklist
To get a working demo fastest, complete in order:
1. ✅ 1.1-1.4 (Setup + both datasets)
2. ✅ 2.1-2.4 (Train detector, test on user's image)
3. ✅ 3.1-3.3 (Train classifier)
4. ✅ 4.1 (Integrate pipeline)
5. ✅ 5.1-5.3 (Basic dashboard)

**MVP achievable in ~8-10 hours of focused work.**
