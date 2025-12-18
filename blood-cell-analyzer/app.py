#!/usr/bin/env python3
"""
Gradio Dashboard for Blood Smear Analysis
"""

import sys
from pathlib import Path
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import json
from collections import Counter
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent  # blood-cell-analyzer/
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_blood_smear import BloodSmearAnalyzer


# Global analyzer instance
analyzer = None


def init_analyzer():
    """Initialize the analyzer if not already done."""
    global analyzer
    if analyzer is None:
        print("Loading models...")
        analyzer = BloodSmearAnalyzer(detection_conf=0.25)
        print("Models loaded!")
    return analyzer


def analyze_image(image, confidence_threshold):
    """Analyze uploaded blood smear image."""
    if image is None:
        return None, "Please upload an image"
    
    # Initialize analyzer
    init_analyzer()
    analyzer.detection_conf = confidence_threshold
    
    # Save temp image
    temp_path = PROJECT_ROOT / "data/samples/temp_upload.jpg"
    Image.fromarray(image).save(temp_path)
    
    # Run analysis
    try:
        results = analyzer.analyze(str(temp_path), save_visualization=True)
        
        # Load visualization
        vis_path = temp_path.parent / f"{temp_path.stem}_analyzed.jpg"
        vis_image = cv2.imread(str(vis_path))
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        # Format report
        report = format_report(results)
        
        return vis_image, report
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def format_report(results: dict) -> str:
    """Format analysis results as markdown report."""
    lines = [
        "# 🔬 Blood Smear Analysis Report",
        "",
        f"**Analyzed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Image:** {Path(results['image_path']).name}",
        f"**Size:** {results['image_size']['width']} x {results['image_size']['height']}",
        "",
        "---",
        "",
        "## 📊 Cell Counts",
        "",
        "| Cell Type | Count |",
        "|-----------|-------|",
        f"| 🔴 RBC | {results['cell_counts']['RBC']} |",
        f"| ⚪ WBC | {results['cell_counts']['WBC']} |",
        f"| 🟣 Platelets | {results['cell_counts']['Platelets']} |",
        f"| **Total** | **{results['summary']['total_cells']}** |",
        "",
    ]
    
    if results.get('differential') and results['summary']['wbc_classified'] > 0:
        lines.extend([
            "---",
            "",
            "## 🔬 WBC Differential",
            "",
            "| Subtype | Count | Percentage |",
            "|---------|-------|------------|",
        ])
        
        for subtype in ["NEUTROPHIL", "LYMPHOCYTE", "MONOCYTE", "EOSINOPHIL"]:
            data = results['differential'].get(subtype, {"count": 0, "percentage": 0})
            icon = {"NEUTROPHIL": "🟣", "LYMPHOCYTE": "🟢", "MONOCYTE": "🔵", "EOSINOPHIL": "🟠"}.get(subtype, "⚪")
            lines.append(f"| {icon} {subtype.capitalize()} | {data['count']} | {data['percentage']:.1f}% |")
        
        lines.extend([
            "",
            "---",
            "",
            "## 📋 Clinical Notes",
            "",
        ])
        
        # Add simple clinical interpretation
        neutrophil_pct = results['differential'].get("NEUTROPHIL", {}).get("percentage", 0)
        lymphocyte_pct = results['differential'].get("LYMPHOCYTE", {}).get("percentage", 0)
        
        if neutrophil_pct > 70:
            lines.append("⚠️ **Elevated Neutrophils** - May indicate bacterial infection or inflammation")
        elif neutrophil_pct < 40:
            lines.append("⚠️ **Low Neutrophils** - May indicate viral infection or bone marrow issues")
        
        if lymphocyte_pct > 40:
            lines.append("⚠️ **Elevated Lymphocytes** - May indicate viral infection or lymphocytic disorder")
        
        eosinophil_pct = results['differential'].get("EOSINOPHIL", {}).get("percentage", 0)
        if eosinophil_pct > 5:
            lines.append("⚠️ **Elevated Eosinophils** - May indicate parasitic infection or allergic condition")
    else:
        lines.extend([
            "---",
            "",
            "## ℹ️ Note",
            "",
            "No WBCs detected in this image. This could be because:",
            "- The field of view doesn't contain WBCs",
            "- The image characteristics differ from training data",
            "- Try adjusting the detection confidence threshold",
        ])
    
    lines.extend([
        "",
        "---",
        "",
        "*This is an AI-assisted analysis. Always verify with a trained professional.*"
    ])
    
    return "\n".join(lines)


def classify_single_cell(image):
    """Classify a single WBC crop."""
    if image is None:
        return "Please upload a cell image"
    
    init_analyzer()
    
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Classify
    subtype, conf, probs = analyzer.classify_wbc(image_bgr)
    
    # Format result
    lines = [
        f"# {subtype}",
        f"**Confidence:** {conf*100:.1f}%",
        "",
        "### All Probabilities:",
        ""
    ]
    
    for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 20)
        lines.append(f"- **{cls}**: {prob*100:.1f}% {bar}")
    
    return "\n".join(lines)


def build_interface():
    """Build the Gradio interface."""
    
    with gr.Blocks(title="Blood Cell Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🩸 Blood Cell Analyzer
        
        AI-powered blood smear analysis for cell detection and WBC subtype classification.
        
        **Models:**
        - 🔍 YOLOv8 Detection (92.7% mAP50)
        - 🧬 ResNet34 WBC Classifier (99.95% accuracy)
        """)
        
        with gr.Tabs():
            # Tab 1: Full Image Analysis
            with gr.TabItem("📷 Full Smear Analysis"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Blood Smear Image", type="numpy")
                        confidence_slider = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="Detection Confidence Threshold"
                        )
                        analyze_btn = gr.Button("🔬 Analyze", variant="primary")
                    
                    with gr.Column():
                        output_image = gr.Image(label="Analysis Result")
                
                report_output = gr.Markdown(label="Analysis Report")
                
                analyze_btn.click(
                    analyze_image,
                    inputs=[input_image, confidence_slider],
                    outputs=[output_image, report_output]
                )
            
            # Tab 2: Single Cell Classification
            with gr.TabItem("🔬 Single Cell Classification"):
                gr.Markdown("""
                Upload a cropped WBC image to classify its subtype.
                
                **Supported subtypes:** Eosinophil, Lymphocyte, Monocyte, Neutrophil
                """)
                
                with gr.Row():
                    cell_input = gr.Image(label="Upload WBC Crop", type="pil")
                    cell_output = gr.Markdown(label="Classification Result")
                
                classify_btn = gr.Button("🔍 Classify", variant="primary")
                classify_btn.click(
                    classify_single_cell,
                    inputs=[cell_input],
                    outputs=[cell_output]
                )
            
            # Tab 3: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This Tool
                
                This blood cell analyzer uses a two-stage deep learning pipeline:
                
                ### Stage 1: Cell Detection
                - **Model:** YOLOv8 nano
                - **Training Data:** BCCD Dataset (364 images)
                - **Classes:** RBC, WBC, Platelets
                - **Performance:** 92.7% mAP50
                
                ### Stage 2: WBC Classification  
                - **Model:** ResNet34 (via fast.ai)
                - **Training Data:** Kaggle Blood Cell Images (9,957 images)
                - **Classes:** Eosinophil, Lymphocyte, Monocyte, Neutrophil
                - **Performance:** 99.95% accuracy
                
                ### Limitations
                - Detection may not generalize well to images from different microscopes
                - Best results with Wright-Giemsa stained smears
                - For research/educational use only - not for clinical diagnosis
                
                ### References
                - BCCD Dataset: https://github.com/Shenggan/BCCD_Dataset
                - Kaggle Blood Cells: https://www.kaggle.com/paultimothymooney/blood-cells
                """)
        
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
