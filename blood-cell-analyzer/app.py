#!/usr/bin/env python3
"""
Gradio Dashboard for Blood Smear Analysis
Enhanced with Shape Analysis and Plain-Language Diagnosis
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
import yaml
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent  # blood-cell-analyzer/
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_blood_smear import BloodSmearAnalyzer
from src.inference.explainer import DiagnosisExplainer, RiskLevel
from src.reports.pdf_generator import generate_pdf_report

# Import Grad-CAM visualization utilities (optional)
try:
    from src.inference.visualization import (
        create_cell_heatmap_gallery, generate_gradcam_legend
    )
    GRADCAM_VIS_AVAILABLE = True
except ImportError:
    GRADCAM_VIS_AVAILABLE = False


def load_config():
    """Load configuration from config.yaml."""
    config_path = PROJECT_ROOT / "configs/config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# Global instances
analyzer = None
explainer = None
config = load_config()
last_results = None  # Store last analysis results for PDF generation
last_vis_path = None  # Store path to visualization
gradcam_enabled = False  # Track Grad-CAM state


def init_analyzer(enable_gradcam: bool = False):
    """Initialize the analyzer if not already done."""
    global analyzer, explainer, gradcam_enabled
    
    # Re-initialize if gradcam state changed
    if analyzer is not None and gradcam_enabled == enable_gradcam:
        return analyzer, explainer
    
    # Update gradcam config
    analysis_config = config.copy()
    if 'gradcam' not in analysis_config:
        analysis_config['gradcam'] = {}
    analysis_config['gradcam']['enabled'] = enable_gradcam
    gradcam_enabled = enable_gradcam
    
    print(f"Loading models... (Grad-CAM: {enable_gradcam})")
    analyzer = BloodSmearAnalyzer(
        detection_conf=0.25,
        enable_shape_analysis=True,
        config=analysis_config
    )
    explainer = DiagnosisExplainer(config=analysis_config)
    print("Models loaded!")
    
    return analyzer, explainer


def analyze_image(image, confidence_threshold, show_gradcam=False):
    """Analyze uploaded blood smear image."""
    global last_results, last_vis_path
    
    if image is None:
        return None, "Please upload an image", "", None, None
    
    # Initialize analyzer with gradcam setting
    analyzer, explainer = init_analyzer(enable_gradcam=show_gradcam)
    analyzer.detection_conf = confidence_threshold
    
    # Save temp image
    temp_path = PROJECT_ROOT / "data/samples/temp_upload.jpg"
    Image.fromarray(image).save(temp_path)
    
    # Run analysis
    try:
        results = analyzer.analyze(str(temp_path), save_visualization=True)
        last_results = results  # Store for PDF generation
        
        # Load visualization
        vis_path = temp_path.parent / f"{temp_path.stem}_analyzed.jpg"
        last_vis_path = str(vis_path)  # Store for PDF
        vis_image = cv2.imread(str(vis_path))
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        # Format technical report
        tech_report = format_report(results)
        
        # Generate plain-language diagnosis
        shape_stats = None
        if results.get('shape_analysis', {}).get('enabled'):
            shape_stats = results['shape_analysis']
        
        diagnosis = explainer.explain_results(results, shape_stats)
        plain_report = explainer.format_as_markdown(diagnosis)
        
        # Generate PDF
        pdf_path = generate_pdf(results, last_vis_path)
        
        # Generate Grad-CAM gallery if enabled and available
        gradcam_gallery = None
        if show_gradcam and GRADCAM_VIS_AVAILABLE:
            heatmaps = results.get('_wbc_heatmaps', [])
            if heatmaps:
                gradcam_gallery = create_cell_heatmap_gallery(heatmaps, cols=4)
            elif results.get('cell_counts', {}).get('WBC', 0) == 0:
                # Create placeholder message when no WBCs found
                import PIL.Image
                import PIL.ImageDraw
                import PIL.ImageFont
                placeholder = PIL.Image.new('RGB', (400, 100), color=(45, 45, 45))
                draw = PIL.ImageDraw.Draw(placeholder)
                draw.text((20, 35), "No WBCs detected - no heatmaps to generate", fill=(200, 200, 200))
                gradcam_gallery = np.array(placeholder)
        
        # Generate pathology report
        pathology_report = format_pathology_report(results)

        return vis_image, tech_report, plain_report, pathology_report, pdf_path, gradcam_gallery
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}", "", "", None, None


def format_pathology_report(results: dict) -> str:
    """Format pathology findings as markdown."""
    lines = []
    
    multi_disorder_risks = results.get('multi_disorder_risks', {})
    
    has_findings = False
    
    # Malaria (YOLO-based detection)
    malaria_detection = results.get('malaria_detection', {})
    if malaria_detection.get('enabled') and 'malaria' in multi_disorder_risks:
        has_findings = True
        data = multi_disorder_risks['malaria']
        
        # Get parasite breakdown
        stages = data.get('parasite_stages', {})
        stage_lines = []
        for stage, count in stages.items():
            if count > 0:
                stage_emoji = {
                    'ring': '🔴',
                    'trophozoite': '🔺',
                    'schizont': '🔵',
                    'gametocyte': '🟬'
                }.get(stage, '⬤')
                stage_lines.append(f"  - {stage_emoji} {stage.title()}: {count}")
        
        lines.extend([
            "## 🦠 Malaria Screening: POSITIVE (YOLO)",
            "",
            f"**⚠️ URGENT ALERT**",
            f"- **Parasitemia:** {data['percentage']:.2f}%",
            f"- **Total Parasites:** {data['infected_cells']}",
            f"- **Dominant Stage:** {data.get('dominant_stage', 'unknown').title() if data.get('dominant_stage') else 'Mixed'}",
            "",
            "**Parasite Breakdown:**",
            *stage_lines,
            "",
            "**Interpretation:**",
            data['interpretation'],
            "",
            "**Method:** YOLOv11n Object Detection (conf=0.55)",
            "",
            "---",
            ""
        ])
    elif malaria_detection.get('enabled'):
        lines.extend([
            "## 🦠 Malaria Screening: Negative (YOLO)",
            "No malaria parasites detected in this sample.",
            "",
            "**Method:** YOLOv11n Object Detection (conf=0.55)",
            "",
            "---",
            ""
        ])
    elif 'malaria' in multi_disorder_risks:
        # Fallback for old CNN results (shouldn't happen with new code)
        has_findings = True
        data = multi_disorder_risks['malaria']
        lines.extend([
            "## 🦠 Malaria Screening: POSITIVE (Legacy CNN)",
            "",
            f"**⚠️ URGENT ALERT**",
            f"- **Parasitemia:** {data['percentage']:.2f}%",
            f"- **Infected Cells:** {data['infected_cells']}",
            "",
            "**Interpretation:**",
            data['interpretation'],
            "",
            "---",
            ""
        ])
    else:
        lines.extend([
            "## 🦠 Malaria Screening: Not Available",
            "Malaria detection disabled or model not found.",
            "",
            "---",
            ""
        ])
        
    # Sickle Cell
    if 'sickle_cell' in multi_disorder_risks:
        has_findings = True
        data = multi_disorder_risks['sickle_cell']
        lines.extend([
            "## 🌙 Sickle Cell Screening: POSITIVE",
            "",
            f"**⚠️ ABNORMAL FINDING**",
            f"- **Sickle Cells:** {data['percentage']:.1f}%",
            f"- **Count:** {data['sickle_cells']}",
            "",
            "**Interpretation:**",
            data['interpretation'],
            "",
            "---",
            ""
        ])
    else:
        lines.extend([
            "## 🌙 Sickle Cell Screening: Negative",
            "No sickle cells detected in this sample.",
            "",
            "---",
            ""
        ])

    # Thalassemia
    if 'thalassemia' in multi_disorder_risks:
        has_findings = True
        data = multi_disorder_risks['thalassemia']
        lines.extend([
            "## 🩸 Thalassemia Screening: POSITIVE",
            "",
            f"**⚠️ ABNORMAL FINDING**",
            f"- **Indicators:** {data['percentage']:.1f}% (Target cells, Teardrops)",
            "",
            "**Interpretation:**",
            data['interpretation'],
            ""
        ])
    else:
        lines.extend([
            "## 🩸 Thalassemia Screening: Negative",
            "No significant thalassemia indicators detected.",
            ""
        ])
        
    if not has_findings:
        lines.insert(0, "### ✅ No Significant Pathological Findings Detected\n")
        
    return "\n".join(lines)


def generate_pdf(results, vis_path, patient_name="", patient_id="", patient_dob=""):
    """Generate PDF report from analysis results."""
    try:
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "data/reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = output_dir / f"blood_analysis_report_{timestamp}.pdf"
        
        # Patient info
        patient_info = None
        if patient_name or patient_id:
            patient_info = {
                'name': patient_name if patient_name else None,
                'id': patient_id if patient_id else None,
                'dob': patient_dob if patient_dob else None,
            }
        
        # Generate PDF
        generate_pdf_report(
            results=results,
            output_path=str(pdf_path),
            analyzed_image_path=vis_path,
            patient_info=patient_info
        )
        
        return str(pdf_path)
    except Exception as e:
        print(f"PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        return None


def regenerate_pdf(patient_name, patient_id, patient_dob):
    """Regenerate PDF with patient information."""
    global last_results, last_vis_path
    
    if last_results is None:
        return None
    
    return generate_pdf(last_results, last_vis_path, patient_name, patient_id, patient_dob)


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
    
    # Add shape analysis section if available
    shape_analysis = results.get('shape_analysis', {})
    if shape_analysis.get('enabled') and shape_analysis.get('shape_distribution'):
        lines.extend([
            "---",
            "",
            "## 🔴 RBC Shape Analysis (Multi-Disorder Screening)",
            "",
        ])
        
        # Add urgent alerts for malaria and sickle cell
        multi_disorder_risks = results.get('multi_disorder_risks', {})
        if 'malaria' in multi_disorder_risks:
            malaria_data = multi_disorder_risks['malaria']
            
            # Get parasite stages
            stages = malaria_data.get('parasite_stages', {})
            stage_text = ', '.join([f"{count} {stage}" for stage, count in stages.items() if count > 0])
            
            lines.extend([
                "### 🚨 URGENT ALERT: MALARIA PARASITES DETECTED",
                "",
                f"**Parasitemia Level:** {malaria_data['percentage']:.2f}% ({malaria_data['infected_cells']} parasites)",
                f"**Parasite Stages:** {stage_text or 'Mixed stages'}",
                f"**Dominant Stage:** {malaria_data.get('dominant_stage', 'unknown').title() if malaria_data.get('dominant_stage') else 'Mixed'}",
                "",
                "**⚠️ IMMEDIATE ACTION REQUIRED:**",
                "- Confirm with thick and thin blood smear microscopy",
                "- Rapid Diagnostic Test (RDT) for malaria antigens",
                "- Begin appropriate antimalarial therapy if confirmed",
                "",
            ])
        
        if 'sickle_cell' in multi_disorder_risks:
            sickle_data = multi_disorder_risks['sickle_cell']
            lines.extend([
                "### ⚠️ SICKLE CELLS DETECTED",
                "",
                f"**Sickle Cell Percentage:** {sickle_data['percentage']:.1f}% ({sickle_data['sickle_cells']} cells)",
                "",
                "**Recommended Follow-up:**",
                "- Hemoglobin electrophoresis for definitive diagnosis",
                "- Sickle cell solubility test",
                "- Genetic counseling if positive",
                "",
            ])
        
        lines.extend([
            "| Shape | Count | Percentage |",
            "|-------|-------|------------|",
        ])
        
        shape_icons = {
            "normal": "⚪",
            "microcyte": "🔵",
            "target": "🎯",
            "teardrop": "💧",
            "spherocyte": "⚫",
            "irregular": "⬛",
            "ring": "🔴",  # Malaria ring stage
            "trophozoite": "🔺",  # Malaria trophozoite
            "sickle": "🌙"  # Sickle cell
        }
        
        shape_dist = shape_analysis['shape_distribution']
        shape_counts = shape_analysis.get('shape_counts', {})
        
        for shape, pct in sorted(shape_dist.items(), key=lambda x: x[1], reverse=True):
            if pct > 0:
                count = shape_counts.get(shape, 0)
                icon = shape_icons.get(shape, "⬜")
                lines.append(f"| {icon} {shape.capitalize()} | {count} | {pct:.1f}% |")
        
        # Add morphology metrics
        lines.extend([
            "",
            "### Morphology Metrics",
            "",
            f"- **Cells Analyzed:** {shape_analysis.get('cells_analyzed', 0)}",
            f"- **Abnormality Index:** {shape_analysis.get('abnormality_index', 0):.2f}",
        ])
        
        # Add disorder-specific metrics (malaria now handled by YOLO detector in pathology section)
        sickle_cells = shape_analysis.get('sickle_cells', 0)
        thal_indicators = shape_analysis.get('thalassemia_indicator_pct', 0)
        
        if sickle_cells > 0:
            sickle_pct = shape_analysis.get('sickle_cell_pct', 0)
            lines.append(f"- **Sickle Cells:** {sickle_cells} ({sickle_pct:.1f}%)")
        
        lines.append(f"- **Thalassemia Indicators:** {thal_indicators:.1f}%")
        
        lines.extend([
            f"- **Mean Circularity:** {shape_analysis.get('mean_circularity', 0):.3f}",
            f"- **Mean Elongation:** {shape_analysis.get('mean_elongation', 0):.3f}",
        ])
        
        # Add multi-disorder risk assessment
        if multi_disorder_risks:
            lines.extend([
                "",
                "### Multi-Disorder Risk Assessment",
                "",
            ])
            
            for disorder, risk_data in multi_disorder_risks.items():
                risk_level = risk_data.get('level', 'unknown')
                risk_emoji = {"urgent": "🔴", "referral": "🟠", "monitor": "🟡", "low": "🟢"}.get(risk_level, "⚪")
                disorder_name = disorder.replace('_', ' ').title()
                lines.append(f"- **{disorder_name}:** {risk_emoji} {risk_level.upper()} - {risk_data.get('interpretation', 'N/A')}")
        else:
            # Legacy thalassemia risk
            risk = results.get('risk_assessment', {})
            if risk:
                risk_level = risk.get('level', 'unknown')
                risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "⚪")
                lines.extend([
                    "",
                    "### Thalassemia Risk Assessment",
                    "",
                    f"- **Risk Level:** {risk_emoji} {risk_level.upper()}",
                    f"- **Risk Score:** {risk.get('score', 0)*100:.1f}%",
                    f"- **Interpretation:** {risk.get('interpretation', 'N/A')}",
                ])
    
    if results.get('differential') and results['summary']['wbc_classified'] > 0:
        lines.extend([
            "---",
            "",
            "## 🔬 WBC Differential",
            "",
            "| Subtype | Count | Percentage |",
            "|---------|-------|------------|",
        ])
        
        for subtype in ["NEUTROPHIL", "LYMPHOCYTE", "MONOCYTE", "EOSINOPHIL", "BASOPHIL"]:
            data = results['differential'].get(subtype, {"count": 0, "percentage": 0})
            icon = {"NEUTROPHIL": "🟣", "LYMPHOCYTE": "🟢", "MONOCYTE": "🔵", "EOSINOPHIL": "🟠", "BASOPHIL": "🔴"}.get(subtype, "⚪")
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
        
        basophil_pct = results['differential'].get("BASOPHIL", {}).get("percentage", 0)
        if basophil_pct > 2:
            lines.append("⚠️ **Elevated Basophils** - May indicate allergic reaction, infection, or myeloproliferative disorder")
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
    
    analyzer, _ = init_analyzer()
    
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
        
        AI-powered blood smear analysis with **thalassemia shape detection** and comprehensive reporting.
        
        | Component | Model | Accuracy | Training Dataset |
        |-----------|-------|----------|------------------|
        | 🔍 Cell Detection | YOLOv11n | **92.8% mAP50** | BCCD Dataset (364 images, 4,888 cells) |
        | 🎭 Instance Segmentation | YOLOv11n-seg | **98.2% mAP50** | BCCD + Masks (1,209 images) |
        | 🧬 WBC Classification | ResNet34 | **97.9% accuracy** | Raabin-WBC (10,175 cells, 5 classes) |
        | 🦠 Malaria/Sickle Cell | ResNet34 | **96.3% accuracy** | Custom Dataset (3 classes) |
        | 🔴 RBC Shape Analysis | Morphometry | Rule-based | Clinical thresholds |
        
        📄 **PDF reports available** with patient info and clinical recommendations.
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
                        gradcam_toggle = gr.Checkbox(
                            label="🔥 Show Grad-CAM Heatmaps",
                            value=False,
                            info="Visualize which regions the AI focuses on for WBC classification"
                        )
                        analyze_btn = gr.Button("🔬 Analyze", variant="primary")
                    
                    with gr.Column():
                        output_image = gr.Image(label="Analysis Result")
                
                with gr.Tabs():
                    with gr.TabItem("📊 Technical Report"):
                        report_output = gr.Markdown(label="Technical Analysis Report")
                    
                    with gr.TabItem("🩺 Patient-Friendly Report"):
                        gr.Markdown("""
                        ### Understanding Your Results
                        This report explains your blood smear analysis in plain language.
                        The traffic light system helps you understand the urgency:
                        - 🟢 **Green** = Normal, no concerns
                        - 🟡 **Yellow** = Borderline, worth monitoring  
                        - 🔴 **Red** = Abnormal, consult a doctor
                        """)
                        plain_report_output = gr.Markdown(label="Plain Language Report")
                    
                    with gr.TabItem("🦠 Pathology Screening"):
                        gr.Markdown("""
                        ### Pathology Screening Results
                        
                        This section highlights specific pathological findings for **Malaria** and **Sickle Cell Disease**.
                        
                        *Note: This is a screening tool, not a diagnostic device. All findings must be confirmed by laboratory tests.*
                        """)
                        pathology_output = gr.Markdown(label="Pathology Findings")
                    
                    with gr.TabItem("� Grad-CAM Attention"):
                        gr.Markdown("""
                        ### Model Attention Visualization (Grad-CAM)
                        
                        When enabled, this shows **which image regions** the AI model focuses on 
                        when classifying white blood cells. This helps verify the model is looking 
                        at the correct cell features.
                        
                        **Color Scale:**
                        - 🔵 **Blue** = Low attention (model ignores this region)
                        - 🟡 **Yellow** = Medium attention
                        - 🔴 **Red** = High attention (model focuses here)
                        
                        *Enable "Show Grad-CAM Heatmaps" checkbox above to generate visualizations.*
                        """)
                        gradcam_gallery = gr.Image(label="WBC Attention Heatmaps", visible=True)
                    
                    with gr.TabItem("�📄 PDF Report"):
                        gr.Markdown("""
                        ### Generate Printable PDF Report
                        Download a comprehensive PDF report with all analysis results.
                        Optionally add patient information for the report.
                        """)
                        
                        with gr.Row():
                            with gr.Column():
                                patient_name = gr.Textbox(label="Patient Name (optional)", placeholder="John Doe")
                                patient_id = gr.Textbox(label="Patient ID (optional)", placeholder="12345")
                                patient_dob = gr.Textbox(label="Date of Birth (optional)", placeholder="1990-01-15")
                                regenerate_btn = gr.Button("🔄 Regenerate PDF with Patient Info", variant="secondary")
                            
                            with gr.Column():
                                pdf_output = gr.File(label="📥 Download PDF Report", file_types=[".pdf"])
                                gr.Markdown("""
                                **Note:** PDF is automatically generated after analysis.
                                Use the button to regenerate with patient information.
                                """)
                        
                        regenerate_btn.click(
                            regenerate_pdf,
                            inputs=[patient_name, patient_id, patient_dob],
                            outputs=[pdf_output]
                        )
                
                analyze_btn.click(
                    analyze_image,
                    inputs=[input_image, confidence_slider, gradcam_toggle],
                    outputs=[output_image, report_output, plain_report_output, pathology_output, pdf_output, gradcam_gallery]
                )
            
            # Tab 2: Single Cell Classification
            with gr.TabItem("🔬 Single Cell Classification"):
                gr.Markdown("""
                Upload a cropped WBC image to classify its subtype.
                
                **Supported subtypes:** Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil
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
                # 🩸 Blood Cell Analyzer
                
                This tool uses artificial intelligence to analyze microscope images of blood smears. 
                It can detect and count different types of blood cells, classify white blood cells, 
                and identify abnormal red blood cell shapes that may indicate health conditions.
                
                ---
                
                ## 📊 Understanding Your Patient-Friendly Report
                
                The **Patient-Friendly Report** tab is designed for people without medical training. 
                Here's what each section means:
                
                ### 🚦 Traffic Light System
                
                Every finding uses a simple color code:
                
                | Color | What It Means | What To Do |
                |-------|---------------|------------|
                | 🟢 **Green** | Normal, healthy finding | No action needed |
                | 🟡 **Yellow** | Slightly outside normal range | Mention to your doctor at next visit |
                | 🔴 **Red** | Significantly abnormal | Schedule a doctor's appointment soon |
                
                ### 📋 Report Sections Explained
                
                #### 1. Cell Counts
                The report shows how many of each cell type were found:
                
                | Cell Type | What It Does | Normal Finding |
                |-----------|--------------|----------------|
                | **Red Blood Cells (RBC)** | Carry oxygen throughout your body | Should be the most numerous |
                | **White Blood Cells (WBC)** | Fight infections and disease | Much fewer than RBCs |
                | **Platelets** | Help blood clot to stop bleeding | Small fragments, varies widely |
                
                #### 2. WBC Differential (White Blood Cell Types)
                
                White blood cells come in 5 main types. Each has a different job:
                
                | WBC Type | Normal Range | What High Levels May Indicate |
                |----------|--------------|-------------------------------|
                | **Neutrophil** | 40-70% | Bacterial infection, inflammation, stress |
                | **Lymphocyte** | 20-40% | Viral infection, immune response |
                | **Monocyte** | 2-8% | Chronic infection, inflammation |
                | **Eosinophil** | 1-4% | Allergies, parasites, asthma |
                | **Basophil** | 0-2% | Allergic reactions, some blood disorders |
                
                #### 3. RBC Shape Analysis (Multi-Disorder Screening)
                
                Red blood cells should be round, disc-shaped, and similar in size. 
                Abnormal shapes can indicate health conditions:
                
                | Shape | Plain Language | What It May Mean |
                |-------|----------------|------------------|
                | **Normal** | Healthy round discs | Your red cells look healthy |
                | **Microcyte** | Cells smaller than normal | Iron deficiency or thalassemia trait |
                | **Target Cell** | Bulls-eye pattern in center | Thalassemia, liver problems, or iron deficiency |
                | **Teardrop** | Elongated like a teardrop | Bone marrow problems, severe anemia |
                | **Spherocyte** | Too round (like a ball) | Hereditary condition or immune issue |
                | **Irregular** | Oddly shaped cells | Various causes, needs doctor review |
                | **Ring** 🔴 | Thin ring inside cell | **URGENT:** Malaria parasite (early stage) |
                | **Trophozoite** 🔺 | Amoeboid form inside cell | **URGENT:** Malaria parasite (mature stage) |
                | **Sickle** 🌙 | Crescent/banana shaped | Sickle cell disease - requires confirmation |
                
                #### 4. Thalassemia Risk Assessment
                
                **What is Thalassemia?**  
                Thalassemia is an inherited blood disorder where the body makes less hemoglobin 
                (the protein in red blood cells that carries oxygen). It's common in people with 
                ancestry from the Mediterranean, Middle East, Africa, and Southeast Asia.
                
                | Risk Level | What It Means |
                |------------|---------------|
                | 🟢 **Low** | Red cell shapes look normal, unlikely to have thalassemia |
                | 🟡 **Medium** | Some abnormal shapes detected, might be thalassemia trait |
                | 🔴 **High** | Many abnormal shapes, strongly recommend blood tests |
                
                **Important:** This screening is NOT a diagnosis. Only lab blood tests 
                (hemoglobin electrophoresis, CBC, iron studies) can diagnose thalassemia.
                
                ---
                
                ## 🔬 How The AI Models Work
                
                ### Stage 1: Finding Cells — YOLOv11n
                
                | What | Details |
                |------|---------|
                | **Task** | Locate and draw boxes around every cell in the image |
                | **AI Model** | YOLOv11n — a fast object detection neural network |
                | **Accuracy** | **92.8%** of cells correctly identified |
                | **Trained On** | 364 blood smear images from BCCD Dataset |
                | **Cell Types** | Red Blood Cells, White Blood Cells, Platelets |
                
                *Per-cell accuracy: RBC 90.3% • WBC 97.5% • Platelets 90.7%*
                
                ### Stage 2: Classifying White Blood Cells — ResNet34
                
                | What | Details |
                |------|---------|
                | **Task** | Identify which type of white blood cell each one is |
                | **AI Model** | ResNet34 — a deep image classification network |
                | **Accuracy** | **97.9%** of WBCs correctly classified |
                | **Trained On** | 10,175 WBC images from Raabin-WBC Dataset |
                | **Classes** | Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil |
                
                **Training Data Breakdown (Raabin-WBC Dataset):**
                | Type | Images | % of Dataset |
                |------|--------|--------------|
                | Neutrophil | 6,231 | 61.3% |
                | Lymphocyte | 2,427 | 23.9% |
                | Eosinophil | 744 | 7.3% |
                | Monocyte | 561 | 5.5% |
                | Basophil | 212 | 2.1% |
                
                > **About Raabin-WBC:** This dataset contains microscopy images collected from 
                > multiple hospital laboratories with expert pathologist annotations. Created by 
                > researchers to advance blood cell classification AI.
                
                ### Stage 3: Analyzing Red Cell Shapes — Morphometry
                
                | What | Details |
                |------|---------|
                | **Task** | Measure shape features to identify abnormalities |
                | **Method** | Mathematical analysis of cell geometry |
                | **Measurements** | Circularity, elongation, size, central pallor |
                | **Shape Categories** | 9 classes: Normal, Microcyte, Target, Teardrop, Spherocyte, Irregular, Ring, Trophozoite, Sickle |
                | **Purpose** | Screen for thalassemia, malaria, and sickle cell disease |
                
                **Multi-Disorder Detection:**
                - **Malaria:** Ring and trophozoite stage parasites (urgent)
                - **Sickle Cell Disease:** Crescent-shaped sickled cells (referral)
                - **Thalassemia:** Microcytes, target cells, teardrops (monitoring)
                
                ---
                
                ## ⚠️ Important Limitations
                
                **This tool is for EDUCATIONAL and RESEARCH purposes only.**
                
                - ❌ **NOT a medical device** — cannot diagnose any condition
                - ❌ **NOT reviewed by FDA** or any regulatory body
                - ❌ **NOT a substitute** for laboratory blood tests or doctor consultation
                
                **Technical limitations:**
                - Accuracy varies with image quality and microscope type
                - Works best with Wright-Giemsa stained blood smears
                - May miss rare cell types or unusual morphologies
                - Shape analysis is approximate without cell segmentation
                - WBC differential needs 100+ cells for clinical significance
                
                **If you have health concerns**, please consult a qualified healthcare provider.
                
                ---
                
                ## 📚 Data Sources & References
                
                | Resource | Description |
                |----------|-------------|
                | [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset) | Blood cell detection training images |
                | [Raabin-WBC Dataset](https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset) | WBC classification training images |
                | [Ultralytics YOLOv11](https://docs.ultralytics.com/) | Object detection framework |
                | [fast.ai](https://docs.fast.ai/) | Deep learning training library |
                """)
        
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
