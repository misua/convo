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
from src.reports.diagnostic_summary import generate_diagnostic_summary

# Import WBC segmentation and N/C ratio analysis
try:
    from ultralytics import YOLO
    from src.inference.nucleus_extractor import NucleusExtractor
    from src.inference.nc_ratio_analyzer import NCRatioAnalyzer
    WBC_SEGMENTATION_AVAILABLE = True
except ImportError:
    WBC_SEGMENTATION_AVAILABLE = False

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

# WBC segmentation instances
wbc_model = None
nucleus_extractor = None
nc_analyzer = None


def init_wbc_segmentation():
    """Initialize WBC segmentation models if not already done."""
    global wbc_model, nucleus_extractor, nc_analyzer
    
    if not WBC_SEGMENTATION_AVAILABLE:
        return None, None, None
    
    if wbc_model is not None:
        return wbc_model, nucleus_extractor, nc_analyzer
    
    try:
        model_path = PROJECT_ROOT / "models/segmentation/wbc_nc_ratio/train/weights/best.pt"
        if not model_path.exists():
            print(f"WBC segmentation model not found at {model_path}")
            return None, None, None
        
        print("Loading WBC segmentation model...")
        wbc_model = YOLO(str(model_path))
        nucleus_extractor = NucleusExtractor(
            method='otsu',
            erosion_iterations=2  # Conservative for healthcare
        )
        nc_analyzer = NCRatioAnalyzer()
        print("WBC segmentation models loaded!")
        
        return wbc_model, nucleus_extractor, nc_analyzer
    except Exception as e:
        print(f"Error loading WBC segmentation: {e}")
        return None, None, None


def analyze_wbc_cells(image_path: str) -> dict:
    """Analyze WBC cells and calculate N/C ratios."""
    wbc_model, nucleus_extractor, nc_analyzer = init_wbc_segmentation()
    
    if wbc_model is None:
        return {'enabled': False, 'error': 'WBC segmentation not available'}
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'enabled': False, 'error': 'Failed to load image'}
        
        # Run WBC segmentation
        results = wbc_model(image, conf=0.25, iou=0.45, verbose=False)[0]
        
        if results.masks is None or len(results.masks) == 0:
            return {
                'enabled': True,
                'cells_detected': 0,
                'cells_analyzed': 0,
                'blast_candidates': [],
                'cell_types': {},
                'nc_ratios': []
            }
        
        # Process each detected cell
        blast_candidates = []
        cell_types = Counter()
        nc_ratios = []
        cells_analyzed = 0
        
        for i, (mask, box, cls) in enumerate(zip(
            results.masks.data.cpu().numpy(),
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy()
        )):
            # Get cell type
            class_id = int(cls)
            cell_type = results.names[class_id]
            cell_types[cell_type] += 1
            
            # Resize mask to image size
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            cell_mask = (mask_resized * 255).astype(np.uint8)
            
            # Extract nucleus
            try:
                nucleus_result = nucleus_extractor.extract(image, cell_mask)
                
                # Calculate N/C ratio
                nc_result = nc_analyzer.analyze_cell(
                    nucleus_mask=nucleus_result.nucleus_mask,
                    cell_mask=cell_mask,
                    cell_class=cell_type
                )
                
                cells_analyzed += 1
                nc_ratios.append(nc_result.nc_ratio)
                
                # Check if blast candidate
                if nc_result.is_suspicious or cell_type == 'blast_cell':
                    blast_candidates.append({
                        'cell_id': i + 1,
                        'type': cell_type,
                        'nc_ratio': nc_result.nc_ratio,
                        'blast_confidence': nc_result.blast_confidence,
                        'interpretation': nc_result.interpretation
                    })
            except Exception as e:
                print(f"Error processing cell {i}: {e}")
                continue
        
        return {
            'enabled': True,
            'cells_detected': len(results.masks),
            'cells_analyzed': cells_analyzed,
            'blast_candidates': blast_candidates,
            'cell_types': dict(cell_types),
            'nc_ratios': nc_ratios,
            'avg_nc_ratio': np.mean(nc_ratios) if nc_ratios else 0.0,
            'max_nc_ratio': max(nc_ratios) if nc_ratios else 0.0
        }
    except Exception as e:
        print(f"WBC analysis error: {e}")
        import traceback
        traceback.print_exc()
        return {'enabled': True, 'error': str(e)}


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
        
        # Add WBC segmentation and N/C ratio analysis
        wbc_analysis = analyze_wbc_cells(str(temp_path))
        results['wbc_segmentation'] = wbc_analysis
        
        # Transform WBC segmentation data for PDF report compatibility
        if wbc_analysis.get('enabled') and not wbc_analysis.get('error'):
            blast_candidates = wbc_analysis.get('blast_candidates', [])
            cells_analyzed = wbc_analysis.get('cells_analyzed', 0)
            
            # Calculate blast percentage
            blast_pct = 0.0
            if cells_analyzed > 0:
                blast_pct = (len(blast_candidates) / cells_analyzed) * 100
            
            # Determine risk level based on blast count
            if len(blast_candidates) > 0:
                max_nc = max([b['nc_ratio'] for b in blast_candidates]) if blast_candidates else 0.0
                if max_nc >= 0.85:
                    risk_level = "URGENT"
                    clinical_action = "⚠️ URGENT: Blast cells detected. Immediate hematology referral required for bone marrow biopsy, flow cytometry, and cytogenetics."
                elif max_nc >= 0.80:
                    risk_level = "HIGH"
                    clinical_action = "⚠️ HIGH RISK: Suspicious cells with high N/C ratios. Urgent hematology consultation recommended."
                else:
                    risk_level = "MODERATE"
                    clinical_action = "Borderline N/C ratios detected. Clinical correlation and follow-up recommended."
            else:
                risk_level = "NEGATIVE"
                clinical_action = "No blast-like cells detected. N/C ratios within normal limits."
            
            # Create nc_ratio_analysis dict for PDF report
            results['nc_ratio_analysis'] = {
                'total_cells_analyzed': cells_analyzed,
                'blast_suspects_count': len(blast_candidates),
                'blast_cell_percentage': blast_pct,
                'average_nc_ratio': wbc_analysis.get('avg_nc_ratio', 0.0),
                'max_nc_ratio': wbc_analysis.get('max_nc_ratio', 0.0),
                'risk_level': risk_level,
                'clinical_action': clinical_action,
                'cell_types': wbc_analysis.get('cell_types', {}),
                'blast_details': blast_candidates
            }
        
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
            wbc_count = results.get('cell_counts', {}).get('WBC', 0)
            gradcam_stats = results.get('gradcam', {})
            heatmaps_generated = gradcam_stats.get('cells_with_heatmaps', 0)
            
            if heatmaps:
                gradcam_gallery = create_cell_heatmap_gallery(heatmaps, cols=4)
            else:
                # Create detailed placeholder with diagnostic information
                import PIL.Image
                import PIL.ImageDraw
                import PIL.ImageFont
                
                placeholder = PIL.Image.new('RGB', (900, 320), color=(250, 250, 250))
                draw = PIL.ImageDraw.Draw(placeholder)
                
                # Try to use a better font
                try:
                    title_font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                    body_font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                except:
                    title_font = PIL.ImageFont.load_default()
                    body_font = PIL.ImageFont.load_default()
                
                # Header bar
                draw.rectangle([(0, 0), (900, 50)], fill=(41, 128, 185))
                draw.text((30, 15), "🔬 Grad-CAM Explainability Heatmaps", fill=(255, 255, 255), font=title_font)
                
                # Status section
                y_pos = 70
                if wbc_count == 0:
                    draw.rectangle([(20, y_pos), (880, y_pos + 60)], outline=(231, 76, 60), width=2)
                    draw.text((30, y_pos + 10), "❌ No WBCs Detected", fill=(231, 76, 60), font=body_font)
                    draw.text((30, y_pos + 35), "Grad-CAM requires WBC detections to generate explanations", fill=(80, 80, 80), font=body_font)
                elif not gradcam_stats.get('enabled', False):
                    draw.rectangle([(20, y_pos), (880, y_pos + 60)], outline=(241, 196, 15), width=2)
                    draw.text((30, y_pos + 10), "⚙️ Grad-CAM Disabled", fill=(241, 196, 15), font=body_font)
                    draw.text((30, y_pos + 35), "Enable in config.yaml to see AI decision heatmaps", fill=(80, 80, 80), font=body_font)
                else:
                    draw.rectangle([(20, y_pos), (880, y_pos + 80)], outline=(231, 76, 60), width=2)
                    draw.text((30, y_pos + 10), f"⚠️ Heatmap Generation Failed ({heatmaps_generated}/{wbc_count} successful)", fill=(231, 76, 60), font=body_font)
                    draw.text((30, y_pos + 35), "• Possible causes: Small cells, poor image quality, low confidence", fill=(80, 80, 80), font=body_font)
                    draw.text((30, y_pos + 55), "• Try: Higher resolution images, better focus, more WBC-rich field", fill=(80, 80, 80), font=body_font)
                
                # Info section
                y_pos = 180
                draw.rectangle([(20, y_pos), (880, y_pos + 120)], fill=(236, 240, 241))
                draw.text((30, y_pos + 10), "💡 What is Grad-CAM?", fill=(41, 128, 185), font=title_font)
                draw.text((30, y_pos + 40), "Grad-CAM highlights which parts of the cell the AI focused on when making", fill=(60, 60, 60), font=body_font)
                draw.text((30, y_pos + 60), "its classification decision. Doctors use this to validate AI reasoning and", fill=(60, 60, 60), font=body_font)
                draw.text((30, y_pos + 80), "ensure the model is looking at clinically relevant features (nucleus, cytoplasm).", fill=(60, 60, 60), font=body_font)
                
                gradcam_gallery = np.array(placeholder)
        elif show_gradcam and not GRADCAM_VIS_AVAILABLE:
            # Grad-CAM module not available
            import PIL.Image
            import PIL.ImageDraw
            import PIL.ImageFont
            
            placeholder = PIL.Image.new('RGB', (900, 250), color=(250, 250, 250))
            draw = PIL.ImageDraw.Draw(placeholder)
            
            try:
                title_font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                body_font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                title_font = PIL.ImageFont.load_default()
                body_font = PIL.ImageFont.load_default()
            
            # Header
            draw.rectangle([(0, 0), (900, 50)], fill=(192, 57, 43))
            draw.text((30, 15), "🔬 Grad-CAM Module Not Available", fill=(255, 255, 255), font=title_font)
            
            # Content
            y_pos = 70
            draw.rectangle([(20, y_pos), (880, y_pos + 80)], outline=(192, 57, 43), width=2)
            draw.text((30, y_pos + 10), "❌ Missing Dependencies", fill=(192, 57, 43), font=title_font)
            draw.text((30, y_pos + 40), "Install with: pip install grad-cam opencv-python", fill=(80, 80, 80), font=body_font)
            
            # Info
            y_pos = 170
            draw.rectangle([(20, y_pos), (880, y_pos + 70)], fill=(236, 240, 241))
            draw.text((30, y_pos + 10), "Grad-CAM provides explainability heatmaps showing which regions", fill=(60, 60, 60), font=body_font)
            draw.text((30, y_pos + 35), "the AI focused on for classification decisions, helping doctors", fill=(60, 60, 60), font=body_font)
            
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
        
        # Get structure breakdown
        structure_types = data.get('structure_types', {})
        structure_lines = []
        for stype, count in structure_types.items():
            if count > 0:
                type_emoji = {
                    'ring': '🔴',
                    'trophozoite': '🔺',
                    'schizont': '🔵',
                    'gametocyte': '🟬'
                }.get(stype, '⬤')
                structure_lines.append(f"  - {type_emoji} {stype.title()}-like: {count}")
        
        lines.extend([
            "# 🔬 Malaria Screening Analysis",
            "",
            "## ⚠️ CANDIDATE STRUCTURES DETECTED - EXPERT REVIEW REQUIRED",
            "",
            f"### Detection Summary:",
            f"- **Detection Rate:** {data['detection_rate']:.2f}% of analyzed cells",
            f"- **Candidate Structures:** {data['candidate_structures']}",
            f"- **Dominant Type:** {data.get('dominant_type', 'unknown').title() if data.get('dominant_type') else 'Mixed'}",
            "",
            "### Structure Type Distribution:",
            *structure_lines,
            "",
            "### Interpretation:",
            data['interpretation'],
            "",
            "---",
            "",
            "### ⚠️ CRITICAL: This is NOT a Diagnosis",
            "- **Method:** YOLOv11n Object Detection (automated screening)",
            "- **Purpose:** Screening assistance only - NOT diagnostic",
            "- **Required:** Thick and thin blood smear microscopy by trained microscopist",
            "- **Action:** Detected structures require expert microscopic confirmation",
            "",
            "---",
            ""
        ])
    elif malaria_detection.get('enabled'):
        lines.extend([
            "## 🔬 Malaria Object Detection Analysis",
            "No parasite-like structures detected in analyzed regions.",
            "",
            "**Method:** YOLOv11n Object Detection (conf=0.55)",
            "",
            "⚠️ **Note:** Absence of detection does not rule out infection.",
            "Consider thick/thin smear microscopy if clinical suspicion exists.",
            "",
            "---",
            ""
        ])
    else:
        lines.extend([
            "## 🔬 Malaria Detection: Not Available",
            "Malaria detection module disabled or model not found.",
            "",
            "---",
            ""
        ])
        
    # Sickle Cell
    if 'sickle_cell' in multi_disorder_risks:
        has_findings = True
        data = multi_disorder_risks['sickle_cell']
        lines.extend([
            "# 🌙 Sickle-Shaped Cells Detected",
            "",
            f"## ⚠️ FINDING REQUIRES CLINICAL EVALUATION",
            f"- **Sickle-Shaped Cells:** {data['percentage']:.1f}% ({data['sickle_cells']} cells)",
            "",
            "**What This Means:**",
            data['interpretation'],
            "",
            "**⚠️ Important:** Sickle cell morphology requires confirmation with:",
            "- Hemoglobin electrophoresis (definitive test)",
            "- Sickle solubility test",
            "- Clinical correlation and family history",
            "",
            "---",
            ""
        ])
    else:
        lines.extend([
            "## 🌙 Sickle Cell Analysis",
            "No sickle-shaped cells detected in analyzed regions.",
            "",
            "---",
            ""
        ])

    # Thalassemia
    if 'thalassemia' in multi_disorder_risks:
        has_findings = True
        data = multi_disorder_risks['thalassemia']
        lines.extend([
            "# 🎯 Thalassemia Indicators Detected",
            "",
            "## ⚠️ MORPHOLOGICAL FINDINGS - FOLLOW-UP RECOMMENDED",
            "",
            f"### Detection Summary:",
            f"- **Indicator Cells:** {data['percentage']:.1f}% (Target cells, Teardrops)",
            f"- **Cell Count:** {data['indicators']} indicator cells identified",
            "",
            "### What This Means:",
            data['interpretation'],
            "",
            "### Recommended Actions:",
            "- Complete blood count (CBC) with red cell indices",
            "- Hemoglobin electrophoresis for definitive diagnosis",
            "- Iron studies to rule out iron deficiency",
            "- Family history evaluation",
            "",
            "---",
            ""
        ])
    else:
        lines.extend([
            "## 🎯 Thalassemia Analysis",
            "No significant thalassemia morphological indicators detected.",
            "",
            "---",
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


def generate_diagnostic_summary_pdf(patient_name, patient_id):
    """Generate 1-page diagnostic summary PDF."""
    global last_results
    
    if last_results is None:
        return None
    
    try:
        # Create output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "data/reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = output_dir / f"diagnostic_summary_{timestamp}.pdf"
        
        # Patient info
        patient_info = None
        if patient_name or patient_id:
            patient_info = {
                'name': patient_name if patient_name else 'Anonymous',
                'id': patient_id if patient_id else 'N/A',
            }
        
        # Generate diagnostic summary
        generate_diagnostic_summary(
            results=last_results,
            output_path=str(pdf_path),
            patient_info=patient_info
        )
        
        return str(pdf_path)
    except Exception as e:
        print(f"Diagnostic summary generation error: {e}")
        import traceback
        traceback.print_exc()
        return None


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
            "# 🔴 RBC Shape Analysis (Multi-Disorder Screening)",
            "",
        ])
        
        # Add findings for malaria and sickle cell
        multi_disorder_risks = results.get('multi_disorder_risks', {})
        if 'malaria' in multi_disorder_risks:
            malaria_data = multi_disorder_risks['malaria']
            
            # Get structure types
            structure_types = malaria_data.get('structure_types', {})
            structure_text = ', '.join([f"{count} {stype}-like" for stype, count in structure_types.items() if count > 0])
            
            lines.extend([
                "## 🔬 Candidate Structures Detected (Malaria Screening)",
                "",
                f"### Detection Summary:",
                f"**Detection Rate:** {malaria_data['detection_rate']:.2f}% ({malaria_data['candidate_structures']} structures)",
                f"**Structure Types:** {structure_text or 'Mixed types'}",
                f"**Dominant Type:** {malaria_data.get('dominant_type', 'unknown').title() if malaria_data.get('dominant_type') else 'Mixed'}",
                "",
                "**⚠️ EXPERT REVIEW RECOMMENDED:**",
                "- Confirmation via thick and thin blood smear microscopy required",
                "- Consider Rapid Diagnostic Test (RDT) for clinical validation",
                "- Results are for screening assistance only, not diagnostic",
                "",
            ])
        
        if 'sickle_cell' in multi_disorder_risks:
            sickle_data = multi_disorder_risks['sickle_cell']
            lines.extend([
                "## 🌙 SICKLE-SHAPED CELLS DETECTED",
                "",
                f"### Finding Details:",
                f"**Sickle-Shaped Cells:** {sickle_data['percentage']:.1f}% ({sickle_data['sickle_cells']} cells)",
                "",
                "**Recommended Follow-up:**",
                "- Hemoglobin electrophoresis for definitive diagnosis",
                "- Sickle cell solubility test",
                "- Clinical correlation and genetic counseling",
                "",
            ])
        
        lines.extend([
            "## Shape Distribution & Clinical Significance",
            "",
            "| Shape | Count | Your % | Normal Range | Clinical Significance |",
            "|-------|-------|--------|--------------|----------------------|",
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
        
        # Reference ranges and clinical significance
        shape_references = {
            "normal": {"range": ">80%", "significance": "Healthy, normal red blood cells"},
            "microcyte": {"range": "<5%", "significance": "Small cells - may indicate iron deficiency/thalassemia"},
            "target": {"range": "<5%", "significance": "Bull's-eye appearance - associated with thalassemia, liver disease"},
            "teardrop": {"range": "<2%", "significance": "Tear-shaped cells - may indicate bone marrow disorders, thalassemia"},
            "spherocyte": {"range": "<1%", "significance": "Round, dense cells - hereditary spherocytosis or immune hemolysis"},
            "irregular": {"range": "<5%", "significance": "Various irregular shapes - requires further evaluation"},
            "ring": {"range": "0%", "significance": "⚠️ Ring-like structures - requires microscopic confirmation"},
            "trophozoite": {"range": "0%", "significance": "⚠️ Irregular structures - requires expert review"},
            "sickle": {"range": "0%", "significance": "⚠️ Sickle-shaped - requires hemoglobin electrophoresis"}
        }
        
        shape_dist = shape_analysis['shape_distribution']
        shape_counts = shape_analysis.get('shape_counts', {})
        
        for shape, pct in sorted(shape_dist.items(), key=lambda x: x[1], reverse=True):
            if pct > 0:
                count = shape_counts.get(shape, 0)
                icon = shape_icons.get(shape, "⬜")
                ref = shape_references.get(shape, {"range": "N/A", "significance": "Unknown"})
                
                # Add status indicator
                status = ""
                if shape == "normal" and pct < 80:
                    status = " ⚠️"
                elif shape in ["target", "teardrop"] and pct > 5:
                    status = " ⚠️"
                elif shape in ["ring", "trophozoite", "sickle"] and pct > 0:
                    status = " ⚠️"
                elif shape in ["microcyte", "irregular"] and pct > 5:
                    status = " ⚠️"
                
                lines.append(f"| {icon} {shape.capitalize()} | {count} | {pct:.1f}%{status} | {ref['range']} | {ref['significance']} |")
        
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
            "| Subtype | Count | Percentage | Normal Range | Interpretation |",
            "|---------|-------|------------|--------------|----------------|",
        ])
        
        # Reference ranges for WBC differential (adult values)
        reference_ranges = {
            "NEUTROPHIL": {
                "range": "40-70%",
                "icon": "🟣",
                "get_interpretation": lambda pct: 
                    "✅ Normal neutrophil levels" if 40 <= pct <= 70 
                    else "⚠️ High (bacterial infection/inflammation)" if pct > 70
                    else "⚠️ Low (viral infection/bone marrow issue)"
            },
            "LYMPHOCYTE": {
                "range": "20-40%",
                "icon": "🟢",
                "get_interpretation": lambda pct:
                    "✅ Normal lymphocyte levels" if 20 <= pct <= 40
                    else "⚠️ High (viral infection/lymphocytic disorder)" if pct > 40
                    else "⚠️ Low (immunodeficiency/stress)"
            },
            "MONOCYTE": {
                "range": "2-10%",
                "icon": "🔵",
                "get_interpretation": lambda pct:
                    "✅ Normal monocyte levels" if 2 <= pct <= 10
                    else "⚠️ High (chronic infection/inflammation)" if pct > 10
                    else "⚠️ Low (bone marrow disorder)"
            },
            "EOSINOPHIL": {
                "range": "1-5%",
                "icon": "🟠",
                "get_interpretation": lambda pct:
                    "✅ Normal eosinophil levels" if 1 <= pct <= 5
                    else "⚠️ High (parasites/allergies/asthma)" if pct > 5
                    else "✅ Low (normal variation)"
            },
            "BASOPHIL": {
                "range": "0-2%",
                "icon": "🔴",
                "get_interpretation": lambda pct:
                    "✅ Normal basophil levels" if pct <= 2
                    else "⚠️ High (allergic reaction/blood disorder)"
            }
        }
        
        for subtype in ["NEUTROPHIL", "LYMPHOCYTE", "MONOCYTE", "EOSINOPHIL", "BASOPHIL"]:
            data = results['differential'].get(subtype, {"count": 0, "percentage": 0})
            ref = reference_ranges[subtype]
            icon = ref["icon"]
            pct = data['percentage']
            interpretation = ref["get_interpretation"](pct)
            lines.append(f"| {icon} {subtype.capitalize()} | {data['count']} | {pct:.1f}% | {ref['range']} | {interpretation} |")
        
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
    
    # Add WBC Segmentation and N/C Ratio Analysis
    wbc_seg = results.get('wbc_segmentation', {})
    if wbc_seg.get('enabled') and not wbc_seg.get('error'):
        lines.extend([
            "",
            "---",
            "",
            "## 🧬 WBC N/C Ratio Analysis (Leukemia Screening)",
            "",
        ])
        
        if wbc_seg.get('cells_detected', 0) > 0:
            lines.extend([
                f"**Cells Analyzed:** {wbc_seg['cells_analyzed']}/{wbc_seg['cells_detected']} cells",
                f"**Average N/C Ratio:** {wbc_seg.get('avg_nc_ratio', 0):.3f}",
                f"**Maximum N/C Ratio:** {wbc_seg.get('max_nc_ratio', 0):.3f}",
                "",
            ])
            
            # Show cell type distribution
            if wbc_seg.get('cell_types'):
                lines.append("### Cell Type Distribution")
                lines.append("")
                for cell_type, count in sorted(wbc_seg['cell_types'].items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {cell_type.replace('_', ' ').title()}: {count}")
                lines.append("")
            
            # Blast cell alerts
            blast_candidates = wbc_seg.get('blast_candidates', [])
            if blast_candidates:
                lines.extend([
                    "### ⚠️ BLAST CELL ALERTS - URGENT HEMATOLOGY REVIEW REQUIRED",
                    "",
                    f"**{len(blast_candidates)} cell(s) flagged for high N/C ratio**",
                    "",
                    "| Cell ID | Type | N/C Ratio | Confidence | Interpretation |",
                    "|---------|------|-----------|------------|----------------|",
                ])
                
                for candidate in blast_candidates[:10]:  # Show max 10
                    cell_id = candidate['cell_id']
                    cell_type = candidate['type'].replace('_', ' ').title()
                    nc_ratio = candidate['nc_ratio']
                    blast_conf = candidate['blast_confidence']
                    interpretation = candidate['interpretation']
                    
                    # Get risk emoji
                    if nc_ratio >= 0.85:
                        risk_emoji = "🔴"
                    elif nc_ratio >= 0.80:
                        risk_emoji = "🟠"
                    else:
                        risk_emoji = "🟡"
                    
                    lines.append(f"| {risk_emoji} Cell #{cell_id} | {cell_type} | {nc_ratio:.3f} | {blast_conf:.1f}% | {interpretation} |")
                
                lines.extend([
                    "",
                    "**⚠️ CRITICAL - IMMEDIATE ACTION REQUIRED:**",
                    "- Elevated N/C ratios (>0.80) are strongly associated with blast cells",
                    "- Blast cells indicate acute leukemia or other hematologic malignancies",
                    "- **Urgent referral to hematology/oncology required**",
                    "- Confirmatory testing: Bone marrow biopsy, flow cytometry, cytogenetics",
                    "- DO NOT DELAY - Early detection improves treatment outcomes",
                    "",
                ])
            else:
                lines.extend([
                    "### ✅ No Blast Cells Detected",
                    "",
                    "All analyzed cells show normal N/C ratios (<0.80) consistent with mature white blood cells.",
                    "",
                ])
            
            # Clinical reference
            lines.extend([
                "### Clinical Reference Ranges (N/C Ratio):",
                "",
                "| Cell Type | Normal N/C Range | Interpretation |",
                "|-----------|------------------|----------------|",
                "| Neutrophils | 0.25 - 0.45 | Segmented nucleus, abundant cytoplasm |",
                "| Lymphocytes | 0.65 - 0.80 | High N/C is normal for lymphocytes |",
                "| Monocytes | 0.35 - 0.50 | Large cell with kidney-shaped nucleus |",
                "| Blast Cells | **>0.80** | 🔴 **Immature cells with huge nuclei** |",
                "",
                "**About N/C Ratio:** The nucleus-to-cytoplasm ratio is a gold standard biomarker",
                "for identifying blast cells in acute leukemia. High ratios (>0.80) indicate",
                "immature cells with abnormally large nuclei - a hallmark of leukemic blasts.",
                "",
            ])
        else:
            lines.extend([
                "No WBC cells detected for N/C ratio analysis.",
                "",
            ])
    elif wbc_seg.get('error'):
        lines.extend([
            "",
            "---",
            "",
            "## 🧬 WBC N/C Ratio Analysis",
            "",
            f"⚠️ Analysis not available: {wbc_seg['error']}",
            "",
        ])
    
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
        
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 15px 0;">
        <strong>⚠️ IMPORTANT:</strong> If you're seeing old cached results or conflicting malaria detection data, 
        <strong>re-upload your image</strong> to get fresh YOLO-based analysis. 
        See <a href="docs/IMPORTANT_TEST_RESULTS_UPDATE.md">this document</a> for details on the recent malaria detection upgrade.
        </div>
        
        | Component | Model | Accuracy | Training Dataset |
        |-----------|-------|----------|------------------|
        | 🔍 Cell Detection | YOLOv11n | **92.8% mAP50** | BCCD Dataset (364 images, 4,888 cells) |
        | 🎭 Instance Segmentation | YOLOv11n-seg | **98.2% mAP50** | BCCD + Masks (1,209 images) |
        | 🧬 WBC Classification | ResNet34 | **97.9% accuracy** | Raabin-WBC (10,175 cells, 5 classes) |
        | 🦠 Malaria Detection | **YOLOv11n (NEW)** | **69.5% mAP50** | BBBC041 (1,328 images, 86K cells) |
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
                                
                                with gr.Row():
                                    generate_summary_btn = gr.Button("⭐ Generate 1-Page Diagnostic Summary", variant="primary")
                                    regenerate_btn = gr.Button("🔄 Regenerate Detailed Report", variant="secondary")
                            
                            with gr.Column():
                                diagnostic_summary_output = gr.File(label="⭐ 1-Page Diagnostic Summary", file_types=[".pdf"])
                                pdf_output = gr.File(label="📥 Detailed Report (Multi-Page)", file_types=[".pdf"])
                                gr.Markdown("""
                                **Note:** Detailed report is automatically generated after analysis.
                                
                                **1-Page Diagnostic Summary** includes:
                                - 🦠 Parasitic Infections (Malaria)
                                - 🔴 Hemoglobinopathies (Sickle Cell, Thalassemia)
                                - 📊 Anemia Screening
                                - 🔬 Infection Indicators (WBC Differential)
                                - ⚕️ Leukemia Screening
                                """)
                        
                        generate_summary_btn.click(
                            generate_diagnostic_summary_pdf,
                            inputs=[patient_name, patient_id],
                            outputs=[diagnostic_summary_output]
                        )
                        
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
