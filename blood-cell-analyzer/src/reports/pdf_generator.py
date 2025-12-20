#!/usr/bin/env python3
"""
PDF Report Generator for Blood Smear Analysis
Creates professional, printable clinical reports
"""

import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart


class BloodSmearPDFReport:
    """Generate professional PDF reports for blood smear analysis."""
    
    def __init__(self, page_size=A4):
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Create custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a5f7a')
        ))
        
        # Subtitle
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            alignment=TA_CENTER,
            textColor=colors.gray
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#2c3e50'),
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5
        ))
        
        # Subsection header
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=5,
            textColor=colors.HexColor('#34495e')
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY
        ))
        
        # Risk level styles
        self.styles.add(ParagraphStyle(
            name='RiskLow',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#27ae60'),
            alignment=TA_CENTER,
            spaceAfter=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskMedium',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#f39c12'),
            alignment=TA_CENTER,
            spaceAfter=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskHigh',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#e74c3c'),
            alignment=TA_CENTER,
            spaceAfter=5
        ))
        
        # Disclaimer style
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER,
            spaceBefore=20
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER
        ))
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: str,
        image_path: Optional[str] = None,
        analyzed_image_path: Optional[str] = None,
        patient_info: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a comprehensive PDF report.
        
        Args:
            results: Analysis results dict from BloodSmearAnalyzer
            output_path: Where to save the PDF
            image_path: Path to original image (optional)
            analyzed_image_path: Path to annotated image (optional)
            patient_info: Optional patient info dict
            
        Returns:
            Path to generated PDF
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        story = []
        
        # Build report sections
        story.extend(self._build_header(results, patient_info))
        story.extend(self._build_methodology())  # Add AI methodology section
        story.extend(self._build_sample_info(results))
        
        # Add analyzed image if available
        if analyzed_image_path and Path(analyzed_image_path).exists():
            story.extend(self._build_image_section(analyzed_image_path))
        
        story.extend(self._build_cell_counts(results))
        story.extend(self._build_wbc_differential(results))
        story.extend(self._build_shape_analysis(results))
        story.extend(self._build_risk_assessment(results))
        story.extend(self._build_clinical_notes(results))
        story.extend(self._build_footer())
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _build_header(self, results: Dict, patient_info: Optional[Dict]) -> list:
        """Build report header with title and patient info."""
        elements = []
        
        # Main title
        elements.append(Paragraph(
            "🩸 Blood Smear Analysis Report",
            self.styles['ReportTitle']
        ))
        
        elements.append(Paragraph(
            "AI-Assisted Morphological Analysis",
            self.styles['ReportSubtitle']
        ))
        
        elements.append(Spacer(1, 10))
        
        # Report info table
        report_date = datetime.now().strftime('%B %d, %Y at %H:%M')
        
        info_data = [
            ['Report Date:', report_date],
            ['Report ID:', datetime.now().strftime('%Y%m%d%H%M%S')],
        ]
        
        if patient_info:
            if patient_info.get('name'):
                info_data.append(['Patient Name:', patient_info['name']])
            if patient_info.get('id'):
                info_data.append(['Patient ID:', patient_info['id']])
            if patient_info.get('dob'):
                info_data.append(['Date of Birth:', patient_info['dob']])
        
        info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(info_table)
        
        elements.append(Spacer(1, 15))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#3498db')))
        
        return elements
    
    def _build_methodology(self) -> list:
        """Build AI methodology and datasets section."""
        elements = []
        
        elements.append(Paragraph("AI Analysis Methodology", self.styles['SectionHeader']))
        
        elements.append(Paragraph(
            "This report was generated using validated machine learning models trained on "
            "peer-reviewed medical imaging datasets. The following table summarizes the "
            "AI components and their performance metrics:",
            self.styles['ReportBody']
        ))
        elements.append(Spacer(1, 8))
        
        # Methodology table
        method_data = [
            ['Component', 'Model', 'Accuracy', 'Training Dataset'],
            ['Cell Detection', 'YOLOv11n', '92.8% mAP50', 'BCCD (364 images, 4,888 cells)'],
            ['Instance Segmentation', 'YOLOv11n-seg', '98.2% mAP50', 'BCCD + Masks (1,209 images)'],
            ['WBC Classification', 'ResNet34', '97.9% accuracy', 'Raabin-WBC (10,175 cells)'],
            ['RBC Shape Analysis', 'Morphometry', 'Rule-based', 'Clinical thresholds'],
        ]
        
        method_table = Table(method_data, colWidths=[1.4*inch, 1.2*inch, 1.1*inch, 2.1*inch])
        method_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Body
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            
            # Alternating rows
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#f8f9fa')),
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#f8f9fa')),
        ]))
        elements.append(method_table)
        elements.append(Spacer(1, 8))
        
        # Dataset references
        elements.append(Paragraph(
            "<b>Dataset References:</b><br/>"
            "• <b>BCCD:</b> Blood Cell Count Dataset - Shenggan/BCCD_Dataset (GitHub)<br/>"
            "• <b>Raabin-WBC:</b> White blood cell dataset with expert annotations (Kaggle)",
            ParagraphStyle('DatasetRef', parent=self.styles['Normal'],
                          fontSize=8, textColor=colors.HexColor('#555555'), spaceAfter=10)
        ))
        
        return elements
    
    def _build_sample_info(self, results: Dict) -> list:
        """Build sample information section."""
        elements = []
        
        elements.append(Paragraph("Sample Information", self.styles['SectionHeader']))
        
        image_name = Path(results.get('image_path', 'Unknown')).name
        image_size = results.get('image_size', {})
        
        info_data = [
            ['Image File:', image_name],
            ['Image Dimensions:', f"{image_size.get('width', 0)} × {image_size.get('height', 0)} pixels"],
            ['Analysis Timestamp:', results.get('timestamp', datetime.now().isoformat())],
            ['Analysis Method:', results.get('shape_analysis', {}).get('method', 'bbox_approximation')],
        ]
        
        info_table = Table(info_data, colWidths=[1.8*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#555555')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 10))
        
        return elements
    
    def _build_image_section(self, image_path: str) -> list:
        """Add the analyzed image to the report."""
        elements = []
        
        elements.append(Paragraph("Analyzed Image", self.styles['SectionHeader']))
        
        try:
            # Calculate image dimensions to fit page
            img = RLImage(image_path)
            max_width = 6 * inch
            max_height = 3.5 * inch
            
            # Scale proportionally
            aspect = img.imageWidth / img.imageHeight
            if aspect > max_width / max_height:
                img.drawWidth = max_width
                img.drawHeight = max_width / aspect
            else:
                img.drawHeight = max_height
                img.drawWidth = max_height * aspect
            
            # Center the image
            img.hAlign = 'CENTER'
            elements.append(img)
            elements.append(Spacer(1, 5))
            elements.append(Paragraph(
                "<i>Annotated blood smear showing detected cells</i>",
                ParagraphStyle('ImageCaption', parent=self.styles['Normal'], 
                              fontSize=9, alignment=TA_CENTER, textColor=colors.gray)
            ))
        except Exception as e:
            elements.append(Paragraph(f"<i>Image could not be loaded: {e}</i>", self.styles['ReportBody']))
        
        elements.append(Spacer(1, 10))
        return elements
    
    def _build_cell_counts(self, results: Dict) -> list:
        """Build cell count summary section."""
        elements = []
        
        elements.append(Paragraph("Cell Count Summary", self.styles['SectionHeader']))
        
        counts = results.get('cell_counts', {})
        total = sum(counts.values())
        
        # Create table with cell counts
        count_data = [
            ['Cell Type', 'Count', 'Visual'],
            ['Red Blood Cells (RBC)', str(counts.get('RBC', 0)), self._make_bar(counts.get('RBC', 0), total)],
            ['White Blood Cells (WBC)', str(counts.get('WBC', 0)), self._make_bar(counts.get('WBC', 0), total)],
            ['Platelets', str(counts.get('Platelets', 0)), self._make_bar(counts.get('Platelets', 0), total)],
            ['TOTAL', str(total), ''],
        ]
        
        count_table = Table(count_data, colWidths=[2.5*inch, 1*inch, 2.5*inch])
        count_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Body
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            
            # Total row
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(count_table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _make_bar(self, count: int, total: int) -> str:
        """Create a text-based bar for visualization."""
        if total == 0:
            return ''
        pct = count / total
        filled = int(pct * 20)
        return '█' * filled + '░' * (20 - filled) + f' ({pct*100:.1f}%)'
    
    def _build_wbc_differential(self, results: Dict) -> list:
        """Build WBC differential section."""
        elements = []
        
        differential = results.get('differential', {})
        if not differential:
            return elements
        
        elements.append(Paragraph("White Blood Cell Differential", self.styles['SectionHeader']))
        
        # Add explanation
        elements.append(Paragraph(
            "The WBC differential shows the percentage of each white blood cell type. "
            "Normal ranges are provided for reference.",
            self.styles['ReportBody']
        ))
        elements.append(Spacer(1, 5))
        
        # WBC table
        wbc_data = [
            ['WBC Type', 'Count', 'Percentage', 'Normal Range', 'Status'],
        ]
        
        normal_ranges = {
            'NEUTROPHIL': (40, 70),
            'LYMPHOCYTE': (20, 40),
            'MONOCYTE': (2, 8),
            'EOSINOPHIL': (1, 4),
            'BASOPHIL': (0, 2),
        }
        
        for subtype in ['NEUTROPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'EOSINOPHIL', 'BASOPHIL']:
            data = differential.get(subtype, {'count': 0, 'percentage': 0})
            pct = data.get('percentage', 0)
            low, high = normal_ranges[subtype]
            
            if pct < low:
                status = '↓ Low'
                status_color = colors.HexColor('#e74c3c')
            elif pct > high:
                status = '↑ High'
                status_color = colors.HexColor('#e74c3c')
            else:
                status = '✓ Normal'
                status_color = colors.HexColor('#27ae60')
            
            wbc_data.append([
                subtype.capitalize(),
                str(data.get('count', 0)),
                f"{pct:.1f}%",
                f"{low}-{high}%",
                status
            ])
        
        wbc_table = Table(wbc_data, colWidths=[1.5*inch, 0.8*inch, 1*inch, 1.2*inch, 1*inch])
        wbc_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Body alignment
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            
            # Alternating rows
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#f8f9fa')),
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#f8f9fa')),
        ]))
        elements.append(wbc_table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_shape_analysis(self, results: Dict) -> list:
        """Build RBC shape analysis section."""
        elements = []
        
        shape_analysis = results.get('shape_analysis', {})
        if not shape_analysis.get('enabled'):
            return elements
        
        elements.append(Paragraph("Red Blood Cell Morphology Analysis", self.styles['SectionHeader']))
        
        method = shape_analysis.get('method', 'bbox_approximation')
        method_text = "Instance Segmentation (pixel-accurate)" if method == 'instance_segmentation' else "Bounding Box Approximation"
        elements.append(Paragraph(
            f"<b>Analysis Method:</b> {method_text}<br/>"
            f"<b>Cells Analyzed:</b> {shape_analysis.get('cells_analyzed', 0)}",
            self.styles['ReportBody']
        ))
        elements.append(Spacer(1, 10))
        
        # Shape distribution table
        shape_dist = shape_analysis.get('shape_distribution', {})
        shape_counts = shape_analysis.get('shape_counts', {})
        
        if shape_dist:
            elements.append(Paragraph("Shape Distribution", self.styles['SubsectionHeader']))
            
            shape_data = [['Shape', 'Count', 'Percentage', 'Clinical Significance']]
            
            shape_significance = {
                'normal': 'Healthy red blood cells',
                'microcyte': 'May indicate iron deficiency or thalassemia',
                'target': 'Associated with thalassemia, liver disease',
                'teardrop': 'May indicate bone marrow disorder',
                'spherocyte': 'Hereditary spherocytosis or immune hemolysis',
                'irregular': 'Various causes, requires evaluation',
            }
            
            for shape, pct in sorted(shape_dist.items(), key=lambda x: x[1], reverse=True):
                if pct > 0:
                    count = shape_counts.get(shape, 0)
                    significance = shape_significance.get(shape, 'Unknown')
                    shape_data.append([
                        shape.capitalize(),
                        str(count),
                        f"{pct:.1f}%",
                        significance
                    ])
            
            shape_table = Table(shape_data, colWidths=[1.2*inch, 0.8*inch, 1*inch, 2.8*inch])
            shape_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (1, 1), (2, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(shape_table)
            elements.append(Spacer(1, 10))
        
        # Morphology metrics
        elements.append(Paragraph("Morphometry Measurements", self.styles['SubsectionHeader']))
        
        metrics_data = [
            ['Metric', 'Value', 'Description'],
            ['Mean Circularity', f"{shape_analysis.get('mean_circularity', 0):.3f}", 
             '1.0 = perfect circle (normal RBC ~0.9)'],
            ['Mean Elongation', f"{shape_analysis.get('mean_elongation', 0):.3f}",
             'Higher values = more elongated cells'],
            ['Size Variation (CV)', f"{shape_analysis.get('size_cv', 0):.1f}%",
             'Normal: 11-15% (RDW proxy)'],
            ['Abnormality Index', f"{shape_analysis.get('abnormality_index', 0):.2f}",
             '0 = all normal, 1 = all abnormal'],
            ['Thalassemia Indicators', f"{shape_analysis.get('thalassemia_indicator_pct', 0):.1f}%",
             'Target + Teardrop + Microcyte cells'],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1*inch, 3.3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#f8f9fa')),
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#f8f9fa')),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_risk_assessment(self, results: Dict) -> list:
        """Build thalassemia risk assessment section."""
        elements = []
        
        risk = results.get('risk_assessment', {})
        if not risk:
            return elements
        
        elements.append(Paragraph("Thalassemia Risk Assessment", self.styles['SectionHeader']))
        
        risk_level = risk.get('level', 'unknown').upper()
        risk_score = risk.get('score', 0) * 100
        
        # Risk level indicator
        if risk_level == 'LOW':
            style = self.styles['RiskLow']
            risk_color = colors.HexColor('#27ae60')
            emoji = "🟢"
        elif risk_level == 'MEDIUM':
            style = self.styles['RiskMedium']
            risk_color = colors.HexColor('#f39c12')
            emoji = "🟡"
        else:
            style = self.styles['RiskHigh']
            risk_color = colors.HexColor('#e74c3c')
            emoji = "🔴"
        
        # Risk box
        risk_data = [[f"{emoji} {risk_level} RISK", f"Score: {risk_score:.0f}%"]]
        risk_table = Table(risk_data, colWidths=[3*inch, 2*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), risk_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BOX', (0, 0), (-1, -1), 2, risk_color),
        ]))
        elements.append(risk_table)
        elements.append(Spacer(1, 10))
        
        # Interpretation
        interpretation = risk.get('interpretation', '')
        if interpretation:
            elements.append(Paragraph(
                f"<b>Interpretation:</b> {interpretation}",
                self.styles['ReportBody']
            ))
        
        elements.append(Spacer(1, 10))
        
        # Recommendation based on risk
        if risk_level == 'LOW':
            recommendation = (
                "Red blood cell morphology appears normal. No specific follow-up required "
                "based on this screening. Continue routine health monitoring."
            )
        elif risk_level == 'MEDIUM':
            recommendation = (
                "Some abnormal red cell shapes detected that may indicate thalassemia trait "
                "or other conditions. <b>Recommended:</b> Complete blood count (CBC), iron studies, "
                "and hemoglobin electrophoresis if not previously done."
            )
        else:
            recommendation = (
                "Significant red cell abnormalities detected. <b>Strongly recommended:</b> "
                "Urgent consultation with a hematologist. Laboratory tests should include CBC, "
                "reticulocyte count, iron studies, hemoglobin electrophoresis, and genetic testing if indicated."
            )
        
        elements.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", self.styles['ReportBody']))
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_clinical_notes(self, results: Dict) -> list:
        """Build clinical notes section."""
        elements = []
        
        elements.append(Paragraph("Clinical Notes", self.styles['SectionHeader']))
        
        notes = []
        
        # WBC findings
        differential = results.get('differential', {})
        if differential:
            neutrophil_pct = differential.get("NEUTROPHIL", {}).get("percentage", 0)
            lymphocyte_pct = differential.get("LYMPHOCYTE", {}).get("percentage", 0)
            eosinophil_pct = differential.get("EOSINOPHIL", {}).get("percentage", 0)
            basophil_pct = differential.get("BASOPHIL", {}).get("percentage", 0)
            
            if neutrophil_pct > 70:
                notes.append("• <b>Neutrophilia:</b> Elevated neutrophils may indicate bacterial infection, inflammation, or stress response.")
            elif neutrophil_pct < 40:
                notes.append("• <b>Neutropenia:</b> Low neutrophils may indicate viral infection, bone marrow suppression, or certain medications.")
            
            if lymphocyte_pct > 40:
                notes.append("• <b>Lymphocytosis:</b> Elevated lymphocytes may indicate viral infection or lymphoproliferative disorder.")
            
            if eosinophil_pct > 5:
                notes.append("• <b>Eosinophilia:</b> Elevated eosinophils may indicate parasitic infection, allergic condition, or drug reaction.")
            
            if basophil_pct > 2:
                notes.append("• <b>Basophilia:</b> Elevated basophils may indicate allergic reaction, infection, or myeloproliferative disorder.")
        
        # RBC findings
        shape_analysis = results.get('shape_analysis', {})
        if shape_analysis.get('enabled'):
            thal_pct = shape_analysis.get('thalassemia_indicator_pct', 0)
            if thal_pct > 30:
                notes.append(f"• <b>Abnormal RBC morphology:</b> {thal_pct:.1f}% of cells show shapes associated with thalassemia or hemoglobinopathy.")
            
            circularity = shape_analysis.get('mean_circularity', 0)
            if circularity < 0.7:
                notes.append(f"• <b>Low circularity ({circularity:.2f}):</b> Many cells deviate from normal disc shape.")
        
        if not notes:
            notes.append("• No significant abnormalities noted in this analysis.")
        
        for note in notes:
            elements.append(Paragraph(note, self.styles['ReportBody']))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _build_footer(self) -> list:
        """Build report footer with disclaimer."""
        elements = []
        
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.gray))
        elements.append(Spacer(1, 10))
        
        disclaimer = (
            "<b>IMPORTANT DISCLAIMER</b><br/><br/>"
            "This report is generated by an AI-assisted analysis tool for <b>educational and research purposes only</b>. "
            "It is <b>NOT</b> a medical diagnosis and should <b>NOT</b> be used as a substitute for professional "
            "medical advice, diagnosis, or treatment. The accuracy of AI analysis can vary based on image quality "
            "and other factors.<br/><br/>"
            "Always consult a qualified healthcare provider for interpretation of blood smear findings and "
            "any health concerns. Laboratory blood tests performed by certified professionals are required "
            "for clinical diagnosis."
        )
        
        elements.append(Paragraph(disclaimer, self.styles['Disclaimer']))
        
        elements.append(Spacer(1, 15))
        elements.append(Paragraph(
            f"Report generated by Blood Cell Analyzer | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Footer']
        ))
        
        return elements


def generate_pdf_report(
    results: Dict[str, Any],
    output_path: str,
    analyzed_image_path: Optional[str] = None,
    patient_info: Optional[Dict[str, str]] = None
) -> str:
    """
    Convenience function to generate a PDF report.
    
    Args:
        results: Analysis results from BloodSmearAnalyzer
        output_path: Where to save the PDF
        analyzed_image_path: Path to the annotated image
        patient_info: Optional dict with 'name', 'id', 'dob'
        
    Returns:
        Path to the generated PDF
    """
    generator = BloodSmearPDFReport()
    return generator.generate_report(
        results=results,
        output_path=output_path,
        analyzed_image_path=analyzed_image_path,
        patient_info=patient_info
    )


if __name__ == "__main__":
    # Test with sample data
    sample_results = {
        "image_path": "test_image.jpg",
        "image_size": {"width": 640, "height": 480},
        "timestamp": datetime.now().isoformat(),
        "cell_counts": {"RBC": 150, "WBC": 5, "Platelets": 20},
        "summary": {"total_cells": 175, "wbc_classified": 5},
        "differential": {
            "NEUTROPHIL": {"count": 3, "percentage": 60.0},
            "LYMPHOCYTE": {"count": 1, "percentage": 20.0},
            "MONOCYTE": {"count": 1, "percentage": 20.0},
            "EOSINOPHIL": {"count": 0, "percentage": 0.0},
            "BASOPHIL": {"count": 0, "percentage": 0.0},
        },
        "shape_analysis": {
            "enabled": True,
            "method": "instance_segmentation",
            "cells_analyzed": 150,
            "shape_distribution": {
                "normal": 60.0,
                "target": 20.0,
                "microcyte": 10.0,
                "teardrop": 5.0,
                "irregular": 5.0,
            },
            "shape_counts": {
                "normal": 90,
                "target": 30,
                "microcyte": 15,
                "teardrop": 8,
                "irregular": 7,
            },
            "thalassemia_indicator_pct": 35.0,
            "abnormality_index": 0.40,
            "mean_circularity": 0.82,
            "mean_elongation": 0.15,
            "size_cv": 14.5,
        },
        "risk_assessment": {
            "score": 0.45,
            "level": "medium",
            "interpretation": "Moderate thalassemia indicators detected"
        }
    }
    
    output = generate_pdf_report(
        sample_results,
        "test_report.pdf",
        patient_info={"name": "Test Patient", "id": "12345"}
    )
    print(f"Generated: {output}")
