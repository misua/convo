#!/usr/bin/env python3
"""
PDF Report Generator for Blood Smear Analysis
Creates professional, printable clinical reports

Report Format: PROFESSIONAL AUTOMATED BLOOD ANALYSIS REPORT
"""

import io
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable, KeepTogether, ListFlowable, ListItem
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart


# Professional color scheme
COLORS = {
    'primary': colors.HexColor('#1a237e'),       # Dark blue
    'secondary': colors.HexColor('#0d47a1'),     # Medium blue  
    'accent': colors.HexColor('#1976d2'),        # Light blue
    'success': colors.HexColor('#2e7d32'),       # Green
    'warning': colors.HexColor('#f57c00'),       # Orange
    'danger': colors.HexColor('#c62828'),        # Red
    'dark': colors.HexColor('#212121'),          # Almost black
    'medium': colors.HexColor('#616161'),        # Gray
    'light': colors.HexColor('#f5f5f5'),         # Light gray
    'white': colors.white,
    'border': colors.HexColor('#e0e0e0'),
}


class BloodSmearPDFReport:
    """Generate professional PDF reports for blood smear analysis."""
    
    def __init__(self, page_size=A4):
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _generate_report_id(self) -> str:
        """Generate unique report ID in format #PH-YYYY-XXXX-A."""
        year = datetime.now().strftime('%Y')
        random_num = str(uuid.uuid4().int)[:4]
        return f"#PH-{year}-{random_num}-A"
        
    def _setup_custom_styles(self):
        """Create custom paragraph styles for professional appearance."""
        # Main title - large, bold, centered
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=22,
            spaceAfter=5,
            alignment=TA_CENTER,
            textColor=COLORS['primary'],
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=15,
            alignment=TA_CENTER,
            textColor=COLORS['medium']
        ))
        
        # Section header - blue background style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=COLORS['primary'],
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderColor=COLORS['accent'],
            borderPadding=8,
            leftIndent=0
        ))
        
        # Subsection header
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=5,
            textColor=COLORS['secondary'],
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT,
            textColor=COLORS['dark']
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            leftIndent=20,
            bulletIndent=10,
            textColor=COLORS['dark']
        ))
        
        # AI Result styles - large and prominent
        self.styles.add(ParagraphStyle(
            name='AIResultHigh',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=COLORS['danger'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='AIResultMedium',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=COLORS['warning'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='AIResultLow',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=COLORS['success'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=5
        ))
        
        # Confidence score
        self.styles.add(ParagraphStyle(
            name='ConfidenceScore',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=COLORS['dark'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Recommendation text
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=COLORS['secondary'],
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique',
            spaceAfter=10
        ))
        
        # Risk level styles
        self.styles.add(ParagraphStyle(
            name='RiskLow',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=COLORS['success'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskMedium',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=COLORS['warning'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskHigh',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=COLORS['danger'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            spaceAfter=5
        ))
        
        # Disclaimer style - small, gray
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=COLORS['medium'],
            alignment=TA_CENTER,
            spaceBefore=15,
            spaceAfter=10
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=COLORS['medium'],
            alignment=TA_CENTER
        ))
        
        # Table header cell style
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=COLORS['white'],
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))
        
        # Finding label style  
        self.styles.add(ParagraphStyle(
            name='FindingLabel',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=COLORS['dark'],
            fontName='Helvetica-Bold'
        ))
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: str,
        image_path: Optional[str] = None,
        analyzed_image_path: Optional[str] = None,
        patient_info: Optional[Dict[str, str]] = None,
        clinic_info: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a comprehensive professional PDF report.
        
        Args:
            results: Analysis results dict from BloodSmearAnalyzer
            output_path: Where to save the PDF
            image_path: Path to original image (optional)
            analyzed_image_path: Path to annotated image (optional)
            patient_info: Optional patient info dict with keys:
                - id: Patient ID
                - name: Patient name (can be "Anonymized")
                - dob: Date of birth
                - gender: Patient gender
            clinic_info: Optional clinic info dict with keys:
                - name: Clinic/lab name
                - physician: Referring physician
                - location: City/location
            
        Returns:
            Path to generated PDF
        """
        # Generate report ID
        report_id = self._generate_report_id()
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=0.6*inch,
            leftMargin=0.6*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        story = []
        
        # Build report sections in professional order
        story.extend(self._build_professional_header(report_id, patient_info, clinic_info))
        story.extend(self._build_ai_screening_result(results))
        
        # Urgent findings first
        story.extend(self._build_malaria_findings(results))
        story.extend(self._build_sickle_cell_findings(results))
        
        story.extend(self._build_microscopical_findings(results))
        
        # Add analyzed image if available
        if analyzed_image_path and Path(analyzed_image_path).exists():
            story.extend(self._build_image_section(analyzed_image_path))
        
        story.extend(self._build_gradcam_visual_explanation(results))
        story.extend(self._build_cell_counts(results))
        story.extend(self._build_wbc_differential(results))
        story.extend(self._build_shape_analysis(results))
        story.extend(self._build_comparison_to_normal(results))
        story.extend(self._build_methodology())
        story.extend(self._build_clinical_notes(results))
        # Malaria/Sickle findings moved to top
        story.extend(self._build_professional_footer())
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _build_professional_header(
        self, 
        report_id: str,
        patient_info: Optional[Dict],
        clinic_info: Optional[Dict]
    ) -> list:
        """Build professional header with title, IDs, and patient/clinic info."""
        elements = []
        
        # Main title with border
        title_data = [['PROFESSIONAL AUTOMATED BLOOD ANALYSIS REPORT']]
        title_table = Table(title_data, colWidths=[6.3*inch])
        title_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['primary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 16),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(title_table)
        elements.append(Spacer(1, 10))
        
        # Report ID and Date row
        report_date = datetime.now().strftime('%Y-%m-%d')
        id_data = [[f'Date: {report_date}', f'Report ID: {report_id}']]
        id_table = Table(id_data, colWidths=[3.15*inch, 3.15*inch])
        id_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['light']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['dark']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(id_table)
        elements.append(Spacer(1, 12))
        
        # Patient & Clinic Information section
        section_header = Table([['Patient & Clinic Information']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        # Build patient/clinic info table
        patient_id = patient_info.get('id', 'N/A') if patient_info else 'N/A'
        patient_name = patient_info.get('name', 'Anonymized') if patient_info else 'Anonymized'
        patient_dob = patient_info.get('dob', 'N/A') if patient_info else 'N/A'
        patient_gender = patient_info.get('gender', 'N/A') if patient_info else 'N/A'
        
        clinic_name = clinic_info.get('name', 'N/A') if clinic_info else 'N/A'
        physician = clinic_info.get('physician', 'N/A') if clinic_info else 'N/A'
        location = clinic_info.get('location', 'N/A') if clinic_info else 'N/A'
        test_date = datetime.now().strftime('%Y-%m-%d')
        
        info_data = [
            ['Patient ID:', patient_id, 'Referring Clinic:', clinic_name],
            ['Name:', patient_name, 'Physician:', physician],
            ['Date of Birth:', patient_dob, 'Location:', location],
            ['Gender:', patient_gender, 'Date of Test:', test_date],
        ]
        
        info_table = Table(info_data, colWidths=[1.2*inch, 1.7*inch, 1.4*inch, 2*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), COLORS['medium']),
            ('TEXTCOLOR', (2, 0), (2, -1), COLORS['medium']),
            ('TEXTCOLOR', (1, 0), (1, -1), COLORS['dark']),
            ('TEXTCOLOR', (3, 0), (3, -1), COLORS['dark']),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('ALIGN', (3, 0), (3, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['light']),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['border']),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_ai_screening_result(self, results: Dict) -> list:
        """Build prominent AI screening result banner."""
        elements = []
        
        # Section header
        section_header = Table([['AI SCREENING RESULT']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        # Get risk assessment
        risk = results.get('risk_assessment', {})
        risk_level = risk.get('level', 'unknown').upper()
        risk_score = risk.get('score', 0) * 100
        
        # Determine result text and colors
        if risk_level == 'HIGH':
            result_text = "HIGH PROBABILITY OF THALASSEMIA"
            bg_color = COLORS['danger']
            style_name = 'AIResultHigh'
        elif risk_level == 'MEDIUM':
            result_text = "MODERATE PROBABILITY OF THALASSEMIA"
            bg_color = COLORS['warning']
            style_name = 'AIResultMedium'
        else:
            result_text = "LOW PROBABILITY OF THALASSEMIA"
            bg_color = COLORS['success']
            style_name = 'AIResultLow'
        
        # Large result box
        result_data = [
            [f'AI Screening Result: {result_text}'],
            [f'AI Confidence: {risk_score:.1f}%'],
        ]
        
        if risk_level == 'HIGH':
            result_data.append(['Recommendation: Further Genetic Testing & Pathologist Review Required'])
        elif risk_level == 'MEDIUM':
            result_data.append(['Recommendation: CBC, Iron Studies, and Hb Electrophoresis Advised'])
        else:
            result_data.append(['Recommendation: Continue Routine Health Monitoring'])
        
        result_table = Table(result_data, colWidths=[6.3*inch])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['light']),
            ('TEXTCOLOR', (0, 0), (0, 0), bg_color),
            ('TEXTCOLOR', (0, 1), (0, 1), COLORS['dark']),
            ('TEXTCOLOR', (0, 2), (0, 2), COLORS['secondary']),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, 1), 'Helvetica-Bold'),
            ('FONTNAME', (0, 2), (0, 2), 'Helvetica-Oblique'),
            ('FONTSIZE', (0, 0), (0, 0), 14),
            ('FONTSIZE', (0, 1), (0, 1), 12),
            ('FONTSIZE', (0, 2), (0, 2), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 2, bg_color),
        ]))
        elements.append(result_table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _build_microscopical_findings(self, results: Dict) -> list:
        """Build automated microscopical findings section."""
        elements = []
        
        # Section header
        section_header = Table([['Automated Microscopical Findings']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        # Get shape analysis data
        shape_analysis = results.get('shape_analysis', {})
        shape_dist = shape_analysis.get('shape_distribution', {})
        
        # Calculate key morphological findings
        microcyte_pct = shape_dist.get('microcyte', 0)
        target_pct = shape_dist.get('target', 0)
        teardrop_pct = shape_dist.get('teardrop', 0)
        irregular_pct = shape_dist.get('irregular', 0)
        normal_pct = shape_dist.get('normal', 100)
        
        mean_circularity = shape_analysis.get('mean_circularity', 0.9)
        size_cv = shape_analysis.get('size_cv', 12)
        thal_indicator_pct = shape_analysis.get('thalassemia_indicator_pct', 0)
        
        # Subsection: Key Morphological Findings
        elements.append(Paragraph(
            "<b>Key Morphological Findings:</b>",
            self.styles['SubsectionHeader']
        ))
        
        findings = []
        
        # Microcytosis finding
        if microcyte_pct > 5:
            severity = "Marked" if microcyte_pct > 30 else "Moderate" if microcyte_pct > 15 else "Mild"
            findings.append(
                f"• <b>Microcytosis:</b> {microcyte_pct:.1f}% of RBCs show reduced volume (MCV &lt; 80 fL estimated)"
            )
        
        # Hypochromia (estimated from circularity/pallor)
        if mean_circularity < 0.85:
            findings.append(
                f"• <b>Hypochromia:</b> Increased central pallor detected (MCH &lt; 27 pg estimated)"
            )
        
        # Target cells
        if target_pct > 2:
            findings.append(
                f"• <b>Target Cells (Codocytes):</b> {target_pct:.1f}% of RBCs show bull's-eye pattern"
            )
        
        # Teardrop cells
        if teardrop_pct > 1:
            findings.append(
                f"• <b>Teardrop Cells (Dacrocytes):</b> {teardrop_pct:.1f}% detected"
            )
        
        # Anisocytosis (size variation)
        if size_cv > 15:
            severity = "Marked" if size_cv > 20 else "Moderate"
            findings.append(
                f"• <b>Anisocytosis:</b> {severity} variation in RBC size (RDW-CV: {size_cv:.1f}%)"
            )
        
        # Irregular cells
        if irregular_pct > 3:
            findings.append(
                f"• <b>Poikilocytosis:</b> {irregular_pct:.1f}% irregular/abnormal cell shapes"
            )
        
        # If mostly normal
        if not findings:
            findings.append("• <b>Normal Morphology:</b> No significant abnormalities detected")
        
        for finding in findings:
            elements.append(Paragraph(finding, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 10))
        
        # Summary metrics table
        metrics_data = [
            ['Metric', 'Value', 'Reference Range', 'Status'],
            ['Microcytes', f'{microcyte_pct:.1f}%', '< 5%', self._get_status_text(microcyte_pct, 5, 15)],
            ['Target Cells', f'{target_pct:.1f}%', '< 2%', self._get_status_text(target_pct, 2, 5)],
            ['Teardrop Cells', f'{teardrop_pct:.1f}%', '< 1%', self._get_status_text(teardrop_pct, 1, 3)],
            ['Size Variation (CV)', f'{size_cv:.1f}%', '11-15%', self._get_cv_status(size_cv)],
            ['Thalassemia Indicators', f'{thal_indicator_pct:.1f}%', '< 10%', self._get_status_text(thal_indicator_pct, 10, 25)],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[1.8*inch, 1.1*inch, 1.4*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 2), (-1, 2), COLORS['light']),
            ('BACKGROUND', (0, 4), (-1, 4), COLORS['light']),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _get_status_text(self, value: float, warn_threshold: float, danger_threshold: float) -> str:
        """Get status text based on thresholds."""
        if value >= danger_threshold:
            return "⚠️ HIGH"
        elif value >= warn_threshold:
            return "⚡ ELEVATED"
        else:
            return "✓ Normal"
    
    def _get_cv_status(self, cv: float) -> str:
        """Get status for coefficient of variation."""
        if cv > 20:
            return "⚠️ HIGH (Anisocytosis)"
        elif cv > 15:
            return "⚡ ELEVATED"
        elif cv < 11:
            return "⚡ LOW"
        else:
            return "✓ Normal"
    
    def _build_gradcam_visual_explanation(self, results: Dict) -> list:
        """Build Grad-CAM AI visual explanation section."""
        elements = []
        
        gradcam = results.get('gradcam', {})
        if not gradcam.get('enabled'):
            return elements
        
        # Section header
        section_header = Table([['AI Visual Explanation (Grad-CAM)']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        elements.append(Paragraph(
            "The AI model's attention regions are highlighted below. <b>Red/Yellow = High attention</b> "
            "(regions the model focused on), <b>Blue = Low attention</b> (ignored regions).",
            self.styles['ReportBody']
        ))
        elements.append(Spacer(1, 8))
        
        # List cells with explanations
        heatmaps = results.get('_wbc_heatmaps', [])
        shape_analysis = results.get('shape_analysis', {})
        
        explanations = []
        
        # WBC attention
        if heatmaps:
            elements.append(Paragraph("<b>WBC Classification Attention:</b>", self.styles['SubsectionHeader']))
            for hm in heatmaps[:4]:
                subtype = hm.get('subtype', 'Unknown')
                conf = hm.get('confidence', 0)
                explanations.append(f"• {subtype}: AI focused on cytoplasm granularity and nuclear shape ({conf*100:.1f}% confidence)")
        
        # RBC shape attention
        if shape_analysis.get('enabled'):
            elements.append(Paragraph("<b>RBC Morphology Attention:</b>", self.styles['SubsectionHeader']))
            shape_dist = shape_analysis.get('shape_distribution', {})
            
            if shape_dist.get('target', 0) > 2:
                explanations.append("• <b>Target Cells:</b> AI detected central hemoglobin density (bull's-eye pattern)")
            if shape_dist.get('microcyte', 0) > 5:
                explanations.append("• <b>Microcytes:</b> AI measured reduced cell diameter relative to normal")
            if shape_dist.get('teardrop', 0) > 1:
                explanations.append("• <b>Teardrop Cells:</b> AI identified elongated cell shape with pointed end")
        
        for exp in explanations:
            elements.append(Paragraph(exp, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _build_image_section(self, image_path: str) -> list:
        """Add the analyzed image to the report with professional styling."""
        elements = []
        
        # Section header
        section_header = Table([['Annotated Blood Smear Image']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        try:
            img = RLImage(image_path)
            max_width = 5.5 * inch
            max_height = 3.2 * inch
            
            aspect = img.imageWidth / img.imageHeight
            if aspect > max_width / max_height:
                img.drawWidth = max_width
                img.drawHeight = max_width / aspect
            else:
                img.drawHeight = max_height
                img.drawWidth = max_height * aspect
            
            img.hAlign = 'CENTER'
            elements.append(Spacer(1, 5))
            elements.append(img)
            elements.append(Spacer(1, 5))
            elements.append(Paragraph(
                "<i>AI-annotated blood smear showing detected cells. "
                "Colors: RBC (red), WBC (by subtype), Platelets (blue).</i>",
                ParagraphStyle('ImageCaption', parent=self.styles['Normal'], 
                              fontSize=8, alignment=TA_CENTER, textColor=COLORS['medium'])
            ))
        except Exception as e:
            elements.append(Paragraph(f"<i>Image could not be loaded: {e}</i>", self.styles['ReportBody']))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _build_cell_counts(self, results: Dict) -> list:
        """Build cell count summary section."""
        elements = []
        
        section_header = Table([['Cell Count Summary']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        counts = results.get('cell_counts', {})
        total = sum(counts.values())
        
        count_data = [
            ['Cell Type', 'Count', 'Percentage', 'Visual Distribution'],
            ['Red Blood Cells (RBC)', str(counts.get('RBC', 0)), 
             f"{counts.get('RBC', 0)/total*100:.1f}%" if total > 0 else "0%",
             self._make_bar(counts.get('RBC', 0), total)],
            ['White Blood Cells (WBC)', str(counts.get('WBC', 0)),
             f"{counts.get('WBC', 0)/total*100:.1f}%" if total > 0 else "0%",
             self._make_bar(counts.get('WBC', 0), total)],
            ['Platelets', str(counts.get('Platelets', 0)),
             f"{counts.get('Platelets', 0)/total*100:.1f}%" if total > 0 else "0%",
             self._make_bar(counts.get('Platelets', 0), total)],
            ['TOTAL', str(total), '100%', ''],
        ]
        
        count_table = Table(count_data, colWidths=[1.8*inch, 0.8*inch, 0.9*inch, 2.8*inch])
        count_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),
            ('BACKGROUND', (0, -1), (-1, -1), COLORS['light']),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
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
        return '█' * filled + '░' * (20 - filled)
    
    def _build_wbc_differential(self, results: Dict) -> list:
        """Build WBC differential section."""
        elements = []
        
        differential = results.get('differential', {})
        if not differential:
            return elements
        
        section_header = Table([['White Blood Cell Differential']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        elements.append(Paragraph(
            "WBC differential count showing percentage of each cell type. "
            "Normal ranges provided for reference.",
            self.styles['ReportBody']
        ))
        elements.append(Spacer(1, 5))
        
        wbc_data = [['WBC Type', 'Count', 'Percentage', 'Normal Range', 'Status']]
        
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
            
            if pct < low and data.get('count', 0) > 0:
                status = '↓ Low'
            elif pct > high:
                status = '↑ High'
            else:
                status = '✓ Normal'
            
            wbc_data.append([
                subtype.capitalize(),
                str(data.get('count', 0)),
                f"{pct:.1f}%",
                f"{low}-{high}%",
                status
            ])
        
        wbc_table = Table(wbc_data, colWidths=[1.4*inch, 0.8*inch, 1*inch, 1.1*inch, 2*inch])
        wbc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['accent']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BACKGROUND', (0, 2), (-1, 2), COLORS['light']),
            ('BACKGROUND', (0, 4), (-1, 4), COLORS['light']),
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
        
        section_header = Table([['Red Blood Cell Morphology Analysis']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        method = shape_analysis.get('method', 'bbox_approximation')
        method_text = "Instance Segmentation" if method == 'instance_segmentation' else "Bounding Box"
        elements.append(Paragraph(
            f"<b>Method:</b> {method_text} | <b>Cells Analyzed:</b> {shape_analysis.get('cells_analyzed', 0)}",
            self.styles['ReportBody']
        ))
        elements.append(Spacer(1, 8))
        
        # Shape distribution
        shape_dist = shape_analysis.get('shape_distribution', {})
        shape_counts = shape_analysis.get('shape_counts', {})
        
        if shape_dist:
            elements.append(Paragraph("<b>Shape Distribution:</b>", self.styles['SubsectionHeader']))
            
            shape_significance = {
                'normal': 'Healthy red blood cells',
                'microcyte': 'May indicate iron deficiency or thalassemia',
                'target': 'Associated with thalassemia, liver disease',
                'teardrop': 'May indicate bone marrow disorder',
                'spherocyte': 'Hereditary spherocytosis or immune hemolysis',
                'irregular': 'Various causes, requires evaluation',
                'ring': '⚠️ URGENT - Malaria parasite detected (ring stage)',
                'trophozoite': '⚠️ URGENT - Mature malaria parasite detected',
                'sickle': 'May indicate sickle cell disease - requires confirmation',
            }
            
            shape_data = [['Shape', 'Count', 'Percentage', 'Clinical Significance']]
            for shape, pct in sorted(shape_dist.items(), key=lambda x: x[1], reverse=True):
                if pct > 0:
                    count = shape_counts.get(shape, 0)
                    significance = shape_significance.get(shape, 'Unknown')
                    shape_data.append([shape.capitalize(), str(count), f"{pct:.1f}%", significance])
            
            shape_table = Table(shape_data, colWidths=[1.1*inch, 0.7*inch, 0.9*inch, 3.6*inch])
            shape_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), COLORS['danger']),
                ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (1, 1), (2, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(shape_table)
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _build_comparison_to_normal(self, results: Dict) -> list:
        """Build comparison to normal RBC section."""
        elements = []
        
        shape_analysis = results.get('shape_analysis', {})
        if not shape_analysis.get('enabled'):
            return elements
        
        section_header = Table([['Comparison to Normal RBC']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        shape_dist = shape_analysis.get('shape_distribution', {})
        microcyte_pct = shape_dist.get('microcyte', 0)
        size_cv = shape_analysis.get('size_cv', 12)
        circularity = shape_analysis.get('mean_circularity', 0.9)
        
        ref_data = [
            ['Parameter', 'Normal Value', 'Patient Estimate', 'Interpretation'],
            ['MCV (cell size)', '80-100 fL', self._estimate_mcv_str(microcyte_pct), self._mcv_interpretation(microcyte_pct)],
            ['MCH (hemoglobin)', '27-33 pg', self._estimate_mch_str(circularity), self._mch_interpretation(circularity)],
            ['RDW (size variation)', '11-15%', f"~{size_cv:.1f}%", self._rdw_interpretation(size_cv)],
            ['Central Pallor', '1/3 diameter', self._pallor_status(circularity), self._pallor_interpretation(circularity)],
        ]
        
        ref_table = Table(ref_data, colWidths=[1.3*inch, 1.2*inch, 1.4*inch, 2.4*inch])
        ref_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BACKGROUND', (0, 2), (-1, 2), COLORS['light']),
            ('BACKGROUND', (0, 4), (-1, 4), COLORS['light']),
        ]))
        elements.append(ref_table)
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _estimate_mcv_str(self, microcyte_pct: float) -> str:
        if microcyte_pct > 30: return "~65-75 fL (Low)"
        elif microcyte_pct > 15: return "~75-80 fL (Borderline)"
        else: return "~80-90 fL (Normal)"
    
    def _mcv_interpretation(self, microcyte_pct: float) -> str:
        if microcyte_pct > 30: return "Suggests thalassemia/iron deficiency"
        elif microcyte_pct > 15: return "Mild microcytosis"
        else: return "Within normal limits"
    
    def _estimate_mch_str(self, circularity: float) -> str:
        if circularity < 0.75: return "~22-25 pg (Low)"
        elif circularity < 0.85: return "~25-27 pg (Borderline)"
        else: return "~27-31 pg (Normal)"
    
    def _mch_interpretation(self, circularity: float) -> str:
        if circularity < 0.75: return "Hypochromia - hemoglobin disorder"
        elif circularity < 0.85: return "Mild hypochromia"
        else: return "Adequate hemoglobin"
    
    def _rdw_interpretation(self, cv: float) -> str:
        if cv > 20: return "Marked anisocytosis"
        elif cv > 15: return "Moderate anisocytosis"
        else: return "Normal variation"
    
    def _pallor_status(self, circularity: float) -> str:
        return "Increased (> 1/3)" if circularity < 0.80 else "Normal (≈ 1/3)"
    
    def _pallor_interpretation(self, circularity: float) -> str:
        return "Hypochromia indicator" if circularity < 0.80 else "Normal hemoglobin distribution"
    
    def _build_methodology(self) -> list:
        """Build AI methodology section."""
        elements = []
        
        section_header = Table([['AI Analysis Methodology']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        elements.append(Paragraph(
            "This report was generated using validated deep learning models:",
            self.styles['ReportBody']
        ))
        
        method_data = [
            ['Component', 'Model', 'Accuracy', 'Training Data'],
            ['Cell Detection', 'YOLOv11n', '92.8% mAP50', 'BCCD (364 images)'],
            ['Segmentation', 'YOLOv11n-seg', '98.2% mAP50', 'BCCD + Masks'],
            ['WBC Classification', 'ResNet34', '97.9%', 'Raabin-WBC (10,175 cells)'],
            ['RBC Morphology', 'Morphometry', 'Rule-based', 'Clinical thresholds'],
        ]
        
        method_table = Table(method_data, colWidths=[1.4*inch, 1.2*inch, 1*inch, 2.7*inch])
        method_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['dark']),
            ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(method_table)
        elements.append(Spacer(1, 10))
        
        return elements
    
    def _build_clinical_notes(self, results: Dict) -> list:
        """Build clinical notes section."""
        elements = []
        
        section_header = Table([['Clinical Notes']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['secondary']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        notes = []
        
        # WBC findings
        differential = results.get('differential', {})
        if differential:
            neutrophil_pct = differential.get("NEUTROPHIL", {}).get("percentage", 0)
            lymphocyte_pct = differential.get("LYMPHOCYTE", {}).get("percentage", 0)
            eosinophil_pct = differential.get("EOSINOPHIL", {}).get("percentage", 0)
            
            if neutrophil_pct > 70:
                notes.append("• <b>Neutrophilia:</b> May indicate bacterial infection or inflammation")
            elif neutrophil_pct < 40 and differential.get("NEUTROPHIL", {}).get("count", 0) > 0:
                notes.append("• <b>Neutropenia:</b> May indicate viral infection or bone marrow suppression")
            if lymphocyte_pct > 40:
                notes.append("• <b>Lymphocytosis:</b> May indicate viral infection")
            if eosinophil_pct > 5:
                notes.append("• <b>Eosinophilia:</b> May indicate parasitic infection or allergy")
        
        # RBC findings - Check for multi-disorder risks
        shape_analysis = results.get('shape_analysis', {})
        multi_disorder_risks = results.get('multi_disorder_risks', {})
        
        if shape_analysis.get('enabled'):
            # Malaria findings
            if 'malaria' in multi_disorder_risks:
                malaria_data = multi_disorder_risks['malaria']
                notes.append(f"• <b>⚠️ URGENT - Malaria Parasites Detected:</b> {malaria_data['infected_cells']} infected RBCs ({malaria_data['percentage']:.1f}%). Immediate lab confirmation required.")
            
            # Sickle cell findings
            if 'sickle_cell' in multi_disorder_risks:
                sickle_data = multi_disorder_risks['sickle_cell']
                notes.append(f"• <b>Sickle Cells Detected:</b> {sickle_data['sickle_cells']} cells ({sickle_data['percentage']:.1f}%). Consider sickle cell disease workup.")
            
            # Thalassemia findings
            thal_pct = shape_analysis.get('thalassemia_indicator_pct', 0)
            if thal_pct > 30:
                notes.append(f"• <b>Thalassemia Indicators:</b> {thal_pct:.1f}% target cells and teardrops detected. Suggestive of thalassemia.")
        
        if not notes:
            notes.append("• No significant abnormalities noted")
        
        for note in notes:
            elements.append(Paragraph(note, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _build_malaria_findings(self, results: Dict) -> list:
        """Build detailed malaria findings section if parasites detected."""
        elements = []
        
        multi_disorder_risks = results.get('multi_disorder_risks', {})
        if 'malaria' not in multi_disorder_risks:
            return elements
        
        shape_analysis = results.get('shape_analysis', {})
        parasite_stages = shape_analysis.get('parasite_stages', {})
        
        # Section header with urgent styling
        section_header = Table([['⚠️ URGENT: MALARIA PARASITE FINDINGS']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['danger']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['white']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        malaria_data = multi_disorder_risks['malaria']
        
        elements.append(Paragraph(
            f"<b>Parasitemia Level:</b> {malaria_data['percentage']:.2f}% ({malaria_data['infected_cells']} infected cells detected)",
            self.styles['ReportBody']
        ))
        elements.append(Spacer(1, 8))
        
        # Parasite stage breakdown
        if parasite_stages:
            elements.append(Paragraph("<b>Parasite Stage Distribution:</b>", self.styles['SubsectionHeader']))
            
            stage_data = [['Stage', 'Count', 'Clinical Significance']]
            
            ring_count = parasite_stages.get('ring', 0)
            tropho_count = parasite_stages.get('trophozoite', 0)
            
            if ring_count > 0:
                stage_data.append([
                    'Ring Stage',
                    str(ring_count),
                    'Early infection - thin ring appearance'
                ])
            
            if tropho_count > 0:
                stage_data.append([
                    'Trophozoite',
                    str(tropho_count),
                    'Mature parasite - amoeboid form'
                ])
            
            stage_table = Table(stage_data, colWidths=[1.8*inch, 1.0*inch, 3.5*inch])
            stage_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), COLORS['danger']),
                ('TEXTCOLOR', (0, 0), (-1, 0), COLORS['white']),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(stage_table)
            elements.append(Spacer(1, 8))
        
        # Urgent recommendations
        elements.append(Paragraph("<b>⚠️ Immediate Actions Required:</b>", self.styles['SubsectionHeader']))
        recommendations = [
            "• <b>URGENT:</b> Confirm with thick and thin blood smear microscopy",
            "• Rapid Diagnostic Test (RDT) for malaria antigens",
            "• Complete blood count with differential",
            "• Begin appropriate antimalarial therapy if confirmed",
            "• Monitor for complications (cerebral malaria, severe anemia)"
        ]
        
        for rec in recommendations:
            elements.append(Paragraph(rec, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _build_sickle_cell_findings(self, results: Dict) -> list:
        """Build detailed sickle cell findings section if sickle cells detected."""
        elements = []
        
        multi_disorder_risks = results.get('multi_disorder_risks', {})
        if 'sickle_cell' not in multi_disorder_risks:
            return elements
        
        # Section header with warning styling
        section_header = Table([['SICKLE CELL FINDINGS']], colWidths=[6.3*inch])
        section_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['warning']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['dark']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_header)
        
        sickle_data = multi_disorder_risks['sickle_cell']
        
        elements.append(Paragraph(
            f"<b>Sickle Cell Percentage:</b> {sickle_data['percentage']:.1f}% ({sickle_data['sickle_cells']} sickled cells detected)",
            self.styles['ReportBody']
        ))
        elements.append(Spacer(1, 8))
        
        # Morphology description
        elements.append(Paragraph("<b>Morphological Features:</b>", self.styles['SubsectionHeader']))
        morph_features = [
            "• <b>Crescent/Sickle Shape:</b> RBCs with characteristic elongated, curved shape",
            "• <b>Reduced Deformability:</b> Cells may appear rigid and pointed at one or both ends",
            "• <b>Polymerization Pattern:</b> Suggests hemoglobin S polymerization under deoxygenation"
        ]
        
        for feature in morph_features:
            elements.append(Paragraph(feature, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 8))
        
        # Recommended follow-up
        elements.append(Paragraph("<b>Recommended Follow-up Tests:</b>", self.styles['SubsectionHeader']))
        followup = [
            "• <b>Hemoglobin Electrophoresis:</b> Definitive test for sickle cell disease (HbS detection)",
            "• <b>Sickle Cell Solubility Test:</b> Rapid screening for HbS",
            "• <b>Complete Blood Count:</b> Assess for chronic anemia",
            "• <b>Reticulocyte Count:</b> Evaluate bone marrow response",
            "• <b>Genetic Counseling:</b> If positive, family screening recommended"
        ]
        
        for fu in followup:
            elements.append(Paragraph(fu, self.styles['BulletPoint']))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _build_professional_footer(self) -> list:
        """Build professional footer with disclaimer and contact."""
        elements = []
        
        elements.append(HRFlowable(width="100%", thickness=1, color=COLORS['border']))
        elements.append(Spacer(1, 10))
        
        # Disclaimer box
        disclaimer_data = [[
            "IMPORTANT DISCLAIMER\n\n"
            "This report is a computer-aided screening tool for EDUCATIONAL and RESEARCH purposes only. "
            "It is NOT a substitute for professional medical diagnosis or advice. "
            "The accuracy of AI analysis can vary based on image quality and other factors.\n\n"
            "Always consult with a qualified hematologist or healthcare provider for interpretation "
            "of blood smear findings and any health concerns. Laboratory tests performed by certified "
            "professionals are required for clinical diagnosis."
        ]]
        
        disclaimer_table = Table(disclaimer_data, colWidths=[6.3*inch])
        disclaimer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['light']),
            ('TEXTCOLOR', (0, 0), (-1, -1), COLORS['medium']),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['border']),
        ]))
        elements.append(disclaimer_table)
        
        elements.append(Spacer(1, 10))
        
        # Contact and timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elements.append(Paragraph(
            f"Blood Cell Analyzer v1.0 | Report Generated: {timestamp}<br/>"
            "Contact: support@hema-ai.com | https://hema-ai.com",
            self.styles['Footer']
        ))
        
        return elements


def generate_pdf_report(
    results: Dict[str, Any],
    output_path: str,
    analyzed_image_path: Optional[str] = None,
    patient_info: Optional[Dict[str, str]] = None,
    clinic_info: Optional[Dict[str, str]] = None
) -> str:
    """
    Convenience function to generate a PDF report.
    
    Args:
        results: Analysis results from BloodSmearAnalyzer
        output_path: Where to save the PDF
        analyzed_image_path: Path to the annotated image
        patient_info: Optional dict with 'name', 'id', 'dob', 'gender'
        clinic_info: Optional dict with 'name', 'physician', 'location'
        
    Returns:
        Path to the generated PDF
    """
    generator = BloodSmearPDFReport()
    return generator.generate_report(
        results=results,
        output_path=output_path,
        analyzed_image_path=analyzed_image_path,
        patient_info=patient_info,
        clinic_info=clinic_info
    )


if __name__ == "__main__":
    # Test with sample data
    sample_results = {
        "image_path": "test_image.jpg",
        "image_size": {"width": 1152, "height": 1703},
        "timestamp": datetime.now().isoformat(),
        "cell_counts": {"RBC": 245, "WBC": 8, "Platelets": 32},
        "summary": {"total_cells": 285, "wbc_classified": 8},
        "differential": {
            "NEUTROPHIL": {"count": 5, "percentage": 62.5},
            "LYMPHOCYTE": {"count": 2, "percentage": 25.0},
            "MONOCYTE": {"count": 1, "percentage": 12.5},
            "EOSINOPHIL": {"count": 0, "percentage": 0.0},
            "BASOPHIL": {"count": 0, "percentage": 0.0},
        },
        "shape_analysis": {
            "enabled": True,
            "method": "instance_segmentation",
            "cells_analyzed": 245,
            "shape_distribution": {
                "normal": 52.0,
                "target": 18.0,
                "microcyte": 15.0,
                "teardrop": 8.0,
                "irregular": 7.0,
            },
            "shape_counts": {
                "normal": 127,
                "target": 44,
                "microcyte": 37,
                "teardrop": 20,
                "irregular": 17,
            },
            "thalassemia_indicator_pct": 41.0,
            "abnormality_index": 0.48,
            "mean_circularity": 0.78,
            "mean_elongation": 1.35,
            "size_cv": 18.5,
        },
        "risk_assessment": {
            "score": 0.85,
            "level": "high",
            "interpretation": "Significant thalassemia indicators detected"
        },
        "gradcam": {"enabled": False}
    }
    
    output = generate_pdf_report(
        sample_results,
        "test_professional_report.pdf",
        patient_info={"name": "Anonymized", "id": "PH-7892", "gender": "Female"},
        clinic_info={"name": "MedCare Diagnostics", "physician": "Dr. Santos", "location": "Manila"}
    )
    print(f"Generated: {output}")
