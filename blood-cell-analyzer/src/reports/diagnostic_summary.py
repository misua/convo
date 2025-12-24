#!/usr/bin/env python3
"""
One-Page Diagnostic Summary PDF Generator
Organized by disease conditions with clear tables and clinical interpretation
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
)

# Clinical color scheme
COLORS = {
    'header': colors.HexColor('#1a237e'),
    'positive': colors.HexColor('#c62828'),  # Red for detected
    'negative': colors.HexColor('#2e7d32'),  # Green for not detected
    'warning': colors.HexColor('#f57c00'),   # Orange for borderline
    'section': colors.HexColor('#0d47a1'),
    'light_bg': colors.HexColor('#f5f5f5'),
    'border': colors.HexColor('#e0e0e0'),
}


class DiagnosticSummaryPDF:
    """Generate 1-page diagnostic summary organized by disease conditions."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()
    
    def _setup_styles(self):
        """Create compact, professional styles for 1-page layout."""
        
        # Use custom style names to avoid conflicts
        self.styles.add(ParagraphStyle(
            name='DiagTitle',
            fontSize=14,
            textColor=COLORS['header'],
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='DiagSectionHeader',
            fontSize=10,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT,
            leftIndent=4,
            spaceAfter=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='DiagSmallText',
            fontSize=7,
            textColor=colors.HexColor('#424242'),
            alignment=TA_LEFT
        ))
        
        self.styles.add(ParagraphStyle(
            name='DiagTinyText',
            fontSize=6,
            textColor=colors.HexColor('#616161'),
            alignment=TA_CENTER
        ))
    
    def generate(
        self,
        results: Dict[str, Any],
        output_path: str,
        patient_info: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate compact 1-page diagnostic summary."""
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.4*inch,
            leftMargin=0.4*inch,
            topMargin=0.3*inch,
            bottomMargin=0.3*inch
        )
        
        story = []
        
        # Header (compact)
        story.extend(self._build_header(patient_info))
        story.append(Spacer(1, 8))
        
        # DISEASE-BASED SECTIONS
        story.extend(self._build_parasitic_infections(results))
        story.append(Spacer(1, 6))
        
        story.extend(self._build_hemoglobinopathies(results))
        story.append(Spacer(1, 6))
        
        story.extend(self._build_anemia_findings(results))
        story.append(Spacer(1, 6))
        
        story.extend(self._build_infection_indicators(results))
        story.append(Spacer(1, 6))
        
        story.extend(self._build_malignancy_screening(results))
        story.append(Spacer(1, 8))
        
        # Footer disclaimer
        story.extend(self._build_footer())
        
        doc.build(story)
        return output_path
    
    def _build_header(self, patient_info: Optional[Dict]) -> list:
        """Compact header with patient info."""
        elements = []
        
        # Title
        title = Paragraph(
            "<b>BLOOD SMEAR DIAGNOSTIC SUMMARY</b>",
            self.styles['DiagTitle']
        )
        elements.append(title)
        
        # Patient info row
        patient_id = patient_info.get('id', 'N/A') if patient_info else 'N/A'
        patient_name = patient_info.get('name', 'Anonymous') if patient_info else 'Anonymous'
        test_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        header_data = [[
            f"Patient: {patient_name}",
            f"ID: {patient_id}",
            f"Date: {test_date}"
        ]]
        
        header_table = Table(header_data, colWidths=[2.4*inch, 1.8*inch, 2.2*inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['light_bg']),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
            ('ALIGN', (1, 0), (1, 0), 'CENTER'),
            ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(header_table)
        
        return elements
    
    def _build_parasitic_infections(self, results: Dict) -> list:
        """Section 1: Malaria & Parasitic Infections."""
        elements = []
        
        # Section header
        header = Table([['PARASITIC INFECTIONS']], colWidths=[6.6*inch])
        header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['section']),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(header)
        
        # Malaria detection data
        malaria = results.get('malaria_detection', {})
        if malaria.get('enabled'):
            parasite_count = malaria.get('structure_count', 0)
            detection_rate = malaria.get('detection_rate', 0.0)
            method = "YOLO Object Detection"
            
            if parasite_count > 0:
                status = "⚠️ DETECTED"
                status_color = COLORS['positive']
                
                # Get stage breakdown
                stages = malaria.get('structure_breakdown', {})
                stage_details = ", ".join([
                    f"{stage.title()}: {count}" 
                    for stage, count in stages.items() if count > 0
                ])
                
                interpretation = f"Candidate parasite-like structures detected. Expert review required."
                finding = f"{parasite_count} structures ({detection_rate:.2f}% detection rate)"
                recommendation = "URGENT: Confirm with thick/thin blood smear microscopy by trained microscopist"
            else:
                status = "✓ NOT DETECTED"
                status_color = COLORS['negative']
                stage_details = "None"
                interpretation = "No parasite-like structures identified in analyzed regions."
                finding = "0 structures detected"
                recommendation = "Negative screening. Clinical suspicion requires confirmation with microscopy."
        else:
            status = "N/A"
            status_color = colors.grey
            stage_details = "Module disabled"
            interpretation = "Malaria detection not performed"
            finding = "N/A"
            recommendation = "N/A"
        
        # Malaria table
        malaria_data = [
            ['Test', 'Result', 'Finding', 'Clinical Interpretation'],
            [
                'Malaria\n(P. falciparum/vivax)',
                status,
                finding,
                interpretation
            ],
            [
                'Method',
                method if malaria.get('enabled') else 'N/A',
                f'Stages: {stage_details}' if malaria.get('enabled') else '',
                recommendation if malaria.get('enabled') else ''
            ]
        ]
        
        malaria_table = Table(malaria_data, colWidths=[1.2*inch, 1.0*inch, 1.6*inch, 2.8*inch])
        malaria_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['light_bg']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            
            # Data rows
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('ALIGN', (2, 1), (-1, -1), 'LEFT'),
            
            # Status cell color
            ('TEXTCOLOR', (1, 1), (1, 1), status_color),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
            
            # Borders
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['border']),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(malaria_table)
        
        return elements
    
    def _build_hemoglobinopathies(self, results: Dict) -> list:
        """Section 2: Hemoglobinopathies (Sickle Cell, Thalassemia)."""
        elements = []
        
        header = Table([['HEMOGLOBINOPATHIES (Sickle Cell & Thalassemia)']], colWidths=[6.6*inch])
        header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['section']),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(header)
        
        # Get shape analysis data
        shape = results.get('shape_analysis', {})
        risks = results.get('multi_disorder_risks', {})
        
        # Sickle cell
        sickle_pct = shape.get('sickle_cell_pct', 0.0) if shape.get('enabled') else 0.0
        sickle_count = shape.get('sickle_cells', 0) if shape.get('enabled') else 0
        sickle_risk = risks.get('sickle_cell', {})
        
        if sickle_pct > 5.0:
            sickle_status = "⚠️ DETECTED"
            sickle_color = COLORS['positive']
            sickle_interp = f"Sickle cells present ({sickle_pct:.1f}%). Consider sickle cell disease workup."
            sickle_rec = "Hemoglobin electrophoresis recommended"
        elif sickle_pct > 0:
            sickle_status = "○ TRACE"
            sickle_color = COLORS['warning']
            sickle_interp = f"Trace sickle cells ({sickle_pct:.1f}%). May indicate carrier status."
            sickle_rec = "Clinical correlation advised"
        else:
            sickle_status = "✓ NEGATIVE"
            sickle_color = COLORS['negative']
            sickle_interp = "No sickle cells detected."
            sickle_rec = "Normal finding"
        
        # Thalassemia
        thal_pct = shape.get('thalassemia_indicator_pct', 0.0) if shape.get('enabled') else 0.0
        thal_count = shape.get('thalassemia_indicators', 0) if shape.get('enabled') else 0
        thal_risk = risks.get('thalassemia', {})
        
        if thal_pct > 30.0:
            thal_status = "⚠️ DETECTED"
            thal_color = COLORS['positive']
            thal_interp = f"Significant thalassemia indicators ({thal_pct:.1f}%): target cells, teardrops, microcytes."
            thal_rec = "Hemoglobin electrophoresis + CBC with RBC indices recommended"
        elif thal_pct > 15.0:
            thal_status = "○ BORDERLINE"
            thal_color = COLORS['warning']
            thal_interp = f"Moderate thalassemia indicators ({thal_pct:.1f}%). Monitor."
            thal_rec = "CBC with RBC indices + iron studies"
        else:
            thal_status = "✓ NEGATIVE"
            thal_color = COLORS['negative']
            thal_interp = "No significant thalassemia indicators."
            thal_rec = "Normal RBC morphology"
        
        # Hemoglobinopathy table
        hemo_data = [
            ['Condition', 'Result', 'Findings', 'Interpretation'],
            [
                'Sickle Cell',
                sickle_status,
                f"{sickle_count} cells\n({sickle_pct:.1f}%)" if shape.get('enabled') else 'N/A',
                sickle_interp
            ],
            [
                'Thalassemia',
                thal_status,
                f"{thal_count} indicators\n({thal_pct:.1f}%)" if shape.get('enabled') else 'N/A',
                thal_interp
            ],
            [
                'Next Steps',
                '',
                '',
                f"Sickle: {sickle_rec}\nThalassemia: {thal_rec}"
            ]
        ]
        
        hemo_table = Table(hemo_data, colWidths=[1.0*inch, 0.9*inch, 1.1*inch, 3.6*inch])
        hemo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['light_bg']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            
            ('FONTSIZE', (0, 1), (-1, -1), 6.5),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('ALIGN', (2, 1), (2, -1), 'CENTER'),
            ('ALIGN', (3, 1), (3, -1), 'LEFT'),
            
            ('TEXTCOLOR', (1, 1), (1, 1), sickle_color),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (1, 2), (1, 2), thal_color),
            ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
            
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['border']),
            
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(hemo_table)
        
        return elements
    
    def _build_anemia_findings(self, results: Dict) -> list:
        """Section 3: Anemia Types."""
        elements = []
        
        header = Table([['ANEMIA SCREENING (RBC Morphology)']], colWidths=[6.6*inch])
        header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['section']),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(header)
        
        shape = results.get('shape_analysis', {})
        
        if shape.get('enabled'):
            shape_dist = shape.get('shape_distribution', {})
            
            # Extract morphology indicators
            microcyte_pct = shape_dist.get('microcyte', 0.0)
            size_cv = shape.get('size_cv', 0.0)
            
            # Iron deficiency anemia (microcytic, hypochromic)
            if microcyte_pct > 20:
                iron_status = "⚠️ SUSPECTED"
                iron_color = COLORS['positive']
                iron_finding = f"{microcyte_pct:.1f}% microcytes"
                iron_interp = "Microcytic pattern. Consider iron deficiency or thalassemia minor."
            elif microcyte_pct > 10:
                iron_status = "○ BORDERLINE"
                iron_color = COLORS['warning']
                iron_finding = f"{microcyte_pct:.1f}% microcytes"
                iron_interp = "Mild microcytosis. Iron studies recommended."
            else:
                iron_status = "✓ NORMAL SIZE"
                iron_color = COLORS['negative']
                iron_finding = f"{microcyte_pct:.1f}% microcytes"
                iron_interp = "Normal RBC size distribution."
            
            # Megaloblastic anemia (macrocytic) - would need size analysis
            # For now, using size variation (RDW proxy)
            if size_cv > 20:
                mega_status = "○ ANISOCYTOSIS"
                mega_color = COLORS['warning']
                mega_finding = f"RDW proxy: {size_cv:.1f}%"
                mega_interp = "High size variation. Consider B12/folate deficiency."
            else:
                mega_status = "✓ NORMAL"
                mega_color = COLORS['negative']
                mega_finding = f"Size CV: {size_cv:.1f}%"
                mega_interp = "Normal size variation."
        else:
            iron_status = mega_status = "N/A"
            iron_color = mega_color = colors.grey
            iron_finding = mega_finding = "Analysis disabled"
            iron_interp = mega_interp = "RBC shape analysis not performed"
        
        anemia_data = [
            ['Anemia Type', 'Indicator', 'Findings', 'Interpretation'],
            [
                'Iron Deficiency\n(Microcytic)',
                iron_status,
                iron_finding,
                iron_interp
            ],
            [
                'Size Variation\n(Anisocytosis)',
                mega_status,
                mega_finding,
                mega_interp
            ],
        ]
        
        anemia_table = Table(anemia_data, colWidths=[1.2*inch, 1.0*inch, 1.4*inch, 2.9*inch])
        anemia_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['light_bg']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('ALIGN', (2, 1), (-1, -1), 'LEFT'),
            
            ('TEXTCOLOR', (1, 1), (1, 1), iron_color),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (1, 2), (1, 2), mega_color),
            ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
            
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['border']),
            
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(anemia_table)
        
        return elements
    
    def _build_infection_indicators(self, results: Dict) -> list:
        """Section 4: Bacterial/Viral Infection Indicators (WBC Differential)."""
        elements = []
        
        header = Table([['INFECTION INDICATORS (WBC Differential Count)']], colWidths=[6.6*inch])
        header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['section']),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(header)
        
        differential = results.get('differential', {})
        
        # WBC differential with clinical interpretation
        wbc_data = [
            ['Cell Type', 'Count', '%', 'Normal Range', 'Clinical Significance'],
        ]
        
        # Reference ranges and interpretations
        wbc_info = {
            'NEUTROPHIL': {
                'range': '40-70%',
                'high': 'Bacterial infection, inflammation',
                'low': 'Viral infection, bone marrow disorder'
            },
            'LYMPHOCYTE': {
                'range': '20-40%',
                'high': 'Viral infection (COVID, flu, hepatitis)',
                'low': 'Immunodeficiency, steroid use'
            },
            'MONOCYTE': {
                'range': '2-10%',
                'high': 'Chronic infection, TB, malaria',
                'low': 'Bone marrow disorder'
            },
            'EOSINOPHIL': {
                'range': '1-5%',
                'high': 'Parasites, allergies, asthma',
                'low': 'Normal (stress response)'
            },
            'BASOPHIL': {
                'range': '0-2%',
                'high': 'Allergic reaction, blood disorder',
                'low': 'Normal'
            }
        }
        
        infection_findings = []
        
        for cell_type in ['NEUTROPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'EOSINOPHIL', 'BASOPHIL']:
            data = differential.get(cell_type, {'count': 0, 'percentage': 0.0})
            count = data.get('count', 0)
            pct = data.get('percentage', 0.0)
            info = wbc_info[cell_type]
            
            # Determine status
            range_str = info['range']
            low, high = map(lambda x: float(x.strip('%')), range_str.split('-'))
            
            if pct > high:
                status = '↑ HIGH'
                significance = info['high']
                infection_findings.append(f"↑ {cell_type.capitalize()} ({pct:.1f}%): {info['high']}")
            elif pct < low:
                status = '↓ LOW'
                significance = info['low']
                if cell_type not in ['EOSINOPHIL', 'BASOPHIL']:  # Only flag if clinically significant
                    infection_findings.append(f"↓ {cell_type.capitalize()} ({pct:.1f}%): {info['low']}")
            else:
                status = '✓ Normal'
                significance = 'Within normal limits'
            
            # Add to table
            wbc_data.append([
                cell_type.capitalize(),
                str(count),
                f"{pct:.1f}%",
                range_str,
                significance if pct > high or (pct < low and cell_type in ['NEUTROPHIL', 'LYMPHOCYTE']) else '✓'
            ])
        
        wbc_table = Table(wbc_data, colWidths=[1.0*inch, 0.5*inch, 0.6*inch, 0.9*inch, 3.5*inch])
        wbc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['light_bg']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            
            ('FONTSIZE', (0, 1), (-1, -1), 6.5),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),
            ('ALIGN', (3, 1), (3, -1), 'CENTER'),
            ('ALIGN', (4, 1), (4, -1), 'LEFT'),
            
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['border']),
            
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(wbc_table)
        
        # Summary interpretation
        if infection_findings:
            summary_text = "<b>Clinical Interpretation:</b> " + "; ".join(infection_findings[:3])
        else:
            summary_text = "<b>Clinical Interpretation:</b> WBC differential within normal limits. No infection indicators detected."
        
        summary = Paragraph(summary_text, self.styles['DiagSmallText'])
        elements.append(Spacer(1, 3))
        elements.append(summary)
        
        return elements
    
    def _build_malignancy_screening(self, results: Dict) -> list:
        """Section 5: Leukemia/Malignancy Screening."""
        elements = []
        
        header = Table([['LEUKEMIA / MALIGNANCY SCREENING']], colWidths=[6.6*inch])
        header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLORS['section']),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(header)
        
        differential = results.get('differential', {})
        wbc_count = results.get('cell_counts', {}).get('WBC', 0)
        
        # Check for N/C ratio analysis (if available)
        nc_ratio_data = results.get('nc_ratio_analysis', {})
        has_nc_ratio = nc_ratio_data.get('total_cells_analyzed', 0) > 0
        
        # Screen for leukemia indicators
        lymph_pct = differential.get('LYMPHOCYTE', {}).get('percentage', 0.0)
        neut_pct = differential.get('NEUTROPHIL', {}).get('percentage', 0.0)
        mono_pct = differential.get('MONOCYTE', {}).get('percentage', 0.0)
        
        findings = []
        
        # Screen for blast cells using N/C ratio (if available)
        if has_nc_ratio:
            blast_pct = nc_ratio_data.get('blast_cell_percentage', 0.0)
            blast_count = nc_ratio_data.get('blast_suspects_count', 0)
            avg_nc = nc_ratio_data.get('average_nc_ratio', 0.0)
            max_nc = nc_ratio_data.get('max_nc_ratio', 0.0)
            risk_level = nc_ratio_data.get('risk_level', 'NEGATIVE')
            
            if blast_count > 0:
                blast_status = "DETECTED"
                blast_color = COLORS['positive']
                blast_finding = f"{blast_count} blast-like cells\n({blast_pct:.1f}%)\nN/C: {avg_nc:.2f}"
                # Shorten clinical action to fit column
                if risk_level == 'URGENT':
                    blast_interp = "URGENT: Immediate hematology referral for bone marrow biopsy required."
                elif risk_level == 'HIGH':
                    blast_interp = "HIGH RISK: Urgent hematology consultation recommended."
                else:
                    blast_interp = "MODERATE: Clinical correlation and follow-up recommended."
                findings.append(f"Blast cells: {blast_count} detected ({blast_pct:.1f}%)")
            else:
                blast_status = "NEGATIVE"
                blast_color = COLORS['negative']
                blast_finding = f"N/C normal\nAvg: {avg_nc:.2f}\nMax: {max_nc:.2f}"
                blast_interp = "No blast-like cells detected."
        
        # Acute Lymphoblastic Leukemia (ALL) screening - count based
        if lymph_pct > 50:
            all_status = "SUSPICIOUS"
            all_color = COLORS['positive']
            all_finding = f"Lymphocyte predominance: {lymph_pct:.1f}%"
            all_interp = "Marked lymphocytosis - possible ALL. Urgent hematology referral."
            findings.append("ALL screening: Suspicious")
        else:
            all_status = "NEGATIVE"
            all_color = COLORS['negative']
            all_finding = f"Lymphocytes: {lymph_pct:.1f}%"
            all_interp = "No lymphocytic predominance."
        
        # Chronic Myeloid Leukemia (CML) screening - "left shift"
        if neut_pct > 75 or mono_pct > 15:
            cml_status = "SUSPICIOUS"
            cml_color = COLORS['positive']
            cml_finding = f"Neutrophils: {neut_pct:.1f}%\nMonocytes: {mono_pct:.1f}%"
            cml_interp = "Myeloid predominance - possible CML. Hematology referral."
            findings.append("CML screening: Suspicious")
        else:
            cml_status = "NEGATIVE"
            cml_color = COLORS['negative']
            cml_finding = f"Neutrophils: {neut_pct:.1f}%"
            cml_interp = "No myeloid predominance."
        
        leuk_data = [
            ['Type', 'Result', 'Findings', 'Clinical Action'],
        ]
        
        # Add N/C ratio blast screening if available
        if has_nc_ratio:
            leuk_data.append([
                'Blast Cells\n(N/C Ratio)',
                blast_status,
                blast_finding,
                blast_interp
            ])
        
        # Add count-based screenings
        leuk_data.extend([
            [
                'ALL\n(Acute Lympho.)',
                all_status,
                all_finding,
                all_interp
            ],
            [
                'CML\n(Chronic Myeloid)',
                cml_status,
                cml_finding,
                cml_interp
            ],
        ])
        
        leuk_table = Table(leuk_data, colWidths=[0.9*inch, 0.85*inch, 1.4*inch, 3.3*inch])
        leuk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLORS['light_bg']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 6.5),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            
            ('FONTSIZE', (0, 1), (-1, -1), 6),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('ALIGN', (2, 1), (-1, -1), 'LEFT'),
            ('WORDWRAP', (0, 0), (-1, -1), True),
            
            ('TEXTCOLOR', (1, 1), (1, 1), blast_color if has_nc_ratio else all_color),
            ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (1, 2), (1, 2), all_color if has_nc_ratio else cml_color),
            ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
            ('TEXTCOLOR', (1, 3), (1, 3), cml_color if has_nc_ratio else colors.grey),
            ('FONTNAME', (1, 3), (1, 3), 'Helvetica-Bold'),
            
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
            ('BOX', (0, 0), (-1, -1), 1, COLORS['border']),
            
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(leuk_table)
        
        # Important note
        if has_nc_ratio:
            note_text = ("<b>N/C Ratio Analysis Enabled:</b> Nucleus-to-cytoplasm ratio calculated from segmentation masks. "
                        "Blast cells (N/C >0.80) flagged for manual review. Atypical morphology requires hematologist confirmation.")
        else:
            note_text = ("<b>IMPORTANT:</b> This system classifies 5 normal WBC types only. "
                        "Atypical cells, blasts, or immature forms cannot be detected without N/C ratio analysis. "
                        "Manual review by hematologist required for definitive blast cell identification.")
        note = Paragraph(note_text, self.styles['DiagSmallText'])
        elements.append(Spacer(1, 3))
        elements.append(note)
        
        return elements
    
    def _build_footer(self) -> list:
        """Compact footer with disclaimers."""
        elements = []
        
        # Disclaimer
        disclaimer = Paragraph(
            "<b>DISCLAIMER:</b> This is an AI-assisted screening report for research and educational purposes only. "
            "NOT FOR DIAGNOSTIC USE. All findings require confirmation by trained medical professionals using "
            "standard laboratory methods (thick/thin smear microscopy, CBC, hemoglobin electrophoresis, flow cytometry).",
            self.styles['DiagTinyText']
        )
        elements.append(disclaimer)
        
        # Methods
        methods = Paragraph(
            f"<b>Methods:</b> YOLO11n object detection (malaria), ResNet34 classification (WBC, RBC), "
            f"Instance segmentation (RBC morphology). Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['DiagTinyText']
        )
        elements.append(Spacer(1, 2))
        elements.append(methods)
        
        return elements


def generate_diagnostic_summary(
    results: Dict[str, Any],
    output_path: str,
    patient_info: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate 1-page diagnostic summary PDF.
    
    Args:
        results: Analysis results from BloodSmearAnalyzer
        output_path: Where to save the PDF
        patient_info: Optional patient info dict with 'name', 'id', 'gender'
    
    Returns:
        Path to generated PDF
    """
    generator = DiagnosticSummaryPDF()
    return generator.generate(results, output_path, patient_info)


if __name__ == "__main__":
    # Test with sample data
    sample_results = {
        "malaria_detection": {
            "enabled": True,
            "method": "yolo_object_detection",
            "structure_count": 0,
            "detection_rate": 0.0,
            "structure_breakdown": {"ring": 0, "trophozoite": 0, "schizont": 0, "gametocyte": 0}
        },
        "shape_analysis": {
            "enabled": True,
            "sickle_cell_pct": 2.3,
            "sickle_cells": 6,
            "thalassemia_indicator_pct": 36.3,
            "thalassemia_indicators": 93,
            "shape_distribution": {"microcyte": 15.2, "target": 2.0, "teardrop": 34.4},
            "size_cv": 18.5
        },
        "multi_disorder_risks": {
            "thalassemia": {"level": "monitor", "percentage": 36.3}
        },
        "differential": {
            "NEUTROPHIL": {"count": 1, "percentage": 50.0},
            "LYMPHOCYTE": {"count": 0, "percentage": 0.0},
            "MONOCYTE": {"count": 0, "percentage": 0.0},
            "EOSINOPHIL": {"count": 1, "percentage": 50.0},
            "BASOPHIL": {"count": 0, "percentage": 0.0}
        },
        "cell_counts": {"RBC": 187, "WBC": 2, "Platelets": 0}
    }
    
    output = generate_diagnostic_summary(
        sample_results,
        "diagnostic_summary_test.pdf",
        patient_info={"name": "Test Patient", "id": "PH-12345"}
    )
    print(f"✅ Generated: {output}")
