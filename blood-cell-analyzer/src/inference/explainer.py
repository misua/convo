#!/usr/bin/env python3
"""
Plain-Language Diagnosis Explainer

Converts technical blood cell analysis findings into understandable
explanations for non-medical users.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class RiskLevel(Enum):
    """Traffic light risk levels."""
    LOW = "low"        # Green
    MODERATE = "moderate"  # Yellow
    HIGH = "high"      # Red


@dataclass
class PlainFinding:
    """A single finding explained in plain language."""
    technical_term: str
    plain_name: str
    icon: str
    status: str  # "normal", "elevated", "low"
    value: str
    explanation: str
    significance: str


@dataclass
class PlainDiagnosis:
    """Complete plain-language diagnosis report."""
    risk_level: RiskLevel
    risk_score: float
    traffic_light_color: str
    headline: str
    summary: str
    cell_count_summary: str
    findings: List[PlainFinding]
    what_this_means: str
    next_steps: List[str]
    disclaimer: str
    technical_summary: Optional[str] = None


class DiagnosisExplainer:
    """
    Converts technical analysis results to plain-language explanations.
    
    Uses template-based generation for consistency and medical accuracy.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        
        # Traffic light thresholds
        diag_config = self.config.get('diagnosis', {}).get('traffic_light', {})
        self.green_max = diag_config.get('green_max', 0.3)
        self.yellow_max = diag_config.get('yellow_max', 0.6)
        
        # Shape term translations
        self.shape_translations = {
            "normal": {
                "plain": "Normal cells",
                "icon": "✅",
                "explanation": "These cells have a healthy, round shape."
            },
            "microcyte": {
                "plain": "Smaller than typical cells",
                "icon": "📏",
                "explanation": "Some cells are smaller than normal, which can happen in various conditions."
            },
            "target": {
                "plain": "Bull's-eye shaped cells",
                "icon": "🎯",
                "explanation": "Some cells have an unusual pattern that looks like a target or bull's-eye."
            },
            "teardrop": {
                "plain": "Teardrop-shaped cells",
                "icon": "💧",
                "explanation": "Some cells have an elongated shape, like a teardrop."
            },
            "spherocyte": {
                "plain": "Round, dense cells",
                "icon": "⚪",
                "explanation": "Some cells are rounder and denser than typical red blood cells."
            },
            "irregular": {
                "plain": "Irregularly shaped cells",
                "icon": "⚠️",
                "explanation": "Some cells have unusual shapes that don't fit normal patterns."
            }
        }
        
        # Finding templates
        self.finding_templates = {
            "microcytosis": {
                "technical": "Microcytic cells",
                "plain": "Cell size",
                "icon": "📏",
                "normal_text": "Cell sizes are within normal range",
                "elevated_text": "{pct:.0f}% of cells are smaller than typical",
                "significance_normal": "This is a good sign - your cells are a healthy size.",
                "significance_elevated": "Smaller cells are sometimes seen in conditions like iron deficiency or thalassemia trait."
            },
            "hypochromia": {
                "technical": "Hypochromic cells",
                "plain": "Cell color",
                "icon": "🎨",
                "normal_text": "Cells have normal color (hemoglobin content)",
                "elevated_text": "{pct:.0f}% of cells appear paler than normal",
                "significance_normal": "Your cells contain a healthy amount of hemoglobin.",
                "significance_elevated": "Paler cells may indicate lower hemoglobin content, which can occur in iron deficiency or thalassemia."
            },
            "target_cells": {
                "technical": "Target cells (codocytes)",
                "plain": "Bull's-eye pattern",
                "icon": "🎯",
                "normal_text": "No unusual cell patterns detected",
                "elevated_text": "{pct:.1f}% of cells have a bull's-eye pattern",
                "significance_normal": "Cell patterns look normal.",
                "significance_elevated": "Bull's-eye patterns are commonly seen in thalassemia trait. Many people with this pattern live completely normal lives."
            },
            "teardrop_cells": {
                "technical": "Dacrocytes (teardrop cells)",
                "plain": "Teardrop shape",
                "icon": "💧",
                "normal_text": "No teardrop-shaped cells detected",
                "elevated_text": "{pct:.1f}% of cells have a teardrop shape",
                "significance_normal": "Cell shapes look healthy.",
                "significance_elevated": "Teardrop shapes can be associated with various blood conditions including thalassemia."
            },
            "rdw": {
                "technical": "RDW-proxy (size variation)",
                "plain": "Size variation",
                "icon": "📊",
                "normal_text": "Cell sizes are consistent (RDW: {val:.1f}%)",
                "elevated_text": "More variation in cell sizes than typical (RDW: {val:.1f}%)",
                "significance_normal": "Consistent cell sizes are a good sign.",
                "significance_elevated": "Higher variation in cell sizes can occur in various conditions."
            }
        }
        
        # Risk level messages
        self.risk_messages = {
            RiskLevel.LOW: {
                "color": "🟢",
                "headline": "Normal Findings",
                "summary": "Your blood cell analysis shows results within normal ranges.",
                "action": "No immediate follow-up is typically needed based on these findings."
            },
            RiskLevel.MODERATE: {
                "color": "🟡",
                "headline": "Moderate Findings - Follow-up Recommended",
                "summary": "Your analysis shows some findings that may warrant a conversation with your doctor.",
                "action": "We recommend discussing these results with a healthcare provider."
            },
            RiskLevel.HIGH: {
                "color": "🔴",
                "headline": "Significant Findings - Medical Consultation Advised",
                "summary": "Your analysis shows findings that should be reviewed by a healthcare professional.",
                "action": "Please schedule an appointment with your doctor to discuss these results."
            }
        }
        
        # Standard disclaimer
        self.disclaimer = (
            "⚠️ **Important:** This is an AI-assisted screening tool, NOT a medical diagnosis. "
            "The results should be interpreted by a qualified healthcare professional. "
            "This tool is for educational and research purposes only. "
            "Always consult with a doctor for proper diagnosis and treatment."
        )
    
    def get_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score."""
        if risk_score <= self.green_max:
            return RiskLevel.LOW
        elif risk_score <= self.yellow_max:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH
    
    def explain_results(
        self,
        analysis_results: dict,
        shape_stats: Optional[dict] = None
    ) -> PlainDiagnosis:
        """
        Generate plain-language diagnosis from analysis results.
        
        Args:
            analysis_results: Full analysis results dict
            shape_stats: Optional shape population statistics
            
        Returns:
            PlainDiagnosis with all explanations
        """
        findings = []
        
        # Extract values
        cell_counts = analysis_results.get('cell_counts', {})
        rbc_count = cell_counts.get('RBC', 0)
        
        # Get morphology results if available
        morphology = analysis_results.get('morphology', {})
        microcyte_pct = morphology.get('microcyte_percentage', 0)
        hypochromic_pct = morphology.get('hypochromic_percentage', 0)
        rdw_proxy = morphology.get('rdw_proxy', 12.0)
        
        # Get shape results if available
        if shape_stats:
            # Try shape_distribution first (from analyze_blood_smear), fallback to shape_percentages
            shape_pcts = shape_stats.get('shape_distribution', shape_stats.get('shape_percentages', {}))
            target_pct = shape_pcts.get('target', 0)
            teardrop_pct = shape_pcts.get('teardrop', 0)
            abnormality_index = shape_stats.get('abnormality_index', 0)
        else:
            target_pct = 0
            teardrop_pct = 0
            abnormality_index = 0
        
        # Calculate risk score
        risk_config = self.config.get('risk_scoring', {})
        risk_score = self._calculate_risk_score(
            microcyte_pct=microcyte_pct,
            hypochromic_pct=hypochromic_pct,
            rdw_proxy=rdw_proxy,
            shape_abnormality=abnormality_index,
            config=risk_config
        )
        
        risk_level = self.get_risk_level(risk_score)
        risk_msg = self.risk_messages[risk_level]
        
        # Build findings list
        
        # 1. Cell size (microcytosis)
        findings.append(self._create_finding(
            "microcytosis",
            microcyte_pct,
            threshold=10.0  # >10% is elevated
        ))
        
        # 2. Cell color (hypochromia)
        findings.append(self._create_finding(
            "hypochromia",
            hypochromic_pct,
            threshold=20.0  # >20% is elevated
        ))
        
        # 3. Target cells (if shape analysis available)
        if shape_stats:
            findings.append(self._create_finding(
                "target_cells",
                target_pct,
                threshold=5.0  # >5% is notable
            ))
            
            # 4. Teardrop cells
            findings.append(self._create_finding(
                "teardrop_cells",
                teardrop_pct,
                threshold=2.0  # >2% is notable
            ))
        
        # 5. Size variation (RDW)
        rdw_elevated = rdw_proxy > 14.5  # Normal range 11.5-14.5%
        findings.append(self._create_finding(
            "rdw",
            rdw_proxy,
            threshold=14.5,
            is_percentage=False
        ))
        
        # Cell count summary
        cell_summary = f"We analyzed **{rbc_count}** red blood cells from your sample."
        if rbc_count < 100:
            cell_summary += " (Note: More cells would improve accuracy)"
        
        # What this means section
        what_this_means = self._generate_what_this_means(
            risk_level, target_pct, teardrop_pct, microcyte_pct
        )
        
        # Next steps
        next_steps = self._generate_next_steps(risk_level)
        
        # Technical summary for optional display
        tech_summary = self._generate_technical_summary(
            analysis_results, shape_stats, risk_score
        )
        
        return PlainDiagnosis(
            risk_level=risk_level,
            risk_score=risk_score,
            traffic_light_color=risk_msg["color"],
            headline=risk_msg["headline"],
            summary=risk_msg["summary"],
            cell_count_summary=cell_summary,
            findings=findings,
            what_this_means=what_this_means,
            next_steps=next_steps,
            disclaimer=self.disclaimer,
            technical_summary=tech_summary
        )
    
    def _calculate_risk_score(
        self,
        microcyte_pct: float,
        hypochromic_pct: float,
        rdw_proxy: float,
        shape_abnormality: float,
        config: dict
    ) -> float:
        """Calculate composite thalassemia risk score."""
        # Get weights
        shape_weight = config.get('shape_abnormality_weight', 0.35)
        micro_weight = config.get('microcytic_weight', 0.30)
        hypo_weight = config.get('hypochromic_weight', 0.25)
        rdw_weight = config.get('rdw_weight', 0.10)
        
        # Normalize values to 0-1 scale
        micro_score = min(1.0, microcyte_pct / 50.0)  # 50% = max score
        hypo_score = min(1.0, hypochromic_pct / 50.0)
        
        # RDW: normal is 11.5-14.5, elevated > 14.5
        if rdw_proxy <= 14.5:
            rdw_score = 0.0
        else:
            rdw_score = min(1.0, (rdw_proxy - 14.5) / 5.0)  # 19.5 = max score
        
        # Composite score
        score = (
            shape_abnormality * shape_weight +
            micro_score * micro_weight +
            hypo_score * hypo_weight +
            rdw_score * rdw_weight
        )
        
        return min(1.0, score)
    
    def _create_finding(
        self,
        finding_type: str,
        value: float,
        threshold: float,
        is_percentage: bool = True
    ) -> PlainFinding:
        """Create a PlainFinding from template."""
        template = self.finding_templates[finding_type]
        
        is_elevated = value > threshold
        
        if is_percentage:
            value_str = f"{value:.1f}%"
        else:
            value_str = f"{value:.1f}"
        
        if is_elevated:
            status = "elevated"
            explanation = template["elevated_text"].format(pct=value, val=value)
            significance = template["significance_elevated"]
        else:
            status = "normal"
            explanation = template["normal_text"].format(pct=value, val=value)
            significance = template["significance_normal"]
        
        # Status icon
        status_icon = "⚠️" if is_elevated else "✅"
        
        return PlainFinding(
            technical_term=template["technical"],
            plain_name=template["plain"],
            icon=template["icon"],
            status=status,
            value=value_str,
            explanation=f"{status_icon} {explanation}",
            significance=significance
        )
    
    def _generate_what_this_means(
        self,
        risk_level: RiskLevel,
        target_pct: float,
        teardrop_pct: float,
        microcyte_pct: float
    ) -> str:
        """Generate the 'What This Means' section."""
        if risk_level == RiskLevel.LOW:
            return (
                "Your blood cell analysis shows results that are within normal ranges. "
                "The cells appear healthy in terms of size, shape, and color. "
                "This is reassuring, but remember that this is just one screening tool."
            )
        
        findings = []
        
        if target_pct > 5:
            findings.append("bull's-eye shaped cells")
        if teardrop_pct > 2:
            findings.append("teardrop-shaped cells")
        if microcyte_pct > 10:
            findings.append("smaller than typical cells")
        
        if findings:
            finding_text = ", ".join(findings)
            
            if risk_level == RiskLevel.MODERATE:
                return (
                    f"The patterns we found ({finding_text}) are sometimes seen in people with "
                    f"**thalassemia trait** - a common inherited blood condition. "
                    f"Many people have this trait and live completely normal, healthy lives. "
                    f"It's usually not a disease that needs treatment, but it's helpful to know about "
                    f"for family planning and to avoid unnecessary iron supplements."
                )
            else:  # HIGH
                return (
                    f"We found several patterns ({finding_text}) that are often associated with "
                    f"**thalassemia** or similar blood conditions. While this screening suggests "
                    f"these patterns are present, only a doctor can make a proper diagnosis. "
                    f"The good news is that thalassemia trait is very common and usually doesn't "
                    f"cause health problems. Your doctor can explain what this means for you."
                )
        
        return (
            "Some of your blood cell characteristics are outside typical ranges. "
            "This doesn't necessarily mean there's a problem, but it's worth discussing "
            "with your healthcare provider for proper evaluation."
        )
    
    def _generate_next_steps(self, risk_level: RiskLevel) -> List[str]:
        """Generate recommended next steps based on risk level."""
        if risk_level == RiskLevel.LOW:
            return [
                "📋 Save this report for your records",
                "🩺 Mention at your next routine checkup if you'd like",
                "✅ No urgent action needed based on these findings"
            ]
        elif risk_level == RiskLevel.MODERATE:
            return [
                "🏥 Visit your doctor to discuss these findings",
                "🧪 Ask about a **Hemoglobin Electrophoresis** test - this is the standard test to check for thalassemia",
                "📋 Bring this report to your appointment",
                "👨‍👩‍👧‍👦 If confirmed, consider informing family members as thalassemia trait is inherited"
            ]
        else:  # HIGH
            return [
                "🏥 **Schedule an appointment** with your doctor soon to discuss these findings",
                "🧪 Your doctor will likely order a **Complete Blood Count (CBC)** and **Hemoglobin Electrophoresis**",
                "📋 Print or save this report to share with your healthcare provider",
                "❓ Prepare questions about what these findings might mean for you",
                "💚 Remember: Many blood conditions are manageable with proper care"
            ]
    
    def _generate_technical_summary(
        self,
        analysis_results: dict,
        shape_stats: Optional[dict],
        risk_score: float
    ) -> str:
        """Generate technical summary for healthcare providers."""
        lines = [
            "### Technical Summary (For Healthcare Providers)",
            "",
            "| Metric | Value | Reference |",
            "|--------|-------|-----------|"
        ]
        
        cell_counts = analysis_results.get('cell_counts', {})
        morphology = analysis_results.get('morphology', {})
        
        lines.append(f"| RBC Count | {cell_counts.get('RBC', 'N/A')} | - |")
        lines.append(f"| Microcyte % | {morphology.get('microcyte_percentage', 0):.1f}% | <10% |")
        lines.append(f"| Hypochromic % | {morphology.get('hypochromic_percentage', 0):.1f}% | <20% |")
        lines.append(f"| RDW-proxy | {morphology.get('rdw_proxy', 0):.1f}% | 11.5-14.5% |")
        
        if shape_stats:
            shape_pcts = shape_stats.get('shape_percentages', {})
            lines.append(f"| Target cells | {shape_pcts.get('target', 0):.1f}% | <5% |")
            lines.append(f"| Teardrop cells | {shape_pcts.get('teardrop', 0):.1f}% | <2% |")
            lines.append(f"| Shape Abnormality Index | {shape_stats.get('abnormality_index', 0):.2f} | <0.15 |")
        
        lines.append(f"| **Composite Risk Score** | **{risk_score:.2f}** | <0.30 |")
        
        return "\n".join(lines)
    
    def format_as_markdown(self, diagnosis: PlainDiagnosis) -> str:
        """Format PlainDiagnosis as Markdown for display with clear comparison tables."""
        lines = [
            f"# 🩸 Your Blood Analysis Results",
            "",
            f"## {diagnosis.traffic_light_color} {diagnosis.headline}",
            "",
            diagnosis.summary,
            "",
            "---",
            "",
            f"### 📊 {diagnosis.cell_count_summary}",
            "",
            "---",
            "",
            "## 📋 Quick Summary Table",
            "",
            "| Test | Your Result | Normal Range | Status | What It Means |",
            "|------|-------------|--------------|--------|---------------|",
        ]
        
        # Add findings as table rows for easy scanning
        for finding in diagnosis.findings:
            # Clean up explanation to remove emoji/status icon for table
            clean_explanation = finding.explanation.replace("✅ ", "").replace("⚠️ ", "")
            
            # Determine status icon based on finding status
            status_icon = "✅" if finding.status == "normal" else "⚠️"
            status_text = "Normal" if finding.status == "normal" else "Abnormal"
            
            # Extract value from explanation if present, otherwise use finding.value
            value_text = finding.value
            
            # Get threshold from template (simplified)
            threshold_map = {
                "Cell size": "6-8 µm diameter",
                "Cell color": "Normal hemoglobin",
                "Bull's-eye pattern": "<5%",
                "Teardrop shape": "<2%",
                "Size variation": "11.5-14.5%"
            }
            threshold = threshold_map.get(finding.plain_name, "Normal")
            
            # Simplified significance
            significance = finding.significance.split(".")[0]  # First sentence only
            
            lines.append(f"| {finding.icon} {finding.plain_name} | {value_text} | {threshold} | {status_icon} {status_text} | {significance} |")
        
        lines.extend([
            "",
            "---",
            "",
            "## 📖 Detailed Explanations",
            "",
            "### What We Found:",
            ""
        ])
        
        # Add detailed findings
        for finding in diagnosis.findings:
            lines.append(f"**{finding.icon} {finding.plain_name}**")
            lines.append(f"> {finding.explanation}")
            lines.append(f"> *Clinical Note:* {finding.significance}")
            lines.append("")
        
        lines.extend([
            "---",
            "",
            "## 💡 What This Might Mean",
            "",
            diagnosis.what_this_means,
            "",
            "---",
            "",
            "## 👨‍⚕️ Recommended Next Steps",
            ""
        ])
        
        for step in diagnosis.next_steps:
            lines.append(f"- {step}")
        
        lines.extend([
            "",
            "---",
            "",
            diagnosis.disclaimer,
            ""
        ])
        
        # Add technical summary if available
        if diagnosis.technical_summary:
            lines.extend([
                "",
                "<details>",
                "<summary>📋 Technical Details (Click to expand)</summary>",
                "",
                diagnosis.technical_summary,
                "",
                "</details>"
            ])
        
        return "\n".join(lines)
