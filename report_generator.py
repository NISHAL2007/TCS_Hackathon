"""
report_generator.py
===================
Incident Audit Report and PDF Generator for Exam Portal Slowdown Root-Cause Analysis.
Generates structured executive markdown summaries and high-resolution ReportLab PDFs
grounded exclusively in verified telemetry and model predictions.
"""

from typing import Dict, Any, List, Optional
import io
import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)


class IncidentReportGenerator:
    """
    Generates structured markdown incident reports and production-grade PDF documents.
    """
    def __init__(self):
        pass

    def build_structured_report_data(
        self,
        incident_id: str,
        sample: Dict[str, Any],
        ml_prediction: Dict[str, Any],
        shap_explanation: Dict[str, Any],
        hypotheses_result: Dict[str, Any],
        time_window: str = "2026-10-29 09:30:00 - 11:40:00"
    ) -> Dict[str, Any]:
        """
        Synthesizes all diagnostic, ML, and telemetry facets into a unified report dictionary.
        """
        severity_info = hypotheses_result.get("severity_assessment", {})
        top_hyp = hypotheses_result.get("top_rule_hypothesis", "Unknown")
        top_score = hypotheses_result.get("top_rule_score", 0.0)
        next_check = hypotheses_result.get("next_best_diagnostic_check", {})

        # Evidence-grounded corrective actions based on leading root cause
        corrective_actions = self._generate_corrective_actions(top_hyp, sample)
        prevention_recommendations = self._generate_prevention_recommendations(top_hyp)

        return {
            "incident_id": incident_id,
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "observation_timestamp": str(sample.get("timestamp", "N/A")),
            "time_window": time_window,
            "severity": severity_info.get("severity_level", "Medium"),
            "sla_target": severity_info.get("sla_target", "Investigate within 15 mins"),
            "predicted_incident_type": ml_prediction.get("predicted_incident_type", "Unknown"),
            "ml_confidence_pct": round(ml_prediction.get("confidence", 0.0) * 100, 1),
            "top_rule_hypothesis": top_hyp,
            "top_rule_score_pct": top_score,
            "telemetry_summary": {
                "active_users": sample.get("active_users", 0),
                "response_time_ms": sample.get("response_time_ms", 0),
                "cpu_usage_pct": sample.get("cpu_usage_pct", 0),
                "memory_usage_pct": sample.get("memory_usage_pct", 0),
                "db_query_time_ms": sample.get("db_query_time_ms", 0),
                "bandwidth_mbps": sample.get("bandwidth_mbps", 0),
                "error_count": sample.get("error_count", 0),
                "network_latency_ms": sample.get("network_latency_ms", 0),
                "packet_loss_pct": sample.get("packet_loss_pct", 0),
                "complaint_text": sample.get("complaint_text", "None reported")
            },
            "shap_attribution": shap_explanation.get("contributions", [])[:5],
            "ranked_hypotheses": hypotheses_result.get("hypotheses", []),
            "next_best_diagnostic_check": next_check,
            "corrective_actions": corrective_actions,
            "prevention_recommendations": prevention_recommendations,
            "limitations_disclosure": (
                "This report was compiled autonomously by the AI Root-Cause Investigation System. "
                "Feature attributions explain statistical classifier mechanics and do not replace physical verification. "
                "Recommendations must be reviewed and executed by exam infrastructure administrators."
            )
        }

    def generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """
        Formats report dictionary into rich markdown for interactive viewing.
        """
        t = report_data["telemetry_summary"]
        s = report_data["severity"]

        md = f"""# 📋 INCIDENT ROOT-CAUSE INVESTIGATION REPORT: `{report_data['incident_id']}`

**Generated:** {report_data['generated_at']} | **Observation Window:** {report_data['observation_timestamp']}  
**Assessed Severity:** **{s}** | **SLA Mitigation Target:** {report_data['sla_target']}

---

### 1. Executive Summary & Diagnostic Findings
- **Machine Learning Classification:** `{report_data['predicted_incident_type']}` (Model Confidence: **{report_data['ml_confidence_pct']}%**)
- **Leading Hypothesis (Rule Engine):** `{report_data['top_rule_hypothesis']}` (Evidence Match: **{report_data['top_rule_score_pct']}%**)
- **Student Impact Narrative:** "{t['complaint_text']}"
- **Key Degraded Telemetry:** Response Time: `{t['response_time_ms']} ms` | DB Query: `{t['db_query_time_ms']} ms` | CPU: `{t['cpu_usage_pct']}%` | Errors: `{t['error_count']}`

---

### 2. SHAP Model Attribution (Top Features Driving Prediction)
| Telemetry Feature | Observed Value | SHAP Impact | Direction |
| :--- | :--- | :--- | :--- |
"""
        for feat in report_data["shap_attribution"]:
            md += f"| `{feat['feature']}` | {feat['value']} | **{feat['attribution']:+.3f}** | {feat['impact_direction']} |\n"

        md += """
---

### 3. Competing Root-Cause Hypotheses Evaluation
"""
        for h in report_data["ranked_hypotheses"]:
            status_emoji = "🟢" if h["status"] == "Leading Hypothesis" else ("🟡" if h["status"] == "Plausible Contributor" else "⚪")
            md += f"#### {status_emoji} {h['hypothesis']} (Evidence Score: **{h['evidence_score_pct']}%**)\n"
            md += f"- **Supporting Evidence:** {'; '.join(h['supporting_evidence'])}\n"
            md += f"- **Contradictory Evidence:** {'; '.join(h['contradictory_evidence'])}\n"
            md += f"- **Missing Information:** {h['missing_information']}\n\n"

        chk = report_data["next_best_diagnostic_check"]
        md += f"""---

### 4. Recommended Next Best Diagnostic Check
**Inspection Action:** `{chk.get('check_title', 'N/A')}`  
**Recommended Tool / Probe:** `{chk.get('recommended_tool', 'N/A')}`  
**Rationale:** {chk.get('why_recommended', 'N/A')}  
**Telemetry Status:** `{chk.get('telemetry_status', 'N/A')}`

---

### 5. Corrective Actions (Post-Validation)
"""
        for i, act in enumerate(report_data["corrective_actions"], 1):
            md += f"{i}. **{act['action']}**: {act['details']}\n"

        md += """
### 6. Prevention Recommendations
"""
        for i, prev in enumerate(report_data["prevention_recommendations"], 1):
            md += f"{i}. **{prev['title']}**: {prev['rationale']}\n"

        md += f"""
---
> ⚠️ **Limitation & Verification Notice:** {report_data['limitations_disclosure']}
"""
        return md

    def generate_pdf_report(self, report_data: Dict[str, Any]) -> bytes:
        """
        Builds a polished multi-page PDF document using ReportLab.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1e293b")
        )
        h2_style = ParagraphStyle(
            "DocH2",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155")
        )
        body_bold = ParagraphStyle(
            "DocBodyBold",
            parent=body_style,
            fontName="Helvetica-Bold"
        )

        story = []

        # Header Title
        story.append(Paragraph(f"Exam Incident Root-Cause Investigation: {report_data['incident_id']}", title_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"<b>Generated:</b> {report_data['generated_at']} | <b>Window:</b> {report_data['observation_timestamp']} | <b>Severity:</b> <font color='red'><b>{report_data['severity']}</b></font>",
            body_style
        ))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceBefore=2, spaceAfter=8))

        # Executive Metrics Table
        t = report_data["telemetry_summary"]
        metric_data = [
            [
                Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style),
                Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style)
            ],
            [
                Paragraph("Response Time", body_style), Paragraph(f"{t['response_time_ms']} ms", body_bold),
                Paragraph("DB Query Time", body_style), Paragraph(f"{t['db_query_time_ms']} ms", body_bold)
            ],
            [
                Paragraph("CPU Utilization", body_style), Paragraph(f"{t['cpu_usage_pct']}%", body_bold),
                Paragraph("Memory Utilization", body_style), Paragraph(f"{t['memory_usage_pct']}%", body_bold)
            ],
            [
                Paragraph("Active Exam Users", body_style), Paragraph(f"{t['active_users']}", body_bold),
                Paragraph("HTTP Error Count", body_style), Paragraph(f"{t['error_count']}", body_bold)
            ],
            [
                Paragraph("Network Latency", body_style), Paragraph(f"{t['network_latency_ms']} ms", body_bold),
                Paragraph("Packet Loss", body_style), Paragraph(f"{t['packet_loss_pct']}%", body_bold)
            ]
        ]
        t_table = Table(metric_data, colWidths=[130, 130, 130, 130])
        t_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_table)
        story.append(Spacer(1, 10))

        # ML & Diagnostic Verdict
        verdict_data = [
            [
                Paragraph("<b>Model Prediction</b>", body_style),
                Paragraph(f"{report_data['predicted_incident_type']} (Confidence: {report_data['ml_confidence_pct']}%)", body_bold)
            ],
            [
                Paragraph("<b>Rule Engine Hypothesis</b>", body_style),
                Paragraph(f"{report_data['top_rule_hypothesis']} (Evidence Match: {report_data['top_rule_score_pct']}%)", body_bold)
            ],
            [
                Paragraph("<b>Student Complaint Log</b>", body_style),
                Paragraph(f"<i>\"{t['complaint_text']}\"</i>", body_style)
            ]
        ]
        v_table = Table(verdict_data, colWidths=[160, 360])
        v_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f8fafc")),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(v_table)
        story.append(Spacer(1, 10))

        # SHAP Attributions Table
        story.append(Paragraph("Top Driving Features (SHAP Explainability)", h2_style))
        shap_table_data = [
            [Paragraph("<b>Feature</b>", body_style), Paragraph("<b>Observed</b>", body_style),
             Paragraph("<b>Attribution</b>", body_style), Paragraph("<b>Impact</b>", body_style)]
        ]
        for item in report_data["shap_attribution"]:
            shap_table_data.append([
                Paragraph(f"<code>{item['feature']}</code>", body_style),
                Paragraph(str(item["value"]), body_style),
                Paragraph(f"{item['attribution']:+.3f}", body_bold),
                Paragraph(item["impact_direction"], body_style)
            ])
        s_table = Table(shap_table_data, colWidths=[160, 100, 120, 140])
        s_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(s_table)
        story.append(Spacer(1, 10))

        # Competing Hypotheses Summary
        story.append(Paragraph("Competing Hypotheses Evaluation", h2_style))
        for h in report_data["ranked_hypotheses"][:4]:
            h_text = f"<b>{h['hypothesis']}</b> (Score: {h['evidence_score_pct']}%) - <i>{h['status']}</i><br/>"
            h_text += f"• <b>Supporting:</b> {'; '.join(h['supporting_evidence'][:2])}<br/>"
            h_text += f"• <b>Missing Info:</b> {h['missing_information']}"
            story.append(Paragraph(h_text, body_style))
            story.append(Spacer(1, 4))

        # Next Best Diagnostic Check
        chk = report_data["next_best_diagnostic_check"]
        story.append(Paragraph("Recommended Next Diagnostic Action", h2_style))
        chk_text = f"<b>Inspection:</b> {chk.get('check_title', 'N/A')}<br/>"
        chk_text += f"<b>Tool / Probe:</b> {chk.get('recommended_tool', 'N/A')} | <b>Status:</b> {chk.get('telemetry_status', 'N/A')}<br/>"
        chk_text += f"<b>Rationale:</b> {chk.get('why_recommended', 'N/A')}"
        story.append(Paragraph(chk_text, body_style))
        story.append(Spacer(1, 8))

        # Corrective & Prevention Actions
        story.append(Paragraph("Corrective Actions & Prevention Checklist", h2_style))
        for act in report_data["corrective_actions"][:3]:
            story.append(Paragraph(f"• <b>{act['action']}:</b> {act['details']}", body_style))
        story.append(Spacer(1, 4))
        for prev in report_data["prevention_recommendations"][:2]:
            story.append(Paragraph(f"• <b>[Prevention] {prev['title']}:</b> {prev['rationale']}", body_style))

        # Footer Notice
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceBefore=2, spaceAfter=4))
        story.append(Paragraph(f"<b>Disclaimer:</b> {report_data['limitations_disclosure']}", ParagraphStyle("Footnote", parent=body_style, fontSize=7, leading=9, textColor=colors.HexColor("#64748b"))))

        doc.build(story)
        return buffer.getvalue()

    def _generate_corrective_actions(self, incident_type: str, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generates evidence-grounded corrective remediation steps."""
        if incident_type == "Database Bottleneck":
            return [
                {"action": "Kill Blocking Transactions", "details": "Query pg_stat_activity / SHOW PROCESSLIST and terminate transactions holding exclusive locks on exam answer tables."},
                {"action": "Temporarily Increase Pool Capacity", "details": "Scale PgBouncer or MySQL connection pool max_connections by 50% to prevent queue backlog."},
                {"action": "Add Emergency Query Indexing", "details": "Ensure composite indexes exist on (student_id, exam_id, question_id) for rapid idempotent upserts."}
            ]
        elif incident_type == "CPU/Server Overload":
            return [
                {"action": "Horizontal Worker Auto-Scaling", "details": "Spin up 4 additional exam portal app containers behind the load balancer to distribute compute."},
                {"action": "Disable Non-Critical Background Tasks", "details": "Pause non-essential cron jobs, report generation, and student formula image rendering during live exam."},
                {"action": "Enable Rate-Limiting on Page Refresh", "details": "Enforce 500ms debounce on browser question navigation to stop student refresh loops."}
            ]
        elif incident_type == "Memory Pressure":
            return [
                {"action": "Execute Graceful Worker Rolling Restart", "details": "Perform a zero-downtime rolling restart of web server workers to reclaim leaked heap RAM."},
                {"action": "Enlarge Container Memory Limits", "details": "Increase pod memory limit from 2Gi to 4Gi in Kubernetes/Docker spec."},
                {"action": "Tune Garbage Collector Parameters", "details": "Adjust JVM `-XX:+UseG1GC` with `-XX:InitiatingHeapOccupancyPercent=45` for continuous compaction."}
            ]
        elif incident_type == "Network Congestion":
            return [
                {"action": "Shift Static Exam Assets to CDN", "details": "Route diagrams, audio, and JS bundles through Cloudflare/CloudFront edge nodes."},
                {"action": "Engage Upstream ISP Gateway Support", "details": "Report border gateway packet loss and re-route traffic via secondary multi-homed BGP link."},
                {"action": "Lower Asset Payload Resolution", "details": "Enable WebP compression and lazy-loading on high-res exam question diagrams."}
            ]
        elif incident_type == "Application/Dependency Failure":
            return [
                {"action": "Re-establish Session Store Connection", "details": "Inspect Redis master health; restart sentinel cluster if quorum is degraded."},
                {"action": "Deploy Hotfix or Roll Back Build", "details": "Revert recent application commit if 500 exceptions stem from syntax or unhandled null pointer."},
                {"action": "Enable Circuit Breaker Fallback", "details": "Activate circuit breaker to allow local browser caching of answers when downstream validator is down."}
            ]
        else:
            return [
                {"action": "Standard Observability Monitoring", "details": "Maintain telemetry monitoring; no immediate corrective intervention required."}
            ]

    def _generate_prevention_recommendations(self, incident_type: str) -> List[Dict[str, str]]:
        """Generates architectural prevention strategies for exam portal stability."""
        return [
            {"title": "Automated Pre-Exam Load Simulation", "rationale": "Run 2x peak synthetic concurrency testing (e.g., Locust or k6) 48 hours prior to high-stakes examinations."},
            {"title": "Read/Write Database Separation", "rationale": "Direct student read requests (question viewing) to read-replicas, reserving master exclusively for answer writes."},
            {"title": "Graceful Offline Client-Side Answer Queue", "rationale": "Implement IndexedDB local storage in the student's browser so answers buffer safely during network drops."},
            {"title": "Proactive Metric Drift Alerting", "rationale": "Configure Prometheus alerts when response time exceeds 400ms for 3 consecutive collection intervals."}
        ]
