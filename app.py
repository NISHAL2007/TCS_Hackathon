"""
app.py
======
ExamGuard AI — Premium Real-Time Incident Intelligence & Observability Platform.
Engineered for online exam reliability, autonomous multi-telemetry correlation,
SHAP explainability, competing root-cause hypotheses, active diagnostics,
what-if counterfactual simulations, complaint NLP, and executive PDF audit reporting.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# Core application modules (preserved from existing working architecture)
from data_generator import (
    get_raw_data, FEATURE_COLUMNS, NORMAL_THRESHOLDS, ALL_INCIDENT_TYPES
)
from model import get_trained_model_suite
from explainability import IncidentExplainer
from correlation import TemporalCorrelationEngine
from diagnostics import DiagnosticEngine
from counterfactual import CounterfactualEngine
from complaint_analysis import ComplaintCategorizer
from report_generator import IncidentReportGenerator

# ------------------------------------------------------------------------------
# PAGE & THEME CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="ExamGuard AI | Incident Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Datadog/Grafana Observability Shell Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Dark Theme Settings */
    html, body, [class*="css"], .stApp {
        background-color: #0B0F19 !important;
        color: #F1F5F9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    code, pre, .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Navigation Shell */
    .top-navbar {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    /* Status Badges */
    .badge-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-high {
        background: rgba(249, 115, 22, 0.15);
        color: #FB923C;
        border: 1px solid #F97316;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid #F59E0B;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-normal {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-ai {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid #6366F1;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Observability KPI Cards */
    .kpi-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #374151;
    }

    .kpi-label {
        color: #94A3B8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-val {
        color: #F8FAFC;
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 4px;
        letter-spacing: -0.02em;
    }

    .kpi-sub {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Graph & Content Containers */
    .panel-container {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }

    .panel-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1F2937;
        padding-bottom: 8px;
    }

    /* Live Pulsing Dot */
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        margin-right: 6px;
        box-shadow: 0 0 8px #10B981;
    }

    .live-dot-red {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #EF4444;
        margin-right: 6px;
        box-shadow: 0 0 8px #EF4444;
    }

    /* Custom Streamlit adjustments for dark theme */
    div[data-testid="stSidebar"] {
        background-color: #0D1117 !important;
        border-right: 1px solid #1F2937 !important;
    }

    div[data-testid="stSidebar"] button {
        background-color: #161B22 !important;
        color: #F1F5F9 !important;
        border: 1px solid #30363D !important;
    }

    .stSelectbox label, .stSlider label {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    /* Clean evidence pill tags */
    .pill-supports {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        display: inline-block;
        margin: 2px;
    }

    .pill-contradicts {
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        display: inline-block;
        margin: 2px;
    }

    .pill-neutral {
        background: rgba(148, 163, 184, 0.1);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# CACHED SERVICE INITIALIZATION
# ------------------------------------------------------------------------------
@st.cache_resource
def init_examguard_services():
    """Initializes and caches the complete telemetry, model, and diagnostic suite."""
    df_raw = get_raw_data()
    model_suite = get_trained_model_suite()
    explainer = IncidentExplainer(model_suite.active_model, df_raw)
    correlation_engine = TemporalCorrelationEngine(df_raw)
    diagnostic_engine = DiagnosticEngine(NORMAL_THRESHOLDS)
    counterfactual_engine = CounterfactualEngine(model_suite, diagnostic_engine)
    complaint_categorizer = ComplaintCategorizer()
    report_generator = IncidentReportGenerator()
    return (
        df_raw, model_suite, explainer, correlation_engine,
        diagnostic_engine, counterfactual_engine, complaint_categorizer, report_generator
    )


(
    DF_RAW, MODEL_SUITE, EXPLAINER, CORRELATION_ENGINE,
    DIAGNOSTIC_ENGINE, COUNTERFACTUAL_ENGINE, COMPLAINT_CATEGORIZER, REPORT_GENERATOR
) = init_examguard_services()


# Shared Plotly Dark Theme Config
PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(17, 24, 39, 0)",
    plot_bgcolor="rgba(17, 24, 39, 0.5)",
    font=dict(color="#94A3B8", family="Inter, sans-serif", size=11),
    xaxis=dict(
        gridcolor="#1F2937",
        linecolor="#374151",
        zerolinecolor="#1F2937",
        tickfont=dict(color="#94A3B8")
    ),
    yaxis=dict(
        gridcolor="#1F2937",
        linecolor="#374151",
        zerolinecolor="#1F2937",
        tickfont=dict(color="#94A3B8")
    ),
    margin=dict(l=35, r=20, t=35, b=30),
    hoverlabel=dict(
        bgcolor="#1E293B",
        bordercolor="#38BDF8",
        font=dict(color="#F8FAFC", family="JetBrains Mono", size=12)
    )
)


# ------------------------------------------------------------------------------
# PERSISTENT SIDEBAR NAVIGATION & TELEMETRY SELECTOR
# ------------------------------------------------------------------------------
with st.sidebar:
    # Product Brand & Logo
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom: 16px;">
        <div style="background: linear-gradient(135deg, #3B82F6, #6366F1); width:32px; height:32px; border-radius:6px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:16px; color:white;">
            EG
        </div>
        <div>
            <div style="font-weight:700; font-size:1.1rem; color:#F8FAFC; letter-spacing:-0.01em;">ExamGuard AI</div>
            <div style="font-size:0.7rem; color:#64748B; letter-spacing:0.04em; text-transform:uppercase;">Incident Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status Pill
    first_anom = CORRELATION_ENGINE.detect_first_anomaly()
    has_active_anom = first_anom["has_anomaly"]
    dot_class = "live-dot-red" if has_active_anom else "live-dot"
    status_label = "INCIDENT ACTIVE" if has_active_anom else "SYSTEM HEALTHY"
    status_color = "#EF4444" if has_active_anom else "#10B981"

    st.markdown(f"""
    <div style="background:#161B22; border:1px solid #30363D; border-radius:6px; padding:8px 12px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:0.75rem; color:#94A3B8;"><span class="{dot_class}"></span>{status_label}</span>
        <span style="font-size:0.7rem; color:#38BDF8; font-family:'JetBrains Mono';">36 SNAPSHOTS</span>
    </div>
    """, unsafe_allow_html=True)

    # Navigation menu
    PAGE_OPTIONS = [
        "Overview",
        "Incident Timeline",
        "Root-Cause Analysis",
        "SHAP Explainability",
        "What-If Simulator",
        "Complaint Analytics",
        "Incident Reports",
        "Model Evaluation"
    ]

    selected_nav = st.radio(
        "NAVIGATION WORKSPACES",
        PAGE_OPTIONS,
        index=0,
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#94A3B8; margin-bottom:6px;'>TELEMETRY SNAPSHOT SELECTOR</div>", unsafe_allow_html=True)

    # Snapshot Picker
    timestamps = [f"[{i:02d}] {str(row['timestamp'])[-8:]} — {row['incident_type']}" for i, row in DF_RAW.iterrows()]
    selected_idx = st.selectbox(
        "Choose timestamp to investigate:",
        options=list(range(len(DF_RAW))),
        format_func=lambda i: timestamps[i],
        index=8  # Default to severe Database Bottleneck snapshot
    )

    selected_sample = DF_RAW.iloc[selected_idx].to_dict()
    selected_ts_str = str(selected_sample["timestamp"])

    # Precompute sample diagnostic and ML results
    sample_pred = MODEL_SUITE.predict_sample(selected_sample)
    sample_diag = DIAGNOSTIC_ENGINE.evaluate_sample_hypotheses(selected_sample, sample_pred)
    sev_info = sample_diag["severity_assessment"]

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.75rem; color:#64748B;">
        <b>Target Scenario:</b> <code style="color:#94A3B8;">{selected_sample.get('scenario_id', 'N/A')}</code><br/>
        <b>Ground Truth:</b> <span style="color:#F1F5F9;">{selected_sample['incident_type']}</span><br/>
        <b>Severity SLA:</b> <span style="color:#F87171;">{sev_info['sla_target']}</span>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# SLIM TOP OBSERVABILITY BAR
# ------------------------------------------------------------------------------
sev_badge_class = (
    "badge-critical" if sev_info["severity_level"] == "Critical"
    else ("badge-high" if sev_info["severity_level"] == "High"
    else ("badge-medium" if sev_info["severity_level"] == "Medium"
    else "badge-normal"))
)

st.markdown(f"""
<div class="top-navbar">
    <div style="display:flex; align-items:center; gap:16px;">
        <span class="{sev_badge_class}">{sev_info['severity_level']} SEVERITY</span>
        <div>
            <span style="font-size:0.88rem; font-weight:600; color:#F8FAFC;">Incident Focus: {selected_sample['incident_type']}</span>
            <span style="font-size:0.78rem; color:#64748B; margin-left:12px;">Timestamp: <code style="color:#38BDF8;">{selected_ts_str}</code></span>
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:20px;">
        <div style="text-align:right;">
            <div style="font-size:0.70rem; color:#64748B; text-transform:uppercase;">Model Verdict</div>
            <div style="font-size:0.85rem; font-weight:600; color:#818CF8;">{sample_pred['predicted_incident_type']} ({sample_pred['confidence']:.1%})</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.70rem; color:#64748B; text-transform:uppercase;">Top Rule Hypothesis</div>
            <div style="font-size:0.85rem; font-weight:600; color:#34D399;">{sample_diag['top_rule_hypothesis']} ({sample_diag['top_rule_score']}%)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. OVERVIEW DASHBOARD (VISUAL-FIRST)
# ==============================================================================
if selected_nav == "Overview":
    # 1. Compact KPI Strip
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)

    with kpi_c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Portal Response Time</div>
            <div class="kpi-val" style="color:{'#EF4444' if selected_sample['response_time_ms'] > 1000 else '#10B981'};">{selected_sample['response_time_ms']:,} ms</div>
            <div class="kpi-sub">Baseline: &le; {NORMAL_THRESHOLDS['response_time_ms']} ms ({((selected_sample['response_time_ms']-NORMAL_THRESHOLDS['response_time_ms'])/NORMAL_THRESHOLDS['response_time_ms'])*100:+.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">DB Query Latency</div>
            <div class="kpi-val" style="color:{'#EF4444' if selected_sample['db_query_time_ms'] > 500 else '#10B981'};">{selected_sample['db_query_time_ms']:,} ms</div>
            <div class="kpi-sub">Baseline: &le; {NORMAL_THRESHOLDS['db_query_time_ms']} ms ({((selected_sample['db_query_time_ms']-NORMAL_THRESHOLDS['db_query_time_ms'])/NORMAL_THRESHOLDS['db_query_time_ms'])*100:+.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Server CPU Load</div>
            <div class="kpi-val" style="color:{'#EF4444' if selected_sample['cpu_usage_pct'] > 85 else ('#F59E0B' if selected_sample['cpu_usage_pct'] > 70 else '#10B981')};">{selected_sample['cpu_usage_pct']:.1f}%</div>
            <div class="kpi-sub">Baseline: &le; {NORMAL_THRESHOLDS['cpu_usage_pct']}%</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">HTTP 5xx Error Spike</div>
            <div class="kpi-val" style="color:{'#EF4444' if selected_sample['error_count'] > 5 else '#10B981'};">{selected_sample['error_count']}</div>
            <div class="kpi-sub">Tolerance: &le; {NORMAL_THRESHOLDS['error_count']} errors/window</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Exam Takers</div>
            <div class="kpi-val" style="color:#38BDF8;">{selected_sample['active_users']:,}</div>
            <div class="kpi-sub">Correlated telemetry streams: 9/9</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Main Performance Timeline & Diagnostic Summary
    main_col1, main_col2 = st.columns([2.3, 1])

    with main_col1:
        st.markdown('<div class="panel-header"><span>Telemetry Performance Timeline & Anomaly Shading</span><span style="font-size:0.75rem; color:#64748B;">Dual Synchronized Scales</span></div>', unsafe_allow_html=True)

        # Plotly dual-subplot performance timeline to avoid cramming ms vs %
        fig_perf = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=["Latency Metrics (ms)", "Compute & Errors (%, Count)"],
            row_heights=[0.6, 0.4]
        )

        # Top row: Latency (Response time, DB latency, Network latency)
        fig_perf.add_trace(
            go.Scatter(
                x=DF_RAW["timestamp"], y=DF_RAW["response_time_ms"],
                name="Response Time (ms)", mode="lines+markers",
                line=dict(color="#38BDF8", width=2.5),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        fig_perf.add_trace(
            go.Scatter(
                x=DF_RAW["timestamp"], y=DF_RAW["db_query_time_ms"],
                name="DB Query Time (ms)", mode="lines",
                line=dict(color="#EC4899", width=2, dash="dot")
            ),
            row=1, col=1
        )
        fig_perf.add_trace(
            go.Scatter(
                x=DF_RAW["timestamp"], y=DF_RAW["network_latency_ms"],
                name="Network Latency (ms)", mode="lines",
                line=dict(color="#A855F7", width=1.5)
            ),
            row=1, col=1
        )

        # Threshold reference for response time
        fig_perf.add_hline(
            y=NORMAL_THRESHOLDS["response_time_ms"], line_dash="dash",
            line_color="rgba(239, 68, 68, 0.5)", annotation_text="Response Baseline (350ms)",
            annotation_position="top right", row=1, col=1
        )

        # Bottom row: CPU, Memory, Errors
        fig_perf.add_trace(
            go.Scatter(
                x=DF_RAW["timestamp"], y=DF_RAW["cpu_usage_pct"],
                name="CPU Usage (%)", mode="lines",
                line=dict(color="#F59E0B", width=2)
            ),
            row=2, col=1
        )
        fig_perf.add_trace(
            go.Scatter(
                x=DF_RAW["timestamp"], y=DF_RAW["memory_usage_pct"],
                name="Memory Usage (%)", mode="lines",
                line=dict(color="#10B981", width=1.5)
            ),
            row=2, col=1
        )
        fig_perf.add_trace(
            go.Bar(
                x=DF_RAW["timestamp"], y=DF_RAW["error_count"],
                name="Error Count", marker_color="rgba(239, 68, 68, 0.65)"
            ),
            row=2, col=1
        )

        # Mark selected snapshot on both plots using ISO string
        fig_perf.add_vline(
            x=str(selected_sample["timestamp"]), line_width=1.5,
            line_dash="dash", line_color="#F8FAFC"
        )

        fig_perf.update_layout(
            **PLOTLY_DARK_LAYOUT,
            height=380,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with main_col2:
        st.markdown('<div class="panel-header"><span>Diagnostic Synthesis</span><span class="badge-ai">AI RCA</span></div>', unsafe_allow_html=True)

        top_hyp = sample_diag["hypotheses"][0]
        st.markdown(f"""
        <div style="background:#161B22; border:1px solid #1F2937; border-radius:6px; padding:12px 14px; margin-bottom:12px;">
            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase;">Primary Incident Suspect</div>
            <div style="font-size:1.15rem; font-weight:700; color:#38BDF8; margin-top:2px;">{top_hyp['hypothesis']}</div>
            <div style="font-size:0.80rem; color:#10B981; font-weight:600; margin-top:2px;">Evidence Score: {top_hyp['evidence_score_pct']}% Match</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#94A3B8; margin-bottom:4px;'>KEY SUPPORTING EVIDENCE</div>", unsafe_allow_html=True)
        for sup in top_hyp["supporting_evidence"][:2]:
            st.markdown(f"<div class='pill-supports'>✓ {sup}</div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#94A3B8; margin-top:8px; margin-bottom:4px;'>CONTRADICTORY / MISSING SIGNAL</div>", unsafe_allow_html=True)
        con_text = top_hyp["contradictory_evidence"][0] if top_hyp["contradictory_evidence"] and "No contradictory" not in top_hyp["contradictory_evidence"][0] else top_hyp["missing_information"][:70] + "..."
        st.markdown(f"<div class='pill-contradicts'>! {con_text}</div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.75rem; font-weight:600; color:#94A3B8; margin-top:12px; margin-bottom:4px;'>NEXT BEST DIAGNOSTIC ACTION</div>", unsafe_allow_html=True)
        chk = sample_diag["next_best_diagnostic_check"]
        st.markdown(f"""
        <div style="background:rgba(99, 102, 241, 0.08); border:1px solid #4338CA; border-radius:6px; padding:8px 10px; font-size:0.80rem; color:#C7D2FE;">
            <b>{chk['check_title']}</b><br/>
            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#93C5FD;">Tool: {chk['recommended_tool']}</span>
        </div>
        """, unsafe_allow_html=True)

    # 3. Compact Incident Event Stream
    st.markdown('<div class="panel-header"><span>Chronological Event & Anomaly Stream</span><span style="font-size:0.75rem; color:#94A3B8;">Click Snapshot to Inspect</span></div>', unsafe_allow_html=True)

    # Group significant events
    event_cols = st.columns(6)
    notable_indices = [0, 8, 14, 19, 24, 30]  # Key representative scenario snapshots
    for col, idx in zip(event_cols, notable_indices):
        if idx < len(DF_RAW):
            row = DF_RAW.iloc[idx]
            is_active_card = idx == selected_idx
            border_color = "#38BDF8" if is_active_card else "#1F2937"
            bg_color = "rgba(56, 189, 248, 0.08)" if is_active_card else "#111827"

            with col:
                st.markdown(f"""
                <div style="background:{bg_color}; border:1px solid {border_color}; border-radius:6px; padding:10px; text-align:center;">
                    <div style="font-family:'JetBrains Mono'; font-size:0.72rem; color:#38BDF8;">{str(row['timestamp'])[-8:-3]}</div>
                    <div style="font-size:0.80rem; font-weight:600; color:#F1F5F9; margin:4px 0;">{row['incident_type'][:16]}</div>
                    <div style="font-size:0.72rem; color:{'#EF4444' if row['response_time_ms'] > 1000 else '#10B981'};">{row['response_time_ms']} ms | {row['error_count']} err</div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# 2. INCIDENT TIMELINE
# ==============================================================================
elif selected_nav == "Incident Timeline":
    st.markdown('<div class="panel-header"><span>Unified Multi-Telemetry Interactive Timeline</span><span style="font-size:0.75rem; color:#94A3B8;">3-Hour Continuous Monitored Exam</span></div>', unsafe_allow_html=True)

    # Interactive Metric Multi-selection
    tl_col1, tl_col2 = st.columns([3, 1])
    with tl_col1:
        metrics_to_plot = st.multiselect(
            "Select active telemetry series to compare:",
            options=FEATURE_COLUMNS,
            default=["response_time_ms", "db_query_time_ms", "cpu_usage_pct", "error_count"]
        )

    # Multi-series synchronized timeline
    if metrics_to_plot:
        fig_tl = go.Figure()
        colors = ["#38BDF8", "#EC4899", "#F59E0B", "#10B981", "#A855F7", "#F43F5E", "#14B8A6", "#6366F1", "#84CC16"]

        for i, m in enumerate(metrics_to_plot):
            fig_tl.add_trace(
                go.Scatter(
                    x=DF_RAW["timestamp"],
                    y=DF_RAW[m],
                    name=m.replace("_", " ").title(),
                    mode="lines+markers",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                )
            )

        # Selected vertical marker using add_shape + add_annotation (avoids plotly pandas timestamp bug)
        ts_str = str(selected_sample["timestamp"])
        fig_tl.add_shape(
            type="line",
            x0=ts_str, x1=ts_str,
            y0=0, y1=1, yref="paper",
            line=dict(color="#F8FAFC", width=1.5, dash="dash")
        )
        fig_tl.add_annotation(
            x=ts_str, y=1, yref="paper",
            text="Active Snapshot", showarrow=False,
            xanchor="left", yanchor="top",
            font=dict(color="#F8FAFC", size=10)
        )

        fig_tl.update_layout(
            **PLOTLY_DARK_LAYOUT,
            height=360,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    # Lead-Lag Analysis & Patient Zero
    st.markdown('<div class="panel-header"><span>Lead-Lag Cross-Correlation Analysis (Preceding Indicator vs Coincident Symptom)</span></div>', unsafe_allow_html=True)
    lead_lag = CORRELATION_ENGINE.compute_lead_lag_correlations("response_time_ms")
    df_ll = pd.DataFrame(lead_lag)

    ll_c1, ll_c2 = st.columns([1.5, 1])
    with ll_c1:
        fig_ll = px.bar(
            df_ll,
            x="lead1_correlation",
            y="metric",
            orientation="h",
            color="lead1_correlation",
            color_continuous_scale="Viridis",
            title="Lead Correlation (t-1) against Response Time (t)",
            labels={"lead1_correlation": "Cross-Correlation Coefficient", "metric": "Metric"}
        )
        fig_ll.update_layout(**PLOTLY_DARK_LAYOUT, height=260)
        st.plotly_chart(fig_ll, use_container_width=True)

    with ll_c2:
        st.markdown(f"""
        <div style="background:#161B22; border:1px solid #1F2937; border-radius:6px; padding:14px; height:260px;">
            <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase;">Patient Zero Analysis</div>
            <div style="font-size:1.05rem; font-weight:700; color:#EF4444; margin:6px 0;">{first_anom['first_breached_metric']} Breach</div>
            <div style="font-size:0.80rem; color:#CBD5E1; line-height:1.4;">{first_anom['narrative']}</div>
            <div style="margin-top:12px; font-size:0.75rem; color:#64748B;">
                <b>Timestamp:</b> <code>{first_anom['first_anomaly_timestamp']}</code><br/>
                <b>Earliest Student Ticket:</b> <i>"{first_anom['complaint_at_first_anomaly']}"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 3. ROOT-CAUSE ANALYSIS (GRAPH-DRIVEN INVESTIGATION)
# ==============================================================================
elif selected_nav == "Root-Cause Analysis":
    rc_tab1, rc_tab2, rc_tab3 = st.tabs([
        "⚖️ Ranked Hypotheses",
        "📋 Diagnostic Evidence Matrix",
        "🕸️ Causal Incident Graph"
    ])

    # A. Ranked Hypotheses
    with rc_tab1:
        st.markdown('<div class="panel-header"><span>Ranked Competing Root-Cause Hypotheses</span><span style="font-size:0.75rem; color:#94A3B8;">Domain Rules + Calibrated Probabilities</span></div>', unsafe_allow_html=True)

        hyp_list = sample_diag["hypotheses"]
        df_hyps = pd.DataFrame([{
            "Hypothesis": h["hypothesis"],
            "Evidence Score (%)": h["evidence_score_pct"],
            "Status": h["status"]
        } for h in hyp_list])

        fig_hyps = px.bar(
            df_hyps.sort_values("Evidence Score (%)", ascending=True),
            x="Evidence Score (%)",
            y="Hypothesis",
            orientation="h",
            color="Evidence Score (%)",
            color_continuous_scale=["#374151", "#3B82F6", "#10B981"],
            text=df_hyps.sort_values("Evidence Score (%)", ascending=True)["Evidence Score (%)"].apply(lambda s: f"{s:.1f}%")
        )
        fig_hyps.update_layout(**PLOTLY_DARK_LAYOUT, height=280)
        st.plotly_chart(fig_hyps, use_container_width=True)

        # Hypothesis Deep-Dive Inspector
        chosen_hyp_name = st.selectbox(
            "Select hypothesis to inspect underlying evidence signals:",
            options=[h["hypothesis"] for h in hyp_list],
            index=0
        )
        chosen_hyp = next(h for h in hyp_list if h["hypothesis"] == chosen_hyp_name)

        det_c1, det_c2, det_c3 = st.columns(3)
        with det_c1:
            st.markdown("<div style='font-size:0.80rem; font-weight:600; color:#10B981;'>SUPPORTING EVIDENCE</div>", unsafe_allow_html=True)
            for s in chosen_hyp["supporting_evidence"]:
                st.markdown(f"<div class='pill-supports'>✓ {s}</div>", unsafe_allow_html=True)

        with det_c2:
            st.markdown("<div style='font-size:0.80rem; font-weight:600; color:#EF4444;'>CONTRADICTORY EVIDENCE</div>", unsafe_allow_html=True)
            for c in chosen_hyp["contradictory_evidence"]:
                st.markdown(f"<div class='pill-contradicts'>✕ {c}</div>", unsafe_allow_html=True)

        with det_c3:
            st.markdown("<div style='font-size:0.80rem; font-weight:600; color:#38BDF8;'>RECOMMENDED IMMEDIATE ACTION</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#161B22; border:1px solid #1F2937; border-radius:6px; padding:10px; font-size:0.82rem; color:#CBD5E1;">
                {chosen_hyp['recommended_immediate_action']}
            </div>
            """, unsafe_allow_html=True)

    # B. Visual Evidence Matrix
    with rc_tab2:
        st.markdown('<div class="panel-header"><span>Visual Diagnostic Evidence Matrix</span><span style="font-size:0.75rem; color:#94A3B8;">Hypothesis vs Observed Signals</span></div>', unsafe_allow_html=True)

        matrix_data = DIAGNOSTIC_ENGINE.generate_evidence_matrix(selected_sample)

        # Build clean HTML table
        table_html = """
        <table style="width:100%; border-collapse:collapse; font-size:0.82rem; background:#111827; border:1px solid #1F2937; border-radius:8px;">
            <thead>
                <tr style="border-bottom:1px solid #374151; background:#161B22; color:#94A3B8; text-align:left;">
                    <th style="padding:10px 14px;">Candidate Cause</th>
                    <th style="padding:10px;">Response Time</th>
                    <th style="padding:10px;">CPU Load</th>
                    <th style="padding:10px;">Memory</th>
                    <th style="padding:10px;">DB Latency</th>
                    <th style="padding:10px;">Network</th>
                    <th style="padding:10px;">Errors</th>
                    <th style="padding:10px;">Student Feedback</th>
                </tr>
            </thead>
            <tbody>
        """

        for row in matrix_data:
            table_html += f"<tr style='border-bottom:1px solid #1F2937;'><td style='padding:10px 14px; font-weight:600; color:#F8FAFC;'>{row['hypothesis']}</td>"
            for sig in ["Response Time", "CPU", "Memory", "DB Latency", "Network", "Errors", "Complaints"]:
                cell = row["signals"].get(sig, {"status": "NEUTRAL", "reason": "N/A"})
                status = cell["status"]
                badge_style = (
                    "background:rgba(16, 185, 129, 0.15); color:#34D399; border:1px solid #10B981;" if status == "SUPPORTS"
                    else ("background:rgba(239, 68, 68, 0.15); color:#F87171; border:1px solid #EF4444;" if status == "CONTRADICTS"
                    else "background:rgba(148, 163, 184, 0.1); color:#94A3B8; border:1px solid #374151;")
                )
                icon = "✓" if status == "SUPPORTS" else ("✕" if status == "CONTRADICTS" else "—")
                cell_reason = cell.get("reason", "")
                table_html += f"<td style='padding:8px 10px;' title='{cell_reason}'><span style='{badge_style} padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600;'>{icon} {status}</span></td>"
            table_html += "</tr>"

        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption("ℹ️ Hover over any matrix cell to inspect the precise rule and numerical comparison.")

    # C. Incident Relationship Graph
    with rc_tab3:
        st.markdown('<div class="panel-header"><span>Causal Incident Dependency Graph</span><span style="font-size:0.75rem; color:#94A3B8;">Event Flow & Hypothesis Propagation</span></div>', unsafe_allow_html=True)

        G = nx.DiGraph()
        G.add_node("Exam Concurrency Surge", node_type="Observed Fact", pos=(0, 2), color="#38BDF8")
        G.add_node("DB Connection Contention", node_type="Abnormal Metric", pos=(-2, 1), color="#F59E0B")
        G.add_node("Compute Core Saturation", node_type="Abnormal Metric", pos=(0, 1), color="#F59E0B")
        G.add_node("HTTP 5xx Exception Spike", node_type="Abnormal Metric", pos=(2, 1), color="#F59E0B")
        G.add_node("Database Bottleneck", node_type="Root Cause Hypothesis", pos=(-2, 0), color="#EF4444")
        G.add_node("CPU/Server Overload", node_type="Root Cause Hypothesis", pos=(0, 0), color="#EF4444")
        G.add_node("Application Failure", node_type="Root Cause Hypothesis", pos=(2, 0), color="#EF4444")
        G.add_node("Execute Next Best Check", node_type="Proposed Action", pos=(0, -1), color="#10B981")

        G.add_edges_from([
            ("Exam Concurrency Surge", "DB Connection Contention"),
            ("Exam Concurrency Surge", "Compute Core Saturation"),
            ("Exam Concurrency Surge", "HTTP 5xx Exception Spike"),
            ("DB Connection Contention", "Database Bottleneck"),
            ("Compute Core Saturation", "CPU/Server Overload"),
            ("HTTP 5xx Exception Spike", "Application Failure"),
            ("Database Bottleneck", "Execute Next Best Check"),
            ("CPU/Server Overload", "Execute Next Best Check"),
            ("Application Failure", "Execute Next Best Check")
        ])

        pos = nx.get_node_attributes(G, 'pos')
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#374151'), mode='lines', hoverinfo='none')
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_colors = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=[node for node in G.nodes()],
            textposition="top center",
            marker=dict(size=22, color=node_colors, line=dict(width=2, color='#1E293B')),
            hoverinfo='text'
        )

        fig_rel = go.Figure(data=[edge_trace, node_trace])
        fig_rel.update_layout(
            **PLOTLY_DARK_LAYOUT,
            showlegend=False,
            height=320,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig_rel, use_container_width=True)


# ==============================================================================
# 4. SHAP EXPLAINABILITY
# ==============================================================================
elif selected_nav == "SHAP Explainability":
    st.markdown('<div class="panel-header"><span>SHAP Mathematical Feature Attribution (TreeExplainer)</span><span class="badge-ai">Game-Theoretic Shapley Values</span></div>', unsafe_allow_html=True)

    shap_res = EXPLAINER.explain_sample(selected_sample, sample_pred["predicted_incident_type"])

    shap_c1, shap_c2 = st.columns([1.6, 1])

    with shap_c1:
        st.markdown("#### Local Feature Contribution Waterfall (Selected Snapshot)")
        df_contributions = pd.DataFrame(shap_res["contributions"])

        fig_shap_local = px.bar(
            df_contributions.sort_values("attribution", ascending=True),
            x="attribution",
            y="feature",
            orientation="h",
            color="attribution",
            color_continuous_scale=["#38BDF8", "#374151", "#EF4444"],
            labels={"attribution": "SHAP Marginal Value (Log-Odds Impact)", "feature": "Telemetry Feature"},
            title=f"Feature Push toward '{sample_pred['predicted_incident_type']}'"
        )
        fig_shap_local.update_layout(**PLOTLY_DARK_LAYOUT, height=340)
        st.plotly_chart(fig_shap_local, use_container_width=True)

    with shap_c2:
        st.markdown("#### Global Feature Importance (Dataset Wide)")
        global_imp = EXPLAINER.get_global_importance()
        df_g = pd.DataFrame(global_imp["ranking"])

        fig_g = px.bar(
            df_g.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Purples",
            labels={"importance": "Global Gini Importance", "feature": "Telemetry Feature"}
        )
        fig_g.update_layout(**PLOTLY_DARK_LAYOUT, height=340)
        st.plotly_chart(fig_g, use_container_width=True)

    # Plain Language Summary
    st.markdown('<div class="panel-header"><span>Administrator Operational Narrative</span></div>', unsafe_allow_html=True)
    st.markdown(shap_res["plain_language_summary"])
    st.caption("⚠️ " + shap_res["causation_disclaimer"])


# ==============================================================================
# 5. WHAT-IF SIMULATOR
# ==============================================================================
elif selected_nav == "What-If Simulator":
    st.markdown('<div class="panel-header"><span>Counterfactual "What-If?" Interactive Sensitivity Studio</span><span style="font-size:0.75rem; color:#94A3B8;">Hypothetical Operational Interventions</span></div>', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1.4])

    with sim_col1:
        st.markdown("##### 🎛️ Adjust Hypothetical Parameters")

        sim_rt = st.slider("Response Time (ms)", 50, 4000, int(selected_sample["response_time_ms"]), 25)
        sim_db = st.slider("DB Query Latency (ms)", 10, 3500, int(selected_sample["db_query_time_ms"]), 25)
        sim_cpu = st.slider("CPU Usage (%)", 5.0, 100.0, float(selected_sample["cpu_usage_pct"]), 1.0)
        sim_mem = st.slider("Memory Usage (%)", 20.0, 100.0, float(selected_sample["memory_usage_pct"]), 1.0)
        sim_net = st.slider("Network Latency (ms)", 10.0, 700.0, float(selected_sample["network_latency_ms"]), 10.0)
        sim_err = st.slider("Error Count", 0, 100, int(selected_sample["error_count"]), 1)

        overrides = {
            "response_time_ms": sim_rt,
            "db_query_time_ms": sim_db,
            "cpu_usage_pct": sim_cpu,
            "memory_usage_pct": sim_mem,
            "network_latency_ms": sim_net,
            "error_count": sim_err
        }

    with sim_col2:
        sim_res = COUNTERFACTUAL_ENGINE.simulate_intervention(selected_sample, overrides)

        orig_class = sim_res["original_prediction"]["incident_type"]
        sim_class = sim_res["simulated_prediction"]["incident_type"]
        is_shifted = sim_res["prediction_shifted"]

        st.markdown("##### 🔬 Simulated Classification Outcome")
        c_kpi1, c_kpi2 = st.columns(2)
        with c_kpi1:
            st.markdown(f"""
            <div style="background:#161B22; border:1px solid #1F2937; border-radius:6px; padding:12px; text-align:center;">
                <div style="font-size:0.72rem; color:#94A3B8;">OBSERVED VERDICT</div>
                <div style="font-size:1.15rem; font-weight:700; color:#F8FAFC;">{orig_class}</div>
                <div style="font-size:0.75rem; color:#38BDF8;">Confidence: {sim_res['original_prediction']['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with c_kpi2:
            st.markdown(f"""
            <div style="background:rgba(16, 185, 129, 0.08); border:1px solid #10B981; border-radius:6px; padding:12px; text-align:center;">
                <div style="font-size:0.72rem; color:#34D399;">SIMULATED VERDICT</div>
                <div style="font-size:1.15rem; font-weight:700; color:#34D399;">{sim_class}</div>
                <div style="font-size:0.75rem; color:#34D399;">Confidence: {sim_res['simulated_prediction']['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        # Before / After Probability Comparison Chart
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        prob_orig = dict(sim_res["original_prediction"]["sorted_probabilities"])
        prob_sim = dict(sim_res["simulated_prediction"]["sorted_probabilities"])

        all_cats = list(prob_orig.keys())
        fig_cf_compare = go.Figure(data=[
            go.Bar(name='Original', x=all_cats, y=[prob_orig.get(c, 0) for c in all_cats], marker_color='#38BDF8'),
            go.Bar(name='Simulated', x=all_cats, y=[prob_sim.get(c, 0) for c in all_cats], marker_color='#10B981')
        ])
        fig_cf_compare.update_layout(
            **PLOTLY_DARK_LAYOUT,
            barmode='group',
            height=240,
            title="Class Probability Distribution Shift"
        )
        st.plotly_chart(fig_cf_compare, use_container_width=True)

        # Coupling warnings
        if sim_res["coupling_warnings"]:
            for w in sim_res["coupling_warnings"]:
                st.warning("⚠️ " + w)

        st.caption("🔬 " + sim_res["scientific_disclaimer"])


# ==============================================================================
# 6. COMPLAINT ANALYTICS
# ==============================================================================
elif selected_nav == "Complaint Analytics":
    st.markdown('<div class="panel-header"><span>Student Complaint & Support Ticket Analytics</span><span style="font-size:0.75rem; color:#94A3B8;">NLP Topic Categorization</span></div>', unsafe_allow_html=True)

    df_comp = COMPLAINT_CATEGORIZER.process_dataframe(DF_RAW)
    c_stats = COMPLAINT_CATEGORIZER.get_summary_statistics(df_comp)

    ca_c1, ca_c2 = st.columns([1, 1.4])

    with ca_c1:
        # Donut distribution
        df_dist = pd.DataFrame(list(c_stats["category_distribution"].items()), columns=["Category", "Count"])
        fig_pie = px.pie(
            df_dist,
            names="Category",
            values="Count",
            hole=0.55,
            color_discrete_sequence=["#38BDF8", "#EC4899", "#F59E0B", "#10B981", "#A855F7", "#F43F5E"],
            title="Complaint Topic Breakdown"
        )
        fig_pie.update_layout(**PLOTLY_DARK_LAYOUT, height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

    with ca_c2:
        # Timeline overlay
        fig_comp_tl = px.scatter(
            df_comp,
            x="timestamp",
            y="response_time_ms",
            color="complaint_category",
            size="error_count",
            hover_data=["complaint_text", "incident_type"],
            title="Complaint Category vs Portal Latency Over Time"
        )
        fig_comp_tl.update_layout(**PLOTLY_DARK_LAYOUT, height=280)
        st.plotly_chart(fig_comp_tl, use_container_width=True)

    # Representative Tickets Table
    st.markdown('<div class="panel-header"><span>Representative Student Support Tickets</span></div>', unsafe_allow_html=True)
    st.dataframe(
        df_comp[["timestamp", "complaint_category", "complaint_text", "error_count", "response_time_ms"]].rename(columns={
            "complaint_category": "Topic Category",
            "complaint_text": "Student Feedback Message",
            "error_count": "Errors",
            "response_time_ms": "Latency (ms)"
        }),
        use_container_width=True
    )


# ==============================================================================
# 7. INCIDENT REPORTS
# ==============================================================================
elif selected_nav == "Incident Reports":
    st.markdown('<div class="panel-header"><span>Official Incident Post-Mortem Audit Report</span><span style="font-size:0.75rem; color:#94A3B8;">Structured Executive Export</span></div>', unsafe_allow_html=True)

    incident_id = f"INC-EXAM-{selected_ts_str[-8:].replace(':', '')}"
    shap_data = EXPLAINER.explain_sample(selected_sample, sample_pred["predicted_incident_type"])
    report_dict = REPORT_GENERATOR.build_structured_report_data(
        incident_id=incident_id,
        sample=selected_sample,
        ml_prediction=sample_pred,
        shap_explanation=shap_data,
        hypotheses_result=sample_diag,
        time_window=f"{selected_ts_str} (Window: 5 mins)"
    )

    pdf_bytes = REPORT_GENERATOR.generate_pdf_report(report_dict)

    col_dl1, col_dl2 = st.columns([1, 3])
    with col_dl1:
        st.download_button(
            label="📄 Download Official PDF Report",
            data=pdf_bytes,
            file_name=f"{incident_id}_audit_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with col_dl2:
        st.caption(f"Compiled using ReportLab. Binary payload size: {len(pdf_bytes):,} bytes.")

    st.markdown("---")
    # Clean visual preview
    st.markdown(REPORT_GENERATOR.generate_markdown_report(report_dict))


# ==============================================================================
# 8. MODEL EVALUATION
# ==============================================================================
elif selected_nav == "Model Evaluation":
    st.markdown('<div class="panel-header"><span>Machine Learning Classification & Validation Suite</span><span class="badge-ai">Zero Target Leakage</span></div>', unsafe_allow_html=True)

    comp_data = MODEL_SUITE.evaluation_results["comparison"]

    me_c1, me_c2 = st.columns(2)

    with me_c1:
        # Confusion matrix heatmap
        rf_cm = comp_data["Random Forest"]["confusion_matrix"]
        rf_classes = comp_data["Random Forest"]["test_classes"]

        fig_cm = px.imshow(
            rf_cm,
            x=rf_classes,
            y=rf_classes,
            color_continuous_scale="Blues",
            text_auto=True,
            title="Confusion Matrix (Random Forest Holdout Set)"
        )
        fig_cm.update_layout(**PLOTLY_DARK_LAYOUT, height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    with me_c2:
        # Model Comparison Chart
        df_comp_chart = pd.DataFrame([
            {"Model": m, "Accuracy": res["test_accuracy"], "Type": "Holdout Test"}
            for m, res in comp_data.items()
        ] + [
            {"Model": m, "Accuracy": res["cv_mean_accuracy"], "Type": "3-Fold CV Mean"}
            for m, res in comp_data.items()
        ])

        fig_m_comp = px.bar(
            df_comp_chart,
            x="Model",
            y="Accuracy",
            color="Type",
            barmode="group",
            title="Algorithm Comparison: Holdout vs Cross-Validation",
            color_discrete_map={"Holdout Test": "#38BDF8", "3-Fold CV Mean": "#818CF8"}
        )
        fig_m_comp.update_layout(**PLOTLY_DARK_LAYOUT, height=320)
        st.plotly_chart(fig_m_comp, use_container_width=True)

    st.markdown('<div class="panel-header"><span>Synthetic Evaluation Limitations Disclosure</span></div>', unsafe_allow_html=True)
    st.warning(MODEL_SUITE.evaluation_results["limitations_disclosure"])
