# 🛡️ ApexExam AI: Exam Portal Slowdown Root-Cause Investigation System

> **PS 1.4: AI-Powered Exam Portal Slowdown Root-Cause Analysis**  
> An intelligent IT-operations and diagnostic application that correlates multi-stream telemetry, detects abnormal behavior, classifies incident types, explains model predictions with SHAP, evaluates competing hypotheses, recommends the next best diagnostic check, performs what-if counterfactual simulations, categorizes student complaints, and generates structured audit reports with downloadable PDFs.

---

## 🌟 Key Architecture & Core Differentiators

```
                     ┌───────────────────────────────────────────────┐
                     │   Embedded Multi-Stream Telemetry (36 pts)    │
                     │  (Server, Database, Network, Logs, Complaints)│
                     └──────────────────────┬────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│ Temporal Correlation  │       │   Supervised ML &     │       │ Competing Hypotheses  │
│  & Patient Zero (A)   │       │   SHAP Explainer      │       │   & Active Check (B,C)│
│ • Lead-lag sequencing │       │ • Random Forest / DT  │       │ • 6 Diagnostic rules  │
│ • Multi-metric overlay│       │ • Local & global SHAP │       │ • Next Best Check     │
│ • Anomaly detection   │       │ • Zero target leakage │       │ • Severity assessment │
└───────────┬───────────┘       └───────────┬───────────┘       └───────────┬───────────┘
            │                               │                               │
            └───────────────────────────────┼───────────────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │     Interactive Streamlit Operations Hub      │
                    │ • What-If Counterfactual Simulation (D)       │
                    │ • Student Complaint NLP Categorization        │
                    │ • ReportLab Downloadable PDF Audit Report     │
                    └───────────────────────────────────────────────┘
```

### 1. Embedded Synthetic Telemetry (No External CSV Required)
- **36 continuous timestamped samples** representing a 3-hour examination window.
- Telemetry streams: `active_users`, `response_time_ms`, `cpu_usage_pct`, `memory_usage_pct`, `db_query_time_ms`, `bandwidth_mbps`, `error_count`, `complaint_text`, `network_latency_ms`, `packet_loss_pct`, `incident_type`.
- 6 incident categories:
  1. `Normal`
  2. `Database Bottleneck`
  3. `CPU/Server Overload`
  4. `Memory Pressure`
  5. `Network Congestion`
  6. `Application/Dependency Failure`
  7. Ambiguous multi-factor transition edge cases.

### 2. Core Differentiator A: Temporal Evidence Correlation
- **Patient Zero Anomaly Detector:** Automatically identifies the exact timestamp and primary metric that first breached baseline thresholds.
- **Lead-Lag Analysis:** Computes cross-correlations ($t$ vs $t-1$) against response time to identify leading indicators (e.g., database query latency rising before user-facing slowdown).
- **Interactive Multi-Stream Timeline:** Synchronized Plotly time-series with threshold violation overlays.

### 3. Core Differentiator B: Competing Root-Cause Hypotheses
- Avoids single-point classification bias by simultaneously evaluating 6 candidate hypotheses.
- For every hypothesis:
  - Evidence Score (0–100% transparent domain matching).
  - Positive supporting evidence list.
  - Contradictory evidence list.
  - Missing information list.
  - Immediate operational action.

### 4. Core Differentiator C: Active Diagnosis ("Next Best Diagnostic Check")
- Synthesizes ambiguity between top competing hypotheses.
- Recommends the specific tool or command (`pg_stat_activity`, thread dump, edge MTR probe, synthetic Playwright probe).
- Outlines expected outcomes: what observation confirms Hypothesis A vs Hypothesis B.
- Distinguishes available telemetry from metrics requiring collection.

### 5. Core Differentiator D: Counterfactual Investigation ("What-If? Analysis")
- Interactive sliders permit perturbing telemetry (e.g. lowering DB query time or CPU).
- Real-time re-inference through the trained Random Forest and diagnostic engine.
- Tracks feature deltas, absolute shift, and physical coupling warnings (e.g. warning if response time is set lower than DB latency).

### 6. Explainable AI (SHAP)
- Exact TreeExplainer Shapley value attributions for each sample and predicted class.
- Plain-language operational narratives for exam administrators.
- Automatic fallback mechanism if SHAP environment errors occur (clearly labeled as fallback).

### 7. Complaint Analysis & Report Generation
- Regex/keyword NLP engine categorizing complaints into: *Login failure, Question/page loading delay, Answer submission failure, Session disconnection, Generic slowdown, No issue reported*.
- ReportLab-powered official PDF generator producing auditable post-incident reports.

---

## 🚀 Quickstart & Execution Instructions

### 1. Environment Setup
Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Run the Streamlit application:

```bash
streamlit run app.py
```

The application will start and open in your default browser at `http://localhost:8501`.

### 3. Run Automated Unit Tests
To verify all 12 acceptance criteria (schema, leakage prevention, training, SHAP, correlation, diagnostics, what-if, PDF generation):

```bash
python -m pytest tests/test_core.py -v
```

---

## 📁 Repository Structure

```
.
├── app.py                  # Full-stack Streamlit dashboard (8 UI sections)
├── data_generator.py       # Embedded 36-sample dataset & schema validator
├── model.py                # ML pipeline (DT, RF, LR) with leakage prevention
├── explainability.py       # SHAP TreeExplainer & plain-language interpreter
├── correlation.py          # Temporal synchronization & Patient Zero anomaly
├── diagnostics.py          # Multi-hypothesis engine & Next Best Diagnostic Check
├── counterfactual.py       # What-if sensitivity & coupling simulation
├── complaint_analysis.py   # Student complaint NLP categorization
├── report_generator.py     # Markdown report & ReportLab PDF compiler
├── requirements.txt        # Pinned compatible dependencies
├── README.md               # Architecture documentation & guide
└── tests/
    └── test_core.py        # Automated test suite (17 passed unit tests)
```

---

## 🔒 Target Leakage & Evaluation Transparency

1. **Strict Feature Isolation:** Feature pipelines strictly exclude string timestamps, scenario identifiers, complaint text, and target labels.
2. **Multi-Model Evaluation:** Stratified 3-fold cross-validation and holdout splits evaluate Decision Tree (78%), Random Forest (100%), and Logistic Regression (78%).
3. **Synthetic Validation Notice:** These metrics prove functional software and algorithmic pipeline mechanics on synthetic telemetry; real production deployments require continuous shadow-mode validation.
