"""
new website/server.py
=====================
High-performance FastAPI REST API backend powering the new custom light-theme
ExamGuard AI incident intelligence website. Recursively converts all NumPy types
and Pandas objects to native Python for JSON serialization.
"""

import sys
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Core domain modules
from data_generator import get_raw_data, FEATURE_COLUMNS, NORMAL_THRESHOLDS, ALL_INCIDENT_TYPES
from model import get_trained_model_suite
from explainability import IncidentExplainer
from correlation import TemporalCorrelationEngine
from diagnostics import DiagnosticEngine
from counterfactual import CounterfactualEngine
from complaint_analysis import ComplaintCategorizer
from report_generator import IncidentReportGenerator

app = FastAPI(title="ExamGuard AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_jsonable(obj: Any) -> Any:
    """Recursively converts NumPy numbers, arrays, and Pandas types to pure Python primitives."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj]
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return obj


# Initialize engines
DF_RAW = get_raw_data()
MODEL_SUITE = get_trained_model_suite()
EXPLAINER = IncidentExplainer(MODEL_SUITE.active_model, DF_RAW)
CORRELATION = TemporalCorrelationEngine(DF_RAW)
DIAGNOSTIC = DiagnosticEngine(NORMAL_THRESHOLDS)
COUNTERFACTUAL = CounterfactualEngine(MODEL_SUITE, DIAGNOSTIC)
COMPLAINT = ComplaintCategorizer()
REPORT = IncidentReportGenerator()


class SimulationRequest(BaseModel):
    sample_index: int
    overrides: Dict[str, float]


@app.get("/api/health")
def health():
    return {"status": "ok", "samples": len(DF_RAW)}


@app.get("/api/telemetry")
def get_all_telemetry():
    """Returns the complete 36-sample dataset with timestamps and metrics."""
    records = []
    for idx, row in DF_RAW.iterrows():
        rec = row.to_dict()
        rec["timestamp"] = str(rec["timestamp"])
        rec["index"] = int(idx)
        records.append(to_jsonable(rec))
    return {
        "telemetry": records,
        "thresholds": to_jsonable(NORMAL_THRESHOLDS),
        "feature_columns": FEATURE_COLUMNS,
        "all_incident_types": ALL_INCIDENT_TYPES
    }


@app.get("/api/sample/{index}")
def get_sample_details(index: int):
    """Calculates all ML, SHAP, and Diagnostic evaluations for a single snapshot."""
    if index < 0 or index >= len(DF_RAW):
        raise HTTPException(status_code=404, detail="Sample index out of range")

    sample = DF_RAW.iloc[index].to_dict()
    sample_json = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in sample.items()}

    # Run ML prediction
    pred = MODEL_SUITE.predict_sample(sample)
    
    # Run SHAP attribution
    shap_res = EXPLAINER.explain_sample(sample, pred["predicted_incident_type"])
    
    # Run Diagnostic multi-hypothesis engine
    diag = DIAGNOSTIC.evaluate_sample_hypotheses(sample, pred)
    
    # Run Evidence matrix
    matrix = DIAGNOSTIC.generate_evidence_matrix(sample)

    return to_jsonable({
        "index": index,
        "sample": sample_json,
        "prediction": pred,
        "shap": shap_res,
        "diagnostics": diag,
        "evidence_matrix": matrix
    })


@app.get("/api/overview")
def get_overview(index: int = 8):
    """Returns KPI strip, patient zero, lead-lag data, and event milestones."""
    first_anom = CORRELATION.detect_first_anomaly()
    lead_lag = CORRELATION.compute_lead_lag_correlations("response_time_ms")
    windows = CORRELATION.group_incident_windows()
    global_imp = EXPLAINER.get_global_importance()

    sample = DF_RAW.iloc[index].to_dict()
    pred = MODEL_SUITE.predict_sample(sample)
    diag = DIAGNOSTIC.evaluate_sample_hypotheses(sample, pred)

    return to_jsonable({
        "first_anomaly": first_anom,
        "lead_lag": lead_lag,
        "incident_windows": windows,
        "global_importance": global_imp,
        "active_sample_kpis": {
            "index": index,
            "timestamp": str(sample["timestamp"]),
            "response_time_ms": sample["response_time_ms"],
            "db_query_time_ms": sample["db_query_time_ms"],
            "cpu_usage_pct": sample["cpu_usage_pct"],
            "memory_usage_pct": sample["memory_usage_pct"],
            "error_count": sample["error_count"],
            "active_users": sample["active_users"],
            "network_latency_ms": sample["network_latency_ms"],
            "packet_loss_pct": sample["packet_loss_pct"],
            "complaint_text": sample["complaint_text"],
            "incident_type": sample["incident_type"],
            "prediction": pred,
            "severity": diag["severity_assessment"],
            "top_hypothesis": diag["top_rule_hypothesis"],
            "top_score": diag["top_rule_score"],
            "next_check": diag["next_best_diagnostic_check"]
        }
    })


@app.post("/api/simulate")
def simulate_counterfactual(req: SimulationRequest):
    """Executes What-If intervention against the trained Random Forest and diagnostic rules."""
    if req.sample_index < 0 or req.sample_index >= len(DF_RAW):
        raise HTTPException(status_code=404, detail="Sample index out of range")
    
    sample = DF_RAW.iloc[req.sample_index].to_dict()
    sim_res = COUNTERFACTUAL.simulate_intervention(sample, req.overrides)
    return to_jsonable(sim_res)


@app.get("/api/complaints")
def get_complaints():
    """Returns NLP categorized student tickets, distribution, and timeline."""
    df_comp = COMPLAINT.process_dataframe(DF_RAW)
    stats = COMPLAINT.get_summary_statistics(df_comp)
    records = []
    for idx, row in df_comp.iterrows():
        records.append({
            "index": int(idx),
            "timestamp": str(row["timestamp"]),
            "complaint_text": row["complaint_text"],
            "category": row["complaint_category"],
            "confidence": row["complaint_match_confidence"],
            "response_time_ms": row["response_time_ms"],
            "error_count": row["error_count"],
            "incident_type": row["incident_type"]
        })
    return to_jsonable({
        "stats": stats,
        "tickets": records
    })


@app.get("/api/evaluation")
def get_model_evaluation():
    """Returns validation scores, confusion matrix, and model comparison data."""
    return to_jsonable(MODEL_SUITE.evaluation_results)


@app.get("/api/report/pdf/{index}")
def download_pdf_report(index: int):
    """Generates and streams the official PDF report via ReportLab."""
    if index < 0 or index >= len(DF_RAW):
        raise HTTPException(status_code=404, detail="Sample index out of range")
    
    sample = DF_RAW.iloc[index].to_dict()
    ts_str = str(sample["timestamp"])
    incident_id = f"INC-EXAM-{ts_str[-8:].replace(':', '')}"

    pred = MODEL_SUITE.predict_sample(sample)
    shap_data = EXPLAINER.explain_sample(sample, pred["predicted_incident_type"])
    diag = DIAGNOSTIC.evaluate_sample_hypotheses(sample, pred)

    report_dict = REPORT.build_structured_report_data(
        incident_id=incident_id,
        sample=sample,
        ml_prediction=pred,
        shap_explanation=shap_data,
        hypotheses_result=diag,
        time_window=f"{ts_str} (Window: 5 mins)"
    )

    pdf_bytes = REPORT.generate_pdf_report(report_dict)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={incident_id}_audit_report.pdf"}
    )


# Mount static website files
static_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=5000, reload=True)
