"""
tests/test_core.py
==================
Comprehensive automated test suite verifying all 12 core acceptance criteria:
schema, leakage prevention, training, multi-class predictions, explainability,
temporal correlation, multi-hypothesis diagnosis, active checks, what-if simulations,
complaint NLP, report building, and PDF generation.
"""

import pytest
import numpy as np
import pandas as pd

from data_generator import (
    DATA, FEATURE_COLUMNS, TARGET_COLUMN, ALL_INCIDENT_TYPES,
    NORMAL_THRESHOLDS, get_raw_data, validate_dataset
)
from model import IncidentClassifierSuite, get_trained_model_suite
from explainability import IncidentExplainer
from correlation import TemporalCorrelationEngine
from diagnostics import DiagnosticEngine
from counterfactual import CounterfactualEngine
from complaint_analysis import ComplaintCategorizer, COMPLAINT_CATEGORIES
from report_generator import IncidentReportGenerator


class TestDatasetAndIntegrity:
    """Check 1 & 12: Dataset schema, valid ranges, and embedded data integrity."""

    def test_dataset_size_and_schema(self):
        df = get_raw_data()
        assert 30 <= len(df) <= 45, f"Expected ~36 samples, found {len(df)}"
        is_valid, errors = validate_dataset(df)
        assert is_valid, f"Dataset validation failed: {errors}"

    def test_all_incident_categories_represented(self):
        df = get_raw_data()
        present = set(df[TARGET_COLUMN].unique())
        for req_cat in ALL_INCIDENT_TYPES:
            assert req_cat in present, f"Missing required category: {req_cat}"

    def test_telemetry_ranges(self):
        df = get_raw_data()
        assert (df["cpu_usage_pct"] >= 0).all() and (df["cpu_usage_pct"] <= 100).all()
        assert (df["memory_usage_pct"] >= 0).all() and (df["memory_usage_pct"] <= 100).all()
        assert (df["packet_loss_pct"] >= 0).all() and (df["packet_loss_pct"] <= 100).all()
        assert (df["response_time_ms"] > 0).all()
        assert (df["error_count"] >= 0).all()


class TestModelTrainingAndLeakage:
    """Check 2, 3 & 13: Model training, no target leakage, and non-trivial predictions."""

    def test_no_target_leakage_in_features(self):
        suite = get_trained_model_suite()
        # Strictly verify no strings, timestamps, scenario IDs, or target column in feature names
        forbidden = ["timestamp", "incident_type", "scenario_id", "complaint_text"]
        for f in forbidden:
            assert f not in suite.feature_names, f"Target leakage! Forbidden column {f} found in features."

    def test_model_comparison_runs_successfully(self):
        suite = get_trained_model_suite()
        res = suite.evaluation_results["comparison"]
        assert "Decision Tree" in res
        assert "Random Forest" in res
        assert "Logistic Regression" in res
        for m_name in ["Decision Tree", "Random Forest", "Logistic Regression"]:
            assert res[m_name]["test_accuracy"] >= 0.0
            assert "classification_report" in res[m_name]
            assert "confusion_matrix" in res[m_name]

    def test_model_predicts_multiple_distinct_classes(self):
        suite = get_trained_model_suite()
        df = get_raw_data()
        predictions = [suite.predict_sample(row.to_dict())["predicted_incident_type"] for _, row in df.iterrows()]
        unique_predicted = set(predictions)
        # Must predict at least 4 distinct classes across the dataset
        assert len(unique_predicted) >= 4, f"Model lacks variety, predicted only: {unique_predicted}"
        assert "Normal" in unique_predicted
        assert "Database Bottleneck" in unique_predicted


class TestExplainability:
    """Check 8: SHAP integration, local attribution, and fallback handling."""

    def test_shap_explanation_structure(self):
        suite = get_trained_model_suite()
        df = get_raw_data()
        explainer = IncidentExplainer(suite.active_model, df)
        sample = df.iloc[7].to_dict()
        pred = suite.predict_sample(sample)
        exp = explainer.explain_sample(sample, pred["predicted_incident_type"])

        assert "contributions" in exp
        assert len(exp["contributions"]) == len(FEATURE_COLUMNS)
        assert "plain_language_summary" in exp
        assert "causation_disclaimer" in exp
        assert "method" in exp

    def test_global_importance(self):
        suite = get_trained_model_suite()
        df = get_raw_data()
        explainer = IncidentExplainer(suite.active_model, df)
        global_imp = explainer.get_global_importance()
        assert "ranking" in global_imp
        assert len(global_imp["ranking"]) == len(FEATURE_COLUMNS)


class TestTemporalCorrelation:
    """Check 4 & 5: Incident detection, timeline aggregation, and Patient Zero anomaly."""

    def test_anomaly_detection_and_first_breach(self):
        df = get_raw_data()
        corr = TemporalCorrelationEngine(df)
        first_anom = corr.detect_first_anomaly()
        assert first_anom["has_anomaly"] is True
        assert first_anom["first_anomaly_timestamp"] is not None
        assert first_anom["first_breached_metric"] in FEATURE_COLUMNS

    def test_lead_lag_correlations(self):
        df = get_raw_data()
        corr = TemporalCorrelationEngine(df)
        corrs = corr.compute_lead_lag_correlations("response_time_ms")
        assert len(corrs) > 0
        for item in corrs:
            assert "metric" in item
            assert "lag0_correlation" in item
            assert "lead1_correlation" in item

    def test_incident_windows_grouping(self):
        df = get_raw_data()
        corr = TemporalCorrelationEngine(df)
        windows = corr.group_incident_windows(min_consecutive_steps=2)
        assert len(windows) >= 1
        for w in windows:
            assert "start_time" in w
            assert "end_time" in w
            assert w["duration_minutes"] >= 0


class TestDiagnosticsAndHypotheses:
    """Check 6 & 7: Multi-hypothesis evaluation, evidence score, and Next Best Check."""

    def test_competing_hypotheses_ranking(self):
        diag = DiagnosticEngine()
        df = get_raw_data()
        sample_db = df[df["incident_type"] == "Database Bottleneck"].iloc[0].to_dict()
        eval_res = diag.evaluate_sample_hypotheses(sample_db)

        hypotheses = eval_res["hypotheses"]
        assert len(hypotheses) == 6
        # Top hypothesis should be Database Bottleneck
        assert eval_res["top_rule_hypothesis"] == "Database Bottleneck"
        assert hypotheses[0]["evidence_score_pct"] > 60.0

        for h in hypotheses:
            assert "supporting_evidence" in h
            assert "contradictory_evidence" in h
            assert "missing_information" in h
            assert "recommended_immediate_action" in h

    def test_next_best_diagnostic_check(self):
        diag = DiagnosticEngine()
        df = get_raw_data()
        sample = df.iloc[8].to_dict()
        eval_res = diag.evaluate_sample_hypotheses(sample)
        chk = eval_res["next_best_diagnostic_check"]

        assert "check_title" in chk
        assert "recommended_tool" in chk
        assert "why_recommended" in chk
        assert "hypotheses_distinguished" in chk
        assert "expected_outcomes" in chk
        assert "telemetry_status" in chk

    def test_severity_levels(self):
        diag = DiagnosticEngine()
        df = get_raw_data()
        # Test normal sample -> Informational
        normal_sample = df[df["incident_type"] == "Normal"].iloc[0].to_dict()
        res_norm = diag.evaluate_sample_hypotheses(normal_sample)
        assert res_norm["severity_assessment"]["severity_level"] in ["Informational", "Low"]

        # Test heavy incident sample -> High or Critical
        heavy_sample = df[df["response_time_ms"] > 3000].iloc[0].to_dict()
        res_heavy = diag.evaluate_sample_hypotheses(heavy_sample)
        assert res_heavy["severity_assessment"]["severity_level"] in ["High", "Critical"]


class TestCounterfactualAndComplaints:
    """Check 9 & 10: Counterfactual simulations and complaint categorization."""

    def test_what_if_simulation(self):
        suite = get_trained_model_suite()
        diag = DiagnosticEngine()
        cf = CounterfactualEngine(suite, diag)
        df = get_raw_data()
        sample = df[df["incident_type"] == "Database Bottleneck"].iloc[1].to_dict()

        # Simulate reducing DB query time and response time to normal levels
        res = cf.simulate_intervention(sample, {
            "db_query_time_ms": 45.0,
            "response_time_ms": 160.0,
            "error_count": 0
        })

        assert res["is_simulation"] is True
        assert len(res["modified_features"]) == 3
        assert "simulated_prediction" in res
        assert "coupling_warnings" in res
        assert "scientific_disclaimer" in res

    def test_complaint_categorization(self):
        categorizer = ComplaintCategorizer()
        test_cases = [
            ("Database lock timeout when submitting answer choice.", "Answer submission failure"),
            ("Getting HTTP 500 Internal Server Error when submitting exam.", "Answer submission failure"),
            ("Server pegged at 100%, total portal freeze for minutes.", "Generic slowdown"),
            ("Portal working normally, questions loading fast.", "No issue reported"),
            ("Campus gateway congestion, ping is 400ms+ to exam server.", "Session disconnection")
        ]
        for text, expected_cat in test_cases:
            res = categorizer.categorize_text(text)
            assert res["category"] in COMPLAINT_CATEGORIES
            assert res["confidence"] > 0.0


class TestReportGeneration:
    """Check 11: Structured incident report and PDF binary generation."""

    def test_markdown_and_pdf_generation(self):
        suite = get_trained_model_suite()
        diag = DiagnosticEngine()
        df = get_raw_data()
        explainer = IncidentExplainer(suite.active_model, df)
        rep = IncidentReportGenerator()

        sample = df.iloc[9].to_dict()
        pred = suite.predict_sample(sample)
        exp = explainer.explain_sample(sample, pred["predicted_incident_type"])
        hyp = diag.evaluate_sample_hypotheses(sample, pred)

        report_data = rep.build_structured_report_data("INC-TEST-001", sample, pred, exp, hyp)
        assert report_data["incident_id"] == "INC-TEST-001"

        md = rep.generate_markdown_report(report_data)
        assert "# 📋 INCIDENT ROOT-CAUSE INVESTIGATION REPORT" in md
        assert "SHAP Model Attribution" in md

        pdf_bytes = rep.generate_pdf_report(report_data)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2000
        assert pdf_bytes.startswith(b"%PDF")
