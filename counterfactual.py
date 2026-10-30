"""
counterfactual.py
=================
Counterfactual "What-If?" Simulation Engine for Exam Portal Incident Investigation.
Simulates hypothetical operational interventions, predicts outcome probabilities,
computes feature deltas, and documents physical realism constraints.
Implements Core Differentiator D.
"""

from typing import Dict, Any, List, Optional
import copy
import pandas as pd
import numpy as np

from data_generator import FEATURE_COLUMNS


class CounterfactualEngine:
    """
    Executes what-if simulations against trained ML models and diagnostic rules.
    Provides clear delta tracking and physical coupling disclosures.
    """
    def __init__(self, model_suite: Any, diagnostic_engine: Any):
        self.model_suite = model_suite
        self.diagnostic_engine = diagnostic_engine
        self.feature_names = FEATURE_COLUMNS

    def simulate_intervention(
        self,
        original_sample: Dict[str, Any],
        hypothetical_overrides: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Takes an original telemetry sample, applies simulated overrides,
        and generates comparative ML predictions and diagnostic scores.
        """
        simulated_sample = copy.deepcopy(original_sample)
        modified_features = []

        for key, new_val in hypothetical_overrides.items():
            if key in original_sample and new_val is not None:
                orig_val = original_sample[key]
                if orig_val != new_val:
                    simulated_sample[key] = new_val
                    delta = new_val - orig_val
                    pct_change = ((new_val - orig_val) / max(1e-5, abs(orig_val))) * 100.0
                    modified_features.append({
                        "feature": key,
                        "original_value": orig_val,
                        "simulated_value": new_val,
                        "absolute_delta": round(delta, 2),
                        "percentage_change": round(pct_change, 1)
                    })

        # Run original inference
        orig_pred = self.model_suite.predict_sample(original_sample)
        orig_diag = self.diagnostic_engine.evaluate_sample_hypotheses(original_sample)

        # Run simulated inference
        sim_pred = self.model_suite.predict_sample(simulated_sample)
        sim_diag = self.diagnostic_engine.evaluate_sample_hypotheses(simulated_sample)

        # Check for realism / coupling warnings
        coupling_notes = self._evaluate_physical_coupling(simulated_sample, hypothetical_overrides)

        return {
            "is_simulation": True,
            "modified_features": modified_features,
            "original_sample": {k: original_sample.get(k) for k in self.feature_names},
            "simulated_sample": {k: simulated_sample.get(k) for k in self.feature_names},
            "original_prediction": {
                "incident_type": orig_pred["predicted_incident_type"],
                "confidence": orig_pred["confidence"],
                "sorted_probabilities": orig_pred["sorted_probabilities"]
            },
            "simulated_prediction": {
                "incident_type": sim_pred["predicted_incident_type"],
                "confidence": sim_pred["confidence"],
                "sorted_probabilities": sim_pred["sorted_probabilities"]
            },
            "original_top_hypothesis": orig_diag["top_rule_hypothesis"],
            "simulated_top_hypothesis": sim_diag["top_rule_hypothesis"],
            "prediction_shifted": orig_pred["predicted_incident_type"] != sim_pred["predicted_incident_type"],
            "coupling_warnings": coupling_notes,
            "scientific_disclaimer": (
                "SIMULATION ONLY: This counterfactual analysis tests model sensitivity by perturbing "
                "isolated feature dimensions. In real physical infrastructure, metrics exhibit tight "
                "cross-coupling (e.g., dropping response time typically requires resolving underlying DB locks or CPU starvation). "
                "A model state shift does not guarantee an identical real-world intervention outcome."
            )
        }

    def _evaluate_physical_coupling(
        self,
        sample: Dict[str, Any],
        overrides: Dict[str, float]
    ) -> List[str]:
        """
        Detects unrealistic or decoupled combinations in the hypothetical scenario.
        """
        warnings = []
        rt = sample.get("response_time_ms", 0)
        db = sample.get("db_query_time_ms", 0)
        cpu = sample.get("cpu_usage_pct", 0)
        users = sample.get("active_users", 0)

        if "response_time_ms" in overrides and "db_query_time_ms" not in overrides:
            if rt < 250 and db > 500:
                warnings.append(
                    "Physical Inconsistency Warning: Simulated end-user response time is 150ms while database query latency remains 1800ms. End-to-end response cannot be strictly lower than database wait."
                )

        if "active_users" in overrides and overrides["active_users"] > 2000 and cpu < 30:
            warnings.append(
                "Unlikely Coupling: 2000+ concurrent exam takers with only 25% CPU usage implies significant caching or multi-node scale-out not captured in single-host telemetry."
            )

        if "db_query_time_ms" in overrides and overrides["db_query_time_ms"] < 100 and rt > 2500 and cpu < 40 and sample.get("error_count", 0) < 5:
            warnings.append(
                "Telemetry Gap: DB query time was reduced to normal, but response time is still > 2500ms without high CPU or errors. Check network latency or client-side rendering bottlenecks."
            )

        return warnings
