"""
explainability.py
=================
SHAP Explainability Module for Exam Portal Slowdown Root-Cause Analysis.
Provides global feature importance, local sample feature attributions,
waterfall/bar visual data structures, plain-language interpretations,
and graceful fallback handling.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap

from data_generator import FEATURE_COLUMNS, ALL_INCIDENT_TYPES


class IncidentExplainer:
    """
    Computes SHAP feature attributions for model predictions.
    Includes fallback mechanisms and human-readable explanations.
    """
    def __init__(self, model: Any, background_data: pd.DataFrame):
        self.model = model
        self.feature_names = FEATURE_COLUMNS
        self.background_df = background_data[self.feature_names]
        self.explainer: Optional[Any] = None
        self.is_shap_available: bool = False
        self.init_error: Optional[str] = None
        
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initializes shap.TreeExplainer on the fitted model with error handling."""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.is_shap_available = True
        except Exception as e:
            self.is_shap_available = False
            self.init_error = str(e)

    def explain_sample(self, sample_dict: Dict[str, float], predicted_class: str) -> Dict[str, Any]:
        """
        Calculates local feature contributions for the specified observation.
        Returns attribution scores, ranking, and administrator-friendly summary.
        """
        features_vec = np.array([[sample_dict[col] for col in self.feature_names]])
        classes = list(self.model.classes_)
        
        if predicted_class in classes:
            class_idx = classes.index(predicted_class)
        else:
            class_idx = 0

        if self.is_shap_available and self.explainer is not None:
            try:
                # TreeExplainer returns shape: (n_samples, n_features, n_classes) or list of ndarrays
                raw_shap = self.explainer.shap_values(features_vec)
                
                if isinstance(raw_shap, list):
                    # List of classes
                    class_shap_vals = raw_shap[class_idx][0]
                elif isinstance(raw_shap, np.ndarray):
                    if raw_shap.ndim == 3:
                        class_shap_vals = raw_shap[0, :, class_idx]
                    elif raw_shap.ndim == 2:
                        class_shap_vals = raw_shap[0]
                    else:
                        raise ValueError(f"Unexpected SHAP array dimension: {raw_shap.ndim}")
                else:
                    raise TypeError(f"Unexpected SHAP return type: {type(raw_shap)}")

                method_used = "SHAP TreeExplainer (Exact Shapley Values)"
                is_fallback = False
            except Exception as ex:
                class_shap_vals = self._compute_fallback_attribution(sample_dict, predicted_class)
                method_used = f"Heuristic Feature Attribution Fallback (SHAP error: {str(ex)[:60]})"
                is_fallback = True
        else:
            class_shap_vals = self._compute_fallback_attribution(sample_dict, predicted_class)
            method_used = f"Heuristic Feature Attribution Fallback (SHAP init note: {str(self.init_error)[:60]})"
            is_fallback = True

        # Build feature contributions list
        contributions = []
        for feat_name, shap_val in zip(self.feature_names, class_shap_vals):
            feat_val = sample_dict[feat_name]
            contributions.append({
                "feature": feat_name,
                "value": feat_val,
                "attribution": float(shap_val),
                "abs_attribution": float(abs(shap_val)),
                "impact_direction": "Supports Class" if shap_val > 0 else "Opposes Class"
            })

        # Rank by absolute attribution magnitude
        contributions.sort(key=lambda x: x["abs_attribution"], reverse=True)

        plain_language = self._generate_plain_language_narrative(
            predicted_class, contributions, sample_dict
        )

        return {
            "method": method_used,
            "is_fallback": is_fallback,
            "predicted_class": predicted_class,
            "contributions": contributions,
            "plain_language_summary": plain_language,
            "causation_disclaimer": (
                "Feature attribution indicates how individual telemetry measurements influenced "
                "the machine-learning model's decision boundary. High attribution does not prove "
                "isolated physical causality without corroborating infrastructure evidence."
            )
        }

    def _compute_fallback_attribution(self, sample_dict: Dict[str, float], predicted_class: str) -> np.ndarray:
        """
        Calculates fallback attribution based on model feature importances and deviation from baseline.
        Explicitly labeled as a fallback method.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        # Baseline mean
        means = self.background_df.mean()
        stds = self.background_df.std().replace(0, 1.0)

        # Normalized z-score deviation multiplied by global tree importance
        z_scores = np.array([(sample_dict[col] - means[col]) / stds[col] for col in self.feature_names])
        fallback_vals = z_scores * importances
        return fallback_vals

    def get_global_importance(self) -> Dict[str, Any]:
        """
        Calculates global feature importance across all features.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            ranking = [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in zip(self.feature_names, importances)
            ]
            ranking.sort(key=lambda x: x["importance"], reverse=True)
            return {
                "method": "Gini Impurity Feature Importance (Random Forest)",
                "ranking": ranking
            }
        else:
            return {
                "method": "Equal weighting baseline",
                "ranking": [{"feature": f, "importance": 1.0 / len(self.feature_names)} for f in self.feature_names]
            }

    def _generate_plain_language_narrative(
        self,
        predicted_class: str,
        contributions: List[Dict[str, Any]],
        sample_dict: Dict[str, float]
    ) -> str:
        """
        Translates mathematical feature attributions into a clear operational narrative for exam admins.
        """
        top_positive = [c for c in contributions if c["attribution"] > 0][:3]
        top_negative = [c for c in contributions if c["attribution"] < 0][:2]

        narrative = f"**Incident Assessment: {predicted_class}**\n\n"
        
        if top_positive:
            pos_desc = ", ".join([
                f"`{c['feature']}` = {c['value']} (+{c['attribution']:.2f})" for c in top_positive
            ])
            narrative += f"- **Primary Driving Telemetry:** The model's classification is most strongly supported by {pos_desc}.\n"

        if top_negative:
            neg_desc = ", ".join([
                f"`{c['feature']}` = {c['value']} ({c['attribution']:.2f})" for c in top_negative
            ])
            narrative += f"- **Countervailing Telemetry:** Lower or normal values in {neg_desc} partially lowered confidence.\n"

        narrative += f"- **Investigative Takeaway:** Review the timeline to confirm if `{top_positive[0]['feature'] if top_positive else 'metrics'}` spiked first."
        return narrative
