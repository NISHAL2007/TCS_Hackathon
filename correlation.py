"""
correlation.py
==============
Temporal Evidence Correlation Engine for Exam Portal Telemetry.
Synchronizes metrics on a unified timeline, flags multi-variate anomalies,
identifies the "Patient Zero" initial anomaly, and evaluates lead-lag sequences.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from data_generator import FEATURE_COLUMNS, NORMAL_THRESHOLDS, get_raw_data


class TemporalCorrelationEngine:
    """
    Performs temporal synchronization, anomaly detection, lead-lag analysis,
    and event grouping across heterogeneous telemetry streams.
    """
    def __init__(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            self.df = get_raw_data()
        else:
            self.df = df.copy()

        self.thresholds = NORMAL_THRESHOLDS
        self._enrich_telemetry_stream()

    def _enrich_telemetry_stream(self):
        """
        Computes anomaly flags, z-scores, and severity index for every row.
        """
        # Ensure timestamp is datetime
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # Flag threshold anomalies for each key metric
        for metric, thresh in self.thresholds.items():
            if metric in self.df.columns:
                self.df[f"{metric}_is_abnormal"] = self.df[metric] > thresh
            else:
                self.df[f"{metric}_is_abnormal"] = False

        # Compute composite anomaly score (count of abnormal metrics)
        abnormal_cols = [f"{m}_is_abnormal" for m in self.thresholds.keys() if f"{m}_is_abnormal" in self.df.columns]
        self.df["anomaly_count"] = self.df[abnormal_cols].sum(axis=1)
        self.df["is_incident_active"] = self.df["anomaly_count"] >= 1

    def detect_first_anomaly(self) -> Dict[str, Any]:
        """
        Identifies the 'Patient Zero' anomaly: the exact timestamp and metric
        that first breached normal operating thresholds.
        """
        incident_rows = self.df[self.df["is_incident_active"]].copy()
        if incident_rows.empty:
            return {
                "has_anomaly": False,
                "first_anomaly_timestamp": None,
                "first_breached_metric": None,
                "first_breached_value": None,
                "narrative": "All telemetry is currently within normal operating bounds."
            }

        first_row = incident_rows.iloc[0]
        first_time = first_row["timestamp"]

        # Determine which specific metrics breached at this initial timestamp
        breached_metrics = []
        for metric, thresh in self.thresholds.items():
            val = first_row.get(metric, 0)
            if val > thresh:
                breached_metrics.append({
                    "metric": metric,
                    "value": val,
                    "threshold": thresh,
                    "excess_pct": round(((val - thresh) / thresh) * 100, 1)
                })

        # Sort by relative excess above threshold
        breached_metrics.sort(key=lambda x: x["excess_pct"], reverse=True)
        primary_breach = breached_metrics[0] if breached_metrics else None

        return {
            "has_anomaly": True,
            "first_anomaly_timestamp": str(first_time),
            "first_breached_metric": primary_breach["metric"] if primary_breach else "Unknown",
            "first_breached_value": primary_breach["value"] if primary_breach else 0,
            "all_breached_metrics": breached_metrics,
            "complaint_at_first_anomaly": first_row.get("complaint_text", "No complaints logged"),
            "narrative": (
                f"First anomalous deviation occurred at {first_time.strftime('%H:%M:%S')}, "
                f"triggered primarily by '{primary_breach['metric'] if primary_breach else 'metric'}' "
                f"reaching {primary_breach['value'] if primary_breach else 0} "
                f"(threshold: {primary_breach['threshold'] if primary_breach else 0})."
            )
        }

    def compute_lead_lag_correlations(self, target_metric: str = "response_time_ms") -> List[Dict[str, Any]]:
        """
        Computes lag-0 and lag-1 cross-correlations against response_time_ms
        to assess which operational metrics precede or coincide with user-facing slowdowns.
        """
        results = []
        metrics_to_test = [
            "db_query_time_ms", "cpu_usage_pct", "memory_usage_pct",
            "network_latency_ms", "bandwidth_mbps", "error_count"
        ]

        target_series = self.df[target_metric]

        for m in metrics_to_test:
            if m not in self.df.columns:
                continue
            m_series = self.df[m]

            # Pearson correlation at lag 0 (concurrent)
            corr_lag0 = float(target_series.corr(m_series))

            # Lag 1: metric at t-1 vs target at t (leading indicator)
            corr_lead = float(target_series.iloc[1:].corr(m_series.iloc[:-1].reset_index(drop=True)))

            timing_type = "Concurrent"
            if abs(corr_lead) > abs(corr_lag0) and abs(corr_lead) > 0.4:
                timing_type = "Leading Indicator (Spikes 5m prior)"
            elif abs(corr_lag0) > 0.5:
                timing_type = "Coincident (Rises simultaneously)"
            else:
                timing_type = "Weakly Correlated"

            results.append({
                "metric": m,
                "lag0_correlation": round(corr_lag0, 3),
                "lead1_correlation": round(corr_lead, 3),
                "timing_relationship": timing_type,
                "max_abs_corr": max(abs(corr_lag0), abs(corr_lead))
            })

        results.sort(key=lambda x: x["max_abs_corr"], reverse=True)
        return results


    def detect_onset_for_window(self, target_idx: int) -> dict:
        target_row = self.df.iloc[target_idx]
        incident_type = target_row.get('incident_type', 'Normal')
        scenario_id = target_row.get('scenario_id', 'unknown')

        scenario_slice = self.df[self.df['scenario_id'] == scenario_id] if 'scenario_id' in self.df.columns else self.df[self.df['incident_type'] == incident_type]

        if scenario_slice.empty or incident_type == 'Normal':
            ts_str = target_row['timestamp'].strftime('%H:%M:%S') if hasattr(target_row['timestamp'], 'strftime') else str(target_row['timestamp'])
            return {
                'window_label': 'Baseline Normal Operation',
                'onset_timestamp': str(target_row['timestamp']),
                'first_breached_metric': 'All Metrics Nominal',
                'first_breached_value': 0,
                'complaint_at_first_anomaly': target_row.get('complaint_text', 'No complaints logged'),
                'narrative': f'System operating normally at {ts_str}. Telemetry metrics remain within baseline thresholds.'
            }

        first_row = scenario_slice.iloc[0]
        first_time = first_row['timestamp']

        breached = []
        for metric, thresh in self.thresholds.items():
            val = first_row.get(metric, 0)
            if val > thresh:
                breached.append({
                    'metric': metric,
                    'value': val,
                    'threshold': thresh,
                    'excess_pct': round(((val - thresh) / thresh) * 100, 1)
                })

        if not breached:
            for metric, thresh in self.thresholds.items():
                val = target_row.get(metric, 0)
                if val > thresh:
                    breached.append({
                        'metric': metric,
                        'value': val,
                        'threshold': thresh,
                        'excess_pct': round(((val - thresh) / thresh) * 100, 1)
                    })

        breached.sort(key=lambda x: x['excess_pct'], reverse=True)
        primary = breached[0] if breached else None

        metric_name = primary['metric'] if primary else 'System Load'
        metric_val = primary['value'] if primary else 'Elevated'
        metric_thresh = primary['threshold'] if primary else 'Normal'
        time_str = first_time.strftime('%H:%M:%S') if hasattr(first_time, 'strftime') else str(first_time)

        return {
            'window_label': f'{incident_type} Window Onset',
            'onset_timestamp': str(first_time),
            'first_breached_metric': metric_name,
            'first_breached_value': metric_val,
            'complaint_at_first_anomaly': first_row.get('complaint_text', target_row.get('complaint_text', 'No complaint logged')),
            'narrative': (
                f'{incident_type} anomaly onset at {time_str}, '
                f'initiated by \'{metric_name}\' reaching {metric_val} '
                f'(normal baseline: {metric_thresh}).'
            )
        }

    def group_incident_windows(self, min_consecutive_steps: int = 2) -> List[Dict[str, Any]]:
        """
        Identifies continuous time windows of active degradation for incident demarcation.
        """
        windows = []
        in_incident = False
        start_idx = None

        for idx, row in self.df.iterrows():
            if row["is_incident_active"] and not in_incident:
                in_incident = True
                start_idx = idx
            elif not row["is_incident_active"] and in_incident:
                in_incident = False
                end_idx = idx - 1
                if (end_idx - start_idx + 1) >= min_consecutive_steps:
                    windows.append(self._summarize_window(start_idx, end_idx))
                start_idx = None

        # If incident active at the end of the series
        if in_incident and start_idx is not None:
            end_idx = len(self.df) - 1
            if (end_idx - start_idx + 1) >= min_consecutive_steps:
                windows.append(self._summarize_window(start_idx, end_idx))

        return windows

    def _summarize_window(self, start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Summarizes an incident interval."""
        sub_df = self.df.iloc[start_idx : end_idx + 1]
        start_time = sub_df["timestamp"].iloc[0]
        end_time = sub_df["timestamp"].iloc[-1]
        duration_minutes = (end_time - start_time).total_seconds() / 60.0

        dominant_type = sub_df["incident_type"].mode()[0] if "incident_type" in sub_df.columns else "Unknown"

        return {
            "window_id": f"INC-{start_time.strftime('%H%M')}",
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration_minutes": duration_minutes,
            "sample_count": len(sub_df),
            "max_response_time_ms": float(sub_df["response_time_ms"].max()),
            "total_errors": int(sub_df["error_count"].sum()),
            "dominant_type": dominant_type,
            "max_cpu": float(sub_df["cpu_usage_pct"].max()),
            "max_db_query": float(sub_df["db_query_time_ms"].max()),
            "max_memory": float(sub_df["memory_usage_pct"].max())
        }
