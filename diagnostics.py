"""
diagnostics.py
==============
Multi-Hypothesis Diagnostic Engine, Active Diagnostic Check Generator,
and Incident Severity Evaluator for Exam Portal Incidents.
Implements Core Differentiators B and C.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from data_generator import NORMAL_THRESHOLDS, ALL_INCIDENT_TYPES


class DiagnosticEngine:
    """
    Evaluates competing root-cause hypotheses, calculates transparent evidence scores,
    synthesizes supporting/contradictory evidence, and formulates Next Best Diagnostic Checks.
    """
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or NORMAL_THRESHOLDS

    def evaluate_sample_hypotheses(
        self,
        sample: Dict[str, Any],
        ml_prediction: Optional[Dict[str, Any]] = None,
        time_window: str = "Selected Observation Window"
    ) -> Dict[str, Any]:
        """
        Evaluates all candidate hypotheses for a given telemetry record.
        Produces transparent evidence scores (0-100%), supporting/contradictory facts,
        and missing evidence.
        """
        rt = sample.get("response_time_ms", 0)
        cpu = sample.get("cpu_usage_pct", 0)
        mem = sample.get("memory_usage_pct", 0)
        db = sample.get("db_query_time_ms", 0)
        net_lat = sample.get("network_latency_ms", 0)
        pkt_loss = sample.get("packet_loss_pct", 0)
        errors = sample.get("error_count", 0)
        complaint = sample.get("complaint_text", "").lower()
        users = sample.get("active_users", 0)

        hypotheses = []

        # -------------------------------------------------------------
        # Hypothesis 1: Database Bottleneck
        # -------------------------------------------------------------
        db_supporting = []
        db_contradictory = []
        db_score = 10.0

        if db > self.thresholds["db_query_time_ms"]:
            ratio = db / self.thresholds["db_query_time_ms"]
            db_score += min(60.0, ratio * 20.0)
            db_supporting.append(f"Database query time is {db}ms (normal baseline <= {self.thresholds['db_query_time_ms']}ms, {ratio:.1f}x threshold).")
        else:
            db_score -= 20.0
            db_contradictory.append(f"Database query latency is normal ({db}ms).")

        if rt > 600 and db > 300:
            db_score += 15.0
            db_supporting.append(f"Elevated end-user response time ({rt}ms) correlates with sluggish query execution.")

        if any(kw in complaint for kw in ["database", "save", "saving", "queries", "lock"]):
            db_score += 15.0
            db_supporting.append(f"Student complaints reference saving or query delays: '{sample.get('complaint_text', '')}'.")

        if cpu < 70 and db > 500:
            db_score += 5.0
            db_supporting.append(f"Server CPU is moderate ({cpu}%), isolating the slowdown to the persistence layer.")
        elif cpu > 90 and db < 200:
            db_contradictory.append(f"High CPU ({cpu}%) with fast DB ({db}ms) indicates server compute saturation rather than database lock.")

        db_score = max(5.0, min(98.0, db_score))
        hypotheses.append({
            "hypothesis": "Database Bottleneck",
            "evidence_score_pct": round(db_score, 1),
            "status": "Leading Hypothesis" if db_score >= 70 else ("Plausible Contributor" if db_score >= 40 else "Unlikely"),
            "supporting_evidence": db_supporting or ["No positive database telemetry indicators observed."],
            "contradictory_evidence": db_contradictory or ["No contradictory facts noted."],
            "key_metrics": {"db_query_time_ms": db, "response_time_ms": rt, "cpu_usage_pct": cpu},
            "time_interval": time_window,
            "missing_information": "Database lock contention logs, slow query logs, connection pool active/idle connection counts.",
            "recommended_immediate_action": "Inspect PostgreSQL/MySQL `pg_stat_activity` or process list for row locks and connection pool exhaustion."
        })

        # -------------------------------------------------------------
        # Hypothesis 2: CPU/Server Overload
        # -------------------------------------------------------------
        cpu_supporting = []
        cpu_contradictory = []
        cpu_score = 10.0

        if cpu > self.thresholds["cpu_usage_pct"]:
            cpu_excess = cpu - self.thresholds["cpu_usage_pct"]
            cpu_score += min(65.0, cpu_excess * 2.2)
            cpu_supporting.append(f"Server CPU utilization is critically elevated at {cpu:.1f}% (threshold: {self.thresholds['cpu_usage_pct']}%).")
        else:
            cpu_score -= 25.0
            cpu_contradictory.append(f"CPU utilization is within normal parameters ({cpu:.1f}%).")

        if any(kw in complaint for kw in ["freeze", "sluggish", "cpu", "spinning", "unresponsive", "timer"]):
            cpu_score += 15.0
            cpu_supporting.append(f"Student feedback highlights browser/server freezes: '{sample.get('complaint_text', '')}'.")

        if users > 1500 and cpu > 80:
            cpu_score += 10.0
            cpu_supporting.append(f"High concurrency load ({users} active exam takers) driving thread contention.")

        if cpu < 50:
            cpu_contradictory.append(f"Low compute load ({cpu}%) strongly refutes processor exhaustion.")

        cpu_score = max(5.0, min(98.0, cpu_score))
        hypotheses.append({
            "hypothesis": "CPU/Server Overload",
            "evidence_score_pct": round(cpu_score, 1),
            "status": "Leading Hypothesis" if cpu_score >= 70 else ("Plausible Contributor" if cpu_score >= 40 else "Unlikely"),
            "supporting_evidence": cpu_supporting or ["No positive compute overload indicators observed."],
            "contradictory_evidence": cpu_contradictory or ["No contradictory facts noted."],
            "key_metrics": {"cpu_usage_pct": cpu, "active_users": users, "response_time_ms": rt},
            "time_interval": time_window,
            "missing_information": "Thread pool dump, process-level CPU profiling, Gunicorn/Uvicorn worker queue length.",
            "recommended_immediate_action": "Profile active web worker processes (`top`/`htop`), inspect thread backtraces, and autoscale worker instances."
        })

        # -------------------------------------------------------------
        # Hypothesis 3: Memory Pressure
        # -------------------------------------------------------------
        mem_supporting = []
        mem_contradictory = []
        mem_score = 10.0

        if mem > self.thresholds["memory_usage_pct"]:
            mem_excess = mem - self.thresholds["memory_usage_pct"]
            mem_score += min(65.0, mem_excess * 3.0)
            mem_supporting.append(f"RAM utilization is at critical threshold {mem:.1f}% (baseline <= {self.thresholds['memory_usage_pct']}%).")
        else:
            mem_score -= 20.0
            mem_contradictory.append(f"Memory headroom is adequate ({mem:.1f}% in use).")

        if any(kw in complaint for kw in ["memory", "heap", "gc", "stuttering", "reload", "session"]):
            mem_score += 15.0
            mem_supporting.append(f"Complaints report heap exhaustion/session reloads: '{sample.get('complaint_text', '')}'.")

        if mem > 85 and rt > 800:
            mem_score += 10.0
            mem_supporting.append("Elevated response times consistent with Stop-the-World garbage collection pauses.")

        mem_score = max(5.0, min(98.0, mem_score))
        hypotheses.append({
            "hypothesis": "Memory Pressure",
            "evidence_score_pct": round(mem_score, 1),
            "status": "Leading Hypothesis" if mem_score >= 70 else ("Plausible Contributor" if mem_score >= 40 else "Unlikely"),
            "supporting_evidence": mem_supporting or ["No memory leak or pressure indicators observed."],
            "contradictory_evidence": mem_contradictory or ["No contradictory facts noted."],
            "key_metrics": {"memory_usage_pct": mem, "response_time_ms": rt, "error_count": errors},
            "time_interval": time_window,
            "missing_information": "JVM/Node.js heap memory graphs, Garbage Collection pause durations, OS swap space thrashing telemetry.",
            "recommended_immediate_action": "Trigger heap memory dump for leak analysis, increase container cgroup memory limits, restart failing worker pods."
        })

        # -------------------------------------------------------------
        # Hypothesis 4: Network Congestion
        # -------------------------------------------------------------
        net_supporting = []
        net_contradictory = []
        net_score = 10.0

        if net_lat > self.thresholds["network_latency_ms"] or pkt_loss > self.thresholds["packet_loss_pct"]:
            if net_lat > self.thresholds["network_latency_ms"]:
                net_supporting.append(f"High network round-trip latency of {net_lat}ms (normal <= {self.thresholds['network_latency_ms']}ms).")
                net_score += min(40.0, (net_lat / 50.0) * 10.0)
            if pkt_loss > self.thresholds["packet_loss_pct"]:
                net_supporting.append(f"Severe packet loss rate of {pkt_loss}% (normal <= {self.thresholds['packet_loss_pct']}%).")
                net_score += min(40.0, pkt_loss * 5.0)
        else:
            net_score -= 25.0
            net_contradictory.append(f"Network transport metrics are healthy (latency: {net_lat}ms, packet loss: {pkt_loss}%).")

        if cpu < 60 and db < 120 and rt > 800:
            net_score += 15.0
            net_supporting.append("Server-side compute and database are idle, localizing degradation to edge or transit connectivity.")

        if any(kw in complaint for kw in ["network", "packet", "ping", "connection", "disconnected", "dropped"]):
            net_score += 15.0
            net_supporting.append(f"User reports echo connectivity failures: '{sample.get('complaint_text', '')}'.")

        net_score = max(5.0, min(98.0, net_score))
        hypotheses.append({
            "hypothesis": "Network Congestion",
            "evidence_score_pct": round(net_score, 1),
            "status": "Leading Hypothesis" if net_score >= 70 else ("Plausible Contributor" if net_score >= 40 else "Unlikely"),
            "supporting_evidence": net_supporting or ["Transport network metrics within normal parameters."],
            "contradictory_evidence": net_contradictory or ["No contradictory facts noted."],
            "key_metrics": {"network_latency_ms": net_lat, "packet_loss_pct": pkt_loss, "bandwidth_mbps": sample.get("bandwidth_mbps", 0)},
            "time_interval": time_window,
            "missing_information": "Campus gateway router throughput, CDN edge latency breakdown, DNS resolution timings, ISP transit BGP routing tables.",
            "recommended_immediate_action": "Check upstream border gateway router QoS queues, verify CDN edge caching health, and test multi-region ping probes."
        })

        # -------------------------------------------------------------
        # Hypothesis 5: Application/Dependency Failure
        # -------------------------------------------------------------
        app_supporting = []
        app_contradictory = []
        app_score = 10.0

        if errors > self.thresholds["error_count"]:
            err_ratio = errors / max(1, self.thresholds["error_count"])
            app_score += min(60.0, err_ratio * 12.0)
            app_supporting.append(f"High HTTP error volume ({errors} errors in interval; threshold <= {self.thresholds['error_count']}).")
        else:
            app_score -= 20.0
            app_contradictory.append(f"Minimal application errors detected ({errors} errors).")

        if any(kw in complaint for kw in ["500", "502", "503", "error", "exception", "failed", "crashed", "redis"]):
            app_score += 20.0
            app_supporting.append(f"Complaints explicitly cite HTTP 5xx or unhandled exceptions: '{sample.get('complaint_text', '')}'.")

        if db < 200 and cpu < 70 and errors > 15:
            app_score += 15.0
            app_supporting.append("Infrastructure telemetry is healthy despite high failure rates, implicating code regression or third-party API outage.")

        app_score = max(5.0, min(98.0, app_score))
        hypotheses.append({
            "hypothesis": "Application/Dependency Failure",
            "evidence_score_pct": round(app_score, 1),
            "status": "Leading Hypothesis" if app_score >= 70 else ("Plausible Contributor" if app_score >= 40 else "Unlikely"),
            "supporting_evidence": app_supporting or ["No application error spikes detected."],
            "contradictory_evidence": app_contradictory or ["No contradictory facts noted."],
            "key_metrics": {"error_count": errors, "response_time_ms": rt, "db_query_time_ms": db},
            "time_interval": time_window,
            "missing_information": "Application stack traces, Redis/Cache connectivity metrics, downstream grading microservice health checks.",
            "recommended_immediate_action": "Inspect web server 5xx error logs (Sentry/Datadog), verify Redis session store latency, and roll back recent deployments if faulty."
        })

        # -------------------------------------------------------------
        # Hypothesis 6: Normal Operation / Insufficient Evidence
        # -------------------------------------------------------------
        norm_supporting = []
        norm_contradictory = []
        norm_score = 10.0

        all_normal = (
            rt <= self.thresholds["response_time_ms"] and
            cpu <= self.thresholds["cpu_usage_pct"] and
            mem <= self.thresholds["memory_usage_pct"] and
            db <= self.thresholds["db_query_time_ms"] and
            errors <= self.thresholds["error_count"] and
            net_lat <= self.thresholds["network_latency_ms"] and
            pkt_loss <= self.thresholds["packet_loss_pct"]
        )

        if all_normal:
            norm_score = 95.0
            norm_supporting.append("All 8 telemetry metrics are strictly within normal baseline operational limits.")
        else:
            norm_contradictory.append("One or more telemetry metrics have breached anomaly thresholds.")
            norm_score = max(5.0, 40.0 - (errors * 2.0) - (rt / 100.0))

        if any(kw in complaint for kw in ["normal", "fast", "smooth", "properly"]):
            norm_score += 15.0
            norm_supporting.append(f"Positive student feedback confirms regular performance: '{sample.get('complaint_text', '')}'.")

        norm_score = max(5.0, min(98.0, norm_score))
        hypotheses.append({
            "hypothesis": "Normal Operation",
            "evidence_score_pct": round(norm_score, 1),
            "status": "Leading Hypothesis" if norm_score >= 70 else ("Plausible Contributor" if norm_score >= 40 else "Unlikely"),
            "supporting_evidence": norm_supporting or ["No strong normal telemetry indicators."],
            "contradictory_evidence": norm_contradictory or ["No contradictory facts noted."],
            "key_metrics": {"response_time_ms": rt, "cpu_usage_pct": cpu, "error_count": errors},
            "time_interval": time_window,
            "missing_information": "Real-time client telemetry beacons, synthetic end-to-end browser heartbeat tests.",
            "recommended_immediate_action": "Maintain routine baseline monitoring; no mitigation required."
        })

        # Sort hypotheses by evidence score descending
        hypotheses.sort(key=lambda h: h["evidence_score_pct"], reverse=True)

        # Generate Next Best Diagnostic Check
        next_check = self.determine_next_best_diagnostic_check(hypotheses, sample)

        # Assess Severity
        severity = self.assess_severity(sample, hypotheses)

        return {
            "hypotheses": hypotheses,
            "top_rule_hypothesis": hypotheses[0]["hypothesis"],
            "top_rule_score": hypotheses[0]["evidence_score_pct"],
            "contributing_hypotheses": [h for h in hypotheses[1:] if h["evidence_score_pct"] >= 45.0],
            "next_best_diagnostic_check": next_check,
            "severity_assessment": severity,
            "epistemic_note": (
                "Rule evidence scores (0-100%) reflect transparent domain telemetry matching. "
                "They are deliberately non-probabilistic and complement statistical ML model attributions."
            )
        }

    def determine_next_best_diagnostic_check(
        self,
        ranked_hypotheses: List[Dict[str, Any]],
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Core Differentiator C: Formulates the Next Best Diagnostic Check to resolve
        ambiguity between the top competing hypotheses or verify the leading suspect.
        """
        top_hyp = ranked_hypotheses[0]
        runner_up = ranked_hypotheses[1] if len(ranked_hypotheses) > 1 else None

        top_name = top_hyp["hypothesis"]
        runner_name = runner_up["hypothesis"] if runner_up else "None"
        score_diff = top_hyp["evidence_score_pct"] - (runner_up["evidence_score_pct"] if runner_up else 0)

        is_ambiguous = score_diff < 20.0

        # Scenario A: Database Bottleneck competing with App Failure
        if {top_name, runner_name} == {"Database Bottleneck", "Application/Dependency Failure"}:
            return {
                "check_title": "Database Connection Pool & Slow Query Execution Inspection",
                "recommended_tool": "PostgreSQL pg_stat_activity / MySQL Performance Schema",
                "why_recommended": (
                    f"Evidence is split between DB Bottleneck ({top_hyp['evidence_score_pct']}%) "
                    f"and App Failure ({runner_up['evidence_score_pct']}%). Both cause end-user delays and submission timeouts."
                ),
                "hypotheses_distinguished": ["Database Bottleneck", "Application/Dependency Failure"],
                "expected_outcomes": {
                    "Supports Database Bottleneck": "High active connections waiting on exclusive table/row locks, query execution time > 1500ms.",
                    "Supports Application Failure": "Zero query locks, idle DB connections, but upstream web server throwing 500s due to code/redis exceptions."
                },
                "telemetry_status": "Requires Infrastructure Log Collection",
                "urgency": "High",
                "ambiguity_flag": is_ambiguous
            }

        # Scenario B: CPU Overload competing with Memory Pressure
        elif {top_name, runner_name} == {"CPU/Server Overload", "Memory Pressure"}:
            return {
                "check_title": "JVM / OS Garbage Collection Pause & Thread Profile",
                "recommended_tool": "Java jcmd / Go pprof / Linux `top -H` and `vmstat 1`",
                "why_recommended": (
                    f"CPU ({sample.get('cpu_usage_pct', 0)}%) and RAM ({sample.get('memory_usage_pct', 0)}%) "
                    "are both high. Memory pressure often triggers heavy GC loops that mimic CPU overload."
                ),
                "hypotheses_distinguished": ["CPU/Server Overload", "Memory Pressure"],
                "expected_outcomes": {
                    "Supports Memory Pressure": "Major GC pauses (>2000ms), high kernel swap I/O, JVM heap above 92%.",
                    "Supports CPU Overload": "Low GC activity, user-space math/rendering threads consuming 100% core cycles."
                },
                "telemetry_status": "Requires Infrastructure Profiler Collection",
                "urgency": "High",
                "ambiguity_flag": is_ambiguous
            }

        # Scenario C: Network Congestion competing with Server Overload
        elif {top_name, runner_name} == {"Network Congestion", "CPU/Server Overload"}:
            return {
                "check_title": "Edge Gateway Packet Capture & DNS RTT Validation",
                "recommended_tool": "MTR / PingPlotter / Cloudflare Edge Analytics",
                "why_recommended": (
                    "Users report page unresponsiveness. Need to differentiate between ingress packet loss and backend compute starvation."
                ),
                "hypotheses_distinguished": ["Network Congestion", "CPU/Server Overload"],
                "expected_outcomes": {
                    "Supports Network Congestion": "Campus router packet drops > 5%, TCP SYN retransmissions, edge RTT > 350ms.",
                    "Supports CPU Overload": "Edge RTT < 30ms, 0% transport packet loss, but server TTFB (time-to-first-byte) delayed."
                },
                "telemetry_status": "Requires Edge Network Probe Collection",
                "urgency": "Medium",
                "ambiguity_flag": is_ambiguous
            }

        # Scenario D: Top is Database Bottleneck
        elif top_name == "Database Bottleneck":
            return {
                "check_title": "Database Lock Contention & Transaction Isolation Audit",
                "recommended_tool": "Database Engine Metrics (`SHOW ENGINE INNODB STATUS` / `pg_locks`)",
                "why_recommended": f"Database query latency reached {sample.get('db_query_time_ms', 0)}ms, driving overall user lag.",
                "hypotheses_distinguished": ["Database Bottleneck", "External Storage Degradation"],
                "expected_outcomes": {
                    "Supports Database Bottleneck": "Row lock wait chains on exam question submission tables.",
                    "Supports Storage Degradation": "High disk write latency (EBS/SAN IOPS saturation) despite low row locks."
                },
                "telemetry_status": "Available in Enhanced DB Telemetry / Collector",
                "urgency": "High",
                "ambiguity_flag": False
            }

        # Scenario E: Top is CPU Overload
        elif top_name == "CPU/Server Overload":
            return {
                "check_title": "Application Worker Thread Dump & Request Queue Inspection",
                "recommended_tool": "Application APM (Sentry, OpenTelemetry, py-spy)",
                "why_recommended": f"CPU reached {sample.get('cpu_usage_pct', 0)}% under concurrency of {sample.get('active_users', 0)} students.",
                "hypotheses_distinguished": ["CPU/Server Overload", "Infinite Loop Bug"],
                "expected_outcomes": {
                    "Supports Concurrency Overload": "All worker processes active and request backlog queue depth > 50.",
                    "Supports Algorithmic Bug": "Single worker thread locked in 100% regex/JSON serialization loop."
                },
                "telemetry_status": "Available in Web Server Worker Metrics",
                "urgency": "High",
                "ambiguity_flag": False
            }

        # Default / Normal Check
        else:
            return {
                "check_title": "Synthetic User Journey Heartbeat & End-to-End Latency Check",
                "recommended_tool": "Playwright / Selenium Automated Exam Login Probe",
                "why_recommended": "Telemetry indicates normal operation or sub-threshold drift; verify real browser experience.",
                "hypotheses_distinguished": ["Normal Operation", "Silent Client-Side Failures"],
                "expected_outcomes": {
                    "Supports Normal Operation": "Synthetic student test completes login, answer choice, and submission in < 400ms.",
                    "Supports Silent Failure": "Client browser JavaScript error on specific student browser versions."
                },
                "telemetry_status": "Available to Run On-Demand",
                "urgency": "Low",
                "ambiguity_flag": False
            }

    def assess_severity(
        self,
        sample: Dict[str, Any],
        hypotheses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determines incident severity level (Informational, Low, Medium, High, Critical)
        based on response time, error rate, student complaints, and resource saturation.
        """
        rt = sample.get("response_time_ms", 0)
        errors = sample.get("error_count", 0)
        cpu = sample.get("cpu_usage_pct", 0)
        mem = sample.get("memory_usage_pct", 0)
        top_hyp = hypotheses[0]["hypothesis"]

        if errors > 40 or rt > 3000 or (cpu > 95 and rt > 2500):
            level = "Critical"
            color = "#ef4444"
            desc = "Severe exam interruption. Multiple students cannot submit answers or load questions."
            sla_target = "Immediate intervention (< 5 mins)"
        elif errors > 15 or rt > 1800 or cpu > 85 or mem > 90:
            level = "High"
            color = "#f97316"
            desc = "Noticeable degradation. High latency and intermittent timeouts during exam session."
            sla_target = "Mitigate within 15 minutes"
        elif errors > 5 or rt > 700 or cpu > 75:
            level = "Medium"
            color = "#eab308"
            desc = "Sub-optimal portal response. Sluggish navigation but submissions generally succeed."
            sla_target = "Investigate within 30 minutes"
        elif rt > 350 or errors > 2:
            level = "Low"
            color = "#3b82f6"
            desc = "Minor latency fluctuation within acceptable exam operational limits."
            sla_target = "Routine monitoring"
        else:
            level = "Informational"
            color = "#10b981"
            desc = "Exam portal operating normally. Telemetry matches baseline standards."
            sla_target = "Standard observation"

        return {
            "severity_level": level,
            "badge_color": color,
            "description": desc,
            "sla_target": sla_target,
            "primary_driver": f"Response Time: {rt}ms, Errors: {errors}, Top Hypothesis: {top_hyp}"
        }

    def generate_evidence_matrix(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Builds a comprehensive evidence matrix mapping each candidate cause against
        observed diagnostic signals (Response Time, CPU, Memory, DB Latency, Network, Errors, Complaints).
        Returns a list of dicts with statuses ('SUPPORTS', 'CONTRADICTS', 'NEUTRAL') and explanatory tooltips.
        """
        rt = sample.get("response_time_ms", 0)
        cpu = sample.get("cpu_usage_pct", 0)
        mem = sample.get("memory_usage_pct", 0)
        db = sample.get("db_query_time_ms", 0)
        net_lat = sample.get("network_latency_ms", 0)
        pkt_loss = sample.get("packet_loss_pct", 0)
        errors = sample.get("error_count", 0)
        complaint = sample.get("complaint_text", "").lower()

        th_rt = self.thresholds["response_time_ms"]
        th_cpu = self.thresholds["cpu_usage_pct"]
        th_mem = self.thresholds["memory_usage_pct"]
        th_db = self.thresholds["db_query_time_ms"]
        th_net = self.thresholds["network_latency_ms"]
        th_err = self.thresholds["error_count"]

        matrix = []

        # 1. Database Bottleneck
        matrix.append({
            "hypothesis": "Database Bottleneck",
            "signals": {
                "Response Time": {
                    "status": "SUPPORTS" if rt > th_rt and db > th_db else ("CONTRADICTS" if rt <= th_rt else "NEUTRAL"),
                    "value": f"{rt} ms",
                    "reason": f"Response time {rt}ms {'correlates with database wait' if rt > th_rt and db > th_db else 'is normal'}"
                },
                "CPU": {
                    "status": "SUPPORTS" if cpu < 70 and db > th_db else ("CONTRADICTS" if cpu > 90 and db <= th_db else "NEUTRAL"),
                    "value": f"{cpu:.1f}%",
                    "reason": f"CPU {cpu:.1f}% {'is moderate, pointing to DB' if cpu < 70 and db > th_db else 'indicates compute issue rather than DB' if cpu > 90 and db <= th_db else 'neutral'}"
                },
                "Memory": {
                    "status": "NEUTRAL",
                    "value": f"{mem:.1f}%",
                    "reason": f"RAM {mem:.1f}% does not directly rule in or out database lock contention"
                },
                "DB Latency": {
                    "status": "SUPPORTS" if db > th_db else "CONTRADICTS",
                    "value": f"{db} ms",
                    "reason": f"DB query time {db}ms {'exceeds threshold of ' + str(th_db) + 'ms' if db > th_db else 'is healthy (<= ' + str(th_db) + 'ms)'}"
                },
                "Network": {
                    "status": "CONTRADICTS" if net_lat > 200 or pkt_loss > 3.0 else "NEUTRAL",
                    "value": f"{net_lat}ms / {pkt_loss}%",
                    "reason": f"Network transport is {'congested, suggesting edge delay' if net_lat > 200 or pkt_loss > 3.0 else 'healthy'}"
                },
                "Errors": {
                    "status": "SUPPORTS" if 0 < errors <= 25 and db > th_db else "NEUTRAL",
                    "value": f"{errors}",
                    "reason": f"Error count {errors} {'matches query timeouts' if 0 < errors <= 25 and db > th_db else 'unremarkable'}"
                },
                "Complaints": {
                    "status": "SUPPORTS" if any(w in complaint for w in ["database", "save", "saving", "queries", "lock"]) else "NEUTRAL",
                    "value": sample.get("complaint_text", "")[:35] + "...",
                    "reason": f"Complaint text {'cites saving/query latency' if any(w in complaint for w in ['database', 'save', 'saving', 'queries', 'lock']) else 'does not mention query issues'}"
                }
            }
        })

        # 2. CPU/Server Overload
        matrix.append({
            "hypothesis": "CPU/Server Overload",
            "signals": {
                "Response Time": {
                    "status": "SUPPORTS" if rt > th_rt and cpu > th_cpu else ("CONTRADICTS" if rt <= th_rt else "NEUTRAL"),
                    "value": f"{rt} ms",
                    "reason": f"Slowdown {rt}ms {'aligns with host compute starvation' if rt > th_rt and cpu > th_cpu else 'is within bounds'}"
                },
                "CPU": {
                    "status": "SUPPORTS" if cpu > th_cpu else "CONTRADICTS",
                    "value": f"{cpu:.1f}%",
                    "reason": f"CPU {cpu:.1f}% {'breaches saturation limit of ' + str(th_cpu) + '%' if cpu > th_cpu else 'is below threshold'}"
                },
                "Memory": {
                    "status": "SUPPORTS" if mem > 70 and cpu > th_cpu else "NEUTRAL",
                    "value": f"{mem:.1f}%",
                    "reason": f"Memory {mem:.1f}% {'under concurrent load' if mem > 70 and cpu > th_cpu else 'unremarkable'}"
                },
                "DB Latency": {
                    "status": "SUPPORTS" if db <= th_db and cpu > th_cpu else ("CONTRADICTS" if db > 1000 and cpu < 60 else "NEUTRAL"),
                    "value": f"{db} ms",
                    "reason": f"DB latency {db}ms {'is low, isolating bottleneck to CPU' if db <= th_db and cpu > th_cpu else 'points to database delay' if db > 1000 and cpu < 60 else 'neutral'}"
                },
                "Network": {
                    "status": "NEUTRAL",
                    "value": f"{net_lat}ms",
                    "reason": f"Network latency {net_lat}ms does not differentiate CPU capacity"
                },
                "Errors": {
                    "status": "SUPPORTS" if errors > 5 and cpu > th_cpu else "NEUTRAL",
                    "value": f"{errors}",
                    "reason": f"HTTP errors {errors} {'stem from request queue drops' if errors > 5 and cpu > th_cpu else 'low'}"
                },
                "Complaints": {
                    "status": "SUPPORTS" if any(w in complaint for w in ["freeze", "sluggish", "cpu", "spinning", "unresponsive", "timer"]) else "NEUTRAL",
                    "value": sample.get("complaint_text", "")[:35] + "...",
                    "reason": f"Complaints {'highlight UI freezing or render lag' if any(w in complaint for w in ['freeze', 'sluggish', 'cpu', 'spinning', 'unresponsive', 'timer']) else 'unrelated'}"
                }
            }
        })

        # 3. Memory Pressure
        matrix.append({
            "hypothesis": "Memory Pressure",
            "signals": {
                "Response Time": {
                    "status": "SUPPORTS" if rt > 800 and mem > th_mem else "NEUTRAL",
                    "value": f"{rt} ms",
                    "reason": f"High response latency {'characteristic of GC pauses' if rt > 800 and mem > th_mem else 'unremarkable'}"
                },
                "CPU": {
                    "status": "SUPPORTS" if 60 <= cpu <= 85 and mem > th_mem else "NEUTRAL",
                    "value": f"{cpu:.1f}%",
                    "reason": f"CPU {cpu:.1f}% {'elevated from garbage collector compaction' if 60 <= cpu <= 85 and mem > th_mem else 'neutral'}"
                },
                "Memory": {
                    "status": "SUPPORTS" if mem > th_mem else "CONTRADICTS",
                    "value": f"{mem:.1f}%",
                    "reason": f"Memory {mem:.1f}% {'exceeds threshold of ' + str(th_mem) + '%' if mem > th_mem else 'is within limits'}"
                },
                "DB Latency": {
                    "status": "NEUTRAL",
                    "value": f"{db} ms",
                    "reason": f"DB latency {db}ms independent of host RAM exhaustion"
                },
                "Network": {
                    "status": "NEUTRAL",
                    "value": f"{net_lat}ms",
                    "reason": "Network latency independent of application heap memory"
                },
                "Errors": {
                    "status": "SUPPORTS" if errors > 10 and mem > th_mem else "NEUTRAL",
                    "value": f"{errors}",
                    "reason": f"Errors {errors} {'consistent with out-of-memory container crashes' if errors > 10 and mem > th_mem else 'low'}"
                },
                "Complaints": {
                    "status": "SUPPORTS" if any(w in complaint for w in ["memory", "heap", "gc", "stuttering", "reload", "session"]) else "NEUTRAL",
                    "value": sample.get("complaint_text", "")[:35] + "...",
                    "reason": f"Complaints {'cite session reload or memory lag' if any(w in complaint for w in ['memory', 'heap', 'gc', 'stuttering', 'reload', 'session']) else 'unrelated'}"
                }
            }
        })

        # 4. Network Congestion
        matrix.append({
            "hypothesis": "Network Congestion",
            "signals": {
                "Response Time": {
                    "status": "SUPPORTS" if rt > th_rt and (net_lat > th_net or pkt_loss > 2.0) else "NEUTRAL",
                    "value": f"{rt} ms",
                    "reason": f"Client latency {rt}ms {'driven by transit network delay' if rt > th_rt and (net_lat > th_net or pkt_loss > 2.0) else 'normal'}"
                },
                "CPU": {
                    "status": "SUPPORTS" if cpu < 60 and (net_lat > th_net or pkt_loss > 2.0) else "NEUTRAL",
                    "value": f"{cpu:.1f}%",
                    "reason": f"Low CPU {cpu:.1f}% {'confirms server is idle while network lags' if cpu < 60 and (net_lat > th_net or pkt_loss > 2.0) else 'neutral'}"
                },
                "Memory": {
                    "status": "NEUTRAL",
                    "value": f"{mem:.1f}%",
                    "reason": "Host memory healthy"
                },
                "DB Latency": {
                    "status": "SUPPORTS" if db < th_db and (net_lat > th_net or pkt_loss > 2.0) else "NEUTRAL",
                    "value": f"{db} ms",
                    "reason": f"Database {db}ms {'is responsive, ruling out DB layer' if db < th_db and (net_lat > th_net or pkt_loss > 2.0) else 'neutral'}"
                },
                "Network": {
                    "status": "SUPPORTS" if (net_lat > th_net or pkt_loss > 1.0) else "CONTRADICTS",
                    "value": f"{net_lat}ms / {pkt_loss}%",
                    "reason": f"Network latency {net_lat}ms / packet loss {pkt_loss}% {'breaches transit SLA' if (net_lat > th_net or pkt_loss > 1.0) else 'is healthy'}"
                },
                "Errors": {
                    "status": "SUPPORTS" if errors > 5 and pkt_loss > 3.0 else "NEUTRAL",
                    "value": f"{errors}",
                    "reason": f"Error count {errors} {'correlates with TCP drops' if errors > 5 and pkt_loss > 3.0 else 'low'}"
                },
                "Complaints": {
                    "status": "SUPPORTS" if any(w in complaint for w in ["network", "packet", "ping", "connection", "disconnected", "dropped"]) else "NEUTRAL",
                    "value": sample.get("complaint_text", "")[:35] + "...",
                    "reason": f"Complaints {'cite gateway or connection disconnects' if any(w in complaint for w in ['network', 'packet', 'ping', 'connection', 'disconnected', 'dropped']) else 'unrelated'}"
                }
            }
        })

        # 5. Application/Dependency Failure
        matrix.append({
            "hypothesis": "Application/Dependency Failure",
            "signals": {
                "Response Time": {
                    "status": "SUPPORTS" if rt > th_rt and errors > th_err else "NEUTRAL",
                    "value": f"{rt} ms",
                    "reason": f"Elevated latency {rt}ms {'with error responses' if rt > th_rt and errors > th_err else 'normal'}"
                },
                "CPU": {
                    "status": "SUPPORTS" if cpu < 70 and errors > 15 else "NEUTRAL",
                    "value": f"{cpu:.1f}%",
                    "reason": f"Server CPU {cpu:.1f}% {'is moderate while errors surge, implicating software bug' if cpu < 70 and errors > 15 else 'neutral'}"
                },
                "Memory": {
                    "status": "NEUTRAL",
                    "value": f"{mem:.1f}%",
                    "reason": "Host memory within limits"
                },
                "DB Latency": {
                    "status": "SUPPORTS" if db < th_db and errors > 15 else "NEUTRAL",
                    "value": f"{db} ms",
                    "reason": f"DB latency {db}ms {'is fast, isolating failure to web/app layer' if db < th_db and errors > 15 else 'neutral'}"
                },
                "Network": {
                    "status": "CONTRADICTS" if net_lat > 300 and errors <= th_err else "NEUTRAL",
                    "value": f"{net_lat}ms",
                    "reason": "Network latency normal"
                },
                "Errors": {
                    "status": "SUPPORTS" if errors > th_err else "CONTRADICTS",
                    "value": f"{errors}",
                    "reason": f"Error count {errors} {'severely breaches threshold of ' + str(th_err) if errors > th_err else 'is below threshold'}"
                },
                "Complaints": {
                    "status": "SUPPORTS" if any(w in complaint for w in ["500", "502", "503", "error", "exception", "failed", "crashed", "redis"]) else "NEUTRAL",
                    "value": sample.get("complaint_text", "")[:35] + "...",
                    "reason": f"Complaints {'explicitly report HTTP 5xx or crash exceptions' if any(w in complaint for w in ['500', '502', '503', 'error', 'exception', 'failed', 'crashed', 'redis']) else 'unrelated'}"
                }
            }
        })

        # 6. Normal Operation
        matrix.append({
            "hypothesis": "Normal Operation",
            "signals": {
                "Response Time": {
                    "status": "SUPPORTS" if rt <= th_rt else "CONTRADICTS",
                    "value": f"{rt} ms",
                    "reason": f"Response time {rt}ms {'within normal range (<= ' + str(th_rt) + 'ms)' if rt <= th_rt else 'exceeds normal baseline'}"
                },
                "CPU": {
                    "status": "SUPPORTS" if cpu <= th_cpu else "CONTRADICTS",
                    "value": f"{cpu:.1f}%",
                    "reason": f"CPU {cpu:.1f}% {'healthy (<= ' + str(th_cpu) + '%)' if cpu <= th_cpu else 'elevated'}"
                },
                "Memory": {
                    "status": "SUPPORTS" if mem <= th_mem else "CONTRADICTS",
                    "value": f"{mem:.1f}%",
                    "reason": f"RAM {mem:.1f}% {'healthy (<= ' + str(th_mem) + '%)' if mem <= th_mem else 'elevated'}"
                },
                "DB Latency": {
                    "status": "SUPPORTS" if db <= th_db else "CONTRADICTS",
                    "value": f"{db} ms",
                    "reason": f"DB query latency {db}ms {'healthy (<= ' + str(th_db) + 'ms)' if db <= th_db else 'elevated'}"
                },
                "Network": {
                    "status": "SUPPORTS" if net_lat <= th_net and pkt_loss <= 1.0 else "CONTRADICTS",
                    "value": f"{net_lat}ms",
                    "reason": f"Network transport {'healthy' if net_lat <= th_net and pkt_loss <= 1.0 else 'degraded'}"
                },
                "Errors": {
                    "status": "SUPPORTS" if errors <= th_err else "CONTRADICTS",
                    "value": f"{errors}",
                    "reason": f"Errors {errors} {'minimal (<= ' + str(th_err) + ')' if errors <= th_err else 'exceeds tolerance'}"
                },
                "Complaints": {
                    "status": "SUPPORTS" if any(w in complaint for w in ["normal", "fast", "smooth", "properly"]) else ("CONTRADICTS" if len(complaint) > 10 and not any(w in complaint for w in ["normal", "fast", "smooth", "properly"]) else "NEUTRAL"),
                    "value": sample.get("complaint_text", "")[:35] + "...",
                    "reason": f"Student feedback {'confirms smooth portal operation' if any(w in complaint for w in ['normal', 'fast', 'smooth', 'properly']) else 'reports exam difficulty'}"
                }
            }
        })

        return matrix
