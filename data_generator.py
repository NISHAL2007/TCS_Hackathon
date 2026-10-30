"""
data_generator.py
=================
Embedded Synthetic Dataset and Preprocessing for Exam Portal Slowdown Root-Cause Analysis.
Dataset strictly embedded in source code with 36 timestamped observations representing
realistic exam portal telemetry and distinct failure scenarios.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Deterministic synthetic dataset (36 timestamped samples across 6 scenarios + ambiguous cases)
# Required fields: timestamp, active_users, response_time_ms, cpu_usage_pct, memory_usage_pct,
# db_query_time_ms, bandwidth_mbps, error_count, complaint_text, network_latency_ms, packet_loss_pct, incident_type.
# Also includes scenario_id for leakage-free group-based cross-validation.
DATA: List[Dict[str, Any]] = [
    # Scenario 1: Normal Baseline Examination (09:00 - 09:30)
    {
        "timestamp": "2026-10-29 09:00:00",
        "active_users": 420,
        "response_time_ms": 145,
        "cpu_usage_pct": 28.5,
        "memory_usage_pct": 42.0,
        "db_query_time_ms": 45,
        "bandwidth_mbps": 32.0,
        "error_count": 0,
        "complaint_text": "Portal working normally, questions loading fast.",
        "network_latency_ms": 18.2,
        "packet_loss_pct": 0.05,
        "incident_type": "Normal",
        "scenario_id": "scen_1_normal"
    },
    {
        "timestamp": "2026-10-29 09:05:00",
        "active_users": 650,
        "response_time_ms": 160,
        "cpu_usage_pct": 34.0,
        "memory_usage_pct": 44.5,
        "db_query_time_ms": 52,
        "bandwidth_mbps": 48.5,
        "error_count": 1,
        "complaint_text": "Logged in smoothly, no delay observed.",
        "network_latency_ms": 19.5,
        "packet_loss_pct": 0.08,
        "incident_type": "Normal",
        "scenario_id": "scen_1_normal"
    },
    {
        "timestamp": "2026-10-29 09:10:00",
        "active_users": 850,
        "response_time_ms": 185,
        "cpu_usage_pct": 39.2,
        "memory_usage_pct": 46.8,
        "db_query_time_ms": 58,
        "bandwidth_mbps": 62.0,
        "error_count": 0,
        "complaint_text": "System running properly, answered section 1.",
        "network_latency_ms": 21.0,
        "packet_loss_pct": 0.10,
        "incident_type": "Normal",
        "scenario_id": "scen_1_normal"
    },
    {
        "timestamp": "2026-10-29 09:15:00",
        "active_users": 980,
        "response_time_ms": 210,
        "cpu_usage_pct": 43.5,
        "memory_usage_pct": 49.0,
        "db_query_time_ms": 65,
        "bandwidth_mbps": 75.4,
        "error_count": 2,
        "complaint_text": "All good so far, timer counting down normally.",
        "network_latency_ms": 22.4,
        "packet_loss_pct": 0.12,
        "incident_type": "Normal",
        "scenario_id": "scen_1_normal"
    },
    {
        "timestamp": "2026-10-29 09:20:00",
        "active_users": 1050,
        "response_time_ms": 225,
        "cpu_usage_pct": 45.0,
        "memory_usage_pct": 51.2,
        "db_query_time_ms": 70,
        "bandwidth_mbps": 81.0,
        "error_count": 1,
        "complaint_text": "Slight pause loading images, but overall fast.",
        "network_latency_ms": 24.1,
        "packet_loss_pct": 0.15,
        "incident_type": "Normal",
        "scenario_id": "scen_1_normal"
    },
    {
        "timestamp": "2026-10-29 09:25:00",
        "active_users": 1100,
        "response_time_ms": 230,
        "cpu_usage_pct": 46.8,
        "memory_usage_pct": 52.0,
        "db_query_time_ms": 74,
        "bandwidth_mbps": 84.5,
        "error_count": 1,
        "complaint_text": "Exam navigation working as expected.",
        "network_latency_ms": 23.0,
        "packet_loss_pct": 0.11,
        "incident_type": "Normal",
        "scenario_id": "scen_1_normal"
    },

    # Scenario 2: Database Bottleneck & Lock Contention (09:30 - 09:55)
    # Characterized by extreme db_query_time_ms, elevated response time, moderate CPU, complaints on saving
    {
        "timestamp": "2026-10-29 09:30:00",
        "active_users": 1250,
        "response_time_ms": 850,
        "cpu_usage_pct": 48.0,
        "memory_usage_pct": 54.0,
        "db_query_time_ms": 680,
        "bandwidth_mbps": 86.0,
        "error_count": 4,
        "complaint_text": "Saving answer to question takes several seconds.",
        "network_latency_ms": 25.0,
        "packet_loss_pct": 0.15,
        "incident_type": "Database Bottleneck",
        "scenario_id": "scen_2_db"
    },
    {
        "timestamp": "2026-10-29 09:35:00",
        "active_users": 1380,
        "response_time_ms": 2200,
        "cpu_usage_pct": 52.0,
        "memory_usage_pct": 56.5,
        "db_query_time_ms": 1850,
        "bandwidth_mbps": 78.0,
        "error_count": 9,
        "complaint_text": "Database timeout when submitting answer choice.",
        "network_latency_ms": 28.0,
        "packet_loss_pct": 0.20,
        "incident_type": "Database Bottleneck",
        "scenario_id": "scen_2_db"
    },
    {
        "timestamp": "2026-10-29 09:40:00",
        "active_users": 1420,
        "response_time_ms": 3400,
        "cpu_usage_pct": 55.4,
        "memory_usage_pct": 58.0,
        "db_query_time_ms": 2950,
        "bandwidth_mbps": 70.2,
        "error_count": 18,
        "complaint_text": "Question/page loading delay, stuck at saving prompt.",
        "network_latency_ms": 26.5,
        "packet_loss_pct": 0.18,
        "incident_type": "Database Bottleneck",
        "scenario_id": "scen_2_db"
    },
    {
        "timestamp": "2026-10-29 09:45:00",
        "active_users": 1450,
        "response_time_ms": 3800,
        "cpu_usage_pct": 58.0,
        "memory_usage_pct": 59.2,
        "db_query_time_ms": 3400,
        "bandwidth_mbps": 65.0,
        "error_count": 22,
        "complaint_text": "Answer submission failure, database lock timeout error.",
        "network_latency_ms": 27.2,
        "packet_loss_pct": 0.22,
        "incident_type": "Database Bottleneck",
        "scenario_id": "scen_2_db"
    },
    {
        "timestamp": "2026-10-29 09:50:00",
        "active_users": 1400,
        "response_time_ms": 2900,
        "cpu_usage_pct": 54.0,
        "memory_usage_pct": 57.5,
        "db_query_time_ms": 2450,
        "bandwidth_mbps": 72.0,
        "error_count": 14,
        "complaint_text": "Portal spinning wheel on save, queries are crawling.",
        "network_latency_ms": 26.0,
        "packet_loss_pct": 0.16,
        "incident_type": "Database Bottleneck",
        "scenario_id": "scen_2_db"
    },
    {
        "timestamp": "2026-10-29 09:55:00",
        "active_users": 1300,
        "response_time_ms": 1450,
        "cpu_usage_pct": 49.5,
        "memory_usage_pct": 55.0,
        "db_query_time_ms": 1150,
        "bandwidth_mbps": 80.0,
        "error_count": 7,
        "complaint_text": "Slow response when moving to next section.",
        "network_latency_ms": 24.5,
        "packet_loss_pct": 0.14,
        "incident_type": "Database Bottleneck",
        "scenario_id": "scen_2_db"
    },

    # Scenario 3: CPU & Server Compute Overload (10:00 - 10:25)
    # Characterized by 90%+ CPU, high response time, db_query_time remains low/moderate, UI freeze
    {
        "timestamp": "2026-10-29 10:00:00",
        "active_users": 1650,
        "response_time_ms": 1200,
        "cpu_usage_pct": 86.5,
        "memory_usage_pct": 62.0,
        "db_query_time_ms": 95,
        "bandwidth_mbps": 110.0,
        "error_count": 6,
        "complaint_text": "Server feels sluggish, rendering delay on questions.",
        "network_latency_ms": 23.0,
        "packet_loss_pct": 0.12,
        "incident_type": "CPU/Server Overload",
        "scenario_id": "scen_3_cpu"
    },
    {
        "timestamp": "2026-10-29 10:05:00",
        "active_users": 1820,
        "response_time_ms": 2650,
        "cpu_usage_pct": 94.2,
        "memory_usage_pct": 65.5,
        "db_query_time_ms": 110,
        "bandwidth_mbps": 118.0,
        "error_count": 16,
        "complaint_text": "CPU overload, page unresponsive during formula render.",
        "network_latency_ms": 25.4,
        "packet_loss_pct": 0.18,
        "incident_type": "CPU/Server Overload",
        "scenario_id": "scen_3_cpu"
    },
    {
        "timestamp": "2026-10-29 10:10:00",
        "active_users": 1950,
        "response_time_ms": 3900,
        "cpu_usage_pct": 98.7,
        "memory_usage_pct": 68.0,
        "db_query_time_ms": 125,
        "bandwidth_mbps": 122.0,
        "error_count": 28,
        "complaint_text": "Server pegged at 100%, total portal freeze for minutes.",
        "network_latency_ms": 29.0,
        "packet_loss_pct": 0.25,
        "incident_type": "CPU/Server Overload",
        "scenario_id": "scen_3_cpu"
    },
    {
        "timestamp": "2026-10-29 10:15:00",
        "active_users": 1880,
        "response_time_ms": 3200,
        "cpu_usage_pct": 95.1,
        "memory_usage_pct": 66.8,
        "db_query_time_ms": 115,
        "bandwidth_mbps": 115.0,
        "error_count": 21,
        "complaint_text": "Generic slowdown, high server load makes UI lag.",
        "network_latency_ms": 26.8,
        "packet_loss_pct": 0.20,
        "incident_type": "CPU/Server Overload",
        "scenario_id": "scen_3_cpu"
    },
    {
        "timestamp": "2026-10-29 10:20:00",
        "active_users": 1700,
        "response_time_ms": 2100,
        "cpu_usage_pct": 89.0,
        "memory_usage_pct": 63.5,
        "db_query_time_ms": 102,
        "bandwidth_mbps": 105.0,
        "error_count": 12,
        "complaint_text": "Exam portal sluggish, timer lag observed.",
        "network_latency_ms": 24.2,
        "packet_loss_pct": 0.15,
        "incident_type": "CPU/Server Overload",
        "scenario_id": "scen_3_cpu"
    },

    # Scenario 4: Memory Pressure & Leak (10:25 - 10:50)
    # Characterized by 90%+ memory_usage_pct, JVM/GC thrashing, memory exhaustion
    {
        "timestamp": "2026-10-29 10:25:00",
        "active_users": 1400,
        "response_time_ms": 820,
        "cpu_usage_pct": 52.0,
        "memory_usage_pct": 87.5,
        "db_query_time_ms": 85,
        "bandwidth_mbps": 90.0,
        "error_count": 5,
        "complaint_text": "Portal stuttering intermittently, page lag.",
        "network_latency_ms": 22.0,
        "packet_loss_pct": 0.10,
        "incident_type": "Memory Pressure",
        "scenario_id": "scen_4_mem"
    },
    {
        "timestamp": "2026-10-29 10:30:00",
        "active_users": 1450,
        "response_time_ms": 1950,
        "cpu_usage_pct": 64.0,
        "memory_usage_pct": 92.4,
        "db_query_time_ms": 95,
        "bandwidth_mbps": 88.0,
        "error_count": 15,
        "complaint_text": "Garbage collection freeze, sessions suddenly terminating.",
        "network_latency_ms": 23.5,
        "packet_loss_pct": 0.12,
        "incident_type": "Memory Pressure",
        "scenario_id": "scen_4_mem"
    },
    {
        "timestamp": "2026-10-29 10:35:00",
        "active_users": 1480,
        "response_time_ms": 3100,
        "cpu_usage_pct": 71.5,
        "memory_usage_pct": 96.8,
        "db_query_time_ms": 105,
        "bandwidth_mbps": 82.0,
        "error_count": 31,
        "complaint_text": "Session disconnection, out of memory error in portal tab.",
        "network_latency_ms": 24.8,
        "packet_loss_pct": 0.14,
        "incident_type": "Memory Pressure",
        "scenario_id": "scen_4_mem"
    },
    {
        "timestamp": "2026-10-29 10:40:00",
        "active_users": 1420,
        "response_time_ms": 2800,
        "cpu_usage_pct": 68.2,
        "memory_usage_pct": 95.2,
        "db_query_time_ms": 98,
        "bandwidth_mbps": 84.0,
        "error_count": 26,
        "complaint_text": "App heap memory exhausted, forced reload logged me out.",
        "network_latency_ms": 23.9,
        "packet_loss_pct": 0.13,
        "incident_type": "Memory Pressure",
        "scenario_id": "scen_4_mem"
    },
    {
        "timestamp": "2026-10-29 10:45:00",
        "active_users": 1350,
        "response_time_ms": 1750,
        "cpu_usage_pct": 58.0,
        "memory_usage_pct": 90.1,
        "db_query_time_ms": 90,
        "bandwidth_mbps": 86.5,
        "error_count": 11,
        "complaint_text": "Memory leak causing high latency during exam transition.",
        "network_latency_ms": 22.8,
        "packet_loss_pct": 0.11,
        "incident_type": "Memory Pressure",
        "scenario_id": "scen_4_mem"
    },

    # Scenario 5: Network Congestion & Edge Connectivity Blip (10:50 - 11:15)
    # Characterized by high network_latency_ms (250-700ms), high packet_loss_pct (4-14%), server CPU/DB normal
    {
        "timestamp": "2026-10-29 10:50:00",
        "active_users": 1200,
        "response_time_ms": 1100,
        "cpu_usage_pct": 36.0,
        "memory_usage_pct": 52.0,
        "db_query_time_ms": 65,
        "bandwidth_mbps": 195.0,
        "error_count": 8,
        "complaint_text": "Network connection dropped while loading next question.",
        "network_latency_ms": 240.0,
        "packet_loss_pct": 4.5,
        "incident_type": "Network Congestion",
        "scenario_id": "scen_5_net"
    },
    {
        "timestamp": "2026-10-29 10:55:00",
        "active_users": 1220,
        "response_time_ms": 2900,
        "cpu_usage_pct": 38.5,
        "memory_usage_pct": 53.5,
        "db_query_time_ms": 72,
        "bandwidth_mbps": 240.0,
        "error_count": 19,
        "complaint_text": "Session disconnection, server unreachable timeout.",
        "network_latency_ms": 480.0,
        "packet_loss_pct": 9.8,
        "incident_type": "Network Congestion",
        "scenario_id": "scen_5_net"
    },
    {
        "timestamp": "2026-10-29 11:00:00",
        "active_users": 1180,
        "response_time_ms": 3600,
        "cpu_usage_pct": 39.0,
        "memory_usage_pct": 54.0,
        "db_query_time_ms": 68,
        "bandwidth_mbps": 265.0,
        "error_count": 27,
        "complaint_text": "High packet loss, exam portal connection timed out.",
        "network_latency_ms": 620.0,
        "packet_loss_pct": 13.5,
        "incident_type": "Network Congestion",
        "scenario_id": "scen_5_net"
    },
    {
        "timestamp": "2026-10-29 11:05:00",
        "active_users": 1150,
        "response_time_ms": 2400,
        "cpu_usage_pct": 37.0,
        "memory_usage_pct": 52.8,
        "db_query_time_ms": 62,
        "bandwidth_mbps": 220.0,
        "error_count": 15,
        "complaint_text": "Campus gateway congestion, ping is 400ms+ to exam server.",
        "network_latency_ms": 390.0,
        "packet_loss_pct": 7.2,
        "incident_type": "Network Congestion",
        "scenario_id": "scen_5_net"
    },
    {
        "timestamp": "2026-10-29 11:10:00",
        "active_users": 1100,
        "response_time_ms": 1300,
        "cpu_usage_pct": 35.0,
        "memory_usage_pct": 51.5,
        "db_query_time_ms": 58,
        "bandwidth_mbps": 180.0,
        "error_count": 7,
        "complaint_text": "Network stabilizing, intermittent lag on submit.",
        "network_latency_ms": 185.0,
        "packet_loss_pct": 3.1,
        "incident_type": "Network Congestion",
        "scenario_id": "scen_5_net"
    },

    # Scenario 6: Application & Dependency Failure (11:15 - 11:40)
    # Characterized by high error_count (HTTP 500/502/503), moderate CPU/DB, broken endpoint
    {
        "timestamp": "2026-10-29 11:15:00",
        "active_users": 1250,
        "response_time_ms": 950,
        "cpu_usage_pct": 42.0,
        "memory_usage_pct": 53.0,
        "db_query_time_ms": 78,
        "bandwidth_mbps": 82.0,
        "error_count": 32,
        "complaint_text": "Getting HTTP 500 Internal Server Error when submitting exam.",
        "network_latency_ms": 22.0,
        "packet_loss_pct": 0.15,
        "incident_type": "Application/Dependency Failure",
        "scenario_id": "scen_6_app"
    },
    {
        "timestamp": "2026-10-29 11:20:00",
        "active_users": 1280,
        "response_time_ms": 1850,
        "cpu_usage_pct": 46.5,
        "memory_usage_pct": 55.0,
        "db_query_time_ms": 85,
        "bandwidth_mbps": 79.0,
        "error_count": 64,
        "complaint_text": "Answer submission failure, auth microservice returning 502.",
        "network_latency_ms": 23.5,
        "packet_loss_pct": 0.20,
        "incident_type": "Application/Dependency Failure",
        "scenario_id": "scen_6_app"
    },
    {
        "timestamp": "2026-10-29 11:25:00",
        "active_users": 1300,
        "response_time_ms": 2700,
        "cpu_usage_pct": 49.0,
        "memory_usage_pct": 57.0,
        "db_query_time_ms": 92,
        "bandwidth_mbps": 76.0,
        "error_count": 88,
        "complaint_text": "HTTP 503 Service Unavailable, question validation crashed.",
        "network_latency_ms": 24.1,
        "packet_loss_pct": 0.22,
        "incident_type": "Application/Dependency Failure",
        "scenario_id": "scen_6_app"
    },
    {
        "timestamp": "2026-10-29 11:30:00",
        "active_users": 1240,
        "response_time_ms": 2100,
        "cpu_usage_pct": 47.0,
        "memory_usage_pct": 56.0,
        "db_query_time_ms": 88,
        "bandwidth_mbps": 78.5,
        "error_count": 55,
        "complaint_text": "Submission failed with unexpected runtime exception.",
        "network_latency_ms": 23.0,
        "packet_loss_pct": 0.18,
        "incident_type": "Application/Dependency Failure",
        "scenario_id": "scen_6_app"
    },
    {
        "timestamp": "2026-10-29 11:35:00",
        "active_users": 1150,
        "response_time_ms": 1200,
        "cpu_usage_pct": 43.5,
        "memory_usage_pct": 54.0,
        "db_query_time_ms": 80,
        "bandwidth_mbps": 81.0,
        "error_count": 28,
        "complaint_text": "Login failure due to redis session store timeout.",
        "network_latency_ms": 21.8,
        "packet_loss_pct": 0.14,
        "incident_type": "Application/Dependency Failure",
        "scenario_id": "scen_6_app"
    },

    # Scenario 7: Ambiguous / Borderline Multi-Factor Cases (11:40 - 11:55)
    # Demonstrating real-world noise where metrics are borderline
    {
        "timestamp": "2026-10-29 11:40:00",
        "active_users": 950,
        "response_time_ms": 480,
        "cpu_usage_pct": 68.0,
        "memory_usage_pct": 69.5,
        "db_query_time_ms": 280,
        "bandwidth_mbps": 65.0,
        "error_count": 4,
        "complaint_text": "Minor slowdown when loading images, still usable.",
        "network_latency_ms": 45.0,
        "packet_loss_pct": 0.8,
        "incident_type": "Normal",
        "scenario_id": "scen_7_borderline"
    },
    {
        "timestamp": "2026-10-29 11:45:00",
        "active_users": 1120,
        "response_time_ms": 920,
        "cpu_usage_pct": 74.0,
        "memory_usage_pct": 72.0,
        "db_query_time_ms": 480,
        "bandwidth_mbps": 95.0,
        "error_count": 6,
        "complaint_text": "Delay saving answers, unsure if network or portal.",
        "network_latency_ms": 68.0,
        "packet_loss_pct": 1.2,
        "incident_type": "Database Bottleneck",
        "scenario_id": "scen_7_borderline"
    },
    {
        "timestamp": "2026-10-29 11:50:00",
        "active_users": 1200,
        "response_time_ms": 1150,
        "cpu_usage_pct": 82.0,
        "memory_usage_pct": 75.0,
        "db_query_time_ms": 310,
        "bandwidth_mbps": 115.0,
        "error_count": 8,
        "complaint_text": "LMS loading slowly, multiple questions pending render.",
        "network_latency_ms": 75.0,
        "packet_loss_pct": 1.5,
        "incident_type": "CPU/Server Overload",
        "scenario_id": "scen_7_borderline"
    },
    {
        "timestamp": "2026-10-29 11:55:00",
        "active_users": 1050,
        "response_time_ms": 650,
        "cpu_usage_pct": 55.0,
        "memory_usage_pct": 61.0,
        "db_query_time_ms": 180,
        "bandwidth_mbps": 70.0,
        "error_count": 3,
        "complaint_text": "Portal recovered slightly, occasional delay on submit.",
        "network_latency_ms": 32.0,
        "packet_loss_pct": 0.4,
        "incident_type": "Normal",
        "scenario_id": "scen_7_borderline"
    }
]

# Feature definitions
FEATURE_COLUMNS = [
    "active_users",
    "response_time_ms",
    "cpu_usage_pct",
    "memory_usage_pct",
    "db_query_time_ms",
    "bandwidth_mbps",
    "error_count",
    "network_latency_ms",
    "packet_loss_pct"
]

TARGET_COLUMN = "incident_type"

ALL_INCIDENT_TYPES = [
    "Normal",
    "Database Bottleneck",
    "CPU/Server Overload",
    "Memory Pressure",
    "Network Congestion",
    "Application/Dependency Failure"
]

# Baseline threshold references for normal operation (derived from domain knowledge & normal samples)
NORMAL_THRESHOLDS = {
    "response_time_ms": 350.0,
    "cpu_usage_pct": 70.0,
    "memory_usage_pct": 75.0,
    "db_query_time_ms": 150.0,
    "bandwidth_mbps": 150.0,
    "error_count": 3,
    "network_latency_ms": 50.0,
    "packet_loss_pct": 1.0
}


def get_raw_data() -> pd.DataFrame:
    """Returns a pandas DataFrame of the embedded dataset with properly typed columns."""
    df = pd.DataFrame(DATA)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates dataset against hackathon schema, data integrity and sanity ranges.
    Returns (is_valid, validation_errors).
    """
    errors = []
    
    # 1. Row count check (~30-40 samples)
    if not (30 <= len(df) <= 45):
        errors.append(f"Dataset length is {len(df)}; expected approximately 36 samples (30-45).")
        
    # 2. Schema check
    required_fields = [
        "timestamp", "active_users", "response_time_ms", "cpu_usage_pct",
        "memory_usage_pct", "db_query_time_ms", "bandwidth_mbps", "error_count",
        "complaint_text", "network_latency_ms", "packet_loss_pct", "incident_type"
    ]
    for field in required_fields:
        if field not in df.columns:
            errors.append(f"Missing required field: '{field}'")
            
    # 3. Numeric range validations
    if "cpu_usage_pct" in df.columns and (df["cpu_usage_pct"].min() < 0 or df["cpu_usage_pct"].max() > 100):
        errors.append("cpu_usage_pct out of valid [0, 100] range.")
    if "memory_usage_pct" in df.columns and (df["memory_usage_pct"].min() < 0 or df["memory_usage_pct"].max() > 100):
        errors.append("memory_usage_pct out of valid [0, 100] range.")
    if "packet_loss_pct" in df.columns and (df["packet_loss_pct"].min() < 0 or df["packet_loss_pct"].max() > 100):
        errors.append("packet_loss_pct out of valid [0, 100] range.")
    if "response_time_ms" in df.columns and df["response_time_ms"].min() < 0:
        errors.append("response_time_ms has negative values.")
    if "error_count" in df.columns and df["error_count"].min() < 0:
        errors.append("error_count has negative values.")

    # 4. Target class variety check
    if TARGET_COLUMN in df.columns:
        present_classes = set(df[TARGET_COLUMN].unique())
        for req_class in ALL_INCIDENT_TYPES:
            if req_class not in present_classes:
                errors.append(f"Missing required incident category: '{req_class}'")

    return (len(errors) == 0, errors)


if __name__ == "__main__":
    df = get_raw_data()
    is_valid, errs = validate_dataset(df)
    print(f"Dataset Size: {len(df)} rows")
    print(f"Validation Valid: {is_valid}")
    if errs:
        print("Errors:", errs)
    else:
        print("All schema and sanity checks passed!")
        print("Class distributions:")
        print(df[TARGET_COLUMN].value_counts())
