import pytest
import tempfile
import os
import json
from fastapi.testclient import TestClient
from data.session_store import SessionStore
import server

def test_observability_api_metrics():
    # Setup temp DB
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    store = None
    old_db = server.db
    
    try:
        # Create temp store and insert mock sessions and executions
        store = SessionStore(db_path=db_path)
        
        # Patch server's global db
        server.db = store
        
        client = TestClient(server.app)
        
        # 1. Verify response on empty DB
        response = client.get("/api/observability/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 0
        assert data["overall_success_rate"] == 0.0
        
        # 2. Insert mock sessions and execution traces
        # Run 1: Completed, 2 parallel steps, 1 memory hit
        sid1 = "session-1"
        store.create_session(sid1, "titanic.csv", 891, 12)
        
        plan1 = {
            "steps": [
                {"id": "step_1", "agent": "schema", "depends_on": []},
                {"id": "step_2", "agent": "quality", "depends_on": []}
            ]
        }
        
        state1 = {
            "status": "completed",
            "filename": "titanic.csv",
            "graph_version": "phase4_langgraph_v1",
            "graph_build_time": "2026-06-09T20:00:00Z",
            "trace": [
                {"step_id": "step_1", "agent": "schema", "status": "completed", "duration_sec": 1.2},
                {"step_id": "step_2", "agent": "quality", "status": "completed", "duration_sec": 0.8}
            ],
            "node_telemetry": {
                "executor": {"attempts": 1, "durations": [1.0]},  # parallel runtime = 1.0s, sum of steps = 2.0s
                "step_1": {"attempts": 1, "durations": [1.2]},
                "step_2": {"attempts": 1, "durations": [0.8]}
            },
            "planner_context": {
                "advisory_memories": {
                    "similar_datasets": [
                        {"rank_score": 0.85, "filename": "titanic_v2.csv"}
                    ]
                }
            },
            "checkpoint_count": 3,
            "resume_events": [{"action": "approved"}]
        }
        store.save_execution_state(sid1, plan1, state1)
        
        # Run 2: Failed, 1 retry attempt
        sid2 = "session-2"
        store.create_session(sid2, "sales.csv", 100, 5)
        
        plan2 = {
            "steps": [
                {"id": "step_1", "agent": "schema", "depends_on": []}
            ]
        }
        
        state2 = {
            "status": "failed",
            "filename": "sales.csv",
            "graph_version": "phase4_langgraph_v2",
            "graph_build_time": "2026-06-09T21:00:00Z",
            "trace": [
                {"step_id": "step_1", "agent": "schema", "status": "failed", "duration_sec": 1.5}
            ],
            "node_telemetry": {
                "executor": {"attempts": 1, "durations": [1.5]},
                "step_1": {"attempts": 2, "durations": [0.7, 0.8]}  # 1 retry
            },
            "planner_context": {
                "advisory_memories": {
                    "similar_datasets": [
                        {"rank_score": 0.50, "filename": "sales_v2.csv"} # rejected!
                    ]
                }
            },
            "checkpoint_count": 2,
            "resume_events": []
        }
        store.save_execution_state(sid2, plan2, state2)
        
        # Fetch metrics again
        response = client.get("/api/observability/metrics")
        assert response.status_code == 200
        data = response.json()
        
        # Assertions
        assert data["total_runs"] == 2
        assert data["overall_success_rate"] == 50.0  # 1 of 2 successful
        
        # Parallel speedup calculation:
        # Sum durations of steps for Run 1 = 1.2 + 0.8 = 2.0s
        # Parallel duration (executor) for Run 1 = 1.0s
        # Sum durations of steps for Run 2 = 1.5s
        # Parallel duration (executor) for Run 2 = 1.5s
        # Total sum step durations = 3.5s
        # Total parallel executor duration = 2.5s
        # Speedup = (3.5 - 2.5) / 3.5 = 28.57% -> 28.6%
        assert data["parallel_speedup"] == 28.6
        
        # Memory metrics
        assert data["memory_hit_rate"] == 100.0  # both have advisory_memories context
        assert data["memory_metrics"]["acceptance_rate"] == 50.0 # 1 accept (0.85), 1 reject (0.50)
        assert data["memory_metrics"]["rejection_rate"] == 50.0
        assert data["memory_metrics"]["avg_confidence"] == 0.68  # avg of 0.85 and 0.50
        
        # Checkpoint restores
        assert data["checkpoint_restores"] == 1
        
        # Retry frequency
        # Run 1: 2 steps, 0 retries
        # Run 2: 1 step, 1 retry
        # Total steps executed = 3, total retries = 1. Frequency = 1 / 3 = 33.3%
        assert data["retry_frequency"] == 33.3
        
        # Version distribution
        assert any(v["version"] == "phase4_langgraph_v1" and v["runs"] == 1 for v in data["version_distribution"])
        assert any(v["version"] == "phase4_langgraph_v2" and v["runs"] == 1 for v in data["version_distribution"])
        
    finally:
        server.db = old_db
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
