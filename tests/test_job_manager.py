"""
Test suite for Phase 2: Persistent Job Manager & Backend Cancellation.

Validates SQLite job persistence, idempotency, cooperative cancellation signals,
multi-agent pipeline early termination, and FastAPI job endpoints.
"""

from __future__ import annotations

import tempfile
import threading
import time
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents.router import run_agent_pipeline
from data.job_manager import JobManager
from server import app


@pytest.fixture
def temp_job_mgr():
    """Create a temporary JobManager with an isolated SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    mgr = JobManager(db_path=db_path)
    yield mgr
    mgr.close()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ═══════════════════════════════════════════════
# 1. Job Manager CRUD & Lifecycle Tests
# ═══════════════════════════════════════════════

def test_job_create_and_lifecycle(temp_job_mgr: JobManager):
    """Test job creation, progress tracking, and successful completion."""
    job = temp_job_mgr.create_job("sess_123", "auto_eda")
    job_id = job["job_id"]

    assert job_id.startswith("job_")
    assert job["status"] == "pending"
    assert job["progress_pct"] == 0

    # Update progress
    temp_job_mgr.update_progress(job_id, 40, "hypothesis", status="running")
    updated = temp_job_mgr.get_job(job_id)
    assert updated is not None
    assert updated["status"] == "running"
    assert updated["progress_pct"] == 40
    assert updated["stage_name"] == "hypothesis"

    # Complete job
    temp_job_mgr.complete_job(job_id, result={"insights": ["Found pattern A"]})
    completed = temp_job_mgr.get_job(job_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["progress_pct"] == 100
    assert completed["completed_at"] is not None
    assert completed["result"] == {"insights": ["Found pattern A"]}


def test_job_failure_lifecycle(temp_job_mgr: JobManager):
    """Test marking a job as failed."""
    job = temp_job_mgr.create_job("sess_fail", "ml_train")
    job_id = job["job_id"]

    temp_job_mgr.fail_job(job_id, "Out of memory error")
    failed = temp_job_mgr.get_job(job_id)
    assert failed is not None
    assert failed["status"] == "failed"
    assert "Out of memory error" in failed["error_message"]
    assert failed["completed_at"] is not None


# ═══════════════════════════════════════════════
# 2. Idempotency & Concurrency Tests
# ═══════════════════════════════════════════════

def test_job_idempotency(temp_job_mgr: JobManager):
    """Test that creating a duplicate job for the same session returns the existing active job."""
    job1 = temp_job_mgr.create_job("sess_idem", "auto_eda", idempotent=True)
    temp_job_mgr.update_progress(job1["job_id"], 25, "schema", status="running")

    # Second creation call should return the active job, not insert a duplicate
    job2 = temp_job_mgr.create_job("sess_idem", "auto_eda", idempotent=True)
    assert job1["job_id"] == job2["job_id"]

    # Once completed, a new job can be created
    temp_job_mgr.complete_job(job1["job_id"], {"status": "ok"})
    job3 = temp_job_mgr.create_job("sess_idem", "auto_eda", idempotent=True)
    assert job3["job_id"] != job1["job_id"]


# ═══════════════════════════════════════════════
# 3. Cancellation Signal & Session Scoping
# ═══════════════════════════════════════════════

def test_job_cancellation_signal(temp_job_mgr: JobManager):
    """Test that cancel_job sets the in-memory threading event and updates DB state."""
    job = temp_job_mgr.create_job("sess_cancel", "auto_eda")
    job_id = job["job_id"]

    event = temp_job_mgr.get_cancel_event(job_id)
    assert not event.is_set()
    assert not temp_job_mgr.is_cancelled(job_id)

    # Trigger cancellation
    success = temp_job_mgr.cancel_job(job_id)
    assert success is True
    assert event.is_set()
    assert temp_job_mgr.is_cancelled(job_id)

    cancelled_job = temp_job_mgr.get_job(job_id)
    assert cancelled_job["status"] == "cancelled"
    assert cancelled_job["completed_at"] is not None


def test_session_scoped_cancellation(temp_job_mgr: JobManager):
    """Test cancelling all active jobs belonging to a specific session."""
    j1 = temp_job_mgr.create_job("sess_A", "auto_eda")
    j2 = temp_job_mgr.create_job("sess_A", "ml_train", idempotent=False)
    j3 = temp_job_mgr.create_job("sess_B", "auto_eda")

    cancelled_ids = temp_job_mgr.cancel_session_jobs("sess_A")
    assert j1["job_id"] in cancelled_ids
    assert j2["job_id"] in cancelled_ids
    assert j3["job_id"] not in cancelled_ids

    # sess_B should still be pending
    b_job = temp_job_mgr.get_job(j3["job_id"])
    assert b_job["status"] == "pending"


# ═══════════════════════════════════════════════
# 4. Cooperative Agent Pipeline Cancellation
# ═══════════════════════════════════════════════

def test_pipeline_cooperative_cancellation():
    """Test that run_agent_pipeline halts cooperatively when cancel_event is set."""
    sample_df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "salary": [50000, 60000, 75000, 90000],
    })
    intel = {
        "dataset_name": "test.csv",
        "row_count": 4,
        "column_count": 2,
        "columns": {"age": {"dtype": "int64"}, "salary": {"dtype": "int64"}},
    }

    cancel_ev = threading.Event()
    # Pre-set the cancel event so it aborts before any stage
    cancel_ev.set()

    stages = ["schema", "quality", "hypothesis", "eda", "insight"]
    results = run_agent_pipeline(stages, intel, df=sample_df, cancel_event=cancel_ev)

    assert "_pipeline_cancelled" in results
    assert results["_pipeline_cancelled"]["cancelled"] is True
    assert results["_pipeline_cancelled"]["cancelled_at_stage"] == "schema"
    assert "insight" not in results


def test_pipeline_mid_flight_cancellation():
    """Test that setting cancel_event during pipeline execution halts remaining stages."""
    sample_df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "salary": [50000, 60000, 75000, 90000],
    })
    intel = {
        "dataset_name": "test.csv",
        "row_count": 4,
        "column_count": 2,
        "columns": {"age": {"dtype": "int64"}, "salary": {"dtype": "int64"}},
    }

    cancel_ev = threading.Event()

    # Progress callback that triggers cancellation after 'schema' stage completes
    def on_progress(stage: str, pct: int):
        if stage == "schema" and pct > 0:
            cancel_ev.set()

    stages = ["schema", "quality", "hypothesis", "eda", "insight"]
    results = run_agent_pipeline(
        stages,
        intel,
        df=sample_df,
        cancel_event=cancel_ev,
        progress_callback=on_progress,
    )

    assert "_pipeline_cancelled" in results
    assert results["_pipeline_cancelled"]["cancelled"] is True
    # Subsequent stages should NOT have executed
    assert "hypothesis" not in results
    assert "insight" not in results


# ═══════════════════════════════════════════════
# 5. FastAPI Endpoints Integration
# ═══════════════════════════════════════════════

def test_api_jobs_crud_and_status(client: TestClient):
    """Test FastAPI /api/jobs/... endpoints."""
    # 1. Create Job
    resp = client.post("/api/jobs/create", json={"session_id": "test_api_sess", "job_type": "auto_eda"})
    assert resp.status_code == 200
    data = resp.json()
    job_id = data["job_id"]
    assert job_id.startswith("job_")
    assert data["status"] == "pending"

    # 2. Get Job Status
    resp = client.get(f"/api/jobs/{job_id}/status")
    assert resp.status_code == 200
    status_data = resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] == "pending"

    # 3. Cancel Job
    resp = client.post(f"/api/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is True

    # 4. Verify Status Updated to Cancelled
    resp = client.get(f"/api/jobs/{job_id}/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_api_cancel_session_endpoint(client: TestClient):
    """Test FastAPI /api/cancel/{sid} endpoint."""
    resp = client.post("/api/jobs/create", json={"session_id": "sess_bulk_cancel", "job_type": "auto_eda"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    cancel_resp = client.post("/api/cancel/sess_bulk_cancel")
    assert cancel_resp.status_code == 200
    cancel_data = cancel_resp.json()
    assert cancel_data["session_id"] == "sess_bulk_cancel"
    assert job_id in cancel_data["cancelled_jobs"]

    # Verify job status
    status_resp = client.get(f"/api/jobs/{job_id}/status")
    assert status_resp.json()["status"] == "cancelled"


# ═══════════════════════════════════════════════
# 6. Recovery & Terminal State Monotonicity
# ═══════════════════════════════════════════════

def test_startup_stale_job_recovery():
    """Test that jobs left running/pending during a server crash are recovered as 'interrupted'."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # 1. Initialize first manager without auto-recovery
    mgr1 = JobManager(db_path=db_path, auto_recover=False)
    j1 = mgr1.create_job("sess_stale_1", "auto_eda")
    mgr1.update_progress(j1["job_id"], 50, "hypothesis", status="running")
    j2 = mgr1.create_job("sess_stale_2", "ml_train")
    mgr1.close()

    # 2. Simulate server restart: initialize second manager with auto_recover=True
    mgr2 = JobManager(db_path=db_path, auto_recover=True)
    rec_j1 = mgr2.get_job(j1["job_id"])
    rec_j2 = mgr2.get_job(j2["job_id"])

    assert rec_j1["status"] == "interrupted"
    assert "Process terminated unexpectedly" in rec_j1["error_message"]
    assert rec_j1["completed_at"] is not None

    assert rec_j2["status"] == "interrupted"
    assert rec_j2["completed_at"] is not None
    mgr2.close()


def test_transactional_unique_active_job_constraint(temp_job_mgr: JobManager):
    """Test that the database-level unique partial index prevents duplicate active jobs."""
    import sqlite3

    j = temp_job_mgr.create_job("sess_unique_test", "auto_eda")
    # Attempt direct raw SQL insertion of another running job for the same session and type
    with pytest.raises(sqlite3.IntegrityError):
        temp_job_mgr._conn.execute(
            """
            INSERT INTO jobs (job_id, session_id, job_type, status, progress_pct)
            VALUES ('job_duplicate_direct', 'sess_unique_test', 'auto_eda', 'running', 10)
            """
        )


def test_terminal_state_monotonicity(temp_job_mgr: JobManager):
    """Test that once a job reaches a terminal state, late callbacks cannot overwrite it."""
    j = temp_job_mgr.create_job("sess_mono", "auto_eda")
    job_id = j["job_id"]

    # Cancel the job (terminal state)
    temp_job_mgr.cancel_job(job_id)
    assert temp_job_mgr.get_job(job_id)["status"] == "cancelled"

    # Attempt late progress update
    updated = temp_job_mgr.update_progress(job_id, 99, "insight", status="running")
    assert updated is False
    assert temp_job_mgr.get_job(job_id)["status"] == "cancelled"

    # Attempt late complete_job
    completed = temp_job_mgr.complete_job(job_id, {"late": "result"})
    assert completed is False
    assert temp_job_mgr.get_job(job_id)["status"] == "cancelled"


