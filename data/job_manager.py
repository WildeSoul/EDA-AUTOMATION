"""
SQLite-backed Persistent Job Manager and Backend Cancellation Protocol.

Provides durable tracking of long-running analytics jobs (Auto-EDA, Feature Engineering,
ML Training) with cooperative thread-safe cancellation across multi-agent workflows.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from config.settings import SESSION_DB_PATH

logger = logging.getLogger(__name__)

_JOB_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    job_type       TEXT NOT NULL,
    status         TEXT NOT NULL,
    progress_pct   INTEGER DEFAULT 0,
    stage_name     TEXT DEFAULT '',
    result_json    TEXT,
    error_message  TEXT,
    created_at     TEXT DEFAULT (datetime('now')),
    updated_at     TEXT DEFAULT (datetime('now')),
    completed_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_job_per_session 
ON jobs(session_id, job_type) 
WHERE status IN ('pending', 'running');
"""


class JobManager:
    """Manages SQLite job lifecycle, statuses, and cooperative thread cancellation."""

    def __init__(self, db_path: str | None = None, auto_recover: bool = True) -> None:
        self._db_path = db_path or SESSION_DB_PATH
        self._lock = threading.RLock()
        self._cancellation_events: dict[str, threading.Event] = {}
        self._active_worker_pids: dict[str, int] = {}

        try:
            self._conn = sqlite3.connect(
                self._db_path,
                timeout=30.0,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            try:
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA synchronous = NORMAL")
            except sqlite3.Error as pragma_err:
                logger.debug("Failed setting WAL pragma on jobs db: %s", pragma_err)
            self._conn.executescript(_JOB_SCHEMA_SQL)
            self._conn.commit()
            logger.info("JobManager initialized on %s", self._db_path)

            if auto_recover:
                self.recover_stale_jobs()
        except sqlite3.Error as exc:
            logger.error("Failed to initialize JobManager: %s", exc)
            raise

    def recover_stale_jobs(self) -> int:
        """Detect jobs left in pending/running state on startup and mark them interrupted."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            cur = self._conn.execute(
                """
                UPDATE jobs 
                SET status = 'interrupted',
                    error_message = 'Process terminated unexpectedly (server restart / crash recovery)',
                    completed_at = ?,
                    updated_at = ?
                WHERE status IN ('pending', 'running')
                """,
                (now_iso, now_iso),
            )
            self._conn.commit()
            if cur.rowcount > 0:
                logger.warning("Recovered and marked %d stale jobs as 'interrupted'", cur.rowcount)
            return cur.rowcount

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception as e:
                logger.debug("Error closing JobManager connection: %s", e)

    def register_worker_pid(self, job_id: str, pid: int) -> None:
        """Register active worker process PID for hard process termination upon cancellation."""
        with self._lock:
            self._active_worker_pids[job_id] = pid

    def unregister_worker_pid(self, job_id: str) -> None:
        """Unregister active worker process PID upon completion."""
        with self._lock:
            self._active_worker_pids.pop(job_id, None)

    def create_job(
        self,
        session_id: str,
        job_type: str = "auto_eda",
        idempotent: bool = True,
    ) -> dict[str, Any]:
        """Create a new job record with transactional idempotency guarantees."""
        with self._lock:
            if idempotent:
                cur = self._conn.execute(
                    """
                    SELECT * FROM jobs 
                    WHERE session_id = ? AND job_type = ? AND status IN ('pending', 'running')
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id, job_type),
                )
                row = cur.fetchone()
                if row:
                    job_dict = dict(row)
                    if job_dict["job_id"] not in self._cancellation_events:
                        self._cancellation_events[job_dict["job_id"]] = threading.Event()
                    return job_dict

            job_id = f"job_{uuid.uuid4().hex[:12]}"
            cancel_ev = threading.Event()
            self._cancellation_events[job_id] = cancel_ev

            now_iso = datetime.now(timezone.utc).isoformat()
            try:
                self._conn.execute(
                    """
                    INSERT INTO jobs (job_id, session_id, job_type, status, progress_pct, stage_name, created_at, updated_at)
                    VALUES (?, ?, ?, 'pending', 0, 'initialized', ?, ?)
                    """,
                    (job_id, session_id, job_type, now_iso, now_iso),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                # Concurrent race condition: another request committed an active job first
                cur = self._conn.execute(
                    """
                    SELECT * FROM jobs 
                    WHERE session_id = ? AND job_type = ? AND status IN ('pending', 'running')
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id, job_type),
                )
                row = cur.fetchone()
                if row:
                    job_dict = dict(row)
                    if job_dict["job_id"] not in self._cancellation_events:
                        self._cancellation_events[job_dict["job_id"]] = threading.Event()
                    return job_dict
                raise

            return {
                "job_id": job_id,
                "session_id": session_id,
                "job_type": job_type,
                "status": "pending",
                "progress_pct": 0,
                "stage_name": "initialized",
                "created_at": now_iso,
                "updated_at": now_iso,
                "completed_at": None,
                "result_json": None,
                "error_message": None,
            }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve job record by ID."""
        with self._lock:
            cur = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("result_json"):
                try:
                    data["result"] = json.loads(data["result_json"])
                except Exception:
                    data["result"] = data["result_json"]
            else:
                data["result"] = None
            return data

    def get_active_jobs_for_session(self, session_id: str) -> list[dict[str, Any]]:
        """Return all active jobs for a session."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE session_id = ? AND status IN ('pending', 'running')",
                (session_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def update_progress(
        self,
        job_id: str,
        progress_pct: int,
        stage_name: str = "",
        status: str = "running",
    ) -> bool:
        """Update job execution progress and current stage. Monotonic: fails if terminal."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            cur = self._conn.execute(
                """
                UPDATE jobs 
                SET progress_pct = ?, stage_name = ?, status = ?, updated_at = ?
                WHERE job_id = ? AND status IN ('pending', 'running')
                """,
                (progress_pct, stage_name, status, now_iso, job_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def complete_job(self, job_id: str, result: Any = None) -> bool:
        """Mark job as successfully completed. Monotonic: cannot overwrite terminal states."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            res_str = json.dumps(result) if result is not None else None
            cur = self._conn.execute(
                """
                UPDATE jobs 
                SET status = 'completed', progress_pct = 100, result_json = ?, completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status IN ('pending', 'running')
                """,
                (res_str, now_iso, now_iso, job_id),
            )
            self._conn.commit()
            self._cancellation_events.pop(job_id, None)
            self._active_worker_pids.pop(job_id, None)
            return cur.rowcount > 0

    def fail_job(self, job_id: str, error_message: str) -> bool:
        """Mark job as failed with error details. Monotonic: cannot overwrite terminal states."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            cur = self._conn.execute(
                """
                UPDATE jobs 
                SET status = 'failed', error_message = ?, completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status IN ('pending', 'running')
                """,
                (str(error_message), now_iso, now_iso, job_id),
            )
            self._conn.commit()
            self._cancellation_events.pop(job_id, None)
            self._active_worker_pids.pop(job_id, None)
            return cur.rowcount > 0

    def cancel_job(self, job_id: str) -> bool:
        """Signal cooperative cancellation, terminate active worker processes, and update status."""
        with self._lock:
            # 1. Trigger cooperative cancellation event in memory
            ev = self._cancellation_events.get(job_id)
            if ev:
                ev.set()

            # 2. Terminate active worker process tree if registered
            pid = self._active_worker_pids.pop(job_id, None)
            if pid:
                try:
                    import psutil
                    parent = psutil.Process(pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except Exception:
                            pass
                    parent.kill()
                except Exception:
                    pass

            now_iso = datetime.now(timezone.utc).isoformat()
            cur = self._conn.execute(
                """
                UPDATE jobs 
                SET status = 'cancelled', completed_at = ?, updated_at = ?
                WHERE job_id = ? AND status IN ('pending', 'running')
                """,
                (now_iso, now_iso, job_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def cancel_session_jobs(self, session_id: str) -> list[str]:
        """Cancel all pending or running jobs for a session."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT job_id FROM jobs WHERE session_id = ? AND status IN ('pending', 'running')",
                (session_id,),
            )
            job_ids = [r[0] for r in cur.fetchall()]
            for jid in job_ids:
                self.cancel_job(jid)
            return job_ids

    def get_cancel_event(self, job_id: str) -> threading.Event:
        """Retrieve or create cooperative cancellation event for a job."""
        with self._lock:
            if job_id not in self._cancellation_events:
                self._cancellation_events[job_id] = threading.Event()
            return self._cancellation_events[job_id]

    def is_cancelled(self, job_id: str) -> bool:
        """Check if job has received a cancellation request."""
        with self._lock:
            ev = self._cancellation_events.get(job_id)
            if ev and ev.is_set():
                return True
            cur = self._conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            return bool(row and row[0] == "cancelled")


# Global singleton instance
job_manager = JobManager()
