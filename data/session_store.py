"""
SQLite-backed session persistence for the AI Data Intelligence Platform.

Stores sessions, DataFrames (as Parquet blobs), chat history, and
per-agent results in a single thread-safe SQLite database.
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import threading
from typing import Any

import pandas as pd

from config.settings import SESSION_DB_PATH

logger = logging.getLogger(__name__)

# ── SQL DDL ──────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    sid            TEXT PRIMARY KEY,
    filename       TEXT NOT NULL,
    created_at     TEXT DEFAULT (datetime('now')),
    row_count      INTEGER,
    col_count      INTEGER,
    intelligence   TEXT,
    quality_score  REAL,
    quality_grade  TEXT
);

CREATE TABLE IF NOT EXISTS dataframes (
    sid   TEXT PRIMARY KEY,
    data  BLOB,
    FOREIGN KEY (sid) REFERENCES sessions(sid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sid         TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    agent_name  TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (sid) REFERENCES sessions(sid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sid         TEXT NOT NULL,
    agent_name  TEXT NOT NULL,
    result      TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (sid) REFERENCES sessions(sid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS execution_plans (
    sid          TEXT PRIMARY KEY,
    plan_json    TEXT NOT NULL,
    state_json   TEXT NOT NULL,
    updated_at   TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (sid) REFERENCES sessions(sid) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    fingerprint TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    version     INTEGER NOT NULL,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS dataset_links (
    source_fingerprint  TEXT NOT NULL,
    target_fingerprint  TEXT NOT NULL,
    relationship_type   TEXT NOT NULL, -- "version_update", "derived_subset", "similar_schema", "cross_domain"
    similarity_score    REAL,
    created_at          TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (source_fingerprint, target_fingerprint, relationship_type)
);

"""


class SessionStore:
    """Thread-safe SQLite store for session data, DataFrames, chat logs,
    and agent results."""

    # ── lifecycle ────────────────────────────────────────────────────

    def __init__(self, db_path: str | None = None) -> None:
        """Open (or create) the SQLite database and ensure tables exist.

        Args:
            db_path: Filesystem path for the .db file.
                     Defaults to ``SESSION_DB_PATH`` from project config.
        """
        self._db_path: str = db_path or SESSION_DB_PATH
        self.lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(
                self._db_path,
                timeout=30.0,
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            try:
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA synchronous = NORMAL")
            except sqlite3.Error as pragma_err:
                logger.debug("Failed to set WAL/synchronous pragmas: %s", pragma_err)
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
            logger.info("SessionStore initialised — db=%s", self._db_path)
        except sqlite3.Error as exc:
            logger.error("Failed to initialise SessionStore: %s", exc)
            raise

        # Dynamically wrap all public methods to be thread-safe using RLock
        import functools
        def thread_safe_wrap(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.lock:
                    return func(*args, **kwargs)
            return wrapper

        for attr_name in list(dir(self)):
            if not attr_name.startswith("_") and attr_name not in ("lock",):
                attr = getattr(self, attr_name)
                if callable(attr) and attr_name != "close":
                    setattr(self, attr_name, thread_safe_wrap(attr))

    def close(self) -> None:
        """Close the underlying database connection."""
        with self.lock:
            try:
                self._conn.close()
                logger.info("SessionStore connection closed.")
            except sqlite3.Error as exc:
                logger.error("Error closing SessionStore: %s", exc)

    # ── sessions ─────────────────────────────────────────────────────

    def create_session(
        self,
        sid: str,
        filename: str,
        row_count: int,
        col_count: int,
        intelligence: dict[str, Any] | None = None,
    ) -> str:
        """Insert a new session row.

        Args:
            sid: Unique session identifier.
            filename: Name of the uploaded file.
            row_count: Number of rows in the dataset.
            col_count: Number of columns in the dataset.
            intelligence: Optional intelligence dict to store as JSON.

        Returns:
            The session id that was created.
        """
        intel_json: str | None = None
        quality_score: float | None = None
        quality_grade: str | None = None

        if intelligence:
            intel_json = json.dumps(intelligence, default=str)
            quality_score = intelligence.get("quality_score")
            quality_grade = intelligence.get("quality_grade")

        try:
            self._conn.execute(
                """
                INSERT INTO sessions
                    (sid, filename, row_count, col_count,
                     intelligence, quality_score, quality_grade)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, filename, row_count, col_count,
                 intel_json, quality_score, quality_grade),
            )
            self._conn.commit()
            logger.info("Session created: sid=%s, file=%s", sid, filename)
        except sqlite3.Error as exc:
            logger.error("create_session failed for sid=%s: %s", sid, exc)
            raise
        return sid

    def get_session(self, sid: str) -> dict[str, Any] | None:
        """Fetch a single session by id.

        Returns:
            A dict with session metadata, or ``None`` if not found.
        """
        try:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE sid = ?", (sid,)
            ).fetchone()
        except sqlite3.Error as exc:
            logger.error("get_session failed for sid=%s: %s", sid, exc)
            return None

        if row is None:
            return None

        result = dict(row)
        if result.get("intelligence"):
            try:
                result["intelligence"] = json.loads(result["intelligence"])
            except json.JSONDecodeError:
                logger.warning("Corrupt intelligence JSON for sid=%s", sid)
        return result

    def update_intelligence(self, sid: str, intelligence: dict[str, Any]) -> None:
        """Overwrite the intelligence payload and derived quality columns.

        Args:
            sid: Session identifier.
            intelligence: Full intelligence dict produced by the analysis
                          pipeline.
        """
        intel_json = json.dumps(intelligence, default=str)
        quality_score = intelligence.get("quality_score")
        quality_grade = intelligence.get("quality_grade")

        try:
            self._conn.execute(
                """
                UPDATE sessions
                SET intelligence  = ?,
                    quality_score = ?,
                    quality_grade = ?
                WHERE sid = ?
                """,
                (intel_json, quality_score, quality_grade, sid),
            )
            self._conn.commit()
            logger.info("Intelligence updated for sid=%s", sid)
        except sqlite3.Error as exc:
            logger.error("update_intelligence failed for sid=%s: %s", sid, exc)
            raise

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent sessions (newest first).

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            A list of session dicts.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT sid, filename, created_at, row_count, col_count,
                       quality_score, quality_grade
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.error("list_sessions failed: %s", exc)
            return []

    def delete_session(self, sid: str) -> None:
        """Delete a session and all related data (cascading).

        Args:
            sid: Session identifier to remove.
        """
        try:
            self._conn.execute("DELETE FROM sessions WHERE sid = ?", (sid,))
            self._conn.commit()
            logger.info("Session deleted: sid=%s", sid)
        except sqlite3.Error as exc:
            logger.error("delete_session failed for sid=%s: %s", sid, exc)
            raise

    # ── dataframe persistence ────────────────────────────────────────

    def save_dataframe(self, sid: str, df: pd.DataFrame) -> None:
        """Serialize a DataFrame to Parquet bytes and store in the DB.

        Args:
            sid: Session identifier.
            df: The pandas DataFrame to persist.
        """
        try:
            buf = io.BytesIO()
            df.to_parquet(buf, engine="pyarrow", index=False)
            blob = buf.getvalue()
            self._conn.execute(
                """
                INSERT INTO dataframes (sid, data) VALUES (?, ?)
                ON CONFLICT(sid) DO UPDATE SET data = excluded.data
                """,
                (sid, blob),
            )
            self._conn.commit()
            logger.info(
                "DataFrame saved for sid=%s (%d rows, %.1f KB)",
                sid, len(df), len(blob) / 1024,
            )
        except Exception as exc:
            logger.error("save_dataframe failed for sid=%s: %s", sid, exc)
            raise

    def load_dataframe(self, sid: str) -> pd.DataFrame | None:
        """Deserialize a DataFrame from stored Parquet bytes.

        Args:
            sid: Session identifier.

        Returns:
            The reconstituted DataFrame, or ``None`` if not found.
        """
        try:
            row = self._conn.execute(
                "SELECT data FROM dataframes WHERE sid = ?", (sid,)
            ).fetchone()
        except sqlite3.Error as exc:
            logger.error("load_dataframe query failed for sid=%s: %s", sid, exc)
            return None

        if row is None or row["data"] is None:
            return None

        try:
            buf = io.BytesIO(row["data"])
            return pd.read_parquet(buf, engine="pyarrow")
        except Exception as exc:
            logger.error("load_dataframe deserialization failed for sid=%s: %s", sid, exc)
            return None

    # ── chat history ─────────────────────────────────────────────────

    def save_chat(
        self,
        sid: str,
        role: str,
        content: str,
        agent_name: str | None = None,
    ) -> None:
        """Append a chat message to history.

        Args:
            sid: Session identifier.
            role: Message role (e.g. ``"user"``, ``"assistant"``).
            content: Message body.
            agent_name: Optional agent that produced the message.
        """
        try:
            self._conn.execute(
                """
                INSERT INTO chat_history (sid, role, content, agent_name)
                VALUES (?, ?, ?, ?)
                """,
                (sid, role, content, agent_name),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("save_chat failed for sid=%s: %s", sid, exc)
            raise

    def get_chat_history(
        self, sid: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Retrieve chat messages for a session (oldest first).

        Args:
            sid: Session identifier.
            limit: Maximum number of messages to return.

        Returns:
            A list of message dicts with keys
            ``id, sid, role, content, agent_name, created_at``.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT id, sid, role, content, agent_name, created_at
                FROM chat_history
                WHERE sid = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (sid, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.error("get_chat_history failed for sid=%s: %s", sid, exc)
            return []

    # ── agent results ────────────────────────────────────────────────

    def save_agent_result(
        self, sid: str, agent_name: str, result: dict[str, Any]
    ) -> None:
        """Persist a JSON-serialisable result produced by an agent.

        Args:
            sid: Session identifier.
            agent_name: Name of the agent (e.g. ``"schema"``, ``"quality"``).
            result: Arbitrary dict to store.
        """
        try:
            result_json = json.dumps(result, default=str)
            self._conn.execute(
                """
                INSERT INTO agent_results (sid, agent_name, result)
                VALUES (?, ?, ?)
                """,
                (sid, agent_name, result_json),
            )
            self._conn.commit()
            logger.info("Agent result saved: sid=%s, agent=%s", sid, agent_name)
        except Exception as exc:
            logger.error(
                "save_agent_result failed for sid=%s, agent=%s: %s",
                sid, agent_name, exc,
            )
            raise

    def get_agent_result(
        self, sid: str, agent_name: str
    ) -> dict[str, Any] | None:
        """Return the latest result for a given agent in a session.

        Args:
            sid: Session identifier.
            agent_name: Agent whose result to fetch.

        Returns:
            The deserialised result dict, or ``None`` if absent.
        """
        try:
            row = self._conn.execute(
                """
                SELECT result FROM agent_results
                WHERE sid = ? AND agent_name = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (sid, agent_name),
            ).fetchone()
        except sqlite3.Error as exc:
            logger.error(
                "get_agent_result failed for sid=%s, agent=%s: %s",
                sid, agent_name, exc,
            )
            return None

        if row is None:
            return None

        try:
            return json.loads(row["result"])
        except json.JSONDecodeError:
            logger.warning(
                "Corrupt agent result JSON for sid=%s, agent=%s", sid, agent_name
            )
            return None

    def get_all_agent_results(
        self, sid: str
    ) -> dict[str, dict[str, Any]]:
        """Fetch the latest result for every agent in a session.

        Args:
            sid: Session identifier.

        Returns:
            Mapping of ``{agent_name: result_dict}``.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT agent_name, result
                FROM agent_results
                WHERE sid = ?
                ORDER BY id ASC
                """,
                (sid,),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.error("get_all_agent_results failed for sid=%s: %s", sid, exc)
            return {}

        results: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                results[row["agent_name"]] = json.loads(row["result"])
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping corrupt result for agent=%s in sid=%s",
                    row["agent_name"], sid,
                )
        return results

    # ── execution plans ──────────────────────────────────────────────

    def save_execution_state(
        self, sid: str, plan: dict[str, Any], state: dict[str, Any]
    ) -> None:
        """Persist a JSON-serialisable execution plan and running state trace.

        Args:
            sid: Session identifier.
            plan: Execution plan dict.
            state: Engine execution state dict.
        """
        try:
            plan_json = json.dumps(plan, default=str)
            state_json = json.dumps(state, default=str)
            self._conn.execute(
                """
                INSERT INTO execution_plans (sid, plan_json, state_json, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(sid) DO UPDATE SET
                    plan_json = excluded.plan_json,
                    state_json = excluded.state_json,
                    updated_at = datetime('now')
                """,
                (sid, plan_json, state_json),
            )
            self._conn.commit()
            logger.info("Execution state saved: sid=%s", sid)
        except Exception as exc:
            logger.error("save_execution_state failed for sid=%s: %s", sid, exc)
            raise

    def get_execution_state(
        self, sid: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Retrieve the persisted plan and state trace for a session.

        Args:
            sid: Session identifier.

        Returns:
            Tuple of (plan, state) or None if absent.
        """
        try:
            row = self._conn.execute(
                "SELECT plan_json, state_json FROM execution_plans WHERE sid = ?",
                (sid,),
            ).fetchone()
        except sqlite3.Error as exc:
            logger.error("get_execution_state failed for sid=%s: %s", sid, exc)
            return None

        if row is None:
            return None

        try:
            plan = json.loads(row["plan_json"])
            state = json.loads(row["state_json"])
            return plan, state
        except json.JSONDecodeError:
            logger.warning("Corrupt execution plan/state JSON for sid=%s", sid)
            return None

    def get_all_execution_states(self) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        """Retrieve all persisted plans and state traces.

        Returns:
            List of Tuples of (sid, plan, state).
        """
        try:
            rows = self._conn.execute(
                "SELECT sid, plan_json, state_json FROM execution_plans"
            ).fetchall()
        except sqlite3.Error as exc:
            logger.error("get_all_execution_states failed: %s", exc)
            return []

        results = []
        for row in rows:
            try:
                plan = json.loads(row["plan_json"])
                state = json.loads(row["state_json"])
                results.append((row["sid"], plan, state))
            except json.JSONDecodeError:
                logger.warning("Corrupt execution plan/state JSON for sid=%s", row["sid"])
        return results

    # ── dataset versions & links ─────────────────────────────────────

    def get_dataset_version(self, fingerprint: str) -> int | None:
        """Get the stored version number for a dataset fingerprint."""
        try:
            row = self._conn.execute(
                "SELECT version FROM dataset_versions WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            return row["version"] if row else None
        except sqlite3.Error as exc:
            logger.error("get_dataset_version failed: %s", exc)
            return None

    def register_dataset_version(self, fingerprint: str, filename: str) -> int:
        """Register a dataset fingerprint and return its version number.

        Increments the version automatically if the filename has been seen before
        but with a different structural fingerprint.
        """
        try:
            # Check if fingerprint is already registered
            existing_ver = self.get_dataset_version(fingerprint)
            if existing_ver is not None:
                return existing_ver

            # Find maximum version for this filename
            row = self._conn.execute(
                "SELECT MAX(version) as max_v FROM dataset_versions WHERE filename = ?",
                (filename,),
            ).fetchone()
            
            max_v = row["max_v"] if row and row["max_v"] is not None else 0
            new_version = max_v + 1

            # Insert version mapping
            self._conn.execute(
                """
                INSERT INTO dataset_versions (fingerprint, filename, version)
                VALUES (?, ?, ?)
                """,
                (fingerprint, filename, new_version),
            )
            self._conn.commit()
            logger.info("Dataset version registered: file=%s, ver=%d", filename, new_version)

            # If this is a version increment (v2+), link it automatically to the previous version (v-1)
            if new_version > 1:
                prev_row = self._conn.execute(
                    "SELECT fingerprint FROM dataset_versions WHERE filename = ? AND version = ?",
                    (filename, new_version - 1),
                ).fetchone()
                if prev_row:
                    prev_fingerprint = prev_row["fingerprint"]
                    self.add_dataset_link(prev_fingerprint, fingerprint, "version_update", 1.0)

            return new_version
        except sqlite3.Error as exc:
            logger.error("register_dataset_version failed: %s", exc)
            raise

    def add_dataset_link(
        self, source_fingerprint: str, target_fingerprint: str, relationship_type: str, similarity_score: float
    ) -> None:
        """Insert a directional relationship link between two dataset structures."""
        try:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO dataset_links
                    (source_fingerprint, target_fingerprint, relationship_type, similarity_score)
                VALUES (?, ?, ?, ?)
                """,
                (source_fingerprint, target_fingerprint, relationship_type, similarity_score),
            )
            self._conn.commit()
            logger.info(
                "Dataset link added: source=%s ➔ target=%s type=%s",
                source_fingerprint[:8], target_fingerprint[:8], relationship_type
            )
        except sqlite3.Error as exc:
            logger.error("add_dataset_link failed: %s", exc)
            raise

    def get_dataset_links(self, fingerprint: str) -> list[dict[str, Any]]:
        """Retrieve all dataset relationships linking to or from this fingerprint."""
        try:
            rows = self._conn.execute(
                """
                SELECT source_fingerprint, target_fingerprint, relationship_type, similarity_score, created_at
                FROM dataset_links
                WHERE source_fingerprint = ? OR target_fingerprint = ?
                """,
                (fingerprint, fingerprint),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.error("get_dataset_links failed: %s", exc)
            return []
