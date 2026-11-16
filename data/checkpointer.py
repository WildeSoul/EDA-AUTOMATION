"""
SQLite-backed checkpointer for LangGraph, integrating with the platform's sessions database.
"""

from __future__ import annotations

import logging
import random
import sqlite3
import threading
from typing import Any, Iterator, Sequence, Optional, AsyncIterator, Tuple

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    RunnableConfig,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


class SQLiteCheckpointer(BaseCheckpointSaver):
    """A thread-safe SQLite checkpoint saver for LangGraph."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
        serde: Optional[Any] = None,
    ) -> None:
        super().__init__(serde=serde or JsonPlusSerializer())
        self.lock = threading.Lock()
        self._owned_conn = False

        if conn is not None:
            self.conn = conn
        else:
            from config.settings import SESSION_DB_PATH
            path = db_path or SESSION_DB_PATH
            self.conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            try:
                self.conn.execute("PRAGMA journal_mode = WAL")
                self.conn.execute("PRAGMA synchronous = NORMAL")
            except sqlite3.Error as pragma_err:
                logger.debug("Failed to set WAL/synchronous pragmas: %s", pragma_err)
            self._owned_conn = True

        self._create_tables()

    def _create_tables(self) -> None:
        with self.lock:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_history (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                checkpoint_type TEXT NOT NULL,
                checkpoint_bin BLOB NOT NULL,
                metadata_type TEXT NOT NULL,
                metadata_bin BLOB NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
            """)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                type TEXT NOT NULL,
                blob BLOB NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
            );
            """)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value_bin BLOB NOT NULL,
                task_path TEXT,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
            """)
            self.conn.commit()

    def close(self) -> None:
        """Close the database connection if it was opened by this checkpointer."""
        with self.lock:
            if self._owned_conn and hasattr(self, "conn") and self.conn:
                try:
                    self.conn.close()
                    logger.info("SQLiteCheckpointer connection closed.")
                except Exception as exc:
                    logger.error("Error closing SQLiteCheckpointer connection: %s", exc)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self.lock:
            if checkpoint_id:
                row = self.conn.execute(
                    """
                    SELECT parent_checkpoint_id, checkpoint_type, checkpoint_bin, metadata_type, metadata_bin
                    FROM checkpoint_history
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id)
                ).fetchone()
            else:
                row = self.conn.execute(
                    """
                    SELECT checkpoint_id, parent_checkpoint_id, checkpoint_type, checkpoint_bin, metadata_type, metadata_bin
                    FROM checkpoint_history
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns)
                ).fetchone()
                if row:
                    checkpoint_id = row["checkpoint_id"]

            if not row:
                return None

            parent_checkpoint_id = row["parent_checkpoint_id"]
            checkpoint_type = row["checkpoint_type"]
            checkpoint_bin = row["checkpoint_bin"]
            metadata_type = row["metadata_type"]
            metadata_bin = row["metadata_bin"]

            # Load writes
            writes_rows = self.conn.execute(
                """
                SELECT task_id, idx, channel, value_type, value_bin, task_path
                FROM checkpoint_writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_ns, checkpoint_id)
            ).fetchall()

            # Load checkpoint & metadata
            checkpoint_ = self.serde.loads_typed((checkpoint_type, checkpoint_bin))
            metadata = self.serde.loads_typed((metadata_type, metadata_bin))

            # Load channel values (blobs)
            channel_values = {}
            if checkpoint_ and "channel_versions" in checkpoint_:
                for channel, version in checkpoint_["channel_versions"].items():
                    blob_row = self.conn.execute(
                        """
                        SELECT type, blob
                        FROM checkpoint_blobs
                        WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?
                        """,
                        (thread_id, checkpoint_ns, channel, version)
                    ).fetchone()
                    if blob_row and blob_row["type"] != "empty":
                        channel_values[channel] = self.serde.loads_typed((blob_row["type"], blob_row["blob"]))

            pending_writes = [
                (
                    w["task_id"],
                    w["channel"],
                    self.serde.loads_typed((w["value_type"], w["value_bin"]))
                )
                for w in writes_rows
            ]

            parent_config = None
            if parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint={
                    **checkpoint_,
                    "channel_values": channel_values,
                },
                metadata=metadata,
                pending_writes=pending_writes,
                parent_config=parent_config,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        c = checkpoint.copy()
        values = c.pop("channel_values", {})

        with self.lock:
            # 1. Save new blobs
            for k, v in new_versions.items():
                if k in values:
                    t, d = self.serde.dumps_typed(values[k])
                else:
                    t, d = "empty", b""
                self.conn.execute(
                    """
                    INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id, checkpoint_ns, channel, version) DO UPDATE SET
                        type = excluded.type,
                        blob = excluded.blob
                    """,
                    (thread_id, checkpoint_ns, k, v, t, d)
                )

            # 2. Save checkpoint history
            c_type, c_bin = self.serde.dumps_typed(c)
            m_type, m_bin = self.serde.dumps_typed(metadata)

            self.conn.execute(
                """
                INSERT INTO checkpoint_history (
                    thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    checkpoint_type, checkpoint_bin, metadata_type, metadata_bin
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET
                    parent_checkpoint_id = excluded.parent_checkpoint_id,
                    checkpoint_type = excluded.checkpoint_type,
                    checkpoint_bin = excluded.checkpoint_bin,
                    metadata_type = excluded.metadata_type,
                    metadata_bin = excluded.metadata_bin
                """,
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, c_type, c_bin, m_type, m_bin)
            )
            self.conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self.lock:
            for idx, (channel, value) in enumerate(writes):
                t, d = self.serde.dumps_typed(value)
                self.conn.execute(
                    """
                    INSERT INTO checkpoint_writes (
                        thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, value_type, value_bin, task_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
                        channel = excluded.channel,
                        value_type = excluded.value_type,
                        value_bin = excluded.value_bin,
                        task_path = excluded.task_path
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, t, d, task_path)
                )
            self.conn.commit()

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"] if config else None
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "") if config else ""

        with self.lock:
            query = """
                SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint_type, checkpoint_bin, metadata_type, metadata_bin
                FROM checkpoint_history
            """
            params = []
            conditions = []

            if thread_id:
                conditions.append("thread_id = ?")
                params.append(thread_id)
            if checkpoint_ns:
                conditions.append("checkpoint_ns = ?")
                params.append(checkpoint_ns)

            if before:
                before_id = get_checkpoint_id(before)
                if before_id:
                    conditions.append("checkpoint_id < ?")
                    params.append(before_id)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY checkpoint_id DESC"

            rows = self.conn.execute(query, params).fetchall()

            count = 0
            for row in rows:
                if limit is not None and count >= limit:
                    break

                t_id = row["thread_id"]
                ns = row["checkpoint_ns"]
                c_id = row["checkpoint_id"]

                metadata = self.serde.loads_typed((row["metadata_type"], row["metadata_bin"]))

                # Filter by metadata if filter is provided
                if filter:
                    if not all(metadata.get(k) == v for k, v in filter.items()):
                        continue

                checkpoint_ = self.serde.loads_typed((row["checkpoint_type"], row["checkpoint_bin"]))

                # Load channel values
                channel_values = {}
                if checkpoint_ and "channel_versions" in checkpoint_:
                    for channel, version in checkpoint_["channel_versions"].items():
                        blob_row = self.conn.execute(
                            """
                            SELECT type, blob
                            FROM checkpoint_blobs
                            WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?
                            """,
                            (t_id, ns, channel, version)
                        ).fetchone()
                        if blob_row and blob_row["type"] != "empty":
                            channel_values[channel] = self.serde.loads_typed((blob_row["type"], blob_row["blob"]))

                # Load writes
                writes_rows = self.conn.execute(
                    """
                    SELECT task_id, idx, channel, value_type, value_bin, task_path
                    FROM checkpoint_writes
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (t_id, ns, c_id)
                ).fetchall()

                pending_writes = [
                    (
                        w["task_id"],
                        w["channel"],
                        self.serde.loads_typed((w["value_type"], w["value_bin"]))
                    )
                    for w in writes_rows
                ]

                parent_config = None
                if row["parent_checkpoint_id"]:
                    parent_config = {
                        "configurable": {
                            "thread_id": t_id,
                            "checkpoint_ns": ns,
                            "checkpoint_id": row["parent_checkpoint_id"],
                        }
                    }

                count += 1
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": t_id,
                            "checkpoint_ns": ns,
                            "checkpoint_id": c_id,
                        }
                    },
                    checkpoint={
                        **checkpoint_,
                        "channel_values": channel_values,
                    },
                    metadata=metadata,
                    pending_writes=pending_writes,
                    parent_config=parent_config,
                )

    # ── Async counterparts ──────────────────────────────────────────

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
