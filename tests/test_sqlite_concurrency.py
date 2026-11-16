"""
Concurrency tests for SessionStore and SQLiteCheckpointer under WAL mode.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
import threading
import pytest
import sqlite3
import pandas as pd
from data.session_store import SessionStore
from data.checkpointer import SQLiteCheckpointer


def test_sqlite_concurrency_no_locks():
    """Verify that multiple threads can read and write concurrently in WAL mode without database locked errors."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    try:
        # Initialise Store and Checkpointer on the same database
        store = SessionStore(db_path=db_path)
        checkpointer = SQLiteCheckpointer(db_path=db_path)
        
        errors = []
        
        # Reader thread target
        def reader_target(thread_idx):
            try:
                for i in range(15):
                    store.list_sessions(limit=5)
                    store.get_session(f"session-{thread_idx}-{i}")
            except Exception as e:
                errors.append(f"Reader {thread_idx} error: {e}")

        # Writer thread target
        def writer_target(thread_idx):
            try:
                for i in range(15):
                    sid = f"session-{thread_idx}-{i}"
                    store.create_session(sid, "test.csv", 100, 5)
                    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
                    store.save_dataframe(sid, df)
            except Exception as e:
                errors.append(f"Writer {thread_idx} error: {e}")

        threads = []
        # Create 10 reader and 10 writer threads
        for idx in range(10):
            threads.append(threading.Thread(target=reader_target, args=(idx,)))
            threads.append(threading.Thread(target=writer_target, args=(idx,)))

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Assert no lock errors
        assert len(errors) == 0, f"Concurrency errors: {errors}"
        
    finally:
        try:
            store.close()
        except:
            pass
        try:
            checkpointer.close()
        except:
            pass
        
        # Clean up files (including WAL files if present)
        for suffix in ["", "-wal", "-shm"]:
            path = db_path + suffix
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
