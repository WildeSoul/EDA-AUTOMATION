import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tempfile
import pytest
import pandas as pd
import numpy as np

from data.session_store import SessionStore

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_session_store_crud(temp_db):
    store = SessionStore(db_path=temp_db)
    
    sid = "test-session-123"
    filename = "test.csv"
    row_count = 100
    col_count = 5
    intelligence = {"quality_score": 85.0, "quality_grade": "B", "dataset_type": "sales"}
    
    # Create session
    created_id = store.create_session(sid, filename, row_count, col_count, intelligence)
    assert created_id == sid
    
    # Get session
    session = store.get_session(sid)
    assert session is not None
    assert session["filename"] == filename
    assert session["row_count"] == row_count
    assert session["col_count"] == col_count
    assert session["quality_score"] == 85.0
    assert session["quality_grade"] == "B"
    assert session["intelligence"]["dataset_type"] == "sales"
    
    # Update intelligence
    new_intel = {"quality_score": 95.0, "quality_grade": "A", "dataset_type": "churn"}
    store.update_intelligence(sid, new_intel)
    
    session = store.get_session(sid)
    assert session["quality_score"] == 95.0
    assert session["quality_grade"] == "A"
    assert session["intelligence"]["dataset_type"] == "churn"
    
    # List sessions
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["sid"] == sid
    
    # Delete session
    store.delete_session(sid)
    assert store.get_session(sid) is None
    
    store.close()

def test_session_store_dataframe(temp_db):
    store = SessionStore(db_path=temp_db)
    sid = "df-session"
    store.create_session(sid, "data.csv", 5, 2)
    
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })
    
    store.save_dataframe(sid, df)
    loaded_df = store.load_dataframe(sid)
    
    assert loaded_df is not None
    assert loaded_df.shape == (3, 2)
    assert list(loaded_df["a"]) == [1, 2, 3]
    assert list(loaded_df["b"]) == ["x", "y", "z"]
    
    store.close()

def test_session_store_chat(temp_db):
    store = SessionStore(db_path=temp_db)
    sid = "chat-session"
    store.create_session(sid, "data.csv", 5, 2)
    
    store.save_chat(sid, "user", "Hello agent")
    store.save_chat(sid, "assistant", "Hello user", agent_name="insight")
    
    history = store.get_chat_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello agent"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hello user"
    assert history[1]["agent_name"] == "insight"
    
    store.close()

def test_session_store_agent_results(temp_db):
    store = SessionStore(db_path=temp_db)
    sid = "agent-session"
    store.create_session(sid, "data.csv", 5, 2)
    
    result_data = {"score": 99}
    store.save_agent_result(sid, "quality", result_data)
    
    retrieved = store.get_agent_result(sid, "quality")
    assert retrieved == result_data
    
    all_results = store.get_all_agent_results(sid)
    assert "quality" in all_results
    assert all_results["quality"] == result_data
    
    store.close()
