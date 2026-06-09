"""
Unit tests for Phase 3 — Memory & Knowledge Layer.

Validates version increments, dataset links, ChromaDB persistence,
Score Fusion ranking, memory confidence gates, and explanation rationale.
"""

from __future__ import annotations
import os
import shutil
import tempfile
import pytest
from datetime import datetime, timezone, timedelta

import pandas as pd

from config.settings import MEMORY_INJECTION_THRESHOLD
from data.session_store import SessionStore
from data.memory_store import MemoryStore
from agents.memory_agent import MemoryAgent


@pytest.fixture(autouse=True)
def configure_test_environment(temp_db_dir):
    """Ensure all MemoryStore initialisations default to the temp test directory."""
    chroma_dir = os.path.join(temp_db_dir, "chroma_db")
    os.environ["TESTING_CHROMA_DIR"] = chroma_dir
    yield
    if "TESTING_CHROMA_DIR" in os.environ:
        del os.environ["TESTING_CHROMA_DIR"]


@pytest.fixture
def temp_db_dir():
    """Fixture to provide a clean temporary directory for SQLite and ChromaDB."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


@pytest.fixture
def session_store(temp_db_dir):
    """Fixture providing a clean SQLite SessionStore instance."""
    db_path = os.path.join(temp_db_dir, "test_sessions.db")
    store = SessionStore(db_path=db_path)
    yield store
    store.close()


@pytest.fixture
def memory_store(temp_db_dir):
    """Fixture providing a clean ChromaDB MemoryStore instance."""
    chroma_dir = os.path.join(temp_db_dir, "chroma_db")
    store = MemoryStore(db_dir=chroma_dir)
    yield store
    store.close()


def test_sqlite_versioning(session_store):
    """Verify that dataset registering increments version numbers correctly."""
    f1 = "fingerprint_1"
    f2 = "fingerprint_2"
    filename = "sales.csv"

    # Register f1 for the first time
    v1 = session_store.register_dataset_version(f1, filename)
    assert v1 == 1

    # Registering f1 again returns the same version
    assert session_store.get_dataset_version(f1) == 1
    assert session_store.register_dataset_version(f1, filename) == 1

    # Registering f2 (modified structure) for the same file increments version
    v2 = session_store.register_dataset_version(f2, filename)
    assert v2 == 2

    # Verify relationships are created automatically for version updates
    links = session_store.get_dataset_links(f2)
    assert len(links) == 1
    assert links[0]["relationship_type"] == "version_update"
    assert links[0]["source_fingerprint"] == f1
    assert links[0]["target_fingerprint"] == f2


def test_sqlite_dataset_linkage(session_store):
    """Verify manual linkage of dataset structures."""
    f1 = "fingerprint_a"
    f2 = "fingerprint_b"
    
    session_store.add_dataset_link(f1, f2, "similar_schema", 0.85)
    links = session_store.get_dataset_links(f1)
    
    assert len(links) == 1
    assert links[0]["source_fingerprint"] == f1
    assert links[0]["target_fingerprint"] == f2
    assert links[0]["relationship_type"] == "similar_schema"
    assert links[0]["similarity_score"] == 0.85


def test_chromadb_basic_crud(memory_store):
    """Verify that datasets, insights, and cleanups are correctly persisted and fetched."""
    sid = "session-123"
    fingerprint = "test-fingerprint"
    filename = "customers.csv"
    intel = {
        "filename": filename,
        "dataset_type": "customer_churn",
        "domain": "telecom",
        "row_count": 100,
        "quality_score": 0.90,
        "columns": {"Age": {"semantic_type": "numeric"}, "Churn": {"semantic_type": "binary_target"}}
    }

    # Add profile
    memory_store.add_dataset_profile(sid, fingerprint, filename, intel, version=1)

    # Add insight
    memory_store.add_insight(sid, fingerprint, "insight", "High Churn in Age 18-25", "Description text", 0.95)

    # Add hypothesis
    memory_store.add_hypothesis(sid, fingerprint, "Younger users churn more", "validated", 0.88, 0.90)

    # Add cleanup
    memory_store.add_cleaning_action(sid, fingerprint, "Age", "imputation", "Imputed missing age with median")

    # Retrieve memory
    mems = memory_store.retrieve_dataset_memory(fingerprint)
    assert len(mems["insights"]) == 1
    assert mems["insights"][0]["title"] == "High Churn in Age 18-25"
    assert mems["insights"][0]["confidence"] == 0.95

    assert len(mems["hypotheses"]) == 1
    assert mems["hypotheses"][0]["hypothesis"] == "Younger users churn more"
    assert mems["hypotheses"][0]["status"] == "validated"

    assert len(mems["cleaning_actions"]) == 1
    assert mems["cleaning_actions"][0]["column"] == "Age"
    assert mems["cleaning_actions"][0]["action_type"] == "imputation"


def test_score_fusion_ranking(memory_store):
    """Verify the Score Fusion ranking weights (similarity, confidence, reuse, recency)."""
    # Insert two datasets: one recent and one old
    sid1 = "session-new"
    fp1 = "fp-new"
    intel1 = {
        "filename": "telecom_new.csv",
        "dataset_type": "customer_churn",
        "domain": "telecom",
        "quality_score": 0.80, # Confidence weight
        "columns": {"Age": {}, "Churn": {}}
    }
    memory_store.add_dataset_profile(sid1, fp1, "telecom_new.csv", intel1, version=1)
    
    # Overwrite created_at to simulate time decay (e.g. 30 days ago)
    res = memory_store.col_datasets.get(ids=[fp1])
    meta = res["metadatas"][0]
    old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    meta["created_at"] = old_time
    # Metadata-only update to avoid dimension re-embedding errors
    memory_store.col_datasets.update(ids=[fp1], metadatas=[meta])

    # Search similar using identical profile structure
    matches = memory_store.search_similar_datasets(intel1, limit=5)
    assert len(matches) == 1
    
    # Check recency factor calculation: e^(-0.05 * 30) = e^(-1.5) = 0.223
    # Similarity = 1.0 -> Weight 0.5 * 1.0 = 0.50
    # Confidence = 0.8 -> Weight 0.2 * 0.8 = 0.16
    # Reuse = 0 -> Weight 0.2 * 0 = 0.00
    # Recency = 0.223 -> Weight 0.1 * 0.223 = 0.0223
    # Total score: 0.50 + 0.16 + 0.0223 = 0.68
    rank_score = matches[0]["rank_score"]
    assert abs(rank_score - 0.68) < 0.02


def test_memory_confidence_gate(memory_store):
    """Verify that MemoryAgent filters out memories under the confidence threshold."""
    # Add a dataset with low baseline score (e.g. 300 days old, low quality)
    sid = "session-low"
    fp = "fp-low"
    intel = {
        "filename": "telecom_stale.csv",
        "dataset_type": "churn",
        "domain": "telecom",
        "quality_score": 0.20,
        "columns": {"A": {}, "B": {}}
    }
    memory_store.add_dataset_profile(sid, fp, "telecom_stale.csv", intel, version=1)
    
    res = memory_store.col_datasets.get(ids=[fp])
    meta = res["metadatas"][0]
    meta["created_at"] = (datetime.now(timezone.utc) - timedelta(days=300)).isoformat()
    memory_store.col_datasets.update(ids=[fp], metadatas=[meta])
    
    # Store sub-memories
    memory_store.add_insight(sid, fp, "insight", "Low correlation", "Low correlation values", 0.50)

    # Invoke memory agent with default threshold (0.65)
    agent = MemoryAgent()
    try:
        res_agent = agent.run(intel)
        # Check that similar dataset was ignored because rank score fell below threshold
        assert len(res_agent["result"]["similar_datasets"]) == 0
        assert len(res_agent["result"]["injected_memories"]["insights"]) == 0
    finally:
        agent._store.close()


def test_explainability_reasons(memory_store):
    """Verify explainability overlap checker compiles descriptive reason lists."""
    sid = "session-abc"
    fp = "fp-abc"
    intel = {
        "filename": "old_telecom.csv",
        "dataset_type": "customer_churn",
        "domain": "telecom",
        "quality_score": 1.0,
        "columns": {"Age": {}, "Churn": {}, "Income": {}}
    }
    memory_store.add_dataset_profile(sid, fp, "old_telecom.csv", intel, version=1)

    # Search with identical metadata to trigger same domain and type overlap checks
    matches = memory_store.search_similar_datasets(intel)
    assert len(matches) == 1
    
    reasons = matches[0]["reasons"]
    assert "same domain: telecom" in reasons
    assert "same target type: customer_churn" in reasons
    assert any("similar feature structure" in r for r in reasons)


def test_prompt_isolation_and_contamination_header(memory_store):
    """Verify that PlannerAgent prompt includes the advisory isolates correctly."""
    # Add a valid high-score memory to ensure it gets retrieved
    sid = "session-ok"
    fp = "fp-ok"
    intel = {
        "filename": "telecom_good.csv",
        "dataset_type": "customer_churn",
        "domain": "telecom",
        "quality_score": 1.0,
        "columns": {"Age": {}, "Churn": {}}
    }
    memory_store.add_dataset_profile(sid, fp, "telecom_good.csv", intel, version=1)
    memory_store.add_insight(sid, fp, "insight", "Key Pattern", "Details of pattern", 0.90)

    # Run planner agent
    from agents.planner_agent import PlannerAgent
    planner = PlannerAgent()
    
    # We can capture the prompt by patching query_ollama
    captured_prompt = ""
    def mock_query(model, prompt, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt
        return json_dict_mock_plan()
    
    def json_dict_mock_plan():
        import json
        return json.dumps({
            "rationale": "Mock planning pass.",
            "estimates": {"estimated_steps": 1, "estimated_time_sec": 1.0, "estimated_llm_calls": 1},
            "steps": [{"id": "step_1", "agent": "schema", "depends_on": [], "description": "Mock pass", "input_mapping": {}}]
        })

    from unittest.mock import patch
    with patch("agents.planner_agent.query_ollama", side_effect=mock_query):
        planner.run(intel)
        
        # Assert advisory context sections are isolated and constraints exist
        assert "## ADVISORY HISTORY (MEMORY)" in captured_prompt
        assert "Historical knowledge is advisory only." in captured_prompt
        assert "Strict Separators" in captured_prompt
