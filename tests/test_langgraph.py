import time
import pytest
import pandas as pd
from datetime import datetime, timezone

from core.agent_engine import AgentState, ExecutionEngine
from core.langgraph_engine import build_langgraph_engine
from agents import AGENT_REGISTRY
from data.checkpointer import SQLiteCheckpointer
from data.session_store import SessionStore


# ═══════════════════════════════════════════════
# Mock Specialists for Testing
# ═══════════════════════════════════════════════

class MockSchemaAgent:
    name = "schema"
    model = "mock-model"
    def run_cached(self, intelligence, **kwargs):
        return {"agent": "schema", "success": True, "result": {"schema_ok": True}}


class MockQualityAgent:
    name = "quality"
    model = "mock-model"
    def run_cached(self, intelligence, **kwargs):
        return {"agent": "quality", "success": True, "result": {"quality_ok": True}}


class MockHypothesisAgent:
    name = "hypothesis"
    model = "mock-model"
    def run_cached(self, intelligence, **kwargs):
        return {"agent": "hypothesis", "success": True, "result": {"hypothesis_ok": True}}


class MockEDAAgent:
    name = "eda"
    model = "mock-model"
    def run_cached(self, intelligence, **kwargs):
        return {"agent": "eda", "success": True, "result": {"eda_ok": True}}


class MockInsightAgent:
    name = "insight"
    model = "mock-model"
    def run_cached(self, intelligence, **kwargs):
        return {"agent": "insight", "success": True, "result": {"markdown_report": "Plan complete"}}


class MockFailingAgent:
    name = "schema"
    model = "mock-model"
    call_count = 0
    def run_cached(self, intelligence, **kwargs):
        MockFailingAgent.call_count += 1
        if MockFailingAgent.call_count < 3:
            return {"agent": "schema", "success": False, "error": "Temporary connection issue"}
        return {"agent": "schema", "success": True, "result": {"schema_ok": True}}


# ═══════════════════════════════════════════════
# Setup / Teardown Fixtures
# ═══════════════════════════════════════════════

@pytest.fixture
def mock_registry(monkeypatch):
    monkeypatch.setitem(AGENT_REGISTRY, "schema", MockSchemaAgent)
    monkeypatch.setitem(AGENT_REGISTRY, "quality", MockQualityAgent)
    monkeypatch.setitem(AGENT_REGISTRY, "hypothesis", MockHypothesisAgent)
    monkeypatch.setitem(AGENT_REGISTRY, "eda", MockEDAAgent)
    monkeypatch.setitem(AGENT_REGISTRY, "insight", MockInsightAgent)


@pytest.fixture
def test_db():
    # In-memory SQLite DB
    store = SessionStore(":memory:")
    yield store
    store.close()


# ═══════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════

def test_conditional_memory_routing_skipped(mock_registry, test_db):
    """If dataset is simple (v1, domain unknown), memory node is skipped."""
    engine = ExecutionEngine(test_db)
    
    plan = {
        "steps": [
            {"id": "step_1", "agent": "schema", "depends_on": [], "input_mapping": {}}
        ]
    }
    
    state = AgentState()
    state["sid"] = "session_skip_memory"
    # version=1, domain=unknown -> Memory Router should skip
    state["intelligence"] = {"dataset_fingerprint": "f1", "version": 1, "domain": "unknown"}
    
    test_db.create_session(state["sid"], "test.csv", 100, 5, state["intelligence"])
    df = pd.DataFrame({"a": [1, 2]})
    
    final_state = engine.execute_plan(plan, df, state)
    
    assert final_state["status"] == "completed"
    # Check that "memory" telemetry is empty (skipped)
    assert "memory" not in final_state["node_telemetry"]
    assert "planner" not in final_state["node_telemetry"]  # skipped plan generation since plan is supplied


def test_conditional_memory_routing_triggered(mock_registry, test_db, monkeypatch):
    """If dataset is v2+ or domain is known, memory node runs."""
    engine = ExecutionEngine(test_db)
    
    # Mock MemoryAgent to avoid DB dependency or network lookups in Chroma
    class MockMemoryAgent:
        name = "memory"
        model = "mock-model"
        def run(self, intelligence, **kwargs):
            return {
                "agent": "memory",
                "success": True,
                "result": {"similar_datasets": [], "injected_memories": {}}
            }
    monkeypatch.setitem(AGENT_REGISTRY, "memory", MockMemoryAgent)

    plan = {
        "steps": [
            {"id": "step_1", "agent": "schema", "depends_on": [], "input_mapping": {}}
        ]
    }
    
    state = AgentState()
    state["sid"] = "session_run_memory"
    # version=2 -> Memory Router should route to memory node
    state["intelligence"] = {"dataset_fingerprint": "f2", "version": 2, "domain": "unknown"}
    
    test_db.create_session(state["sid"], "test.csv", 100, 5, state["intelligence"])
    df = pd.DataFrame({"a": [1, 2]})
    
    final_state = engine.execute_plan(plan, df, state)
    
    assert final_state["status"] == "completed"
    # Verify that memory node was executed
    assert "memory" in final_state["node_telemetry"]
    assert final_state["node_telemetry"]["memory"]["attempts"] == 1


def test_parallel_execution_timing(mock_registry, test_db, monkeypatch):
    """Independent steps are batch-executed in parallel."""
    engine = ExecutionEngine(test_db)

    class SleepingSchemaAgent:
        name = "schema"
        model = "mock-model"
        def run_cached(self, intelligence, **kwargs):
            time.sleep(0.5)
            return {"agent": "schema", "success": True, "result": {}}

    class SleepingQualityAgent:
        name = "quality"
        model = "mock-model"
        def run_cached(self, intelligence, **kwargs):
            time.sleep(0.5)
            return {"agent": "quality", "success": True, "result": {}}

    monkeypatch.setitem(AGENT_REGISTRY, "schema", SleepingSchemaAgent)
    monkeypatch.setitem(AGENT_REGISTRY, "quality", SleepingQualityAgent)

    plan = {
        "steps": [
            # Two independent steps that can run concurrently
            {"id": "step_1", "agent": "schema", "depends_on": [], "input_mapping": {}},
            {"id": "step_2", "agent": "quality", "depends_on": [], "input_mapping": {}}
        ]
    }

    state = AgentState()
    state["sid"] = "session_parallel"
    state["intelligence"] = {"dataset_fingerprint": "f3", "version": 1, "domain": "unknown"}
    test_db.create_session(state["sid"], "test.csv", 100, 5, state["intelligence"])
    df = pd.DataFrame({"a": [1, 2]})

    start = time.time()
    final_state = engine.execute_plan(plan, df, state)
    elapsed = time.time() - start

    assert final_state["status"] == "completed"
    # If sequential, it takes >= 1.0s. If parallel, it takes ~0.5s - 0.7s.
    assert elapsed < 0.95


def test_retry_policy_failing_node(test_db, monkeypatch):
    """Nodes failing temporarily are retried according to policy."""
    engine = ExecutionEngine(test_db)

    # Reset call count
    MockFailingAgent.call_count = 0
    monkeypatch.setitem(AGENT_REGISTRY, "schema", MockFailingAgent)

    plan = {
        "steps": [
            {"id": "step_1", "agent": "schema", "depends_on": [], "input_mapping": {}}
        ]
    }

    state = AgentState()
    state["sid"] = "session_retries"
    state["intelligence"] = {"dataset_fingerprint": "f4", "version": 1, "domain": "unknown"}
    test_db.create_session(state["sid"], "test.csv", 100, 5, state["intelligence"])
    df = pd.DataFrame({"a": [1, 2]})

    final_state = engine.execute_plan(plan, df, state)

    assert final_state["status"] == "completed"
    # Check telemetry tracks all 3 attempts
    assert final_state["node_telemetry"]["step_1"]["attempts"] == 3
    assert MockFailingAgent.call_count == 3


def test_human_in_the_loop_approval_and_restore_time(mock_registry, test_db):
    """Execution halts on high duration and resumes cleanly when approved (< 2s)."""
    engine = ExecutionEngine(test_db)

    # 1. Plan with estimated_time_sec > 30 to trigger approval halt
    plan = {
        "estimates": {
            "estimated_steps": 2,
            "estimated_time_sec": 45.0,  # > 30s triggers approval node
            "estimated_llm_calls": 2
        },
        "steps": [
            {"id": "step_1", "agent": "schema", "depends_on": [], "input_mapping": {}},
            {"id": "step_2", "agent": "insight", "depends_on": ["step_1"], "input_mapping": {}}
        ]
    }

    state = AgentState()
    state["sid"] = "session_hitl"
    state["intelligence"] = {"dataset_fingerprint": "f5", "version": 1, "domain": "unknown"}
    test_db.create_session(state["sid"], "test.csv", 100, 5, state["intelligence"])
    df = pd.DataFrame({"a": [1, 2]})

    # Run execution -> should halt at approval gate
    final_state = engine.execute_plan(plan, df, state)

    assert final_state["status"] == "awaiting_approval"
    assert final_state["requires_approval"] is True
    assert final_state["approval_granted"] is False
    assert "step_1" not in final_state["outputs"]  # Has not run executor node yet!

    # 2. Approve and resume the plan
    # Measure restore latency
    start_restore = time.time()
    resumed_state = engine.resume_execution(
        sid=state["sid"],
        adjusted_plan=None,
        df=df
    )
    restore_latency = time.time() - start_restore

    # Verify checkpoint restore time benchmark is < 2.0s
    assert restore_latency < 2.0, f"Checkpoint restore time was {restore_latency}s (target < 2s)"

    # Verify that execution has successfully resumed and completed
    assert resumed_state["status"] == "completed"
    assert resumed_state["requires_approval"] is False
    assert resumed_state["approval_granted"] is True
    assert "step_1" in resumed_state["outputs"]
    assert "step_2" in resumed_state["outputs"]
    assert len(resumed_state["resume_events"]) == 1
    assert resumed_state["resume_events"][0]["action"] == "approved"
