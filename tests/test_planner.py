import pytest
import pandas as pd
from core.plan_validator import validate_plan
from core.agent_engine import AgentState, ExecutionEngine
from agents import AGENT_REGISTRY

# ═══════════════════════════════════════════════
# Plan Validation Tests
# ═══════════════════════════════════════════════

def test_validation_clean_plan():
    plan = {
        "steps": [
            {
                "id": "step_1",
                "agent": "schema",
                "depends_on": [],
                "description": "Read schema",
                "input_mapping": {}
            },
            {
                "id": "step_2",
                "agent": "quality",
                "depends_on": ["step_1"],
                "description": "Read quality",
                "input_mapping": {
                    "schema_result": "step_1"
                }
            }
        ]
    }
    errors = validate_plan(plan)
    assert len(errors) == 0

def test_validation_cycle_detection():
    # Simple self-loop cycle
    plan = {
        "steps": [
            {
                "id": "step_1",
                "agent": "schema",
                "depends_on": ["step_1"],
                "description": "Self cycle"
            }
        ]
    }
    errors = validate_plan(plan)
    assert any("cycle" in err.lower() for err in errors)

    # Multi-step cycle
    plan_cycle = {
        "steps": [
            {
                "id": "step_1",
                "agent": "schema",
                "depends_on": ["step_2"]
            },
            {
                "id": "step_2",
                "agent": "quality",
                "depends_on": ["step_1"]
            }
        ]
    }
    errors_cycle = validate_plan(plan_cycle)
    assert any("cycle" in err.lower() for err in errors_cycle)

def test_validation_duplicate_ids():
    plan = {
        "steps": [
            {
                "id": "step_1",
                "agent": "schema",
                "depends_on": []
            },
            {
                "id": "step_1",
                "agent": "quality",
                "depends_on": []
            }
        ]
    }
    errors = validate_plan(plan)
    assert any("duplicate step id" in err.lower() for err in errors)

def test_validation_invalid_agent():
    plan = {
        "steps": [
            {
                "id": "step_1",
                "agent": "non_existent_agent_name",
                "depends_on": []
            }
        ]
    }
    errors = validate_plan(plan)
    assert any("invalid agent name" in err.lower() for err in errors)

def test_validation_missing_dependency():
    plan = {
        "steps": [
            {
                "id": "step_1",
                "agent": "schema",
                "depends_on": ["missing_step"]
            }
        ]
    }
    errors = validate_plan(plan)
    assert any("depends on non-existent step" in err.lower() for err in errors)

def test_validation_invalid_input_mapping():
    plan = {
        "steps": [
            {
                "id": "step_1",
                "agent": "schema",
                "depends_on": []
            },
            {
                "id": "step_2",
                "agent": "quality",
                "depends_on": [],
                "description": "No dependencies",
                "input_mapping": {
                    "schema_result": "step_1"
                }
            }
        ]
    }
    errors = validate_plan(plan)
    assert any("not listed in its 'depends_on'" in err.lower() for err in errors)


# ═══════════════════════════════════════════════
# Execution Engine Tests
# ═══════════════════════════════════════════════

def test_engine_topological_sort():
    engine = ExecutionEngine()
    steps = [
        {"id": "step_3", "agent": "hypothesis", "depends_on": ["step_2"]},
        {"id": "step_1", "agent": "schema", "depends_on": []},
        {"id": "step_2", "agent": "quality", "depends_on": ["step_1"]}
    ]
    sorted_steps = engine._topological_sort(steps)
    assert sorted_steps[0]["id"] == "step_1"
    assert sorted_steps[1]["id"] == "step_2"
    assert sorted_steps[2]["id"] == "step_3"

def test_engine_execution_success(monkeypatch):
    engine = ExecutionEngine()
    
    # Mocking agents to avoid real LLM calls
    class MockSchemaAgent:
        model = "mock-model"
        def run_cached(self, intelligence, **kwargs):
            return {"agent": "schema", "success": True, "result": {"schema_keys": True}}
            
    class MockQualityAgent:
        model = "mock-model"
        def run_cached(self, intelligence, **kwargs):
            # Verify input mapping mapping worked
            assert "schema_result" in kwargs
            assert kwargs["schema_result"].get("result", {}).get("schema_keys") is True
            return {"agent": "quality", "success": True, "result": {"quality_score": 90}}

    class MockInsightAgent:
        model = "mock-model"
        def run_cached(self, intelligence, **kwargs):
            return {"agent": "insight", "success": True, "result": {"markdown_report": "All clean"}}

    monkeypatch.setitem(AGENT_REGISTRY, "schema", MockSchemaAgent)
    monkeypatch.setitem(AGENT_REGISTRY, "quality", MockQualityAgent)
    monkeypatch.setitem(AGENT_REGISTRY, "insight", MockInsightAgent)

    plan = {
        "steps": [
            {
                "id": "step_1",
                "agent": "schema",
                "depends_on": [],
                "input_mapping": {}
            },
            {
                "id": "step_2",
                "agent": "quality",
                "depends_on": ["step_1"],
                "input_mapping": {
                    "schema_result": "step_1"
                }
            },
            {
                "id": "step_3",
                "agent": "insight",
                "depends_on": ["step_2"],
                "input_mapping": {
                    "quality_result": "step_2"
                }
            }
        ]
    }

    df = pd.DataFrame({"a": [1, 2]})
    state = AgentState()
    state["intelligence"] = {"dataset_type": "test"}

    final_state = engine.execute_plan(plan, df, state)

    assert final_state["status"] == "completed"
    assert "step_1" in final_state["outputs"]
    assert "step_2" in final_state["outputs"]
    assert "step_3" in final_state["outputs"]
    assert len(final_state["trace"]) == 3
    assert all(t["status"] == "completed" for t in final_state["trace"])

def test_engine_execution_failure_and_recovery(monkeypatch):
    engine = ExecutionEngine()

    class MockSchemaAgent:
        model = "mock-model"
        def run_cached(self, intelligence, **kwargs):
            return {"agent": "schema", "success": False, "error": "Schema crashed"}

    class MockQualityAgent:
        model = "mock-model"
        def run_cached(self, intelligence, **kwargs):
            return {"agent": "quality", "success": True}

    monkeypatch.setitem(AGENT_REGISTRY, "schema", MockSchemaAgent)
    monkeypatch.setitem(AGENT_REGISTRY, "quality", MockQualityAgent)

    plan = {
        "steps": [
            {
                "id": "step_1",
                "agent": "schema",
                "depends_on": [],
                "input_mapping": {}
            },
            {
                "id": "step_2",
                "agent": "quality",
                "depends_on": ["step_1"],
                "input_mapping": {
                    "schema_result": "step_1"
                }
            }
        ]
    }

    df = pd.DataFrame({"a": [1, 2]})
    state = AgentState()
    state["intelligence"] = {"dataset_type": "test"}

    final_state = engine.execute_plan(plan, df, state)

    assert final_state["status"] == "failed"
    # step_1 fails, step_2 depends on step_1 and is skipped/failed
    assert final_state["trace"][0]["status"] == "failed"
    assert final_state["trace"][1]["status"] == "failed"
    
    # Recovery insight runs
    assert "recovery_insight" in final_state["outputs"]
    assert any(t["step_id"] == "recovery_insight" for t in final_state["trace"])

def test_execution_state_persistence():
    import tempfile
    import os
    from data.session_store import SessionStore

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    try:
        store = SessionStore(db_path=db_path)
        sid = "persist-session"
        store.create_session(sid, "test.csv", 10, 2)
        
        plan = {
            "rationale": "Persistence test plan",
            "estimates": {"estimated_steps": 1, "estimated_time_sec": 1.5, "estimated_llm_calls": 1},
            "steps": [{"id": "step_1", "agent": "schema"}]
        }
        
        state = {
            "status": "completed",
            "trace": [{"step_id": "step_1", "status": "completed"}],
            "outputs": {"step_1": {"success": True}}
        }
        
        # Save
        store.save_execution_state(sid, plan, state)
        
        # Load
        loaded = store.get_execution_state(sid)
        assert loaded is not None
        
        loaded_plan, loaded_state = loaded
        assert loaded_plan["rationale"] == "Persistence test plan"
        assert loaded_state["status"] == "completed"
        assert loaded_state["outputs"]["step_1"]["success"] is True
        
        store.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
