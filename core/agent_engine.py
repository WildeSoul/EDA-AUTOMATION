"""
Agent Execution Engine — Phase 4 LangGraph Migration.

Manages execution state, LangGraph StateGraph orchestration, SQLite checkpointing,
human approval gates, parallel batch execution, and automatic legacy engine recovery.
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from agents import AGENT_REGISTRY
# Ensure all agents are registered
import agents.schema_agent
import agents.quality_agent
import agents.hypothesis_agent
import agents.eda_agent
import agents.insight_agent
import agents.planner_agent

from core.plan_validator import validate_plan
from core.langgraph_engine import build_langgraph_engine, AgentState, execute_single_step
from langgraph.errors import NodeInterrupt
try:
    from langgraph.errors import GraphInterrupt
except ImportError:
    GraphInterrupt = NodeInterrupt

logger = logging.getLogger(__name__)


class LegacyExecutionEngine:
    """Fallback sequential loop execution engine. Kept for backward compatibility and crash recovery."""

    def __init__(self, session_store: Any = None) -> None:
        self.session_store = session_store

    def _topological_sort(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort plan steps topologically according to depends_on references."""
        step_map = {step["id"]: step for step in steps}
        graph = {step["id"]: step.get("depends_on", []) for step in steps}
        
        sorted_ids: list[str] = []
        visited = set()
        
        def visit(node_id: str):
            if node_id in visited:
                return
            for dep in graph.get(node_id, []):
                if dep in step_map:
                    visit(dep)
            visited.add(node_id)
            sorted_ids.append(node_id)
            
        for node_id in graph:
            visit(node_id)
            
        return [step_map[sid] for sid in sorted_ids]

    def execute_plan(
        self,
        plan: dict[str, Any],
        df: pd.DataFrame | None,
        state: AgentState,
        ws_callback: Callable[[dict[str, Any]], None] | None = None
    ) -> AgentState:
        """Run the legacy sequential execution loop, updating state and executing agents."""
        logger.info("Executing plan via LegacyExecutionEngine fallback.")
        state["status"] = "executing"
        state["trace"] = []
        
        # 1. Plan validation
        validation_errors = validate_plan(plan)
        if validation_errors:
            error_msg = f"Plan validation failed: {'; '.join(validation_errors)}"
            logger.error(error_msg)
            state["status"] = "failed"
            state["trace"].append({
                "step_id": "validation",
                "agent": "validator",
                "description": "Validate LLM-generated plan",
                "status": "failed",
                "duration_sec": 0.0,
                "model": "system",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return state

        steps = plan.get("steps", [])
        
        # 2. Topological sort
        sorted_steps = self._topological_sort(steps)
        logger.info("Plan sorted topologically (legacy): %s", [s["id"] for s in sorted_steps])

        # Initialize trace entries
        for step in sorted_steps:
            state["trace"].append({
                "step_id": step["id"],
                "agent": step["agent"],
                "description": step.get("description", ""),
                "status": "pending",
                "duration_sec": 0.0,
                "model": "",
                "error": None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        # Keep track of failed steps
        failed_steps = set()

        for step in sorted_steps:
            step_id = step["id"]
            agent_name = step["agent"]
            depends_on = step.get("depends_on", [])
            input_mapping = step.get("input_mapping", {})
            description = step.get("description", "")

            # Find trace item index
            trace_idx = next(i for i, t in enumerate(state["trace"]) if t["step_id"] == step_id)

            # Check if any parent dependencies failed
            failed_deps = [dep for dep in depends_on if dep in failed_steps]
            if failed_deps:
                logger.warning("Step '%s' skipped because parent step(s) failed: %s", step_id, failed_deps)
                state["trace"][trace_idx]["status"] = "failed"
                state["trace"][trace_idx]["error"] = f"Dependency failed: {failed_deps}"
                failed_steps.add(step_id)
                continue

            # Update trace status to running
            state["trace"][trace_idx]["status"] = "running"
            state["trace"][trace_idx]["timestamp"] = datetime.now(timezone.utc).isoformat()

            # WebSocket progress start event
            if ws_callback:
                try:
                    ws_callback({
                        "event": "step_started",
                        "step_id": step_id,
                        "agent": agent_name,
                        "description": description
                    })
                except Exception as ws_err:
                    logger.debug("Failed sending step_started WS notification: %s", ws_err)

            # Instantiating target agent
            agent_cls = AGENT_REGISTRY.get(agent_name)
            if agent_cls is None:
                err = f"Agent '{agent_name}' not found in registry"
                logger.error(err)
                state["trace"][trace_idx]["status"] = "failed"
                state["trace"][trace_idx]["error"] = err
                failed_steps.add(step_id)
                continue

            agent = agent_cls()
            state["trace"][trace_idx]["model"] = agent.model

            # Construct agent inputs based on mapping
            kwargs = {}
            for param_name, source_step_id in input_mapping.items():
                if source_step_id in state["outputs"]:
                    kwargs[param_name] = state["outputs"][source_step_id]

            # Inject DataFrame if needed
            if agent_name in ("quality", "eda"):
                kwargs["df"] = df

            # Inject user query context into insight agent
            if agent_name == "insight" and state["user_question"]:
                kwargs["user_question"] = state["user_question"]
                for prev_step_id, prev_out in state["outputs"].items():
                    prev_agent = prev_out.get("agent")
                    if prev_agent == "schema" and "schema_result" not in kwargs:
                        kwargs["schema_result"] = prev_out
                    elif prev_agent == "quality" and "quality_result" not in kwargs:
                        kwargs["quality_result"] = prev_out
                    elif prev_agent == "hypothesis" and "hypothesis_result" not in kwargs:
                        kwargs["hypothesis_result"] = prev_out
                    elif prev_agent == "eda" and "eda_result" not in kwargs:
                        kwargs["eda_result"] = prev_out

            start_time = time.time()
            logger.info("Executing step '%s' with agent '%s' (legacy)", step_id, agent_name)

            try:
                result = agent.run_cached(state["intelligence"], **kwargs)
                duration = round(time.time() - start_time, 3)

                if result.get("success", False):
                    state["outputs"][step_id] = result
                    state["trace"][trace_idx]["status"] = "completed"
                    state["trace"][trace_idx]["duration_sec"] = duration
                    
                    if "result" in result and isinstance(result["result"], dict):
                        state["variables"].update(result["result"])
                else:
                    err_msg = result.get("error", "Unknown agent error")
                    logger.error("Step '%s' failed: %s", step_id, err_msg)
                    state["trace"][trace_idx]["status"] = "failed"
                    state["trace"][trace_idx]["error"] = err_msg
                    state["trace"][trace_idx]["duration_sec"] = duration
                    failed_steps.add(step_id)

            except Exception as exc:
                duration = round(time.time() - start_time, 3)
                logger.exception("Step '%s' crashed with exception: %s", step_id, exc)
                state["trace"][trace_idx]["status"] = "failed"
                state["trace"][trace_idx]["error"] = str(exc)
                state["trace"][trace_idx]["duration_sec"] = duration
                failed_steps.add(step_id)

            # WebSocket progress completed/failed event
            if ws_callback:
                try:
                    ws_callback({
                        "event": "step_completed" if step_id not in failed_steps else "step_failed",
                        "step_id": step_id,
                        "agent": agent_name,
                        "duration": state["trace"][trace_idx]["duration_sec"],
                        "error": state["trace"][trace_idx]["error"]
                    })
                except Exception as ws_err:
                    logger.debug("Failed sending WS completion notification: %s", ws_err)

        # 3. Handle engine recovery pathway
        final_insight_ok = any(
            t["agent"] == "insight" and t["status"] == "completed" 
            for t in state["trace"]
        )

        if not final_insight_ok:
            logger.warning("Legacy execution engine: running fallback recovery insight.")
            recovery_result = self._run_recovery_insight(state, failed_steps)
            state["outputs"]["recovery_insight"] = recovery_result
            state["trace"].append({
                "step_id": "recovery_insight",
                "agent": "insight",
                "description": "Fallback synthesis after legacy loop failure.",
                "status": "completed",
                "duration_sec": 0.5,
                "model": "system",
                "error": None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        state["status"] = "failed" if failed_steps else "completed"
        
        # Persist legacy execution state
        if self.session_store and "sid" in state:
            try:
                self.session_store.save_execution_state(state["sid"], plan, state)
            except Exception as db_err:
                logger.error("Failed persisting execution state: %s", db_err)

        return state

    def _run_recovery_insight(self, state: AgentState, failed_steps: set[str]) -> dict[str, Any]:
        """Synthesize a diagnostic summary indicating which steps failed and what succeeded."""
        trace_summary = []
        for t in state["trace"]:
            trace_summary.append(f"- Step '{t['step_id']}' ({t['agent']}): {t['status']} " + (f"[Error: {t['error']}]" if t['error'] else ""))
        
        summary_text = f"""### ⚠️ Pipeline Execution Partially Failed

The autonomous execution plan encountered issues during processing.

#### Trace Log:
{chr(10).join(trace_summary)}

#### Synthesis:
The platform successfully initialized but failed to complete some reasoning steps. Please inspect the dataset diagnostics or check your Ollama server configuration.
"""
        return {
            "agent": "insight",
            "success": True,
            "result": {
                "summary": "Pipeline execution completed with failures.",
                "insights": [{"title": "Execution Issues", "text": "Some steps failed during pipeline execution. See the trace log for details."}],
                "actionable_recommendations": ["Verify Ollama model availability", "Check dataset cleanliness"]
            },
            "evidence": [f"Failed steps: {list(failed_steps)}"],
            "confidence": 0.30,
            "error": None,
            "duration_sec": 0.1,
            "model": "system",
            "cached": False,
            "markdown_report": summary_text
        }


class ExecutionEngine:
    """Orchestrates plan execution using a compiled LangGraph StateGraph,

    with SQLite checkpointing, human-in-the-loop gates, conditional routing,
    and fallback recovery to a sequential executor if needed.
    """

    def __init__(self, session_store: Any = None) -> None:
        self.session_store = session_store

    def execute_plan(
        self,
        plan: dict[str, Any],
        df: pd.DataFrame | None,
        state: AgentState,
        ws_callback: Callable[[dict[str, Any]], None] | None = None
    ) -> AgentState:
        """Run the execution plan, updating state and executing agents via LangGraph."""
        # Ensure state is the LangGraph AgentState
        lg_state = AgentState(state)
        lg_state["plan"] = plan
        lg_state["status"] = "executing"

        # Instantiate checkpointer
        from data.checkpointer import SQLiteCheckpointer
        if self.session_store and hasattr(self.session_store, "_conn"):
            checkpointer = SQLiteCheckpointer(conn=self.session_store._conn)
        else:
            checkpointer = SQLiteCheckpointer()

        try:
            graph = build_langgraph_engine(checkpointer)

            # Setup config
            config = {
                "configurable": {
                    "thread_id": lg_state.get("sid", "default_thread"),
                    "df": df,
                    "ws_callback": ws_callback
                }
            }

            # Invoke graph
            final_state_dict = graph.invoke(lg_state, config)
            final_state = AgentState(final_state_dict)

            # Save execution state to session store
            if self.session_store and "sid" in final_state:
                try:
                    self.session_store.save_execution_state(final_state["sid"], plan, final_state)
                except Exception as db_err:
                    logger.error("Failed persisting execution state: %s", db_err)

            return final_state

        except Exception as exc:
            if isinstance(exc, (NodeInterrupt, GraphInterrupt)):
                logger.info("LangGraph execution paused for human approval.")
                try:
                    state_val = graph.get_state(config).values
                    lg_state.update(state_val)
                except Exception:
                    pass
                lg_state["status"] = "awaiting_approval"
                lg_state["requires_approval"] = True
                
                # Persist intermediate state
                if self.session_store and "sid" in lg_state:
                    try:
                        self.session_store.save_execution_state(lg_state["sid"], plan, lg_state)
                    except Exception as db_err:
                        logger.error("Failed persisting intermediate state: %s", db_err)
                return lg_state

            logger.exception("LangGraph execution failed; falling back to LegacyExecutionEngine: %s", exc)
            
            # Fall back to legacy sequential loop
            legacy_engine = LegacyExecutionEngine(self.session_store)
            return legacy_engine.execute_plan(plan, df, state, ws_callback)

    def resume_execution(
        self,
        sid: str,
        adjusted_plan: Optional[dict[str, Any]],
        df: pd.DataFrame | None,
        ws_callback: Callable[[dict[str, Any]], None] | None = None
    ) -> AgentState:
        """Resume execution of a plan paused at a human approval gate."""
        # Instantiate checkpointer
        from data.checkpointer import SQLiteCheckpointer
        if self.session_store and hasattr(self.session_store, "_conn"):
            checkpointer = SQLiteCheckpointer(conn=self.session_store._conn)
        else:
            checkpointer = SQLiteCheckpointer()

        graph = build_langgraph_engine(checkpointer)

        config = {
            "configurable": {
                "thread_id": sid,
                "df": df,
                "ws_callback": ws_callback
            }
        }

        # 1. Update state in checkpointer
        state_updates = {"approval_granted": True}
        if adjusted_plan:
            state_updates["plan"] = adjusted_plan

        graph.update_state(config, state_updates)

        try:
            # 2. Invoke graph with None input to resume
            final_state_dict = graph.invoke(None, config)
            final_state = AgentState(final_state_dict)

            # Save execution state to session store
            if self.session_store:
                self.session_store.save_execution_state(sid, final_state.get("plan", {}), final_state)

            return final_state

        except Exception as exc:
            if isinstance(exc, (NodeInterrupt, GraphInterrupt)):
                logger.info("LangGraph execution paused again for human approval.")
                try:
                    state_val = graph.get_state(config).values
                    final_state = AgentState(state_val)
                except Exception:
                    final_state = AgentState()
                final_state["status"] = "awaiting_approval"
                final_state["requires_approval"] = True
                return final_state

            logger.exception("LangGraph resume failed; falling back to LegacyExecutionEngine: %s", exc)
            try:
                state_val = graph.get_state(config).values
                curr_state = AgentState(state_val)
            except Exception:
                curr_state = AgentState()
                curr_state["sid"] = sid

            legacy_engine = LegacyExecutionEngine(self.session_store)
            return legacy_engine.execute_plan(curr_state.get("plan", {}), df, curr_state, ws_callback)

    def _topological_sort(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort plan steps topologically (delegated to legacy sort)."""
        return LegacyExecutionEngine()._topological_sort(steps)
