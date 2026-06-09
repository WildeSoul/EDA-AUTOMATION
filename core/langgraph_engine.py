"""
LangGraph Execution Engine — Phase 4.

Manages execution state using a compiled StateGraph, SQLite checkpoints,
conditional memory routing, human approval gates, node telemetry,
and parallel execution of independent steps.
"""

from __future__ import annotations

import time
import logging
import concurrent.futures
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

from langgraph.graph import StateGraph, START, END
from langgraph.errors import NodeInterrupt
from langgraph.checkpoint.base import get_checkpoint_id
from langchain_core.runnables import RunnableConfig

from agents import AGENT_REGISTRY
from data.checkpointer import SQLiteCheckpointer
from core.plan_validator import validate_plan

logger = logging.getLogger(__name__)


class AgentState(dict):
    """Central state object for LangGraph execution.

    Keeps track of intelligence, parameters, outputs, shared variables,
    observability traces, telemetry, and control flags.
    """
    # Core Data
    sid: str
    filename: str
    intelligence: dict[str, Any]
    user_question: str
    
    # Execution DAG
    plan: dict[str, Any]
    outputs: dict[str, Any]
    
    # Context & Variables
    planner_context: dict[str, Any]
    variables: dict[str, Any]
    
    # Graph Observability & Telemetry
    trace: list[dict[str, Any]]
    node_telemetry: dict[str, dict[str, Any]]
    checkpoint_count: int
    resume_events: list[dict[str, Any]]
    graph_version: str
    graph_build_time: str
    
    # Control Flags
    status: str
    requires_approval: bool
    approval_granted: bool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setdefault("sid", "")
        self.setdefault("filename", "")
        self.setdefault("intelligence", {})
        self.setdefault("user_question", "")
        self.setdefault("planner_context", {
            "previous_question": "",
            "previous_plan": {},
            "current_goal": ""
        })
        self.setdefault("outputs", {})
        self.setdefault("variables", {})
        self.setdefault("trace", [])
        
        # Telemetry & Observability
        self.setdefault("node_telemetry", {})
        self.setdefault("checkpoint_count", 0)
        self.setdefault("resume_events", [])
        self.setdefault("graph_version", "phase4_langgraph_v1")
        self.setdefault("graph_build_time", datetime.now(timezone.utc).isoformat())

        # Control Flags
        self.setdefault("status", "idle")
        self.setdefault("requires_approval", False)
        self.setdefault("approval_granted", False)


# ── LangGraph Node Definitions ───────────────────────────────────────

def memory_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Retrieves semantic cross-session memory context."""
    logger.info("Executing MemoryNode")
    start_time = time.time()
    
    # Track attempt
    state["checkpoint_count"] += 1
    checkpoint_id = get_checkpoint_id(config) or "unknown"
    telemetry = state["node_telemetry"].setdefault("memory", {"attempts": 0, "durations": []})
    telemetry["attempts"] += 1
    telemetry["checkpoint_id"] = checkpoint_id

    # Retrieve memory
    from agents.memory_agent import MemoryAgent
    mem_agent = MemoryAgent()
    mem_res = mem_agent.run(state["intelligence"])

    duration = round(time.time() - start_time, 3)
    telemetry["durations"].append(duration)

    if mem_res.get("success"):
        state["planner_context"]["advisory_memories"] = mem_res["result"]
    else:
        logger.warning("MemoryNode failed: %s", mem_res.get("error"))

    return state


def planner_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generates execution DAG via PlannerAgent."""
    if state.get("plan") and state["plan"].get("steps"):
        logger.info("PlannerNode: plan already present in state, skipping generation.")
        return state

    logger.info("Executing PlannerNode")
    start_time = time.time()

    state["checkpoint_count"] += 1
    checkpoint_id = get_checkpoint_id(config) or "unknown"
    telemetry = state["node_telemetry"].setdefault("planner", {"attempts": 0, "durations": []})
    telemetry["attempts"] += 1
    telemetry["checkpoint_id"] = checkpoint_id

    # Execute Planner
    from agents.planner_agent import PlannerAgent
    planner = PlannerAgent()
    
    # We pass pre-retrieved memories explicitly if they exist in state
    advisory_mem = state["planner_context"].get("advisory_memories")
    result = planner.run(
        state["intelligence"],
        user_question=state["user_question"],
        planner_context=state["planner_context"],
        advisory_memories=advisory_mem
    )

    duration = round(time.time() - start_time, 3)
    telemetry["durations"].append(duration)

    if result.get("success"):
        state["plan"] = result["result"]
    else:
        logger.error("PlannerNode failed: %s", result.get("error"))
        state["status"] = "failed"
        state["plan"] = {}

    return state


def pre_approval_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Pre-approval node to persist state modifications before interruption."""
    logger.info("Executing PreApprovalNode")
    state["checkpoint_count"] += 1
    checkpoint_id = get_checkpoint_id(config) or "unknown"
    telemetry = state["node_telemetry"].setdefault("pre_approval", {"attempts": 0, "durations": []})
    telemetry["attempts"] += 1
    telemetry["checkpoint_id"] = checkpoint_id

    state["requires_approval"] = True
    state["status"] = "awaiting_approval"

    # Stream status via WS if callback is provided
    ws_callback = config.get("configurable", {}).get("ws_callback")
    if ws_callback:
        try:
            ws_callback({
                "event": "awaiting_approval",
                "sid": state.get("sid"),
                "plan": state.get("plan"),
                "checkpoint_id": checkpoint_id
            })
        except Exception as ws_err:
            logger.debug("WS approval notification failed: %s", ws_err)

    return state


def approval_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Human-in-the-loop validation node.

    Halts execution to await user verification if estimated duration is high
    or memory match scores are too low.
    """
    logger.info("Executing ApprovalNode")
    state["checkpoint_count"] += 1
    
    checkpoint_id = get_checkpoint_id(config) or "unknown"
    telemetry = state["node_telemetry"].setdefault("approval", {"attempts": 0, "durations": []})
    telemetry["attempts"] += 1
    telemetry["checkpoint_id"] = checkpoint_id

    if state.get("approval_granted", False):
        # Human has resumed execution and granted approval
        logger.info("Human approval verified. Resuming execution.")
        state["requires_approval"] = False
        state["status"] = "executing"
        
        # Log a resume event
        state["resume_events"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_id": checkpoint_id,
            "action": "approved"
        })
        return state

    logger.warning("Interrupted execution: awaiting human approval.")
    raise NodeInterrupt("Plan requires human approval. Confirm or adjust the plan to proceed.")


def executor_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Executes planned specialist agent steps topologically with retry policies

    and batch-level parallel execution of independent steps.
    """
    logger.info("Executing ExecutorNode")
    start_time = time.time()

    state["checkpoint_count"] += 1
    checkpoint_id = get_checkpoint_id(config) or "unknown"
    telemetry = state["node_telemetry"].setdefault("executor", {"attempts": 0, "durations": []})
    telemetry["attempts"] += 1
    telemetry["checkpoint_id"] = checkpoint_id

    plan = state.get("plan", {})
    steps = plan.get("steps", [])
    df = config.get("configurable", {}).get("df")
    ws_callback = config.get("configurable", {}).get("ws_callback")

    # 1. Validate plan
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

    # 2. Build trace outline
    state["trace"] = []
    for step in steps:
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

    completed_steps: Set[str] = set()
    failed_steps: Set[str] = set()

    step_map = {step["id"]: step for step in steps}

    # Helper function to check if dependencies are met
    def can_run(step_id: str) -> bool:
        step = step_map[step_id]
        for dep in step.get("depends_on", []):
            if dep not in completed_steps:
                return False
        return True

    # 3. Batch Parallel Execution Loop
    with concurrent.futures.ThreadPoolExecutor() as pool:
        while len(completed_steps) + len(failed_steps) < len(steps):
            # Check for steps that are blocked by failed/skipped dependencies
            blocked_steps = []
            for step_id, step in step_map.items():
                if step_id not in completed_steps and step_id not in failed_steps:
                    failed_deps = [dep for dep in step.get("depends_on", []) if dep in failed_steps]
                    if failed_deps:
                        blocked_steps.append((step_id, failed_deps))

            if blocked_steps:
                for step_id, failed_deps in blocked_steps:
                    logger.warning("Step '%s' skipped because parent step(s) failed: %s", step_id, failed_deps)
                    trace_idx = next(i for i, t in enumerate(state["trace"]) if t["step_id"] == step_id)
                    state["trace"][trace_idx]["status"] = "failed"
                    state["trace"][trace_idx]["error"] = f"Dependency failed: {failed_deps}"
                    failed_steps.add(step_id)
                continue

            # Find all steps that are ready to run (dependencies satisfied)
            ready_steps = [
                step_id for step_id in step_map 
                if step_id not in completed_steps and step_id not in failed_steps and can_run(step_id)
            ]

            if not ready_steps:
                # Cycle or unresolvable dependencies (already caught by validation)
                remaining = [sid for sid in step_map if sid not in completed_steps and sid not in failed_steps]
                logger.error("Deadlock/Cycle detected in topological executor for steps: %s", remaining)
                for sid in remaining:
                    trace_idx = next(i for i, t in enumerate(state["trace"]) if t["step_id"] == sid)
                    state["trace"][trace_idx]["status"] = "failed"
                    state["trace"][trace_idx]["error"] = "Deadlock/Cycle detected in topological executor"
                    failed_steps.add(sid)
                break

            # Execute the batch of ready steps in parallel
            logger.info("Executing batch of parallel steps: %s", ready_steps)
            futures = {
                pool.submit(execute_single_step, step_map[sid], state, df, ws_callback, checkpoint_id): sid 
                for sid in ready_steps
            }

            for future in concurrent.futures.as_completed(futures):
                sid = futures[future]
                try:
                    success, result, duration, error_msg = future.result()
                    if success:
                        state["outputs"][sid] = result
                        completed_steps.add(sid)
                        # Store any global variables returned by agent to variables pool
                        if "result" in result and isinstance(result["result"], dict):
                            state["variables"].update(result["result"])
                    else:
                        logger.error("Step '%s' failed: %s", sid, error_msg)
                        failed_steps.add(sid)
                except Exception as exc:
                    logger.exception("Step '%s' raised exception: %s", sid, exc)
                    failed_steps.add(sid)

    duration = round(time.time() - start_time, 3)
    telemetry["durations"].append(duration)

    state["status"] = "failed" if failed_steps else "completed"
    return state


def execute_single_step(
    step: dict[str, Any],
    state: AgentState,
    df: Any,
    ws_callback: Optional[Callable[[dict[str, Any]], None]],
    checkpoint_id: str
) -> Tuple[bool, dict[str, Any], float, Optional[str]]:
    """Executes a single plan step with built-in retries and backoffs."""
    step_id = step["id"]
    agent_name = step["agent"]
    depends_on = step.get("depends_on", [])
    input_mapping = step.get("input_mapping", {})
    description = step.get("description", "")

    # Find trace item index
    trace_idx = next(i for i, t in enumerate(state["trace"]) if t["step_id"] == step_id)

    # Update trace status to running
    state["trace"][trace_idx]["status"] = "running"
    state["trace"][trace_idx]["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Stream starting progress
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
        state["trace"][trace_idx]["status"] = "failed"
        state["trace"][trace_idx]["error"] = err
        return False, {}, 0.0, err

    agent = agent_cls()
    state["trace"][trace_idx]["model"] = agent.model

    # Construct inputs
    kwargs = {}
    for param_name, source_step_id in input_mapping.items():
        if source_step_id in state["outputs"]:
            kwargs[param_name] = state["outputs"][source_step_id]

    if agent_name in ("quality", "eda"):
        kwargs["df"] = df

    if agent_name == "insight" and state["user_question"]:
        kwargs["user_question"] = state["user_question"]
        # Map preceding agent results automatically if not mapped explicitly
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

    # Step-level telemetry
    step_telemetry = state["node_telemetry"].setdefault(step_id, {"attempts": 0, "durations": []})
    step_telemetry["checkpoint_id"] = checkpoint_id

    # 3-Attempt Retry Policy with Backoff
    max_attempts = 3
    backoff = 1.0
    last_error = None
    start_time = time.time()

    for attempt in range(1, max_attempts + 1):
        step_telemetry["attempts"] += 1
        logger.info("Executing step '%s' (attempt %d/%d)", step_id, attempt, max_attempts)
        try:
            result = agent.run_cached(state["intelligence"], **kwargs)
            duration = round(time.time() - start_time, 3)
            step_telemetry["durations"].append(duration)

            if result.get("success", False):
                state["trace"][trace_idx]["status"] = "completed"
                state["trace"][trace_idx]["duration_sec"] = duration

                # Stream completion
                if ws_callback:
                    try:
                        ws_callback({
                            "event": "step_completed",
                            "step_id": step_id,
                            "agent": agent_name,
                            "duration": duration,
                            "error": None
                        })
                    except Exception as ws_err:
                        logger.debug("Failed sending WS completion notification: %s", ws_err)
                return True, result, duration, None
            else:
                last_error = result.get("error", "Unknown agent error")
        except Exception as exc:
            logger.exception("Step '%s' raised exception: %s", step_id, exc)
            last_error = str(exc)

        # Backoff sleep
        if attempt < max_attempts:
            time.sleep(backoff)
            backoff *= 2.0

    duration = round(time.time() - start_time, 3)
    state["trace"][trace_idx]["status"] = "failed"
    state["trace"][trace_idx]["error"] = last_error
    state["trace"][trace_idx]["duration_sec"] = duration

    if ws_callback:
        try:
            ws_callback({
                "event": "step_failed",
                "step_id": step_id,
                "agent": agent_name,
                "duration": duration,
                "error": last_error
            })
        except Exception as ws_err:
            logger.debug("Failed sending WS completion notification: %s", ws_err)

    return False, {}, duration, last_error


def recovery_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Runs fallback recovery synthesis if the pipeline failed."""
    logger.info("Executing RecoveryNode")
    state["checkpoint_count"] += 1
    checkpoint_id = get_checkpoint_id(config) or "unknown"
    telemetry = state["node_telemetry"].setdefault("recovery", {"attempts": 0, "durations": []})
    telemetry["attempts"] += 1
    telemetry["checkpoint_id"] = checkpoint_id

    final_insight_ok = any(
        t["agent"] == "insight" and t["status"] == "completed"
        for t in state["trace"]
    )

    if not final_insight_ok:
        logger.warning("Pipeline execution failed or lacked final insight. Running recovery diagnostic.")
        trace_summary = []
        failed_steps = []
        for t in state["trace"]:
            trace_summary.append(f"- Step '{t['step_id']}' ({t['agent']}): {t['status']} " + (f"[Error: {t['error']}]" if t['error'] else ""))
            if t["status"] == "failed":
                failed_steps.append(t["step_id"])

        summary_text = f"""### ⚠️ Pipeline Execution Partially Failed

The autonomous execution plan encountered issues during processing.

#### Trace Log:
{chr(10).join(trace_summary)}

#### Synthesis:
The platform successfully initialized but failed to complete some reasoning steps. Please inspect the dataset diagnostics or check your Ollama server configuration.
"""
        recovery_result = {
            "agent": "insight",
            "success": True,
            "result": {
                "summary": "Pipeline execution completed with failures.",
                "insights": [{"title": "Execution Issues", "text": "Some steps failed during pipeline execution. See the trace log for details."}],
                "actionable_recommendations": ["Verify Ollama model availability", "Check dataset cleanliness"]
            },
            "evidence": [f"Failed steps: {failed_steps}"],
            "confidence": 0.30,
            "error": None,
            "duration_sec": 0.1,
            "model": "system",
            "cached": False,
            "markdown_report": summary_text
        }

        state["outputs"]["recovery_insight"] = recovery_result
        state["trace"].append({
            "step_id": "recovery_insight",
            "agent": "insight",
            "description": "Fallback diagnostic synthesis after pipeline failure.",
            "status": "completed",
            "duration_sec": 0.5,
            "model": "system",
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    return state


# ── Conditional Routing Functions ────────────────────────────────────

def memory_router(state: AgentState) -> str:
    """Inspects intelligence schema metadata to route to memory or skip it."""
    intel = state.get("intelligence", {})
    # Skip memory query for standard simple datasets to minimize latency
    if intel.get("version", 1) > 1 or intel.get("domain") != "unknown":
        logger.info("Memory Router: YES, routing to memory agent.")
        return "memory"
    logger.info("Memory Router: NO, skipping to planner.")
    return "planner"


def approval_gate(state: AgentState) -> str:
    """Halt execution for human approval if plan estimated time is high

    or if retrieval match confidence score is low.
    """
    if state.get("approval_granted", False):
        return "executor"

    plan = state.get("plan", {})
    estimates = plan.get("estimates", {})
    estimated_time_sec = estimates.get("estimated_time_sec", 0.0)

    # Retrieve memory confidence
    similar_datasets = state.get("planner_context", {}).get("advisory_memories", {}).get("similar_datasets", [])
    memory_confidence = 1.0
    if similar_datasets:
        memory_confidence = max(item.get("rank_score", 0.0) for item in similar_datasets)

    if estimated_time_sec > 30.0 or memory_confidence < 0.70:
        logger.info("Approval Gate: Triggered pause. Estimated time: %.1fs, Memory confidence: %.2f", 
                    estimated_time_sec, memory_confidence)
        return "pre_approval"
    
    logger.info("Approval Gate: Safe execution allowed. Skipping human approval gate.")
    return "executor"


def recovery_router(state: AgentState) -> str:
    """Routes to recovery node if any steps failed, otherwise ends."""
    failed = any(t["status"] == "failed" for t in state.get("trace", []))
    if failed:
        return "recovery"
    return END


# ── StateGraph Constructor ───────────────────────────────────────────

def build_langgraph_engine(checkpointer: SQLiteCheckpointer) -> StateGraph:
    """Compiles and registers the StateGraph with the SQLite checkpointer."""
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("memory", memory_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("pre_approval", pre_approval_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("recovery", recovery_node)

    # 2. Setup Routing Edges
    workflow.add_conditional_edges(
        START,
        memory_router,
        {
            "memory": "memory",
            "planner": "planner"
        }
    )

    workflow.add_edge("memory", "planner")

    workflow.add_conditional_edges(
        "planner",
        approval_gate,
        {
            "pre_approval": "pre_approval",
            "executor": "executor"
        }
    )

    # Pre-approval transitions to approval node to halt
    workflow.add_edge("pre_approval", "approval")

    # After approval node resumes, check approval gate to route to executor
    workflow.add_conditional_edges(
        "approval",
        approval_gate,
        {
            "pre_approval": "pre_approval",
            "executor": "executor"
        }
    )

    workflow.add_conditional_edges(
        "executor",
        recovery_router,
        {
            "recovery": "recovery",
            "__end__": END
        }
    )

    workflow.add_edge("recovery", END)

    # Compile with persistence SQLiteCheckpointer
    return workflow.compile(checkpointer=checkpointer)
