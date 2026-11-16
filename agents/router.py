"""
Agent Router — Simple routing logic for Phase 1.

Determines which agents to run based on user action or query intent.
No LangGraph — just clean if/else routing that's easy to migrate later.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from agents import AGENT_REGISTRY
from agents.schema_agent import SchemaAgent
from agents.quality_agent import QualityAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.eda_agent import EDAAgent
from agents.insight_agent import InsightAgent

logger = logging.getLogger(__name__)

# ── Intent keywords for chat routing ─────────────────────────────

_QUALITY_KEYWORDS = {
    "quality", "missing", "null", "clean", "dirty", "corrupt",
    "fix", "issue", "problem", "invalid", "mismatch", "duplicate",
}

_SCHEMA_KEYWORDS = {
    "schema", "column", "type", "dtype", "identifier", "target",
    "variable", "feature", "field", "structure", "describe",
}

_HYPOTHESIS_KEYWORDS = {
    "hypothesis", "hypotheses", "predict", "expect", "theory",
    "investigate", "test", "why", "cause", "reason", "influence",
}


def _detect_intent(query: str) -> str:
    """Detect user intent from a chat query.

    Returns one of: 'quality', 'schema', 'hypothesis', 'eda', 'general'
    """
    lower = query.lower()
    words = set(lower.split())

    if words & _QUALITY_KEYWORDS:
        return "quality"
    if words & _SCHEMA_KEYWORDS:
        return "schema"
    if words & _HYPOTHESIS_KEYWORDS:
        return "hypothesis"
    return "eda"


def route_request(
    action: str,
    intelligence: dict[str, Any],
    query: str = "",
) -> list[str]:
    """Determine which agents to run based on action and query.

    Args:
        action: The action type — 'upload', 'analyze', 'chat', 'quality', 'schema'.
        intelligence: Dataset Intelligence dict.
        query: Optional user query for intent detection.

    Returns:
        Ordered list of agent names to run.
    """
    if action == "upload":
        # Auto-run on upload: schema → quality
        return ["schema", "quality"]

    if action == "full_analysis":
        # Full pipeline: schema → quality → hypothesis → eda → insight
        return ["schema", "quality", "hypothesis", "eda", "insight"]

    if action == "schema":
        return ["schema"]

    if action == "quality":
        return ["quality"]

    if action == "hypothesis":
        return ["schema", "hypothesis"]

    if action == "analyze":
        return ["eda", "insight"]

    if action == "chat" and query:
        intent = _detect_intent(query)
        if intent == "quality":
            return ["quality", "insight"]
        if intent == "schema":
            return ["schema"]
        if intent == "hypothesis":
            return ["hypothesis", "eda", "insight"]
        return ["eda", "insight"]

    # Default: run EDA + insight
    return ["eda", "insight"]


def run_agent_pipeline(
    agent_names: list[str],
    intelligence: dict[str, Any],
    df: pd.DataFrame | None = None,
    query: str = "",
    use_cache: bool = True,
    cancel_event: Any | None = None,
    progress_callback: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Execute a pipeline of agents in order, passing results forward.

    Supports cooperative cancellation via ``cancel_event`` and stage progress
    reporting via ``progress_callback(stage_name, progress_pct)``.

    Args:
        agent_names: Ordered list of agent names to run.
        intelligence: Dataset Intelligence dict.
        df: The DataFrame (required for quality and eda agents).
        query: Optional user question.
        use_cache: Whether to use agent caching.
        cancel_event: Optional threading.Event signaling cancellation.
        progress_callback: Optional callable(stage_name: str, pct: int).

    Returns:
        {agent_name: agent_result_dict} for all agents run.
    """
    results: dict[str, dict[str, Any]] = {}
    total_stages = len(agent_names)

    for i, name in enumerate(agent_names):
        # 1. Check cooperative cancellation before stage begins
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Pipeline cancelled cooperatively before starting stage '%s'", name)
            results["_pipeline_cancelled"] = {
                "cancelled": True,
                "cancelled_at_stage": name,
                "completed_stages": list(results.keys()),
            }
            break

        # 2. Report progress
        if progress_callback:
            try:
                progress_callback(name, int((i / max(1, total_stages)) * 100))
            except Exception as cb_err:
                logger.debug("Progress callback failed: %s", cb_err)

        agent_cls = AGENT_REGISTRY.get(name)
        if agent_cls is None:
            logger.warning("Agent '%s' not found in registry, skipping", name)
            continue

        agent = agent_cls()
        kwargs: dict[str, Any] = {}

        # Pass DataFrame to agents that need it
        if name in ("quality", "eda"):
            if df is None:
                results[name] = {
                    "agent": name,
                    "success": False,
                    "error": "DataFrame required but not available",
                }
                continue
            kwargs["df"] = df

        # Pass previous results forward
        if name == "hypothesis":
            kwargs["schema_result"] = results.get("schema", {})
            kwargs["quality_result"] = results.get("quality", {})

        if name == "eda":
            hyp_result = results.get("hypothesis", {})
            if hyp_result.get("success"):
                kwargs["hypotheses"] = hyp_result.get("result", {}).get("hypotheses", [])

        if name == "insight":
            kwargs["schema_result"] = results.get("schema", {})
            kwargs["quality_result"] = results.get("quality", {})
            kwargs["hypothesis_result"] = results.get("hypothesis", {})
            kwargs["eda_result"] = results.get("eda", {})
            if query:
                kwargs["user_question"] = query

        # Run agent (with or without cache)
        logger.info("Agent '%s' started on model '%s'", name, agent.model)
        start_time = time.time()
        try:
            if use_cache:
                result = agent.run_cached(intelligence, **kwargs)
            else:
                result = agent.run(intelligence, **kwargs)
                result["duration_sec"] = round(time.time() - start_time, 3)
                result["model"] = agent.model
                result["cached"] = False
            results[name] = result
            logger.info(
                "Agent '%s' completed on model '%s': success=%s, duration=%.3fs, cached=%s",
                name, result.get("model", agent.model), result.get("success"),
                result.get("duration_sec", 0.0), result.get("cached", False)
            )
        except Exception as exc:
            logger.error("Agent '%s' crashed: %s", name, exc)
            results[name] = {
                "agent": name,
                "success": False,
                "error": str(exc),
                "evidence": [],
                "confidence": 0.0,
                "model": agent.model,
                "duration_sec": round(time.time() - start_time, 3),
                "cached": False,
            }

        # 3. Report progress after stage completion
        if progress_callback:
            try:
                progress_callback(name, int(((i + 1) / max(1, total_stages)) * 100))
            except Exception as cb_err:
                logger.debug("Progress callback failed: %s", cb_err)

        # 4. Check cooperative cancellation immediately after stage completes
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Pipeline cancelled cooperatively after finishing stage '%s'", name)
            results["_pipeline_cancelled"] = {
                "cancelled": True,
                "cancelled_at_stage": name,
                "completed_stages": list(results.keys()),
            }
            break

    return results
