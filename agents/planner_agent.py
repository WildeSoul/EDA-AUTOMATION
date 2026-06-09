"""
Planner Agent — Phase 2.

Determines an optimal, dependency-correct execution plan (DAG) of specialist
agents based on the dataset's profile and user query. Includes task estimations
and incorporates context memory.
"""

from __future__ import annotations
import json
import logging
from typing import Any

from agents import BaseAgent, register_agent
from config.settings import AGENT_MODELS, TIMEOUT_CLOUD
from core.llm import query_ollama

logger = logging.getLogger(__name__)

@register_agent
class PlannerAgent(BaseAgent):
    """Lead Data Scientist Planner Agent.

    Dynamically selects which specialist agents to execute, orders them in a DAG,
    attaches cost/latency estimations, and sustains request-level memory context.
    """

    name = "planner"
    model = AGENT_MODELS["reasoning"]

    def run(self, intelligence: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Generate an execution plan.

        Args:
            intelligence: Dataset Intelligence dict.
            **kwargs:
                user_question: The user's query.
                planner_context: Context memory dict.

        Returns:
            Plan JSON dict.
        """
        user_question = kwargs.get("user_question", "")
        planner_context = kwargs.get("planner_context", {})

        # Context memory format
        prev_q = planner_context.get("previous_question", "None")
        prev_plan = json.dumps(planner_context.get("previous_plan", {}), default=str)
        curr_goal = planner_context.get("current_goal", "Analyze dataset structure and generate insights.")

        # Build column summary
        cols = list(intelligence.get("columns", {}).keys())
        target = intelligence.get("target_candidates", ["None"])

        # ── Retrieve Cross-Session Advisory Memory ──
        advisory_mem_str = ""
        pre_retrieved_mem = kwargs.get("advisory_memories") or planner_context.get("advisory_memories")
        
        # If pre_retrieved_mem is provided as an empty dict or similar, we consider memory skipped.
        has_pre_retrieved = pre_retrieved_mem is not None
        mem_res = None
        
        if has_pre_retrieved:
            if pre_retrieved_mem and pre_retrieved_mem.get("similar_datasets"):
                mem_res = {"success": True, "result": pre_retrieved_mem}
        elif "advisory_memories" in planner_context:
            # Explicitly skipped by Memory Router
            pass
        else:
            # Fallback legacy behavior: query MemoryAgent inside PlannerAgent
            try:
                from agents.memory_agent import MemoryAgent
                mem_agent = MemoryAgent()
                mem_res_raw = mem_agent.run(intelligence)
                if mem_res_raw.get("success"):
                    mem_res = mem_res_raw
            except Exception as exc:
                logger.warning("Failed to retrieve advisory memory in PlannerAgent fallback: %s", exc)

        if mem_res and mem_res.get("success"):
            similar_datasets = mem_res["result"]["similar_datasets"]
            injected_memories = mem_res["result"]["injected_memories"]
            
            if similar_datasets:
                advisory_mem_str += "## ADVISORY HISTORY (MEMORY)\n"
                advisory_mem_str += "The following historically similar datasets and their key findings were found:\n"
                for item in similar_datasets:
                    advisory_mem_str += f"- Dataset: {item['dataset']} (Similarity: {item['similarity']:.2f}, Rank: {item['rank_score']:.2f})\n"
                    advisory_mem_str += f"  Reasons: {', '.join(item['reasons'])}\n"
                
                insights = injected_memories.get("insights", [])
                if insights:
                    advisory_mem_str += "\nHistorical Insights:\n"
                    for ins in insights[:3]:
                        advisory_mem_str += f"- [{ins['source_dataset']}] {ins['title']}: {ins['content'][:150]}...\n"
                        
                hypotheses = injected_memories.get("hypotheses", [])
                if hypotheses:
                    advisory_mem_str += "\nHistorical Hypotheses:\n"
                    for hyp in hypotheses[:3]:
                        advisory_mem_str += f"- [{hyp['source_dataset']}] {hyp['hypothesis']} (Status: {hyp['status']})\n"
                        
                cleanups = injected_memories.get("cleaning_actions", [])
                if cleanups:
                    advisory_mem_str += "\nHistorical Cleaning Actions:\n"
                    for cl in cleanups[:3]:
                        advisory_mem_str += f"- [{cl['source_dataset']}] Column {cl['column']}: {cl['description']}\n"
                        
                advisory_mem_str += """
> [!CAUTION]
> **Historical knowledge is advisory only.**
> - All retrieved insights, hypotheses, and cleanups must be validated against the current dataset's live schema, statistics, and distributions before being used in conclusions.
> - Never report data points, column values, or findings from past datasets that are not statistically supported by the current session's live data.
> - Strict Separators: Column metadata and profiles are never merged. The current dataset's live intelligence layer output is the sole source of truth for schema definitions and feature roles.
"""

        prompt = f"""You are the Lead Data Scientist Planner Agent for an autonomous AI Data Scientist.
Your job is to generate a custom step-by-step execution plan of specialized agents to answer a user's question.

## DATASET INFO
- Dataset Type: {intelligence.get('dataset_type', 'unknown')}
- Domain: {intelligence.get('domain', 'unknown')}
- Columns: {cols}
- Target column(s): {target}
- Row count: {intelligence.get('row_count', 0)}

{advisory_mem_str}

## PLANNER CONTEXT MEMORY
- Previous Question: {prev_q}
- Previous Plan: {prev_plan}
- Current Goal: {curr_goal}

## USER QUESTION
"{user_question if user_question else 'Generate a full data intelligence report.'}"

## AVAILABLE SPECIALIST AGENTS
1. "schema": Deep column metadata, inter-column relationships, type fixes, encoding suggestions.
2. "quality": Diagnostics of nulls, duplicates, outliers, constant columns.
3. "hypothesis": Formulates scientific testable hypotheses based on schema/quality results.
4. "eda": Performs statistical analysis, PCA, kmeans, anomalies, correlations, and generates plots.
5. "insight": Synthesizes all agent outputs to directly answer the user question and provide recommendations.

## INSTRUCTIONS
Construct an execution plan (DAG) of steps. You must return a JSON object with this EXACT structure (no other text):
{{
    "rationale": "Why you chose these agents and this execution sequence.",
    "estimates": {{
        "estimated_steps": 3,
        "estimated_time_sec": 7.5,
        "estimated_llm_calls": 2
    }},
    "steps": [
        {{
            "id": "step_1",
            "agent": "schema",
            "depends_on": [],
            "description": "Examine column profiles and identify relationships.",
            "input_mapping": {{}}
        }},
        {{
            "id": "step_2",
            "agent": "eda",
            "depends_on": ["step_1"],
            "description": "Check correlation matrices and distribution metrics.",
            "input_mapping": {{
                "schema_result": "step_1"
            }}
        }}
    ]
}}

RULES:
- step 'id' must be unique.
- 'agent' must be one of: "schema", "quality", "hypothesis", "eda", "insight".
- 'depends_on' lists step ids that must finish before this step runs.
- 'input_mapping' maps parameter names (like "schema_result", "quality_result", "hypothesis_result", "eda_result") to the step id that provides them.
- If the user wants a full report, execute all agents in a logical sequence.
- If the user asks a very specific question, select only the necessary agents (e.g. schema + eda + insight).
- Keep plans optimal. Do not add redundant steps.
- Return ONLY valid JSON, no explanations or markdown blocks."""

        system = (
            "You are a principal AI planner orchestrating a team of data science agents. "
            "Output ONLY valid JSON following the execution plan schema."
        )

        try:
            raw = query_ollama(self.model, prompt, system=system, timeout=TIMEOUT_CLOUD)
            
            # Extract JSON from response
            json_str = raw.strip()
            if "```" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]

            plan_dict = json.loads(json_str)
            
            # Simple sanitization
            if "steps" not in plan_dict:
                raise ValueError("LLM JSON response missing 'steps' key.")
            
            return self._make_result(
                success=True,
                result=plan_dict,
                evidence=[plan_dict.get("rationale", "")],
                confidence=0.90
            )

        except Exception as exc:
            logger.error("Planner agent failed to generate plan: %s", exc)
            # Fallback plan: static pipeline based on user action
            steps = [
                {"id": "step_1", "agent": "schema", "depends_on": [], "description": "Auto-plan schema pass.", "input_mapping": {}},
                {"id": "step_2", "agent": "quality", "depends_on": ["step_1"], "description": "Auto-plan quality diagnostics.", "input_mapping": {"schema_result": "step_1"}},
                {"id": "step_3", "agent": "hypothesis", "depends_on": ["step_1", "step_2"], "description": "Auto-plan hypothesis generation.", "input_mapping": {"schema_result": "step_1", "quality_result": "step_2"}},
                {"id": "step_4", "agent": "eda", "depends_on": ["step_1", "step_3"], "description": "Auto-plan statistical analysis.", "input_mapping": {"schema_result": "step_1"}},
                {"id": "step_5", "agent": "insight", "depends_on": ["step_1", "step_2", "step_3", "step_4"], "description": "Auto-plan insight synthesis.", "input_mapping": {"schema_result": "step_1", "quality_result": "step_2", "hypothesis_result": "step_3", "eda_result": "step_4"}}
            ]
            
            if user_question:
                steps[-1]["user_question"] = user_question
                
            fallback_plan = {
                "rationale": f"Fallback static pipeline triggered due to planning error: {exc}",
                "estimates": {
                    "estimated_steps": 5,
                    "estimated_time_sec": 12.0,
                    "estimated_llm_calls": 4
                },
                "steps": steps
            }
            
            return self._make_result(
                success=True,
                result=fallback_plan,
                evidence=["Planner fallback triggered"],
                confidence=0.50
            )
