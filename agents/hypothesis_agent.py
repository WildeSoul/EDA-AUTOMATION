"""
Hypothesis Agent — Generates testable hypotheses from dataset intelligence.

Sits between Schema and EDA in the pipeline:
  Schema Agent → Hypothesis Agent → EDA Agent

Creates the "AI scientist" feel by formulating specific, testable
hypotheses that the EDA agent can then investigate.
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
class HypothesisAgent(BaseAgent):
    """Generates testable hypotheses about the dataset.

    Uses the intelligence layer's initial hypotheses as a seed, then
    generates deeper, more specific hypotheses based on:
    - Column relationships from Schema Agent
    - Quality issues from Quality Agent
    - Statistical properties of the data
    """

    name = "hypothesis"
    model = AGENT_MODELS["reasoning"]

    def run(self, intelligence: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Generate testable hypotheses.

        Args:
            intelligence: Dataset Intelligence dict.
            **kwargs:
                schema_result (dict): Optional output from Schema Agent.
                quality_result (dict): Optional output from Quality Agent.

        Returns:
            Agent result with categorized hypotheses and test plans.
        """
        schema_result = kwargs.get("schema_result", {})
        quality_result = kwargs.get("quality_result", {})

        columns = intelligence.get("columns", {})
        if not columns:
            return self._make_result(
                success=False,
                error="No column metadata available for hypothesis generation",
            )

        # Build context for LLM
        col_summaries = []
        for col, info in list(columns.items())[:25]:
            col_summaries.append(
                f"- {col}: {info['semantic_type']} "
                f"(unique={info['nunique']}, missing={info['missing_pct']}%)"
            )
        col_text = "\n".join(col_summaries)

        # Add schema relationships if available
        relationships_text = ""
        if schema_result and schema_result.get("success"):
            rels = schema_result.get("result", {}).get("relationships", [])
            if rels:
                rel_lines = [
                    f"- {r['type']}: {' ↔ '.join(r['columns'])} — {r['description']}"
                    for r in rels[:5]
                ]
                relationships_text = "\n## KNOWN RELATIONSHIPS\n" + "\n".join(rel_lines)

        # Add quality context
        quality_text = ""
        if quality_result and quality_result.get("success"):
            q_res = quality_result.get("result", {})
            quality_text = (
                f"\n## QUALITY CONTEXT\n"
                f"Overall: {q_res.get('overall_assessment', 'unknown')}, "
                f"{q_res.get('total_issues', 0)} issues found"
            )

        # Seed hypotheses from intelligence layer
        seed = intelligence.get("hypotheses", [])
        seed_text = ""
        if seed:
            seed_text = "\n## INITIAL HYPOTHESES (expand on these)\n" + "\n".join(
                f"- {h}" for h in seed
            )

        targets = intelligence.get("target_candidates", [])
        target_text = f"\n## TARGET CANDIDATES: {targets}" if targets else ""

        prompt = f"""You are a senior data scientist formulating testable hypotheses.

## DATASET: {intelligence.get('dataset_type', 'unknown')} ({intelligence.get('domain', 'unknown')})
## SIZE: {intelligence.get('row_count', 0)} rows × {intelligence.get('column_count', 0)} columns

## COLUMNS
{col_text}
{target_text}
{relationships_text}
{quality_text}
{seed_text}

## TASK
Generate 5-8 specific, testable hypotheses. Each hypothesis must:
1. Name specific columns
2. State the expected relationship or pattern
3. Suggest the statistical test to validate it
4. Estimate business impact if confirmed

Return a JSON object:
{{
    "hypotheses": [
        {{
            "id": "H1",
            "statement": "Customers with month-to-month contracts have significantly higher churn rates than those with yearly contracts",
            "columns_involved": ["Contract", "Churn"],
            "category": "relationship|distribution|comparison|trend|anomaly",
            "test_method": "Chi-square test of independence",
            "expected_direction": "positive|negative|significant_difference",
            "business_impact": "If confirmed, targeted retention campaigns for month-to-month customers could reduce churn by X%",
            "priority": "high|medium|low"
        }}
    ],
    "investigation_order": ["H1", "H3", "H2", ...],
    "summary": "Brief overview of the hypothesis landscape"
}}

Return ONLY valid JSON."""

        system = (
            "You are a principal data scientist who thinks like a researcher. "
            "Generate specific, testable hypotheses with clear statistical test "
            "recommendations. Prioritize high-impact hypotheses. Return ONLY valid JSON."
        )

        try:
            raw = query_ollama(self.model, prompt, system=system, timeout=TIMEOUT_CLOUD)

            # Parse JSON
            json_str = raw.strip()
            if "```" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]

            result = json.loads(json_str)
            hypotheses = result.get("hypotheses", [])

            # Build evidence
            evidence = [
                f"Generated {len(hypotheses)} hypotheses",
                f"High priority: {sum(1 for h in hypotheses if h.get('priority') == 'high')}",
                f"Categories: {set(h.get('category', '') for h in hypotheses)}",
            ]

            return self._make_result(
                success=True,
                result=result,
                evidence=evidence,
                confidence=0.80,
            )

        except json.JSONDecodeError as exc:
            logger.warning("Hypothesis agent: invalid JSON from LLM: %s", exc)
            # Fallback: return seed hypotheses
            fallback = {
                "hypotheses": [
                    {
                        "id": f"H{i+1}",
                        "statement": h,
                        "columns_involved": [],
                        "category": "unknown",
                        "test_method": "To be determined",
                        "expected_direction": "unknown",
                        "business_impact": "To be assessed",
                        "priority": "medium",
                    }
                    for i, h in enumerate(seed)
                ],
                "investigation_order": [f"H{i+1}" for i in range(len(seed))],
                "summary": "Fallback hypotheses from initial intelligence scan",
            }
            return self._make_result(
                success=True,
                result=fallback,
                evidence=["Using fallback hypotheses (LLM JSON parse failed)"],
                confidence=0.50,
            )
        except Exception as exc:
            logger.error("Hypothesis agent failed: %s", exc)
            return self._make_result(
                success=False,
                error=str(exc),
            )
