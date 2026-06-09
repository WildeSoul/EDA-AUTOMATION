"""
Insight Agent — Executive synthesis with explainability metadata.

The final agent in the pipeline. Synthesizes outputs from all other agents
into polished, executive-level insights with clear evidence trails.

Uses kimi-k2.6:cloud for report-quality writing.
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
class InsightAgent(BaseAgent):
    """Synthesizes all agent outputs into executive-level insights.

    Takes the combined output of Schema, Quality, Hypothesis, and EDA
    agents and produces:
    - Executive summary
    - Top findings ranked by business impact
    - Actionable recommendations with thresholds
    - Evidence trails for every insight
    - "What to investigate next" suggestions
    """

    name = "insight"
    model = AGENT_MODELS["reporting"]  # kimi-k2.6 — best for reports

    def run(self, intelligence: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Generate executive insights from all agent results.

        Args:
            intelligence: Dataset Intelligence dict.
            **kwargs:
                schema_result (dict): Output from Schema Agent.
                quality_result (dict): Output from Quality Agent.
                hypothesis_result (dict): Output from Hypothesis Agent.
                eda_result (dict): Output from EDA Agent.
                user_question (str): Optional specific question to address.

        Returns:
            Agent result with structured executive report.
        """
        schema_result = kwargs.get("schema_result", {})
        quality_result = kwargs.get("quality_result", {})
        hypothesis_result = kwargs.get("hypothesis_result", {})
        eda_result = kwargs.get("eda_result", {})
        user_question = kwargs.get("user_question", "")

        # ── Collect all evidence ──
        all_evidence = []

        # Schema evidence
        if schema_result.get("success"):
            for e in schema_result.get("evidence", []):
                all_evidence.append(f"[Schema] {e}")
            summary = schema_result.get("result", {}).get("summary", "")
            if summary:
                all_evidence.append(f"[Schema] {summary}")

        # Quality evidence
        if quality_result.get("success"):
            q = quality_result.get("result", {})
            all_evidence.append(
                f"[Quality] Overall: {q.get('overall_assessment', '?')}, "
                f"{q.get('total_issues', 0)} issues "
                f"({q.get('critical_count', 0)} critical, "
                f"{q.get('warning_count', 0)} warnings)"
            )
            for e in quality_result.get("evidence", [])[:5]:
                all_evidence.append(f"[Quality] {e}")

        # Hypothesis evidence
        if hypothesis_result.get("success"):
            hyps = hypothesis_result.get("result", {}).get("hypotheses", [])
            for h in hyps[:5]:
                all_evidence.append(
                    f"[Hypothesis] {h.get('id', '?')}: {h.get('statement', '?')} "
                    f"(priority: {h.get('priority', '?')})"
                )

        # EDA evidence
        if eda_result.get("success"):
            for e in eda_result.get("evidence", []):
                all_evidence.append(f"[EDA] {e}")
            interp = eda_result.get("result", {}).get("interpretation", "")
            if interp:
                all_evidence.append(f"[EDA Interpretation] {interp[:300]}")

        evidence_text = "\n".join(f"- {e}" for e in all_evidence[:30])

        # ── Build prompt for executive synthesis ──
        question_text = ""
        if user_question:
            question_text = f"\n## USER'S SPECIFIC QUESTION\n{user_question}\n"

        prompt = f"""You are creating an executive intelligence brief for a dataset.

## DATASET
- Type: {intelligence.get('dataset_type', 'unknown')}
- Domain: {intelligence.get('domain', 'unknown')}
- Size: {intelligence.get('row_count', 0)} rows × {intelligence.get('column_count', 0)} columns
- Quality: {intelligence.get('quality_grade', '?')} ({intelligence.get('quality_score', 0)}/100)
- Target candidates: {intelligence.get('target_candidates', [])}
{question_text}
## EVIDENCE FROM ALL AGENTS
{evidence_text}

## TASK
Create a structured executive intelligence brief. Return a JSON object:

{{
    "executive_summary": "2-3 sentence overview of the most important finding",
    "top_findings": [
        {{
            "rank": 1,
            "finding": "Clear statement of the finding",
            "evidence": ["Specific evidence point 1", "Evidence point 2"],
            "business_impact": "What this means for the business",
            "confidence": "high|medium|low",
            "action_required": true
        }}
    ],
    "recommendations": [
        {{
            "priority": 1,
            "action": "Specific action to take",
            "rationale": "Why this matters",
            "implementation": "How to do it (specific threshold/code/steps)",
            "expected_impact": "What improvement to expect"
        }}
    ],
    "data_readiness": {{
        "ml_ready": true,
        "blockers": ["List any blockers"],
        "preparation_steps": ["Steps needed before ML"]
    }},
    "next_investigations": [
        "Specific analysis to run next"
    ]
}}

RULES:
- Rank findings by BUSINESS IMPACT, not statistical significance
- Every finding must cite specific evidence
- Recommendations must be actionable with specific steps
- Be honest about confidence levels
- Return ONLY valid JSON"""

        system = (
            "You are a principal data scientist creating executive intelligence briefs. "
            "Combine statistical rigor with business clarity. "
            "Every insight must be backed by evidence. Return ONLY valid JSON."
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

            report = json.loads(json_str)

            return self._make_result(
                success=True,
                result=report,
                evidence=all_evidence,
                confidence=0.85,
            )

        except json.JSONDecodeError as exc:
            logger.warning("Insight agent: invalid JSON from LLM: %s", exc)
            # Fallback: return raw text as summary
            return self._make_result(
                success=True,
                result={
                    "executive_summary": raw[:500] if 'raw' in dir() else "Analysis completed.",
                    "top_findings": [],
                    "recommendations": [],
                    "data_readiness": {"ml_ready": False, "blockers": ["Report generation incomplete"]},
                    "next_investigations": [],
                },
                evidence=all_evidence,
                confidence=0.50,
            )
        except Exception as exc:
            logger.error("Insight agent failed: %s", exc)
            return self._make_result(
                success=False,
                error=str(exc),
            )
