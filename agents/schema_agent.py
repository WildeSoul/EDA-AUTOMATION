"""
Schema Agent — Deep semantic understanding of dataset columns.

Uses DeepSeek V4 Pro to analyze column relationships, data modeling
suggestions, and enhanced semantic classification beyond the
intelligence layer's initial pass.
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
class SchemaAgent(BaseAgent):
    """Performs deep semantic analysis of dataset schema.

    Goes beyond the intelligence layer's heuristic + LLM classification by:
    - Identifying inter-column relationships (e.g., TotalCharges ≈ Monthly × Tenure)
    - Suggesting data modeling decisions (encode, drop, transform)
    - Detecting composite keys and hierarchies
    - Recommending the target variable for ML
    """

    name = "schema"
    model = AGENT_MODELS["reasoning"]

    def run(self, intelligence: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Run deep schema analysis.

        Args:
            intelligence: Dataset Intelligence dict.

        Returns:
            Agent result with column analysis, relationships, and modeling advice.
        """
        columns = intelligence.get("columns", {})
        if not columns:
            return self._make_result(
                success=False,
                error="No column metadata in intelligence dict",
            )

        # Build compact schema summary
        schema_lines = []
        for col, info in list(columns.items())[:30]:
            samples = ", ".join(str(s) for s in info.get("sample_values", [])[:3])
            schema_lines.append(
                f"- {col}: {info['dtype']} | {info['semantic_type']} "
                f"(conf={info['confidence']:.2f}) | "
                f"unique={info['nunique']} | missing={info['missing_pct']}% | "
                f"samples: {samples}"
            )
        schema_text = "\n".join(schema_lines)

        prompt = f"""You are analyzing a {intelligence.get('dataset_type', 'unknown')} dataset
from the {intelligence.get('domain', 'unknown')} domain.

## DATASET: {intelligence.get('row_count', 0)} rows × {intelligence.get('column_count', 0)} columns

## COLUMN PROFILES
{schema_text}

## TARGET CANDIDATES: {intelligence.get('target_candidates', [])}
## ID COLUMNS: {intelligence.get('id_columns', [])}

## TASK
Perform deep schema analysis. Return a JSON object with this EXACT structure:

{{
    "relationships": [
        {{"columns": ["col_a", "col_b"], "type": "derived|correlated|hierarchical", "description": "explanation"}},
    ],
    "modeling_advice": [
        {{"column": "col_name", "action": "drop|encode|transform|keep|engineer", "reason": "why", "code_hint": "optional python snippet"}}
    ],
    "recommended_target": "column_name or null",
    "recommended_target_reason": "why this is the best target",
    "data_type_fixes": [
        {{"column": "col_name", "current": "object", "suggested": "float64", "fix_code": "df['col'] = pd.to_numeric(df['col'], errors='coerce')"}}
    ],
    "summary": "2-3 sentence executive summary of schema quality and readiness"
}}

Return ONLY valid JSON."""

        system = (
            "You are a principal data engineer performing schema analysis. "
            "Identify column relationships, recommend data modeling decisions, "
            "and suggest the best ML target. Return ONLY valid JSON."
        )

        try:
            raw = query_ollama(self.model, prompt, system=system, timeout=TIMEOUT_CLOUD)

            # Parse JSON from response
            json_str = raw.strip()
            if "```" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]

            analysis = json.loads(json_str)

            # Build evidence list
            evidence = []
            for rel in analysis.get("relationships", []):
                evidence.append(
                    f"{rel['type']}: {' ↔ '.join(rel['columns'])} — {rel['description']}"
                )
            for fix in analysis.get("data_type_fixes", []):
                evidence.append(
                    f"Type fix: {fix['column']} ({fix['current']} → {fix['suggested']})"
                )

            return self._make_result(
                success=True,
                result=analysis,
                evidence=evidence,
                confidence=0.85,
            )

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Schema agent LLM call failed or returned invalid JSON (%s); using heuristic schema analysis.", exc)
            fallback = self._build_heuristic_analysis(intelligence)
            evidence = [
                f"Modeling advice generated for {len(fallback['modeling_advice'])} columns",
                f"Target recommendation: {fallback.get('recommended_target') or 'None'}",
            ]
            return self._make_result(
                success=True,
                result=fallback,
                evidence=evidence,
                confidence=0.75,
            )

    def _build_heuristic_analysis(self, intelligence: dict[str, Any]) -> dict[str, Any]:
        """Deterministic rule-based schema analysis when LLM is unavailable."""
        columns = intelligence.get("columns", {})
        col_names = list(columns.keys())
        target = intelligence.get("target_column") or (intelligence.get("target_candidates", [None])[0] if intelligence.get("target_candidates") else None)
        relationships = []
        modeling_advice = []
        data_type_fixes = []

        for name, meta in columns.items():
            sem = meta.get("semantic_type", "")
            dtype = meta.get("dtype", "")
            null_pct = meta.get("missing_pct", meta.get("null_pct", 0.0))

            if null_pct > 50:
                modeling_advice.append({
                    "column": name,
                    "action": "drop",
                    "reason": f"High missing rate ({null_pct:.1f}%)",
                    "code_hint": f"df = df.drop(columns=['{name}'])"
                })
            elif sem in ("identifier", "id") or meta.get("is_identifier"):
                modeling_advice.append({
                    "column": name,
                    "action": "drop",
                    "reason": "Identifier column with high uniqueness has no predictive value",
                    "code_hint": f"df = df.drop(columns=['{name}'])"
                })
            elif sem in ("category", "categorical", "binary", "binary_target"):
                modeling_advice.append({
                    "column": name,
                    "action": "encode",
                    "reason": "Categorical feature needs encoding for ML modeling",
                    "code_hint": f"df = pd.get_dummies(df, columns=['{name}'], drop_first=True)"
                })
            elif dtype == "object" and sem not in ("text", "identifier"):
                data_type_fixes.append({
                    "column": name,
                    "current": "object",
                    "suggested": "category",
                    "fix_code": f"df['{name}'] = df['{name}'].astype('category')"
                })

        if not target and col_names:
            for c in col_names:
                if any(w in c.lower() for w in ("churn", "target", "label", "outcome", "status", "price", "revenue")):
                    target = c
                    break

        return {
            "relationships": relationships,
            "modeling_advice": modeling_advice,
            "recommended_target": target,
            "recommended_target_reason": f"Selected '{target}' based on domain target heuristics" if target else "No clear target column detected",
            "data_type_fixes": data_type_fixes,
            "summary": f"Heuristic schema analysis evaluated {len(col_names)} columns. Target: {target or 'None'}."
        }
