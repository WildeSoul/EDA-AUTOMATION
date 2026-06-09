"""
Quality Agent — Comprehensive data quality diagnostics.

Goes beyond the existing health score with deeper checks:
type mismatches, inconsistent categories, suspicious constants,
range violations, and encoding issues. Generates actionable fix
recommendations with Python code.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from agents import BaseAgent, register_agent
from config.settings import AGENT_MODELS, TIMEOUT_CLOUD
from core.llm import query_ollama

logger = logging.getLogger(__name__)


def _detect_type_mismatches(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Find columns stored as strings that should be numeric."""
    issues = []
    for col in df.select_dtypes(include=["object"]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        sample = non_null.head(100).astype(str)
        # Check if values look numeric (strip $, commas, %)
        cleaned = sample.str.replace(r"[$,€£%]", "", regex=True).str.strip()
        numeric_count = pd.to_numeric(cleaned, errors="coerce").notna().sum()
        if numeric_count / len(sample) > 0.8:
            issues.append({
                "column": col,
                "issue": "type_mismatch",
                "severity": "warning",
                "description": (
                    f"Column '{col}' is stored as text but {numeric_count}/{len(sample)} "
                    f"sample values are numeric (after removing currency/% symbols)"
                ),
                "fix_code": (
                    f"df['{col}'] = pd.to_numeric("
                    f"df['{col}'].astype(str).str.replace(r'[$,€£%]', '', regex=True), "
                    f"errors='coerce')"
                ),
                "auto_fixable": True,
            })
    return issues


def _detect_inconsistent_categories(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Find categorical columns with likely duplicates (case/whitespace variants)."""
    issues = []
    for col in df.select_dtypes(include=["object"]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0 or non_null.nunique() > 50:
            continue

        values = non_null.unique()
        # Check for case variants
        lower_map: dict[str, list[str]] = {}
        for v in values:
            key = str(v).strip().lower()
            lower_map.setdefault(key, []).append(str(v))

        duplicates = {k: v for k, v in lower_map.items() if len(v) > 1}
        if duplicates:
            examples = []
            for _key, variants in list(duplicates.items())[:3]:
                examples.append(f"  {' vs '.join(repr(v) for v in variants)}")

            issues.append({
                "column": col,
                "issue": "inconsistent_categories",
                "severity": "warning",
                "description": (
                    f"Column '{col}' has {len(duplicates)} groups of "
                    f"case/whitespace variants:\n" + "\n".join(examples)
                ),
                "fix_code": f"df['{col}'] = df['{col}'].str.strip().str.lower()",
                "auto_fixable": True,
            })
    return issues


def _detect_suspicious_constants(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Find columns where 95%+ of values are the same."""
    issues = []
    for col in df.columns:
        if len(df) < 10:
            continue
        top_freq = df[col].value_counts(normalize=True, dropna=False)
        if len(top_freq) > 0 and top_freq.iloc[0] >= 0.95:
            dominant = top_freq.index[0]
            pct = round(top_freq.iloc[0] * 100, 1)
            issues.append({
                "column": col,
                "issue": "suspicious_constant",
                "severity": "info",
                "description": (
                    f"Column '{col}' has {pct}% of values = '{dominant}' — "
                    f"may have no analytical value"
                ),
                "fix_code": f"df = df.drop(columns=['{col}'])",
                "auto_fixable": True,
            })
    return issues


def _detect_range_violations(df: pd.DataFrame, intelligence: dict) -> list[dict[str, Any]]:
    """Find numeric values outside reasonable ranges based on semantic type."""
    issues = []
    columns = intelligence.get("columns", {})

    for col in df.select_dtypes(include=[np.number]).columns:
        col_info = columns.get(col, {})
        sem_type = col_info.get("semantic_type", "")
        col_lower = col.lower()

        # Age checks
        if "age" in col_lower or sem_type == "age":
            invalid = df[(df[col] < 0) | (df[col] > 150)][col]
            if len(invalid) > 0:
                issues.append({
                    "column": col,
                    "issue": "range_violation",
                    "severity": "critical",
                    "description": (
                        f"Column '{col}' has {len(invalid)} values outside "
                        f"valid age range (0-150): min={invalid.min()}, max={invalid.max()}"
                    ),
                    "fix_code": (
                        f"df.loc[(df['{col}'] < 0) | (df['{col}'] > 150), '{col}'] = np.nan"
                    ),
                    "auto_fixable": True,
                })

        # Price/currency checks (negative values)
        if any(kw in col_lower for kw in ["price", "cost", "revenue", "amount", "charge", "fee"]):
            negatives = df[df[col] < 0][col]
            if len(negatives) > 0:
                issues.append({
                    "column": col,
                    "issue": "range_violation",
                    "severity": "warning",
                    "description": (
                        f"Column '{col}' has {len(negatives)} negative values — "
                        f"unusual for a price/amount field (min={negatives.min()})"
                    ),
                    "fix_code": f"df['{col}'] = df['{col}'].abs()  # or filter out",
                    "auto_fixable": False,
                })

        # Percentage checks
        if "pct" in col_lower or "percent" in col_lower or "rate" in col_lower:
            outliers = df[(df[col] < 0) | (df[col] > 100)][col]
            if len(outliers) > 0:
                issues.append({
                    "column": col,
                    "issue": "range_violation",
                    "severity": "warning",
                    "description": (
                        f"Column '{col}' has {len(outliers)} values outside "
                        f"0-100% range: min={outliers.min()}, max={outliers.max()}"
                    ),
                    "fix_code": f"df['{col}'] = df['{col}'].clip(0, 100)",
                    "auto_fixable": True,
                })

    return issues


@register_agent
class QualityAgent(BaseAgent):
    """Comprehensive data quality diagnostics agent.

    Checks for:
    - Type mismatches (numeric stored as string)
    - Inconsistent categories (case/whitespace variants)
    - Suspicious constants (95%+ same value)
    - Range violations (impossible ages, negative prices)
    - Missing data patterns
    - LLM-generated quality recommendations
    """

    name = "quality"
    model = AGENT_MODELS["reasoning"]

    def run(self, intelligence: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Run quality diagnostics.

        Args:
            intelligence: Dataset Intelligence dict.
            **kwargs:
                df (pd.DataFrame): Required — the actual DataFrame to analyze.

        Returns:
            Agent result with categorized issues and fix recommendations.
        """
        df = kwargs.get("df")
        if df is None:
            return self._make_result(
                success=False,
                error="DataFrame required for quality analysis (pass df=...)",
            )

        logger.info("Quality agent running on %d×%d dataset", *df.shape)

        # ── Run all heuristic checks ──
        all_issues: list[dict[str, Any]] = []
        all_issues.extend(_detect_type_mismatches(df))
        all_issues.extend(_detect_inconsistent_categories(df))
        all_issues.extend(_detect_suspicious_constants(df))
        all_issues.extend(_detect_range_violations(df, intelligence))

        # Add missing data issues from intelligence
        for col, info in intelligence.get("columns", {}).items():
            pct = info.get("missing_pct", 0)
            if pct > 50:
                all_issues.append({
                    "column": col,
                    "issue": "high_missing",
                    "severity": "critical",
                    "description": f"Column '{col}' is {pct}% missing — consider dropping",
                    "fix_code": f"df = df.drop(columns=['{col}'])",
                    "auto_fixable": True,
                })
            elif pct > 20:
                all_issues.append({
                    "column": col,
                    "issue": "moderate_missing",
                    "severity": "warning",
                    "description": f"Column '{col}' is {pct}% missing — needs imputation strategy",
                    "fix_code": f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())",
                    "auto_fixable": True,
                })

        # ── Categorize by severity ──
        critical = [i for i in all_issues if i["severity"] == "critical"]
        warnings = [i for i in all_issues if i["severity"] == "warning"]
        info = [i for i in all_issues if i["severity"] == "info"]

        # ── LLM-generated quality narrative ──
        narrative = self._generate_narrative(intelligence, all_issues)

        # ── Build evidence ──
        evidence = []
        for issue in all_issues:
            evidence.append(
                f"[{issue['severity'].upper()}] {issue['column']}: {issue['issue']}"
            )

        # ── Overall quality assessment ──
        quality_score = intelligence.get("quality_score", 0)
        if len(critical) > 3:
            overall = "poor"
        elif len(critical) > 0 or len(warnings) > 5:
            overall = "needs_attention"
        elif len(warnings) > 0:
            overall = "good"
        else:
            overall = "excellent"

        return self._make_result(
            success=True,
            result={
                "overall_assessment": overall,
                "total_issues": len(all_issues),
                "critical_count": len(critical),
                "warning_count": len(warnings),
                "info_count": len(info),
                "issues": all_issues,
                "critical_issues": critical,
                "warnings": warnings,
                "info_issues": info,
                "narrative": narrative,
                "auto_fixable_count": sum(1 for i in all_issues if i.get("auto_fixable")),
            },
            evidence=evidence,
            confidence=0.90,
        )

    def _generate_narrative(
        self,
        intelligence: dict[str, Any],
        issues: list[dict[str, Any]],
    ) -> str:
        """Use LLM to generate a quality narrative with recommendations."""
        if not issues:
            return "✅ Dataset passes all quality checks. No issues detected."

        issues_text = "\n".join(
            f"- [{i['severity']}] {i['column']}: {i['description']}"
            for i in issues[:15]  # Limit to avoid prompt bloat
        )

        prompt = f"""You are a data quality expert reviewing a dataset.

DATASET: {intelligence.get('dataset_type', 'unknown')} ({intelligence.get('domain', 'unknown')})
SIZE: {intelligence.get('row_count', 0)} rows × {intelligence.get('column_count', 0)} columns
QUALITY SCORE: {intelligence.get('quality_score', 0)}/100

ISSUES FOUND:
{issues_text}

Write a concise 3-5 paragraph quality report:
1. Overall assessment (one sentence)
2. Critical issues that MUST be fixed before analysis
3. Recommended fixes in priority order
4. What's good about this dataset

Be specific. Reference column names and numbers. Keep it under 200 words."""

        system = (
            "You are a data quality engineer. Write clear, actionable quality reports. "
            "Be specific about what needs fixing and why."
        )

        try:
            return query_ollama(self.model, prompt, system=system, timeout=TIMEOUT_CLOUD)
        except Exception as exc:
            logger.warning("Quality narrative generation failed: %s", exc)
            return f"Found {len(issues)} quality issues. Review the detailed list below."
