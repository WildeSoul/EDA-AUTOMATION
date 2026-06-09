"""
EDA Agent — Statistical analysis wrapper around existing modules.

Delegates to existing analysis/ functions and enhances results with
LLM-powered interpretation. Wraps:
- analysis.stats (correlations, normality, chi-square, PCA, clustering)
- analysis.features (skewness, date features, interactions)
- analysis.anomaly (isolation forest, z-score)
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np
import pandas as pd

from agents import BaseAgent, register_agent
from config.settings import AGENT_MODELS, TIMEOUT_CLOUD
from core.llm import query_ollama
from analysis.stats import (
    compute_correlation_with_pvalues,
    run_normality_tests,
    run_chi_square_tests,
)
from analysis.features import compute_skewness

logger = logging.getLogger(__name__)


@register_agent
class EDAAgent(BaseAgent):
    """Statistical analysis agent that wraps existing EDA modules.

    Runs correlations, normality tests, chi-square tests, and skewness
    analysis. Uses DeepSeek V4 Pro to interpret results in business context.
    """

    name = "eda"
    model = AGENT_MODELS["reasoning"]

    def run(self, intelligence: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Run statistical analysis.

        Args:
            intelligence: Dataset Intelligence dict.
            **kwargs:
                df (pd.DataFrame): Required — the actual DataFrame.
                focus (str): Optional — specific analysis focus
                    ('correlations', 'normality', 'distributions', 'all').
                hypotheses (list[dict]): Optional — hypotheses to test.

        Returns:
            Agent result with statistical findings and LLM interpretation.
        """
        df = kwargs.get("df")
        if df is None:
            return self._make_result(
                success=False,
                error="DataFrame required for EDA (pass df=...)",
            )

        focus = kwargs.get("focus", "all")
        hypotheses = kwargs.get("hypotheses", [])

        numeric_cols = intelligence.get("numeric_columns", [])
        categorical_cols = intelligence.get("categorical_columns", [])

        # Filter to columns actually in the DataFrame
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        categorical_cols = [c for c in categorical_cols if c in df.columns]

        findings: dict[str, Any] = {}
        evidence: list[str] = []

        # ── Correlation Analysis ──
        if focus in ("all", "correlations") and len(numeric_cols) >= 2:
            try:
                corr_result = compute_correlation_with_pvalues(df, numeric_cols)
                if isinstance(corr_result, dict):
                    findings["correlations"] = corr_result
                    # Extract top correlations for evidence
                    top_corrs = corr_result.get("top_correlations", [])
                    for tc in top_corrs[:5]:
                        evidence.append(
                            f"Correlation: {tc.get('col1', '?')} ↔ {tc.get('col2', '?')} "
                            f"= {tc.get('correlation', 0):.3f} "
                            f"(p={tc.get('p_value', 'N/A')})"
                        )
                elif isinstance(corr_result, pd.DataFrame):
                    findings["correlation_matrix"] = corr_result.to_dict()
                    evidence.append(f"Correlation matrix: {len(numeric_cols)} numeric columns")
            except Exception as exc:
                logger.warning("Correlation analysis failed: %s", exc)
                findings["correlations_error"] = str(exc)

        # ── Normality Tests ──
        if focus in ("all", "normality") and numeric_cols:
            try:
                norm_results = run_normality_tests(df, numeric_cols[:10])
                findings["normality"] = norm_results
                if isinstance(norm_results, list):
                    non_normal = [r for r in norm_results if not r.get("is_normal", True)]
                    evidence.append(
                        f"Normality: {len(non_normal)}/{len(norm_results)} columns "
                        f"are non-normal (Shapiro-Wilk)"
                    )
                elif isinstance(norm_results, dict):
                    evidence.append("Normality tests completed")
            except Exception as exc:
                logger.warning("Normality tests failed: %s", exc)
                findings["normality_error"] = str(exc)

        # ── Chi-Square Tests ──
        if focus in ("all", "categorical") and len(categorical_cols) >= 2:
            try:
                chi2_results = run_chi_square_tests(df, categorical_cols[:10])
                findings["chi_square"] = chi2_results
                if isinstance(chi2_results, list):
                    significant = [r for r in chi2_results if r.get("p_value", 1) < 0.05]
                    evidence.append(
                        f"Chi-square: {len(significant)}/{len(chi2_results)} pairs "
                        f"have significant association (p<0.05)"
                    )
            except Exception as exc:
                logger.warning("Chi-square tests failed: %s", exc)
                findings["chi_square_error"] = str(exc)

        # ── Skewness Analysis ──
        if focus in ("all", "distributions") and numeric_cols:
            try:
                skew_df = compute_skewness(df, numeric_cols)
                if skew_df is not None and isinstance(skew_df, pd.DataFrame):
                    findings["skewness"] = skew_df.to_dict(orient="records")
                    highly_skewed = skew_df[skew_df["skewness"].abs() > 1] if "skewness" in skew_df.columns else pd.DataFrame()
                    if len(highly_skewed) > 0:
                        evidence.append(
                            f"Skewness: {len(highly_skewed)} columns highly skewed (|skew| > 1)"
                        )
            except Exception as exc:
                logger.warning("Skewness analysis failed: %s", exc)
                findings["skewness_error"] = str(exc)

        # ── Basic descriptive stats ──
        try:
            if numeric_cols:
                desc = df[numeric_cols].describe().to_dict()
                findings["descriptive_stats"] = desc
                evidence.append(f"Descriptive stats computed for {len(numeric_cols)} numeric columns")
        except Exception as exc:
            logger.warning("Descriptive stats failed: %s", exc)

        # ── LLM Interpretation ──
        interpretation = self._interpret_findings(intelligence, findings, evidence, hypotheses)
        findings["interpretation"] = interpretation

        return self._make_result(
            success=True,
            result=findings,
            evidence=evidence,
            confidence=0.85,
        )

    def _interpret_findings(
        self,
        intelligence: dict,
        findings: dict,
        evidence: list[str],
        hypotheses: list[dict],
    ) -> str:
        """Use LLM to interpret statistical findings in business context."""
        evidence_text = "\n".join(f"- {e}" for e in evidence[:20])

        hypothesis_text = ""
        if hypotheses:
            h_lines = [
                f"- {h.get('id', '?')}: {h.get('statement', '?')}"
                for h in hypotheses[:5]
            ]
            hypothesis_text = "\n\n## HYPOTHESES TO EVALUATE\n" + "\n".join(h_lines)

        prompt = f"""You are interpreting EDA results for a {intelligence.get('dataset_type', 'unknown')} dataset.

## STATISTICAL EVIDENCE
{evidence_text}
{hypothesis_text}

## TASK
Write a concise interpretation (150-200 words):
1. Key statistical patterns found
2. Which hypotheses are supported/rejected by the evidence (if any)
3. Surprising or unexpected findings
4. Business implications
5. What to investigate next

Be specific — cite numbers from the evidence."""

        system = (
            "You are a senior data scientist interpreting statistical results. "
            "Be specific, cite exact numbers, and provide actionable business insights."
        )

        try:
            return query_ollama(self.model, prompt, system=system, timeout=TIMEOUT_CLOUD)
        except Exception as exc:
            logger.warning("EDA interpretation failed: %s", exc)
            return "Statistical analysis completed. See evidence list for key findings."
