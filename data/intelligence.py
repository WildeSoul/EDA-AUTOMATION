"""
Dataset Intelligence Layer — The Brain of the Platform.

Runs automatically after file upload. Produces a structured intelligence dict
that every agent consumes. Includes:
- Column semantic classification (heuristic + LLM-enhanced)
- Dataset fingerprinting for caching
- Confidence scores on all classifications
- Quality scoring (reuses existing health module)
- LLM-generated hypotheses
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config.settings import AGENT_MODELS, TIMEOUT_CLOUD, AGENT_CACHE_TTL
from core.llm import query_ollama
from analysis.features import detect_date_columns
from analysis.health import compute_data_health_score

logger = logging.getLogger(__name__)

# ── In-memory intelligence cache ─────────────────────────────────
# {fingerprint: (timestamp, intelligence_dict)}
_intelligence_cache: dict[str, tuple[float, dict]] = {}


# ═══════════════════════════════════════════════
# Dataset Fingerprinting
# ═══════════════════════════════════════════════

def compute_fingerprint(df: pd.DataFrame) -> str:
    """Compute a stable MD5 fingerprint of a dataset's structure and content.

    Hash components:
    - Sorted column names + dtypes
    - Row count
    - First 5 rows (for content uniqueness)

    Returns:
        MD5 hex string.
    """
    parts = []
    # Schema
    for col in sorted(df.columns):
        parts.append(f"{col}:{df[col].dtype}")
    # Shape
    parts.append(f"rows:{len(df)}")
    # Content sample (first 5 rows)
    sample = df.head(5).to_csv(index=False)
    parts.append(sample)

    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()


# ═══════════════════════════════════════════════
# Heuristic Column Classification (fast, no LLM)
# ═══════════════════════════════════════════════

def _classify_columns_heuristic(df: pd.DataFrame) -> dict[str, dict]:
    """Fast heuristic classification of columns without LLM calls.

    Classifies each column into semantic types based on dtype, nunique,
    column name patterns, and value distributions.

    Returns:
        {col_name: {dtype, semantic_type, confidence, nunique, missing_pct,
                     sample_values, is_identifier, is_target_candidate, description}}
    """
    n_rows = len(df)
    date_cols = set()
    try:
        date_cols = set(detect_date_columns(df))
    except Exception:
        pass

    result = {}
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        nunique = int(df[col].nunique())
        missing_pct = round(float(df[col].isna().mean()) * 100, 2)
        col_lower = col.lower().replace("_", "").replace(" ", "")

        # Sample values (up to 5 non-null, truncated strings)
        non_null = df[col].dropna()
        samples = non_null.head(5).tolist()
        samples = [
            str(v)[:50] if isinstance(v, str) and len(str(v)) > 50 else v
            for v in samples
        ]

        # ── Heuristic semantic type detection ──
        semantic_type = "unknown"
        confidence = 0.5
        is_identifier = False
        is_target_candidate = False

        # Date columns (from existing detector)
        if col in date_cols or "datetime" in dtype_str:
            semantic_type = "date"
            confidence = 0.85

        # Identifier patterns
        elif (nunique == n_rows and n_rows > 10) or any(
            kw in col_lower for kw in ["id", "index", "key", "code"]
        ):
            if nunique >= n_rows * 0.95:
                semantic_type = "identifier"
                is_identifier = True
                confidence = 0.80

        # Binary target candidate
        elif np.issubdtype(df[col].dtype, np.number) and nunique == 2:
            semantic_type = "binary_target"
            is_target_candidate = True
            confidence = 0.65

        # Object/category with low cardinality and binary
        elif df[col].dtype == object and nunique == 2:
            semantic_type = "binary_target"
            is_target_candidate = True
            confidence = 0.60

        # Numeric
        elif np.issubdtype(df[col].dtype, np.number):
            if nunique <= 10:
                semantic_type = "categorical_encoded"
                confidence = 0.55
            else:
                semantic_type = "numeric"
                confidence = 0.75

        # Text (long strings)
        elif df[col].dtype == object and non_null.any():
            avg_len = non_null.astype(str).str.len().mean()
            if avg_len > 50:
                semantic_type = "text"
                confidence = 0.70
            elif nunique <= 20:
                semantic_type = "categorical"
                confidence = 0.75
            else:
                semantic_type = "categorical"
                confidence = 0.55

        # Categorical fallback
        elif df[col].dtype == object:
            semantic_type = "categorical"
            confidence = 0.60

        # Check common target column names (including regression targets like revenue/sales)
        if any(kw in col_lower for kw in [
            "target", "label", "class", "churn", "survived",
            "outcome", "default", "fraud", "spam", "diagnosis",
            "revenue", "sales", "price", "attrition", "y"
        ]):
            is_target_candidate = True
            confidence = max(confidence, 0.75)

        # Check geo patterns
        if any(kw in col_lower for kw in ["lat", "lng", "lon", "latitude", "longitude"]):
            semantic_type = "geo"
            confidence = 0.80

        # Check email/phone/url patterns
        if any(kw in col_lower for kw in ["email", "phone", "url", "website"]):
            semantic_type = "identifier"
            is_identifier = True
            confidence = 0.80

        # Set initial feature role
        feature_role = "feature"
        if is_identifier:
            feature_role = "ignore"
        elif is_target_candidate:
            feature_role = "target"

        result[col] = {
            "dtype": dtype_str,
            "semantic_type": semantic_type,
            "confidence": confidence,
            "description": "",  # Filled by LLM
            "is_identifier": is_identifier,
            "is_target_candidate": is_target_candidate,
            "feature_role": feature_role,
            "nunique": nunique,
            "missing_pct": missing_pct,
            "sample_values": samples,
        }

    return result


# ═══════════════════════════════════════════════
# LLM-Enhanced Classification (deep, uses DeepSeek V4)
# ═══════════════════════════════════════════════

def _generate_candidates(filename: str) -> tuple[list[str], list[str]]:
    """Generate candidate dataset types and domains based on filename keywords."""
    fn_lower = filename.lower()
    if "titanic" in fn_lower:
        return (
            ["passenger_survival", "maritime_disaster", "demographics_classification"],
            ["maritime", "historical", "safety"]
        )
    elif "churn" in fn_lower:
        return (
            ["customer_churn", "telecom_retention", "subscription_analytics"],
            ["telecom", "business", "marketing"]
        )
    elif "sales" in fn_lower:
        return (
            ["sales_transactions", "retail_sales", "ecommerce_sales", "revenue_trends"],
            ["retail", "commerce", "finance"]
        )
    elif "credit" in fn_lower or "loan" in fn_lower:
        return (
            ["credit_risk", "loan_default", "financial_risk_assessment"],
            ["finance", "banking", "risk_management"]
        )
    elif "attrition" in fn_lower:
        return (
            ["employee_attrition", "hr_analytics", "talent_retention"],
            ["hr", "corporate", "management"]
        )
    elif "patient" in fn_lower or "treatment" in fn_lower:
        return (
            ["patient_treatment", "clinical_outcomes", "healthcare_effectiveness"],
            ["healthcare", "medicine", "clinical"]
        )
    
    return (
        ["tabular_analytics", "general_dataset"],
        ["cross_domain"]
    )


def _classify_columns_llm(
    df: pd.DataFrame,
    heuristic: dict[str, dict],
    filename: str = "",
) -> tuple[dict[str, dict], str, str, list[str]]:
    """Enhance heuristic classification with LLM semantic analysis.

    Sends column metadata to DeepSeek V4 Pro for deeper understanding.

    Args:
        df: The DataFrame.
        heuristic: Output from _classify_columns_heuristic().
        filename: Optional filename context for domain matching.

    Returns:
        (updated_columns, dataset_type, domain, hypotheses)
    """
    model = AGENT_MODELS["reasoning"]

    # Generate domain & dataset type candidates
    candidate_types, candidate_domains = _generate_candidates(filename)

    # Build compact column summary table (saves tokens)
    col_lines = []
    cols_to_send = list(heuristic.keys())[:30]  # Limit to 30 columns
    for col in cols_to_send:
        info = heuristic[col]
        samples_str = ", ".join(str(s) for s in info["sample_values"][:3])
        col_lines.append(
            f"| {col} | {info['dtype']} | {info['nunique']} | "
            f"{info['missing_pct']}% | {info['semantic_type']} | {samples_str} |"
        )
    col_table = "\n".join(col_lines)

    prompt = f"""Analyze this dataset and classify each column semantically.

## DATASET: {filename} ({len(df)} rows × {len(df.columns)} columns)

## CANDIDATE SUGGESTIONS (Use these to guide your validation, but you can choose variants if needed)
- Candidate Dataset Types: {candidate_types}
- Candidate Domains: {candidate_domains}

## COLUMNS
| Column | Dtype | Unique | Missing% | Heuristic_Type | Sample_Values |
|--------|-------|--------|----------|----------------|---------------|
{col_table}

## INSTRUCTIONS
Return a JSON object with this EXACT structure (no other text):
{{
    "dataset_type": "The selected dataset type from candidates, or a specialized variant",
    "domain": "The selected industry domain from candidates, or a specialized variant",
    "columns": {{
        "ColumnName": {{
            "semantic_type": "one of: identifier, numeric, categorical, binary_target, date, text, geo, currency, percentage, ordinal, target",
            "confidence": 0.0 to 1.0,
            "description": "one-line description of what this column represents"
        }}
    }},
    "hypotheses": [
        "Testable hypothesis 1 mentioning specific column names",
        "Testable hypothesis 2 mentioning specific column names",
        "Testable hypothesis 3 mentioning specific column names"
    ]
}}

RULES:
- Classify ALL columns listed above
- confidence should reflect how certain you are (0.7+ for clear cases)
- hypotheses should be specific and testable with the data
- Return ONLY valid JSON, no markdown, no explanation"""

    system = (
        "You are a senior data scientist classifying dataset columns. "
        "Return ONLY valid JSON. Be precise about semantic types."
    )

    try:
        raw = query_ollama(model, prompt, system=system, timeout=TIMEOUT_CLOUD)
        logger.debug("LLM classification raw response: %s", raw[:200])

        # Extract JSON from response (handle markdown code blocks)
        json_str = raw.strip()
        if "```" in json_str:
            # Extract from code block
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = json_str[start:end]

        llm_result = json.loads(json_str)

        # Merge LLM output with heuristic
        llm_columns = llm_result.get("columns", {})
        updated = dict(heuristic)

        for col, llm_info in llm_columns.items():
            if col in updated:
                llm_conf = float(llm_info.get("confidence", 0.5))
                heuristic_type = updated[col]["semantic_type"]
                llm_type = llm_info.get("semantic_type", heuristic_type)

                # Confidence Calibration & Semantic Typing Validation
                if llm_conf > 0.7:
                    updated[col]["semantic_type"] = llm_type
                    
                    # Calibrate confidence
                    if llm_type == heuristic_type:
                        updated[col]["confidence"] = 0.95
                    else:
                        updated[col]["confidence"] = 0.85
                else:
                    if llm_type == heuristic_type:
                        updated[col]["confidence"] = 0.80
                    else:
                        updated[col]["confidence"] = 0.70

                # Always take the description
                desc = llm_info.get("description", "")
                if desc:
                    updated[col]["description"] = desc

                # Feature Role Assignment & Type flags update
                final_type = updated[col]["semantic_type"]
                if final_type in ("target", "binary_target"):
                    updated[col]["is_target_candidate"] = True
                    updated[col]["feature_role"] = "target"
                elif final_type == "identifier":
                    updated[col]["is_identifier"] = True
                    updated[col]["feature_role"] = "ignore"
                else:
                    updated[col]["feature_role"] = "feature"

        dataset_type = llm_result.get("dataset_type", "unknown")
        domain = llm_result.get("domain", "unknown")
        hypotheses = llm_result.get("hypotheses", [])

        # Validate dataset_type and domain match candidates
        if dataset_type == "unknown" or not dataset_type:
            dataset_type = candidate_types[0]
        if domain == "unknown" or not domain:
            domain = candidate_domains[0]

        logger.info(
            "LLM classification: type=%s, domain=%s, %d hypotheses",
            dataset_type, domain, len(hypotheses),
        )
        return updated, dataset_type, domain, hypotheses

    except Exception as exc:
        logger.warning("LLM classification failed, using heuristic fallback: %s", exc)
        fallback_type = candidate_types[0]
        fallback_domain = candidate_domains[0]
        
        fallback_hyps = []
        if "titanic" in filename.lower():
            fallback_hyps = ["Passengers in higher classes had higher survival rates."]
        elif "churn" in filename.lower():
            fallback_hyps = ["Customers with longer tenure are less likely to churn."]
        elif "sales" in filename.lower():
            fallback_hyps = ["Total sales revenue peaks during weekends or holidays."]
        elif "credit" in filename.lower() or "loan" in filename.lower():
            fallback_hyps = ["Borrowers with lower credit scores default more frequently."]
        elif "attrition" in filename.lower():
            fallback_hyps = ["Employees with lower monthly income have higher attrition rates."]
        elif "patient" in filename.lower():
            fallback_hyps = ["Certain treatment types correlate with a higher recovery outcome."]
        else:
            fallback_hyps = ["Variables show significant pairwise correlations."]

        # Calibrate heuristic-only confidences
        updated = dict(heuristic)
        for col in updated:
            updated[col]["confidence"] = min(updated[col]["confidence"], 0.70)
            
        return updated, fallback_type, fallback_domain, fallback_hyps


# ═══════════════════════════════════════════════
# Quality Grading
# ═══════════════════════════════════════════════

def _compute_quality_grade(score: int) -> str:
    """Convert numeric quality score (0-100) to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def _extract_quality_issues(df: pd.DataFrame, columns: dict) -> list[str]:
    """Extract human-readable quality issues from column metadata."""
    issues = []

    for col, info in columns.items():
        pct = info["missing_pct"]
        if pct > 50:
            issues.append(f"⚠️ {col}: {pct}% missing — consider dropping")
        elif pct > 10:
            issues.append(f"⚠️ {col}: {pct}% missing — needs imputation")
        elif pct > 0:
            issues.append(f"ℹ️ {col}: {pct}% missing — minor")

    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues.append(f"⚠️ {col}: constant value — no analytical value")

    # Check for high cardinality categoricals
    for col, info in columns.items():
        if info["semantic_type"] == "categorical" and info["nunique"] > 50:
            issues.append(
                f"ℹ️ {col}: high cardinality ({info['nunique']} unique) — "
                f"consider grouping"
            )

    return issues


# ═══════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════

def build_dataset_intelligence(
    df: pd.DataFrame,
    filename: str,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Build the complete Dataset Intelligence dict.

    This is the BRAIN of the platform. Every agent consumes this output.

    Args:
        df: The uploaded DataFrame.
        filename: Original filename.
        use_cache: If True, return cached result for identical datasets.

    Returns:
        Full intelligence dict with schema, quality, hypotheses, and metadata.
    """
    start_time = time.time()
    logger.info("Building dataset intelligence for '%s' (%d×%d)", filename, *df.shape)

    # ── Step 1: Fingerprint ──
    fingerprint = compute_fingerprint(df)
    logger.debug("Fingerprint: %s", fingerprint)

    # ── Step 2: Cache check ──
    if use_cache and fingerprint in _intelligence_cache:
        ts, cached = _intelligence_cache[fingerprint]
        if time.time() - ts < AGENT_CACHE_TTL:
            logger.info("Intelligence cache HIT for %s (%.1fs old)", filename, time.time() - ts)
            return cached
        else:
            del _intelligence_cache[fingerprint]

    # ── Step 3: Heuristic classification ──
    columns = _classify_columns_heuristic(df)

    # ── Step 4: LLM-enhanced classification ──
    columns, dataset_type, domain, hypotheses = _classify_columns_llm(df, columns, filename)

    # ── Step 5: Compute quality score ──
    try:
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        categorical_cols = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
        health_result = compute_data_health_score(df, numeric_cols, categorical_cols)
        quality_score = health_result.get("total_score", 0) if isinstance(health_result, dict) else 0
    except Exception as exc:
        logger.warning("Health score computation failed: %s", exc)
        quality_score = 0

    quality_grade = _compute_quality_grade(quality_score)
    quality_issues = _extract_quality_issues(df, columns)

    # ── Step 6: Categorize columns ──
    id_columns = [c for c, i in columns.items() if i["is_identifier"]]
    target_candidates = [c for c, i in columns.items() if i["is_target_candidate"]]
    date_columns = [c for c, i in columns.items() if i["semantic_type"] == "date"]
    text_columns = [c for c, i in columns.items() if i["semantic_type"] == "text"]
    numeric_columns = [
        c for c, i in columns.items()
        if i["semantic_type"] in ("numeric", "currency", "percentage")
        or (np.issubdtype(df[c].dtype, np.number) and not i["is_identifier"])
    ]
    categorical_columns = [
        c for c, i in columns.items()
        if i["semantic_type"] in ("categorical", "ordinal", "categorical_encoded")
    ]

    # ── Step 7: Build final dict ──
    intelligence: dict[str, Any] = {
        "filename": filename,
        "dataset_fingerprint": fingerprint,
        "dataset_type": dataset_type,
        "domain": domain,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "target_candidates": target_candidates,
        "id_columns": id_columns,
        "date_columns": date_columns,
        "text_columns": text_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "quality_score": quality_score,
        "quality_grade": quality_grade,
        "quality_issues": quality_issues,
        "hypotheses": hypotheses,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Step 8: Cache result ──
    _intelligence_cache[fingerprint] = (time.time(), intelligence)

    elapsed = time.time() - start_time
    logger.info(
        "Intelligence built for '%s': type=%s, domain=%s, quality=%s (%d), "
        "%d hypotheses, %.1fs elapsed",
        filename, dataset_type, domain, quality_grade, quality_score,
        len(hypotheses), elapsed,
    )

    return intelligence
