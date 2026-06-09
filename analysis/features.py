"""
Automated Feature Engineering — date extraction, interactions, skewness, imputation.
"""
import numpy as np
import pandas as pd


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Auto-detect columns that look like dates."""
    date_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(50)
            try:
                parsed = pd.to_datetime(sample, format="mixed", dayfirst=False)
                if parsed.notna().sum() > len(sample) * 0.7:
                    date_cols.append(col)
            except (ValueError, TypeError):
                continue
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
    return date_cols


def engineer_date_features(df: pd.DataFrame, date_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Extract temporal features from date columns."""
    new_features = []
    for col in date_cols:
        dt = pd.to_datetime(df[col], errors="coerce")
        prefix = col.replace(" ", "_")
        df[f"{prefix}_year"] = dt.dt.year
        df[f"{prefix}_month"] = dt.dt.month
        df[f"{prefix}_day"] = dt.dt.day
        df[f"{prefix}_dayofweek"] = dt.dt.dayofweek
        df[f"{prefix}_quarter"] = dt.dt.quarter
        df[f"{prefix}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        new_features.extend([
            f"{prefix}_year", f"{prefix}_month", f"{prefix}_day",
            f"{prefix}_dayofweek", f"{prefix}_quarter", f"{prefix}_is_weekend",
        ])
    return df, new_features


def engineer_interaction_features(
    df: pd.DataFrame, numeric_cols: list[str], top_n: int = 3
) -> tuple[pd.DataFrame, list[str]]:
    """Create interaction features from top correlated numeric pairs."""
    new_features = []
    if len(numeric_cols) < 2:
        return df, new_features

    corr_matrix = df[numeric_cols].corr().abs()
    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            pairs.append((c1, c2, corr_matrix.loc[c1, c2]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for c1, c2, _ in pairs[:top_n]:
        feat_name = f"{c1}_x_{c2}"
        df[feat_name] = df[c1] * df[c2]
        new_features.append(feat_name)
        if (df[c2] != 0).all():
            ratio_name = f"{c1}_div_{c2}"
            df[ratio_name] = df[c1] / df[c2].replace(0, np.nan)
            new_features.append(ratio_name)

    return df, new_features


def compute_skewness(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Identify skewed features and suggest transforms."""
    skew_data = []
    for col in numeric_cols:
        skew_val = df[col].skew()
        if abs(skew_val) > 1:
            direction = "right-skewed" if skew_val > 0 else "left-skewed"
            suggestion = "log transform" if skew_val > 1 else "square transform"
            skew_data.append({
                "Column": col,
                "Skewness": round(skew_val, 3),
                "Direction": direction,
                "Suggested Fix": suggestion,
            })
    return pd.DataFrame(skew_data) if skew_data else pd.DataFrame()


def impute_missing(
    df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Impute missing values — median for numeric, mode for categorical."""
    report = {}
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report[col] = f"filled {n_missing} NaN → median ({median_val:.2f})"
    for col in categorical_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            report[col] = f"filled {n_missing} NaN → mode ('{mode_val}')"
    return df, report


def detect_high_cardinality(
    df: pd.DataFrame,
    categorical_cols: list[str],
    threshold: float = 0.10,
) -> list[dict]:
    """
    Detect categorical columns with dangerously high cardinality.

    Flags columns where unique values > threshold * total_rows.
    This prevents memory crashes from One-Hot Encoding and the
    'Curse of Dimensionality.'

    Returns list of dicts: {column, unique_count, total_rows, ratio, recommendation}
    """
    n_rows = len(df)
    if n_rows == 0:
        return []

    flagged = []
    for col in categorical_cols:
        n_unique = df[col].nunique()
        ratio = n_unique / n_rows

        if ratio > threshold:
            # Determine recommendation
            if ratio > 0.90:
                recommendation = "🔴 Drop — likely an ID/hash column"
                severity = "critical"
            elif ratio > 0.50:
                recommendation = "🟡 Feature Hash — use hashing trick to reduce dimensions"
                severity = "high"
            else:
                recommendation = "🔵 Target Encode — encode using target variable statistics"
                severity = "medium"

            flagged.append({
                "column": col,
                "unique_count": n_unique,
                "total_rows": n_rows,
                "ratio": round(ratio, 4),
                "ratio_pct": f"{ratio:.1%}",
                "recommendation": recommendation,
                "severity": severity,
            })

    # Sort by ratio descending (worst offenders first)
    flagged.sort(key=lambda x: x["ratio"], reverse=True)
    return flagged
