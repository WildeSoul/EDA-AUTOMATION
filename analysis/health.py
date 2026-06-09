"""
Data Health Score — quantifies data readiness for ML on a 0-100 scale.

Scoring Breakdown:
  Completeness  (25 pts): % non-null cells
  Uniqueness    (20 pts): low duplicates + reasonable cardinality
  Consistency   (20 pts): type homogeneity, valid ranges
  Distribution  (20 pts): normality, low extreme skewness
  Outlier Ratio (15 pts): % of outliers via IQR
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _score_completeness(df: pd.DataFrame) -> tuple[float, list[str]]:
    """Score based on percentage of non-null cells. Max 25 points."""
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0, ["Empty dataset"]

    non_null = df.notnull().sum().sum()
    pct_complete = non_null / total_cells

    score = round(pct_complete * 25, 1)
    notes = []

    if pct_complete >= 0.99:
        notes.append("Excellent — less than 1% missing data")
    elif pct_complete >= 0.95:
        notes.append(f"{(1 - pct_complete) * 100:.1f}% missing — minor imputation needed")
    elif pct_complete >= 0.80:
        notes.append(f"{(1 - pct_complete) * 100:.1f}% missing — significant imputation required")
    else:
        notes.append(f"{(1 - pct_complete) * 100:.1f}% missing — critical data gaps, consider dropping sparse columns")

    # Flag columns with >50% missing
    col_missing = df.isnull().mean()
    bad_cols = col_missing[col_missing > 0.50].index.tolist()
    if bad_cols:
        notes.append(f"Columns with >50% missing: {', '.join(bad_cols[:5])}")

    return score, notes


def _score_uniqueness(df: pd.DataFrame) -> tuple[float, list[str]]:
    """Score based on duplicate rows and cardinality issues. Max 20 points."""
    notes = []
    n_rows = len(df)
    if n_rows == 0:
        return 0.0, ["Empty dataset"]

    # Duplicate rows penalty (10 pts)
    dup_count = df.duplicated().sum()
    dup_pct = dup_count / n_rows
    dup_score = max(0, 10 * (1 - dup_pct * 5))  # 20% dups → 0 pts

    if dup_pct > 0.10:
        notes.append(f"⚠️ {dup_pct:.1%} duplicate rows — consider deduplication")
    elif dup_pct > 0:
        notes.append(f"{dup_count} duplicate rows ({dup_pct:.1%})")
    else:
        notes.append("No duplicate rows — clean")

    # Cardinality reasonableness (10 pts)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    card_score = 10.0
    for col in cat_cols:
        ratio = df[col].nunique() / n_rows
        if ratio > 0.90:  # Likely an ID column
            card_score -= 2
            notes.append(f"⚠️ '{col}' looks like an ID (uniqueness: {ratio:.0%})")
    card_score = max(0, card_score)

    return round(dup_score + card_score, 1), notes


def _score_consistency(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[float, list[str]]:
    """Score based on type homogeneity and valid ranges. Max 20 points."""
    notes = []
    n_cols = len(df.columns)
    if n_cols == 0:
        return 0.0, ["No columns"]

    # Mixed-type check (10 pts)
    mixed_count = 0
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(100)
            numeric_count = pd.to_numeric(sample, errors="coerce").notna().sum()
            if 0 < numeric_count < len(sample) * 0.8:
                mixed_count += 1
                if mixed_count <= 3:
                    notes.append(f"⚠️ '{col}' has mixed types (numbers + text)")

    type_score = max(0, 10 - mixed_count * 2.5)

    # Constant columns check (5 pts)
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    const_score = max(0, 5 - len(constant_cols) * 1.5)
    if constant_cols:
        notes.append(f"Constant columns (zero variance): {', '.join(constant_cols[:3])}")

    # Negative values in typically-positive columns (5 pts)
    neg_score = 5.0
    for col in numeric_cols:
        if any(kw in col.lower() for kw in ["age", "price", "count", "quantity", "salary", "income"]):
            if (df[col] < 0).any():
                neg_score -= 1.5
                notes.append(f"⚠️ '{col}' has negative values (unexpected)")
    neg_score = max(0, neg_score)

    if not notes:
        notes.append("Data types are consistent and clean")

    return round(type_score + const_score + neg_score, 1), notes


def _score_distribution(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[float, list[str]]:
    """Score based on normality and skewness. Max 20 points."""
    notes = []
    if not numeric_cols:
        return 20.0, ["No numeric columns to test"]

    # Skewness check (10 pts)
    skew_vals = df[numeric_cols].skew()
    extreme_skew = (skew_vals.abs() > 2).sum()
    moderate_skew = ((skew_vals.abs() > 1) & (skew_vals.abs() <= 2)).sum()
    skew_penalty = extreme_skew * 2 + moderate_skew * 0.5
    skew_score = max(0, 10 - skew_penalty)

    if extreme_skew > 0:
        worst = skew_vals.abs().idxmax()
        notes.append(f"{extreme_skew} features have extreme skew (worst: '{worst}' = {skew_vals[worst]:.1f})")
    elif moderate_skew > 0:
        notes.append(f"{moderate_skew} moderately skewed features — log/sqrt transform recommended")
    else:
        notes.append("Distributions are well-balanced")

    # Normality check via Shapiro on a sample (10 pts)
    normal_count = 0
    tested = 0
    for col in numeric_cols[:10]:  # Test up to 10 columns
        data = df[col].dropna()
        if len(data) < 8:
            continue
        sample = data.sample(min(2000, len(data)), random_state=42)
        try:
            _, p = scipy_stats.shapiro(sample)
            tested += 1
            if p > 0.05:
                normal_count += 1
        except Exception:
            continue

    if tested > 0:
        norm_ratio = normal_count / tested
        norm_score = round(norm_ratio * 10, 1)
        notes.append(f"{normal_count}/{tested} features pass normality test")
    else:
        norm_score = 5.0  # Neutral

    return round(skew_score + norm_score, 1), notes


def _score_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[float, list[str]]:
    """Score based on outlier ratio via IQR. Max 15 points."""
    notes = []
    if not numeric_cols:
        return 15.0, ["No numeric columns to check"]

    total_outliers = 0
    total_values = 0
    worst_col = None
    worst_pct = 0

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 10:
            continue

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        n_outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
        total_outliers += n_outliers
        total_values += len(data)

        pct = n_outliers / len(data)
        if pct > worst_pct:
            worst_pct = pct
            worst_col = col

    if total_values == 0:
        return 15.0, ["No outlier analysis possible"]

    outlier_ratio = total_outliers / total_values
    score = max(0, 15 * (1 - outlier_ratio * 10))  # 10% outliers → 0 pts

    if outlier_ratio > 0.05:
        notes.append(f"⚠️ {outlier_ratio:.1%} outlier ratio — worst: '{worst_col}' ({worst_pct:.1%})")
    elif outlier_ratio > 0.01:
        notes.append(f"{outlier_ratio:.1%} outlier ratio — within acceptable range")
    else:
        notes.append("Very low outlier ratio — clean data")

    return round(score, 1), notes


def compute_data_health_score(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict:
    """
    Compute a comprehensive data health score (0-100).

    Returns dict with:
        total_score: int (0-100)
        grade: str (A-F)
        breakdown: dict of {category: {score, max, notes}}
        recommendations: list[str]
    """
    completeness_score, completeness_notes = _score_completeness(df)
    uniqueness_score, uniqueness_notes = _score_uniqueness(df)
    consistency_score, consistency_notes = _score_consistency(df, numeric_cols)
    distribution_score, distribution_notes = _score_distribution(df, numeric_cols)
    outlier_score, outlier_notes = _score_outliers(df, numeric_cols)

    total = round(
        completeness_score + uniqueness_score + consistency_score
        + distribution_score + outlier_score,
        0,
    )
    total = int(min(100, max(0, total)))

    # Letter grade
    if total >= 90:
        grade = "A"
    elif total >= 80:
        grade = "B"
    elif total >= 65:
        grade = "C"
    elif total >= 50:
        grade = "D"
    else:
        grade = "F"

    # Recommendations based on weakest areas
    recommendations = []
    scores = [
        ("Completeness", completeness_score, 25),
        ("Uniqueness", uniqueness_score, 20),
        ("Consistency", consistency_score, 20),
        ("Distribution", distribution_score, 20),
        ("Outlier Ratio", outlier_score, 15),
    ]
    scores.sort(key=lambda x: x[1] / x[2])  # Sort by ratio (weakest first)

    for name, sc, mx in scores:
        ratio = sc / mx if mx > 0 else 1
        if ratio < 0.6:
            if name == "Completeness":
                recommendations.append("Impute or drop columns with high missing rates")
            elif name == "Uniqueness":
                recommendations.append("Remove duplicate rows and drop ID-like columns")
            elif name == "Consistency":
                recommendations.append("Fix mixed-type columns and remove constant features")
            elif name == "Distribution":
                recommendations.append("Apply log/sqrt transforms to highly skewed features")
            elif name == "Outlier Ratio":
                recommendations.append("Investigate outliers — clip, transform, or remove if errors")

    if not recommendations:
        recommendations.append("Data is ML-ready! Consider feature engineering for further gains.")

    return {
        "total_score": total,
        "grade": grade,
        "breakdown": {
            "Completeness": {
                "score": completeness_score, "max": 25,
                "notes": completeness_notes,
            },
            "Uniqueness": {
                "score": uniqueness_score, "max": 20,
                "notes": uniqueness_notes,
            },
            "Consistency": {
                "score": consistency_score, "max": 20,
                "notes": consistency_notes,
            },
            "Distribution": {
                "score": distribution_score, "max": 20,
                "notes": distribution_notes,
            },
            "Outlier Ratio": {
                "score": outlier_score, "max": 15,
                "notes": outlier_notes,
            },
        },
        "recommendations": recommendations,
    }
