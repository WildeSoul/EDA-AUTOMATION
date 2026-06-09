"""
Build rich data context strings for LLM consumption.
"""
import re
import numpy as np
import pandas as pd


def build_data_context(
    df: pd.DataFrame,
    uploaded_name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    engineered_features: list[str] | None = None,
    imputation_report: dict | None = None,
    skew_df: pd.DataFrame | None = None,
    date_cols: list[str] | None = None,
    user_question: str = "",
) -> str:
    """Build a comprehensive data context string for the LLM."""
    ctx = []
    ctx.append(f"DATASET: {uploaded_name}")
    ctx.append(f"SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    ctx.append(f"COLUMNS & TYPES: {dict(zip(df.columns, df.dtypes.astype(str)))}")

    # Date range
    if date_cols:
        for dc in date_cols:
            dt_series = pd.to_datetime(df[dc], errors="coerce")
            ctx.append(f"DATE RANGE ({dc}): {dt_series.min()} to {dt_series.max()}")

    # Descriptive statistics
    if numeric_cols:
        ctx.append(f"\n--- NUMERICAL STATISTICS ---\n{df[numeric_cols].describe().to_string()}")

    # Categorical breakdown
    if categorical_cols:
        ctx.append("\n--- CATEGORICAL BREAKDOWN ---")
        for c in categorical_cols[:8]:
            vc = df[c].value_counts().head(7).to_dict()
            ctx.append(f"  '{c}': {vc}")

    # Top correlations
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack().reset_index()
        )
        pairs.columns = ["Var1", "Var2", "Correlation"]
        top_pos = pairs.nlargest(5, "Correlation")
        top_neg = pairs.nsmallest(5, "Correlation")
        ctx.append("\n--- TOP POSITIVE CORRELATIONS ---")
        for _, r in top_pos.iterrows():
            ctx.append(f"  {r.Var1} ↔ {r.Var2}: {r.Correlation:.3f}")
        ctx.append("\n--- TOP NEGATIVE CORRELATIONS ---")
        for _, r in top_neg.iterrows():
            ctx.append(f"  {r.Var1} ↔ {r.Var2}: {r.Correlation:.3f}")

    # Outlier counts (IQR)
    if numeric_cols:
        ctx.append("\n--- OUTLIER COUNTS (IQR method) ---")
        for c in numeric_cols:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                out_count = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
                if out_count > 0:
                    ctx.append(f"  {c}: {out_count} outliers ({out_count / len(df) * 100:.1f}%)")

    # Missing & duplicates
    ctx.append(f"\nMISSING VALUES: {int(df.isnull().sum().sum())} total")
    per_col_missing = df.isnull().sum()
    per_col_missing = per_col_missing[per_col_missing > 0]
    if not per_col_missing.empty:
        ctx.append(f"  Per column: {per_col_missing.to_dict()}")
    ctx.append(f"DUPLICATE ROWS: {int(df.duplicated().sum())}")

    # Engineered features
    if engineered_features:
        ctx.append(f"\n--- ENGINEERED FEATURES ({len(engineered_features)} new) ---")
        ctx.append(f"  {', '.join(engineered_features)}")

    # Imputation report
    if imputation_report:
        ctx.append("\n--- DATA IMPUTATION PERFORMED ---")
        for col, action in imputation_report.items():
            ctx.append(f"  {col}: {action}")

    # Skewness
    if skew_df is not None and not skew_df.empty:
        ctx.append("\n--- SKEWED FEATURES ---")
        for _, row in skew_df.iterrows():
            ctx.append(
                f"  {row['Column']}: skewness={row['Skewness']} "
                f"({row['Direction']}) → {row['Suggested Fix']}"
            )

    # Relevant rows for the question
    if user_question:
        numbers_in_q = re.findall(r"[\d,]+\.?\d*", user_question.replace(",", ""))
        numbers_in_q = [float(n) for n in numbers_in_q if n]
        keywords = [w for w in user_question.lower().split() if len(w) > 2]

        relevant_rows = pd.DataFrame()
        if numbers_in_q and numeric_cols:
            target_val = max(numbers_in_q)
            for nc in numeric_cols:
                mask = (df[nc] >= target_val * 0.8) & (df[nc] <= target_val * 1.2)
                matched = df[mask]
                if len(matched) > 0:
                    relevant_rows = matched.head(10)
                    ctx.append(
                        f"\n--- ROWS MATCHING ~{target_val:,.0f} (±20%) in "
                        f"'{nc}' ({len(matched)} total) ---"
                    )
                    ctx.append(relevant_rows.to_string(index=False))
                    break

        if relevant_rows.empty and keywords and categorical_cols:
            for cat_col in categorical_cols:
                str_vals = df[cat_col].astype(str).str.lower()
                for kw in keywords:
                    mask = str_vals.str.contains(kw, na=False)
                    if mask.sum() > 0:
                        relevant_rows = df[mask].head(10)
                        ctx.append(
                            f"\n--- ROWS MATCHING keyword '{kw}' in "
                            f"'{cat_col}' ({mask.sum()} total) ---"
                        )
                        ctx.append(relevant_rows.to_string(index=False))
                        break
                if not relevant_rows.empty:
                    break

        if relevant_rows.empty:
            sample = df.sample(min(10, len(df)), random_state=42)
            ctx.append(f"\n--- SAMPLE ROWS (random 10) ---\n{sample.to_string(index=False)}")

    return "\n".join(ctx)
