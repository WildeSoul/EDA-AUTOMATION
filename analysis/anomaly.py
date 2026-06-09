"""
Anomaly Detection — Isolation Forest, Z-Score, IQR multivariate.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config.settings import ANOMALY_CONTAMINATION_DEFAULT, ANOMALY_ZSCORE_THRESHOLD


def run_isolation_forest(
    df: pd.DataFrame,
    numeric_cols: list[str],
    contamination: float = ANOMALY_CONTAMINATION_DEFAULT,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run Isolation Forest on numeric columns.
    Returns (df_with_labels, anomaly_mask, anomaly_scores).
    """
    clean_df = df[numeric_cols].dropna()
    if len(clean_df) < 10 or len(numeric_cols) < 1:
        return df, np.array([]), np.array([])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df)

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
    )
    predictions = iso_forest.fit_predict(scaled)
    scores = iso_forest.decision_function(scaled)

    # -1 = anomaly, 1 = normal → convert to boolean mask
    anomaly_mask = predictions == -1

    # Add labels back to original df (aligned by index)
    result_df = df.copy()
    result_df.loc[clean_df.index, "anomaly_label"] = np.where(anomaly_mask, "🔴 Anomaly", "🟢 Normal")
    result_df.loc[clean_df.index, "anomaly_score"] = scores

    return result_df, anomaly_mask, scores


def run_zscore_detection(
    df: pd.DataFrame,
    numeric_cols: list[str],
    threshold: float = ANOMALY_ZSCORE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag rows where ANY numeric feature exceeds Z-score threshold.
    Returns (summary_df, anomalous_rows).
    """
    if not numeric_cols:
        return pd.DataFrame(), pd.DataFrame()

    clean_df = df[numeric_cols].dropna()
    z_scores = np.abs((clean_df - clean_df.mean()) / clean_df.std())

    # Per-column anomaly counts
    summary = []
    for col in numeric_cols:
        count = (z_scores[col] > threshold).sum()
        if count > 0:
            summary.append({
                "Column": col,
                "Anomalies": count,
                "% of Data": round(count / len(clean_df) * 100, 2),
                "Max Z-Score": round(z_scores[col].max(), 2),
            })

    # Rows where any feature is anomalous
    row_mask = (z_scores > threshold).any(axis=1)
    anomalous_rows = df.loc[clean_df.index[row_mask]]

    return (
        pd.DataFrame(summary) if summary else pd.DataFrame(),
        anomalous_rows,
    )


def run_iqr_multivariate(
    df: pd.DataFrame,
    numeric_cols: list[str],
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multivariate IQR anomaly detection — flag rows exceeding IQR bounds
    in multiple features simultaneously.
    Returns (summary_df, anomalous_rows).
    """
    if not numeric_cols:
        return pd.DataFrame(), pd.DataFrame()

    clean_df = df[numeric_cols].dropna()
    outlier_counts = pd.Series(0, index=clean_df.index)

    summary = []
    for col in numeric_cols:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = (clean_df[col] < lower) | (clean_df[col] > upper)
        outlier_counts += mask.astype(int)
        count = mask.sum()
        if count > 0:
            summary.append({
                "Column": col,
                "Outliers": count,
                "Lower Bound": round(lower, 2),
                "Upper Bound": round(upper, 2),
                "% Outliers": round(count / len(clean_df) * 100, 2),
            })

    # Rows that are outliers in 2+ features = multivariate anomalies
    multi_mask = outlier_counts >= 2
    anomalous_rows = df.loc[clean_df.index[multi_mask]].copy()
    anomalous_rows["outlier_features_count"] = outlier_counts[multi_mask].values

    return (
        pd.DataFrame(summary) if summary else pd.DataFrame(),
        anomalous_rows,
    )
