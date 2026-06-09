"""
Advanced Statistical Tests — normality, chi-square, PCA, clustering.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def run_normality_tests(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Run Shapiro-Wilk and D'Agostino normality tests on numeric columns."""
    results = []
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 8:
            continue
        sample = data.sample(min(5000, len(data)), random_state=42)
        try:
            sw_stat, sw_p = scipy_stats.shapiro(sample)
        except Exception:
            sw_stat, sw_p = np.nan, np.nan
        try:
            da_stat, da_p = scipy_stats.normaltest(sample)
        except Exception:
            da_stat, da_p = np.nan, np.nan

        is_normal = (
            (sw_p > 0.05) and (da_p > 0.05)
            if not (np.isnan(sw_p) or np.isnan(da_p))
            else None
        )
        results.append({
            "Column": col,
            "Shapiro-Wilk stat": round(sw_stat, 4) if not np.isnan(sw_stat) else "N/A",
            "Shapiro p-value": round(sw_p, 4) if not np.isnan(sw_p) else "N/A",
            "D'Agostino stat": round(da_stat, 4) if not np.isnan(da_stat) else "N/A",
            "D'Agostino p-value": round(da_p, 4) if not np.isnan(da_p) else "N/A",
            "Normal?": "✅ Yes" if is_normal else ("❌ No" if is_normal is not None else "⚠️ N/A"),
        })
    return pd.DataFrame(results) if results else pd.DataFrame()


def run_chi_square_tests(
    df: pd.DataFrame, categorical_cols: list[str], max_pairs: int = 10
) -> pd.DataFrame:
    """Run chi-square independence tests between pairs of categorical columns."""
    results = []
    pairs_tested = 0
    for i, c1 in enumerate(categorical_cols):
        if pairs_tested >= max_pairs:
            break
        for c2 in categorical_cols[i + 1:]:
            if pairs_tested >= max_pairs:
                break
            if df[c1].nunique() > 20 or df[c2].nunique() > 20:
                continue
            try:
                contingency = pd.crosstab(df[c1], df[c2])
                chi2, p_val, dof, _ = scipy_stats.chi2_contingency(contingency)
                cramers_v = (
                    np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1)))
                    if min(contingency.shape) > 1
                    else 0
                )
                results.append({
                    "Variable 1": c1,
                    "Variable 2": c2,
                    "Chi² Statistic": round(chi2, 2),
                    "p-value": round(p_val, 4),
                    "Cramér's V": round(cramers_v, 3),
                    "Significant?": "✅ Yes" if p_val < 0.05 else "❌ No",
                    "Strength": (
                        "🔴 Strong" if cramers_v > 0.5
                        else "🟡 Moderate" if cramers_v > 0.3
                        else "🔵 Weak"
                    ),
                })
                pairs_tested += 1
            except Exception:
                continue
    return pd.DataFrame(results) if results else pd.DataFrame()


def run_pca_analysis(
    df: pd.DataFrame, numeric_cols: list[str], n_components: int = 0
):
    """Run PCA on standardized numeric columns. Returns dict with results."""
    clean_df = df[numeric_cols].dropna()
    if len(clean_df) < 3 or len(numeric_cols) < 2:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df)
    n_comp = min(
        n_components if n_components > 0 else len(numeric_cols),
        len(numeric_cols),
        len(clean_df),
    )
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(scaled)

    loadings = pd.DataFrame(
        pca.components_, columns=numeric_cols,
        index=[f"PC{i+1}" for i in range(n_comp)],
    )

    return {
        "model": pca,
        "transformed": transformed,
        "index": clean_df.index,
        "columns": numeric_cols,
        "variance_explained": pca.explained_variance_ratio_,
        "loadings": loadings,
    }


def run_kmeans_clustering(df: pd.DataFrame, numeric_cols: list[str], max_k: int = 8):
    """Find optimal k using elbow method and run KMeans. Returns dict with results."""
    clean_df = df[numeric_cols].dropna()
    if len(clean_df) < 4 or len(numeric_cols) < 2:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean_df)

    # Use first 5 PCs for clustering
    from sklearn.decomposition import PCA as PCA_
    n_pcs = min(5, scaled.shape[1])
    pca = PCA_(n_components=n_pcs)
    transformed = pca.fit_transform(scaled)

    max_k = min(max_k, len(clean_df) - 1)
    if max_k < 2:
        return None

    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(transformed)
        inertias.append(km.inertia_)

    if len(inertias) >= 2:
        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        ratios = [
            diffs[i] / diffs[i + 1] if diffs[i + 1] > 0 else 1
            for i in range(len(diffs) - 1)
        ]
        best_k = list(k_range)[np.argmax(ratios) + 1] if ratios else 2
    else:
        best_k = 2

    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_km.fit_predict(transformed)

    cluster_stats = df.loc[clean_df.index].copy()
    cluster_stats["Cluster"] = labels

    return {
        "labels": labels,
        "optimal_k": best_k,
        "inertias": list(zip(k_range, inertias)),
        "transformed": transformed,
        "index": clean_df.index,
        "cluster_stats": cluster_stats.groupby("Cluster")[numeric_cols].mean().round(2),
    }


def compute_correlation_with_pvalues(
    df: pd.DataFrame, numeric_cols: list[str], method: str = "pearson",
) -> dict:
    """
    Compute correlation matrix WITH p-values for statistical significance.

    Returns dict with:
        corr_matrix: pd.DataFrame — correlation coefficients
        pvalue_matrix: pd.DataFrame — p-values for each pair
        significant_pairs: list[dict] — pairs where p < 0.05
        method: str — "pearson" or "spearman"
    """
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return {"corr_matrix": pd.DataFrame(), "pvalue_matrix": pd.DataFrame(),
                "significant_pairs": [], "method": method}

    n = len(cols)
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.ones((n, n))

    corr_func = scipy_stats.pearsonr if method == "pearson" else scipy_stats.spearmanr

    for i in range(n):
        corr_matrix[i, i] = 1.0
        pval_matrix[i, i] = 0.0
        for j in range(i + 1, n):
            data_i = df[cols[i]]
            data_j = df[cols[j]]
            # Drop rows where either is NaN
            mask = data_i.notna() & data_j.notna()
            if mask.sum() < 3:
                continue
            try:
                corr, pval = corr_func(data_i[mask], data_j[mask])
                corr_matrix[i, j] = corr_matrix[j, i] = corr
                pval_matrix[i, j] = pval_matrix[j, i] = pval
            except Exception:
                continue

    corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols)
    pval_df = pd.DataFrame(pval_matrix, index=cols, columns=cols)

    # Extract significant pairs
    significant_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            p = pval_matrix[i, j]
            c = corr_matrix[i, j]
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            if p < 0.05:
                significant_pairs.append({
                    "Feature 1": cols[i],
                    "Feature 2": cols[j],
                    "Correlation": round(c, 4),
                    "|Correlation|": round(abs(c), 4),
                    "p-value": round(p, 6),
                    "Significance": significance,
                    "Strength": (
                        "🔴 Strong" if abs(c) > 0.5
                        else "🟡 Moderate" if abs(c) > 0.3
                        else "🔵 Weak"
                    ),
                })

    significant_pairs.sort(key=lambda x: x["|Correlation|"], reverse=True)

    return {
        "corr_matrix": corr_df,
        "pvalue_matrix": pval_df,
        "significant_pairs": significant_pairs,
        "method": method,
    }
