"""
HTML Report Export — generates a downloadable styled HTML report.
"""
import io
import base64
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded


def _generate_distribution_chart(df, numeric_cols) -> str:
    """Generate a grid of distribution charts."""
    import seaborn as sns
    cols_to_plot = numeric_cols[:9]
    n = len(cols_to_plot)
    if n == 0:
        return ""
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, col in enumerate(cols_to_plot):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#3498db")
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    encoded = _fig_to_base64(fig)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{encoded}" style="max-width:100%;">'


def _generate_correlation_chart(df, numeric_cols) -> str:
    """Generate a correlation heatmap."""
    import seaborn as sns
    cols = numeric_cols[:15]
    if len(cols) < 2:
        return ""
    fig, ax = plt.subplots(figsize=(min(12, len(cols)), min(10, len(cols) * 0.8)))
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="RdBu_r", fmt=".2f",
                ax=ax, center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    encoded = _fig_to_base64(fig)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{encoded}" style="max-width:100%;">'


def generate_html_report(
    df: pd.DataFrame,
    uploaded_name: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    ai_summary: str = "",
    anomaly_count: int = 0,
    ml_results: dict | None = None,
) -> str:
    """Generate a comprehensive styled HTML report."""

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Data overview stats
    n_rows, n_cols = df.shape
    n_missing = int(df.isnull().sum().sum())
    n_duplicates = int(df.duplicated().sum())
    memory_kb = df.memory_usage(deep=True).sum() / 1024

    # Missing values table
    missing_series = df.isnull().sum()
    missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
    missing_html = ""
    if not missing_series.empty:
        missing_df = missing_series.reset_index()
        missing_df.columns = ["Column", "Missing"]
        missing_df["% Missing"] = (missing_df["Missing"] / n_rows * 100).round(2)
        missing_html = missing_df.to_html(index=False, classes="data-table")

    # Descriptive stats
    desc_html = ""
    if numeric_cols:
        desc = df[numeric_cols].describe().T.round(3)
        desc["skewness"] = df[numeric_cols].skew().round(3)
        desc_html = desc.to_html(classes="data-table")

    # Charts
    dist_chart = _generate_distribution_chart(df, numeric_cols)
    corr_chart = _generate_correlation_chart(df, numeric_cols)

    # Categorical summary
    cat_html = ""
    if categorical_cols:
        cat_data = []
        for col in categorical_cols[:10]:
            cat_data.append({
                "Column": col,
                "Unique": df[col].nunique(),
                "Top Value": str(df[col].mode()[0]) if not df[col].mode().empty else "N/A",
                "Top Freq": int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else 0,
            })
        cat_html = pd.DataFrame(cat_data).to_html(index=False, classes="data-table")

    # ML results section
    ml_html = ""
    if ml_results and ml_results.get("success"):
        if ml_results["task_type"] == "classification":
            ml_html = f"""
            <h2>🤖 ML Model Results</h2>
            <p><strong>Model:</strong> {ml_results['model_name']} (Classification)</p>
            <p><strong>Accuracy:</strong> {ml_results['accuracy']}</p>
            <p><strong>F1 Score:</strong> {ml_results['f1_score']}</p>
            <p><strong>CV Score:</strong> {ml_results['cv_mean']} ± {ml_results['cv_std']}</p>
            """
        else:
            ml_html = f"""
            <h2>🤖 ML Model Results</h2>
            <p><strong>Model:</strong> {ml_results['model_name']} (Regression)</p>
            <p><strong>R² Score:</strong> {ml_results['r2']}</p>
            <p><strong>RMSE:</strong> {ml_results['rmse']}</p>
            <p><strong>CV Score:</strong> {ml_results['cv_mean']} ± {ml_results['cv_std']}</p>
            """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report — {uploaded_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f1117; color: #e0e0e0; padding: 40px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 2.5em; margin-bottom: 10px;
        }}
        h2 {{
            color: #667eea; border-bottom: 2px solid #667eea33;
            padding-bottom: 8px; margin: 30px 0 15px;
        }}
        .meta {{ color: #888; margin-bottom: 30px; }}
        .stats-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin: 20px 0;
        }}
        .stat-card {{
            background: #1a1d29; border-radius: 12px; padding: 20px;
            border: 1px solid #2a2d3a; text-align: center;
        }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .stat-card .label {{ color: #888; font-size: 0.9em; margin-top: 5px; }}
        .data-table {{
            width: 100%; border-collapse: collapse; margin: 15px 0;
            background: #1a1d29; border-radius: 8px; overflow: hidden;
        }}
        .data-table th {{
            background: #667eea33; color: #667eea; padding: 12px 15px;
            text-align: left; font-weight: 600;
        }}
        .data-table td {{
            padding: 10px 15px; border-bottom: 1px solid #2a2d3a;
        }}
        .data-table tr:hover {{ background: #2a2d3a; }}
        .ai-summary {{
            background: #1a1d29; border-left: 4px solid #667eea;
            padding: 20px; border-radius: 0 8px 8px 0; margin: 20px 0;
            white-space: pre-wrap;
        }}
        .anomaly-badge {{
            background: #e74c3c33; color: #e74c3c; padding: 4px 12px;
            border-radius: 20px; display: inline-block;
        }}
        img {{ border-radius: 8px; margin: 15px 0; }}
        .footer {{ text-align: center; color: #666; margin-top: 40px; padding-top: 20px;
                   border-top: 1px solid #2a2d3a; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 EDA Report</h1>
        <p class="meta">Dataset: <strong>{uploaded_name}</strong> | Generated: {now}</p>

        <h2>📋 Data Overview</h2>
        <div class="stats-grid">
            <div class="stat-card"><div class="value">{n_rows:,}</div><div class="label">Rows</div></div>
            <div class="stat-card"><div class="value">{n_cols}</div><div class="label">Columns</div></div>
            <div class="stat-card"><div class="value">{n_missing:,}</div><div class="label">Missing Values</div></div>
            <div class="stat-card"><div class="value">{n_duplicates:,}</div><div class="label">Duplicates</div></div>
            <div class="stat-card"><div class="value">{memory_kb:,.0f} KB</div><div class="label">Memory</div></div>
            <div class="stat-card"><div class="value">{anomaly_count}</div><div class="label">Anomalies Detected</div></div>
        </div>

        {"<h2>⚠️ Missing Values</h2>" + missing_html if missing_html else ""}

        {"<h2>📈 Descriptive Statistics</h2>" + desc_html if desc_html else ""}

        {"<h2>📊 Distributions</h2>" + dist_chart if dist_chart else ""}

        {"<h2>🔥 Correlation Matrix</h2>" + corr_chart if corr_chart else ""}

        {"<h2>📦 Categorical Summary</h2>" + cat_html if cat_html else ""}

        {ml_html}

        {"<h2>🤖 AI Executive Summary</h2><div class='ai-summary'>" + ai_summary + "</div>" if ai_summary else ""}

        <div class="footer">
            <p>Generated by AI Data Analyst Pro | {now}</p>
        </div>
    </div>
</body>
</html>"""

    return html
