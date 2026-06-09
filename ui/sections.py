"""
All UI sections — each function renders one section of the dashboard.
"""
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config.settings import (
    SYSTEM_ROLE, TIMEOUT_CLOUD, TIMEOUT_LOCAL,
    CLOUD_MODELS, NEMOTRON_MODELS,
)
from core.llm import query_ollama, stream_ollama, unload_ollama_model
from core.context import build_data_context
from core.agentic import agentic_plan, agentic_execute, agentic_synthesize
from analysis.features import (
    detect_date_columns, engineer_date_features,
    engineer_interaction_features, compute_skewness, impute_missing,
    detect_high_cardinality,
)
from analysis.stats import (
    run_normality_tests, run_chi_square_tests,
    run_pca_analysis, run_kmeans_clustering,
    compute_correlation_with_pvalues,
)
from analysis.anomaly import run_isolation_forest, run_zscore_detection, run_iqr_multivariate
from analysis.ml import train_quick_model, get_available_models
from analysis.health import compute_data_health_score
from tools.code_sandbox import execute_code_safely, generate_code_prompt
from tools.nl_viz import detect_viz_intent, generate_visualization
from tools.report import generate_html_report



# ════════════════════════════════════════════════
# 1. Data Inspection & Health Check
# ════════════════════════════════════════════════
def render_data_inspection(df: pd.DataFrame):
    st.header("🔍 1. Data Inspection & Health Check")

    # ── Data Health Score (0-100) ──
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    health = compute_data_health_score(df, numeric_cols, categorical_cols)

    # Health Score Gauge
    grade_colors = {"A": "#2ecc71", "B": "#27ae60", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c"}
    grade_color = grade_colors.get(health["grade"], "#888")

    health_col, metrics_col = st.columns([1, 2])
    with health_col:
        # Plotly gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health["total_score"],
            title={"text": f"Data Health — Grade {health['grade']}", "font": {"size": 16, "color": "#e0e0e0"}},
            number={"suffix": "/100", "font": {"size": 36, "color": grade_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333"},
                "bar": {"color": grade_color},
                "bgcolor": "#1a1d29",
                "borderwidth": 2,
                "bordercolor": "#2a2d3a",
                "steps": [
                    {"range": [0, 50], "color": "#2a1015"},
                    {"range": [50, 65], "color": "#2a2010"},
                    {"range": [65, 80], "color": "#1a2a10"},
                    {"range": [80, 100], "color": "#102a15"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": health["total_score"],
                },
            },
        ))
        fig_gauge.update_layout(
            height=220, margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e0e0e0"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with metrics_col:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Duplicates", int(df.duplicated().sum()))
        c4.metric("Missing Cells", int(df.isnull().sum().sum()))
        c5.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB")

        # Health breakdown bar
        breakdown_data = []
        for cat, info in health["breakdown"].items():
            breakdown_data.append({
                "Category": cat,
                "Score": info["score"],
                "Max": info["max"],
                "Pct": round(info["score"] / info["max"] * 100, 1) if info["max"] > 0 else 0,
            })
        bd_df = pd.DataFrame(breakdown_data)

        fig_bar = px.bar(
            bd_df, x="Pct", y="Category", orientation="h",
            color="Pct",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            range_color=[0, 100],
            text=bd_df.apply(lambda r: f"{r['Score']}/{r['Max']}", axis=1),
        )
        fig_bar.update_layout(
            height=180, margin=dict(l=0, r=0, t=5, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e0e0e0", "size": 11},
            showlegend=False, coloraxis_showscale=False,
            xaxis={"visible": False, "range": [0, 100]},
            yaxis={"title": ""},
        )
        fig_bar.update_traces(textposition="auto")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Health recommendations
    if health["recommendations"]:
        with st.expander("💡 Data Health Recommendations"):
            for rec in health["recommendations"]:
                st.markdown(f"- {rec}")
            # Detailed breakdown
            for cat, info in health["breakdown"].items():
                for note in info["notes"]:
                    st.caption(f"**{cat}**: {note}")

    with st.expander("📋 Data Preview (first 20 rows)", expanded=True):
        st.dataframe(df.head(20), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Column Data Types")
        dtypes_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(dtypes_df, width="stretch")

    with right:
        st.subheader("Missing Values per Column")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.write("No missing values found! 🎉")
        else:
            missing_df = missing.reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            missing_df["% Missing"] = (missing_df["Missing Count"] / len(df) * 100).round(2)
            st.dataframe(missing_df, width="stretch")
            fig_missing = px.bar(
                missing_df, x="% Missing", y="Column", orientation="h",
                color="% Missing",
                color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                title="Missing Values Distribution",
            )
            fig_missing.update_layout(
                height=max(250, len(missing) * 30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e0e0e0"},
                showlegend=False, coloraxis_showscale=False,
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(fig_missing, use_container_width=True)


# ════════════════════════════════════════════════
# 2. Statistical Profiling
# ════════════════════════════════════════════════
def render_statistical_profiling(df, numeric_cols, categorical_cols):
    st.header("📈 2. Statistical Profiling")

    tab_num, tab_cat, tab_full = st.tabs(["Numerical Summary", "Categorical Summary", "Full Describe"])
    with tab_num:
        if numeric_cols:
            desc = df[numeric_cols].describe().T
            desc["skewness"] = df[numeric_cols].skew()
            desc["kurtosis"] = df[numeric_cols].kurtosis()
            st.dataframe(desc.round(3), width="stretch")
        else:
            st.info("No numerical columns detected.")
    with tab_cat:
        if categorical_cols:
            cat_summary = []
            for col in categorical_cols:
                cat_summary.append({
                    "Column": col,
                    "Unique Values": df[col].nunique(),
                    "Top Value": df[col].mode()[0] if not df[col].mode().empty else "N/A",
                    "Top Frequency": df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0,
                    "Missing": df[col].isnull().sum(),
                })
            st.dataframe(pd.DataFrame(cat_summary), width="stretch")
        else:
            st.info("No categorical columns detected.")
    with tab_full:
        st.dataframe(df.describe(include="all").T, width="stretch")


# ════════════════════════════════════════════════
# 3. Feature Engineering
# ════════════════════════════════════════════════
def render_feature_engineering(df, numeric_cols, categorical_cols):
    """Returns (df, engineered_features, imputation_report, skew_df, date_cols, numeric_cols)."""
    st.header("🔧 3. Automated Feature Engineering")

    engineered_features = []
    imputation_report = {}
    skew_df = pd.DataFrame()

    # Date features
    date_cols = detect_date_columns(df)
    if date_cols:
        df, date_feats = engineer_date_features(df, date_cols)
        engineered_features.extend(date_feats)
        st.markdown(f"- 📅 **Date features**: Extracted {len(date_feats)} temporal features from `{'`, `'.join(date_cols)}`")

    # Imputation
    num_for_impute = df.select_dtypes(include=np.number).columns.tolist()
    cat_for_impute = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if df.isnull().sum().sum() > 0:
        df, imputation_report = impute_missing(df, num_for_impute, cat_for_impute)
        st.markdown(f"- 🩹 **Imputation**: Fixed {len(imputation_report)} columns")

    # Skewness
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    skew_df = compute_skewness(df, numeric_cols)
    if not skew_df.empty:
        st.markdown(f"- 📐 **Skewness detected**: {len(skew_df)} features are highly skewed")

    # Interactions
    original_numeric = [c for c in numeric_cols if c not in engineered_features]
    if len(original_numeric) >= 2:
        df, interaction_feats = engineer_interaction_features(df, original_numeric, top_n=3)
        engineered_features.extend(interaction_feats)
        st.markdown(f"- ✖️ **Interaction features**: Created {len(interaction_feats)} features")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ── High Cardinality Guard ──
    cat_cols_current = df.select_dtypes(include=["object", "category"]).columns.tolist()
    high_card = detect_high_cardinality(df, cat_cols_current)
    if high_card:
        st.markdown(f"- ⚠️ **High Cardinality**: {len(high_card)} columns flagged (risk of dimension explosion)")
        with st.expander(f"⚠️ High Cardinality Warning ({len(high_card)} columns)", expanded=True):
            st.markdown(
                "**Why this matters:** One-Hot Encoding these columns would create thousands of sparse features, "
                "crashing memory or degrading model performance (Curse of Dimensionality)."
            )
            hc_df = pd.DataFrame([{
                "Column": h["column"],
                "Unique Values": h["unique_count"],
                "Total Rows": h["total_rows"],
                "Uniqueness": h["ratio_pct"],
                "Recommendation": h["recommendation"],
            } for h in high_card])
            st.dataframe(hc_df, width="stretch")

    # Details
    if imputation_report:
        with st.expander("🩹 Imputation Details"):
            st.dataframe(
                pd.DataFrame([{"Column": k, "Action": v} for k, v in imputation_report.items()]),
                width="stretch",
            )
    if not skew_df.empty:
        with st.expander("📐 Skewness Details"):
            st.dataframe(skew_df, width="stretch")
    if engineered_features:
        with st.expander(f"🆕 Engineered Features ({len(engineered_features)} new)"):
            preview = [c for c in engineered_features if c in df.columns]
            st.dataframe(df[preview].head(10), width="stretch")

    return df, engineered_features, imputation_report, skew_df, date_cols, numeric_cols


# ════════════════════════════════════════════════
# 4. Automated Visualizations
# ════════════════════════════════════════════════
def render_visualizations(df, numeric_cols, categorical_cols, date_cols, section_num):
    st.header(f"🎨 {section_num}. Automated Visualizations")

    viz_tabs = st.tabs(["📊 Distributions", "📦 Categories", "🔥 Correlations",
                        "📈 Pairwise", "📉 Time Series"])

    with viz_tabs[0]:
        if numeric_cols:
            sel = st.selectbox("Pick a numerical column:", numeric_cols, key="dist_num")
            data = df[sel].dropna()

            col_a, col_b = st.columns(2)
            with col_a:
                # Interactive histogram with KDE
                fig_hist = px.histogram(
                    data, nbins=40, marginal="box",
                    title=f"Distribution — {sel}",
                    color_discrete_sequence=["#667eea"],
                    opacity=0.8,
                )
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e0e0e0"},
                    xaxis_title=sel, yaxis_title="Count",
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_b:
                # Interactive violin + box
                fig_violin = go.Figure()
                fig_violin.add_trace(go.Violin(
                    y=data, box_visible=True, meanline_visible=True,
                    fillcolor="#764ba2", opacity=0.7,
                    line_color="#c99af8", name=sel,
                ))
                fig_violin.update_layout(
                    title=f"Violin + Box — {sel}",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e0e0e0"},
                    showlegend=False,
                )
                st.plotly_chart(fig_violin, use_container_width=True)

            # Quick stats
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Mean", f"{data.mean():.3f}")
            sc2.metric("Median", f"{data.median():.3f}")
            sc3.metric("Skew", f"{data.skew():.3f}")
            sc4.metric("Kurtosis", f"{data.kurtosis():.3f}")
        else:
            st.info("No numerical columns.")

    with viz_tabs[1]:
        if categorical_cols:
            sel = st.selectbox("Pick a categorical column:", categorical_cols, key="cat_col")
            top = df[sel].value_counts().nlargest(15)
            fig_cat = px.bar(
                x=top.values, y=top.index, orientation="h",
                title=f"Top Categories — {sel}",
                color=top.values,
                color_continuous_scale="Viridis",
                labels={"x": "Count", "y": sel},
                text=top.values,
            )
            fig_cat.update_layout(
                height=max(350, len(top) * 35),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e0e0e0"},
                showlegend=False, coloraxis_showscale=False,
                yaxis={"autorange": "reversed"},
            )
            fig_cat.update_traces(textposition="auto")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No categorical columns.")

    with viz_tabs[2]:
        if len(numeric_cols) > 1:
            corr_cols = numeric_cols[:15]

            # P-value validated correlation
            corr_method = st.radio(
                "Correlation method:", ["Pearson (linear)", "Spearman (non-linear)"],
                horizontal=True, key="corr_method",
            )
            method = "pearson" if "Pearson" in corr_method else "spearman"

            with st.spinner("Computing correlations with p-values..."):
                corr_result = compute_correlation_with_pvalues(df, corr_cols, method=method)

            corr_matrix = corr_result["corr_matrix"]
            pval_matrix = corr_result["pvalue_matrix"]

            # Create annotation text with significance stars
            annot_text = []
            for i in range(len(corr_cols)):
                row = []
                for j in range(len(corr_cols)):
                    c = corr_matrix.iloc[i, j]
                    p = pval_matrix.iloc[i, j]
                    if i == j:
                        row.append("1.00")
                    elif p < 0.001:
                        row.append(f"{c:.2f}***")
                    elif p < 0.01:
                        row.append(f"{c:.2f}**")
                    elif p < 0.05:
                        row.append(f"{c:.2f}*")
                    else:
                        row.append(f"{c:.2f}")
                annot_text.append(row)

            # Plotly heatmap with hover showing p-values
            hover_text = []
            for i in range(len(corr_cols)):
                row = []
                for j in range(len(corr_cols)):
                    c = corr_matrix.iloc[i, j]
                    p = pval_matrix.iloc[i, j]
                    sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
                    row.append(
                        f"{corr_cols[i]} vs {corr_cols[j]}<br>"
                        f"Correlation: {c:.4f}<br>"
                        f"P-value: {p:.6f}<br>"
                        f"Status: {sig}"
                    )
                hover_text.append(row)

            # Mask upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
            masked_corr = corr_matrix.values.copy().astype(float)
            masked_corr[mask] = np.nan

            fig_corr = go.Figure(data=go.Heatmap(
                z=masked_corr,
                x=corr_cols,
                y=corr_cols,
                text=annot_text,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertext=hover_text,
                hoverinfo="text",
                colorscale="RdBu_r",
                zmid=0, zmin=-1, zmax=1,
                showscale=True,
                colorbar={"title": "Corr"},
            ))
            fig_corr.update_layout(
                title=f"{method.title()} Correlation Matrix (with p-value significance)",
                height=max(500, len(corr_cols) * 45),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e0e0e0"},
                xaxis={"side": "bottom"},
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05. Grey cells are not statistically significant.")

            # Significant pairs table
            with st.expander(f"📊 Statistically Significant Correlations ({len(corr_result['significant_pairs'])} pairs)"):
                if corr_result["significant_pairs"]:
                    sig_df = pd.DataFrame(corr_result["significant_pairs"])
                    st.dataframe(sig_df.drop(columns=["|Correlation|"]).round(4), width="stretch")
                else:
                    st.info("No statistically significant correlations found at p < 0.05.")
        else:
            st.info("Need at least 2 numerical columns.")


    with viz_tabs[3]:
        if 2 <= len(numeric_cols) <= 10:
            pair_cols = numeric_cols[:6]
            st.caption(f"Pairplot for: {', '.join(pair_cols)}")
            hue_col = None
            if categorical_cols and df[categorical_cols[0]].nunique() <= 6:
                hue_col = categorical_cols[0]
            fig = sns.pairplot(
                df[pair_cols + ([hue_col] if hue_col else [])].dropna(),
                hue=hue_col, diag_kind="kde",
                palette="Set2" if hue_col else None,
                plot_kws={"alpha": 0.6, "s": 20},
            )
            st.pyplot(fig.figure)
            plt.close(fig.figure)
        elif len(numeric_cols) > 10:
            st.info("Too many columns for auto-pairplot. Select up to 6:")
            pair_sel = st.multiselect("Select columns:", numeric_cols,
                                      default=numeric_cols[:4], key="pair_select")
            if 2 <= len(pair_sel) <= 6:
                fig = sns.pairplot(df[pair_sel].dropna(), diag_kind="kde",
                                  plot_kws={"alpha": 0.6, "s": 20})
                st.pyplot(fig.figure)
                plt.close(fig.figure)
        else:
            st.info("Need at least 2 numerical columns.")

    with viz_tabs[4]:
        if date_cols:
            ts_date = st.selectbox("Date column:", date_cols, key="ts_date")
            ts_val = st.selectbox("Value column:", numeric_cols, key="ts_val")
            ts_df = df[[ts_date, ts_val]].copy()
            ts_df[ts_date] = pd.to_datetime(ts_df[ts_date], errors="coerce")
            ts_df = ts_df.dropna().sort_values(ts_date)
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(ts_df[ts_date], ts_df[ts_val], color="#3498db", alpha=0.7, linewidth=1)
            if len(ts_df) > 10:
                window = max(7, len(ts_df) // 30)
                rolling = ts_df[ts_val].rolling(window=window, center=True).mean()
                ax.plot(ts_df[ts_date], rolling, color="#e74c3c", linewidth=2,
                        label=f"{window}-pt moving avg")
                ax.legend()
            ax.set_title(f"{ts_val} over time")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No date columns detected.")


# ════════════════════════════════════════════════
# 5. Advanced Statistical Tests
# ════════════════════════════════════════════════
def render_advanced_stats(df, numeric_cols, categorical_cols, section_num):
    """Returns pca_data for clustering."""
    st.header(f"📐 {section_num}. Advanced Statistical Tests")
    pca_data = None
    pca_idx = None

    tabs = st.tabs(["🔔 Normality", "🔗 Chi-Square", "📉 PCA", "🎯 Clustering"])

    with tabs[0]:
        st.subheader("Distribution Normality Tests")
        st.caption("p > 0.05 = normal distribution")
        if numeric_cols:
            with st.spinner("Running normality tests..."):
                norm_df = run_normality_tests(df, numeric_cols)
            if not norm_df.empty:
                n_yes = (norm_df["Normal?"] == "✅ Yes").sum()
                n_no = (norm_df["Normal?"] == "❌ No").sum()
                c1, c2 = st.columns(2)
                c1.metric("Normal", n_yes)
                c2.metric("Non-Normal", n_no)
                st.dataframe(norm_df, width="stretch")
            else:
                st.info("Not enough data.")
        else:
            st.info("No numeric columns.")

    with tabs[1]:
        st.subheader("Chi-Square Independence Tests")
        if len(categorical_cols) >= 2:
            with st.spinner("Running chi-square tests..."):
                chi_df = run_chi_square_tests(df, categorical_cols)
            if not chi_df.empty:
                st.dataframe(chi_df, width="stretch")
            else:
                st.info("No suitable pairs found.")
        else:
            st.info("Need at least 2 categorical columns.")

    with tabs[2]:
        st.subheader("Principal Component Analysis (PCA)")
        if len(numeric_cols) >= 2:
            with st.spinner("Running PCA..."):
                pca_result = run_pca_analysis(df, numeric_cols)
            if pca_result is not None:
                pca_model = pca_result["model"]
                pca_data = pca_result["transformed"]
                pca_idx = pca_result["index"]
                pca_cols = pca_result["columns"]

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                cumulative = np.cumsum(pca_model.explained_variance_ratio_)
                n_pc = len(pca_model.explained_variance_ratio_)
                axes[0].bar(range(1, n_pc + 1), pca_model.explained_variance_ratio_,
                            color="#3498db", alpha=0.7, label="Individual")
                axes[0].plot(range(1, n_pc + 1), cumulative, "ro-", label="Cumulative")
                axes[0].axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="95%")
                axes[0].set_xlabel("PC")
                axes[0].set_ylabel("Explained Variance")
                axes[0].set_title("PCA Explained Variance")
                axes[0].legend()
                if pca_data.shape[1] >= 2:
                    axes[1].scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.5, s=15, c="#3498db")
                    axes[1].set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})")
                    axes[1].set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})")
                    axes[1].set_title("PCA 2D Projection")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                n_show = min(5, n_pc)
                with st.expander(f"Top {n_show} Component Loadings"):
                    st.dataframe(pca_result["loadings"].head(n_show).round(3), width="stretch")
            else:
                st.info("Not enough data for PCA.")
        else:
            st.info("Need 2+ numeric columns.")

    with tabs[3]:
        st.subheader("Auto-Clustering (KMeans)")
        if len(numeric_cols) >= 2:
            with st.spinner("Finding clusters..."):
                cluster_result = run_kmeans_clustering(df, numeric_cols)
            if cluster_result is not None:
                labels = cluster_result["labels"]
                best_k = cluster_result["optimal_k"]
                elbow = cluster_result["inertias"]
                transformed = cluster_result["transformed"]

                st.success(f"Optimal clusters: **{best_k}** (elbow method)")
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                ks, inertias = zip(*elbow)
                axes[0].plot(ks, inertias, "bo-")
                axes[0].axvline(x=best_k, color="red", linestyle="--", label=f"k={best_k}")
                axes[0].set_xlabel("k")
                axes[0].set_ylabel("Inertia")
                axes[0].set_title("Elbow Method")
                axes[0].legend()
                if transformed.shape[1] >= 2:
                    sc = axes[1].scatter(transformed[:, 0], transformed[:, 1],
                                         c=labels, cmap="Set2", alpha=0.6, s=15)
                    axes[1].set_xlabel("PC1")
                    axes[1].set_ylabel("PC2")
                    axes[1].set_title(f"Clusters (k={best_k})")
                    plt.colorbar(sc, ax=axes[1], label="Cluster")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                with st.expander("Cluster Statistics"):
                    st.dataframe(cluster_result["cluster_stats"], width="stretch")
            else:
                st.info("Not enough data for clustering.")
        else:
            st.info("Need 2+ numeric columns for clustering.")

    return pca_data, pca_idx


# ════════════════════════════════════════════════
# 6. Advanced Analysis (Target, Importance, Outliers)
# ════════════════════════════════════════════════
def render_advanced_analysis(df, numeric_cols, categorical_cols, section_num):
    """Returns target_col for downstream use."""
    st.header(f"🧠 {section_num}. Advanced Analysis")

    tabs = st.tabs(["🎯 Target Analysis", "📊 Feature Importance", "🔎 Outlier Deep Dive"])

    target_col = "(none)"
    with tabs[0]:
        st.subheader("Target Variable Analysis")
        target_col = st.selectbox(
            "Select target:", ["(none)"] + numeric_cols + categorical_cols,
            key="target_select",
        )
        if target_col != "(none)":
            if target_col in numeric_cols:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                sns.histplot(df[target_col].dropna(), kde=True, ax=axes[0], color="#e74c3c")
                axes[0].set_title(f"Distribution — {target_col}")
                sorted_vals = np.sort(df[target_col].dropna().values)
                axes[1].plot(sorted_vals, marker=".", linestyle="none", alpha=0.5, color="#3498db")
                axes[1].set_title(f"Sorted Values — {target_col}")
                axes[1].set_xlabel("Rank")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                ca, cb, cc = st.columns(3)
                ca.metric("Mean", f"{df[target_col].mean():.3f}")
                cb.metric("Median", f"{df[target_col].median():.3f}")
                cc.metric("Std Dev", f"{df[target_col].std():.3f}")
            elif target_col in categorical_cols:
                vc = df[target_col].value_counts()
                fig, ax = plt.subplots(figsize=(8, 8))
                if len(vc) <= 10:
                    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                           colors=sns.color_palette("Set2", len(vc)))
                else:
                    top_vc = vc.head(10)
                    ax.barh(range(len(top_vc)), top_vc.values,
                            color=sns.color_palette("viridis", len(top_vc)))
                    ax.set_yticks(range(len(top_vc)))
                    ax.set_yticklabels(top_vc.index)
                    ax.set_title(f"Top 10 — {target_col}")
                    ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with tabs[1]:
        st.subheader("Feature Importance (Correlation-based)")
        if target_col != "(none)" and target_col in numeric_cols:
            importances = df[numeric_cols].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, max(4, len(importances) * 0.35)))
            colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.3 else "#3498db" for v in importances.values]
            ax.barh(range(len(importances)), importances.values, color=colors)
            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels(importances.index)
            ax.set_title(f"Feature Importance for '{target_col}'")
            ax.set_xlabel("|Correlation|")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(
                pd.DataFrame({
                    "Feature": importances.index,
                    "|Correlation|": importances.values.round(3),
                    "Strength": ["🔴 Strong" if v > 0.5 else "🟡 Moderate" if v > 0.3 else "🔵 Weak" for v in importances.values],
                }),
                width="stretch",
            )
        else:
            st.info("Select a numeric target above.")

    with tabs[2]:
        st.subheader("Outlier Deep Dive")
        if numeric_cols:
            col = st.selectbox("Column:", numeric_cols, key="outlier_col")
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = df[(df[col] < lb) | (df[col] > ub)]
            cx, cy, cz = st.columns(3)
            cx.metric("Outliers", len(outliers))
            cy.metric("Lower Bound", f"{lb:.2f}")
            cz.metric("Upper Bound", f"{ub:.2f}")
            if len(outliers) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                normal = df[(df[col] >= lb) & (df[col] <= ub)]
                axes[0].scatter(range(len(normal)), normal[col], alpha=0.3, s=10, color="#3498db", label="Normal")
                axes[0].scatter(range(len(normal), len(normal) + len(outliers)),
                                outliers[col], alpha=0.7, s=20, color="#e74c3c", label="Outliers")
                axes[0].axhline(y=ub, color="orange", linestyle="--", alpha=0.7)
                axes[0].axhline(y=lb, color="orange", linestyle="--", alpha=0.7)
                axes[0].legend()
                axes[0].set_title(f"Outliers in {col}")
                sns.boxplot(x=df[col], ax=axes[1], color="#2ecc71")
                axes[1].set_title(f"Box Plot — {col}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                with st.expander(f"View outlier rows ({len(outliers)})"):
                    st.dataframe(outliers.head(50), width="stretch")
        else:
            st.info("No numeric columns.")

    return target_col


# ════════════════════════════════════════════════
# NEW: Anomaly Detection
# ════════════════════════════════════════════════
def render_anomaly_detection(df, numeric_cols, section_num):
    """Returns anomaly_count for report."""
    st.header(f"🔍 {section_num}. Anomaly Detection")
    anomaly_count = 0

    tabs = st.tabs(["🌲 Isolation Forest", "📏 Z-Score", "📦 IQR Multivariate"])

    with tabs[0]:
        st.subheader("Isolation Forest")
        contamination = st.slider("Contamination Rate", 0.01, 0.20, 0.05, 0.01, key="iso_contam")
        if numeric_cols:
            with st.spinner("Running Isolation Forest..."):
                result_df, mask, scores = run_isolation_forest(df, numeric_cols, contamination)
            if len(mask) > 0:
                anomaly_count = int(mask.sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Anomalies", anomaly_count)
                c2.metric("% Anomalous", f"{anomaly_count / len(mask) * 100:.1f}%")
                c3.metric("Normal Points", int((~mask).sum()))

                # PCA scatter with anomaly colors
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA
                clean = df[numeric_cols].dropna()
                if len(clean) > 0 and len(numeric_cols) >= 2:
                    scaled = StandardScaler().fit_transform(clean)
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(scaled)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ["#e74c3c" if m else "#3498db" for m in mask]
                    ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.6, s=15)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title(f"Anomalies on PCA Space ({anomaly_count} detected)")
                    # Legend
                    from matplotlib.lines import Line2D
                    legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Anomaly'),
                              Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Normal')]
                    ax.legend(handles=legend)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with st.expander(f"View Anomalous Rows ({anomaly_count})"):
                    anomalous = result_df[result_df["anomaly_label"] == "🔴 Anomaly"]
                    st.dataframe(anomalous.head(100), width="stretch")
                    csv = anomalous.to_csv(index=False)
                    st.download_button("📥 Download Anomalies CSV", csv, "anomalies.csv", "text/csv")
        else:
            st.info("No numeric columns.")

    with tabs[1]:
        st.subheader("Z-Score Detection")
        threshold = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.5, key="z_thresh")
        if numeric_cols:
            with st.spinner("Running Z-Score detection..."):
                summary_df, anomalous_rows = run_zscore_detection(df, numeric_cols, threshold)
            if not summary_df.empty:
                st.dataframe(summary_df, width="stretch")
                st.metric("Total Anomalous Rows", len(anomalous_rows))
                with st.expander("View Anomalous Rows"):
                    st.dataframe(anomalous_rows.head(100), width="stretch")
            else:
                st.success("No Z-Score anomalies found!")

    with tabs[2]:
        st.subheader("IQR Multivariate")
        st.caption("Rows that are outliers in 2+ features simultaneously")
        if numeric_cols:
            with st.spinner("Running multivariate IQR..."):
                iqr_summary, iqr_anomalous = run_iqr_multivariate(df, numeric_cols)
            if not iqr_summary.empty:
                st.dataframe(iqr_summary, width="stretch")
                st.metric("Multi-Feature Outlier Rows", len(iqr_anomalous))
                with st.expander("View Multi-Feature Outliers"):
                    st.dataframe(iqr_anomalous.head(100), width="stretch")
            else:
                st.success("No multivariate outliers found!")

    return anomaly_count


# ════════════════════════════════════════════════
# NEW: Quick ML Modeling
# ════════════════════════════════════════════════
def render_ml_modeling(df, numeric_cols, categorical_cols, section_num):
    """Returns ml_results dict."""
    st.header(f"🤖 {section_num}. Quick ML Model")
    ml_results = None

    target = st.selectbox("Select target variable:", ["(none)"] + numeric_cols + categorical_cols, key="ml_target")

    if target != "(none)":
        # Determine task type for model selection
        if target in categorical_cols or (target in numeric_cols and df[target].nunique() <= 10):
            task = "classification"
        else:
            task = "regression"

        available = get_available_models(task)
        st.caption(f"Detected task: **{task.title()}**")

        feature_cols = [c for c in numeric_cols + categorical_cols if c != target]
        sel_features = st.multiselect("Select features:", feature_cols, default=feature_cols[:15], key="ml_features")
        model_name = st.selectbox("Model:", available, key="ml_model")

        if st.button("🚀 Train Model", key="train_btn"):
            if len(sel_features) < 1:
                st.error("Select at least 1 feature.")
            else:
                with st.spinner(f"Training {model_name}..."):
                    ml_results = train_quick_model(df, target, sel_features, model_name)

                if ml_results and ml_results["success"]:
                    st.success(f"✅ Model trained on {ml_results['n_samples']} samples!")
                    if ml_results.get("leakage_protected"):
                        st.markdown(
                            '<span style="background:#1a4d2e;color:#2ecc71;padding:4px 12px;'
                            'border-radius:12px;font-size:0.8rem;">🛡️ Data Leakage Protected — '
                            'sklearn Pipeline ensures scaler fits only on training data</span>',
                            unsafe_allow_html=True,
                        )

                    if task == "classification":
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Accuracy", ml_results["accuracy"])
                        c2.metric("F1 (weighted)", ml_results["f1_score"])
                        c3.metric("CV Score", f"{ml_results['cv_mean']} ± {ml_results['cv_std']}")

                        # Confusion matrix
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(ml_results["confusion_matrix"], annot=True, fmt="d",
                                    cmap="Blues", ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        ax.set_title("Confusion Matrix")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("R² Score", ml_results["r2"])
                        c2.metric("RMSE", ml_results["rmse"])
                        c3.metric("CV Score", f"{ml_results['cv_mean']} ± {ml_results['cv_std']}")

                        # Actual vs Predicted
                        if "y_test" in ml_results and "y_pred" in ml_results:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(ml_results["y_test"], ml_results["y_pred"], alpha=0.5, s=20, color="#3498db")
                            mn = min(ml_results["y_test"].min(), ml_results["y_pred"].min())
                            mx = max(ml_results["y_test"].max(), ml_results["y_pred"].max())
                            ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect")
                            ax.set_xlabel("Actual")
                            ax.set_ylabel("Predicted")
                            ax.set_title("Actual vs Predicted")
                            ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                    # Feature importance
                    if "feature_importance" in ml_results:
                        with st.expander("📊 Feature Importance"):
                            imp = ml_results["feature_importance"].head(20)
                            fig, ax = plt.subplots(figsize=(10, max(4, len(imp) * 0.35)))
                            ax.barh(range(len(imp)), imp["Importance"].values,
                                    color=sns.color_palette("viridis", len(imp)))
                            ax.set_yticks(range(len(imp)))
                            ax.set_yticklabels(imp["Feature"].values)
                            ax.set_title("Feature Importance")
                            ax.invert_yaxis()
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            st.dataframe(imp.round(4), width="stretch")

                    # Model Explainability (SHAP)
                    if "shap_values" in ml_results and ml_results["shap_values"] is not None:
                        with st.expander("🧠 Model Explainability (SHAP)"):
                            st.markdown(
                                "SHAP (SHapley Additive exPlanations) shows how each feature "
                                "impacts the model's prediction. Red = higher feature value, "
                                "Blue = lower feature value. Position on X-axis shows the impact."
                            )
                            import shap
                            try:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                shap.summary_plot(
                                    ml_results["shap_values"],
                                    ml_results["shap_test_data"],
                                    show=False,
                                    plot_size=(10, 6)
                                )
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as e:
                                st.warning(f"Could not render SHAP plot: {e}")

                elif ml_results:
                    st.error(f"❌ {ml_results['error']}")

    return ml_results


# ════════════════════════════════════════════════
# NEW: Code Lab
# ════════════════════════════════════════════════
def render_code_lab(df, numeric_cols, categorical_cols, selected_model, section_num):
    st.header(f"🐍 {section_num}. AI Code Lab")
    st.markdown("Describe what you want in English → AI writes & executes Python code on your data.")

    code_request = st.text_area(
        "What do you want to analyze?",
        placeholder="e.g., Show average salary by department, Find the top 10 customers by revenue...",
        key="code_lab_input",
    )

    if st.button("⚡ Generate & Run", key="code_lab_run"):
        if not code_request.strip():
            st.warning("Enter a request first.")
        elif selected_model.startswith("("):
            st.error("No model available.")
        else:
            cols_info = f"Numeric: {numeric_cols[:20]}\nCategorical: {categorical_cols[:15]}"
            prompt = generate_code_prompt(code_request, df, cols_info)

            with st.spinner("Generating code..."):
                import re
                code_model = NEMOTRON_MODELS.get("code_gen", selected_model)
                timeout = TIMEOUT_CLOUD if code_model in CLOUD_MODELS else TIMEOUT_LOCAL
                code = query_ollama(code_model, prompt,
                                    system="Generate ONLY executable Python code. No markdown.",
                                    timeout=timeout)
                # Clean markdown fences
                code = re.sub(r"^```python\s*\n?", "", code, flags=re.MULTILINE)
                code = re.sub(r"^```\s*$", "", code, flags=re.MULTILINE)
                code = code.strip()

            if code and not code.startswith("❌"):
                with st.expander("📝 Generated Code", expanded=True):
                    st.code(code, language="python")

                with st.spinner("Executing..."):
                    result = execute_code_safely(code, df)

                if result["success"]:
                    if result["stdout"]:
                        st.subheader("📄 Output")
                        st.text(result["stdout"])
                    if result["result_df"] is not None:
                        st.subheader("📊 Result Table")
                        st.dataframe(result["result_df"], width="stretch")
                    for fig in result["figures"]:
                        st.pyplot(fig)
                        plt.close(fig)
                    if not result["stdout"] and result["result_df"] is None and not result["figures"]:
                        st.info("Code executed successfully but produced no output.")
                else:
                    st.error(f"Execution error:\n{result['error']}")
            else:
                st.error(code or "Failed to generate code.")


# ════════════════════════════════════════════════
# AI Auto-Summary
# ════════════════════════════════════════════════
def render_ai_summary(df, uploaded_name, numeric_cols, categorical_cols,
                      engineered_features, imputation_report, skew_df, date_cols,
                      selected_model, section_num):
    st.header(f"📝 {section_num}. AI Executive Summary")
    key = f"auto_summary_{uploaded_name}"

    if key not in st.session_state:
        ctx = build_data_context(
            df=df, uploaded_name=uploaded_name,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            engineered_features=engineered_features,
            imputation_report=imputation_report,
            skew_df=skew_df, date_cols=date_cols,
        )
        if not selected_model.startswith("("):
            with st.spinner(f"Generating summary with {selected_model}..."):
                prompt = (
                    f"Analyze this dataset and provide a concise executive summary.\n\n"
                    f"{ctx}\n\n"
                    f"Provide:\n"
                    f"1. **Data Quality Grade** (A-F) with justification\n"
                    f"2. **Top 3 Key Patterns** with specific numbers\n"
                    f"3. **Top 3 Actionable Recommendations**\n"
                    f"4. **Predicted Best Use Cases**\n"
                    f"5. **Red Flags**\n"
                    f"Be specific and cite exact numbers."
                )
                timeout = TIMEOUT_CLOUD if selected_model in CLOUD_MODELS else TIMEOUT_LOCAL
                st.session_state[key] = query_ollama(selected_model, prompt,
                                                     system=SYSTEM_ROLE, timeout=timeout)
        else:
            st.session_state[key] = "❌ No model available."

    st.markdown(st.session_state[key])
    if st.button("🔄 Regenerate", key="regen_summary"):
        del st.session_state[key]
        st.rerun()

    return st.session_state.get(key, "")


# ════════════════════════════════════════════════
# AI Q&A Chat (with Agentic & Consensus modes)
# ════════════════════════════════════════════════
def render_qa_chat(df, uploaded_name, numeric_cols, categorical_cols,
                   engineered_features, imputation_report, skew_df, date_cols,
                   selected_model, available_models, section_num,
                   run_agentic=False, run_consensus=False):
    st.header(f"💡 {section_num}. AI-Powered Q&A")
    st.markdown("Ask anything about your data — get **detailed answers with insights & solutions**.")

    if run_agentic:
        st.caption("🧠 **Agentic Deep Analysis** ON — Plan→Execute→Synthesize pipeline")
    if run_consensus:
        st.caption("🔀 **Multi-Model Consensus** ON")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about your data...")

    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        data_context = build_data_context(
            df=df, uploaded_name=uploaded_name,
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            engineered_features=engineered_features,
            imputation_report=imputation_report,
            skew_df=skew_df, date_cols=date_cols,
            user_question=user_q,
        )
        max_ctx = 4000 if selected_model in CLOUD_MODELS else 3500
        if len(data_context) > max_ctx:
            data_context = data_context[:max_ctx] + "\n... [truncated]"

        # Check for visualization intent
        if detect_viz_intent(user_q):
            with st.chat_message("assistant"):
                with st.spinner("Generating visualization..."):
                    viz_result = generate_visualization(
                        user_q, df, selected_model, numeric_cols, categorical_cols,
                    )
                if viz_result["success"]:
                    for fig in viz_result["figures"]:
                        st.pyplot(fig)
                        plt.close(fig)
                    with st.expander("📝 Generated Code"):
                        st.code(viz_result["code"], language="python")
                    response = f"📊 Visualization generated!\n\n```python\n{viz_result['code']}\n```"
                else:
                    st.warning(f"Couldn't generate chart: {viz_result['error']}")
                    st.markdown("Falling back to text response...")
                    response = st.write_stream(
                        stream_ollama(selected_model, user_q + "\n\n" + data_context, system=SYSTEM_ROLE)
                    )
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            return

        # Previous conversation context
        prev_ctx = ""
        if len(st.session_state.chat_history) > 2:
            recent = st.session_state.chat_history[-6:]
            prev_ctx = "\n## PREVIOUS CONVERSATION\n"
            for msg in recent[:-1]:
                role = "USER" if msg["role"] == "user" else "ANALYST"
                prev_ctx += f"{role}: {msg['content'][:500]}\n"

        prompt = textwrap.dedent(f"""\
You are a senior data analyst. Think step-by-step.

## USER'S QUESTION
{user_q}

## DATA CONTEXT
{data_context}
{prev_ctx}

## RESPONSE FORMAT:
### 📌 DIRECT ANSWER
### 🔍 KEY INSIGHTS (3-5)
### ✅ SOLUTION & RECOMMENDATIONS
### ⚠️ DATA QUALITY WARNINGS
### 💡 FOLLOW-UP ANALYSIS
### 🎯 CONFIDENCE LEVEL
CRITICAL: Use SPECIFIC numbers. Be concrete and actionable.
""")

        if selected_model.startswith("("):
            with st.chat_message("assistant"):
                st.error("No model available.")

        elif run_agentic:
            # ── Agentic Pipeline ──
            with st.chat_message("assistant"):
                with st.spinner("🧠 Step 1/3 — Planning analysis..."):
                    plan_text, steps = agentic_plan(user_q, data_context)

                with st.expander("📋 Analysis Plan", expanded=False):
                    st.markdown(plan_text)

                with st.spinner(f"⚡ Step 2/3 — Executing {len(steps)} sub-tasks..."):
                    exec_results = agentic_execute(steps, data_context)

                for i, r in enumerate(exec_results, 1):
                    with st.expander(f"Sub-task {i}: {r['step'][:80]}...", expanded=False):
                        st.markdown(r["result"])

                with st.spinner("✨ Step 3/3 — Synthesizing findings..."):
                    response = agentic_synthesize(user_q, plan_text, exec_results)

                st.markdown("### ✨ Synthesized Answer")
                st.markdown(response)

            st.session_state.chat_history.append({"role": "assistant", "content": response})

        elif run_consensus:
            # ── Multi-Model Consensus ──
            valid = [m for m in available_models if not m.startswith("(")][:3]
            with st.chat_message("assistant"):
                if len(valid) < 2:
                    response = st.write_stream(stream_ollama(selected_model, prompt, system=SYSTEM_ROLE))
                else:
                    responses = {}
                    prev_mdl = None
                    for mdl in valid:
                        if prev_mdl and prev_mdl not in CLOUD_MODELS:
                            unload_ollama_model(prev_mdl)
                        label = f"☁️ {mdl}" if mdl in CLOUD_MODELS else f"💻 {mdl}"
                        with st.spinner(f"Querying {label}..."):
                            t = TIMEOUT_CLOUD if mdl in CLOUD_MODELS else TIMEOUT_LOCAL
                            responses[mdl] = query_ollama(mdl, prompt, system=SYSTEM_ROLE, timeout=t)
                        prev_mdl = mdl
                    if prev_mdl and prev_mdl not in CLOUD_MODELS:
                        unload_ollama_model(prev_mdl)

                    for mdl, resp in responses.items():
                        label = f"☁️ {mdl}" if mdl in CLOUD_MODELS else f"💻 {mdl}"
                        with st.expander(f"Response from {label}", expanded=False):
                            st.markdown(resp)

                    synth_p = f"Synthesize answers from {len(responses)} models to: '{user_q}'\n\n"
                    for mdl, resp in responses.items():
                        synth_p += f"### {mdl}\n{resp[:1500]}\n\n"
                    synth_p += "Synthesize: agreements, disagreements, best combined answer."
                    st.markdown("---\n### 🔀 Consensus")
                    response = st.write_stream(stream_ollama(selected_model, synth_p, system=SYSTEM_ROLE))

                st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            # ── Single Model ──
            with st.chat_message("assistant"):
                response = st.write_stream(stream_ollama(selected_model, prompt, system=SYSTEM_ROLE))
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    if not st.session_state.chat_history:
        st.info(
            "💬 **Try asking:**\n"
            "- *What are the main patterns in this data?*\n"
            "- *Show me a scatter plot of age vs income*\n"
            "- *Which features predict churn?*\n"
            "- *Are there anomalies I should worry about?*\n"
            "- *Compare top categories by revenue*"
        )


# ════════════════════════════════════════════════
# NEW: Report Export
# ════════════════════════════════════════════════
def render_report_export(df, uploaded_name, numeric_cols, categorical_cols,
                         ai_summary, anomaly_count, ml_results, section_num):
    st.header(f"📄 {section_num}. Export Report")

    if st.button("📥 Generate HTML Report", key="gen_report"):
        with st.spinner("Generating report..."):
            html = generate_html_report(
                df=df, uploaded_name=uploaded_name,
                numeric_cols=numeric_cols, categorical_cols=categorical_cols,
                ai_summary=ai_summary, anomaly_count=anomaly_count,
                ml_results=ml_results,
            )
        st.download_button(
            "⬇️ Download Report",
            html, f"eda_report_{uploaded_name.split('.')[0]}.html",
            "text/html",
            key="download_report",
        )
        st.success("Report generated! Click above to download.")


# ════════════════════════════════════════════════
# Landing Page
# ════════════════════════════════════════════════
def render_landing_page():
    st.markdown("---")
    st.markdown("### 🌟 Multi-Modal Capabilities")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📊 Tabular ML")
        st.markdown("Upload CSV, Excel, Parquet, or SQLite. Get Auto-ML, Feature Engineering, and SHAP Explainability.")
    with c2:
        st.markdown("#### 📝 NLP Engine")
        st.markdown("Upload PDF or TXT. Get Flesch-Kincaid Readability, N-Grams, Lemmatization, and Sentiment Analysis.")
    with c3:
        st.markdown("#### 👁️ Computer Vision")
        st.markdown("Upload PNG or JPG. Get KMeans Color Palettes, Image Quality Scoring, and EasyOCR Extraction.")
    st.markdown("---")
    st.info("👈 Upload any file format from the sidebar to automatically trigger the correct dashboard.")
