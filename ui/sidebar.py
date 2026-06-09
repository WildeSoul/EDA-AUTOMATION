"""
Sidebar components — file upload, model selection, options.
"""
import streamlit as st
from core.llm import get_ollama_models
from config.settings import CLOUD_MODELS


def render_sidebar() -> dict:
    """
    Render the sidebar and return a dict of user selections.
    """
    with st.sidebar:
        st.header("1️⃣  Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a data file", 
            type=[
                "csv", "tsv", "xlsx", "xls", "parquet", "json", 
                "pkl", "pickle", "feather", "sqlite", "db", 
                "html", "xml", "sav", "zsav", "dta", "sas7bdat",
                "txt", "pdf", "png", "jpg", "jpeg"
            ]
        )
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 100:
                st.warning(
                    f"⚠️ Large file detected ({file_size_mb:.0f} MB). "
                    "Consider enabling data sampling below for faster analysis."
                )
            st.caption(f"📎 {uploaded_file.name} — {file_size_mb:.1f} MB")

        st.header("2️⃣  AI Model")
        available_models = get_ollama_models()
        model_labels = {
            m: f"☁️ {m}" if m in CLOUD_MODELS else f"💻 {m}"
            for m in available_models
        }
        selected_model = st.selectbox(
            "Select Model", available_models,
            format_func=lambda m: model_labels.get(m, m),
        )
        if selected_model in CLOUD_MODELS:
            st.caption("☁️ Cloud model — runs via cloud API through Ollama.")
        else:
            st.caption("💻 Local model — runs entirely on your machine.")

        st.header("3️⃣  Analysis Mode")
        analysis_mode = st.radio(
            "Select Dashboard Mode",
            options=["Auto-Detect", "Force Tabular (Rows/Cols)", "Force Unstructured (NLP/Image)"],
            help="Override the automatic file-type detection to force a specific dashboard."
        )

        st.header("4️⃣  Analysis Options")
        run_feature_eng = st.checkbox("🔧 Auto Feature Engineering", value=True)
        run_advanced_stats = st.checkbox(
            "📐 Advanced Statistical Tests", value=True,
            help="Normality tests, Chi-square, PCA & Clustering",
        )
        run_anomaly = st.checkbox(
            "🔍 Anomaly Detection", value=False,
            help="Isolation Forest + Z-Score + IQR multivariate",
        )
        run_quick_ml = st.checkbox(
            "🤖 Quick ML Model", value=False,
            help="Train a baseline ML model (RF, LR, XGBoost)",
        )

        st.header("4️⃣  AI Features")
        run_auto_summary = st.checkbox(
            "📝 AI Auto-Summary", value=False,
            help="Generate an executive AI summary on load",
        )
        run_agentic = st.checkbox(
            "🧠 Agentic Deep Analysis", value=False,
            help="Multi-step Plan→Execute→Synthesize for Q&A",
        )
        run_consensus = st.checkbox(
            "🔀 Multi-Model Consensus", value=False,
            help="Query multiple models and merge answers",
        )
        run_code_lab = st.checkbox(
            "🐍 Code Lab", value=False,
            help="AI generates & executes Python code on your data",
        )

    return {
        "uploaded_file": uploaded_file,
        "selected_model": selected_model,
        "available_models": available_models,
        "analysis_mode": analysis_mode,
        "run_feature_eng": run_feature_eng,
        "run_advanced_stats": run_advanced_stats,
        "run_anomaly": run_anomaly,
        "run_quick_ml": run_quick_ml,
        "run_auto_summary": run_auto_summary,
        "run_agentic": run_agentic,
        "run_consensus": run_consensus,
        "run_code_lab": run_code_lab,
    }
