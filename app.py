"""
AI Data Analyst Pro -- Main Application Entry Point
=====================================================
Industrial-ready modular EDA automation with AI-powered analysis.
Synthetix Dark theme from Google Stitch design system.

Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np

from ui.styles import apply_styles
from ui.sidebar import render_sidebar
from ui.sections import (
    render_data_inspection,
    render_statistical_profiling,
    render_feature_engineering,
    render_visualizations,
    render_advanced_stats,
    render_advanced_analysis,
    render_anomaly_detection,
    render_ml_modeling,
    render_code_lab,
    render_ai_summary,
    render_qa_chat,
    render_report_export,
    render_landing_page,
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:0.5rem;">
    <div style="width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,#97a9ff,#c99af8);display:flex;align-items:center;justify-content:center;box-shadow:0 4px 20px rgba(151,169,255,0.25);">
        <span style="font-size:24px;color:#002283;">&#x1F4CA;</span>
    </div>
    <div>
        <h1 style="margin:0;padding:0;font-size:1.8rem;line-height:1.1;">AI Data Analyst Pro</h1>
        <p style="margin:0;color:#aaaab3;font-size:0.8rem;letter-spacing:0.06em;text-transform:uppercase;font-weight:500;">Precision Intelligence</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p style="color:#aaaab3;font-size:0.9rem;margin-top:0;margin-bottom:1.5rem;">'
    'Upload <strong>any dataset</strong> &rarr; full <strong>EDA + Advanced Analysis + Feature Engineering</strong> '
    '&rarr; <strong>AI-powered Q&A, Code Lab, Anomaly Detection, ML Modeling</strong> &rarr; <strong>Export Reports</strong>.</p>',
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
cfg = render_sidebar()


# ──────────────────────────────────────────────
# Data Loading (universal format support)
# ──────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes, file_name, sheet_name=0):
    """Load data from various file formats."""
    import io
    if file_name.endswith(".parquet"):
        return pd.read_parquet(file_bytes)
    elif file_name.endswith(".json"):
        return pd.read_json(file_bytes)
    elif file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_bytes, sheet_name=sheet_name)
    elif file_name.endswith(".tsv"):
        return pd.read_csv(file_bytes, sep='\t')
    elif file_name.endswith(".pkl") or file_name.endswith(".pickle"):
        return pd.read_pickle(file_bytes)
    elif file_name.endswith(".feather"):
        return pd.read_feather(file_bytes)
    elif file_name.endswith((".sqlite", ".db")):
        import sqlite3
        # write temp file for sqlite since it requires a path
        with open("temp.db", "wb") as f:
            f.write(file_bytes.read() if hasattr(file_bytes, "read") else file_bytes)
        conn = sqlite3.connect("temp.db")
        # Get first table
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            raise ValueError("No tables found in SQLite database.")
        df = pd.read_sql_query(f"SELECT * FROM {tables[0][0]}", conn)
        conn.close()
        import os
        if os.path.exists("temp.db"): os.remove("temp.db")
        return df
    elif file_name.endswith(".html"):
        return pd.read_html(file_bytes)[0]
    elif file_name.endswith(".xml"):
        return pd.read_xml(file_bytes)
    elif file_name.endswith((".sav", ".zsav")):
        return pd.read_spss(file_bytes)
    elif file_name.endswith(".dta"):
        return pd.read_stata(file_bytes)
    elif file_name.endswith(".sas7bdat"):
        return pd.read_sas(file_bytes, format='sas7bdat')
    else:
        return pd.read_csv(file_bytes)


def get_excel_sheets(file_bytes):
    """Get sheet names from an Excel file."""
    try:
        xl = pd.ExcelFile(file_bytes)
        return xl.sheet_names
    except Exception:
        return []


# ──────────────────────────────────────────────
# Main Application Flow
# ──────────────────────────────────────────────
if cfg["uploaded_file"] is not None:
    try:
        # Multi-sheet Excel handling
        sheet_name = 0
        fname = cfg["uploaded_file"].name
        if fname.endswith((".xlsx", ".xls")):
            sheets = get_excel_sheets(cfg["uploaded_file"])
            if len(sheets) > 1:
                sheet_name = st.selectbox(
                    f"📑 This Excel file has {len(sheets)} sheets. Select one:",
                    sheets, key="sheet_select",
                )
            cfg["uploaded_file"].seek(0)  # Reset file pointer after reading sheets

        # Determine if we should route to Unstructured Dashboard
        is_unstructured = False
        if cfg["analysis_mode"] == "Force Unstructured (NLP/Image)":
            is_unstructured = True
        elif cfg["analysis_mode"] == "Auto-Detect":
            if fname.endswith((".pdf", ".txt", ".png", ".jpg", ".jpeg")):
                is_unstructured = True
        # If "Force Tabular", is_unstructured remains False

        if is_unstructured:
            from ui.unstructured import render_nlp_dashboard, render_image_dashboard
            
            # Check for OCR handoff
            is_image = fname.endswith((".png", ".jpg", ".jpeg"))
            if "unstructured_text" in st.session_state and st.session_state.get("unstructured_filename") == f"OCR_{fname}.txt":
                text = st.session_state["unstructured_text"]
                render_nlp_dashboard(text, f"OCR_{fname}.txt")
            elif is_image:
                render_image_dashboard(cfg["uploaded_file"].read(), fname)
            else:
                if fname.endswith(".pdf"):
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(cfg["uploaded_file"]) as pdf:
                        for page in pdf.pages:
                            extracted = page.extract_text()
                            if extracted: text += extracted + "\n"
                else:
                    text = cfg["uploaded_file"].read().decode("utf-8", errors="ignore")
                render_nlp_dashboard(text, fname)
                
            # Stop the standard tabular execution
            st.stop()

        df = load_data(cfg["uploaded_file"], fname, sheet_name)

        # Large file sampling dialog
        if len(df) > 50000:
            st.info(
                f"📊 Large dataset detected: **{len(df):,} rows**. "
                "You can sample for faster analysis."
            )
            sample_choice = st.radio(
                "Data handling:",
                ["Use full dataset", "Sample 50,000 rows (faster)"],
                key="sample_choice", horizontal=True,
            )
            if sample_choice.startswith("Sample"):
                df = df.sample(n=50000, random_state=42).reset_index(drop=True)
                st.caption("✅ Using random sample of 50,000 rows")

        df_original = df.copy()

        # Status badge
        fmt = fname.split(".")[-1].upper()
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;">'
            f'<span class="synthetix-badge badge-success">&#x2705; Data Ready</span>'
            f'<span style="color:#aaaab3;font-size:0.8rem;">'
            f'<strong>{fname}</strong> &mdash; '
            f'{df.shape[0]:,} rows &times; {df.shape[1]} columns &mdash; '
            f'{df.memory_usage(deep=True).sum() / 1024:.0f} KB &mdash; '
            f'Format: {fmt}</span></div>',
            unsafe_allow_html=True,
        )

        # Track section numbering
        section = 1

        # -- 1. Data Inspection --
        render_data_inspection(df)
        section += 1

        # -- 2. Statistical Profiling --
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        render_statistical_profiling(df, numeric_cols, categorical_cols)
        section += 1

        # -- 3. Feature Engineering --
        engineered_features = []
        imputation_report = {}
        skew_df = pd.DataFrame()
        date_cols = []

        if cfg["run_feature_eng"]:
            df, engineered_features, imputation_report, skew_df, date_cols, numeric_cols = \
                render_feature_engineering(df, numeric_cols, categorical_cols)
            section += 1

        # -- 4. Visualizations --
        render_visualizations(df, numeric_cols, categorical_cols, date_cols, section)
        section += 1

        # -- 5. Advanced Stats --
        pca_data = None
        pca_idx = None
        if cfg["run_advanced_stats"]:
            pca_data, pca_idx = render_advanced_stats(df, numeric_cols, categorical_cols, section)
            section += 1

        # -- 6. Advanced Analysis --
        target_col = render_advanced_analysis(df, numeric_cols, categorical_cols, section)
        section += 1

        # -- 7. Anomaly Detection --
        anomaly_count = 0
        if cfg["run_anomaly"]:
            anomaly_count = render_anomaly_detection(df, numeric_cols, section)
            section += 1

        # -- 8. Quick ML Model --
        ml_results = None
        if cfg["run_quick_ml"]:
            ml_results = render_ml_modeling(df, numeric_cols, categorical_cols, section)
            section += 1

        # -- 9. Code Lab --
        if cfg["run_code_lab"]:
            render_code_lab(df, numeric_cols, categorical_cols, cfg["selected_model"], section)
            section += 1

        # -- 10. AI Auto-Summary --
        ai_summary = ""
        if cfg["run_auto_summary"]:
            ai_summary = render_ai_summary(
                df, cfg["uploaded_file"].name, numeric_cols, categorical_cols,
                engineered_features, imputation_report, skew_df, date_cols,
                cfg["selected_model"], section,
            )
            section += 1

        # -- 11. AI Q&A Chat --
        render_qa_chat(
            df, cfg["uploaded_file"].name, numeric_cols, categorical_cols,
            engineered_features if cfg["run_feature_eng"] else None,
            imputation_report if cfg["run_feature_eng"] else None,
            skew_df if cfg["run_feature_eng"] else None,
            date_cols,
            cfg["selected_model"], cfg["available_models"], section,
            run_agentic=cfg["run_agentic"],
            run_consensus=cfg["run_consensus"],
        )
        section += 1

        # -- 12. Report Export --
        render_report_export(
            df, cfg["uploaded_file"].name, numeric_cols, categorical_cols,
            ai_summary, anomaly_count, ml_results, section,
        )

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.exception(e)

else:
    render_landing_page()
