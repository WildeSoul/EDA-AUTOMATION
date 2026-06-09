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
    render_agentic_orchestrator,
    render_memory_knowledge,
    render_telemetry_observability,
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
# Sidebar Workspace Navigation
# ──────────────────────────────────────────────


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
# ──────────────────────────────────────────────
# Sidebar Workspace Navigation
# ──────────────────────────────────────────────
st.sidebar.markdown("### 🗺️ Workspace Navigation")

# Initialize current_page in session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "🚀 Ingest & Profile"

page_options = [
    "🚀 Ingest & Profile",
    "🔬 Specialty Labs",
    "🧠 Agent Orchestrator (DAG)",
    "🗂️ Memory & Knowledge",
    "📊 Telemetry & Observability"
]

default_index = 0
if st.session_state.current_page in page_options:
    default_index = page_options.index(st.session_state.current_page)

selected_page = st.sidebar.radio(
    label="Go to:",
    options=page_options,
    index=default_index,
    key="navigation_radio"
)
st.session_state.current_page = selected_page

# ──────────────────────────────────────────────
# Sidebar Configurations & Settings
# ──────────────────────────────────────────────
cfg = render_sidebar()

# ──────────────────────────────────────────────
# Main Application Flow
# ──────────────────────────────────────────────
if st.session_state.current_page == "🗂️ Memory & Knowledge":
    render_memory_knowledge()
elif st.session_state.current_page == "📊 Telemetry & Observability":
    render_telemetry_observability()
else:
    if cfg["uploaded_file"] is None:
        render_landing_page()
    else:
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
                cfg["uploaded_file"].seek(0)

            # Determine if we should route to Unstructured Dashboard
            is_unstructured = False
            if cfg["analysis_mode"] == "Force Unstructured (NLP/Image)":
                is_unstructured = True
            elif cfg["analysis_mode"] == "Auto-Detect":
                if fname.endswith((".pdf", ".txt", ".png", ".jpg", ".jpeg")):
                    is_unstructured = True

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
                st.stop()

            # Initialize session if needed
            if "session_id" not in st.session_state or st.session_state.get("filename") != fname or st.session_state.get("sheet_name") != sheet_name:
                import uuid
                from data.session_store import SessionStore
                from data.intelligence import build_dataset_intelligence
                from data.duckdb_engine import DuckDBEngine
                from data.memory_store import MemoryStore
                
                sid = str(uuid.uuid4())[:8]
                st.session_state.session_id = sid
                st.session_state.filename = fname
                st.session_state.sheet_name = sheet_name
                
                db = SessionStore()
                
                with st.spinner("🧠 Initializing Dataset Intelligence..."):
                    df = load_data(cfg["uploaded_file"], fname, sheet_name)
                    if len(df) > 50000:
                        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
                    st.session_state.df = df
                    st.session_state.df_original = df.copy()
                    
                    intelligence = build_dataset_intelligence(df, fname)
                st.session_state.intelligence = intelligence
                
                # Version control & ChromaDB
                version = 1
                try:
                    from data.intelligence import compute_fingerprint
                    fingerprint = intelligence.get("dataset_fingerprint") or compute_fingerprint(df)
                    intelligence["dataset_fingerprint"] = fingerprint
                    version = db.register_dataset_version(fingerprint, fname)
                    intelligence["version"] = version
                    
                    store = MemoryStore()
                    store.add_dataset_profile(sid, fingerprint, fname, intelligence, version)
                    
                    # Link
                    similar = store.search_similar_datasets(intelligence, limit=5)
                    for sim in similar:
                        sim_score = sim["similarity"]
                        sim_fingerprint = sim["fingerprint"]
                        if sim_fingerprint and sim_fingerprint != fingerprint:
                            db.add_dataset_link(fingerprint, sim_fingerprint, "similar_schema", sim_score)
                except Exception as ver_err:
                    st.warning(f"Failed to register dataset version: {ver_err}")
                    
                # Create session & save DataFrame
                try:
                    db.create_session(sid, fname, len(df), len(df.columns), intelligence)
                    db.save_dataframe(sid, df)
                except Exception as db_err:
                    st.warning(f"Database error: {db_err}")
                    
                # DuckDB registration
                try:
                    DuckDBEngine().register(df)
                except Exception as ddb_err:
                    st.warning(f"DuckDB registration error: {ddb_err}")
            
            df = st.session_state.df
            sid = st.session_state.session_id
            intelligence = st.session_state.intelligence
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            from analysis.features import detect_date_columns
            date_cols = detect_date_columns(df)

            # RENDER ACTIVE PAGE
            if st.session_state.current_page == "🚀 Ingest & Profile":
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
                render_data_inspection(df)
                
                # Guided Path Next Steps
                st.markdown("---")
                st.markdown("### 💡 Guided Path: Next Steps")
                st.info("Choose a workspace mode below to proceed with your analysis:")
                
                col_chat, col_dag, col_stats = st.columns(3)
                if col_chat.button("💬 Conversational AI Chat & SQL", use_container_width=True):
                    st.session_state.current_page = "🧠 Agent Orchestrator (DAG)"
                    st.session_state.dag_tab_active = False
                    st.rerun()
                if col_dag.button("⚙️ Agentic Orchestration DAG", use_container_width=True):
                    st.session_state.current_page = "🧠 Agent Orchestrator (DAG)"
                    st.session_state.dag_tab_active = True
                    st.rerun()
                if col_stats.button("🔬 Open Specialty Labs", use_container_width=True):
                    st.session_state.current_page = "🔬 Specialty Labs"
                    st.rerun()

            elif st.session_state.current_page == "🔬 Specialty Labs":
                labs_tabs = st.tabs([
                    "📈 Statistical Profiling",
                    "📐 Advanced Analytics",
                    "🔍 Anomaly Detection",
                    "🤖 Predictive ML Modeling",
                    "🐍 Python AI Code Lab"
                ])
                
                with labs_tabs[0]:
                    render_statistical_profiling(df, numeric_cols, categorical_cols)
                    
                with labs_tabs[1]:
                    # Run feature engineering before advanced analytics
                    engineered_features = []
                    imputation_report = {}
                    skew_df = pd.DataFrame()
                    if cfg["run_feature_eng"]:
                        df, engineered_features, imputation_report, skew_df, date_cols, numeric_cols = \
                            render_feature_engineering(df, numeric_cols, categorical_cols)
                    pca_data, pca_idx = render_advanced_stats(df, numeric_cols, categorical_cols, 1)
                    target_col = render_advanced_analysis(df, numeric_cols, categorical_cols, 2)
                    
                with labs_tabs[2]:
                    render_anomaly_detection(df, numeric_cols, 1)
                    
                with labs_tabs[3]:
                    render_ml_modeling(df, numeric_cols, categorical_cols, 1)
                    
                with labs_tabs[4]:
                    render_code_lab(df, numeric_cols, categorical_cols, cfg["selected_model"], 1)

            elif st.session_state.current_page == "🧠 Agent Orchestrator (DAG)":
                dag_tab_active = st.session_state.get("dag_tab_active", False)
                orchestrator_tabs = st.tabs(["💬 Conversational Chat & SQL", "⚙️ Agentic Orchestration (DAG)"])
                
                with orchestrator_tabs[0]:
                    st.session_state.dag_tab_active = False
                    render_qa_chat(
                        df, fname, numeric_cols, categorical_cols,
                        None, None, None, date_cols,
                        cfg["selected_model"], cfg["available_models"], 1,
                        run_agentic=cfg["run_agentic"],
                        run_consensus=cfg["run_consensus"],
                    )
                with orchestrator_tabs[1]:
                    st.session_state.dag_tab_active = True
                    render_agentic_orchestrator(df, sid, cfg["selected_model"])

        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.exception(e)
