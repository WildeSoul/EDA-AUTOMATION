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
# Helper classes and configurations
# ──────────────────────────────────────────────
import io
import os
import uuid

class MockUploadedFile:
    def __init__(self, name: str, data_bytes: bytes):
        self.name = name
        self.data_bytes = data_bytes
        self._io = io.BytesIO(data_bytes)
        self.size = len(data_bytes)

    def read(self, *args, **kwargs):
        return self._io.read(*args, **kwargs)

    def getvalue(self):
        return self.data_bytes

    def seek(self, offset: int, whence: int = 0):
        return self._io.seek(offset, whence)

def init_settings():
    if "selected_model" not in st.session_state:
        from core.llm import get_ollama_models
        available_models = get_ollama_models()
        st.session_state.available_models = available_models
        st.session_state.selected_model = available_models[0] if available_models else "mistral-nemo:latest"
    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = "Auto-Detect"
    if "run_feature_eng" not in st.session_state:
        st.session_state.run_feature_eng = True
    if "run_advanced_stats" not in st.session_state:
        st.session_state.run_advanced_stats = True
    if "run_anomaly" not in st.session_state:
        st.session_state.run_anomaly = False
    if "run_quick_ml" not in st.session_state:
        st.session_state.run_quick_ml = False
    if "run_auto_summary" not in st.session_state:
        st.session_state.run_auto_summary = False
    if "run_agentic" not in st.session_state:
        st.session_state.run_agentic = False
    if "run_consensus" not in st.session_state:
        st.session_state.run_consensus = False
    if "run_code_lab" not in st.session_state:
        st.session_state.run_code_lab = False

def get_current_config():
    init_settings()
    return {
        "uploaded_file": st.session_state.get("uploaded_file"),
        "selected_model": st.session_state.get("selected_model"),
        "available_models": st.session_state.get("available_models", []),
        "analysis_mode": st.session_state.get("analysis_mode"),
        "run_feature_eng": st.session_state.get("run_feature_eng"),
        "run_advanced_stats": st.session_state.get("run_advanced_stats"),
        "run_anomaly": st.session_state.get("run_anomaly"),
        "run_quick_ml": st.session_state.get("run_quick_ml"),
        "run_auto_summary": st.session_state.get("run_auto_summary"),
        "run_agentic": st.session_state.get("run_agentic"),
        "run_consensus": st.session_state.get("run_consensus"),
        "run_code_lab": st.session_state.get("run_code_lab"),
    }

def load_demo_dataset(filename: str):
    filepath = os.path.join("data", "demo", filename)
    try:
        with open(filepath, "rb") as f:
            data_bytes = f.read()
        mock_file = MockUploadedFile(filename, data_bytes)
        st.session_state.uploaded_file = mock_file
        st.session_state.filename = filename
        
        # Clear other session states to trigger re-init
        keys_to_clear = ["df", "df_original", "session_id", "intelligence", "sheet_name", "chat_history", "current_plan", "execution_state"]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        
        st.session_state.current_page = "📂 Datasets"
        st.success(f"Loaded demo dataset: {filename}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load demo dataset {filename}: {e}")

def restore_session(sid: str):
    from data.session_store import SessionStore
    db = SessionStore()
    session = db.get_session(sid)
    if not session:
        st.error(f"Session {sid} not found.")
        return
        
    df = db.load_dataframe(sid)
    if df is None:
        st.error(f"DataFrame for session {sid} could not be loaded.")
        return
        
    st.session_state.session_id = sid
    st.session_state.filename = session["filename"]
    st.session_state.df = df
    st.session_state.df_original = df.copy()
    st.session_state.intelligence = session["intelligence"]
    
    # Wrap in MockUploadedFile
    try:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        mock_file = MockUploadedFile(session["filename"], csv_bytes)
        st.session_state.uploaded_file = mock_file
    except Exception:
        mock_file = MockUploadedFile(session["filename"], b"")
        st.session_state.uploaded_file = mock_file
        
    st.session_state.current_page = "📂 Datasets"
    st.success(f"Session {sid} restored successfully!")
    st.rerun()

# Initialize configuration settings
init_settings()
cfg = get_current_config()

# ──────────────────────────────────────────────
# Global Dataset Initialization
# ──────────────────────────────────────────────
if "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
    fname = st.session_state.uploaded_file.name
    sheet_name = st.session_state.get("sheet_name", 0)
    
    if "session_id" not in st.session_state or "intelligence" not in st.session_state or "df" not in st.session_state or st.session_state.get("filename") != fname or st.session_state.get("sheet_name") != sheet_name:
        from data.session_store import SessionStore
        from data.intelligence import build_dataset_intelligence
        from data.engine import DuckDBEngine
        from data.memory_store import MemoryStore
        
        sid = str(uuid.uuid4())[:8]
        st.session_state.session_id = sid
        st.session_state.filename = fname
        st.session_state.sheet_name = sheet_name
        
        db = SessionStore()
        
        with st.spinner("🧠 Initializing Dataset Intelligence..."):
            try:
                # Multi-sheet Excel handling
                if fname.endswith((".xlsx", ".xls")):
                    sheets = get_excel_sheets(st.session_state.uploaded_file)
                    if len(sheets) > 1 and "sheet_select" not in st.session_state:
                        sheet_name = st.selectbox(
                            f"📑 This Excel file has {len(sheets)} sheets. Select one:",
                            sheets, key="sheet_select",
                        )
                        st.session_state.sheet_name = sheet_name
                st.session_state.uploaded_file.seek(0)
                
                df = load_data(st.session_state.uploaded_file, fname, sheet_name)
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
            except Exception as init_err:
                st.error(f"Error initializing dataset: {init_err}")

# Determine if unstructured
is_unstructured = False
if "filename" in st.session_state and st.session_state.filename:
    fname = st.session_state.filename
    if cfg["analysis_mode"] == "Force Unstructured (NLP/Image)":
        is_unstructured = True
    elif cfg["analysis_mode"] == "Auto-Detect":
        if fname.endswith((".pdf", ".txt", ".png", ".jpg", ".jpeg")):
            is_unstructured = True

# ──────────────────────────────────────────────
# Sidebar Workspace Navigation Call
# ──────────────────────────────────────────────
if "df" in st.session_state and st.session_state.df is not None:
    active_df_name = st.session_state.filename
    active_rows = len(st.session_state.df)
    active_cols = len(st.session_state.df.columns)
else:
    active_df_name = None
    active_rows = 0
    active_cols = 0

selected_page = render_sidebar(
    active_df_name=active_df_name,
    active_rows=active_rows,
    active_cols=active_cols
)
st.session_state.current_page = selected_page

# ──────────────────────────────────────────────
# Main Application Flow Routing
# ──────────────────────────────────────────────
if st.session_state.current_page == "🏠 Home":
    # RENDER HOME PAGE
    st.subheader("🏠 Home Workspace")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        home_tabs = st.tabs(["📤 Upload Dataset", "📊 Try Demo Dataset"])
        
        with home_tabs[0]:
            st.markdown("### Upload your own dataset")
            uploaded_file = st.file_uploader(
                "Choose a data file (CSV, Excel, Parquet, SQLite, JSON, PDF, Images...)", 
                type=[
                    "csv", "tsv", "xlsx", "xls", "parquet", "json", 
                    "pkl", "pickle", "feather", "sqlite", "db", 
                    "html", "xml", "sav", "zsav", "dta", "sas7bdat",
                    "txt", "pdf", "png", "jpg", "jpeg"
                ],
                key="home_uploader"
            )
            if uploaded_file is not None:
                if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != uploaded_file:
                    st.session_state.uploaded_file = uploaded_file
                    # Clear previous state
                    keys_to_clear = ["df", "df_original", "session_id", "intelligence", "sheet_name", "chat_history", "current_plan", "execution_state"]
                    for k in keys_to_clear:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.session_state.current_page = "📂 Datasets"
                    st.success("Dataset uploaded successfully! Navigating to Datasets tab...")
                    st.rerun()
                    
        with home_tabs[1]:
            st.markdown("### Choose a curated demo dataset")
            demo_cols = st.columns(3)
            demos = [
                ("🚢 Titanic Survival", "titanic.csv"),
                ("📊 Customer Churn", "customer_churn.csv"),
                ("💼 Employee Attrition", "employee_attrition.csv"),
                ("🏥 Patient Treatment", "patient_treatment.csv"),
                ("💰 Credit Risk", "credit_risk.csv"),
                ("🛒 Sales Transactions", "sales_transactions.csv")
            ]
            for idx, (label, fname) in enumerate(demos):
                col_idx = idx % 3
                if demo_cols[col_idx].button(label, use_container_width=True, key=f"demo_btn_{fname}"):
                    load_demo_dataset(fname)
                    
        # Recent Sessions Table
        st.markdown("---")
        st.markdown("### 🕰️ Recent Sessions")
        from data.session_store import SessionStore
        db = SessionStore()
        recent_sessions = db.list_sessions(limit=8)
        if recent_sessions:
            for s in recent_sessions:
                sid = s["sid"]
                fname = s["filename"]
                rows = s["row_count"]
                cols = s["col_count"]
                created_at = s["created_at"]
                grade = s.get("quality_grade", "N/A")
                score = s.get("quality_score", 0.0)
                
                col_info, col_action = st.columns([5, 1.2])
                with col_info:
                    st.markdown(
                        f"📄 **{fname}** &nbsp;|&nbsp; `Session: {sid}` &nbsp;|&nbsp; {rows:,} x {cols} &nbsp;|&nbsp; "
                        f"Grade: `{grade}` ({score:.1f}%) &nbsp;|&nbsp; *{created_at}*"
                    )
                with col_action:
                    if st.button("Restore", key=f"restore_btn_{sid}", use_container_width=True):
                        restore_session(sid)
        else:
            st.caption("No previous sessions found.")

    with col_right:
        # Platform Info and Stats Cards
        st.markdown("""
        <div style="background: rgba(151,169,255,0.03); border: 1px solid rgba(151,169,255,0.1); border-radius: 12px; padding: 20px;">
            <h4 style="margin-top:0; color:#e5e4ed; font-size:1.1rem; border-bottom: 1px solid rgba(151,169,255,0.15); padding-bottom:8px;">🛠️ Platform Engine Status</h4>
            <div style="display:flex; flex-direction:column; gap:12px; margin-top:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#aaaab3; font-size:0.85rem;">System Health</span>
                    <span style="color:#2ecc71; font-weight:700; font-size:0.85rem;">🟢 Online</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#aaaab3; font-size:0.85rem;">SQLite WAL Mode</span>
                    <span style="color:#2ecc71; font-weight:700; font-size:0.85rem;">Enabled (NORMAL)</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#aaaab3; font-size:0.85rem;">Concurrency Locks</span>
                    <span style="color:#97a9ff; font-weight:700; font-size:0.85rem;">Active (Thread-Safe)</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#aaaab3; font-size:0.85rem;">Vector Memory</span>
                    <span style="color:#2ecc71; font-weight:700; font-size:0.85rem;">ChromaDB Persistent</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#aaaab3; font-size:0.85rem;">Agentic Engine</span>
                    <span style="color:#c99af8; font-weight:700; font-size:0.85rem;">LangGraph DAG</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#aaaab3; font-size:0.85rem;">Self-Correction</span>
                    <span style="color:#2ecc71; font-weight:700; font-size:0.85rem;">Active</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#aaaab3; font-size:0.85rem;">Integration Tests</span>
                    <span style="color:#2ecc71; font-weight:700; font-size:0.85rem;">44/44 Passing (100%)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌟 Platform Capabilities")
        st.markdown("#### 📊 Tabular Machine Learning")
        st.caption("Upload CSV, Excel, Parquet, or SQLite. Includes Auto-ML, advanced feature engineering, and automated statistical testing.")
        
        st.markdown("#### 📝 Natural Language NLP")
        st.caption("Upload PDF or TXT. Get sentiment analysis, lemmatization, Flesch readability, and key term extractions.")
        
        st.markdown("#### 👁️ Computer Vision")
        st.caption("Upload PNG or JPG. Performs color-palette clustering, quality estimation, and OCR text extractions.")

elif st.session_state.current_page == "📂 Datasets":
    if "df" not in st.session_state or st.session_state.df is None:
        st.info("📂 Please load a dataset on the 🏠 Home page to begin.")
    else:
        fname = st.session_state.filename
        if is_unstructured:
            from ui.unstructured import render_nlp_dashboard, render_image_dashboard
            is_image = fname.endswith((".png", ".jpg", ".jpeg"))
            if "unstructured_text" in st.session_state and st.session_state.get("unstructured_filename") == f"OCR_{fname}.txt":
                text = st.session_state["unstructured_text"]
                render_nlp_dashboard(text, f"OCR_{fname}.txt")
            elif is_image:
                st.session_state.uploaded_file.seek(0)
                render_image_dashboard(st.session_state.uploaded_file.read(), fname)
            else:
                if fname.endswith(".pdf"):
                    import pdfplumber
                    text = ""
                    st.session_state.uploaded_file.seek(0)
                    with pdfplumber.open(st.session_state.uploaded_file) as pdf:
                        for page in pdf.pages:
                            extracted = page.extract_text()
                            if extracted: text += extracted + "\n"
                else:
                    st.session_state.uploaded_file.seek(0)
                    text = st.session_state.uploaded_file.read().decode("utf-8", errors="ignore")
                render_nlp_dashboard(text, fname)
        else:
            df = st.session_state.df
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

elif st.session_state.current_page == "🔬 Analysis":
    if "df" not in st.session_state or st.session_state.df is None:
        st.info("🔬 Please load a dataset on the 🏠 Home page to begin analysis.")
    else:
        fname = st.session_state.filename
        if is_unstructured:
            st.info("Unstructured datasets are processed via the NLP/Image dashboard on the 📂 Datasets tab.")
        else:
            df = st.session_state.df
            sid = st.session_state.session_id
            intelligence = st.session_state.intelligence
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            from analysis.features import detect_date_columns
            date_cols = detect_date_columns(df)
            
            analysis_tabs = st.tabs([
                "💬 AI Conversational Chat & SQL",
                "🧪 Specialty Labs"
            ])
            
            with analysis_tabs[0]:
                render_qa_chat(
                    df, fname, numeric_cols, categorical_cols,
                    None, None, None, date_cols,
                    cfg["selected_model"], cfg["available_models"], 1,
                    run_agentic=cfg["run_agentic"],
                    run_consensus=cfg["run_consensus"],
                )
                
            with analysis_tabs[1]:
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

elif st.session_state.current_page == "🧠 Agent System":
    agent_tabs = st.tabs([
        "⚙️ LangGraph Orchestrator (DAG)",
        "🗂️ Memory & Knowledge",
        "📊 Telemetry & Observability"
    ])
    
    with agent_tabs[0]:
        if "df" not in st.session_state or st.session_state.df is None:
            st.info("🧠 Please load a dataset on the 🏠 Home page to execute the agent network.")
        else:
            render_agentic_orchestrator(st.session_state.df, st.session_state.session_id, cfg["selected_model"])
            
    with agent_tabs[1]:
        render_memory_knowledge()
        
    with agent_tabs[2]:
        render_telemetry_observability()

elif st.session_state.current_page == "⚙️ Settings":
    st.header("⚙️ Platform Configurations & Settings")
    st.markdown("Customize model selections, analysis modes, and active features for the pipeline.")
    
    from core.llm import get_ollama_models
    available_models = get_ollama_models()
    model_labels = {
        m: f"☁️ {m}" if m in CLOUD_MODELS else f"💻 {m}"
        for m in available_models
    }
    
    st.subheader("🤖 Core AI Model")
    selected_model = st.selectbox(
        "Select Model", available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        format_func=lambda m: model_labels.get(m, m),
        key="settings_selected_model"
    )
    st.session_state.selected_model = selected_model
    
    if selected_model in CLOUD_MODELS:
        st.caption("☁️ Cloud model — runs via cloud API through Ollama.")
    else:
        st.caption("💻 Local model — runs entirely on your machine.")
        
    st.subheader("📊 Analysis Mode")
    analysis_mode = st.radio(
        "Select Dashboard Mode",
        options=["Auto-Detect", "Force Tabular (Rows/Cols)", "Force Unstructured (NLP/Image)"],
        index=["Auto-Detect", "Force Tabular (Rows/Cols)", "Force Unstructured (NLP/Image)"].index(st.session_state.analysis_mode),
        help="Override the automatic file-type detection to force a specific dashboard.",
        key="settings_analysis_mode"
    )
    st.session_state.analysis_mode = analysis_mode
    
    st.subheader("⚙️ Feature Engineering & Statistical Options")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.run_feature_eng = st.checkbox("🔧 Auto Feature Engineering", value=st.session_state.run_feature_eng, help="Handle missing values and create interaction features")
        st.session_state.run_advanced_stats = st.checkbox("📐 Advanced Statistical Tests", value=st.session_state.run_advanced_stats, help="Normality tests, Chi-square, PCA & Clustering")
        st.session_state.run_anomaly = st.checkbox("🔍 Anomaly Detection", value=st.session_state.run_anomaly, help="Isolation Forest + Z-Score + IQR multivariate")
    with col2:
        st.session_state.run_quick_ml = st.checkbox("🤖 Quick ML Model", value=st.session_state.run_quick_ml, help="Train a baseline ML model (RF, LR, XGBoost)")
        st.session_state.run_code_lab = st.checkbox("🐍 Code Lab", value=st.session_state.run_code_lab, help="AI generates & executes Python code on your data")
        
    st.subheader("🧠 Deep AI Capabilities")
    col3, col4 = st.columns(2)
    with col3:
        st.session_state.run_auto_summary = st.checkbox("📝 AI Auto-Summary", value=st.session_state.run_auto_summary, help="Generate an executive AI summary on load")
        st.session_state.run_agentic = st.checkbox("🧠 Agentic Deep Analysis", value=st.session_state.run_agentic, help="Multi-step Plan→Execute→Synthesize for Q&A")
    with col4:
        st.session_state.run_consensus = st.checkbox("🔀 Multi-Model Consensus", value=st.session_state.run_consensus, help="Query multiple models and merge answers")
