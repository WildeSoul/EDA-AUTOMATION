"""
CSS styling for the Streamlit app — Synthetix Dark theme (from Google Stitch design).
Premium dark UI with glassmorphism, gradient accents, and smooth typography.
"""
import streamlit as st


def apply_styles():
    """Inject the Synthetix Dark design system into Streamlit."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

        /* ═══════════════════════════════════════
           SYNTHETIX DARK — Design Tokens
           ═══════════════════════════════════════ */
        :root {
            --surface: #0c0e14;
            --surface-container: #171921;
            --surface-container-low: #11131a;
            --surface-container-high: #1d1f27;
            --surface-container-highest: #23262e;
            --primary: #97a9ff;
            --primary-dim: #8197ff;
            --secondary: #c99af8;
            --tertiary: #c3ffcd;
            --tertiary-dim: #5bef90;
            --error: #ff6e84;
            --on-surface: #e5e4ed;
            --on-surface-variant: #aaaab3;
            --outline: #74757d;
            --outline-variant: #46484f;
        }

        /* ═══════════════════════════════════════
           GLOBAL
           ═══════════════════════════════════════ */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            color: var(--on-surface);
        }
        
        .main .block-container {
            max-width: 1400px;
            padding-top: 2rem;
        }

        /* ═══════════════════════════════════════
           HEADER / TITLE
           ═══════════════════════════════════════ */
        .stMarkdown h1 {
            font-family: 'Manrope', sans-serif !important;
            background: linear-gradient(135deg, #97a9ff 0%, #c99af8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            font-family: 'Manrope', sans-serif !important;
            letter-spacing: -0.01em;
            font-weight: 700;
        }

        /* ═══════════════════════════════════════
           SIDEBAR — Glassmorphism
           ═══════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background: rgba(12, 14, 20, 0.85) !important;
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-right: 1px solid rgba(151, 169, 255, 0.08) !important;
        }

        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2 {
            font-family: 'Manrope', sans-serif !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            background: none;
            -webkit-text-fill-color: var(--on-surface-variant);
            opacity: 0.8;
            margin-bottom: 0.5rem;
        }

        /* Sidebar checkboxes */
        [data-testid="stSidebar"] .stCheckbox label {
            font-size: 0.85rem !important;
            color: var(--on-surface-variant) !important;
        }

        /* ═══════════════════════════════════════
           METRIC CARDS
           ═══════════════════════════════════════ */
        [data-testid="stMetric"] {
            background: var(--surface-container) !important;
            border: 1px solid rgba(70, 72, 79, 0.15);
            border-radius: 12px !important;
            padding: 1rem 1.25rem !important;
            position: relative;
            overflow: hidden;
            transition: background-color 0.2s, transform 0.1s;
        }
        [data-testid="stMetric"]:hover {
            background: var(--surface-container-high) !important;
            transform: translateY(-1px);
        }
        [data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(to right, transparent, rgba(151,169,255,0.3), transparent);
            opacity: 0;
            transition: opacity 0.3s;
        }
        [data-testid="stMetric"]:hover::before { opacity: 1; }

        .stMetricValue {
            font-family: 'Manrope', sans-serif !important;
            font-size: 1.6rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
            color: var(--on-surface) !important;
        }
        .stMetricLabel {
            font-size: 0.75rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--on-surface-variant) !important;
            font-weight: 500;
        }
        .stMetricDelta { font-size: 0.85rem !important; }

        /* ═══════════════════════════════════════
           EXPANDERS — Glass Panel
           ═══════════════════════════════════════ */
        div[data-testid="stExpander"] {
            background: rgba(23, 25, 33, 0.6) !important;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(70, 72, 79, 0.15) !important;
            border-radius: 12px !important;
            overflow: hidden;
        }
        div[data-testid="stExpander"] summary {
            font-family: 'Manrope', sans-serif !important;
            font-weight: 600;
            font-size: 0.9rem;
        }

        /* ═══════════════════════════════════════
           TABS
           ═══════════════════════════════════════ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid rgba(70, 72, 79, 0.2);
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 18px;
            border-radius: 8px 8px 0 0;
            font-weight: 500;
            font-size: 0.85rem;
            transition: all 0.2s;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 2px solid var(--primary) !important;
            color: var(--primary) !important;
        }

        /* ═══════════════════════════════════════
           CHAT
           ═══════════════════════════════════════ */
        [data-testid="stChatMessage"] {
            border-radius: 12px !important;
            border: 1px solid rgba(70, 72, 79, 0.15);
            padding: 1rem !important;
        }
        [data-testid="stChatMessage"][data-testid*="user"] {
            background: rgba(151, 169, 255, 0.08);
            border-color: rgba(151, 169, 255, 0.2);
        }

        /* Chat input */
        .stChatInput {
            border: 1px solid rgba(70, 72, 79, 0.3) !important;
            border-radius: 12px !important;
        }
        .stChatInput:focus-within {
            border-color: rgba(151, 169, 255, 0.5) !important;
            box-shadow: 0 0 0 2px rgba(151, 169, 255, 0.1);
        }

        /* ═══════════════════════════════════════
           BUTTONS
           ═══════════════════════════════════════ */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 600;
            font-family: 'Inter', sans-serif !important;
            letter-spacing: 0.01em;
            transition: all 0.3s ease;
            border: 1px solid rgba(70, 72, 79, 0.2);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(151, 169, 255, 0.2);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #97a9ff, #c99af8) !important;
            color: #002283 !important;
            border: none !important;
            font-weight: 700;
        }

        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #97a9ff 0%, #c99af8 100%) !important;
            color: #002283 !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700;
            padding: 0.6rem 1.5rem;
        }
        .stDownloadButton > button:hover {
            box-shadow: 0 0 20px rgba(151, 169, 255, 0.35);
        }

        /* ═══════════════════════════════════════
           DATA FRAMES / TABLES
           ═══════════════════════════════════════ */
        [data-testid="stDataFrame"] {
            border-radius: 10px !important;
            overflow: hidden;
            border: 1px solid rgba(70, 72, 79, 0.15);
        }

        /* ═══════════════════════════════════════
           ALERTS
           ═══════════════════════════════════════ */
        .stAlert {
            border-radius: 10px !important;
            backdrop-filter: blur(8px);
        }

        /* Success */
        div[data-testid="stAlert"][data-baseweb="notification"] {
            border-radius: 10px !important;
        }

        /* ═══════════════════════════════════════
           SLIDERS
           ═══════════════════════════════════════ */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--primary) !important;
        }

        /* ═══════════════════════════════════════
           SELECT BOXES
           ═══════════════════════════════════════ */
        .stSelectbox, .stMultiSelect {
            font-size: 0.9rem;
        }

        /* ═══════════════════════════════════════
           FILE UPLOADER
           ═══════════════════════════════════════ */
        [data-testid="stFileUploader"] {
            border: 1px dashed rgba(151, 169, 255, 0.3);
            border-radius: 12px;
            padding: 0.5rem;
            transition: border-color 0.3s;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(151, 169, 255, 0.6);
        }

        /* ═══════════════════════════════════════
           SPINNER
           ═══════════════════════════════════════ */
        .stSpinner > div > div {
            border-top-color: var(--primary) !important;
        }

        /* ═══════════════════════════════════════
           SCROLLBAR
           ═══════════════════════════════════════ */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--surface-container-highest); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--outline-variant); }

        /* ═══════════════════════════════════════
           TEXT INPUT / TEXT AREA
           ═══════════════════════════════════════ */
        .stTextArea textarea, .stTextInput input {
            border-radius: 10px !important;
            border: 1px solid rgba(70, 72, 79, 0.25) !important;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: rgba(151, 169, 255, 0.5) !important;
            box-shadow: 0 0 0 2px rgba(151, 169, 255, 0.08);
        }

        /* ═══════════════════════════════════════
           CUSTOM CLASSES (used in markdown)
           ═══════════════════════════════════════ */
        .synthetix-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .badge-success { background: rgba(91,239,144,0.1); color: #5bef90; border: 1px solid rgba(91,239,144,0.2); }
        .badge-warning { background: rgba(255,110,132,0.1); color: #ff6e84; border: 1px solid rgba(255,110,132,0.2); }
        .badge-info { background: rgba(151,169,255,0.1); color: #97a9ff; border: 1px solid rgba(151,169,255,0.2); }
        .badge-cloud { background: rgba(201,154,248,0.1); color: #c99af8; border: 1px solid rgba(201,154,248,0.2); }

    </style>
    """, unsafe_allow_html=True)
