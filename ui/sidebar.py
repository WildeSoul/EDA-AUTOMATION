"""
Sidebar navigation component.
"""
import streamlit as st
from core.llm import get_ollama_models
from config.settings import CLOUD_MODELS

def render_sidebar(active_df_name: str | None = None, active_rows: int = 0, active_cols: int = 0) -> str:
    """Renders the sidebar navigation and returns the selected page."""
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
                <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#97a9ff,#c99af8);display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(151,169,255,0.2);">
                    <span style="font-size:20px;color:#002283;">🧠</span>
                </div>
                <div>
                    <h3 style="margin:0;padding:0;font-size:1.15rem;font-weight:800;background:linear-gradient(135deg,#97a9ff,#c99af8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;">AI Analyst Pro</h3>
                    <p style="margin:0;color:#aaaab3;font-size:0.65rem;letter-spacing:0.06em;text-transform:uppercase;font-weight:500;">v4.0 Agentic Platform</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Options
        page_options = [
            "🏠 Home",
            "📂 Datasets",
            "🔬 Analysis",
            "🧠 Agent System",
            "⚙️ Settings"
        ]
        
        # Get index
        if "current_page" not in st.session_state:
            st.session_state.current_page = "🏠 Home"
            
        default_index = 0
        if st.session_state.current_page in page_options:
            default_index = page_options.index(st.session_state.current_page)
            
        selected_page = st.radio(
            label="Workspace Section:",
            options=page_options,
            index=default_index,
            key="sidebar_navigation_radio",
            label_visibility="collapsed"
        )
        st.session_state.current_page = selected_page
        
        st.markdown("---")
        
        # Active Dataset Status Card
        if active_df_name:
            st.markdown(f"""
            <div style="background: rgba(151, 169, 255, 0.05); border: 1px solid rgba(151, 169, 255, 0.15); border-radius: 10px; padding: 10px; margin-bottom: 1rem;">
                <p style="margin:0; font-size:0.7rem; color:#888890; text-transform:uppercase; font-weight:600; letter-spacing:0.05em;">Active Dataset</p>
                <p style="margin:4px 0 0 0; font-size:0.85rem; font-weight:700; color:#e5e4ed; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">📄 {active_df_name}</p>
                <p style="margin:2px 0 0 0; font-size:0.75rem; color:#aaaab3;">{active_rows:,} rows × {active_cols} cols</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("❌ Close Dataset", use_container_width=True):
                # Clear active dataset from session state
                keys_to_clear = [
                    "df", "df_original", "session_id", "filename", "intelligence", 
                    "sheet_name", "uploaded_file", "chat_history", "current_plan", "execution_state"
                ]
                for k in keys_to_clear:
                    if k in st.session_state:
                        del st.session_state[k]
                st.session_state.current_page = "🏠 Home"
                st.rerun()
        else:
            st.markdown("""
            <div style="background: rgba(255, 110, 132, 0.03); border: 1px dashed rgba(255, 110, 132, 0.15); border-radius: 10px; padding: 10px; text-align: center; margin-bottom: 1rem;">
                <p style="margin:0; font-size:0.8rem; color:#aaaab3; font-weight:500;">No active dataset loaded</p>
                <p style="margin:2px 0 0 0; font-size:0.7rem; color:#888890;">Upload or pick a demo on Home</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Quick Model Indicator
        selected_model = st.session_state.get("selected_model", "mistral-nemo:latest")
        model_badge = "☁️" if selected_model in CLOUD_MODELS else "💻"
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 8px; font-size:0.75rem;">
            <span style="color:#aaaab3;">AI Core:</span> <strong>{model_badge} {selected_model.split(':')[0]}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        return selected_page
