import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import io

from analysis.unstructured import (
    compute_text_health, compute_image_health, 
    extract_ngrams_and_lemmas, extract_dominant_colors, run_ocr
)

def render_nlp_dashboard(text: str, filename: str):
    """Render the NLP Dashboard for text and PDF files."""
    st.title(f"📄 NLP Analysis: {filename}")
    
    # 1. Health Score
    health = compute_text_health(text)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Health Grade", health["grade"], f"Score: {health['total_score']}/100")
    with col2:
        st.metric("Readability", health["readability_raw"], health["complexity_level"])
    with col3:
        st.metric("Stop-Word Density", f"{health['stop_density']}%")
    with col4:
        st.metric("Total Characters", f"{len(text):,}")
        
    st.markdown("---")
    
    # 2. NLP Analysis (N-Grams & Sentiment)
    if text.strip():
        with st.spinner("Extracting N-Grams and running NLP..."):
            nlp_data = extract_ngrams_and_lemmas(text)
            
        if "error" in nlp_data:
            st.error(f"NLP Error: {nlp_data['error']}")
        else:
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Sentiment Analysis")
                sentiment = nlp_data["sentiment"]
                polarity = sentiment.polarity # -1 to 1
                subjectivity = sentiment.subjectivity # 0 to 1
                
                pol_color = "#2ecc71" if polarity > 0 else "#e74c3c" if polarity < 0 else "#95a5a6"
                
                fig_sent = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = polarity,
                    title = {'text': "Polarity (-1 Negative to 1 Positive)"},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': pol_color},
                        'steps': [
                            {'range': [-1, -0.2], 'color': "#ffebee"},
                            {'range': [-0.2, 0.2], 'color': "#f5f5f5"},
                            {'range': [0.2, 1], 'color': "#e8f5e9"}
                        ]
                    }
                ))
                fig_sent.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_sent, use_container_width=True)
                st.metric("Subjectivity (0=Fact, 1=Opinion)", f"{subjectivity:.2f}")

            with c2:
                st.subheader("Top Lemmatized Words")
                if nlp_data["top_words"]:
                    words, counts = zip(*nlp_data["top_words"])
                    fig_words = px.bar(x=counts, y=words, orientation='h', color=counts, color_continuous_scale="Viridis")
                    fig_words.update_layout(yaxis={'categoryorder':'total ascending'}, height=300, showlegend=False, xaxis_title="Frequency", yaxis_title="")
                    st.plotly_chart(fig_words, use_container_width=True)
            
            st.subheader("Contextual Analysis (N-Grams)")
            tab1, tab2 = st.tabs(["Bigrams (2 words)", "Trigrams (3 words)"])
            with tab1:
                if nlp_data["bigrams"]:
                    bg, bg_c = zip(*nlp_data["bigrams"])
                    fig_bg = px.bar(x=bg, y=bg_c, color=bg_c, color_continuous_scale="Blues")
                    fig_bg.update_layout(height=350, xaxis_title="", yaxis_title="Frequency")
                    st.plotly_chart(fig_bg, use_container_width=True)
            with tab2:
                if nlp_data["trigrams"]:
                    tg, tg_c = zip(*nlp_data["trigrams"])
                    fig_tg = px.bar(x=tg, y=tg_c, color=tg_c, color_continuous_scale="Purples")
                    fig_tg.update_layout(height=350, xaxis_title="", yaxis_title="Frequency")
                    st.plotly_chart(fig_tg, use_container_width=True)
    
    st.markdown("---")
    with st.expander("🔍 View Raw Text Content", expanded=False):
        st.text_area("Content", text, height=400, disabled=True)


def render_image_dashboard(image_bytes: bytes, filename: str):
    """Render the Image Analytics Dashboard for PNG/JPG."""
    st.title(f"🖼️ Computer Vision Analysis: {filename}")
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        st.error(f"Failed to open image: {str(e)}")
        return
        
    # 1. Health Score
    health = compute_image_health(image_bytes)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Image Quality Grade", health["grade"], f"Score: {health['total_score']}/100")
    with col2:
        st.metric("Contrast", health["contrast"])
    with col3:
        st.metric("Sharpness (Variance of Laplacian)", health["blurriness"])
    with col4:
        st.metric("Resolution", f"{img.width}x{img.height} ({img.mode})")
        
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Image Preview")
        # Resize for display if huge
        display_img = img.copy()
        display_img.thumbnail((800, 800))
        st.image(display_img, use_container_width=True)
        
    with c2:
        st.subheader("🎨 Dominant Color Palette")
        with st.spinner("Extracting colors via KMeans clustering..."):
            hex_colors = extract_dominant_colors(image_bytes)
            
        if hex_colors:
            cols = st.columns(len(hex_colors))
            for i, hex_code in enumerate(hex_colors):
                with cols[i]:
                    st.markdown(
                        f'''<div style="background-color: {hex_code}; height: 80px; width: 100%; border-radius: 8px; border: 1px solid #ddd;"></div>''', 
                        unsafe_allow_html=True
                    )
                    st.caption(hex_code)
        else:
            st.info("Could not extract dominant colors.")
            
        st.markdown("---")
        st.subheader("📝 Optical Character Recognition (OCR)")
        st.markdown("Extracts embedded text from the image for downstream NLP analysis.")
        
        if "ocr_text" not in st.session_state:
            st.session_state.ocr_text = None

        if st.button("Run EasyOCR Pipeline"):
            with st.spinner("Running deep learning OCR pipeline..."):
                st.session_state.ocr_text = run_ocr(image_bytes)
                
        if st.session_state.ocr_text is not None:
            ocr_text = st.session_state.ocr_text
            if ocr_text.strip() and not ocr_text.startswith("OCR Failed"):
                st.success("Text extracted successfully!")
                st.text_area("Extracted Text", ocr_text, height=150)
                if st.button("Send to NLP Dashboard"):
                    st.session_state["unstructured_text"] = ocr_text
                    st.session_state["unstructured_filename"] = f"OCR_{filename}.txt"
                    st.rerun()
            elif ocr_text.startswith("OCR Failed"):
                st.error(ocr_text)
            else:
                st.info("No text detected in this image.")
