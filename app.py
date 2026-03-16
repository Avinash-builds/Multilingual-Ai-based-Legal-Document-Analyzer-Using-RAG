"""
Legal Document Analysis Platform — Main Entry Point
Run with: .venv/bin/streamlit run app.py
"""

import json
import os
import streamlit as st
from datetime import datetime

from utils.session import init_session_state, clear_session
from translation.handler import get_available_languages
from core.llm_engine import configure_gemini
import ui.document_analysis as doc_analysis_ui
import ui.rag_qa as rag_qa_ui
import ui.fine_tuning as fine_tuning_ui
import ui.analytics as analytics_ui

# ── Gemini Configuration (reads from .env file) ───────────────────────────────
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    configure_gemini(GEMINI_API_KEY)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚖️ Legal Document Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px; border-radius: 15px; text-align: center;
        color: white; margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .clause-box {
        background: #e3f2fd; padding: 15px; border-radius: 8px;
        margin: 10px 0; border-left: 4px solid #2196f3;
    }
    .summary-box {
        background: #f3e5f5; padding: 15px; border-radius: 8px;
        margin: 10px 0; border-left: 4px solid #9c27b0;
    }
    .translation-box {
        background: #fff3e0; padding: 15px; border-radius: 8px;
        margin: 10px 0; border-left: 4px solid #ff9800;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="main-header">
        <h1>⚖️ Legal Document Analysis Platform</h1>
        <p style="font-size:1.1em; margin-top:10px;">
            AI-Powered Legal Document Analysis, Translation &amp; Summarization
        </p>
        <p style="font-size:0.9em; opacity:0.9;">
            English · Hindi · Tamil · Telugu · Marathi &nbsp;|&nbsp;
            Clause Extraction · RAG Q&amp;A · Fine-tuning
        </p>
    </div>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
init_session_state()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Control Panel")

    mode = st.selectbox(
        "Select Mode",
        ["📄 Document Analysis", "🔍 RAG Q&A", "🎓 Model Fine-tuning", "📊 Analytics"],
        help="Choose the operation mode",
    )

    st.markdown("---")

    target_language = st.selectbox(
        "🌐 Target Language",
        get_available_languages(),
        help="Select output language for summaries and answers",
    )

    st.markdown("---")

    if st.button("🗑️ Clear All Data", use_container_width=True):
        clear_session()
        st.rerun()

    if st.button("💾 Export Session Data", use_container_width=True):
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "document": st.session_state.current_pdf_name,
            "chat_history": st.session_state.chat_history,
            "clauses": st.session_state.extracted_clauses,
            "training_data": st.session_state.training_data,
        }
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(session_data, indent=2, default=str),
            file_name=f"legal_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    st.markdown("---")

    if st.session_state.pdf_processed:
        st.markdown("### 📊 Session Stats")
        st.metric("Questions Asked", len(st.session_state.chat_history))
        st.metric("Clauses Extracted", len(st.session_state.extracted_clauses))
        st.metric("Training Samples", len(st.session_state.training_data))

    st.markdown("---")
    st.markdown("""
    ### 💡 Features
    - 📄 Document Summarization
    - 🌐 Multi-language Translation
    - 🔍 Clause Extraction
    - ⚠️ Document Risk Analysis
    - 📋 Standard Practices Comparison
    - 🤖 RAG-based Q&A
    - 🎓 Model Fine-tuning
    - 📊 Performance Analytics
    """)

# ── Route to UI modules ────────────────────────────────────────────────────────
if mode == "📄 Document Analysis":
    doc_analysis_ui.render(target_language)
elif mode == "🔍 RAG Q&A":
    rag_qa_ui.render(target_language)
elif mode == "🎓 Model Fine-tuning":
    fine_tuning_ui.render()
else:
    analytics_ui.render()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
    <div style="text-align:center; padding:20px; color:#666;">
        <p><strong>Legal Document Analysis Platform</strong></p>
        <p style="font-size:0.9em;">
            Powered by FAISS · LangChain · HuggingFace Transformers · Helsinki-NLP<br/>
            <em>For reference purposes only. Consult qualified legal professionals for advice.</em>
        </p>
    </div>
""", unsafe_allow_html=True)
