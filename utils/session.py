"""
Session State Utilities — centralised session state initialisation.
"""
import streamlit as st


def init_session_state():
    """Initialise all session state keys with safe defaults."""
    defaults = {
        "chat_history":      [],
        "pdf_processed":     False,
        "extracted_text":    "",
        "current_pdf_name":  None,
        "training_data":     [],
        "extracted_clauses": [],
        "tmp_pdf_path":      None,
        "training_status":   "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_session():
    """Reset all session state to defaults."""
    st.session_state.chat_history = []
    st.session_state.pdf_processed = False
    st.session_state.extracted_text = ""
    st.session_state.current_pdf_name = None
    st.session_state.training_data = []
    st.session_state.extracted_clauses = []
    st.session_state.tmp_pdf_path = None
    st.session_state.training_status = ""
