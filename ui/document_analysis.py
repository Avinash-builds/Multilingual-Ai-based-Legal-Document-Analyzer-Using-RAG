"""
Document Analysis UI — PDF upload, summarization, translation, clause extraction.
"""
import tempfile
import streamlit as st

from core.rag_engine import extract_text_from_pdf, build_index_from_pdf
from core.llm_engine import llm_summarize, llm_extract_clauses, llm_translate, llm_risk_analysis, llm_compare_standard

FAISS_DIR = "./data/rag_faiss_store"


def _render_clauses(clauses):
    """Render extracted clauses with risk levels and summaries."""
    for i, clause in enumerate(clauses, 1):
        risk = clause.get("risk_level", "unknown")
        risk_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk, "⚪")
        risk_color = {"high": "#ffebee", "medium": "#fff8e1", "low": "#e8f5e9"}.get(risk, "#e3f2fd")
        border_color = {"high": "#e53935", "medium": "#ffa000", "low": "#43a047"}.get(risk, "#2196f3")

        st.markdown(f"""<div style="background:{risk_color}; padding:15px; border-radius:8px;
            margin:10px 0; border-left:4px solid {border_color};">
            <strong>{risk_icon} Clause {i}: {clause.get('type', 'Unknown')}</strong>
            <span style="float:right; font-size:0.8em; color:#666;">Risk: {risk.upper()}</span>
        </div>""", unsafe_allow_html=True)

        if clause.get("summary"):
            st.caption(f"💡 {clause['summary']}")

        text = clause.get("text", "")
        snippet = text[:400] + ("…" if len(text) > 400 else "")
        st.markdown(f"<small>{snippet}</small>", unsafe_allow_html=True)
        st.markdown("---")


def render(target_language: str):
    st.header("📄 Document Analysis & Processing")

    uploaded_file = st.file_uploader(
        "Upload Legal Document (PDF)", type=["pdf"],
        help="Upload a legal document for analysis"
    )

    if not uploaded_file:
        return

    if st.session_state.current_pdf_name != uploaded_file.name:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner("📖 Extracting text from PDF…"):
                st.session_state.extracted_text = extract_text_from_pdf(tmp_path)
                st.session_state.current_pdf_name = uploaded_file.name
                st.session_state.tmp_pdf_path = tmp_path
                build_index_from_pdf(tmp_path, persist_dir=FAISS_DIR)
                st.session_state.pdf_processed = True

            st.success(f"✅ Document processed: **{uploaded_file.name}**")
        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")
            return

    if not st.session_state.pdf_processed:
        return

    # ── Action Buttons ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📝 Generate Summary", use_container_width=True):
            with st.spinner("Generating summary…"):
                try:
                    summary = llm_summarize(
                        st.session_state.extracted_text,
                        target_language=target_language
                    )
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.markdown("### 📝 Document Summary")
                    st.write(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

    with col2:
        if st.button("🌐 Translate Document Snippet", use_container_width=True):
            with st.spinner(f"Translating to {target_language}…"):
                try:
                    snippet = st.session_state.extracted_text[:1000]
                    translated = llm_translate(snippet, target_language)
                    st.markdown('<div class="translation-box">', unsafe_allow_html=True)
                    st.markdown(f"### 🌐 Translation ({target_language})")
                    st.write(translated)
                    st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error translating: {e}")

    with col3:
        if st.button("🔍 Extract Clauses", use_container_width=True):
            with st.spinner("Extracting legal clauses…"):
                try:
                    clauses = llm_extract_clauses(st.session_state.extracted_text)
                    st.session_state.extracted_clauses = clauses
                    st.markdown("### 🔍 Extracted Legal Clauses")
                    _render_clauses(clauses)
                except Exception as e:
                    st.error(f"Error extracting clauses: {e}")

    with col4:
        if st.button("⚠️ Risk Analysis", use_container_width=True):
            with st.spinner("Performing risk analysis…"):
                try:
                    analysis = llm_risk_analysis(st.session_state.extracted_text)
                    st.markdown("### ⚠️ Document Risk Analysis")
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Error in risk analysis: {e}")

    # ── Compare Against Standards ─────────────────────────────────────────
    if st.button("📋 Compare Against Standard Practices", use_container_width=True):
        with st.spinner("Comparing with standard legal practices…"):
            try:
                comparison = llm_compare_standard(st.session_state.extracted_text)
                st.markdown("### 📋 Standard Practices Comparison")
                st.markdown(comparison)
            except Exception as e:
                st.error(f"Error in comparison: {e}")

    with st.expander("📄 View Full Extracted Text"):
        st.text_area("Document Text", st.session_state.extracted_text, height=300)
