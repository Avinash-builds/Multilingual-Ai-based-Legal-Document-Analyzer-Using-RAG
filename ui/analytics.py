"""
Analytics UI — session statistics and document insights.
"""
import streamlit as st


def render():
    st.header("📊 Performance Analytics")

    if not st.session_state.pdf_processed:
        st.info("📊 Upload a document in **Document Analysis** mode to see analytics.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents Processed", 1)
        st.metric("Total Questions", len(st.session_state.chat_history))
    with col2:
        st.metric("Clauses Extracted", len(st.session_state.extracted_clauses))
        st.metric("Training Samples", len(st.session_state.training_data))
    with col3:
        if st.session_state.chat_history:
            avg = sum(c.get("sources", 0) for c in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.metric("Avg. Sources per Answer", f"{avg:.1f}")

    if st.session_state.extracted_text:
        st.markdown("### 📄 Document Insights")
        text = st.session_state.extracted_text
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Character Count", f"{len(text):,}")
        with col2:
            st.metric("Word Count", f"{len(text.split()):,}")

    if st.session_state.extracted_clauses:
        st.markdown("### 🔍 Top Extracted Clauses Preview")
        for i, clause in enumerate(st.session_state.extracted_clauses[:5], 1):
            st.markdown(f"**Clause {i}: {clause['type']}**")
            preview = clause["text"][:200] + ("…" if len(clause["text"]) > 200 else "")
            st.write(preview)
