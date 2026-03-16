"""
RAG Q&A UI — question answering against the indexed document.
"""
import streamlit as st

from core.rag_engine import query_rag
from core.llm_engine import llm_answer_question

FAISS_DIR = "./data/rag_faiss_store"


def render(target_language: str):
    st.header("🔍 RAG-based Question Answering")

    if not st.session_state.pdf_processed:
        st.warning("⚠️ Please upload a document in **Document Analysis** mode first!")
        return

    st.info(f"📄 Active Document: **{st.session_state.current_pdf_name}**")

    with st.form("qa_form"):
        question = st.text_input(
            "Ask a question about the document:",
            placeholder="e.g., What are the payment terms in this contract?",
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            submit = st.form_submit_button("🔍 Ask Question", use_container_width=True)
        with col2:
            k_results = st.selectbox("Chunks", [3, 4, 5, 6], index=1)

    if submit and question:
        with st.spinner("🤖 Searching and generating answer…"):
            try:
                context, docs = query_rag(question, persist_dir=FAISS_DIR, k=k_results)

                if "⚠️" not in context and docs:
                    answer = llm_answer_question(
                        context, question,
                        target_language=target_language
                    )

                    st.markdown("### 🧠 Answer")
                    st.success(answer)

                    with st.expander("📚 View Source Context"):
                        st.markdown(context)

                    _, col_btn = st.columns([3, 1])
                    with col_btn:
                        if st.button("➕ Add to Training Data"):
                            st.session_state.training_data.append({
                                "question": question,
                                "context": context,
                                "answer": answer,
                            })
                            st.success("Added to training data!")
                            st.rerun()

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": len(docs),
                    })
                else:
                    st.warning(context)
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 Question History")
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            idx = len(st.session_state.chat_history) - i + 1
            with st.expander(f"Q{idx}: {chat['question'][:60]}…"):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.caption(f"📚 Based on {chat.get('sources', 0)} excerpts")
