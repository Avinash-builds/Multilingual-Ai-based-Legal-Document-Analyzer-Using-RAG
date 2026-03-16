"""
QA Engine — wraps the Flan-T5 text2text-generation pipeline for RAG Q&A.
"""

import streamlit as st

QA_MODEL = "google/flan-t5-base"


@st.cache_resource
def load_qa_model():
    from transformers import pipeline
    return pipeline("text2text-generation", model=QA_MODEL)


def answer_question(context: str, question: str, max_new_tokens: int = 256) -> str:
    """Generate an answer to a question given retrieved context."""
    qa = load_qa_model()
    prompt = f"""You are a legal document assistant. Answer the question based on the context provided and explain it in detail.

Context:
{context}

Question: {question}

Provide a clear, accurate answer:"""

    result = qa(prompt, max_new_tokens=max_new_tokens, clean_up_tokenization_spaces=True)
    return result[0]["generated_text"].strip()
