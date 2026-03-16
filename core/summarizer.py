"""
Summarizer — wraps the BART summarization pipeline.
"""

import streamlit as st

SUMMARIZATION_MODEL = "facebook/bart-large-cnn"


@st.cache_resource
def load_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model=SUMMARIZATION_MODEL)


def generate_summary(text: str, max_length: int = 300, min_length: int = 100) -> str:
    """Generate a concise summary of the provided text."""
    summarizer = load_summarizer()
    result = summarizer(text[:4000], max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"]
