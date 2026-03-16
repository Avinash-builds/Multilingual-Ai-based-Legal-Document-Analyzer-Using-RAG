"""
Translation Handler — Helsinki-NLP dedicated translation models.
Supports: Hindi, Tamil, Telugu, Marathi.
"""

import re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

TRANSLATION_MODELS = {
    "Hindi":   {"model": "Helsinki-NLP/opus-mt-en-hi", "name": "हिंदी"},
    "Tamil":   {"model": "Helsinki-NLP/opus-mt-en-ta", "name": "தமிழ்"},
    "Telugu":  {"model": "Helsinki-NLP/opus-mt-en-te", "name": "తెలుగు"},
    "Marathi": {"model": "Helsinki-NLP/opus-mt-en-mr", "name": "मराठी"},
}


@st.cache_resource
def load_translation_model(language: str):
    """Load and cache a Helsinki-NLP translation model."""
    if language not in TRANSLATION_MODELS:
        raise ValueError(f"Unsupported language: {language}")
    model_name = TRANSLATION_MODELS[language]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def _chunk_text(text: str, max_chars: int = 450) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            for part in re.split(r'[,;]', sentence):
                if len(current) + len(part) < max_chars:
                    current += part + ", "
                else:
                    if current:
                        chunks.append(current.strip())
                    current = part + ", "
        else:
            if len(current) + len(sentence) < max_chars:
                current += sentence + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks


def _translate_chunk(text: str, model, tokenizer) -> str:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_text(text: str, target_language: str, max_chars: int = 2000) -> str:
    """
    Translate English text to the target language.
    Returns original text if target_language is 'English' or unsupported.
    """
    if not text or not text.strip():
        return ""
    if target_language == "English" or target_language not in TRANSLATION_MODELS:
        return text

    model, tokenizer = load_translation_model(target_language)
    chunks = _chunk_text(text[:max_chars])
    return " ".join(_translate_chunk(chunk, model, tokenizer) for chunk in chunks)


def get_available_languages() -> List[str]:
    """Return list of all supported output languages."""
    return ["English"] + list(TRANSLATION_MODELS.keys())
