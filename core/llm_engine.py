"""
LLM Engine — Google Gemini integration for advanced legal document analysis.
Provides: summarization, clause extraction, Q&A, and translation via Gemini API.
"""

import os
import json
import re
import streamlit as st
import google.generativeai as genai
from typing import List, Dict, Optional


# ── Configuration ─────────────────────────────────────────────────────────────

def configure_gemini(api_key: str) -> bool:
    """Configure Gemini with the provided API key. Returns True on success."""
    try:
        genai.configure(api_key=api_key)
        # Quick validation — list models to confirm key works
        list(genai.list_models())
        return True
    except Exception:
        return False


def is_gemini_configured() -> bool:
    """Check if Gemini is configured and ready."""
    return st.session_state.get("gemini_configured", False)


@st.cache_resource
def get_gemini_model(model_name: str = "gemini-2.5-flash"):
    """Load and cache the Gemini generative model."""
    return genai.GenerativeModel(model_name)


# ── Summarization ─────────────────────────────────────────────────────────────

def llm_summarize(text: str, target_language: str = "English") -> str:
    """
    Generate a comprehensive legal document summary using Gemini.
    Can handle much longer text than BART (up to ~30K tokens).
    """
    model = get_gemini_model()

    # Take up to ~25K characters (well within Gemini's context window)
    text_chunk = text[:25000]

    lang_instruction = ""
    if target_language != "English":
        lang_instruction = f"\n\nIMPORTANT: Write the summary in {target_language} language."

    prompt = f"""You are an expert legal document analyst. Analyze the following legal document and provide a comprehensive, well-structured summary.

Your summary should include:
1. **Document Type & Parties**: Identify the type of document and key parties involved
2. **Key Terms & Conditions**: Summarize the most important terms
3. **Obligations & Rights**: Highlight obligations of each party
4. **Important Dates & Deadlines**: Note any critical dates or timelines
5. **Financial Terms**: Summarize any monetary amounts, fees, or payment terms
6. **Risk Factors**: Identify any potential risks or unusual clauses
{lang_instruction}

LEGAL DOCUMENT:
{text_chunk}

COMPREHENSIVE SUMMARY:"""

    response = model.generate_content(prompt)
    return response.text


# ── Clause Extraction ─────────────────────────────────────────────────────────

def llm_extract_clauses(text: str) -> List[Dict[str, str]]:
    """
    Extract legal clauses using Gemini's semantic understanding.
    Far more accurate than regex-based extraction.
    """
    model = get_gemini_model()

    text_chunk = text[:20000]

    prompt = f"""You are an expert legal clause extractor. Analyze the following legal document and extract ALL identifiable legal clauses.

For each clause, provide:
- "type": The category of the clause (e.g., "Payment Terms", "Termination", "Confidentiality", "Liability", "Indemnification", "Governing Law", "Dispute Resolution", "Intellectual Property", "Force Majeure", "Warranty", "Non-Compete", "Data Protection", "Insurance", "Assignment", "Amendments", "Severability", "Entire Agreement", "Notice Requirements", etc.)
- "text": The exact relevant text from the document (include enough context to understand the clause, keep it under 500 characters)
- "risk_level": Rate as "low", "medium", or "high" based on potential risk to the parties
- "summary": A brief 1-2 sentence plain-English explanation of what this clause means

Return ONLY a valid JSON array. No markdown, no ```json``` wrapper, no explanation outside the array.

Example format:
[
  {{"type": "Payment Terms", "text": "Tenant shall pay rent of $850...", "risk_level": "low", "summary": "Monthly rent of $850 due on the 1st of each month."}},
  {{"type": "Termination", "text": "Either party may terminate...", "risk_level": "medium", "summary": "30-day notice required for early termination."}}
]

LEGAL DOCUMENT:
{text_chunk}

EXTRACTED CLAUSES JSON ARRAY:"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Clean up any markdown code fencing
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        clauses = json.loads(raw)
        if isinstance(clauses, list):
            return clauses
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON array in the response
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            clauses = json.loads(match.group())
            if isinstance(clauses, list):
                return clauses
        except json.JSONDecodeError:
            pass

    # If all parsing fails, return a single entry with the raw response
    return [{"type": "Analysis", "text": raw[:500], "risk_level": "unknown", "summary": "Could not parse structured clauses."}]


# ── Question Answering ────────────────────────────────────────────────────────

def llm_answer_question(context: str, question: str, target_language: str = "English") -> str:
    """
    Answer a legal question using Gemini, given RAG-retrieved context.
    Produces detailed, well-reasoned answers compared to flan-t5.
    """
    model = get_gemini_model()

    lang_instruction = ""
    if target_language != "English":
        lang_instruction = f"\n\nIMPORTANT: Provide the answer in {target_language} language."

    prompt = f"""You are an expert legal document assistant. Answer the question accurately based on the provided context from a legal document.

INSTRUCTIONS:
- Base your answer STRICTLY on the provided context
- Be thorough and detailed in your explanation
- Use clear, professional language
- If the context doesn't contain enough information, explicitly state that
- Highlight any important legal implications
- Quote relevant parts of the document when helpful
{lang_instruction}

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

DETAILED ANSWER:"""

    response = model.generate_content(prompt)
    return response.text


# ── Translation ───────────────────────────────────────────────────────────────

def llm_translate(text: str, target_language: str) -> str:
    """
    Translate text using Gemini. Handles legal terminology accurately.
    """
    if not text or not text.strip() or target_language == "English":
        return text

    model = get_gemini_model()

    # Chunk for very long text
    text_chunk = text[:10000]

    prompt = f"""Translate the following legal text accurately into {target_language}.

RULES:
- Maintain legal terminology precision
- Keep proper nouns, names, and numbers unchanged
- Preserve the meaning and tone of the original text
- Do not add explanations — only provide the translation

TEXT TO TRANSLATE:
{text_chunk}

{target_language} TRANSLATION:"""

    response = model.generate_content(prompt)
    return response.text


# ── Document Risk Analysis (NEW Advanced Feature) ─────────────────────────────

def llm_risk_analysis(text: str) -> str:
    """
    Perform a risk analysis of the legal document.
    This is a NEW feature only available in the advanced LLM version.
    """
    model = get_gemini_model()

    text_chunk = text[:20000]

    prompt = f"""You are a senior legal consultant. Perform a comprehensive risk analysis of the following legal document.

Provide your analysis in the following structure:

## 🔴 High-Risk Issues
List any clauses or terms that could be particularly problematic or unfavorable.

## 🟡 Medium-Risk Concerns
List areas that may need attention or negotiation.

## 🟢 Favorable Terms
List clauses that are standard or favorable.

## 📋 Recommendations
Provide specific, actionable recommendations for the reader.

## ⚖️ Overall Risk Rating
Rate the overall document risk as: LOW / MEDIUM / HIGH and explain why.

LEGAL DOCUMENT:
{text_chunk}

RISK ANALYSIS:"""

    response = model.generate_content(prompt)
    return response.text


# ── Comparison Analysis (NEW Advanced Feature) ────────────────────────────────

def llm_compare_standard(text: str) -> str:
    """
    Compare the document against standard legal practices.
    """
    model = get_gemini_model()

    text_chunk = text[:20000]

    prompt = f"""You are a legal expert. Compare the following legal document against standard legal practices and industry norms.

Identify:
1. **Missing Standard Clauses** — Important clauses typically found in this type of document that are absent
2. **Non-Standard Terms** — Any terms that deviate significantly from standard practice
3. **Ambiguous Language** — Clauses that could be interpreted in multiple ways
4. **Compliance Concerns** — Any potential regulatory or compliance issues

LEGAL DOCUMENT:
{text_chunk}

COMPARISON ANALYSIS:"""

    response = model.generate_content(prompt)
    return response.text
