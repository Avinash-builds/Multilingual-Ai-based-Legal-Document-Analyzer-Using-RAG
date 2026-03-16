import streamlit as st
import tempfile
import os
import json
from datetime import datetime
from core.rag_index_builder import build_index_from_pdf, query_rag, get_index_stats, extract_text_from_pdf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from deep_translator import GoogleTranslator
import re
from langdetect import detect, DetectorFactory

# Make langdetect deterministic
DetectorFactory.seed = 0

# ------------------ CONFIG ------------------
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
QA_MODEL = "google/flan-t5-base"

# ------------------ TRANSLATION UTIL ------------------
def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text to target_lang using deep_translator.GoogleTranslator.
    target_lang should be one of: "English", "Hindi", "Tamil", "Telugu"
    Handles chunking for long texts.
    Returns translated text or an error message string on failure.
    """
    try:
        if not text or not text.strip():
            return ""

        lang_codes = {"English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te"}
        target_code = lang_codes.get(target_lang, "en")

        # chunk size 4000 characters to avoid limits
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        parts = []
        for chunk in chunks:
            # GoogleTranslator source='auto' detects automatically
            translated = GoogleTranslator(source="auto", target=target_code).translate(chunk)
            parts.append(translated)
        return " ".join(parts)
    except Exception as e:
        return f"🚫 Translation Error: {str(e)}"

def detect_language_of_text(text: str) -> str:
    """Return detected ISO short code like 'en','hi','ta','te' or 'unknown'"""
    try:
        if not text or not text.strip():
            return "unknown"
        lang = detect(text)
        return lang
    except Exception:
        return "unknown"

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_models():
    """Load summarizer (BART) and QA (Flan-T5) models via AutoModel (transformers v5+ compatible)."""
    try:
        print("Loading BART summarizer...")
        sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)
        print("BART loaded.")

        print("Loading Flan-T5 QA model...")
        qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
        qa_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL)
        print("Flan-T5 loaded.")

        return {
            'sum_model': sum_model,
            'sum_tokenizer': sum_tokenizer,
            'qa_model': qa_model,
            'qa_tokenizer': qa_tokenizer,
        }
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

models = load_models()

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="⚖️ Legal Document Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .clause-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    .summary-box {
        background: #f3e5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #9c27b0;
    }
    .translation-box {
        background: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ff9800;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
    <div class="main-header">
        <h1>Legal Document Analysis Platform</h1>
        <p style="font-size: 1.1em; margin-top: 10px;">
            AI-Powered Legal Document Analysis, Translation & Summarization
        </p>
        <p style="font-size: 0.9em; opacity: 0.9;">
            Supporting English, Hindi, Tamil, Telugu | Extract Clauses | Train Custom Models
        </p>
    </div>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE ------------------
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "pdf_processed" not in st.session_state: st.session_state.pdf_processed = False
if "extracted_text" not in st.session_state: st.session_state.extracted_text = ""
if "current_pdf_name" not in st.session_state: st.session_state.current_pdf_name = None
if "training_data" not in st.session_state: st.session_state.training_data = []
if "extracted_clauses" not in st.session_state: st.session_state.extracted_clauses = []
if "tmp_pdf_path" not in st.session_state: st.session_state.tmp_pdf_path = None
if "translated_text" not in st.session_state: st.session_state.translated_text = ""

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("🎛️ Control Panel")

    mode = st.selectbox(
        "Select Mode",
        ["📄 Document Analysis", "🔍 RAG Q&A", "🎓 Model Fine-tuning", "📊 Analytics"],
        help="Choose the operation mode"
    )

    st.markdown("---")

    target_language = st.selectbox(
        "🌐 Target Language",
        ["English", "Hindi", "Tamil", "Telugu"],
        help="Select language for translation (output language)"
    )

    st.markdown("---")

    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pdf_processed = False
        st.session_state.extracted_text = ""
        st.session_state.current_pdf_name = None
        st.session_state.training_data = []
        st.session_state.extracted_clauses = []
        st.session_state.tmp_pdf_path = None
        st.session_state.translated_text = ""
        st.rerun()

    if st.button("💾 Export Session Data", use_container_width=True):
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "document": st.session_state.current_pdf_name,
            "chat_history": st.session_state.chat_history,
            "clauses": st.session_state.extracted_clauses,
            "training_data": st.session_state.training_data
        }
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(session_data, indent=2),
            file_name=f"legal_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
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
        - 🤖 RAG-based Q&A
        - 🎓 Model Fine-tuning
        - 📊 Performance Analytics
    """)

# ------------------ HELPER FUNCTIONS ------------------
def extract_legal_clauses(text):
    """Extract legal clauses (same as original patterns)"""
    clauses = []
    patterns = {
        "Payment Terms": r"payment.*?(?:\.|;|\n\n)",
        "Termination": r"terminat.*?(?:\.|;|\n\n)",
        "Confidentiality": r"confidential.*?(?:\.|;|\n\n)",
        "Liability": r"liability.*?(?:\.|;|\n\n)",
        "Indemnification": r"indemnif.*?(?:\.|;|\n\n)",
        "Governing Law": r"governing law.*?(?:\.|;|\n\n)",
        "Dispute Resolution": r"dispute.*?(?:\.|;|\n\n)",
        "Intellectual Property": r"intellectual property.*?(?:\.|;|\n\n)",
    }
    text_lower = text.lower()
    for clause_type, pattern in patterns.items():
        for match in re.finditer(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            start = match.start()
            context_start = max(0, start - 200)
            context_end = min(len(text), start + 500)
            clause_text = text[context_start:context_end].strip()
            if len(clause_text) > 50:
                clauses.append({"type": clause_type, "text": clause_text})
    if not clauses:
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        for i, para in enumerate(paragraphs[:5], 1):
            clauses.append({"type": f"Section {i}", "text": para[:500]})
    return clauses

# ------------------ MAIN SECTIONS ------------------
if mode == "📄 Document Analysis":
    st.header("📄 Document Analysis & Processing")

    uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type=["pdf"])

    if uploaded_file:
        if st.session_state.current_pdf_name != uploaded_file.name:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_pdf_path = tmp_file.name

                with st.spinner("📖 Extracting text from PDF..."):
                    st.session_state.extracted_text = extract_text_from_pdf(tmp_pdf_path)
                    st.session_state.current_pdf_name = uploaded_file.name
                    st.session_state.tmp_pdf_path = tmp_pdf_path

                    # Build RAG index (persist)
                    build_index_from_pdf(tmp_pdf_path, persist_dir="data/rag_faiss_store_stable")
                    st.session_state.pdf_processed = True

                st.success(f"✅ Document processed: {uploaded_file.name}")

            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")

        # If processed and models loaded
        if st.session_state.pdf_processed and models:
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📝 Generate Summary", use_container_width=True):
                    with st.spinner("Generating summary…"):
                        try:
                            text_for_summary = st.session_state.extracted_text[:4000]
                            inputs = models['sum_tokenizer'](text_for_summary, return_tensors='pt', max_length=1024, truncation=True)
                            summary_ids = models['sum_model'].generate(
                                inputs['input_ids'],
                                max_length=300,
                                min_length=100,
                                do_sample=False
                            )
                            summary_en = models['sum_tokenizer'].decode(summary_ids[0], skip_special_tokens=True)

                            # Translate if needed
                            if target_language != "English":
                                display_summary = translate_text(summary_en, target_language)
                            else:
                                display_summary = summary_en

                            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                            st.markdown("### 📝 Document Summary")
                            st.write(display_summary)
                            if target_language != "English":
                                with st.expander("📚 Original English Summary"):
                                    st.write(summary_en)
                            st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating summary: {e}")

            with col2:
                if st.button("🌐 Translate Document Snippet", use_container_width=True):
                    with st.spinner(f"Translating to {target_language}…"):
                        try:
                            snippet = st.session_state.extracted_text[:500]
                            translated = translate_text(snippet, target_language)
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
                            clauses = extract_legal_clauses(st.session_state.extracted_text)
                            st.session_state.extracted_clauses = clauses
                            st.markdown("### 🔍 Extracted Legal Clauses")
                            for i, clause in enumerate(clauses, 1):
                                st.markdown('<div class="clause-box">', unsafe_allow_html=True)
                                st.markdown(f"**Clause {i}: {clause['type']}**")
                                snippet = clause["text"][:300] + ("…" if len(clause["text"]) > 300 else "")
                                st.write(snippet)
                                st.markdown('</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error extracting clauses: {e}")

            with st.expander("📄 View Full Extracted Text"):
                st.text_area("Document Text", st.session_state.extracted_text, height=300)

# ------------------ RAG Q&A ------------------
elif mode == "🔍 RAG Q&A":
    st.header("🔍 RAG-based Question Answering")

    if not st.session_state.pdf_processed:
        st.warning("⚠️ Please upload a document in Document Analysis mode first!")
    else:
        st.info(f"📄 Active Document: {st.session_state.current_pdf_name}")

        # Question Input
        with st.form("qa_form"):
            question = st.text_input(
                "Ask a question about the document:",
                placeholder="You can type in English, Hindi, Tamil, or Telugu"
            )
            col1, col2 = st.columns([3, 1])
            with col1:
                submit = st.form_submit_button("🔍 Ask Question", use_container_width=True)
            with col2:
                k_results = st.selectbox("Chunks", [4, 5, 6, 8], index=2)

        if submit and question and models:
            with st.spinner("🤖 Searching and generating answer..."):
                try:
                    # Detect question language
                    detected_lang = detect_language_of_text(question)
                    # Translate question -> English for RAG/model if not already English
                    question_for_model = question
                    if detected_lang != "en" and detected_lang != "unknown":
                        question_for_model = translate_text(question, "English")
                    # Query RAG with English question
                    context, docs = query_rag(question_for_model, k=k_results)

                    if "⚠️" not in context and docs:
                        # ── Multi-pass answer generation ──
                        # Generate a focused answer from each retrieved chunk,
                        # then combine for a comprehensive response.
                        chunk_answers = []
                        for doc in docs:
                            chunk_text = doc.page_content.strip()
                            if len(chunk_text) < 30:
                                continue
                            chunk_prompt = (
                                f"Based on the following legal document excerpt, "
                                f"answer the question in detail with full explanation.\n\n"
                                f"Excerpt:\n{chunk_text[:1500]}\n\n"
                                f"Question: {question_for_model}\n\n"
                                f"Detailed answer:"
                            )
                            qa_inputs = models['qa_tokenizer'](
                                chunk_prompt, return_tensors='pt',
                                max_length=512, truncation=True
                            )
                            qa_ids = models['qa_model'].generate(
                                qa_inputs['input_ids'],
                                max_length=256,
                                num_beams=4,
                                length_penalty=1.5,
                                no_repeat_ngram_size=3,
                                early_stopping=True,
                            )
                            part = models['qa_tokenizer'].decode(qa_ids[0], skip_special_tokens=True).strip()
                            if part and part not in chunk_answers:
                                chunk_answers.append(part)

                        # ── Also generate an overall summary answer ──
                        overall_prompt = (
                            f"You are a legal document assistant. "
                            f"Summarize and answer the question based on all the context "
                            f"provided below. Give a thorough, detailed response.\n\n"
                            f"Context:\n{context[:3000]}\n\n"
                            f"Question: {question_for_model}\n\n"
                            f"Comprehensive answer:"
                        )
                        ov_inputs = models['qa_tokenizer'](
                            overall_prompt, return_tensors='pt',
                            max_length=512, truncation=True
                        )
                        ov_ids = models['qa_model'].generate(
                            ov_inputs['input_ids'],
                            max_length=300,
                            num_beams=4,
                            length_penalty=1.5,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                        )
                        overall_answer = models['qa_tokenizer'].decode(ov_ids[0], skip_special_tokens=True).strip()

                        # ── Build a comprehensive combined answer ──
                        # Deduplicate chunk answers (remove ones that are substrings of another)
                        unique_answers = []
                        for ans in chunk_answers:
                            if not any(ans in other for other in chunk_answers if other != ans):
                                unique_answers.append(ans)

                        # Compose the full detailed answer
                        full_answer_parts = []
                        if overall_answer:
                            full_answer_parts.append(f"**Overview:** {overall_answer}")
                        if unique_answers:
                            full_answer_parts.append("\n**Detailed Findings from Document:**")
                            for idx, ans in enumerate(unique_answers, 1):
                                full_answer_parts.append(f"\n{idx}. {ans}")

                        # Also include key relevant excerpts inline
                        full_answer_parts.append("\n\n**Relevant Excerpts from Document:**")
                        for i, doc in enumerate(docs[:4], 1):
                            excerpt = doc.page_content.strip()[:400]
                            full_answer_parts.append(f"\n> **[Excerpt {i}]:** {excerpt}{'…' if len(doc.page_content.strip()) > 400 else ''}")

                        answer_en = "\n".join(full_answer_parts)

                        # Translate if needed
                        answer_translated = answer_en
                        if target_language != "English":
                            answer_translated = translate_text(answer_en, target_language)

                        # ── Display ──
                        st.markdown("### 🧠 Answer")

                        if target_language == "English":
                            st.markdown(f'<div class="feature-card">{answer_en}</div>', unsafe_allow_html=True)
                            st.markdown(answer_en)
                        else:
                            st.markdown(answer_translated)
                            with st.expander("📚 English Answer (original)"):
                                st.markdown(answer_en)

                        # Show full context (English) optionally
                        with st.expander("📚 View All Source Context (English)"):
                            for i, doc in enumerate(docs, 1):
                                st.markdown(f"**[Excerpt {i}]:**")
                                st.text(doc.page_content.strip())
                                st.markdown("---")

                        # Save to training data option (store English versions)
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            if st.button("➕ Add to Training Data"):
                                st.session_state.training_data.append({
                                    "question": question_for_model,
                                    "context": context,
                                    "answer": answer_en
                                })
                                st.success("Added to training data (stored in English)!")
                                st.rerun()

                        # Save to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer_translated,
                            "sources": len(docs)
                        })
                    else:
                        st.warning(context)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 💬 Question History")
            for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {chat['question'][:60]}..."):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
                    st.caption(f"📚 Based on {chat.get('sources', 0)} excerpts")

# ------------------ FINE-TUNING ------------------
elif mode == "🎓 Model Fine-tuning":
    st.header("🎓 Model Fine-tuning")

    st.markdown("""
        ### Train Your Custom Legal Model
        Improve accuracy by fine-tuning on your legal document Q&A pairs.
    """)

    st.markdown(f"### 📊 Training Dataset: {len(st.session_state.training_data)} samples")

    if st.session_state.training_data:
        with st.expander("👀 Preview Training Data"):
            for i, sample in enumerate(st.session_state.training_data[:5], 1):
                st.markdown(f"**Sample {i}:**")
                st.json(sample)

    # Manual data entry (accept input in any language; stored as English)
    st.markdown("### ➕ Add Training Samples Manually (you may write in any supported language)")
    with st.form("training_form"):
        q_input = st.text_input("Question (you can type in Hindi/Tamil/Telugu/English)")
        c_input = st.text_area("Context (preferably English or auto-translated)")
        a_input = st.text_area("Answer (you can type in any language)")

        if st.form_submit_button("Add Sample"):
            if q_input.strip() and c_input.strip() and a_input.strip():
                # Detect and translate all to English for training storage
                q_en = q_input
                c_en = c_input
                a_en = a_input
                try:
                    if detect_language_of_text(q_input) != "en":
                        q_en = translate_text(q_input, "English")
                    if detect_language_of_text(c_input) != "en":
                        c_en = translate_text(c_input, "English")
                    if detect_language_of_text(a_input) != "en":
                        a_en = translate_text(a_input, "English")
                except Exception:
                    # fallback: use raw input
                    pass

                st.session_state.training_data.append({
                    "question": q_en,
                    "context": c_en,
                    "answer": a_en
                })
                st.success("Sample added (stored in English for training).")
                st.rerun()
            else:
                st.warning("Please fill question, context and answer.")

    # Training configuration UI
    st.markdown("### ⚙️ Training Configuration")
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Epochs", 1, 10, 3)
        batch_size = st.selectbox("Batch Size", [2, 4, 8], index=1)
    with col2:
        learning_rate = st.select_slider("Learning Rate", options=[1e-5, 5e-5, 1e-4, 5e-4], value=5e-5)
        output_dir = st.text_input("Output Directory", "fine_tuned_legal_model")

    # Start training (placeholder)
    if st.button("🚀 Start Fine-tuning", use_container_width=True):
        if len(st.session_state.training_data) < 10:
            st.warning(f"⚠️ Need at least 10 training samples. Current: {len(st.session_state.training_data)}")
        else:
            st.info("🎓 Fine-tuning requires a separate training module/script (placeholder here).")
            st.info(f"Would train on {len(st.session_state.training_data)} samples with {epochs} epochs, batch size {batch_size}, lr {learning_rate}.")
            # You can plug in your training pipeline here: save training_data to JSON and call a trainer.

    # Export training data
    if st.button("💾 Export Training Data"):
        training_json = json.dumps(st.session_state.training_data, indent=2)
        st.download_button(
            "📥 Download Training Data",
            training_json,
            file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ------------------ ANALYTICS ------------------
else:  # mode == "📊 Analytics"
    st.header("📊 Performance Analytics")

    if st.session_state.pdf_processed:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Documents Processed", 1)
            st.metric("Total Questions", len(st.session_state.chat_history))

        with col2:
            st.metric("Clauses Extracted", len(st.session_state.extracted_clauses))
            st.metric("Training Samples", len(st.session_state.training_data))

        with col3:
            if st.session_state.chat_history:
                avg_sources = sum(c.get('sources', 0) for c in st.session_state.chat_history) / len(st.session_state.chat_history)
                st.metric("Avg. Sources per Answer", f"{avg_sources:.1f}")

        # Document insights
        if st.session_state.extracted_text:
            st.markdown("### 📄 Document Insights")
            text_length = len(st.session_state.extracted_text)
            word_count = len(st.session_state.extracted_text.split())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Character Count", f"{text_length:,}")
            with col2:
                st.metric("Word Count", f"{word_count:,}")

            # Top clauses preview
            if st.session_state.extracted_clauses:
                st.markdown("### 🔍 Top Extracted Clauses Preview")
                for i, clause in enumerate(st.session_state.extracted_clauses[:5], 1):
                    st.markdown(f"**Clause {i}: {clause['type']}**")
                    st.write(clause['text'][:200] + "..." if len(clause['text']) > 200 else clause['text'])
    else:
        st.info("📊 Upload a document in Document Analysis mode to see analytics.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p><strong>Legal Document Analysis Platform</strong></p>
        <p style="font-size: 0.9em;">
            Powered by FAISS, LangChain, HuggingFace Transformers<br/>
            <em>For reference purposes only. Consult qualified legal professionals for advice.</em>
        </p>
    </div>
""", unsafe_allow_html=True)
