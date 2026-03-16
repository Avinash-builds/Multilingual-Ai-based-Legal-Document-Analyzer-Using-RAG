# ⚖️ Legal Document Analysis Platform

AI-Powered Legal Document Analysis, Translation & Summarization supporting English, Hindi, Tamil, Telugu, and Marathi.

## 📁 Project Structure

```
legal-rag-assistant/
│
├── app.py                          # 🚀 Advanced version (Gemini LLM)
├── app2.py                         # 🟢 Stable version (BART / Flan-T5)
├── requirements.txt                # Python dependencies
├── .env                            # API keys (gitignored)
├── .gitignore
├── README.md
│
├── core/                           # Core business logic
│   ├── __init__.py
│   ├── llm_engine.py               # Gemini LLM integration (app.py)
│   ├── rag_engine.py               # RAG pipeline for app.py
│   ├── rag_index_builder.py        # RAG pipeline for app2.py
│   ├── summarizer.py               # BART summarizer
│   ├── clause_extractor.py         # Regex clause extraction
│   ├── qa_engine.py                # Flan-T5 Q&A
│   └── model_trainer.py            # Fine-tuning module
│
├── ui/                             # Streamlit UI modules (app.py)
│   ├── __init__.py
│   ├── document_analysis.py        # Document upload & analysis
│   ├── rag_qa.py                   # RAG Q&A interface
│   ├── fine_tuning.py              # Model training UI
│   └── analytics.py                # Performance dashboard
│
├── translation/                    # Translation module
│   ├── __init__.py
│   └── handler.py                  # Helsinki-NLP translation
│
├── utils/                          # Utilities
│   ├── __init__.py
│   └── session.py                  # Session state management
│
├── data/                           # Generated data (gitignored)
│   ├── rag_faiss_store/            # FAISS index for app.py
│   └── rag_faiss_store_stable/     # FAISS index for app2.py
│
├── models/                         # Fine-tuned models (gitignored)
│   └── fine_tuned_legal_model/
│
└── docs/                           # Sample legal documents
    ├── sample_rental_agreement.pdf
    ├── legaldoc.pdf
    └── coi_2024-*.pdf
```

## 🚀 Quick Start

### 1. Create Virtual Environment

**Mac / Linux:**
```bash
python3 -m venv .venv
```

**Windows (CMD):**
```cmd
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
```

### 2. Activate Virtual Environment

**Mac / Linux:**
```bash
source .venv/bin/activate
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key (for Advanced version only)
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application

**Advanced Version** (Gemini LLM — requires API key):

| Platform | Command |
|---|---|
| Mac / Linux | `streamlit run app.py` |
| Windows (CMD) | `streamlit run app.py` |
| Windows (PowerShell) | `streamlit run app.py` |

**Stable Version** (BART / Flan-T5 — no API key needed):

| Platform | Command |
|---|---|
| Mac / Linux | `streamlit run app2.py` |
| Windows (CMD) | `streamlit run app2.py` |
| Windows (PowerShell) | `streamlit run app2.py` |

> **Note:** The `streamlit run` command is the same on all platforms. The only difference is how you activate the virtual environment (Step 2).

## 🔧 Two Application Versions

### `app.py` — Advanced Version
- Powered by **Google Gemini** LLM
- AI-powered summarization (handles full documents)
- Semantic clause extraction with risk levels
- **Document Risk Analysis** *(advanced feature)*
- **Standard Practices Comparison** *(advanced feature)*
- Gemini-powered RAG Q&A
- Context-aware translation

### `app2.py` — Stable Version
- Uses **BART** for summarization
- Uses **Flan-T5** for Q&A
- Regex-based clause extraction
- GoogleTranslator for translation
- No API key required

## 💡 Features

| Feature | app.py (Advanced) | app2.py (Stable) |
|---|---|---|
| Summarization | Gemini LLM | BART |
| Clause Extraction | Semantic + Risk Levels | Regex Patterns |
| Q&A | Gemini + RAG | Flan-T5 + RAG |
| Translation | Gemini | GoogleTranslator |
| Risk Analysis | ✅ | ❌ |
| Standard Comparison | ✅ | ❌ |
| API Key Required | ✅ (Gemini) | ❌ |

## 📚 Tech Stack

- **Framework**: Streamlit
- **LLM**: Google Gemini (Advanced)
- **ML Models**: BART, Flan-T5, all-MiniLM-L6-v2 (Stable)
- **Vector Store**: FAISS
- **PDF Processing**: PyMuPDF
- **Orchestration**: LangChain

## ⚠️ Disclaimer

This tool is for reference purposes only. Always consult qualified legal professionals for legal advice.
