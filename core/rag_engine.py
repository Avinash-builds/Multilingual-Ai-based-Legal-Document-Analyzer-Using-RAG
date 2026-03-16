"""
RAG Engine — PDF ingestion, FAISS indexing, and semantic search.
"""

import os
import shutil
from typing import Tuple, List

_embeddings_cache = None


def get_embeddings():
    """Return (and lazily load) the cached HuggingFace embeddings model."""
    global _embeddings_cache
    if _embeddings_cache is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_cache


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required. Run: pip install PyMuPDF")

    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"\n--- Page {page_num + 1} ---\n{text}")
    doc.close()

    full_text = "".join(pages)
    if not full_text.strip():
        raise ValueError("PDF appears empty or contains no extractable text.")
    return full_text


def build_index_from_pdf(pdf_path: str, persist_dir: str = "./data/rag_faiss_store") -> None:
    """Build (or rebuild) a FAISS vector index from a PDF file."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    full_text = extract_text_from_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    base_doc = Document(
        page_content=full_text,
        metadata={"source": pdf_path, "filename": os.path.basename(pdf_path)},
    )
    documents = splitter.split_documents([base_doc])

    if not documents:
        raise ValueError("No chunks created from PDF.")

    embeddings = get_embeddings()
    db = FAISS.from_documents(documents, embeddings)
    os.makedirs(persist_dir, exist_ok=True)
    db.save_local(persist_dir)
    print(f"✅ Index saved ({len(documents)} chunks) → {persist_dir}")


def query_rag(
    query: str,
    persist_dir: str = "./data/rag_faiss_store",
    k: int = 4,
) -> Tuple[str, List]:
    """Perform a semantic search against the FAISS index."""
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(persist_dir):
        return "⚠️ No document index found. Please upload a PDF first.", []

    embeddings = get_embeddings()
    db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

    docs_with_scores = db.similarity_search_with_score(query, k=k)
    if not docs_with_scores:
        return "⚠️ Could not find relevant information in the document.", []

    relevant = [(doc, score) for doc, score in docs_with_scores if score < 1.5]
    if not relevant:
        return "⚠️ No sufficiently relevant information found in the document.", []

    docs = [doc for doc, _ in relevant]
    context = "\n\n".join(
        f"[Excerpt {i}]:\n{doc.page_content.strip()}" for i, doc in enumerate(docs, 1)
    )

    if len(context.strip()) < 50:
        return "⚠️ No relevant legal context found in the document.", docs

    return context, docs


def get_index_stats(persist_dir: str = "./data/rag_faiss_store") -> dict:
    """Return basic statistics about the current FAISS index."""
    if not os.path.exists(persist_dir):
        return {"exists": False}
    try:
        from langchain_community.vectorstores import FAISS
        embeddings = get_embeddings()
        db = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        return {"exists": True, "num_documents": db.index.ntotal, "dimension": db.index.d}
    except Exception as e:
        return {"exists": False, "error": str(e)}
