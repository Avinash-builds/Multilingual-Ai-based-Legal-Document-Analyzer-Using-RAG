"""
RAG Index Builder for Legal Document Analysis
Handles PDF processing, FAISS indexing, and semantic search
"""

import fitz  # PyMuPDF for PDF reading
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Tuple, List

# Cache embeddings model to avoid reloading
_embeddings_cache = None

def get_embeddings():
    """Get or create cached embeddings model"""
    global _embeddings_cache
    if _embeddings_cache is None:
        print("Loading embeddings model...")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Embeddings model loaded successfully!")
    return _embeddings_cache

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF with better error handling
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as string
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        doc.close()
        
        if not text.strip():
            raise ValueError("PDF appears to be empty or contains no extractable text")
        
        print(f"✅ Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def build_index_from_pdf(pdf_path: str, persist_dir: str = "./data/rag_faiss_store_stable") -> None:
    """
    Build FAISS index from PDF with improved chunking
    
    Args:
        pdf_path: Path to the PDF file
        persist_dir: Directory to save the FAISS index
    """
    try:
        print(f"Building index from: {pdf_path}")
        
        # Clear old index to avoid mixing documents
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            print(f"Cleared old index at {persist_dir}")
        
        # Extract text
        full_text = extract_text_from_pdf(pdf_path)
        
        # Split text into chunks with better parameters for legal documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased for better context
            chunk_overlap=200,  # Increased overlap for continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better splitting for legal docs
        )
        
        # Create document with proper metadata
        base_doc = Document(
            page_content=full_text,
            metadata={
                "source": pdf_path,
                "filename": os.path.basename(pdf_path)
            }
        )
        
        documents = text_splitter.split_documents([base_doc])
        
        if not documents:
            raise ValueError("No documents created from PDF. The file might be too short.")
        
        print(f"Created {len(documents)} text chunks")
        
        # Use cached embeddings
        embeddings = get_embeddings()
        
        # Create FAISS index
        print("Creating FAISS index...")
        db = FAISS.from_documents(documents, embeddings)
        
        # Save index
        os.makedirs(persist_dir, exist_ok=True)
        db.save_local(persist_dir)
        
        print(f"✅ Successfully created index with {len(documents)} chunks at {persist_dir}")
        
    except Exception as e:
        raise Exception(f"Error building index: {str(e)}")

def query_rag(
    query: str,
    persist_dir: str = "./data/rag_faiss_store_stable",
    k: int = 4
) -> Tuple[str, List[Document]]:
    """
    Query the RAG system with improved relevance filtering
    
    Args:
        query: User's question
        persist_dir: Directory where FAISS index is stored
        k: Number of similar documents to retrieve
        
    Returns:
        Tuple of (context_string, list_of_documents)
    """
    try:
        # Check if index exists
        if not os.path.exists(persist_dir):
            return "⚠️ No document index found. Please upload a PDF first.", []
        
        print(f"Querying: {query}")
        
        # Load embeddings and index
        embeddings = get_embeddings()
        db = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Perform similarity search with scores
        docs_with_scores = db.similarity_search_with_score(query, k=k)
        
        if not docs_with_scores:
            return "⚠️ Sorry, I could not find relevant information in the document.", []
        
        print(f"Found {len(docs_with_scores)} relevant chunks")
        
        # Filter by relevance score (lower is better for FAISS L2 distance)
        # Adjust threshold based on your needs
        relevant_docs = [
            (doc, score) for doc, score in docs_with_scores
            if score < 2.0  # Threshold for L2 distance
        ]
        
        if not relevant_docs:
            print("No sufficiently relevant docs found")
            return "⚠️ Sorry, no sufficiently relevant information found in the document.", []
        
        # Extract just the documents
        docs = [doc for doc, score in relevant_docs]
        
        # Create context with better formatting
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            context_parts.append(f"[Excerpt {i}]:\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Check if context is meaningful
        if len(context.strip()) < 50:
            return "⚠️ Sorry, no relevant legal context found in the document.", docs
        
        print(f"✅ Generated context with {len(context)} characters")
        return context, docs
        
    except Exception as e:
        error_msg = f"⚠️ Error querying document: {str(e)}"
        print(error_msg)
        return error_msg, []

def get_index_stats(persist_dir: str = "./data/rag_faiss_store_stable") -> dict:
    """
    Get statistics about the current index
    
    Args:
        persist_dir: Directory where FAISS index is stored
        
    Returns:
        Dictionary with index statistics
    """
    try:
        if not os.path.exists(persist_dir):
            return {"exists": False}
        
        embeddings = get_embeddings()
        db = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        return {
            "exists": True,
            "num_documents": db.index.ntotal,
            "dimension": db.index.d
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}

# Test function
if __name__ == "__main__":
    print("RAG Index Builder Module")
    print("Testing embeddings model...")
    try:
        embeddings = get_embeddings()
        print("✅ Embeddings model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")