"""
Ingest documents from the data/ folder into a local Chroma vector store.

Run this once (and again whenever you update your docs):
    python ingest.py
"""
import os
import glob
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks

# Free, local embedding model from Hugging Face (no API key needed)
# Multilingual embedding model — works well for cross-language retrieval
# (e.g. English docs, questions asked in Spanish/Zulu/etc.)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Naive character-based chunker with overlap. Good enough for an MVP."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(filepath: str) -> str:
    """Extract all text from a PDF file, page by page."""
    reader = PdfReader(filepath)
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text)


def load_documents(data_dir: str) -> list[dict]:
    """Load all .txt and .pdf files from data_dir, returning chunks with source metadata."""
    docs = []
    filepaths = glob.glob(os.path.join(data_dir, "*.txt")) + \
        glob.glob(os.path.join(data_dir, "*.pdf"))

    for filepath in filepaths:
        if filepath.lower().endswith(".pdf"):
            print(f"  Extracting text from PDF: {os.path.basename(filepath)}")
            text = extract_text_from_pdf(filepath)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

        if not text.strip():
            print(f"  Warning: no extractable text found in {os.path.basename(filepath)} "
                  f"(likely a scanned/image-based PDF — OCR would be needed).")
            continue

        for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            docs.append({
                "id": f"{os.path.basename(filepath)}_{i}",
                "text": chunk,
                "source": os.path.basename(filepath),
            })
    return docs


def main():
    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    if not docs:
        print(f"No .txt files found in '{DATA_DIR}/'. Add some documents and re-run.")
        return
    print(f"Loaded {len(docs)} chunks from '{DATA_DIR}/'.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing embeddings...")
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    # Start fresh each time so re-running ingest.py doesn't duplicate old chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # Use cosine distance explicitly — it's always bounded 0 (identical) to 2
    # (opposite), regardless of embedding model, which keeps NO_MATCH_THRESHOLD
    # in rag_engine.py stable even if you swap embedding models later.
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[d["id"] for d in docs],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": d["source"]} for d in docs],
    )

    print(f"Done. {len(docs)} chunks stored in '{DB_DIR}/'.")


if __name__ == "__main__":
    main()