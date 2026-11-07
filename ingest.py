"""Ingest a folder of PDFs into Chroma: PDF -> page text -> overlapping chunks -> embeddings.

Usage: python ingest.py --pdf-dir sample_pdfs [--reset]
"""
from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "pdf_pages"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 90


def chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Character windows with overlap, snapped back to a word boundary when possible."""
    text = " ".join(text.split())
    if not text:
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        yield text[start:end].strip()
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
        if text[start - 1] != " ":  # do not start mid-word
            nxt = text.find(" ", start)
            start = nxt + 1 if nxt != -1 and nxt < end else start


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def ingest(pdf_dir: Path, reset: bool = False):
    if reset and DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")
    collection = get_collection()
    model = SentenceTransformer(MODEL_NAME)
    total = 0
    for path in pdfs:
        ids, docs, metas = [], [], []
        reader = PdfReader(str(path))
        for page_no, page in enumerate(reader.pages, start=1):
            for chunk_no, chunk in enumerate(chunks(page.extract_text() or "")):
                ids.append(hashlib.sha256(f"{path.name}:{page_no}:{chunk_no}".encode()).hexdigest())
                docs.append(chunk)
                metas.append({"filename": path.name, "page": page_no, "chunk": chunk_no})
        collection.delete(where={"filename": path.name})  # drop stale chunks of older versions
        if not docs:
            print(f"  WARNING {path.name}: no extractable text (scanned PDF? needs OCR)")
            continue
        vectors = model.encode(docs, batch_size=32).tolist()
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
        print(f"  {path.name}: {len(reader.pages)} pages -> {len(docs)} chunks")
        total += len(docs)
    print(f"Indexed {total} chunks from {len(pdfs)} PDFs into {DB_DIR} (collection size: {collection.count()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index PDF pages into Chroma")
    parser.add_argument("--pdf-dir", type=Path, default=Path(__file__).parent / "sample_pdfs")
    parser.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = parser.parse_args()
    ingest(args.pdf_dir, args.reset)
