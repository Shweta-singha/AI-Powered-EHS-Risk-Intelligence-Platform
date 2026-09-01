import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("data/compliance_docs")
DB_DIR = "data/chroma_db"
COLLECTION_NAME = "osha_compliance_docs"

CHUNK_SIZE = 500  # approx tokens per chunk
CHUNK_OVERLAP = 100  # approx tokens of overlap between consecutive chunks

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Wingdings-style bullet glyphs land in the Private Use Area when pypdf
# extracts them from OSHA's bulleted-list PDFs -- not renderable text, and
# they crash a Windows console's default (cp1252) stdout encoding. Normalize
# to a plain hyphen bullet instead of leaving PUA codepoints in the corpus.
PRIVATE_USE_AREA_RE = re.compile("[-]")


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return PRIVATE_USE_AREA_RE.sub("-", text)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple overlapping splitter. Approximates "tokens" as whitespace-
    delimited words -- close enough for chunk sizing without pulling in a
    model-specific tokenizer for what is a small, fixed document set."""
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def main():
    doc_paths = sorted(p for p in DOCS_DIR.iterdir() if p.suffix.lower() in (".txt", ".pdf"))
    if not doc_paths:
        raise SystemExit(f"No .txt/.pdf documents found in {DOCS_DIR}")

    print(f"Found {len(doc_paths)} source documents in {DOCS_DIR}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh index each run -- avoids duplicate/stale chunks piling up across reruns.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    all_chunks, all_ids, all_metadatas = [], [], []
    for doc_path in doc_paths:
        text = load_text(doc_path)
        chunks = chunk_text(text)
        print(f"  {doc_path.name}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{doc_path.stem}_{i}")
            all_metadatas.append({"source": doc_path.name, "chunk_index": i})

    print(f"\nTotal chunks indexed: {len(all_chunks)}")

    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    collection.add(
        ids=all_ids,
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        metadatas=all_metadatas,
    )

    print(
        f"Indexed {len(all_chunks)} chunks from {len(doc_paths)} documents into "
        f"Chroma collection '{COLLECTION_NAME}' at {DB_DIR}"
    )


if __name__ == "__main__":
    main()
