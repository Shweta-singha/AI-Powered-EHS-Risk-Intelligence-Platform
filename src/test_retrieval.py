import sys

import chromadb
from sentence_transformers import SentenceTransformer

# Some source PDFs contain characters outside a Windows console's default
# (cp1252) encoding; re-encode stdout as UTF-8 so printing chunk text can't
# crash the script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_DIR = "data/chroma_db"
COLLECTION_NAME = "osha_compliance_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# Candidate pool size fetched from Chroma before boilerplate deprioritization
# and re-ranking -- needs to be bigger than TOP_K so a demoted boilerplate
# chunk can be replaced by the next-best genuine candidate.
OVERFETCH_N = 15

# Subtracted from a chunk's similarity score if build_rag_index.py tagged it
# is_boilerplate (generic legal-authority preamble or footer text -- see that
# script for the marker phrases). Earlier testing found such a chunk in the
# struck-by guide out-ranking the actual electrical fact sheet on a
# generically-worded electrical query; this fixes that without deleting the
# chunk outright, in case a query is genuinely about the General Duty Clause.
BOILERPLATE_PENALTY = 0.15

# Derived from the incident data's top risk categories (Day 2 EDA: falls,
# electrical, excavation/trenching, struck-by, scaffolding).
TEST_QUERIES = [
    "fall protection requirements for roofing work",
    "trenching and excavation safety requirements",
    "electrical hazard protection for construction workers",
    "scaffold safety requirements and load capacity",
    "struck-by hazards from falling or moving objects",
]


_model = None
_collection = None


def _get_model_and_collection():
    """Lazily loads and caches the embedding model and Chroma collection so
    repeated calls to search() (e.g. from an agent tool) don't reload either
    on every query."""
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _collection is None:
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _model, _collection


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Runs a query against the compliance-docs collection and returns the
    top_k chunks, overfetching and re-ranking with the boilerplate penalty
    (see BOILERPLATE_PENALTY above) so generic legal-preamble/footer chunks
    don't out-rank genuine topic-specific content."""
    model, collection = _get_model_and_collection()

    query_embedding = model.encode([query], convert_to_numpy=True)
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=OVERFETCH_N)

    candidates = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        similarity = 1 - dist  # cosine distance -> cosine similarity
        is_boilerplate = meta.get("is_boilerplate", False)
        adjusted = similarity - BOILERPLATE_PENALTY if is_boilerplate else similarity
        candidates.append((adjusted, similarity, doc, meta, is_boilerplate))

    candidates.sort(key=lambda c: c[0], reverse=True)

    return [
        {
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "similarity": similarity,
            "is_boilerplate": is_boilerplate,
            "text": doc,
        }
        for _, similarity, doc, meta, is_boilerplate in candidates[:top_k]
    ]


def main():
    _, collection = _get_model_and_collection()
    print(f"Loaded collection '{COLLECTION_NAME}' ({collection.count()} chunks)\n")

    for query in TEST_QUERIES:
        print(f"{'=' * 80}\nQuery: {query}\n{'=' * 80}")

        for rank, result in enumerate(search(query), start=1):
            snippet = result["text"][:400] + ("..." if len(result["text"]) > 400 else "")
            flag = " [boilerplate, deprioritized]" if result["is_boilerplate"] else ""
            print(
                f"\n--- Rank {rank} | source: {result['source']} "
                f"(chunk {result['chunk_index']}) | similarity: {result['similarity']:.4f}{flag} ---"
            )
            print(snippet)
        print()


if __name__ == "__main__":
    main()
