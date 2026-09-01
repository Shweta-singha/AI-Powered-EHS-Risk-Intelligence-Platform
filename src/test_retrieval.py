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

# Derived from the incident data's top risk categories (Day 2 EDA: falls,
# electrical, excavation/trenching, struck-by, scaffolding).
#
# Known limitation (see docs/model_card.md-style honesty -- documented, not
# fixed): the electrical query's correct chunks rank 5th-6th, not top-3.
# Root cause isn't a missing/bad electrical source -- it's that a page of
# generic OSHA General Duty Clause boilerplate in the struck-by guide
# ("employers must provide... free from recognized hazards...") scores
# higher against a generically-worded query than short, specific electrical
# content does. Small corpus + a general-purpose sentence embedding model
# means topic-agnostic legal boilerplate can out-rank on-topic technical text.
TEST_QUERIES = [
    "fall protection requirements for roofing work",
    "trenching and excavation safety requirements",
    "electrical hazard protection for construction workers",
    "scaffold safety requirements and load capacity",
    "struck-by hazards from falling or moving objects",
]


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Loaded collection '{COLLECTION_NAME}' ({collection.count()} chunks)\n")

    for query in TEST_QUERIES:
        print(f"{'=' * 80}\nQuery: {query}\n{'=' * 80}")

        query_embedding = model.encode([query], convert_to_numpy=True)
        results = collection.query(query_embeddings=query_embedding.tolist(), n_results=TOP_K)

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), start=1):
            similarity = 1 - dist  # cosine distance -> cosine similarity
            snippet = doc[:400] + ("..." if len(doc) > 400 else "")
            print(
                f"\n--- Rank {rank} | source: {meta['source']} "
                f"(chunk {meta['chunk_index']}) | similarity: {similarity:.4f} ---"
            )
            print(snippet)
        print()


if __name__ == "__main__":
    main()
