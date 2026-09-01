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


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Loaded collection '{COLLECTION_NAME}' ({collection.count()} chunks)\n")

    for query in TEST_QUERIES:
        print(f"{'=' * 80}\nQuery: {query}\n{'=' * 80}")

        query_embedding = model.encode([query], convert_to_numpy=True)
        results = collection.query(query_embeddings=query_embedding.tolist(), n_results=OVERFETCH_N)

        candidates = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity = 1 - dist  # cosine distance -> cosine similarity
            is_boilerplate = meta.get("is_boilerplate", False)
            adjusted = similarity - BOILERPLATE_PENALTY if is_boilerplate else similarity
            candidates.append((adjusted, similarity, doc, meta, is_boilerplate))

        candidates.sort(key=lambda c: c[0], reverse=True)

        for rank, (adjusted, similarity, doc, meta, is_boilerplate) in enumerate(candidates[:TOP_K], start=1):
            snippet = doc[:400] + ("..." if len(doc) > 400 else "")
            flag = " [boilerplate, deprioritized]" if is_boilerplate else ""
            print(
                f"\n--- Rank {rank} | source: {meta['source']} "
                f"(chunk {meta['chunk_index']}) | similarity: {similarity:.4f}{flag} ---"
            )
            print(snippet)
        print()


if __name__ == "__main__":
    main()
