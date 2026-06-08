"""
embed.py — Embed all chunks with all-MiniLM-L6-v2 and store in ChromaDB.

Run once to build the vector store:
    python embed.py

The ChromaDB collection is persisted in ./chroma_db/. Subsequent runs
of query.py load it from disk without re-embedding.

Dependencies (install with: pip install -r requirements.txt):
    sentence-transformers, chromadb
"""

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import get_chunks

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "bing_housing"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 32  # embed this many chunks at once


def build_vector_store() -> None:
    """Embed all chunks and upsert into ChromaDB."""
    print("Loading documents and chunking...")
    chunks = get_chunks()
    print(f"  {len(chunks)} chunks ready for embedding\n")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection so we start clean on re-run
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\nEmbedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=[f"chunk_{batch_start + i}" for i in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"source": c["source"], "chunk_index": c["chunk_index"]}
                for c in batch
            ],
        )
        print(f"  Embedded chunks {batch_start}–{batch_start + len(batch) - 1}")

    count = collection.count()
    print(f"\nDone. ChromaDB collection '{COLLECTION_NAME}' has {count} vectors.")
    print(f"Vector store saved to: {CHROMA_PATH}/")


if __name__ == "__main__":
    build_vector_store()
