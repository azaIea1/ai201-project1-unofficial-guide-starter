"""
query.py — Retrieve relevant chunks and generate a grounded answer.

Usage:
    python query.py "When should I start looking for off-campus housing?"

Or import ask() from another module:
    from query import ask
    result = ask("Is it cheaper to live off campus?")
    print(result["answer"])
    print(result["sources"])

Dependencies:
    sentence-transformers, chromadb, groq, python-dotenv
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

CHROMA_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "bing_housing"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
LLM_MODEL = "llama-3.3-70b-versatile"

# Module-level singletons — loaded once, reused across calls
_embed_model: SentenceTransformer | None = None
_collection = None
_groq_client: Groq | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Return the top-k chunks most similar to the query.
    Each result: {"text": str, "source": str, "distance": float}
    """
    model = _get_embed_model()
    query_embedding = model.encode([query])[0].tolist()

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "distance": round(dist, 4),
        })
    return chunks


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about \
off-campus housing for college students, specifically near Binghamton University (BU).

You MUST answer ONLY using information from the provided source documents. \
Do not use any knowledge from your training data. \
If the documents do not contain enough information to answer the question, \
respond with exactly: "I don't have enough information in my documents to answer that."

When answering:
- Be specific and quote or paraphrase details from the documents.
- Always cite the source document(s) you drew from using the format: \
  [source: filename] at the end of each relevant sentence or paragraph.
- Do not speculate or add general knowledge beyond what the documents say."""


def ask(question: str) -> dict:
    """
    Full RAG pipeline: retrieve → generate.

    Returns:
        {
          "answer":  str,           # grounded response from the LLM
          "sources": list[str],     # unique source filenames used
          "chunks":  list[dict],    # raw retrieved chunks with distances
        }
    """
    # 1. Retrieve
    chunks = retrieve(question)

    # 2. Build context block
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Document {i} | source: {chunk['source']} | distance: {chunk['distance']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 3. Generate
    user_message = (
        f"SOURCE DOCUMENTS:\n\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using ONLY the information in the source documents above. "
        "Cite the source document filename for each claim you make."
    )

    groq = _get_groq()
    response = groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,   # low temperature = more faithful to context
        max_tokens=600,
    )

    answer = response.choices[0].message.content.strip()

    # 4. Programmatically collect source names (guaranteed attribution)
    unique_sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {
        "answer": answer,
        "sources": unique_sources,
        "chunks": chunks,
    }


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_questions = [
        sys.argv[1] if len(sys.argv) > 1 else None,
        "When should I start looking for off-campus housing near Binghamton University?",
        "Is it cheaper to live off-campus than on-campus at BU?",
        "What are common problems students face with off-campus roommates?",
    ]
    test_questions = [q for q in test_questions if q][:3]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print("="*60)

        # Show retrieval first
        chunks = retrieve(question)
        print(f"\nTop {len(chunks)} retrieved chunks:")
        for i, c in enumerate(chunks, 1):
            print(f"\n  [{i}] source: {c['source']}  distance: {c['distance']}")
            print(f"      {c['text'][:200]}...")

        # Full answer
        result = ask(question)
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")
