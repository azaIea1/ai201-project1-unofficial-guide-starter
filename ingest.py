"""
ingest.py — Load, clean, and chunk all documents in documents/.

No external dependencies required — uses only the Python standard library.

Produces a list of dicts:
    {"text": str, "source": str, "chunk_index": int}

Run standalone to print chunk stats and 5 sample chunks:
    python ingest.py
"""

import re
import random
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Revised from planning.md spec:
#   all-MiniLM-L6-v2 has a hard 256-token limit; original 400-token target
#   would be silently truncated. Reduced to ~200 tokens (800 chars) to stay
#   inside the model window and produce more retrievable chunks.
#   Overlap: ~30 tokens = 120 chars.
#   planning.md Chunking Strategy updated to reflect this change.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_LEN = 100  # drop fragments shorter than this


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents() -> list[dict]:
    """Read all .txt files from documents/, return list of {text, source}."""
    docs = []
    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        docs.append({"text": raw, "source": path.name})
    return docs


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove boilerplate that shouldn't end up in RAG chunks:
    - Reddit sidebar / navigation / ad copy
    - HTML entities
    - Excessive blank lines
    """
    # HTML entities
    text = text.replace("&amp;", "&").replace("&nbsp;", " ").replace("&#39;", "'")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')

    # Tracking / ad URLs (very long alb.reddit.com links)
    text = re.sub(r'https?://alb\.reddit\.com/\S+', '', text)
    # General bare URLs
    text = re.sub(r'https?://\S+', '', text)

    # Reddit markdown link labels that are just navigation junk
    text = re.sub(r'\[r/\w+\]\([^)]*\)', '', text)

    # Boilerplate phrases to strip entirely
    BOILERPLATE = [
        "Reddit Rules", "Privacy Policy", "User Agreement",
        "Your Privacy Choices", "Accessibility",
        "Reddit, Inc. © 2026. All rights reserved.",
        "Collapse Navigation", "Collapse video player",
        "Search CommentsExpand comment search",
        "Comments Section", "Community Info Section",
        "Community Bookmarks", "Sort by:Best",
        "Go to commentsShare", "Upvote", "Downvote",
        "Promoted", "Learn More", "Shop Now", "Sign Up",
        "View More", "Weekly visitors", "Weekly contributions",
        "Procrastinating", '"Studying" Right Now',
        "Public", "Created ", "Helpful Threads",
        "Related Subreddits", "Quick Links",
        "r/college Rules", "r/BinghamtonUniversity Rules",
        "r/Binghamton Rules", "Filter by Flair", "Filter Topics",
        "Message Mods", "View all moderators",
        "Installed Apps", "Photo", "User flair", "Calendar",
        "Moderators",
    ]
    for phrase in BOILERPLATE:
        text = text.replace(phrase, "")

    # Remove lines that are empty or suspiciously short (likely nav items)
    lines = [ln for ln in text.splitlines() if len(ln.strip()) > 1]
    text = "\n".join(lines)

    # Collapse 3+ blank lines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ---------------------------------------------------------------------------
# Chunking — recursive character splitter (no langchain required)
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks using a sliding-window approach.

    Tries to break at paragraph → newline → sentence → word boundaries
    so chunks don't cut through mid-sentence. The overlap is implemented
    by rewinding the start pointer — no text is literally duplicated.
    """
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text] if len(text) >= MIN_CHUNK_LEN else []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        if end >= len(text):
            tail = text[start:].strip()
            if len(tail) >= MIN_CHUNK_LEN:
                chunks.append(tail)
            break

        # Find a clean split point, preferring larger separators
        split_at = end
        for sep in ["\n\n", "\n", ". ", " "]:
            pos = text.rfind(sep, start + CHUNK_SIZE // 2, end)
            if pos != -1:
                split_at = pos + len(sep)
                break

        chunk = text[start:split_at].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)

        # Rewind by overlap amount (never go backwards)
        next_start = split_at - CHUNK_OVERLAP
        if next_start <= start:
            next_start = split_at  # safety: always advance
        start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_chunks() -> list[dict]:
    """Load, clean, and chunk all documents. Returns list of chunk dicts."""
    docs = load_documents()
    chunks = []
    for doc in docs:
        cleaned = clean_text(doc["text"])
        pieces = chunk_text(cleaned)
        for i, piece in enumerate(pieces):
            chunks.append({
                "text": piece.strip(),
                "source": doc["source"],
                "chunk_index": i,
            })
    return chunks


# ---------------------------------------------------------------------------
# Standalone inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents\n")

    # Show cleaning effect on the first document
    sample = docs[0]
    raw_preview = sample["text"][:600]
    cleaned_preview = clean_text(sample["text"])[:600]
    print(f"=== RAW (first 600 chars of {sample['source']}) ===")
    print(raw_preview)
    print(f"\n=== AFTER CLEANING ===")
    print(cleaned_preview)

    # Chunk stats
    all_chunks = get_chunks()
    print(f"\n=== CHUNK STATS ===")
    print(f"Total chunks: {len(all_chunks)}")
    lengths = [len(c["text"]) for c in all_chunks]
    print(f"Min length : {min(lengths)} chars")
    print(f"Max length : {max(lengths)} chars")
    print(f"Avg length : {sum(lengths) // len(lengths)} chars")

    sources: dict[str, int] = {}
    for c in all_chunks:
        sources[c["source"]] = sources.get(c["source"], 0) + 1
    print("\nChunks per document:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    # 5 random representative chunks
    print("\n=== 5 RANDOM CHUNKS ===")
    sample_chunks = random.sample(all_chunks, min(5, len(all_chunks)))
    for i, chunk in enumerate(sample_chunks):
        print(f"\n--- Chunk {i+1} (source: {chunk['source']}, index: {chunk['chunk_index']}) ---")
        print(chunk["text"][:500])
        if len(chunk["text"]) > 500:
            print("  [... truncated ...]")
