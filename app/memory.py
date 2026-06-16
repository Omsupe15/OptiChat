"""OptiChat – Memory Management System (Phase 3).

Responsibilities
────────────────
• Short-term memory  – rolling window of recent messages per chat.
• LRU memory         – most-recently-used important messages, swapped on overflow.
• Long-term memory   – ChromaDB vector store for semantic retrieval.
• Personalized memory – conflict-aware user preference updates.
• All heavy operations run in background threads with proper locking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db.database import (
    CHATS_DIR,
    MEMORY_PATH,
    CHROMA_DIR,
    get_messages,
    load_config,
    list_chats,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════
# Message count limits for short-term & LRU memories.
# "large" = messages with > LARGE_MSG_TOKEN_THRESHOLD tokens.
LARGE_MSG_TOKEN_THRESHOLD = 300
SHORT_TERM_LIMIT_LARGE = 3
SHORT_TERM_LIMIT_SMALL = 5
LRU_LIMIT_LARGE = 3
LRU_LIMIT_SMALL = 5

# Long-term chunking parameters
LT_CHUNK_SIZE = 400       # tokens per chunk
LT_CHUNK_OVERLAP = 50     # overlap tokens between chunks
LT_DEDUP_THRESHOLD = 0.97 # cosine similarity above which we skip insertion
LT_RETRIEVAL_TOP_K = 5    # top-k chunks to retrieve

# ChromaDB collection name
CHROMA_COLLECTION = "optichat_long_term"

# ══════════════════════════════════════════════
#  Thread-safety lock
# ══════════════════════════════════════════════
# A single asyncio lock ensures that memory-read operations wait for
# any in-progress memory-write operations to finish before proceeding.
_memory_lock = asyncio.Lock()


# ══════════════════════════════════════════════
#  Token estimation
# ══════════════════════════════════════════════
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token (GPT-family heuristic)."""
    return max(1, len(text) // 4)


def _is_large_message(content: str) -> bool:
    """Return True if the message is considered 'large'."""
    return estimate_tokens(content) > LARGE_MSG_TOKEN_THRESHOLD


# ══════════════════════════════════════════════
#  File I/O helpers for per-chat JSON files
# ══════════════════════════════════════════════
def _chat_dir(chat_name: str) -> Path:
    """Return the path to a chat's folder under ~/.optichat/chats/."""
    return CHATS_DIR / chat_name


def _read_json_file(path: Path) -> list[dict[str, Any]]:
    """Read a JSON file and return its contents (default: empty list)."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _write_json_file(path: Path, data: list[dict[str, Any]]) -> None:
    """Atomically write JSON data to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ══════════════════════════════════════════════
#  SHORT-TERM MEMORY
# ══════════════════════════════════════════════
def _get_short_term_limit(messages: list[dict[str, Any]]) -> int:
    """Return the message limit for short-term memory.

    If any message in the list is 'large', use the stricter limit (3),
    otherwise use the relaxed limit (5).
    """
    for msg in messages:
        if _is_large_message(msg.get("content", "")):
            return SHORT_TERM_LIMIT_LARGE
    return SHORT_TERM_LIMIT_SMALL


def load_short_term(chat_name: str) -> list[dict[str, Any]]:
    """Load the short-term memory for a chat."""
    return _read_json_file(_chat_dir(chat_name) / "short_term.json")


def save_short_term(chat_name: str, messages: list[dict[str, Any]]) -> None:
    """Persist the short-term memory for a chat."""
    _write_json_file(_chat_dir(chat_name) / "short_term.json", messages)


def add_to_short_term(
    chat_name: str,
    role: str,
    content: str,
) -> list[dict[str, Any]]:
    """Add a message to short-term memory and enforce the rolling window.

    Returns a list of *dropped* messages (the overflow that was evicted),
    which the caller should feed into LRU / long-term ingestion.
    """
    st = load_short_term(chat_name)

    new_msg: dict[str, Any] = {
        "role": role,
        "content": content,
        "token_count": estimate_tokens(content),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    st.append(new_msg)

    limit = _get_short_term_limit(st)
    dropped: list[dict[str, Any]] = []

    # Drop oldest messages until within limit
    while len(st) > limit:
        dropped.append(st.pop(0))

    save_short_term(chat_name, st)
    return dropped


# ══════════════════════════════════════════════
#  LRU CACHED MEMORY
# ══════════════════════════════════════════════
def load_lru(chat_name: str) -> list[dict[str, Any]]:
    """Load the LRU cached memory for a chat."""
    return _read_json_file(_chat_dir(chat_name) / "LRU_cached.json")


def save_lru(chat_name: str, messages: list[dict[str, Any]]) -> None:
    """Persist the LRU cached memory for a chat."""
    _write_json_file(_chat_dir(chat_name) / "LRU_cached.json", messages)


def _get_lru_limit(messages: list[dict[str, Any]]) -> int:
    """Return the message limit for LRU memory."""
    for msg in messages:
        if _is_large_message(msg.get("content", "")):
            return LRU_LIMIT_LARGE
    return LRU_LIMIT_SMALL


def update_lru_cache(chat_name: str, chat_id: str) -> None:
    """Rebuild the LRU cache with the most frequently referenced messages.

    This is triggered every time messages overflow from short-term memory.
    It scans the full conversation history from SQLite, computes word-level
    frequency overlap between recent messages and all past messages, and
    picks the top-N most relevant ones to keep in LRU_cached.json.
    """
    all_messages = get_messages(chat_id)
    if not all_messages:
        return

    st = load_short_term(chat_name)

    # Build a "query" from the current short-term context
    query_text = " ".join(m.get("content", "") for m in st)
    query_words = _tokenize_for_freq(query_text)
    query_counter = Counter(query_words)

    # Score every historical message by word overlap with current context
    scored: list[tuple[float, dict[str, Any]]] = []
    st_contents = {m.get("content", "") for m in st}

    for msg in all_messages:
        content = msg.get("content", "")
        # Skip messages already in short-term
        if content in st_contents:
            continue
        msg_words = _tokenize_for_freq(content)
        msg_counter = Counter(msg_words)
        # Compute overlap score (sum of min counts)
        overlap = sum((query_counter & msg_counter).values())
        if overlap > 0:
            scored.append((overlap, {
                "role": msg["role"],
                "content": content,
                "token_count": msg.get("token_count", estimate_tokens(content)),
                "timestamp": msg.get("timestamp", ""),
            }))

    # Sort by score descending (most relevant first)
    scored.sort(key=lambda x: x[0], reverse=True)

    lru_candidates = [item[1] for item in scored]
    limit = _get_lru_limit(lru_candidates) if lru_candidates else LRU_LIMIT_SMALL
    lru_final = lru_candidates[:limit]

    save_lru(chat_name, lru_final)


def _tokenize_for_freq(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for frequency analysis."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ══════════════════════════════════════════════
#  LONG-TERM MEMORY  (ChromaDB)
# ══════════════════════════════════════════════
_chroma_client = None
_chroma_collection = None
_embedding_function = None


def _get_chroma_collection():
    """Lazily initialise and return the ChromaDB collection."""
    global _chroma_client, _chroma_collection, _embedding_function

    if _chroma_collection is not None:
        return _chroma_collection

    import chromadb
    from chromadb.config import Settings

    _chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Use sentence-transformers as the embedding function
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    _embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )

    _chroma_collection = _chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


def _chunk_text(text: str, chunk_size: int = LT_CHUNK_SIZE, overlap: int = LT_CHUNK_OVERLAP) -> list[str]:
    """Split *text* into chunks of approximately *chunk_size* tokens with *overlap*.

    Uses a simple word-level split and reconstructs strings.
    """
    words = text.split()
    if not words:
        return []

    # Approximate tokens-per-word ≈ 1.3 (average for English text)
    words_per_chunk = max(1, int(chunk_size / 1.3))
    words_overlap = max(0, int(overlap / 1.3))

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - words_overlap

    return chunks


def embed_into_long_term(chat_id: str, role: str, content: str) -> None:
    """Chunk and embed a message into ChromaDB for long-term retrieval.

    Deduplication: if a chunk has cosine similarity > 0.97 with an
    existing document in the same chat, it is skipped.
    """
    collection = _get_chroma_collection()
    chunks = _chunk_text(content)

    for i, chunk in enumerate(chunks):
        # Check for near-duplicates before inserting
        try:
            results = collection.query(
                query_texts=[chunk],
                where={"chat_id": chat_id},
                n_results=1,
            )
            if results and results["distances"]:
                # ChromaDB with cosine space returns distances (1 - similarity)
                # so distance < (1 - 0.97) = 0.03 means similarity > 0.97
                distances = results["distances"][0]
                if distances and distances[0] < (1 - LT_DEDUP_THRESHOLD):
                    logger.debug("Skipping near-duplicate chunk for chat %s", chat_id)
                    continue
        except Exception:
            # If query fails (e.g. empty collection), proceed with insert
            pass

        doc_id = f"{chat_id}_{role}_{datetime.now(timezone.utc).timestamp()}_{i}"
        collection.add(
            documents=[chunk],
            ids=[doc_id],
            metadatas=[{
                "chat_id": chat_id,
                "role": role,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chunk_index": i,
            }],
        )


def retrieve_from_long_term(chat_id: str, query: str, top_k: int = LT_RETRIEVAL_TOP_K) -> list[dict[str, Any]]:
    """Query ChromaDB for the top-k most relevant chunks for a given chat.

    Returns a list of dicts with keys: content, role, score, timestamp.
    """
    collection = _get_chroma_collection()

    try:
        results = collection.query(
            query_texts=[query],
            where={"chat_id": chat_id},
            n_results=top_k,
        )
    except Exception:
        return []

    if not results or not results.get("documents"):
        return []

    retrieved: list[dict[str, Any]] = []
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    for idx, doc in enumerate(documents):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        dist = distances[idx] if idx < len(distances) else 1.0
        similarity = 1.0 - dist  # cosine distance → similarity
        retrieved.append({
            "content": doc,
            "role": meta.get("role", "unknown"),
            "score": round(similarity, 4),
            "timestamp": meta.get("timestamp", ""),
        })

    return retrieved


def filter_by_relevance(
    chunks: list[dict[str, Any]],
    threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Filter chunks below *threshold* and sort by score descending.

    Used by the prompt construction pipeline (Phase 4) after retrieval.
    """
    filtered = [c for c in chunks if c.get("score", 0) >= threshold]
    filtered.sort(key=lambda c: c.get("score", 0), reverse=True)
    return filtered


# ══════════════════════════════════════════════
#  PERSONALIZED MEMORY
# ══════════════════════════════════════════════
# Patterns that signal explicit user preferences
_PREFERENCE_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern, target field)
    (r"(?:i prefer|i like|i want)\s+(\w+)\s+(?:response|answer)s?", "response_length"),
    (r"(?:use|speak in|respond in)\s+(\w+)\s+(?:tone|style)", "tone"),
    (r"(?:i prefer|i like|i want)\s+(\w+)\s+language", "language"),
    (r"(?:don'?t|never|stop)\s+(?:use|do|give)\s+(.+)", "_dislike"),
    (r"(?:i'?m interested in|i like|i love)\s+(.+)", "_interest"),
]


def load_personalized_memory() -> dict[str, Any]:
    """Load the personalized memory JSON from disk."""
    if MEMORY_PATH.exists():
        try:
            loaded = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except (json.JSONDecodeError, OSError):
            pass
    from db.database import DEFAULT_PERSONALIZED_MEMORY
    return dict(DEFAULT_PERSONALIZED_MEMORY)


def save_personalized_memory(memory: dict[str, Any]) -> None:
    """Persist the personalized memory JSON to disk."""
    MEMORY_PATH.write_text(json.dumps(memory, indent=2, default=str), encoding="utf-8")


def update_personalized_memory_from_messages(
    chat_id: str,
    messages: list[dict[str, str]],
) -> list[str]:
    """Scan messages for explicit preference signals and update personalized memory.

    Uses conflict resolution: most-recent-wins for explicit statements.
    All changes are logged in conflict_log.

    Returns a list of human-readable confirmation strings for any updates made.
    """
    memory = load_personalized_memory()
    confirmations: list[str] = []
    now = datetime.now(timezone.utc).isoformat()

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "").lower()

        for pattern, field in _PREFERENCE_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if not match:
                continue

            value = match.group(1).strip().rstrip(".,!?")

            if field == "_dislike":
                # Add to dislikes list
                dislikes = memory.get("dislikes", [])
                if value not in dislikes:
                    dislikes.append(value)
                    memory["dislikes"] = dislikes
                    confirmations.append(f"✓ Noted dislike: {value}")
                continue

            if field == "_interest":
                # Add to interests list
                interests = memory.get("interests", [])
                if value not in interests:
                    interests.append(value)
                    memory["interests"] = interests
                    confirmations.append(f"✓ Noted interest: {value}")
                continue

            # Preference field update with conflict logging
            prefs = memory.get("preferences", {})
            old_value = prefs.get(field)

            if old_value != value:
                # Log the conflict
                conflict_entry = {
                    "field": field,
                    "old_value": old_value,
                    "new_value": value,
                    "changed_at": now,
                    "chat_id": chat_id,
                }
                memory.setdefault("conflict_log", []).append(conflict_entry)
                prefs[field] = value
                memory["preferences"] = prefs
                confirmations.append(f"✓ Preference noted: {field} → {value}")

    if confirmations:
        save_personalized_memory(memory)

    return confirmations


def update_personalized_memory_post_session(
    chat_id: str,
) -> list[str]:
    """Run a lightweight end-of-session analysis to update personalized memory.

    Scans all messages in the chat for preference signals.
    Called on session close (/quit or Ctrl+Q).
    """
    chats = list_chats()
    if len(chats) % 3 == 0:
        all_messages = get_messages(chat_id)
        simple_messages = [{"role": m["role"], "content": m["content"]} for m in all_messages]
        return update_personalized_memory_from_messages(chat_id, simple_messages)
    return []


# ══════════════════════════════════════════════
#  ORCHESTRATOR – async wrappers with locking
# ══════════════════════════════════════════════
async def process_message(
    chat_name: str,
    chat_id: str,
    role: str,
    content: str,
) -> None:
    """Process a new message through the memory pipeline.

    Called after every user message and AI response:
    1. Add to short-term memory (may drop overflow).
    2. If overflow occurred, trigger LRU cache update in background.
    3. Embed assistant responses into long-term memory in background.

    All writes are guarded by _memory_lock so that concurrent reads
    will wait for writes to finish.
    """
    async with _memory_lock:
        # 1. Add to short-term, get dropped messages
        dropped = await asyncio.to_thread(
            add_to_short_term, chat_name, role, content,
        )

        # 2. If messages were dropped, update LRU cache
        if dropped:
            await asyncio.to_thread(update_lru_cache, chat_name, chat_id)

        # 3. Embed into long-term memory (assistant responses only)
        if role == "assistant":
            await asyncio.to_thread(embed_into_long_term, chat_id, role, content)


async def get_context_for_prompt(
    chat_name: str,
    chat_id: str,
    user_query: str,
) -> dict[str, Any]:
    """Gather all memory context needed to build a prompt.

    Waits for any in-progress memory writes to finish, then returns:
    - short_term: list of recent messages
    - lru_cached: list of LRU-cached important messages
    - long_term: list of retrieved chunks from ChromaDB
    - personalized: the full personalized memory dict

    This function is safe to call concurrently – it will wait for
    _memory_lock if a write is in progress.
    """
    async with _memory_lock:
        short_term = await asyncio.to_thread(load_short_term, chat_name)
        lru_cached = await asyncio.to_thread(load_lru, chat_name)
        long_term = await asyncio.to_thread(
            retrieve_from_long_term, chat_id, user_query,
        )
        personalized = await asyncio.to_thread(load_personalized_memory)

    return {
        "short_term": short_term,
        "lru_cached": lru_cached,
        "long_term": long_term,
        "personalized": personalized,
    }


async def on_session_close(chat_id: str) -> list[str]:
    """Called when a session ends. Updates personalized memory.

    Returns list of confirmation strings for any preference updates.
    """
    async with _memory_lock:
        confirmations = await asyncio.to_thread(
            update_personalized_memory_post_session, chat_id,
        )
    return confirmations
