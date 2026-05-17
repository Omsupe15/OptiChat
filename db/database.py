"""OptiChat – SQLite database & ~/.optichat/ folder bootstrap.

Responsibilities
────────────────
• Create ~/.optichat/ and all sub-folders/files on first launch.
• Apply the SQLite schema (chats, messages, sessions).
• Provide async-friendly CRUD helpers for chats & messages.
• Manage config.json and .env read/write.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
OPTICHAT_DIR = Path.home() / ".optichat"
CONFIG_PATH = OPTICHAT_DIR / "config.json"
ENV_PATH = OPTICHAT_DIR / ".env"
MEMORY_PATH = OPTICHAT_DIR / "personalized_memory.json"
DB_PATH = OPTICHAT_DIR / "optichat.db"
CHROMA_DIR = OPTICHAT_DIR / "chroma" / "long_term"
MODELS_DIR = OPTICHAT_DIR / "models"
CHATS_DIR = OPTICHAT_DIR / "chats"


# ──────────────────────────────────────────────
#  Default config / personalized memory
# ──────────────────────────────────────────────
DEFAULT_CONFIG: dict[str, Any] = {
    "default_model": None,        # e.g. "openai/gpt-4o"
    "theme": "dark",
    "streaming": True,
    "memory_enabled": True,
}

DEFAULT_PERSONALIZED_MEMORY: dict[str, Any] = {
    "preferences": {
        "response_length": "detailed",
        "tone": "formal",
        "examples_preferred": True,
        "language": "English",
    },
    "interests": ["computer science", "cybersecurity", "philosophy"],
    "dislikes": ["excessive bullet points", "vague answers"],
    "name": "User",
    "conflict_log": [],
}


# ══════════════════════════════════════════════
#  Bootstrap – folder & file creation
# ══════════════════════════════════════════════
def bootstrap() -> None:
    """Create ~/.optichat/ tree and seed default files if missing."""
    # Directories
    for d in (OPTICHAT_DIR, CHROMA_DIR, MODELS_DIR, CHATS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # config.json
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")

    # .env
    if not ENV_PATH.exists():
        ENV_PATH.write_text(
            "# OptiChat API keys – managed by the app\n",
            encoding="utf-8",
        )

    # personalized_memory.json
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text(
            json.dumps(DEFAULT_PERSONALIZED_MEMORY, indent=2), encoding="utf-8"
        )

    # SQLite
    _init_db()


# ──────────────────────────────────────────────
#  SQLite schema
# ──────────────────────────────────────────────
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chats (
    id            TEXT PRIMARY KEY,
    name          TEXT UNIQUE,
    created_at    DATETIME,
    updated_at    DATETIME,
    model_id      TEXT,
    persona       TEXT,
    message_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id            TEXT PRIMARY KEY,
    chat_id       TEXT REFERENCES chats(id) ON DELETE CASCADE,
    role          TEXT,
    content       TEXT,
    token_count   INTEGER,
    cost_usd      REAL,
    timestamp     DATETIME,
    schema_used   TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    chat_id       TEXT REFERENCES chats(id) ON DELETE CASCADE,
    started_at    DATETIME,
    ended_at      DATETIME,
    total_tokens  INTEGER,
    total_cost    REAL
);
"""


def _get_conn() -> sqlite3.Connection:
    """Return a new connection with WAL mode and FK enforcement."""
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Apply the schema if tables do not yet exist."""
    with _get_conn() as conn:
        conn.executescript(_SCHEMA_SQL)


# ══════════════════════════════════════════════
#  Config helpers
# ══════════════════════════════════════════════
def load_config() -> dict[str, Any]:
    """Load config.json (returns defaults if file missing)."""
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict[str, Any]) -> None:
    """Persist config.json."""
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def get_default_model() -> str | None:
    """Return the configured default model id, or None."""
    return load_config().get("default_model")


def set_default_model(model_id: str) -> None:
    """Save a model id as the global default."""
    cfg = load_config()
    cfg["default_model"] = model_id
    save_config(cfg)


# ══════════════════════════════════════════════
#  .env helpers  (API keys)
# ══════════════════════════════════════════════
_ENV_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def _parse_env() -> dict[str, str]:
    """Parse the .env file into a dict."""
    result: dict[str, str] = {}
    if not ENV_PATH.exists():
        return result
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip("\"'")
    return result


def _write_env(data: dict[str, str]) -> None:
    """Write the full .env dict back."""
    lines = ["# OptiChat API keys – managed by the app"]
    for k, v in data.items():
        lines.append(f'{k}="{v}"')
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_api_key(provider: str, api_key: str) -> None:
    """Store an API key for *provider* in .env."""
    env_key = _ENV_KEY_MAP.get(provider)
    if env_key is None:
        raise ValueError(f"Unknown provider: {provider}")
    env = _parse_env()
    env[env_key] = api_key
    _write_env(env)
    # Also inject into current process env so LangChain picks it up
    os.environ[env_key] = api_key


def get_api_key(provider: str) -> str | None:
    """Return the stored key for *provider*, or None."""
    env_key = _ENV_KEY_MAP.get(provider)
    if env_key is None:
        return None
    return _parse_env().get(env_key)


def get_all_saved_providers() -> list[str]:
    """Return list of provider names that have a saved key."""
    env = _parse_env()
    providers: list[str] = []
    for provider, env_key in _ENV_KEY_MAP.items():
        if env.get(env_key):
            providers.append(provider)
    return providers


def load_env_into_process() -> None:
    """Inject all .env keys into os.environ on startup."""
    for key, value in _parse_env().items():
        os.environ[key] = value


# ══════════════════════════════════════════════
#  Chat CRUD
# ══════════════════════════════════════════════
def create_chat(name: str, model_id: str | None = None) -> str:
    """Insert a new chat row. Returns the chat id."""
    chat_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chats (id, name, created_at, updated_at, model_id, message_count)"
            " VALUES (?, ?, ?, ?, ?, 0)",
            (chat_id, name, now, now, model_id),
        )
    # Create chat-specific folder
    chat_dir = CHATS_DIR / name
    chat_dir.mkdir(parents=True, exist_ok=True)
    (chat_dir / "short_term.json").write_text("[]", encoding="utf-8")
    (chat_dir / "LRU_cached.json").write_text("[]", encoding="utf-8")
    return chat_id


def get_chat_by_name(name: str) -> dict[str, Any] | None:
    """Fetch a single chat row by name."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM chats WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def get_chat_by_id(chat_id: str) -> dict[str, Any] | None:
    """Fetch a single chat row by id."""
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    return dict(row) if row else None


def list_chats() -> list[dict[str, Any]]:
    """Return all chats ordered by most recently updated."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chats ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def rename_chat(chat_id: str, new_name: str) -> None:
    """Rename a chat (also renames its folder)."""
    chat = get_chat_by_id(chat_id)
    if chat is None:
        return
    old_name = chat["name"]
    old_dir = CHATS_DIR / old_name
    new_dir = CHATS_DIR / new_name
    with _get_conn() as conn:
        conn.execute(
            "UPDATE chats SET name = ?, updated_at = ? WHERE id = ?",
            (new_name, datetime.now(timezone.utc).isoformat(), chat_id),
        )
    if old_dir.exists():
        old_dir.rename(new_dir)


def generate_chat_title(user_message: str, max_words: int = 3) -> str:
    """Generate a short 2-3 word chat title from the first user message.

    Strips common stopwords and punctuation, then takes the first
    *max_words* meaningful words and title-cases them.
    """
    import re

    _STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "am", "i", "me",
        "my", "we", "our", "you", "your", "he", "she", "it", "they", "them",
        "its", "his", "her", "their", "this", "that", "these", "those",
        "what", "which", "who", "whom", "how", "when", "where", "why",
        "if", "then", "than", "but", "and", "or", "nor", "not", "no",
        "so", "as", "at", "by", "for", "in", "of", "on", "to", "up",
        "with", "from", "into", "about", "between", "through", "during",
        "before", "after", "above", "below", "just", "also", "very",
        "too", "some", "any", "all", "each", "every", "both", "few",
        "more", "most", "other", "such", "only", "own", "same", "tell",
        "please", "hi", "hello", "hey", "thanks", "thank",
    }

    # Remove punctuation and extra whitespace
    cleaned = re.sub(r"[^\w\s]", " ", user_message.lower())
    words = cleaned.split()

    # Filter stopwords and very short words
    meaningful = [w for w in words if w not in _STOPWORDS and len(w) > 1]

    if not meaningful:
        # Fallback: take any words if filtering removed everything
        meaningful = words[:max_words] if words else ["Untitled"]

    title_words = meaningful[:max_words]
    return " ".join(w.capitalize() for w in title_words)


def auto_rename_chat(chat_id: str, first_message: str) -> str | None:
    """Generate a title from *first_message* and rename the chat.

    Returns the new name on success, or ``None`` if the chat was not found
    or a chat with the generated name already exists (falls back to appending
    the last 4 chars of the chat id for uniqueness).

    Designed to be called from a background thread.
    """
    chat = get_chat_by_id(chat_id)
    if chat is None:
        return None

    new_name = generate_chat_title(first_message)

    # Ensure uniqueness
    if get_chat_by_name(new_name):
        new_name = f"{new_name} {chat_id[-4:]}"

    try:
        rename_chat(chat_id, new_name)
        return new_name
    except Exception:
        return None


def delete_chat(chat_id: str) -> None:
    """Delete a chat and all its messages/sessions + folder."""
    chat = get_chat_by_id(chat_id)
    with _get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    if chat:
        chat_dir = CHATS_DIR / chat["name"]
        if chat_dir.exists():
            shutil.rmtree(chat_dir, ignore_errors=True)


def update_chat_model(chat_id: str, model_id: str) -> None:
    """Change the model assigned to a chat."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE chats SET model_id = ?, updated_at = ? WHERE id = ?",
            (model_id, datetime.now(timezone.utc).isoformat(), chat_id),
        )


# ══════════════════════════════════════════════
#  Message CRUD
# ══════════════════════════════════════════════
def add_message(
    chat_id: str,
    role: str,
    content: str,
    token_count: int = 0,
    cost_usd: float = 0.0,
    schema_used: str | None = None,
) -> str:
    """Insert a message and bump the chat's message_count. Returns msg id."""
    msg_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, token_count, cost_usd, timestamp, schema_used)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, chat_id, role, content, token_count, cost_usd, now, schema_used),
        )
        conn.execute(
            "UPDATE chats SET message_count = message_count + 1, updated_at = ? WHERE id = ?",
            (now, chat_id),
        )
    return msg_id


def get_messages(chat_id: str) -> list[dict[str, Any]]:
    """Return all messages for a chat, oldest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════
#  Session CRUD
# ══════════════════════════════════════════════
def create_session(chat_id: str) -> str:
    """Start a new session for a chat. Returns session id."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, chat_id, started_at, total_tokens, total_cost)"
            " VALUES (?, ?, ?, 0, 0.0)",
            (session_id, chat_id, now),
        )
    return session_id


def end_session(session_id: str, total_tokens: int = 0, total_cost: float = 0.0) -> None:
    """Mark a session as ended."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ?, total_tokens = ?, total_cost = ? WHERE id = ?",
            (now, total_tokens, total_cost, session_id),
        )
