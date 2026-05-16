"""OptiChat – AI model connection layer.

Responsibilities
────────────────
• Validate API keys against each provider.
• List available models from cloud providers (OpenAI / Anthropic / Gemini).
• Detect locally-installed Ollama models.
• Instantiate LangChain chat model objects for actual inference.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


# ══════════════════════════════════════════════
#  Provider registry
# ══════════════════════════════════════════════
PROVIDERS = ("openai", "anthropic", "gemini")


# ══════════════════════════════════════════════
#  API key validation
# ══════════════════════════════════════════════
def validate_api_key(provider: str, api_key: str) -> bool:
    """Return True if *api_key* is accepted by *provider*.

    Each provider check is a lightweight call (list models or a tiny request)
    wrapped in a try/except so a bad key returns False.
    """
    try:
        if provider == "openai":
            return _validate_openai(api_key)
        elif provider == "anthropic":
            return _validate_anthropic(api_key)
        elif provider == "gemini":
            return _validate_gemini(api_key)
        else:
            return False
    except Exception:
        return False


def _validate_openai(api_key: str) -> bool:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    # A successful models.list() call proves the key is valid
    models = client.models.list()
    # Consume at least one item to confirm
    _ = next(iter(models))
    return True


def _validate_anthropic(api_key: str) -> bool:
    from langchain_anthropic import ChatAnthropic

    client = ChatAnthropic(api_key=api_key)
    models = client.models.list()
    # Consume at least one item to confirm
    _ = next(iter(models))
    return True


def _validate_gemini(api_key: str) -> bool:
    from google import genai

    client = genai.Client(api_key=api_key)
    models = list(client.models.list())
    return len(models) > 0


# ══════════════════════════════════════════════
#  List cloud models
# ══════════════════════════════════════════════
def list_cloud_models(provider: str, api_key: str) -> list[dict[str, str]]:
    """Return a list of ``{id, name}`` dicts for available models.

    Only returns chat/completion-capable models where possible.
    """
    try:
        if provider == "openai":
            return _list_openai(api_key)
        elif provider == "anthropic":
            return _list_anthropic(api_key)
        elif provider == "gemini":
            return _list_gemini(api_key)
    except Exception:
        pass
    return []


def _list_openai(api_key: str) -> list[dict[str, str]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    models = client.models.list()
    result: list[dict[str, str]] = []
    for m in models:
        mid = m.id
        # Filter to chat-capable models (gpt- prefix)
        if mid.startswith(("gpt-", "o", "chatgpt")):
            result.append({"id": f"openai/{mid}", "name": mid})
    result.sort(key=lambda x: x["name"])
    return result


def _list_anthropic(api_key: str) -> list[dict[str, str]]:
    from langchain_anthropic import ChatAnthropic

    client = ChatAnthropic(api_key=api_key)
    models = client.models.list()
    result: list[dict[str, str]] = []
    for m in models:
        result.append({"id": f"anthropic/{m.id}", "name": m.display_name or m.id})
    result.sort(key=lambda x: x["name"])
    return result


def _list_gemini(api_key: str) -> list[dict[str, str]]:
    from google import genai

    client = genai.Client(api_key=api_key)
    result: list[dict[str, str]] = []
    for m in client.models.list():
        name = getattr(m, "name", "")
        display = getattr(m, "display_name", name)
        # Only include generative models
        if "gemini" in name.lower():
            result.append({"id": f"gemini/{name}", "name": display})
    result.sort(key=lambda x: x["name"])
    return result


# ══════════════════════════════════════════════
#  Ollama – local model detection
# ══════════════════════════════════════════════
def detect_ollama_models() -> list[dict[str, str]]:
    """Detect locally installed Ollama models.

    Returns a list of ``{id, name, size}`` dicts, or an empty list
    if Ollama is not running / not installed.
    """
    try:
        import ollama

        response = ollama.list()
        result: list[dict[str, str]] = []
        for m in response.models:
            model_name = m.model if hasattr(m, "model") else m.name
            size_bytes = getattr(m, "size", 0)
            size_gb = f"{size_bytes / (1024 ** 3):.1f} GB" if size_bytes else "?"
            result.append({
                "id": f"ollama/{model_name}",
                "name": model_name,
                "size": size_gb,
            })
        return result
    except Exception:
        return []


# ══════════════════════════════════════════════
#  Create a LangChain chat model instance
# ══════════════════════════════════════════════
def get_chat_model(model_id: str) -> BaseChatModel:
    """Instantiate and return a LangChain chat model for *model_id*.

    *model_id* format: ``provider/model_name``
    e.g. ``openai/gpt-4o``, ``anthropic/claude-sonnet-4-20250514``,
         ``gemini/gemini-2.0-flash``, ``ollama/llama3``.
    """
    if "/" not in model_id:
        raise ValueError(f"Invalid model_id format: {model_id!r}. Expected 'provider/model'.")

    provider, model_name = model_id.split("/", 1)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, streaming=True)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, streaming=True)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model_name, streaming=True)

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model=model_name)

    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ══════════════════════════════════════════════
#  Pipeline-aware message sending  (Phase 4)
# ══════════════════════════════════════════════
async def send_message_via_pipeline(
    model_id: str,
    user_input: str,
    chat_name: str,
    chat_id: str,
) -> str:
    """Run the user's message through the full prompt construction pipeline.

    The pipeline handles classification, memory retrieval, prompt assembly,
    LLM invocation, and post-processing (DB + memory storage).

    Returns the assistant reply string.
    """
    from app.pipeline import run_pipeline

    result = await run_pipeline(
        user_input=user_input,
        chat_name=chat_name,
        chat_id=chat_id,
        model_id=model_id,
    )

    error = result.get("error")
    if error:
        return f"*{error}*"

    return result.get("response", "")


# ══════════════════════════════════════════════
#  Legacy: direct send (no pipeline, for fallback)
# ══════════════════════════════════════════════
async def send_message(
    model_id: str,
    messages: list[dict[str, str]],
    chat_name: str | None = None,
    chat_id: str | None = None,
) -> str:
    """Send a list of {role, content} dicts and return the assistant reply.

    If *chat_name* and *chat_id* are provided, the user message and AI
    response are automatically fed through the memory pipeline.

    NOTE: For Phase 4+, prefer ``send_message_via_pipeline()`` which runs
    the full prompt construction pipeline.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    _type_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    lc_messages = [_type_map[m["role"]](content=m["content"]) for m in messages]

    model = get_chat_model(model_id)
    response = await model.ainvoke(lc_messages)
    reply = str(response.content)

    # ── Memory integration (Phase 3) ────────
    if chat_name and chat_id:
        try:
            from app.memory import process_message

            # Store the last user message in memory
            user_msgs = [m for m in messages if m["role"] == "user"]
            if user_msgs:
                await process_message(chat_name, chat_id, "user", user_msgs[-1]["content"])
            # Store the assistant reply in memory
            await process_message(chat_name, chat_id, "assistant", reply)
        except Exception:
            pass  # Memory errors must not block the response

    return reply


async def stream_message(
    model_id: str,
    messages: list[dict[str, str]],
    chat_name: str | None = None,
    chat_id: str | None = None,
):
    """Yield token chunks as an async generator.

    After streaming completes, the accumulated response is fed through
    the memory pipeline if *chat_name* and *chat_id* are provided.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    _type_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    lc_messages = [_type_map[m["role"]](content=m["content"]) for m in messages]

    model = get_chat_model(model_id)
    full_response: list[str] = []
    async for chunk in model.astream(lc_messages):
        text = chunk.content if hasattr(chunk, "content") else str(chunk)
        if text:
            full_response.append(text)
            yield text

    # ── Memory integration (Phase 3) ────────
    if chat_name and chat_id:
        try:
            from app.memory import process_message

            # Store the last user message in memory
            user_msgs = [m for m in messages if m["role"] == "user"]
            if user_msgs:
                await process_message(chat_name, chat_id, "user", user_msgs[-1]["content"])
            # Store the full accumulated assistant response in memory
            accumulated = "".join(full_response)
            if accumulated:
                await process_message(chat_name, chat_id, "assistant", accumulated)
        except Exception:
            pass  # Memory errors must not block the response
