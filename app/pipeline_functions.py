"""OptiChat – Prompt Construction Pipeline Functions (Phase 4 + 5).

LangGraph state definition, node functions, and helpers for the
prompt construction pipeline described in design.md §3.

Pipeline steps
──────────────
1. Classifier  – detect question type, complexity, language; check ST/LRU.
               – (Phase 5) if websearch_enabled, simultaneously fetch top-2
                 DuckDuckGo results and store them in the state.
2. Schema      – detect output schema category + depth (runs in parallel).
3. Memory      – retrieve long-term chunks from ChromaDB (if needed).
4. Relevance   – score & filter retrieved chunks (drop < 0.4).
5. Personalize – inject user preferences from personalized memory.
6. Assemble    – build the final prompt from all gathered context.
               – (Phase 5) injects [WEBSEARCH] block when results are present.
7. Invoke      – call the selected LLM and capture the response.
8. Post-process – store response in DB/memory, trigger background tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import app.memory as mem
import db.database as db

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════
#  LangGraph State
# ══════════════════════════════════════════════

class PipelineState(TypedDict, total=False):
    """Shared state passed through every node in the pipeline graph."""

    # ── Inputs ───────────────────────────────
    user_input: str                   # The raw user message
    chat_name: str                    # Chat folder name
    chat_id: str                      # Chat UUID
    model_id: str                     # e.g. "openai/gpt-4o"

    # ── Classifier outputs ───────────────────
    question_type: str                # e.g. "factual", "how-to", "code"
    complexity: str                   # "simple" | "moderate" | "complex"
    language: str                     # detected language of user input
    needs_long_term: bool             # whether long-term retrieval is needed
    context_hint: str                 # hint text found in ST/LRU (may be empty)

    # ── Schema classifier outputs ────────────
    schema_category: str              # one of the 10 schema categories
    schema_depth: str                 # "quick" | "standard" | "detailed"
    selected_output_schema: str       # formatted schema instructions

    # ── Memory context ───────────────────────
    short_term: list[dict[str, Any]]
    lru_cached: list[dict[str, Any]]
    long_term_raw: list[dict[str, Any]]    # raw retrieved chunks
    long_term_scored: list[dict[str, Any]] # after relevance scoring
    personalized: dict[str, Any]
    memory_used: bool                 # True if long-term retrieval was used

    # ── Websearch (Phase 5) ──────────────────
    websearch_enabled: bool           # True when the websearch toggle is on
    websearch_results: str            # formatted top-2 DuckDuckGo results

    # ── Prompt ───────────────────────────────
    final_prompt: list[dict[str, str]]  # list of {role, content} dicts

    # ── Response ─────────────────────────────
    response: str                     # full assistant reply (cleaned)
    raw_response: str                 # raw LLM output before trace extraction
    trace_log: str                    # chain-of-thought trace / ToDo list
    error: str | None                 # error message if LLM call failed


# ══════════════════════════════════════════════
#  Output schema definitions
# ══════════════════════════════════════════════

OUTPUT_SCHEMAS: dict[str, dict[str, str]] = {
    "factual_definition": {
        "quick": "Provide: Definition (1-2 sentences), Key Points (2-3 bullets).",
        "standard": "Provide: Definition, Key Points (4-5 bullets), Examples (1-2).",
        "detailed": "Provide: Definition, Key Points, Examples, Common Misconceptions.",
    },
    "how_to_procedural": {
        "quick": "Provide: Goal, Steps (numbered, concise).",
        "standard": "Provide: Goal, Prerequisites, Steps (numbered), Result.",
        "detailed": "Provide: Goal, Prerequisites, Steps (numbered, with explanation), Pitfalls, Result.",
    },
    "comparison": {
        "quick": "Provide: Overview, Verdict.",
        "standard": "Provide: Overview, Side-by-Side comparison, Verdict.",
        "detailed": "Provide: Overview, Side-by-Side Table, Detailed Analysis, Verdict.",
    },
    "creative_writing": {
        "quick": "Provide: Body text in the requested style.",
        "standard": "Provide: Introduction, Body, Closing.",
        "detailed": "Provide: Introduction, Body, Closing, Tone Notes.",
    },
    "historical_analytical": {
        "quick": "Provide: Context, Key takeaway.",
        "standard": "Provide: Context, Timeline, Impact.",
        "detailed": "Provide: Context, Timeline, Key Figures, Impact.",
    },
    "scientific_technical": {
        "quick": "Provide: Concept, Application.",
        "standard": "Provide: Concept, Mechanism, Application.",
        "detailed": "Provide: Concept, Mechanism, Formula/Code, Application.",
    },
    "opinion_debate": {
        "quick": "Provide: Core Argument, Conclusion.",
        "standard": "Provide: Core Argument, Supporting Points, Conclusion.",
        "detailed": "Provide: Core Argument, Supporting Points, Counter-Arguments, Conclusion.",
    },
    "personal_advice": {
        "quick": "Provide: Recommendation.",
        "standard": "Provide: Understanding your situation, Recommendation.",
        "detailed": "Provide: Understanding your situation, Options, Recommendation.",
    },
    "code_explanation": {
        "quick": "Provide: What it does, Example.",
        "standard": "Provide: What it does, How it works, Example.",
        "detailed": "Provide: What it does, How it works, Gotchas, Example.",
    },
    "open_ended_conversational": {
        "quick": "Respond naturally in a conversational tone.",
        "standard": "Respond naturally in a conversational tone.",
        "detailed": "Respond naturally in a conversational tone.",
    },
}

# Keywords used by the lightweight classifier to detect schema category
_SCHEMA_KEYWORDS: dict[str, list[str]] = {
    "factual_definition": ["what is", "define", "meaning of", "explain what", "who is", "what are"],
    "how_to_procedural": ["how to", "how do i", "steps to", "guide", "tutorial", "set up", "install"],
    "comparison": ["compare", "versus", "vs", "difference between", "which is better", "pros and cons"],
    "creative_writing": ["write", "compose", "create a story", "poem", "essay", "draft"],
    "historical_analytical": ["history of", "when did", "timeline", "evolution of", "origin of"],
    "scientific_technical": ["how does", "mechanism", "formula", "equation", "algorithm", "theory"],
    "opinion_debate": ["should", "is it good", "argue", "debate", "opinion on", "do you think"],
    "personal_advice": ["advice", "recommend", "suggest", "what should i", "help me decide"],
    "code_explanation": ["code", "function", "class", "debug", "error", "syntax", "programming", "script"],
}

# Keywords for complexity detection
_COMPLEXITY_SIGNALS: dict[str, list[str]] = {
    "complex": ["in detail", "comprehensive", "thorough", "deep dive", "elaborate", "advanced"],
    "simple": ["briefly", "quick", "short", "tldr", "tl;dr", "in one line", "summarize", "simple"],
}

# Relevance threshold – chunks below this are dropped
RELEVANCE_THRESHOLD = 0.4


# ══════════════════════════════════════════════
#  Phase 5 helper: DuckDuckGo Web Search
# ══════════════════════════════════════════════

def perform_web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted top results.

    Uses the ``duckduckgo_search`` library (already in requirements.txt).
    Returns a human-readable string with the top *max_results* results.
    If the search fails or returns nothing, an empty string is returned.

    Parameters
    ----------
    query:
        The search query derived from the user's input.
    max_results:
        Number of top results to fetch (default 2, as per Phase 5 spec).
    """
    try:
        from ddgs import DDGS

        results: list[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query=query, max_results=max_results):
                results.append(r)

        if not results:
            return ""

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "(no title)")
            href = r.get("href", "")
            body = r.get("body", "")
            # Truncate body to keep the prompt concise
            body_snippet = body[:400].strip() if body else "(no snippet)"
            lines.append(
                f"[Result {i}]\n"
                f"Title: {title}\n"
                f"URL: {href}\n"
                f"Snippet: {body_snippet}\n"
            )
        return "\n".join(lines)
    except Exception:
        logger.exception("DuckDuckGo web search failed")
        return ""


# ══════════════════════════════════════════════
#  Node 1: Classifier  (question type + context check)
# ══════════════════════════════════════════════

def classify_question(state: PipelineState) -> PipelineState:
    """Classify the user question's type, complexity, and language.

    Also checks short-term and LRU memory for relevant context hints.
    If sufficient context is found locally, sets ``needs_long_term = False``.

    Phase 5 addition: if ``websearch_enabled`` is True, simultaneously
    fetches the top-2 DuckDuckGo results for the user's query and stores
    them in ``websearch_results``.  This runs synchronously here so the
    results are ready for ``assemble_prompt`` without any extra graph nodes.
    """
    user_input = state["user_input"]
    user_lower = user_input.lower()

    # ── Detect question type ──────────────────
    question_type = "general"
    for qtype, keywords in _SCHEMA_KEYWORDS.items():
        if any(kw in user_lower for kw in keywords):
            question_type = qtype
            break

    # ── Detect complexity ─────────────────────
    complexity: str = "moderate"
    for level, keywords in _COMPLEXITY_SIGNALS.items():
        if any(kw in user_lower for kw in keywords):
            complexity = level
            break

    # ── Detect language (basic heuristic) ─────
    language = "English"  # default; could be extended with langdetect

    # ── Check ST/LRU for existing context ─────
    short_term = state.get("short_term", [])
    lru_cached = state.get("lru_cached", [])

    context_hint = ""
    needs_long_term = True

    # Search short-term and LRU for keyword overlap with the user query
    query_words = set(user_lower.split())
    if len(query_words) > 2:  # skip very short queries from matching
        for source in (short_term, lru_cached):
            for msg in source:
                content_lower = msg.get("content", "").lower()
                content_words = set(content_lower.split())
                overlap = query_words & content_words
                # If >40% of query words appear in a cached message, use it
                if len(overlap) / max(len(query_words), 1) > 0.4:
                    context_hint += msg.get("content", "") + "\n\n"
                    needs_long_term = False

    # ── Phase 5: Web search (runs simultaneously with classification) ──
    websearch_results = ""
    if state.get("websearch_enabled", False):
        logger.info("Websearch enabled — querying DuckDuckGo for: %s", user_input[:80])
        websearch_results = perform_web_search(user_input)

    return {
        **state,
        "question_type": question_type,
        "complexity": complexity,
        "language": language,
        "needs_long_term": needs_long_term,
        "context_hint": context_hint.strip(),
        "websearch_results": websearch_results,
    }


# ══════════════════════════════════════════════
#  Node 2: Schema Classifier (output schema)
# ══════════════════════════════════════════════

def classify_schema(state: PipelineState) -> PipelineState:
    """Determine the output schema category and depth for the response.

    Runs in parallel with the question classifier.
    """
    user_input = state["user_input"]
    user_lower = user_input.lower()

    # ── Detect category ───────────────────────
    category = "open_ended_conversational"  # default fallback
    best_match_count = 0
    for cat, keywords in _SCHEMA_KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw in user_lower)
        if match_count > best_match_count:
            best_match_count = match_count
            category = cat

    # ── Detect depth from complexity signals ──
    depth: str = "standard"  # default
    for kw in _COMPLEXITY_SIGNALS.get("complex", []):
        if kw in user_lower:
            depth = "detailed"
            break
    for kw in _COMPLEXITY_SIGNALS.get("simple", []):
        if kw in user_lower:
            depth = "quick"
            break

    # ── Build schema instruction text ─────────
    schema_text = OUTPUT_SCHEMAS.get(category, {}).get(depth, "")
    if not schema_text:
        schema_text = OUTPUT_SCHEMAS["open_ended_conversational"]["standard"]

    return {
        **state,
        "schema_category": category,
        "schema_depth": depth,
        "selected_output_schema": schema_text,
    }


# ══════════════════════════════════════════════
#  Node 3: Memory Retrieval (long-term)
# ══════════════════════════════════════════════

def retrieve_long_term_memory(state: PipelineState) -> PipelineState:
    """Retrieve top-5 chunks from ChromaDB for the user's question.

    Only runs if ``needs_long_term`` is True.
    """
    if not state.get("needs_long_term", True):
        return {
            **state,
            "long_term_raw": [],
            "memory_used": False,
        }

    chat_id = state["chat_id"]
    user_input = state["user_input"]

    try:
        chunks = mem.retrieve_from_long_term(chat_id, user_input, top_k=5)
    except Exception:
        logger.exception("Long-term memory retrieval failed")
        chunks = []

    return {
        **state,
        "long_term_raw": chunks,
        "memory_used": len(chunks) > 0,
    }


# ══════════════════════════════════════════════
#  Node 4: Relevance Scoring & Ordering
# ══════════════════════════════════════════════

def score_and_filter_chunks(state: PipelineState) -> PipelineState:
    """Apply relevance threshold and sort retrieved chunks.

    Chunks with score < RELEVANCE_THRESHOLD (0.4) are dropped.
    Remaining chunks are sorted by score descending.
    """
    raw_chunks = state.get("long_term_raw", [])

    scored = [c for c in raw_chunks if c.get("score", 0) >= RELEVANCE_THRESHOLD]
    scored.sort(key=lambda c: c.get("score", 0), reverse=True)

    return {
        **state,
        "long_term_scored": scored,
        "memory_used": state.get("memory_used", False) and len(scored) > 0,
    }


# ══════════════════════════════════════════════
#  Node 5: Personalization Layer
# ══════════════════════════════════════════════

def apply_personalization(state: PipelineState) -> PipelineState:
    """Load personalized memory and attach it to the state.

    Also checks whether personalized memory is enabled in config.
    """
    cfg = db.load_config()
    memory_enabled = cfg.get("memory_enabled", True)

    if memory_enabled:
        personalized = mem.load_personalized_memory()
    else:
        personalized = {}

    return {
        **state,
        "personalized": personalized,
    }


# ══════════════════════════════════════════════
#  Node 6: Final Prompt Assembly
# ══════════════════════════════════════════════

def assemble_prompt(state: PipelineState) -> PipelineState:
    """Build the final prompt from all pipeline outputs.

    Uses the template from design.md §3.2.  Additionally injects:
    - **Chain-of-thought trace** instructions: the model must first produce
      a numbered ToDo plan inside ``<TRACE>...</TRACE>`` tags.
    - **Adaptive response** instructions: response length adapts to the
      detected complexity (simple → concise, complex → thorough).
    """
    personalized = state.get("personalized", {})
    prefs = personalized.get("preferences", {})
    name = personalized.get("name", "User")
    tone = prefs.get("tone", "neutral")
    response_length = prefs.get("response_length", "standard")
    interests = ", ".join(personalized.get("interests", []))
    dislikes = ", ".join(personalized.get("dislikes", []))

    # ── [ROLE] block ──────────────────────────
    role_block = (
        f"You are a knowledgeable, thoughtful assistant talking to {name}.\n"
        f"Tone: {tone}. Response style: {response_length}.\n"
    )
    if interests:
        role_block += f"User interests: {interests}.\n"
    if dislikes:
        role_block += f"Avoid: {dislikes}.\n"

    # ── [CONTEXT] block ───────────────────────
    context_block = ""
    memory_used = state.get("memory_used", False)
    scored_chunks = state.get("long_term_scored", [])

    if memory_used and scored_chunks:
        context_block = "[RELEVANT CONTEXT — ordered by relevance score]\n"
        for i, chunk in enumerate(scored_chunks[:5], 1):
            score = chunk.get("score", 0)
            content = chunk.get("content", "")
            context_block += f"  [{i}] (score: {score:.2f}) {content}\n"
    else:
        # Use short-term and LRU context
        st_messages = state.get("short_term", [])
        lru_messages = state.get("lru_cached", [])
        context_hint = state.get("context_hint", "")

        if st_messages:
            context_block += "[SHORT-TERM MEMORY CONTEXT]\n"
            for msg in st_messages[-3:]:
                role_label = "User" if msg.get("role") == "user" else "Assistant"
                context_block += f"  {role_label}: {msg.get('content', '')[:300]}\n"

        if lru_messages:
            context_block += "\n[LRU CACHED CONTEXT]\n"
            for msg in lru_messages[-3:]:
                role_label = "User" if msg.get("role") == "user" else "Assistant"
                context_block += f"  {role_label}: {msg.get('content', '')[:300]}\n"

        if context_hint:
            context_block += f"\n[CONTEXT HINT]\n  {context_hint[:500]}\n"

    # ── [OUTPUT FORMAT] block ─────────────────
    schema_text = state.get("selected_output_schema", "")
    output_block = ""
    if schema_text:
        output_block = (
            "[OUTPUT FORMAT]\n"
            f"Structure your response using this schema:\n{schema_text}\n"
        )

    # ── [CHAIN-OF-THOUGHT TRACE] block ────────
    trace_block = (
        "[CHAIN-OF-THOUGHT TRACE]\n"
        "Before answering, you MUST create a numbered ToDo plan that breaks "
        "the user's question into parts. Output this plan inside "
        "<TRACE> and </TRACE> XML tags at the very beginning of your response.\n"
        "IMPORTANT: If [WEBSEARCH] results are present above, your numbered plan "
        "MUST be grounded in those web results then refine that data using your knowledge base.\n"
        "Inside <TRACE>, list each step as:\n"
        "  1. <step description>\n"
        "  2. <step description>\n"
        "  ...\n"
        "After the closing </TRACE> tag, execute the plan step by step strictly using the websearch and knowledge base results or if websearch is not available then only use the knowledge base.\n"
        "Then write your full response strictly following the output format instructions.\n"
    )

    # ── [ADAPTIVE RESPONSE] block ─────────────
    complexity = state.get("complexity", "moderate")
    adaptive_block = _build_adaptive_instructions(complexity)

    # ── [WEBSEARCH] block (Phase 5) ───────────
    websearch_block = ""
    websearch_results = state.get("websearch_results", "")
    if websearch_results:
        websearch_block = (
            "[WEBSEARCH — DuckDuckGo Results]\n"
            "The following real-time web search results were retrieved to "
            "help answer the user's question. Use them as context "
            "where relevant, and cite the source URLs if you quote them.\n\n"
            f"{websearch_results}\n"
        )

    # ── [USER QUESTION] block ─────────────────
    user_question_block = (
        "[USER QUESTION — give this the most attention]\n"
        f"{state['user_input']}"
    )

    # ── Assemble full system prompt ───────────
    system_content = role_block
    if websearch_block:
        system_content += "\n" + websearch_block
    if context_block:
        system_content += "\n" + context_block
    if output_block:
        system_content += "\n" + output_block
    system_content += "\n" + trace_block
    system_content += "\n" + adaptive_block

    final_prompt: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_question_block},
    ]

    return {
        **state,
        "final_prompt": final_prompt,
    }


# ══════════════════════════════════════════════
#  Node 7: Invoke LLM
# ══════════════════════════════════════════════

async def invoke_model(state: PipelineState) -> PipelineState:
    """Send the assembled prompt to the selected LLM.

    Uses the existing ``get_chat_model`` from connect_models.
    After receiving the raw response, parses ``<TRACE>...</TRACE>``
    to separate the chain-of-thought trace from the user-facing reply.
    """
    from app.connect_models import get_chat_model

    model_id = state["model_id"]
    final_prompt = state.get("final_prompt", [])

    _type_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }

    lc_messages = [
        _type_map[m["role"]](content=m["content"])
        for m in final_prompt
    ]

    try:
        model = get_chat_model(model_id)
        response = await model.ainvoke(lc_messages)
        raw_reply = str(response.content)

        # ── Parse trace from raw response ─────
        trace_log, clean_reply = _extract_trace(raw_reply)

        return {
            **state,
            "raw_response": raw_reply,
            "response": clean_reply,
            "trace_log": trace_log,
            "error": None,
        }
    except Exception as exc:
        error_msg = f"Error communicating with model: {exc}"
        logger.exception(error_msg)
        return {
            **state,
            "raw_response": "",
            "response": "",
            "trace_log": "",
            "error": error_msg,
        }


# ══════════════════════════════════════════════
#  Streaming LLM invocation helpers
# ══════════════════════════════════════════════

class StreamDone:
    """Sentinel object yielded at the end of ``stream_invoke_model``.

    Carries the final ``trace_log`` and the full cleaned ``response``
    so the caller can hand them to ``post_process`` without needing a
    second coroutine call.
    """
    __slots__ = ("trace_log", "response", "raw_response", "error")

    def __init__(
        self,
        trace_log: str,
        response: str,
        raw_response: str,
        error: str | None = None,
    ) -> None:
        self.trace_log = trace_log
        self.response = response
        self.raw_response = raw_response
        self.error = error


async def stream_invoke_model(state: PipelineState):
    """Async generator that streams the LLM response token-by-token.

    Behaviour
    ---------
    1. Buffers all incoming tokens until the closing ``</TRACE>`` tag is
       found.  Once found, the trace log is stored internally and the
       remaining tokens (and all subsequent tokens) are **yielded** as
       plain strings so the UI can update the Markdown widget in real-time.
    2. If the model never emits a ``<TRACE>`` block every token is yielded
       directly (no buffering delay).
    3. When the stream ends a :class:`StreamDone` sentinel is yielded as the
       very last item.  The caller **must** check ``isinstance(item, StreamDone)``
       to know when streaming is finished and to retrieve the ``trace_log``
       and ``response`` for post-processing.

    Yields
    ------
    str | StreamDone
        Individual token strings while streaming, then a single
        :class:`StreamDone` object when done.
    """
    from app.connect_models import get_chat_model

    model_id = state["model_id"]
    final_prompt = state.get("final_prompt", [])

    _type_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }

    lc_messages = [
        _type_map[m["role"]](content=m["content"])
        for m in final_prompt
    ]

    raw_chunks: list[str] = []
    # Accumulate tokens until we are past the <TRACE>...</TRACE> block.
    # ``trace_done`` flips to True once </TRACE> has been flushed.
    trace_done: bool = False
    # Buffer for tokens received before </TRACE> closes
    pre_trace_buf: str = ""

    try:
        model = get_chat_model(model_id)
        async for chunk in model.astream(lc_messages):
            text: str = ""
            if hasattr(chunk, "content"):
                text = chunk.content or ""
            else:
                text = str(chunk)

            if not text:
                continue

            raw_chunks.append(text)

            if trace_done:
                # Trace already extracted – yield token directly
                yield text
            else:
                # Accumulate until we see </TRACE>
                pre_trace_buf += text
                close_tag = "</TRACE>"
                close_tag_lower = close_tag.lower()
                buf_lower = pre_trace_buf.lower()
                idx = buf_lower.find(close_tag_lower)
                if idx != -1:
                    # Found the closing tag – flush remainder
                    after_tag = pre_trace_buf[idx + len(close_tag):]
                    trace_done = True
                    pre_trace_buf = ""
                    if after_tag:
                        yield after_tag
                # else: keep buffering silently

        # --- Stream finished ---
        raw_reply = "".join(raw_chunks)

        # If </TRACE> was never seen but we still have buffered content,
        # flush it now (no trace was produced).
        if not trace_done and pre_trace_buf:
            yield pre_trace_buf

        trace_log, clean_reply = _extract_trace(raw_reply)

        yield StreamDone(
            trace_log=trace_log,
            response=clean_reply,
            raw_response=raw_reply,
            error=None,
        )

    except Exception as exc:
        error_msg = f"Error communicating with model: {exc}"
        logger.exception(error_msg)
        yield StreamDone(
            trace_log="",
            response="",
            raw_response="",
            error=error_msg,
        )


async def run_pipeline_until_prompt(state: PipelineState) -> PipelineState:
    """Run all pipeline nodes up to and including ``assemble_prompt``.

    Returns the populated :class:`PipelineState` with ``final_prompt``
    ready.  This is used by the streaming path so the caller can hand
    the final prompt to :func:`stream_invoke_model` before committing
    to a full non-streaming ``ainvoke`` call.

    Phase 5: when ``websearch_enabled`` is True the function additionally
    runs :func:`generate_cot_plan` (mini model call → CoT plan) and
    :func:`second_web_search` (refined DDGS query from the plan) before
    ``assemble_prompt``, guaranteeing that **streaming only starts after
    both web-search passes are complete**.
    """
    state = classify_question(state)
    state = classify_schema(state)
    state = retrieve_long_term_memory(state)
    state = score_and_filter_chunks(state)
    state = apply_personalization(state)
    # ── Phase 5: dual web-search pass (both finish before streaming) ──
    if state.get("websearch_enabled", False):
        state = await generate_cot_plan(state)
        state = second_web_search(state)
    state = assemble_prompt(state)
    return state


# ══════════════════════════════════════════════
#  Node 8: Post-Response Processing
# ══════════════════════════════════════════════

async def post_process(state: PipelineState) -> PipelineState:
    """Store the response and trigger background memory updates.

    1. Save user message + assistant reply to SQLite.
    2. Feed both through short-term memory.
    3. Trigger LRU update and long-term embedding in background if needed.
    """
    chat_name = state["chat_name"]
    chat_id = state["chat_id"]
    user_input = state["user_input"]
    response = state.get("response", "")
    schema_category = state.get("schema_category")

    if not response:
        return state

    # 1. Save to SQLite
    user_tokens = mem.estimate_tokens(user_input)
    reply_tokens = mem.estimate_tokens(response)

    db.add_message(chat_id, "user", user_input,
                   token_count=user_tokens, schema_used=schema_category)
    db.add_message(chat_id, "assistant", response,
                   token_count=reply_tokens, schema_used=schema_category)

    # 2. Feed through memory pipeline
    try:
        await mem.process_message(chat_name, chat_id, "user", user_input)
        await mem.process_message(chat_name, chat_id, "assistant", response)
    except Exception:
        logger.exception("Memory processing failed during post-process")

    return state


# ══════════════════════════════════════════════
#  Helper: Load initial memory context (called before graph runs)
# ══════════════════════════════════════════════

async def load_memory_context(chat_name: str, chat_id: str) -> dict[str, Any]:
    """Pre-load short-term, LRU, and personalized memory for the pipeline.

    Returns a dict that can be merged into the initial PipelineState.
    """
    short_term = await asyncio.to_thread(mem.load_short_term, chat_name)
    lru_cached = await asyncio.to_thread(mem.load_lru, chat_name)
    personalized = await asyncio.to_thread(mem.load_personalized_memory)

    return {
        "short_term": short_term,
        "lru_cached": lru_cached,
        "personalized": personalized,
    }


# ══════════════════════════════════════════════
#  Conditional edge: decide if we need long-term retrieval
# ══════════════════════════════════════════════

def route_after_classify(state: PipelineState) -> str:
    """Return the next node name based on whether long-term retrieval is needed."""
    if state.get("needs_long_term", True):
        return "retrieve_long_term"
    return "score_and_filter"


def route_after_personalization(state: PipelineState) -> str:
    """Route to CoT-plan generation when websearch is on, else go straight to assembly.

    When websearch is enabled the graph visits ``generate_cot_plan`` →
    ``second_web_search`` before ``assemble_prompt`` so that both DDGS
    passes complete *before* the LLM streaming call begins.
    """
    if state.get("websearch_enabled", False):
        return "generate_cot_plan"
    return "assemble_prompt"


# ══════════════════════════════════════════════
#  Phase 5 Node: Generate CoT Plan (mini model call)
# ══════════════════════════════════════════════

async def generate_cot_plan(state: PipelineState) -> PipelineState:
    """Make a lightweight model call to produce a numbered CoT plan.

    Uses the first-pass DuckDuckGo results + user question to ask the
    model for ONLY a ``<TRACE>`` numbered plan (no full response).  The
    plan is stored in ``trace_log`` and consumed by :func:`second_web_search`
    as a refined search query.

    Skipped silently if ``websearch_enabled`` is False.
    """
    if not state.get("websearch_enabled", False):
        return state

    from app.connect_models import get_chat_model

    websearch_results = state.get("websearch_results", "")
    user_input = state["user_input"]

    ws_section = (
        f"[WEBSEARCH RESULTS — Pass 1]\n{websearch_results}\n\n"
        if websearch_results
        else ""
    )

    plan_system = (
        f"{ws_section}"
        "You are a planning assistant. Using the web search results above "
        "(and your knowledge only where results are absent), produce a "
        "concise numbered ToDo plan that breaks the user question into "
        "answerable steps. "
        "Output the plan inside <TRACE>...</TRACE> tags ONLY. "
        "Do NOT write a full response — just the plan."
    )

    try:
        model = get_chat_model(state["model_id"])
        response = await model.ainvoke([
            SystemMessage(content=plan_system),
            HumanMessage(content=user_input),
        ])
        raw = str(response.content)
        trace_log, _ = _extract_trace(raw)
        # Fallback: model skipped TRACE tags — use full output as plan
        if not trace_log:
            trace_log = raw.strip()
        logger.info("CoT plan generated (%d chars)", len(trace_log))
        return {**state, "trace_log": trace_log}
    except Exception:
        logger.exception("generate_cot_plan failed — second web search will be skipped")
        return state


# ══════════════════════════════════════════════
#  Phase 5 Node: Second Web Search (refined query)
# ══════════════════════════════════════════════

def second_web_search(state: PipelineState) -> PipelineState:
    """Run a second DDGS search using the CoT plan as a refined query.

    Merges Pass-2 results with the first-pass results already stored in
    ``websearch_results``.  After merging, ``trace_log`` is cleared so
    that the streaming response can populate it fresh from the model's
    own ``<TRACE>`` output.

    Skipped silently if ``websearch_enabled`` is False or no plan exists.
    """
    if not state.get("websearch_enabled", False):
        return state

    plan = state.get("trace_log", "").strip()
    if not plan:
        return state

    # Use first 200 chars of the plan as the refined search query
    refined_query = plan[:200]
    logger.info("Second web search — refined query from CoT plan: %s", refined_query[:80])

    second_results = perform_web_search(refined_query, max_results=5)

    first_results = state.get("websearch_results", "")
    parts: list[str] = []
    if first_results:
        parts.append("[Pass-1 Web Results — initial query]\n" + first_results)
    if second_results:
        parts.append("[Pass-2 Web Results — refined from CoT plan]\n" + second_results)
    merged = "\n\n".join(parts)

    return {
        **state,
        "websearch_results": merged,
        "trace_log": "",   # Reset — streaming response repopulates this
    }


# ══════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════

import re as _re

_TRACE_PATTERN = _re.compile(
    r"<TRACE>\s*(.*?)\s*</TRACE>",
    _re.DOTALL | _re.IGNORECASE,
)


def _extract_trace(raw_text: str) -> tuple[str, str]:
    """Extract the ``<TRACE>`` block from the raw LLM response.

    Returns ``(trace_log, cleaned_response)``.
    If no ``<TRACE>`` block is found the trace is empty and the
    full text is returned unchanged.
    """
    match = _TRACE_PATTERN.search(raw_text)
    if not match:
        return "", raw_text.strip()

    trace_log = match.group(1).strip()
    # Remove the entire <TRACE>...</TRACE> block from the reply
    cleaned = raw_text[: match.start()] + raw_text[match.end() :]
    return trace_log, cleaned.strip()


# ── Adaptive-response complexity mapping ──────
_ADAPTIVE_INSTRUCTIONS: dict[str, str] = {
    "simple": (
        "[ADAPTIVE RESPONSE]\n"
        "The question is simple. Provide a concise, focused answer. "
        "Keep it brief — a few sentences or a short paragraph is sufficient. "
        "Do not over-explain.\n"
    ),
    "moderate": (
        "[ADAPTIVE RESPONSE]\n"
        "The question is of moderate complexity. Provide a well-structured "
        "answer with enough detail to be helpful. Use paragraphs, lists, "
        "or examples as appropriate.\n"
    ),
    "complex": (
        "[ADAPTIVE RESPONSE]\n"
        "The question is complex or the user has asked for thorough "
        "explanation. Provide a comprehensive, detailed response. "
        "Cover all relevant aspects, include examples, edge cases, "
        "and explanations even if the response is long. "
        "Do NOT truncate or summarize prematurely.\n"
    ),
}


def _build_adaptive_instructions(complexity: str) -> str:
    """Return adaptive-response instructions for the given *complexity*."""
    return _ADAPTIVE_INSTRUCTIONS.get(
        complexity,
        _ADAPTIVE_INSTRUCTIONS["moderate"],
    )
