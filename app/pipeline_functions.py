"""OptiChat prompt pipeline sub-agent functions.

This module keeps the existing public pipeline contract while refactoring the
Phase 4/5 flow into explicit LLM-backed sub-agent nodes. Deterministic I/O
still stays in code: DuckDuckGo search, database writes, and memory operations
are not delegated to the model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import app.memory as mem
import db.database as db

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    """Shared LangGraph state for the prompt construction pipeline."""

    user_input: str
    chat_name: str
    chat_id: str
    model_id: str

    question_type: str
    complexity: str
    language: str
    needs_long_term: bool
    context_hint: str
    memory_path: str

    schema_category: str
    schema_depth: str
    selected_output_schema: str

    short_term: list[dict[str, Any]]
    lru_cached: list[dict[str, Any]]
    long_term_raw: list[dict[str, Any]]
    long_term_scored: list[dict[str, Any]]
    personalized: dict[str, Any]
    personalization_summary: str
    memory_used: bool

    websearch_enabled: bool
    websearch_results: str
    search_queries: list[str]
    web_sources: list[dict[str, Any]]
    web_summary: str

    needs_memory: bool
    needs_websearch: bool

    agent_logs: list[dict[str, Any]]
    agent_errors: list[str]
    visible_trace_log: str
    answer_plan: list[str]

    final_prompt: list[dict[str, str]]
    response: str
    raw_response: str
    trace_log: str
    error: str | None


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

_COMPLEXITY_SIGNALS: dict[str, list[str]] = {
    "complex": ["in detail", "comprehensive", "thorough", "deep dive", "elaborate", "advanced"],
    "simple": ["briefly", "quick", "short", "tldr", "tl;dr", "in one line", "summarize", "simple"],
}

RELEVANCE_THRESHOLD = 0.4
MAX_WEB_QUERIES = 4
MAX_RESULTS_PER_QUERY = 4


class StreamDone:
    """Sentinel yielded after streaming has completed."""

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


def _type_map() -> dict[str, type[SystemMessage] | type[HumanMessage] | type[AIMessage]]:
    return {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }


def _to_lc_messages(messages: list[dict[str, str]]):
    types = _type_map()
    return [types[m["role"]](content=m["content"]) for m in messages]


def _extract_text_content(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            elif hasattr(part, "text"):
                parts.append(part.text)
            elif hasattr(part, "get") and part.get("text"):
                parts.append(part.get("text"))
        return "".join(parts)
    return str(content) if content is not None else ""


def _compact_json(value: Any, max_chars: int = 5000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _safe_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


async def _agent_json_call(
    state: PipelineState,
    agent_name: str,
    system_prompt: str,
    payload: dict[str, Any],
    fallback: dict[str, Any],
) -> tuple[dict[str, Any], PipelineState]:
    """Call an LLM sub-agent and parse a JSON object, with one repair retry."""
    from app.connect_models import get_chat_model

    model = get_chat_model(state["model_id"])
    messages = [
        SystemMessage(
            content=(
                f"You are the {agent_name} sub-agent in OptiChat. "
                "Return exactly one valid JSON object and no markdown. "
                "Do not include hidden reasoning or chain-of-thought."
                f"\n\n{system_prompt}"
            )
        ),
        HumanMessage(content=_compact_json(payload)),
    ]

    raw = ""
    try:
        response = await model.ainvoke(messages)
        raw = _extract_text_content(response.content)
        parsed = _safe_json_from_text(raw)
        if parsed is not None:
            return parsed, _append_log(state, agent_name, "ok", {"parsed": True})

        repair = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Repair this model output into exactly one valid JSON object. "
                        "No markdown, no explanation."
                    )
                ),
                HumanMessage(content=raw),
            ]
        )
        parsed = _safe_json_from_text(_extract_text_content(repair.content))
        if parsed is not None:
            return parsed, _append_log(state, agent_name, "ok", {"parsed_after_repair": True})

        raise ValueError("sub-agent returned invalid JSON")
    except Exception as exc:
        logger.exception("%s sub-agent failed", agent_name)
        state = _append_error(state, f"{agent_name}: {exc}")
        state = _append_log(
            state,
            agent_name,
            "fallback",
            {"reason": str(exc), "raw": raw[:500]},
        )
        return fallback, state


def _append_log(
    state: PipelineState,
    agent_name: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> PipelineState:
    logs = list(state.get("agent_logs", []))
    logs.append({"agent": agent_name, "status": status, "details": details or {}})
    return {**state, "agent_logs": logs}


def _append_error(state: PipelineState, message: str) -> PipelineState:
    errors = list(state.get("agent_errors", []))
    errors.append(message)
    return {**state, "agent_errors": errors}


def _heuristic_category(user_lower: str) -> str:
    category = "open_ended_conversational"
    best_match_count = 0
    for candidate, keywords in _SCHEMA_KEYWORDS.items():
        count = sum(1 for keyword in keywords if keyword in user_lower)
        if count > best_match_count:
            best_match_count = count
            category = candidate
    return category


def _heuristic_complexity(user_lower: str) -> str:
    for level, keywords in _COMPLEXITY_SIGNALS.items():
        if any(keyword in user_lower for keyword in keywords):
            return level
    return "moderate"


def _memory_overlap_hint(
    user_input: str,
    short_term: list[dict[str, Any]],
    lru_cached: list[dict[str, Any]],
) -> tuple[str, bool]:
    query_words = {w for w in re.findall(r"\w+", user_input.lower()) if len(w) > 2}
    if len(query_words) <= 2:
        return "", True

    hints: list[str] = []
    for source in (short_term, lru_cached):
        for msg in source:
            content = str(msg.get("content", ""))
            content_words = set(re.findall(r"\w+", content.lower()))
            overlap = query_words & content_words
            if len(overlap) / max(len(query_words), 1) > 0.4:
                hints.append(content)

    return "\n\n".join(hints[:3]), not hints


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return default


async def orchestrator_agent(state: PipelineState) -> PipelineState:
    """Single LLM call combining classifier + orchestrator.

    Classifies the question AND decides which sub-agents (memory, websearch)
    are required.  Replaces the old ``unified_classifier_agent``.
    """
    user_input = state["user_input"]
    user_lower = user_input.lower()

    # ── Deterministic pre-computation (free, no LLM) ──────────────
    context_hint, needs_long_term = _memory_overlap_hint(
        user_input,
        state.get("short_term", []),
        state.get("lru_cached", []),
    )
    heuristic_category = _heuristic_category(user_lower)
    heuristic_complexity = _heuristic_complexity(user_lower)
    depth_map = {"simple": "quick", "moderate": "standard", "complex": "detailed"}
    heuristic_depth = depth_map.get(heuristic_complexity, "standard")

    try:
        cfg = db.load_config()
        memory_enabled = cfg.get("memory_enabled", True)
    except Exception:
        logger.exception("Config load failed during orchestrator classification")
        memory_enabled = True

    personalized = mem.load_personalized_memory() if memory_enabled else {}

    websearch_toggle = state.get("websearch_enabled", False)

    # ── Combined fallback ─────────────────────────────────────────
    fallback = {
        "question_type": heuristic_category,
        "complexity": heuristic_complexity,
        "language": "English",
        "needs_long_term": needs_long_term,
        "context_hint": context_hint,
        "memory_path": "long_term" if needs_long_term else "short_lru",
        "schema_category": heuristic_category,
        "schema_depth": heuristic_depth,
        "personalization_summary": (
            "Use default assistant behavior." if not personalized
            else "Use stored user preferences where relevant."
        ),
        "relevant_preferences": (
            personalized.get("preferences", {})
            if isinstance(personalized, dict) else {}
        ),
        "needs_memory": needs_long_term,
        "needs_websearch": websearch_toggle,
    }

    # ── Single LLM call ───────────────────────────────────────────
    result, state = await _agent_json_call(
        state,
        "orchestrator_agent",
        (
            "You are the Orchestrator Agent in OptiChat. Perform ALL of these "
            "tasks in one JSON response:\n"
            "1. UNDERSTAND the question: determine whether user wants a "
            "detailed answer or short answer → return complexity "
            "(simple/moderate/complex).\n"
            "2. SELECT output schema: return schema_category (one of "
            f"{list(OUTPUT_SCHEMAS.keys())}) and schema_depth "
            "(quick/standard/detailed).\n"
            "3. PERSONALIZE: decide if personalisation is relevant → return "
            "personalization_summary (one sentence) and relevant_preferences.\n"
            "4. MEMORY: decide if memory retrieval is required (true only if "
            "the question seems related to previous conversation) → return "
            "needs_memory (bool).\n"
            "5. WEBSEARCH: decide if web search is required (true only if the "
            "question needs recent/current data the model may not have, e.g. "
            "current events, recent news, latest research). The user's websearch "
            f"toggle is {'ON' if websearch_toggle else 'OFF'}. If toggle is OFF, "
            "needs_websearch must be false. → return needs_websearch (bool).\n"
            "6. Also return question_type, language, needs_long_term (bool), "
            "context_hint (string), memory_path (long_term or short_lru).\n\n"
            "Return a single JSON object with ALL these keys."
        ),
        {
            "user_input": user_input,
            "short_term_recent": state.get("short_term", [])[-5:],
            "lru_cached_recent": state.get("lru_cached", [])[-5:],
            "personalized_memory": personalized,
            "available_schemas": OUTPUT_SCHEMAS,
            "websearch_toggle": websearch_toggle,
            "heuristic_fallback": fallback,
        },
        fallback,
    )

    # ── Parse classification fields ───────────────────────────────
    complexity = str(result.get("complexity", fallback["complexity"])).lower()
    if complexity not in {"simple", "moderate", "complex"}:
        complexity = fallback["complexity"]

    question_type = str(result.get("question_type", fallback["question_type"]))
    language = str(result.get("language", fallback["language"]))
    lt = _normalize_bool(result.get("needs_long_term"), fallback["needs_long_term"])
    c_hint = str(result.get("context_hint", fallback["context_hint"])).strip()
    m_path = str(result.get("memory_path", fallback["memory_path"]))

    # ── Parse schema fields ───────────────────────────────────────
    category = str(result.get("schema_category", fallback["schema_category"]))
    if category not in OUTPUT_SCHEMAS:
        category = fallback["schema_category"]
    depth = str(result.get("schema_depth", fallback["schema_depth"]))
    if depth not in OUTPUT_SCHEMAS[category]:
        depth = fallback["schema_depth"]

    # ── Parse personalization fields ──────────────────────────────
    p_summary = str(
        result.get("personalization_summary", fallback["personalization_summary"])
    )
    rel_prefs = result.get("relevant_preferences", fallback["relevant_preferences"])
    if not isinstance(rel_prefs, dict):
        rel_prefs = fallback["relevant_preferences"]
    if isinstance(personalized, dict):
        personalized = {**personalized, "relevant_preferences": rel_prefs}

    # ── Parse sub-agent routing decisions ─────────────────────────
    needs_memory = _normalize_bool(
        result.get("needs_memory"), fallback["needs_memory"]
    )
    needs_websearch = _normalize_bool(
        result.get("needs_websearch"), fallback["needs_websearch"]
    )
    # Websearch cannot be enabled if the user toggle is OFF
    if not websearch_toggle:
        needs_websearch = False

    return {
        **state,
        "question_type": question_type,
        "complexity": complexity,
        "language": language,
        "needs_long_term": lt,
        "context_hint": c_hint,
        "memory_path": m_path,
        "schema_category": category,
        "schema_depth": depth,
        "selected_output_schema": OUTPUT_SCHEMAS[category][depth],
        "personalized": personalized,
        "personalization_summary": p_summary,
        "needs_memory": needs_memory,
        "needs_websearch": needs_websearch,
    }


# ── Legacy wrappers (kept for backward-compat / non-streaming path) ──

unified_classifier_agent = orchestrator_agent


async def classifier_agent(state: PipelineState) -> PipelineState:
    """Legacy: now delegates to orchestrator_agent."""
    return await orchestrator_agent(state)


async def schema_agent(state: PipelineState) -> PipelineState:
    """Legacy: classification + schema handled by orchestrator_agent."""
    return state


async def personalization_agent(state: PipelineState) -> PipelineState:
    """Legacy: personalization handled by orchestrator_agent."""
    return state


async def memory_agent(state: PipelineState) -> PipelineState:
    """LLM sub-agent for memory selection, with deterministic retrieval."""
    raw_chunks: list[dict[str, Any]] = []
    if state.get("needs_long_term", True):
        try:
            raw_chunks = mem.retrieve_from_long_term(state["chat_id"], state["user_input"], top_k=5)
        except Exception as exc:
            logger.exception("Long-term memory retrieval failed")
            state = _append_error(state, f"memory_agent retrieval: {exc}")

    candidate_chunks = [
        c for c in raw_chunks if float(c.get("score", 0) or 0) >= RELEVANCE_THRESHOLD
    ]
    candidate_chunks.sort(key=lambda c: float(c.get("score", 0) or 0), reverse=True)
    fallback = {
        "selected_chunk_indexes": list(range(min(len(candidate_chunks), 5))),
        "memory_used": bool(candidate_chunks),
        "memory_path": "long_term" if candidate_chunks else state.get("memory_path", "short_lru"),
        "context_hint": state.get("context_hint", ""),
    }
    result, state = await _agent_json_call(
        state,
        "memory_agent",
        (
            "Choose which memory context should be used. Return selected_chunk_indexes "
            "as zero-based indexes into candidate_chunks, memory_used, memory_path, "
            "and a short context_hint. Do not invent memory."
        ),
        {
            "user_input": state["user_input"],
            "needs_long_term": state.get("needs_long_term"),
            "short_term": state.get("short_term", [])[-5:],
            "lru_cached": state.get("lru_cached", [])[-5:],
            "candidate_chunks": candidate_chunks,
        },
        fallback,
    )
    indexes = result.get("selected_chunk_indexes", fallback["selected_chunk_indexes"])
    if not isinstance(indexes, list):
        indexes = fallback["selected_chunk_indexes"]
    selected: list[dict[str, Any]] = []
    for index in indexes[:5]:
        if isinstance(index, int) and 0 <= index < len(candidate_chunks):
            selected.append(candidate_chunks[index])

    memory_used = bool(selected) and _normalize_bool(result.get("memory_used"), True)
    return {
        **state,
        "long_term_raw": raw_chunks,
        "long_term_scored": selected,
        "memory_used": memory_used,
        "memory_path": str(result.get("memory_path", fallback["memory_path"])),
        "context_hint": str(result.get("context_hint", fallback["context_hint"])).strip(),
    }


def _run_ddgs_query(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[dict[str, Any]]:
    from ddgs import DDGS

    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for rank, item in enumerate(ddgs.text(query=query, max_results=max_results), 1):
            results.append(
                {
                    "title": str(item.get("title", "")).strip() or "(no title)",
                    "url": str(item.get("href", "")).strip(),
                    "snippet": str(item.get("body", "")).strip()[:700],
                    "query": query,
                    "rank": rank,
                }
            )
    return results


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for source in sources:
        url = str(source.get("url", "")).strip()
        key = url or f"{source.get('title')}::{source.get('snippet')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _format_web_results(sources: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, source in enumerate(sources, 1):
        lines.append(
            f"[Source {index}]\n"
            f"Title: {source.get('title', '(no title)')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Query: {source.get('query', '')}\n"
            f"Snippet: {source.get('snippet', '')}\n"
        )
    return "\n".join(lines)


async def websearch_agent(state: PipelineState) -> PipelineState:
    """LLM sub-agent for query planning, source ranking, and web summary."""
    if not state.get("websearch_enabled", False):
        state = _append_log(state, "websearch_agent", "skipped", {"websearch_enabled": False})
        return {
            **state,
            "search_queries": [],
            "web_sources": [],
            "web_summary": "",
            "websearch_results": "",
        }

    today = date.today().isoformat()
    query_fallback = {
        "queries": [state["user_input"]],
    }
    query_result, state = await _agent_json_call(
        state,
        "websearch_agent.query_planner",
        (
            f"Today is {today}. Generate 2 to 4 targeted DuckDuckGo search "
            "queries for accurate, current evidence. Return {'queries': [...]}."
        ),
        {
            "user_input": state["user_input"],
            "question_type": state.get("question_type"),
            "complexity": state.get("complexity"),
        },
        query_fallback,
    )
    raw_queries = query_result.get("queries", query_fallback["queries"])
    if not isinstance(raw_queries, list):
        raw_queries = query_fallback["queries"]
    queries = [str(q).strip() for q in raw_queries if str(q).strip()][:MAX_WEB_QUERIES]
    if not queries:
        queries = [state["user_input"]]

    sources: list[dict[str, Any]] = []
    for query in queries:
        try:
            sources.extend(await asyncio.to_thread(_run_ddgs_query, query, MAX_RESULTS_PER_QUERY))
        except Exception as exc:
            logger.exception("DDGS query failed")
            state = _append_error(state, f"websearch_agent DDGS query failed for {query!r}: {exc}")
            state = _append_log(state, "websearch_agent", "warning", {"query": query, "error": str(exc)})

    sources = _dedupe_sources(sources)
    rank_fallback = {
        "selected_indexes": list(range(min(len(sources), 6))),
        "web_summary": "Use the selected web sources as supporting context.",
        "missing_fact_query": "",
    }
    rank_result, state = await _agent_json_call(
        state,
        "websearch_agent.source_ranker",
        (
            "Rank the provided sources for answering the user. Return selected_indexes "
            "as zero-based indexes, web_summary, and missing_fact_query. Only set "
            "missing_fact_query if one concrete missing fact needs a second search."
        ),
        {
            "today": today,
            "user_input": state["user_input"],
            "queries": queries,
            "sources": sources,
        },
        rank_fallback,
    )

    missing_query = str(rank_result.get("missing_fact_query", "")).strip()
    if missing_query:
        try:
            sources.extend(await asyncio.to_thread(_run_ddgs_query, missing_query, MAX_RESULTS_PER_QUERY))
            sources = _dedupe_sources(sources)
            queries.append(missing_query)
        except Exception as exc:
            logger.exception("DDGS second-pass query failed")
            state = _append_error(state, f"websearch_agent second-pass failed for {missing_query!r}: {exc}")

    indexes = rank_result.get("selected_indexes", rank_fallback["selected_indexes"])
    if not isinstance(indexes, list):
        indexes = rank_fallback["selected_indexes"]
    selected: list[dict[str, Any]] = []
    for index in indexes[:8]:
        if isinstance(index, int) and 0 <= index < len(sources):
            selected.append(sources[index])
    if not selected:
        selected = sources[:6]

    web_summary = str(rank_result.get("web_summary", rank_fallback["web_summary"])).strip()
    web_results = _format_web_results(selected)
    state = _append_log(
        state,
        "websearch_agent",
        "ok" if selected else "no_results",
        {"queries": queries, "selected_source_count": len(selected)},
    )
    return {
        **state,
        "search_queries": queries,
        "web_sources": selected,
        "web_summary": web_summary,
        "websearch_results": web_results,
    }


def route_after_personalization(state: PipelineState) -> str:
    return "websearch_agent" if state.get("websearch_enabled", False) else "prompt_assembly_agent"


async def prompt_assembly_agent(state: PipelineState) -> PipelineState:
    """LLM sub-agent for visible planning, then deterministic prompt assembly."""
    fallback = {
        "answer_plan": [
            "Identify the user's main request.",
            "Use selected memory and web evidence where relevant.",
            "Answer using the selected response schema.",
        ]
    }
    result, state = await _agent_json_call(
        state,
        "prompt_assembly_agent",
        (
            "Create a concise visible action plan for the final answer. This plan "
            "will be shown to the user. Do not include hidden reasoning. Return "
            "{'answer_plan': ['...']} in a concise precise manner. Use only 5-6 steps."
        ),
        {
            "user_input": state["user_input"],
            "schema": state.get("selected_output_schema"),
            "memory_path": state.get("memory_path"),
            "websearch_enabled": state.get("websearch_enabled"),
            "web_summary": state.get("web_summary"),
            "source_urls": [s.get("url") for s in state.get("web_sources", [])],
        },
        fallback,
    )
    plan = result.get("answer_plan", fallback["answer_plan"])
    if not isinstance(plan, list):
        plan = fallback["answer_plan"]
    answer_plan = [str(item).strip() for item in plan if str(item).strip()][:8]

    final_prompt = _build_final_prompt(state, answer_plan)
    trace_log = _build_visible_trace_log({**state, "answer_plan": answer_plan})
    return {
        **state,
        "answer_plan": answer_plan,
        "final_prompt": final_prompt,
        "visible_trace_log": trace_log,
        "trace_log": trace_log,
    }


def _build_final_prompt(state: PipelineState, answer_plan: list[str]) -> list[dict[str, str]]:
    personalized = state.get("personalized", {})
    prefs = personalized.get("preferences", {}) if isinstance(personalized, dict) else {}
    if not isinstance(prefs, dict):
        prefs = {}
    relevant_prefs = personalized.get("relevant_preferences", {}) if isinstance(personalized, dict) else {}
    if not isinstance(relevant_prefs, dict):
        relevant_prefs = {}
    name = personalized.get("name", "User") if isinstance(personalized, dict) else "User"
    tone = relevant_prefs.get("tone") or prefs.get("tone", "neutral")
    response_length = relevant_prefs.get("response_length") or prefs.get("response_length", "standard")
    interests = ", ".join(personalized.get("interests", [])) if isinstance(personalized, dict) else ""
    dislikes = ", ".join(personalized.get("dislikes", [])) if isinstance(personalized, dict) else ""

    blocks: list[str] = [
        "[ROLE]",
        f"You are a knowledgeable, thoughtful assistant talking to {name}.",
        f"Tone: {tone}. Response style: {response_length}.",
    ]
    if interests:
        blocks.append(f"User interests: {interests}.")
    if dislikes:
        blocks.append(f"Avoid: {dislikes}.")
    if state.get("personalization_summary"):
        blocks.append(f"Personalization note: {state['personalization_summary']}")

    context = _build_context_block(state)
    if context:
        blocks.append(context)

    if state.get("web_sources"):
        blocks.append(
            "[WEBSEARCH SOURCES]\n"
            f"Today is {date.today().isoformat()}.\n"
            "Use these sources only when relevant. Cite source URLs for web-derived claims.\n"
            f"Web summary: {state.get('web_summary', '')}\n\n"
            f"{state.get('websearch_results', '')}"
        )
    elif state.get("websearch_enabled"):
        blocks.append(
            "[WEBSEARCH SOURCES]\n"
            "Websearch was enabled, but no reliable search results were available. "
            "Say when current web evidence is unavailable instead of inventing it."
        )

    if state.get("selected_output_schema"):
        blocks.append(
            "[OUTPUT FORMAT]\n"
            "Structure your response using this schema:\n"
            f"{state['selected_output_schema']}"
        )

    if answer_plan:
        blocks.append(
            "[VISIBLE ACTION PLAN]\n"
            + "\n".join(f"{i}. {step}" for i, step in enumerate(answer_plan, 1))
        )
    blocks.append(_build_adaptive_instructions(state.get("complexity", "moderate")))
    blocks.append(
        "[ANSWER RULES]\n"
        "Answer directly after considering the visible action plan. Do not output XML trace tags. "
        "Do not expose hidden reasoning. If web sources are present, ground current claims in them "
        "and cite URLs inline. Do not write visible plan in final response"
    )

    user_block = (
        "[USER QUESTION - give this the most attention]\n"
        f"{state['user_input']}"
    )
    return [
        {"role": "system", "content": "\n\n".join(blocks)},
        {"role": "user", "content": user_block},
    ]


def _build_context_block(state: PipelineState) -> str:
    if state.get("memory_used") and state.get("long_term_scored"):
        lines = ["[RELEVANT MEMORY CONTEXT - ordered by relevance score]"]
        for index, chunk in enumerate(state.get("long_term_scored", [])[:5], 1):
            score = float(chunk.get("score", 0) or 0)
            content = str(chunk.get("content", "")).strip()
            lines.append(f"[{index}] score={score:.2f}: {content}")
        return "\n".join(lines)

    lines: list[str] = []
    short_term = state.get("short_term", [])
    lru_cached = state.get("lru_cached", [])
    if short_term:
        lines.append("[SHORT-TERM MEMORY CONTEXT]")
        for msg in short_term[-3:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {str(msg.get('content', ''))[:400]}")
    if lru_cached:
        lines.append("[LRU CACHED CONTEXT]")
        for msg in lru_cached[-3:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {str(msg.get('content', ''))[:400]}")
    if state.get("context_hint"):
        lines.append("[CONTEXT HINT]")
        lines.append(str(state["context_hint"])[:700])
    return "\n".join(lines)


async def response_agent(state: PipelineState) -> PipelineState:
    """LLM sub-agent that produces the final non-streaming answer."""
    from app.connect_models import get_chat_model

    try:
        model = get_chat_model(state["model_id"])
        response = await model.ainvoke(_to_lc_messages(state.get("final_prompt", [])))
        raw_reply = _extract_text_content(response.content)
        clean_reply = _strip_trace_tags(raw_reply)
        trace_log = state.get("visible_trace_log") or _build_visible_trace_log(state)
        state = _append_log(state, "response_agent", "ok", {"response_chars": len(clean_reply)})
        return {
            **state,
            "raw_response": raw_reply,
            "response": clean_reply,
            "trace_log": trace_log,
            "visible_trace_log": trace_log,
            "error": None,
        }
    except Exception as exc:
        error_msg = f"Error communicating with model: {exc}"
        logger.exception(error_msg)
        state = _append_error(state, f"response_agent: {exc}")
        return {
            **state,
            "raw_response": "",
            "response": "",
            "trace_log": state.get("visible_trace_log", ""),
            "error": error_msg,
        }


async def stream_invoke_model(state: PipelineState):
    """Stream the final response without asking the model for trace tags."""
    from app.connect_models import get_chat_model

    raw_chunks: list[str] = []
    try:
        model = get_chat_model(state["model_id"])
        async for chunk in model.astream(_to_lc_messages(state.get("final_prompt", []))):
            content = chunk.content if hasattr(chunk, "content") else chunk
            text = _extract_text_content(content)
            if not text:
                continue
            raw_chunks.append(text)
            yield text

        raw_reply = "".join(raw_chunks)
        clean_reply = _strip_trace_tags(raw_reply)
        trace_log = state.get("visible_trace_log") or _build_visible_trace_log(state)
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
            trace_log=state.get("visible_trace_log", ""),
            response="",
            raw_response="",
            error=error_msg,
        )


async def post_process_agent(state: PipelineState) -> PipelineState:
    """LLM-backed post-processing summary plus deterministic storage writes."""
    response = state.get("response", "")
    if not response:
        return state

    fallback = {
        "post_process_summary": "Save user and assistant messages, then update chat memory.",
    }
    _, state = await _agent_json_call(
        state,
        "post_process_agent",
        (
            "Summarize the deterministic post-processing work in one sentence. "
            "Return {'post_process_summary': '...'}."
        ),
        {
            "chat_id": state.get("chat_id"),
            "schema_category": state.get("schema_category"),
            "user_chars": len(state.get("user_input", "")),
            "response_chars": len(response),
        },
        fallback,
    )

    chat_name = state["chat_name"]
    chat_id = state["chat_id"]
    user_input = state["user_input"]
    schema_category = state.get("schema_category")

    user_tokens = mem.estimate_tokens(user_input)
    reply_tokens = mem.estimate_tokens(response)

    db.add_message(chat_id, "user", user_input, token_count=user_tokens, schema_used=schema_category)
    db.add_message(chat_id, "assistant", response, token_count=reply_tokens, schema_used=schema_category)

    try:
        await mem.process_message(chat_name, chat_id, "user", user_input)
        await mem.process_message(chat_name, chat_id, "assistant", response)
    except Exception as exc:
        logger.exception("Memory processing failed during post-process")
        state = _append_error(state, f"post_process_agent memory: {exc}")

    # Periodic personalized memory update (every 3 assistant responses)
    try:
        confirmations = await asyncio.to_thread(
            mem.maybe_update_personalized_memory, chat_id,
        )
        if confirmations:
            state = _append_log(
                state,
                "post_process_agent",
                "personalized_memory_updated",
                {"confirmations": confirmations},
            )
    except Exception as exc:
        logger.exception("Personalized memory periodic update failed")
        state = _append_error(state, f"post_process_agent personalized_memory: {exc}")

    return _append_log(
        state,
        "post_process_agent",
        "ok",
        {"user_tokens": user_tokens, "reply_tokens": reply_tokens},
    )


async def run_pipeline_until_prompt(state: PipelineState) -> PipelineState:
    """Run all pre-response sub-agents before streaming starts (non-streaming).

    Uses the orchestrator agent (1 LLM call) followed by conditional
    memory / websearch and prompt assembly.
    """
    state = await orchestrator_agent(state)
    needs_memory = state.get("needs_memory", False)
    needs_websearch = state.get("needs_websearch", False)

    if needs_memory:
        state = await memory_agent(state)
    if needs_websearch:
        state = await websearch_agent(state)

    state = await prompt_assembly_agent(state)
    return state


class _TraceChunk:
    """Internal sentinel used by run_pipeline_streaming."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


async def run_pipeline_streaming(state: PipelineState):
    """Streaming variant of the pre-response pipeline.

    Yields ``_TraceChunk`` objects as each stage completes so the UI can
    stream trace-log lines into a Collapsible widget in real time.
    After all pre-flight work is done, yields the updated *state* dict as
    the final item.
    """

    # ── 1. Orchestrator agent ─────────────────────────────────────
    state = await orchestrator_agent(state)

    complexity = state.get("complexity", "moderate")
    yield _TraceChunk(f"📋 **Response type:** {complexity}\n")

    category = state.get("schema_category", "unknown")
    depth = state.get("schema_depth", "standard")
    yield _TraceChunk(f"📐 **Output schema:** {category} / {depth}\n")

    p_summary = state.get("personalization_summary", "")
    if p_summary:
        yield _TraceChunk(f"👤 **Personalization:** {p_summary}\n")

    needs_memory = state.get("needs_memory", False)
    needs_websearch = state.get("needs_websearch", False)

    yield _TraceChunk(
        f"🧠 **Memory retrieval:** {'required' if needs_memory else 'not required'}\n"
    )
    yield _TraceChunk(
        f"🌐 **Web search:** {'required' if needs_websearch else 'not required'}\n"
    )

    # ── 2. Conditional sub-agents (concurrent if both needed) ─────
    if needs_memory and needs_websearch:
        yield _TraceChunk("\n🔄 Searching web for sources & recalling from old chats...\n")
        mem_state, web_state = await asyncio.gather(
            memory_agent(state),
            websearch_agent(state),
        )
        state = {**state, **mem_state, **web_state}
    elif needs_memory:
        yield _TraceChunk("\n🔄 Recalling from old chats...\n")
        state = await memory_agent(state)
    elif needs_websearch:
        yield _TraceChunk("\n🔄 Searching web for sources...\n")
        state = await websearch_agent(state)

    # Emit search/memory results into trace if applicable
    if needs_websearch and state.get("search_queries"):
        yield _TraceChunk("\n**Search queries:**\n")
        for q in state.get("search_queries", []):
            yield _TraceChunk(f"- {q}\n")
    if needs_websearch and state.get("web_sources"):
        yield _TraceChunk("\n**Selected sources:**\n")
        for src in state.get("web_sources", [])[:6]:
            yield _TraceChunk(f"- {src.get('title', '')}: {src.get('url', '')}\n")

    # ── 3. Prompt assembly ────────────────────────────────────────
    state = await prompt_assembly_agent(state)

    if state.get("answer_plan"):
        yield _TraceChunk("\n**Action plan:**\n")
        for i, step in enumerate(state.get("answer_plan", []), 1):
            yield _TraceChunk(f"{i}. {step}\n")

    # ── 4. Signal that LLM generation is about to start ───────────
    yield _TraceChunk("\n⏳ Generating...\n")

    # Final item: the fully prepared state
    yield state


async def load_memory_context(chat_name: str, chat_id: str) -> dict[str, Any]:
    """Pre-load short-term, LRU, and personalized memory for the pipeline."""
    short_term = await asyncio.to_thread(mem.load_short_term, chat_name)
    lru_cached = await asyncio.to_thread(mem.load_lru, chat_name)
    personalized = await asyncio.to_thread(mem.load_personalized_memory)
    return {
        "short_term": short_term,
        "lru_cached": lru_cached,
        "personalized": personalized,
    }


def _strip_trace_tags(raw_text: str) -> str:
    return re.sub(
        r"<TRACE>\s*.*?\s*</TRACE>",
        "",
        raw_text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _build_visible_trace_log(state: PipelineState) -> str:
    lines: list[str] = ["### Chat Trace Logs", ""]
    if state.get("answer_plan"):
        lines.append("**Action plan**")
        for index, step in enumerate(state.get("answer_plan", []), 1):
            lines.append(f"{index}. {step}")
        lines.append("")

    lines.append("**Pipeline summary**")
    lines.append(f"- Schema: {state.get('schema_category', 'unknown')} / {state.get('schema_depth', 'standard')}")
    lines.append(f"- Complexity: {state.get('complexity', 'moderate')}")
    lines.append(f"- Memory path: {state.get('memory_path', 'default')}")
    lines.append(f"- Websearch: {'enabled' if state.get('websearch_enabled') else 'disabled'}")

    if state.get("search_queries"):
        lines.append("")
        lines.append("**Search queries**")
        for query in state.get("search_queries", []):
            lines.append(f"- {query}")

    if state.get("web_sources"):
        lines.append("")
        lines.append("**Selected sources**")
        for source in state.get("web_sources", [])[:8]:
            title = source.get("title", "(no title)")
            url = source.get("url", "")
            lines.append(f"- {title}: {url}")

    if state.get("agent_errors"):
        lines.append("")
        lines.append("**Warnings**")
        for error in state.get("agent_errors", [])[:5]:
            lines.append(f"- {error}")

    return "\n".join(lines).strip()


_ADAPTIVE_INSTRUCTIONS: dict[str, str] = {
    "simple": (
        "[ADAPTIVE RESPONSE]\n"
        "The question is simple. Provide a concise, focused answer."
    ),
    "moderate": (
        "[ADAPTIVE RESPONSE]\n"
        "The question is of moderate complexity. Provide a well-structured answer with enough detail."
    ),
    "complex": (
        "[ADAPTIVE RESPONSE]\n"
        "The question is complex or asks for thorough explanation. Provide a comprehensive response."
    ),
}


def _build_adaptive_instructions(complexity: str) -> str:
    return _ADAPTIVE_INSTRUCTIONS.get(complexity, _ADAPTIVE_INSTRUCTIONS["moderate"])

classify_question = orchestrator_agent
classify_schema = orchestrator_agent
retrieve_long_term_memory = memory_agent
score_and_filter_chunks = memory_agent
apply_personalization = orchestrator_agent
assemble_prompt = prompt_assembly_agent
invoke_model = response_agent
post_process = post_process_agent
generate_cot_plan = websearch_agent
second_web_search = websearch_agent


# ══════════════════════════════════════════════
#  Ollama model preloading / unloading
# ══════════════════════════════════════════════
async def preload_ollama_model(model_name: str) -> bool:
    """Send an empty prompt to Ollama to load the model into VRAM.

    Uses ``keep_alive=-1`` which tells Ollama to keep the model loaded
    indefinitely until explicitly unloaded or until Ollama is stopped.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": -1},
            )
            return resp.status_code == 200
    except Exception:
        logger.exception("Failed to preload Ollama model %s", model_name)
        return False


async def unload_ollama_model(model_name: str) -> bool:
    """Unload a model from VRAM by setting ``keep_alive`` to 0."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": 0},
            )
            return resp.status_code == 200
    except Exception:
        logger.exception("Failed to unload Ollama model %s", model_name)
        return False


# ══════════════════════════════════════════════
#  Cloud models optimized pipeline (Changes 5)
# ══════════════════════════════════════════════

async def _cloud_agent_call(
    state: PipelineState,
    agent_name: str,
    system_prompt: str,
    payload_str: str,
) -> tuple[str, PipelineState]:
    """Call a cloud LLM sub-agent without JSON repair retries."""
    from app.connect_models import get_chat_model

    model = get_chat_model(state["model_id"])
    messages = [
        SystemMessage(
            content=(
                f"You are the {agent_name} sub-agent in OptiChat. "
                "Output your response clearly in the requested XML/text format. "
                "Do not include hidden reasoning or chain-of-thought."
                f"\n\n{system_prompt}"
            )
        ),
        HumanMessage(content=payload_str),
    ]

    try:
        response = await model.ainvoke(messages)
        content = _extract_text_content(response.content)
        return content, _append_log(state, agent_name, "ok", {"raw_chars": len(content)})
    except Exception as exc:
        logger.exception("%s sub-agent failed", agent_name)
        state = _append_error(state, f"{agent_name}: {exc}")
        return "", state


def _parse_xml_tag(text: str, tag: str, default: str = "") -> str:
    """Helper to extract content between XML tags case-insensitively."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else default


async def cloud_orchestrator_agent(state: PipelineState) -> PipelineState:
    """Cloud-specific orchestrator agent combining orchestrator classification,
    personalization analysis, and action plan generation into a single LLM call.
    Also programmatically performs lexical and semantic memory search without an LLM call.
    """
    user_input = state["user_input"]
    user_lower = user_input.lower()

    # Deterministic pre-computations
    context_hint, needs_long_term = _memory_overlap_hint(
        user_input,
        state.get("short_term", []),
        state.get("lru_cached", []),
    )
    heuristic_category = _heuristic_category(user_lower)
    heuristic_complexity = _heuristic_complexity(user_lower)
    depth_map = {"simple": "quick", "moderate": "standard", "complex": "detailed"}
    heuristic_depth = depth_map.get(heuristic_complexity, "standard")

    try:
        cfg = db.load_config()
        memory_enabled = cfg.get("memory_enabled", True)
    except Exception:
        memory_enabled = True

    personalized = mem.load_personalized_memory() if memory_enabled else {}

    system_prompt = (
        "You are the combined Orchestrator and Prompt Assembly Agent in OptiChat.\n"
        "Perform all of these tasks and return the result using XML tags:\n"
        "1. Determine complexity (simple/moderate/complex) and put it inside `<complexity>...</complexity>`.\n"
        "2. Select schema category (one of: factual_definition, how_to_procedural, comparison, creative_writing, historical_analytical, scientific_technical, opinion_debate, personal_advice, code_explanation, open_ended_conversational) inside `<schema_category>...</schema_category>`.\n"
        "3. Select schema depth (quick/standard/detailed) inside `<schema_depth>...</schema_depth>`.\n"
        "4. Write a personalization summary inside `<personalization_summary>...</personalization_summary>`.\n"
        "5. Output relevant key-value preferences as a JSON string inside `<relevant_preferences>...</relevant_preferences>`.\n"
        "6. Create a complete action plan inside `<plan>...</plan>` with one step per line (concise, visible steps to answer the query).\n"
        "Do not output any reasoning or markdown outside the tags."
    )

    payload = {
        "user_input": user_input,
        "short_term_recent": state.get("short_term", [])[-5:],
        "lru_cached_recent": state.get("lru_cached", [])[-5:],
        "personalized_memory": personalized,
        "available_schemas": OUTPUT_SCHEMAS,
    }

    content, state = await _cloud_agent_call(
        state,
        "cloud_orchestrator_agent",
        system_prompt,
        _compact_json(payload),
    )

    # Parsing tags
    complexity = _parse_xml_tag(content, "complexity", heuristic_complexity).lower()
    if complexity not in {"simple", "moderate", "complex"}:
        complexity = heuristic_complexity

    category = _parse_xml_tag(content, "schema_category", heuristic_category)
    if category not in OUTPUT_SCHEMAS:
        category = heuristic_category

    depth = _parse_xml_tag(content, "schema_depth", heuristic_depth)
    if depth not in OUTPUT_SCHEMAS.get(category, {}):
        depth = heuristic_depth

    p_summary = _parse_xml_tag(content, "personalization_summary")
    
    prefs_text = _parse_xml_tag(content, "relevant_preferences")
    try:
        rel_prefs = json.loads(prefs_text) if prefs_text else {}
    except Exception:
        rel_prefs = {}

    if not isinstance(rel_prefs, dict):
        rel_prefs = {}
    if isinstance(personalized, dict):
        personalized = {**personalized, "relevant_preferences": rel_prefs}

    # Action plan steps
    plan_text = _parse_xml_tag(content, "plan")
    answer_plan = [line.strip() for line in plan_text.splitlines() if line.strip()]
    if not answer_plan:
        answer_plan = [
            "Identify the user's main request.",
            "Use selected memory context where relevant.",
            "Answer using the selected response schema.",
        ]

    # ── Memory Search (Lexical + Semantic) ──
    matched_memories = []
    
    query_words = {w for w in re.findall(r"\w+", user_input.lower()) if len(w) > 2}
    stopwords = {"the", "and", "a", "an", "of", "to", "in", "is", "that", "it", "was", "for", "on", "are", "as", "with", "his", "they", "i", "you", "he", "she"}
    query_words = query_words - stopwords

    # Lexical search in short-term memory
    for msg in state.get("short_term", []):
        msg_content = msg.get("content", "")
        if msg_content:
            msg_words = {w for w in re.findall(r"\w+", msg_content.lower()) if len(w) > 2} - stopwords
            if query_words & msg_words:
                matched_memories.append({
                    "content": msg_content,
                    "role": msg.get("role", "user"),
                    "score": 1.0,
                    "source": "short_term",
                    "timestamp": msg.get("timestamp", ""),
                })

    # Lexical search in LRU memory
    for msg in state.get("lru_cached", []):
        msg_content = msg.get("content", "")
        if msg_content:
            msg_words = {w for w in re.findall(r"\w+", msg_content.lower()) if len(w) > 2} - stopwords
            if query_words & msg_words:
                if not any(m["content"] == msg_content for m in matched_memories):
                    matched_memories.append({
                        "content": msg_content,
                        "role": msg.get("role", "user"),
                        "score": 1.0,
                        "source": "lru_cached",
                        "timestamp": msg.get("timestamp", ""),
                    })

    # Semantic search in long-term memory vectordb
    raw_chunks = []
    try:
        raw_chunks = mem.retrieve_from_long_term(state["chat_id"], user_input, top_k=5)
    except Exception as exc:
        logger.exception("Long-term memory retrieval failed")
        state = _append_error(state, f"cloud_orchestrator retrieval: {exc}")

    candidate_chunks = [
        c for c in raw_chunks if float(c.get("score", 0) or 0) >= RELEVANCE_THRESHOLD
    ]
    for chunk in candidate_chunks:
        if not any(m["content"] == chunk["content"] for m in matched_memories):
            matched_memories.append({
                "content": chunk["content"],
                "role": chunk.get("role", "user"),
                "score": float(chunk.get("score", 0) or 0),
                "source": "long_term",
                "timestamp": chunk.get("timestamp", ""),
            })

    memory_used = len(matched_memories) > 0

    # If found return the context and add it to action plan as some more context.
    if memory_used:
        for match in matched_memories:
            short_content = match["content"].replace('\n', ' ')
            if len(short_content) > 120:
                short_content = short_content[:120] + "..."
            answer_plan.append(f"Context from memory [{match['source']}]: {short_content}")

    temp_state = {
        **state,
        "question_type": category,
        "complexity": complexity,
        "language": "English",
        "needs_long_term": memory_used,
        "context_hint": context_hint,
        "memory_path": "long_term" if memory_used else "short_lru",
        "schema_category": category,
        "schema_depth": depth,
        "selected_output_schema": OUTPUT_SCHEMAS[category][depth],
        "personalized": personalized,
        "personalization_summary": p_summary,
        "needs_memory": memory_used,
        "needs_websearch": False,
        "long_term_raw": raw_chunks,
        "long_term_scored": matched_memories,
        "memory_used": memory_used,
        "answer_plan": answer_plan,
    }

    final_prompt = _build_final_prompt(temp_state, answer_plan)
    trace_log = _build_visible_trace_log({**temp_state, "answer_plan": answer_plan})

    return {
        **temp_state,
        "final_prompt": final_prompt,
        "visible_trace_log": trace_log,
        "trace_log": trace_log,
    }


async def cloud_memory_agent(state: PipelineState) -> PipelineState:
    """Legacy/compatibility cloud memory agent function."""
    return state


async def cloud_prompt_assembly_agent(state: PipelineState) -> PipelineState:
    """Legacy/compatibility cloud prompt assembly agent function."""
    return state


async def run_cloud_pipeline_streaming(state: PipelineState):
    """Streaming variant of the pre-response cloud pipeline under Changes 5 optimization."""
    # 1. Cloud Orchestrator Agent (runs combined LLM call & memory search)
    state = await cloud_orchestrator_agent(state)

    complexity = state.get("complexity", "moderate")
    yield _TraceChunk(f"📋 **Response type:** {complexity}\n")

    category = state.get("schema_category", "unknown")
    depth = state.get("schema_depth", "standard")
    yield _TraceChunk(f"📐 **Output schema:** {category} / {depth}\n")

    p_summary = state.get("personalization_summary", "")
    if p_summary:
        yield _TraceChunk(f"👤 **Personalization:** {p_summary}\n")

    memory_used = state.get("memory_used", False)
    long_term_scored = state.get("long_term_scored", [])
    yield _TraceChunk(
        f"🧠 **Memory retrieval:** {'recalled' if memory_used else 'no matches found'} ({len(long_term_scored)} matches)\n"
    )

    answer_plan = state.get("answer_plan", [])
    if answer_plan:
        yield _TraceChunk("\n**Action plan:**\n")
        for i, step in enumerate(answer_plan, 1):
            yield _TraceChunk(f"{i}. {step}\n")

    # 2. Signal that generation is starting
    yield _TraceChunk("\n⏳ Generating...\n")
    yield state


async def run_cloud_pipeline_until_prompt(state: PipelineState) -> PipelineState:
    """Non-streaming variant of the pre-response cloud pipeline under Changes 5 optimization."""
    state = await cloud_orchestrator_agent(state)
    return state
