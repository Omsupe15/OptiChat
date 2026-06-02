"""OptiChat – LangGraph Prompt Construction Pipeline (Phase 4 + 5).

Builds a LangGraph ``StateGraph`` that wires together the nodes defined
in ``pipeline_functions.py``.  The graph follows the flow from
design.md §3.1:

    User Input
      → classify_question  ─┐  (parallel)
      → classify_schema    ─┘
      → (conditional) retrieve_long_term
      → score_and_filter
      → apply_personalization
      → assemble_prompt
      → invoke_model
      → post_process
      → END

Phase 5 addition
────────────────
Both ``run_pipeline`` and ``stream_pipeline`` now accept a
``websearch_enabled`` boolean flag.  When True, the classifier node
performs a DuckDuckGo search and injects the results into the final
prompt via the ``[WEBSEARCH]`` block.

Public API
──────────
    run_pipeline(user_input, chat_name, chat_id, model_id,
                 websearch_enabled=False) -> dict
    stream_pipeline(user_input, chat_name, chat_id, model_id,
                    websearch_enabled=False) -> AsyncGenerator
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from app.pipeline_functions import (
    PipelineState,
    StreamDone,
    apply_personalization,
    assemble_prompt,
    classify_question,
    classify_schema,
    generate_cot_plan,
    invoke_model,
    load_memory_context,
    post_process,
    retrieve_long_term_memory,
    route_after_classify,
    route_after_personalization,
    run_pipeline_until_prompt,
    score_and_filter_chunks,
    second_web_search,
    stream_invoke_model,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  Build the LangGraph
# ══════════════════════════════════════════════

def _build_graph() -> StateGraph:
    """Construct and compile the prompt construction pipeline graph."""

    graph = StateGraph(PipelineState)

    # ── Add nodes ─────────────────────────────
    graph.add_node("classify_question", classify_question)
    graph.add_node("classify_schema", classify_schema)
    graph.add_node("retrieve_long_term", retrieve_long_term_memory)
    graph.add_node("score_and_filter", score_and_filter_chunks)
    graph.add_node("apply_personalization", apply_personalization)
    # Phase 5: dual web-search nodes (only visited when websearch is on)
    graph.add_node("generate_cot_plan", generate_cot_plan)
    graph.add_node("second_web_search", second_web_search)
    graph.add_node("assemble_prompt", assemble_prompt)
    graph.add_node("invoke_model", invoke_model)
    graph.add_node("post_process", post_process)

    # ── Entry point ───────────────────────────
    # Both classifiers run from the start (conceptually parallel)
    graph.set_entry_point("classify_question")

    # ── Edges ─────────────────────────────────
    # After question classification, run schema classification
    graph.add_edge("classify_question", "classify_schema")

    # After schema classification, decide if we need long-term retrieval
    graph.add_conditional_edges(
        "classify_schema",
        route_after_classify,
        {
            "retrieve_long_term": "retrieve_long_term",
            "score_and_filter": "score_and_filter",
        },
    )

    # Long-term retrieval feeds into scoring
    graph.add_edge("retrieve_long_term", "score_and_filter")

    # Scoring → personalization
    graph.add_edge("score_and_filter", "apply_personalization")

    # Phase 5: after personalization, conditionally run dual web-search or go
    # straight to assembly.  Both DDGS passes complete before invoke_model.
    graph.add_conditional_edges(
        "apply_personalization",
        route_after_personalization,
        {
            "generate_cot_plan": "generate_cot_plan",
            "assemble_prompt": "assemble_prompt",
        },
    )
    graph.add_edge("generate_cot_plan", "second_web_search")
    graph.add_edge("second_web_search", "assemble_prompt")

    # Assembly → invoke → post-process → END
    graph.add_edge("assemble_prompt", "invoke_model")
    graph.add_edge("invoke_model", "post_process")
    graph.add_edge("post_process", END)

    return graph


# Compile once at module level
_compiled_graph = _build_graph().compile()


# ══════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════

async def run_pipeline(
    user_input: str,
    chat_name: str,
    chat_id: str,
    model_id: str,
    *,
    websearch_enabled: bool = False,
) -> dict[str, Any]:
    """Execute the full prompt construction pipeline.

    Parameters
    ----------
    websearch_enabled:
        When True the classifier node will query DuckDuckGo for the top-2
        results and inject them into the final prompt (Phase 5 feature).

    Returns the final ``PipelineState`` dict.  The caller should read
    ``state["response"]`` for the assistant reply and ``state["error"]``
    for any error message.
    """
    # Pre-load memory context before entering the graph
    mem_context = await load_memory_context(chat_name, chat_id)

    initial_state: PipelineState = {
        "user_input": user_input,
        "chat_name": chat_name,
        "chat_id": chat_id,
        "model_id": model_id,
        # Phase 5: websearch flag
        "websearch_enabled": websearch_enabled,
        "websearch_results": "",
        # Pre-loaded memory
        "short_term": mem_context["short_term"],
        "lru_cached": mem_context["lru_cached"],
        "personalized": mem_context["personalized"],
        # Defaults (filled by nodes)
        "question_type": "",
        "complexity": "",
        "language": "English",
        "needs_long_term": True,
        "context_hint": "",
        "schema_category": "",
        "schema_depth": "standard",
        "selected_output_schema": "",
        "long_term_raw": [],
        "long_term_scored": [],
        "memory_used": False,
        "final_prompt": [],
        "response": "",
        "raw_response": "",
        "trace_log": "",
        "error": None,
    }

    # Run the graph
    result = await _compiled_graph.ainvoke(initial_state)

    return result


async def stream_pipeline(
    user_input: str,
    chat_name: str,
    chat_id: str,
    model_id: str,
    *,
    websearch_enabled: bool = False,
):
    """Streaming variant of the prompt construction pipeline.

    Runs all pipeline stages up to ``assemble_prompt``, then streams
    the LLM response token-by-token.  Post-processing (DB storage +
    memory updates) is performed after the stream completes.

    Parameters
    ----------
    websearch_enabled:
        When True the classifier node will query DuckDuckGo for the top-2
        results and inject them into the final prompt (Phase 5 feature).

    Yields
    ------
    str
        Individual token chunks from the LLM (the ``<TRACE>`` block is
        silently consumed and never yielded).
    StreamDone
        A single :class:`~app.pipeline_functions.StreamDone` sentinel as
        the **very last** item.  The caller must handle this to know
        streaming has finished and to access ``trace_log`` / ``response``.
    """
    # ── 1. Pre-load memory ───────────────────────────────────────────
    mem_context = await load_memory_context(chat_name, chat_id)

    state: PipelineState = {
        "user_input": user_input,
        "chat_name": chat_name,
        "chat_id": chat_id,
        "model_id": model_id,
        # Phase 5: websearch flag
        "websearch_enabled": websearch_enabled,
        "websearch_results": "",
        # Pre-loaded memory
        "short_term": mem_context["short_term"],
        "lru_cached": mem_context["lru_cached"],
        "personalized": mem_context["personalized"],
        # Defaults (filled by nodes)
        "question_type": "",
        "complexity": "",
        "language": "English",
        "needs_long_term": True,
        "context_hint": "",
        "schema_category": "",
        "schema_depth": "standard",
        "selected_output_schema": "",
        "long_term_raw": [],
        "long_term_scored": [],
        "memory_used": False,
        "final_prompt": [],
        "response": "",
        "raw_response": "",
        "trace_log": "",
        "error": None,
    }

    # ── 2. Run pipeline up to assemble_prompt ────────────────────────
    state = await run_pipeline_until_prompt(state)

    # ── 3. Stream LLM tokens ─────────────────────────────────────────
    done: StreamDone | None = None
    async for item in stream_invoke_model(state):
        if isinstance(item, StreamDone):
            done = item
        else:
            yield item  # plain token string – forward to the UI

    # Ensure we always have a done sentinel even if the generator ended
    # unexpectedly (defensive guard).
    if done is None:
        done = StreamDone(
            trace_log="",
            response="",
            raw_response="",
            error="Stream ended without a StreamDone sentinel.",
        )

    # Yield the sentinel so the caller can access trace_log / response.
    yield done

    # ── 4. Post-process (DB + memory) ────────────────────────────────
    if done.error:
        return  # Skip post-processing if streaming failed

    final_state: PipelineState = {
        **state,
        "response": done.response,
        "raw_response": done.raw_response,
        "trace_log": done.trace_log,
        "error": None,
    }
    try:
        await post_process(final_state)
    except Exception:
        logger.exception("post_process failed after streaming")
