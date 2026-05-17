"""OptiChat – LangGraph Prompt Construction Pipeline (Phase 4).

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

Public API
──────────
    run_pipeline(user_input, chat_name, chat_id, model_id) -> str
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from app.pipeline_functions import (
    PipelineState,
    apply_personalization,
    assemble_prompt,
    classify_question,
    classify_schema,
    invoke_model,
    load_memory_context,
    post_process,
    retrieve_long_term_memory,
    route_after_classify,
    score_and_filter_chunks,
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

    # Scoring → personalization → assembly → invoke → post-process → END
    graph.add_edge("score_and_filter", "apply_personalization")
    graph.add_edge("apply_personalization", "assemble_prompt")
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
) -> dict[str, Any]:
    """Execute the full prompt construction pipeline.

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
