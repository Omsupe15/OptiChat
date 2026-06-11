"""OptiChat LangGraph prompt pipeline.

The graph is organized as explicit LLM-backed sub-agent stages while keeping
the public API used by the UI stable.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from app.pipeline_functions import (
    PipelineState,
    StreamDone,
    classifier_agent,
    load_memory_context,
    memory_agent,
    personalization_agent,
    post_process_agent,
    prompt_assembly_agent,
    response_agent,
    route_after_personalization,
    run_pipeline_until_prompt,
    schema_agent,
    stream_invoke_model,
    websearch_agent,
)

logger = logging.getLogger(__name__)


def _build_graph():
    """Construct and compile the sub-agent prompt pipeline graph."""
    graph = StateGraph(PipelineState)

    graph.add_node("classifier_agent", classifier_agent)
    graph.add_node("schema_agent", schema_agent)
    graph.add_node("memory_agent", memory_agent)
    graph.add_node("personalization_agent", personalization_agent)
    graph.add_node("websearch_agent", websearch_agent)
    graph.add_node("prompt_assembly_agent", prompt_assembly_agent)
    graph.add_node("response_agent", response_agent)
    graph.add_node("post_process_agent", post_process_agent)

    graph.set_entry_point("classifier_agent")
    graph.add_edge("classifier_agent", "schema_agent")
    graph.add_edge("schema_agent", "memory_agent")
    graph.add_edge("memory_agent", "personalization_agent")
    graph.add_conditional_edges(
        "personalization_agent",
        route_after_personalization,
        {
            "websearch_agent": "websearch_agent",
            "prompt_assembly_agent": "prompt_assembly_agent",
        },
    )
    graph.add_edge("websearch_agent", "prompt_assembly_agent")
    graph.add_edge("prompt_assembly_agent", "response_agent")
    graph.add_edge("response_agent", "post_process_agent")
    graph.add_edge("post_process_agent", END)

    return graph.compile()


_compiled_graph = _build_graph()


async def _initial_state(
    user_input: str,
    chat_name: str,
    chat_id: str,
    model_id: str,
    *,
    websearch_enabled: bool,
) -> PipelineState:
    mem_context = await load_memory_context(chat_name, chat_id)
    return {
        "user_input": user_input,
        "chat_name": chat_name,
        "chat_id": chat_id,
        "model_id": model_id,
        "websearch_enabled": websearch_enabled,
        "websearch_results": "",
        "search_queries": [],
        "web_sources": [],
        "web_summary": "",
        "short_term": mem_context["short_term"],
        "lru_cached": mem_context["lru_cached"],
        "personalized": mem_context["personalized"],
        "personalization_summary": "",
        "question_type": "",
        "complexity": "moderate",
        "language": "English",
        "needs_long_term": True,
        "context_hint": "",
        "memory_path": "long_term",
        "schema_category": "",
        "schema_depth": "standard",
        "selected_output_schema": "",
        "long_term_raw": [],
        "long_term_scored": [],
        "memory_used": False,
        "agent_logs": [],
        "agent_errors": [],
        "answer_plan": [],
        "visible_trace_log": "",
        "final_prompt": [],
        "response": "",
        "raw_response": "",
        "trace_log": "",
        "error": None,
    }


async def run_pipeline(
    user_input: str,
    chat_name: str,
    chat_id: str,
    model_id: str,
    *,
    websearch_enabled: bool = False,
) -> dict[str, Any]:
    """Execute the full sub-agent prompt construction pipeline."""
    state = await _initial_state(
        user_input,
        chat_name,
        chat_id,
        model_id,
        websearch_enabled=websearch_enabled,
    )
    return await _compiled_graph.ainvoke(state)


async def stream_pipeline(
    user_input: str,
    chat_name: str,
    chat_id: str,
    model_id: str,
    *,
    websearch_enabled: bool = False,
):
    """Streaming variant of the sub-agent pipeline.

    All pre-response agents, including websearch when enabled, finish before
    token streaming starts. The final item yielded is ``StreamDone``.
    """
    state = await _initial_state(
        user_input,
        chat_name,
        chat_id,
        model_id,
        websearch_enabled=websearch_enabled,
    )

    state = await run_pipeline_until_prompt(state)

    done: StreamDone | None = None
    async for item in stream_invoke_model(state):
        if isinstance(item, StreamDone):
            done = item
        else:
            yield item

    if done is None:
        done = StreamDone(
            trace_log=state.get("visible_trace_log", ""),
            response="",
            raw_response="",
            error="Stream ended without a StreamDone sentinel.",
        )

    yield done

    if done.error:
        return

    final_state: PipelineState = {
        **state,
        "response": done.response,
        "raw_response": done.raw_response,
        "trace_log": done.trace_log,
        "visible_trace_log": done.trace_log,
        "error": None,
    }
    try:
        await post_process_agent(final_state)
    except Exception:
        logger.exception("post_process_agent failed after streaming")
