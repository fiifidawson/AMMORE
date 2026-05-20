from typing import Any, Callable, List

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import HeadAndTailChatCompletionContext

from .config import config
from .mmore_client import retrieve
from .web_search import search_web

_search_state = {"failed_streak": 0}
_MAX_FAILED_STREAK = 3


def reset_search_state() -> None:
    _search_state["failed_streak"] = 0


def _ctx():
    return HeadAndTailChatCompletionContext(
        head_size=config.context_head, tail_size=config.context_tail
    )


def _track(result: str) -> str:
    bad = result.startswith("ERROR") or result == "No results found."
    if bad:
        _search_state["failed_streak"] += 1
    else:
        _search_state["failed_streak"] = 0
    if _search_state["failed_streak"] >= _MAX_FAILED_STREAK:
        return (
            "Stop searching: the last few attempts returned nothing. "
            "Report what you already have and let the Critic decide."
        )
    return result


def search_documents(
    query: str, max_matches: int = 5, min_similarity: float = -1.0
) -> str:
    """Search the indexed paper corpus.

    One topic per query. If results are empty or off-topic, retry with a
    higher max_matches, lower min_similarity, or a reworded query before
    giving up.

    max_matches: chunks to return (default 5; bump to 10-15 if sparse).
    min_similarity: -1.0 keeps everything, 0.3-0.5 drops weak matches.
    Returns chunks joined by '---', or "No results found." if nothing hit.
    """
    if not query.strip():
        return "ERROR: 'query' is required. Call again with a specific query string."
    return _track(
        retrieve(query, max_matches=max_matches, min_similarity=min_similarity)
    )


def search_web_tool(query: str) -> str:
    """Tavily web search. Use when the corpus doesn't cover the topic."""
    if not query.strip():
        return "ERROR: 'query' is required. Call again with a specific query string."
    return _track(search_web(query))


PLANNER_PROMPT = (
    "You are a research planner. Given a question, break it down into "
    "3-5 specific search queries to run against a document corpus.\n\n"
    "Output ONLY a numbered list of queries, nothing else.\n"
    "Example:\n"
    "1. What methods are used for X?\n"
    "2. What are the results reported for Y?\n"
    "3. What limitations are mentioned?\n"
)


def create_planner(model_client):
    return AssistantAgent(
        name="Planner",
        model_client=model_client,
        description="Breaks the question into sub-queries.",
        system_message=PLANNER_PROMPT,
        model_context=_ctx(),
    )


def create_agents(model_client):
    tools: List[Callable[..., Any]] = [search_documents]
    retriever_msg = (
        "You are a retrieval agent. Use search_documents to find chunks for "
        "each query.\n"
        "If a call returns nothing or off-topic chunks, don't give up: retry "
        "the same need with adjusted parameters (lower min_similarity, higher "
        "max_matches, or a reworded query) before moving on.\n"
        "Only report what the tool returns, don't invent content."
    )
    if config.websearch_enabled:
        tools.append(search_web_tool)
        retriever_msg = (
            "You are a retrieval agent with two tools:\n"
            "- search_documents: indexed corpus (tune max_matches / min_similarity)\n"
            "- search_web_tool: open web via Tavily\n\n"
            "Always pass a non-empty query string to either tool.\n"
            "For each query: call search_documents first. If it returns nothing "
            "or off-topic chunks, retry ONCE with adjusted parameters (higher "
            "max_matches, lower min_similarity, or a reworded query). If the "
            "second attempt still fails, you MUST call search_web_tool with the "
            "same query before moving on -- do not give up on the corpus only. "
            "Report everything the tools return. Don't invent content."
        )

    retriever = AssistantAgent(
        name="Retriever",
        model_client=model_client,
        description="Searches the document corpus and optionally the web.",
        system_message=retriever_msg,
        tools=list(tools),
        reflect_on_tool_use=False,
        model_context=_ctx(),
    )

    critic = AssistantAgent(
        name="Critic",
        model_client=model_client,
        description="Checks if we have enough information to write the answer.",
        system_message=(
            "Judge whether the retrieved chunks actually address the ORIGINAL "
            "question -- not whether chunks merely exist. Chunks that are on a "
            "different topic count as no coverage.\n\n"
            "Output EXACTLY ONE verdict on the FIRST line:\n"
            "- COVERAGE_OK only if the chunks genuinely answer the question.\n"
            "- NEEDS_MORE if the chunks are off-topic, unrelated, missing key "
            "aspects, or too sparse.\n\n"
            "Never write both verdicts in the same reply.\n"
            "If NEEDS_MORE: list 1-3 follow-up queries. If the corpus chunks are "
            "clearly off-topic for the question, explicitly tell the Retriever to "
            "use search_web_tool for these queries."
        ),
        model_context=_ctx(),
    )

    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="Writes the final synthesis.",
        system_message=(
            "You write a literature review STRICTLY from the retrieved chunks above. "
            "You have no other knowledge.\n\n"
            "Before writing, check whether the chunks actually contain information "
            "that answers the original question. If they do NOT (off-topic, empty, "
            "or unrelated to the question), do not write a review. Instead reply "
            "exactly:\n"
            "  The retrieved sources do not contain information answering this "
            "question. The corpus does not appear to cover this topic.\n"
            "Then end with TERMINATE. Never fill the gap with general knowledge.\n\n"
            "If the chunks DO answer the question, write:\n"
            "## Summary\n## Key Findings\n## Gaps & Limitations\n## Sources\n\n"
            "Every claim must be traceable to a specific chunk. Cite chunks with "
            "their bracketed label exactly as shown, e.g. [Chunk 3 | title] or "
            "[Web 2 | Title | URL]. Do not state anything that is not in a chunk, "
            "and do not cite a source that is not in the chunks above.\n\n"
            "End with: TERMINATE"
        ),
        model_context=_ctx(),
    )

    return retriever, critic, writer
