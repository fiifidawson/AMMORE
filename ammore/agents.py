from typing import Any, Callable, List

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import BufferedChatCompletionContext

from .config import config
from .mmore_client import retrieve
from .web_search import search_web


def _ctx():
    return BufferedChatCompletionContext(buffer_size=config.context_window)


def search_documents(query: str, max_matches: int = 5) -> str:
    """Search the indexed paper corpus for chunks matching the query.

    Args:
        query: a precise natural-language query, one topic at a time.
        max_matches: number of chunks to return. Bump it up when the first
            call returns too few or off-topic results before trying a new
            query. Drop it down when chunks are noisy.

    Returns:
        Chunks separated by '---', each prefixed with a header containing
        the chunk index, the fileId and the source file path.
    """
    return retrieve(query, max_matches=max_matches)


def search_web_tool(query: str) -> str:
    """Search the open web via Tavily.

    Use this when the indexed corpus clearly doesn't cover the topic
    (off-topic chunks, very recent events, very broad question).
    """
    return search_web(query)


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
        "each query. Only report what the tool returns, don't invent content."
    )
    if config.websearch_enabled:
        tools.append(search_web_tool)
        retriever_msg = (
            "You are a retrieval agent with two tools:\n"
            "- search_documents: indexed corpus\n"
            "- search_web_tool: open web via Tavily\n\n"
            "For each query, always call search_documents first. If the chunks "
            "are off-topic or there are too few results, also call search_web_tool. "
            "Report everything both tools return. Don't invent content."
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
            "You review the retrieved chunks and decide if we have enough coverage to "
            "answer the original question.\n\n"
            "Output EXACTLY ONE verdict on the FIRST line of your reply:\n"
            "- COVERAGE_OK if the chunks contain enough information to answer the question.\n"
            "- NEEDS_MORE if the chunks are off-topic, missing key aspects, or too sparse.\n\n"
            "Never write both verdicts in the same reply.\n"
            "If NEEDS_MORE: on the next lines, suggest 1-3 additional queries that would "
            "fill the gaps. Frame these queries so they help the Retriever find missing info "
            "(use the open web if the corpus clearly doesn't cover the topic)."
        ),
        model_context=_ctx(),
    )

    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="Writes the final synthesis.",
        system_message=(
            "Based on the retrieved chunks, write a structured literature review.\n\n"
            "## Summary\n"
            "## Key Findings\n"
            "## Gaps & Limitations\n"
            "## Sources\n"
            "List ONLY the chunks that actually appear above. Use the bracketed labels "
            "as-is, e.g. [Chunk 3 | abc123] for corpus chunks or [Web 2 | Title | URL] for web results. "
            "Do not add any sources from your training knowledge. If a paper or page is not in the chunks, do not cite it.\n\n"
            "End with: TERMINATE"
        ),
        model_context=_ctx(),
    )

    return retriever, critic, writer
