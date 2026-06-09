import os
from typing import Annotated, Any, List

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import HeadAndTailChatCompletionContext
from autogen_core.tools import FunctionTool

from .config import config

# from .critic import CriticAgent  # diagnostic: temporarily reverted to AssistantAgent Critic
from .mmore_client import retrieve
from .web_search import search_web

_search_state = {"failed_streak": 0, "web_allowed": False}
_MAX_FAILED_STREAK = 3


def reset_search_state() -> None:
    _search_state["failed_streak"] = 0
    _search_state["web_allowed"] = False


def allow_web_search() -> None:
    _search_state["web_allowed"] = True


def _ctx():
    return HeadAndTailChatCompletionContext(
        head_size=config.loop.context_head, tail_size=config.loop.context_tail
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
        return _track(
            "ERROR: 'query' is required. Call again with a specific query string."
        )
    return _track(
        retrieve(query, max_matches=max_matches, min_similarity=min_similarity)
    )


def search_web_tool(query: str) -> str:
    """Tavily web search. Use when the corpus doesn't cover the topic."""
    if not _search_state["web_allowed"]:
        return _track(
            "ERROR: use search_web_tool only after the Critic asks for web fallback."
        )
    if not query.strip():
        return _track(
            "ERROR: 'query' is required. Call again with a specific query string."
        )
    return _track(search_web(query))


def _search_documents_tool(
    query: Annotated[str, "Specific non-empty search query."],
    max_matches: Annotated[int, "Chunks to return. Use 5 normally, 10-15 if sparse."],
    min_similarity: Annotated[
        float, "Similarity threshold. Use -1.0 for broad search, 0.3-0.5 to filter."
    ],
) -> str:
    return search_documents(query, max_matches, min_similarity)


def _search_web_tool(
    query: Annotated[str, "Specific non-empty web search query."],
) -> str:
    return search_web_tool(query)


PLANNER_PROMPT = (
    "You are a research planner. Given a question, break it down into "
    "3-5 specific search queries to run against a document corpus.\n\n"
    "For broad questions such as 'what topics are covered', prefer broad "
    "content queries about subjects, themes, methods, findings, and limitations. "
    "Do not ask for filenames, table of contents, abstracts, headings, figures, "
    "or metadata unless the user explicitly asks for them.\n\n"
    "Output ONLY a numbered list of queries, nothing else.\n"
    "Example:\n"
    "1. What main subjects are discussed?\n"
    "2. What methods or concepts are covered?\n"
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
    tools: List[Any] = [
        FunctionTool(
            _search_documents_tool,
            name="search_documents",
            description=(
                "Search the indexed corpus. Always pass query, max_matches, "
                "and min_similarity."
            ),
            strict=True,
        )
    ]
    retriever_msg = (
        "You are a retrieval agent. Use search_documents to find chunks for "
        "each query.\n"
        "If a call returns nothing or off-topic chunks, don't give up: retry "
        "the same need with adjusted parameters (lower min_similarity, higher "
        "max_matches, or a reworded query) before moving on.\n"
        "Only report what the tool returns, don't invent content."
    )
    web_ready = config.websearch.enabled and bool(os.getenv("TAVILY_API_KEY"))
    if config.websearch.enabled and not web_ready:
        print(
            "Warning: websearch is enabled in config.yaml but TAVILY_API_KEY is not set; "
            "the Retriever will only use the corpus."
        )
    if web_ready:
        tools.append(
            FunctionTool(
                _search_web_tool,
                name="search_web_tool",
                description="Search the web with Tavily. Always pass query.",
                strict=True,
            )
        )
        retriever_msg = (
            "You are a retrieval agent with two tools:\n"
            "- search_documents: indexed corpus (tune max_matches / min_similarity)\n"
            "- search_web_tool: open web via Tavily\n\n"
            "Always pass a non-empty query string to either tool.\n"
            "Use search_documents for the first retrieval pass. If it returns "
            "some relevant chunks, report them and let the Critic decide. Do not "
            "use search_web_tool unless the Critic explicitly asks for web "
            "fallback. Report everything the tools return. Don't invent content."
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
            "For broad synthesis questions, chunks that directly summarize the "
            "documents' subjects, methods, findings, or limitations can be enough "
            "for COVERAGE_OK. Do not require document structure, metadata, or "
            "extra details unless the user asked for them.\n\n"
            "Output EXACTLY ONE verdict on the FIRST line:\n"
            "- COVERAGE_OK only if the chunks genuinely answer the question.\n"
            "- NEEDS_MORE if the chunks are off-topic, unrelated, missing key "
            "aspects, or too sparse.\n\n"
            "Never write both verdicts in the same reply.\n"
            "If NEEDS_MORE: list 1-3 follow-up queries. Use web search only when "
            "the original question needs external knowledge and the corpus is "
            "clearly unrelated."
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
            "Use ONLY the chunks that are actually relevant to the original question. "
            "Off-topic chunks are fine to ignore -- you do not need every chunk to be "
            "on-topic, as long as at least one chunk meaningfully addresses the question.\n\n"
            "Only if NO chunk at all addresses the question (all are off-topic, empty, "
            "or unrelated), reply EXACTLY:\n"
            "  The retrieved sources do not contain information answering this "
            "question. The corpus does not appear to cover this topic.\n"
            "Then end with TERMINATE. Never fill the gap with general knowledge.\n\n"
            "Otherwise, write:\n"
            "## Summary\n## Key Findings\n## Gaps & Limitations\n## Sources\n\n"
            "Every claim must be traceable to a specific on-topic chunk. Cite chunks "
            "with their bracketed label exactly as shown, e.g. [Chunk 3 | title] or "
            "[Web 2 | Title | URL]. Do not state anything that is not in a chunk, "
            "and do not cite a source that is not in the chunks above.\n\n"
            "End with: TERMINATE"
        ),
        model_context=_ctx(),
    )

    return retriever, critic, writer
