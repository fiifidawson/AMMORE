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
    "You are a research planner. Given a question, break it down into 4-6 "
    "specific search queries to run against a document corpus.\n\n"
    "Make the queries cover DIFFERENT facets of the question so that together "
    "they retrieve comprehensive evidence -- distinct sub-topics, methods, "
    "datasets, populations, or viewpoints the question touches. For a "
    "comparison or synthesis question, write one query per thing being compared "
    "or per theme, so no aspect is missed.\n"
    "Prefer broad content queries about subjects, themes, methods, findings, "
    "and limitations. Do not ask for filenames, table of contents, headings, or "
    "metadata unless the user explicitly asks for them.\n\n"
    "Output ONLY a numbered list of queries, nothing else.\n"
    "Example:\n"
    "1. What main subjects or tasks are addressed?\n"
    "2. What methods or approaches are used?\n"
    "3. What datasets or populations are studied?\n"
    "4. What limitations or open problems are noted?\n"
)


def create_planner(model_client):
    return AssistantAgent(
        name="Planner",
        model_client=model_client,
        description="Breaks the question into sub-queries.",
        system_message=PLANNER_PROMPT,
        model_context=_ctx(),
    )


def create_agents(
    model_client, web_enabled: bool | None = None, web_only: bool = False
):
    if web_enabled is None:
        web_enabled = config.websearch.enabled

    if web_only:
        allow_web_search()
        retriever = AssistantAgent(
            name="Retriever",
            model_client=model_client,
            description="Searches the web.",
            system_message=(
                "You are a retrieval agent. There is no document corpus. Use "
                "search_web_tool to find sources for each query. Always pass a "
                "non-empty query string. Report everything the tool returns, "
                "don't invent content."
            ),
            tools=[
                FunctionTool(
                    _search_web_tool,
                    name="search_web_tool",
                    description="Search the web with Tavily. Always pass query.",
                    strict=True,
                )
            ],
            reflect_on_tool_use=False,
            model_context=_ctx(),
        )
        return retriever, _make_critic(model_client), _make_writer(model_client)

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
    web_ready = web_enabled and bool(os.getenv("TAVILY_API_KEY"))
    if web_enabled and not web_ready:
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

    return retriever, _make_critic(model_client), _make_writer(model_client)


def _make_critic(model_client):
    return AssistantAgent(
        name="Critic",
        model_client=model_client,
        description="Checks if we have enough information to write the answer.",
        system_message=(
            "Judge whether the retrieved chunks actually address the ORIGINAL "
            "question -- not whether chunks merely exist. Chunks that are on a "
            "different topic count as no coverage.\n\n"
            "For broad synthesis or comparison questions, chunks that cover the "
            "relevant subjects, methods, findings, or limitations across the "
            "documents are enough for COVERAGE_OK -- the answer is built by "
            "combining them, so do NOT require a single chunk to answer the whole "
            "question, and do not require document structure or metadata.\n\n"
            "Output EXACTLY ONE verdict on the FIRST line:\n"
            "- COVERAGE_OK if the chunks together let the Writer answer the "
            "question, even partially.\n"
            "- NEEDS_MORE only if the chunks are off-topic, unrelated, or miss a "
            "key aspect that a targeted follow-up query could still find in the "
            "corpus.\n\n"
            "Prefer COVERAGE_OK once there is usable on-topic material; reserve "
            "NEEDS_MORE for genuinely thin or off-topic results.\n"
            "Never write both verdicts in the same reply.\n"
            "If NEEDS_MORE: list 1-3 follow-up queries. Use web search only when "
            "the original question needs external knowledge and the corpus is "
            "clearly unrelated."
        ),
        model_context=_ctx(),
    )


# shared by the agentic Writer and the single-pass baseline so the only
# difference between them is the loop, not the writing instructions
WRITER_PROMPT = (
    "You write a thorough literature review STRICTLY from the retrieved "
    "chunks. You have no other knowledge.\n\n"
    "Use every chunk relevant to the question's topic; ignore off-topic ones. "
    "For comparison or synthesis questions, build the answer by combining what "
    "the relevant chunks say -- a single chunk need not answer the whole "
    "question on its own. Refuse ONLY if no chunk is relevant to the topic at "
    "all, replying EXACTLY:\n"
    "  The retrieved sources do not contain information answering this "
    "question. The corpus does not appear to cover this topic.\n"
    "Then end with TERMINATE. Never fill gaps with general knowledge.\n\n"
    "Otherwise write a detailed review. Be comprehensive: cover every "
    "relevant point in the chunks, not just a few. Use these sections:\n"
    "## Summary -- two or three paragraphs giving the overall picture.\n"
    "## Key Findings -- group the evidence into 3-6 themes. Start each "
    "theme with a level-3 markdown heading, for example:\n"
    "  ### Datasets and scale\n"
    "Follow it with a full paragraph that explains what the sources say, "
    "where they agree or differ, and the specifics (numbers, methods, "
    "datasets, names). Write prose, not one-line bullets.\n"
    "## Gaps & Limitations -- what the sources leave out or where they are "
    "weak.\n"
    "## Sources -- list every source you cited, one per line, with the full "
    "label including the URL for web sources.\n\n"
    "Every claim must trace to a specific chunk. Cite inline with the "
    "bracketed label exactly as shown, e.g. [Chunk 3 | title] or "
    "[Web 2 | Title | URL]. For a web source keep the whole URL in both the "
    "inline citation and the Sources list. Do not state anything that is "
    "not in a chunk, and do not invent sources.\n\n"
    "End with: TERMINATE"
)


def _make_writer(model_client):
    return AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="Writes the final synthesis.",
        system_message=WRITER_PROMPT,
        model_context=_ctx(),
    )
