from autogen_agentchat.agents import AssistantAgent

from .agents import WRITER_PROMPT
from .mmore_client import reset_citations, retrieve
from .web_search import search_web


async def _write_from(question: str, chunks: str, model_client) -> str:
    writer = AssistantAgent(
        name="BaselineWriter",
        model_client=model_client,
        system_message=WRITER_PROMPT,
    )
    result = await writer.run(task=f"Question: {question}\n\nChunks:\n{chunks}")
    return result.messages[-1].content


async def run_baseline(question: str, model_client, max_matches: int = 10) -> str:
    # single retrieval pass, then the same writer instructions AMMORE uses, so
    # the comparison isolates the loop rather than the prompt
    reset_citations()
    return await _write_from(
        question, retrieve(question, max_matches=max_matches), model_client
    )


async def run_baseline_web(question: str, model_client) -> str:
    # single web search, same writer instructions, the web-only counterpart
    reset_citations()
    return await _write_from(question, search_web(question), model_client)
