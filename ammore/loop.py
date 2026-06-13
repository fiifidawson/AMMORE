from typing import Sequence

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat

from .agents import allow_web_search, create_agents
from .config import config


def make_selector():
    def selector(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if len(messages) <= 1:
            return "Retriever"

        last = messages[-1]
        sender = getattr(last, "source", None)
        text = (
            last.content
            if hasattr(last, "content") and isinstance(last.content, str)
            else ""
        )

        # force the Writer near the cap so we always get a synthesis
        if len(messages) >= config.loop.max_messages - 2 and sender != "Writer":
            return "Writer"

        if sender == "Retriever":
            return "Critic"
        if sender == "Critic":
            if "NEEDS_MORE" in text:
                if "search_web_tool" in text or "web" in text.lower():
                    allow_web_search()
                return "Retriever"
            if "COVERAGE_OK" in text:
                return "Writer"
            return "Retriever"

        return None

    return selector


def build_team(
    model_client, web_enabled: bool | None = None, web_only: bool = False
) -> SelectorGroupChat:
    retriever, critic, writer = create_agents(
        model_client, web_enabled=web_enabled, web_only=web_only
    )
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(
        config.loop.max_messages
    )
    return SelectorGroupChat(
        participants=[retriever, critic, writer],
        model_client=model_client,
        selector_func=make_selector(),
        termination_condition=termination,
    )


def task_text(question: str, approved_plan: str, web_only: bool = False) -> str:
    tool = "search_web_tool" if web_only else "search_documents"
    return (
        f"Original question: {question}\n\n"
        f"Search plan:\n{approved_plan}\n\n"
        f"Retriever: run each query above using {tool}."
    )
