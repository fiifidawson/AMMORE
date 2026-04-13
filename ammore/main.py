import asyncio
import sys
from typing import Sequence

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.ui import Console

from .llm import get_model_client
from .agents import create_agents, create_planner_only


async def run(question: str):
    print(f"\n{'='*60}")
    print(f"AMMORE -- Agentic Literature Review")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    model_client = get_model_client()

    # step 1: planner generates sub-questions
    print("Generating search plan...\n")
    planner_only = create_planner_only(model_client)
    result = await planner_only.run(task=question)
    sub_questions_text = result.messages[-1].content
    print(sub_questions_text)

    # step 2: let the user review and optionally edit the sub-questions
    print("\n" + "-"*60)
    print("Press ENTER to approve, or type your own sub-questions:")
    print("-"*60)
    user_input = input("> ").strip()

    if user_input:
        approved_plan = user_input
        print("\nUsing your version.")
    else:
        approved_plan = sub_questions_text
        print("\nApproved.")

    # step 3: run the retrieval loop
    print("\nStarting retrieval loop...\n")

    planner, retriever, critic, writer = create_agents(model_client)

    task = (
        f"Original question: {question}\n\n"
        f"Search plan:\n{approved_plan}\n\n"
        f"Retriever: run each query above using search_documents."
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(20)

    def selector(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if len(messages) <= 1:
            return "Retriever"

        last = messages[-1]
        sender = getattr(last, "source", None)
        text = last.content if hasattr(last, "content") and isinstance(last.content, str) else ""

        if sender == "Retriever":
            return "Critic"
        if sender == "Critic":
            return "Writer" if "COVERAGE_OK" in text else "Retriever"

        return None  # Writer just spoke, termination condition handles the rest

    team = SelectorGroupChat(
        participants=[retriever, critic, writer],
        model_client=model_client,
        selector_func=selector,
        termination_condition=termination,
    )

    await Console(team.run_stream(task=task))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ammore \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    asyncio.run(run(question))


if __name__ == "__main__":
    main()
