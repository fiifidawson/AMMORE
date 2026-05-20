import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Sequence

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat

from .agents import create_agents, create_planner, reset_search_state
from .config import config
from .corpus import document_metadata_path, prepare_corpus
from .document_metadata import load_title_map
from .llm import get_model_client
from .mmore_client import set_title_map
from .retriever_launcher import auto_retriever


def _corpus_overview(corpus: Path | None) -> str:
    if corpus is None or not config.metadata_in_prompt:
        return ""
    path = document_metadata_path(corpus)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


async def run(question: str, corpus: Path | None = None):
    print(f"\n{'=' * 60}")
    print("AMMORE -- Agentic Literature Review")
    print(f"Question: {question}")
    print(f"{'=' * 60}\n")

    model_client = get_model_client()

    overview = _corpus_overview(corpus)

    # step 1: planner generates sub-questions
    print("Generating search plan...\n")
    planner = create_planner(model_client)
    planner_task = question
    if overview:
        planner_task = f"{question}\n\nThe corpus contains:\n{overview}"
    result = await planner.run(task=planner_task)
    sub_questions_text = result.messages[-1].content
    print(sub_questions_text)

    # step 2: let the user review and optionally edit the sub-questions
    print("\n" + "-" * 60)
    print("Press ENTER to approve, or type your own sub-questions:")
    print("-" * 60)
    user_input = input("> ").strip()

    if user_input:
        approved_plan = user_input
        print("\nUsing your version.")
    else:
        approved_plan = sub_questions_text
        print("\nApproved.")

    # step 3: run the retrieval loop
    print("\nStarting retrieval loop...\n")

    reset_search_state()
    retriever, critic, writer = create_agents(model_client)

    task = (
        f"Original question: {question}\n\n"
        f"Search plan:\n{approved_plan}\n\n"
        f"Retriever: run each query above using search_documents."
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(
        config.max_messages
    )

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
        if len(messages) >= config.max_messages - 2 and sender != "Writer":
            return "Writer"

        if sender == "Retriever":
            return "Critic"
        if sender == "Critic":
            if "NEEDS_MORE" in text:
                return "Retriever"
            if "COVERAGE_OK" in text:
                return "Writer"
            return "Retriever"

        return None

    team = SelectorGroupChat(
        participants=[retriever, critic, writer],
        model_client=model_client,
        selector_func=selector,
        termination_condition=termination,
    )

    collected = []

    try:
        async for msg in team.run_stream(task=task):
            collected.append(msg)
            src = getattr(msg, "source", "?")
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content:
                print(f"---------- {src} ----------\n{content}\n")
            elif isinstance(content, list):
                for item in content:
                    name = getattr(item, "name", None)
                    if name is not None:
                        args = getattr(item, "arguments", "")
                        print(
                            f"---------- {src} (tool call) ----------\n{name}({args})\n"
                        )
    except Exception as e:
        print(f"\nRun stopped early: {e}")

    save_output(question, collected)


def save_output(question, messages):
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    writer_text = ""
    for m in reversed(messages):
        if getattr(m, "source", None) == "Writer":
            writer_text = m.content
            break

    if not writer_text:
        print("\nNo writer output to save.")
        return

    name = datetime.now().strftime("review_%Y%m%d_%H%M%S.md")
    path = out_dir / name
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {question}\n\n{writer_text}\n")
    print(f"\nSaved to {path}")


def main():
    parser = argparse.ArgumentParser(prog="ammore")
    parser.add_argument("question", help="research question to investigate")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="folder of documents to index and search. If omitted, uses the corpus already configured in config.yaml.",
    )
    args = parser.parse_args()

    retriever_cfg = None
    if args.corpus is not None:
        retriever_cfg = prepare_corpus(args.corpus)
        set_title_map(load_title_map(document_metadata_path(args.corpus).parent))

    with auto_retriever(retriever_cfg):
        asyncio.run(run(args.question, args.corpus))


if __name__ == "__main__":
    main()
