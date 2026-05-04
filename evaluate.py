"""
evaluate.py — compare AMMORE vs single-pass baseline on LLM-as-judge metrics.

Usage (from AMMORE directory, with venv active):
    python evaluate.py

Results are printed as a table and saved to eval_results.json.
"""

import asyncio
import json
import os
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat

from ammore.agents import create_agents, create_planner_only
from ammore.llm import get_model_client
from ammore.mmore_client import retrieve

# ── Questions to evaluate ──────────────────────────────────────────────────────
# Replace with questions relevant to your indexed papers
QUESTIONS = [
    "How do CLIP and its variants align visual and language representations, and what role does training data quality play?",
    "What evaluation frameworks exist for multimodal LLMs, and what limitations do they identify?",
    "How do open-weight vision-language models like Molmo and LLaVA differ in their training approach and capabilities?",
    "What zero-shot transfer capabilities does CLIP demonstrate, and what are its limits?",
    "How does visual instruction tuning in LLaVA improve multimodal understanding compared to standard pretraining?",
    "What cultural biases or blind spots are identified in current multimodal LLM benchmarks?",
    "How do different caption generation strategies affect the quality of vision-language model training?",
    "What are the key architectural differences between CLIP, LLaVA, and Molmo?",
    "How do vision-language models handle ambiguous or culturally specific images?",
    "What are the computational costs and scalability challenges of training large vision-language models?",
    "How is contrastive learning used in vision-language models and what are its known failure cases?",
    "What makes PixMo dataset different from other vision-language training datasets?",
]

METRICS = ["coverage", "groundedness", "specificity", "usefulness"]


# ── Baseline: single mmore query → one LLM synthesis call ─────────────────────


async def run_baseline(question: str, model_client) -> str:
    chunks = retrieve(question, max_matches=10)

    writer = AssistantAgent(
        name="BaselineWriter",
        model_client=model_client,
        system_message=(
            "Based on the retrieved chunks below, write a structured synthesis.\n"
            "## Summary\n## Key Findings\n## Gaps & Limitations\n"
            "Use ONLY information from the chunks. Do not add outside knowledge."
        ),
    )
    result = await writer.run(task=f"Question: {question}\n\nChunks:\n{chunks}")
    return result.messages[-1].content


# ── AMMORE: full 4-agent loop, no human validation step ───────────────────────


async def run_ammore(question: str, model_client) -> str:
    # Generate sub-questions automatically (skip human input)
    planner_only = create_planner_only(model_client)
    result = await planner_only.run(task=question)
    sub_questions = result.messages[-1].content

    print(f"  Sub-questions:\n{sub_questions}")

    planner, retriever, critic, writer = create_agents(model_client)

    task = (
        f"Original question: {question}\n\n"
        f"Search plan:\n{sub_questions}\n\n"
        f"Retriever: run each query above using search_documents."
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(20)

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
        if sender == "Retriever":
            return "Critic"
        if sender == "Critic":
            return "Writer" if "COVERAGE_OK" in text else "Retriever"
        return None

    team = SelectorGroupChat(
        participants=[retriever, critic, writer],
        model_client=model_client,
        selector_func=selector,
        termination_condition=termination,
    )

    output = await team.run(task=task)

    critic_turns = [
        m for m in output.messages if getattr(m, "source", None) == "Critic"
    ]
    for i, msg in enumerate(critic_turns, 1):
        decision = "COVERAGE_OK" if "COVERAGE_OK" in msg.content else "NEEDS_MORE"
        print(f"  Critic turn {i} → {decision}")
    print(f"  Total Critic turns: {len(critic_turns)}")

    for msg in reversed(output.messages):
        if getattr(msg, "source", None) == "Writer":
            return msg.content
    return "No Writer output."


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are a strict expert evaluator of literature review quality.
You will receive a question and TWO syntheses (A = baseline, B = AMMORE).
Score EACH on 4 criteria from 1 to 5 using these strict rubrics:

coverage: 1=misses major aspects, 2=covers 1-2 aspects only, 3=covers main points but gaps, 4=thorough with minor gaps, 5=comprehensive
groundedness: 1=mostly hallucinated, 2=several unsupported claims, 3=mostly grounded with some guesses, 4=well-grounded minor issues, 5=fully faithful to sources
specificity: 1=only vague generalities, 2=few concrete details, 3=some specifics but mostly vague, 4=mostly concrete, 5=precise details throughout
usefulness: 1=not helpful, 2=marginally helpful, 3=somewhat useful, 4=clearly useful, 5=highly actionable for a researcher

Be critical. A score of 5 means near-perfect. Most answers should score 2-4.

Output ONLY valid JSON, exactly this format:
{"A": {"coverage": 3, "groundedness": 4, "specificity": 2, "usefulness": 3}, "B": {"coverage": 4, "groundedness": 4, "specificity": 3, "usefulness": 4}, "reasoning": "one sentence explaining key differences"}"""


async def judge(
    question: str, baseline: str, ammore: str, model_client
) -> tuple[dict, dict]:
    judge_agent = AssistantAgent(
        name="Judge",
        model_client=model_client,
        system_message=JUDGE_PROMPT,
    )

    prompt = (
        f"Question: {question}\n\n"
        f"=== Synthesis A (baseline) ===\n{baseline}\n\n"
        f"=== Synthesis B (AMMORE) ===\n{ammore}"
    )
    result = await judge_agent.run(task=prompt)
    text = result.messages[-1].content

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        parsed = json.loads(text[start:end])
        print(f"  Reasoning: {parsed.get('reasoning', '')}")
        return parsed["A"], parsed["B"]
    except Exception:
        print(f"  Warning: could not parse judge output: {text[:100]}")
        return {m: 0 for m in METRICS}, {m: 0 for m in METRICS}


# ── Main ──────────────────────────────────────────────────────────────────────


def print_table(results: list):
    print(f"\n\n{'=' * 60}")
    print("RESULTS — Baseline vs AMMORE (LLM-as-judge, scores 1–5)")
    print(f"{'=' * 60}")

    print(f"\n{'Metric':<16} {'Baseline':>10} {'AMMORE':>10} {'Δ':>6}")
    print("-" * 46)
    for metric in METRICS:
        b_avg = sum(r["baseline"].get(metric, 0) for r in results) / len(results)
        a_avg = sum(r["ammore"].get(metric, 0) for r in results) / len(results)
        delta = a_avg - b_avg
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<16} {b_avg:>10.2f} {a_avg:>10.2f} {sign}{delta:>5.2f}")

    b_total = sum(sum(r["baseline"].get(m, 0) for m in METRICS) for r in results) / len(
        results
    )
    a_total = sum(sum(r["ammore"].get(m, 0) for m in METRICS) for r in results) / len(
        results
    )
    print("-" * 46)
    print(f"{'Average total':<16} {b_total:>10.2f} {a_total:>10.2f}")


async def evaluate():
    model_client = get_model_client()

    # Load existing results to avoid re-running already evaluated questions
    results_file = "eval_results.json"
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        done = {r["question"] for r in all_results}
        print(
            f"Loaded {len(all_results)} existing results, skipping those questions.\n"
        )
    else:
        all_results = []
        done = set()

    for i, question in enumerate(QUESTIONS, 1):
        if question in done:
            print(f"[{i}/{len(QUESTIONS)}] SKIP (already done): {question[:60]}...")
            continue
        print(f"\n[{i}/{len(QUESTIONS)}] {question}")
        print("-" * 60)

        print("  → Baseline...")
        baseline_output = await run_baseline(question, model_client)

        print("  → AMMORE...")
        ammore_output = await run_ammore(question, model_client)

        print("  → Judging...")
        baseline_scores, ammore_scores = await judge(
            question, baseline_output, ammore_output, model_client
        )

        print(f"  Baseline: {baseline_scores}")
        print(f"  AMMORE:   {ammore_scores}")

        all_results.append(
            {
                "question": question,
                "baseline": baseline_scores,
                "ammore": ammore_scores,
                "baseline_output": baseline_output,
                "ammore_output": ammore_output,
            }
        )

    print_table(all_results)

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nFull results saved to eval_results.json")


if __name__ == "__main__":
    asyncio.run(evaluate())
