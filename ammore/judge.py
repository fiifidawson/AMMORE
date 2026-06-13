"""Blind A/B judge over collected answers. Scores baseline vs ammore on the
same question. Position is randomized so the judge doesn't know which is which.
"""

import asyncio
import json
import random
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent

from .llm import get_judge_client

METRIC_DEFS = {
    "coverage": "does it address all aspects of the question?",
    "groundedness": "is every claim traceable to the cited chunks? penalize unsupported claims.",
    "specificity": "precise details vs vague generalities.",
    "ethical_awareness": "does it surface ethical dilemmas, harms, or tradeoffs where they apply?",
    "usefulness": "would a researcher find it actionable?",
}

DOMAINS = {
    "generic": ["coverage", "groundedness", "specificity", "usefulness"],
    "humanitarian": [
        "coverage",
        "groundedness",
        "specificity",
        "ethical_awareness",
        "usefulness",
    ],
}


def build_judge_prompt(metrics: list[str]) -> str:
    criteria = "\n".join(f"{m}: {METRIC_DEFS[m]}" for m in metrics)
    shape_a = ", ".join(f'"{m}": 3' for m in metrics)
    shape_b = ", ".join(f'"{m}": 4' for m in metrics)
    return (
        "You are a strict evaluator of literature reviews. You will see a "
        "question and two syntheses, A and B. You do not know which system "
        "produced which.\n\n"
        f"Score EACH synthesis from 1 to 5 on:\n\n{criteria}\n\n"
        "Be critical. 5 means near-perfect. Most should score 2-4.\n\n"
        "Output ONLY valid JSON, exactly this shape:\n"
        f'{{"A": {{{shape_a}}}, "B": {{{shape_b}}}, '
        '"reasoning": "one sentence on the key differences"}'
    )


async def judge_one(question, a, b, model_client, system_prompt, metrics):
    agent = AssistantAgent(
        name="Judge", model_client=model_client, system_message=system_prompt
    )
    prompt = (
        f"Question: {question}\n\n=== Synthesis A ===\n{a}\n\n=== Synthesis B ===\n{b}"
    )
    result = await agent.run(task=prompt)
    raw = result.messages[-1].content
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        return parsed["A"], parsed["B"], parsed.get("reasoning", "")
    except Exception:
        zero = {m: 0 for m in metrics}
        return zero, zero, f"unparseable: {raw[:120]}"


async def score_all(answers_path: Path, out_path: Path, domain: str):
    metrics = DOMAINS[domain]
    system_prompt = build_judge_prompt(metrics)
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    model_client = get_judge_client()
    scored = []
    if out_path.exists():
        scored = json.loads(out_path.read_text(encoding="utf-8"))
        done = {r["question"] for r in scored}
    else:
        done = set()

    for i, row in enumerate(answers, 1):
        q = row["question"]
        if q in done:
            print(f"[{i}/{len(answers)}] skip: {q[:60]}")
            continue
        print(f"[{i}/{len(answers)}] {q[:80]}")
        baseline_is_a = random.random() < 0.5
        if baseline_is_a:
            text_a, text_b = row["baseline"], row["ammore"]
        else:
            text_a, text_b = row["ammore"], row["baseline"]

        a_scores, b_scores, reasoning = await judge_one(
            q, text_a, text_b, model_client, system_prompt, metrics
        )
        baseline_scores = a_scores if baseline_is_a else b_scores
        ammore_scores = b_scores if baseline_is_a else a_scores

        scored.append(
            {
                "question": q,
                "baseline_scores": baseline_scores,
                "ammore_scores": ammore_scores,
                "reasoning": reasoning,
            }
        )
        out_path.write_text(
            json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  baseline {baseline_scores}")
        print(f"  ammore   {ammore_scores}")
        print(f"  {reasoning[:120]}")

    print_summary(scored, metrics)


def print_summary(scored: list, metrics: list[str]):
    n = len(scored)
    if n == 0:
        return
    print(f"\n{'metric':<20} {'baseline':>10} {'ammore':>10} {'delta':>8}")
    print("-" * 50)
    for m in metrics:
        b = sum(r["baseline_scores"].get(m, 0) for r in scored) / n
        a = sum(r["ammore_scores"].get(m, 0) for r in scored) / n
        d = a - b
        print(f"{m:<20} {b:>10.2f} {a:>10.2f} {d:>+8.2f}")
    bt = sum(sum(r["baseline_scores"].values()) for r in scored) / n
    at = sum(sum(r["ammore_scores"].values()) for r in scored) / n
    print("-" * 50)
    print(f"{'total avg':<20} {bt:>10.2f} {at:>10.2f} {at - bt:>+8.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="ammore.judge")
    parser.add_argument(
        "--answers",
        type=Path,
        default=Path("outputs") / "eval_answers.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / "eval_scores.json",
    )
    parser.add_argument(
        "--domain",
        choices=list(DOMAINS),
        default="generic",
        help="generic (default) or humanitarian (adds ethical_awareness)",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(score_all(args.answers, args.out, args.domain))


if __name__ == "__main__":
    main()
