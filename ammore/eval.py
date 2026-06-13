"""Run AMMORE and a single-pass baseline on a list of questions, save outputs.

The judge / scoring step lives elsewhere; this file just collects answers.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from .agents import create_planner, reset_search_state
from .baseline import run_baseline, run_baseline_web
from .corpus import prepare_corpus
from .llm import get_model_client
from .loop import build_team, task_text
from .retriever_launcher import auto_retriever


async def run_ammore(question: str, model_client, web_only: bool = False) -> str:
    reset_search_state()
    planner = create_planner(model_client)
    plan = await planner.run(task=question)
    sub_questions = plan.messages[-1].content

    team = build_team(model_client, web_only=web_only)
    output = await team.run(task=task_text(question, sub_questions, web_only=web_only))
    for msg in reversed(output.messages):
        if getattr(msg, "source", None) == "Writer":
            return msg.content
    return ""


async def collect(questions: list[str], out_path: Path, web_only: bool = False) -> None:
    model_client = get_model_client()
    results = []
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        done = {r["question"] for r in results}
        print(f"Resuming, {len(results)} answers already collected")
    else:
        done = set()

    for i, q in enumerate(questions, 1):
        if q in done:
            print(f"[{i}/{len(questions)}] skip: {q[:60]}")
            continue
        print(f"\n[{i}/{len(questions)}] {q}")
        try:
            print("  baseline...")
            if web_only:
                baseline_out = await run_baseline_web(q, model_client)
            else:
                baseline_out = await run_baseline(q, model_client)
            print("  ammore...")
            ammore_out = await run_ammore(q, model_client, web_only=web_only)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        results.append(
            {
                "question": q,
                "baseline": baseline_out,
                "ammore": ammore_out,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        out_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"\nSaved {len(results)} answers to {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="ammore.eval")
    parser.add_argument("questions", help="path to a .txt file (one question per line)")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="folder of documents or JSON file (not needed with --web-only)",
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="no corpus: compare AMMORE's web loop vs a single web search",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / "eval_answers.json",
        help="where to write answers (resumes if exists)",
    )
    args = parser.parse_args()

    qs = [
        line.strip()
        for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not qs:
        raise SystemExit("no questions in file")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.web_only:
        asyncio.run(collect(qs, args.out, web_only=True))
    else:
        if args.corpus is None:
            raise SystemExit("--corpus is required unless --web-only")
        retriever_cfg = prepare_corpus(args.corpus)
        with auto_retriever(retriever_cfg):
            asyncio.run(collect(qs, args.out))


if __name__ == "__main__":
    main()
