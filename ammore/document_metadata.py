"""Per-corpus title/abstract/summary overview built once from mmore's output."""

import json
import re
from pathlib import Path

from .config import config

_OVERVIEW_NAME = "document_metadata.md"

_ABSTRACT_RE = re.compile(r"\babstract\b", re.IGNORECASE)
_SECTION_RE = re.compile(
    r"\b(introduction|keywords|1\.?\s|background)\b", re.IGNORECASE
)


def _load_documents(merged_jsonl: Path):
    docs = []
    with open(merged_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            meta = d.get("metadata") or {}
            name = Path(meta.get("file_path", "unknown")).name
            docs.append((name, text))
    return docs


def _cheap_extract(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    title = "Untitled"
    for line in lines[:15]:
        if len(line) > 12 and not line.isupper():
            title = line
            break

    abstract = ""
    m = _ABSTRACT_RE.search(text)
    if m:
        after = text[m.end() :]
        end = _SECTION_RE.search(after)
        abstract = (after[: end.start()] if end else after[:1500]).strip()
    if not abstract:
        abstract = text[:600].strip()

    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    summary = " ".join(sentences[:3]).strip()
    return {"title": title, "abstract": abstract, "summary": summary}


_LLM_PROMPT = (
    "Extract metadata from the start of a research document.\n"
    'Reply with ONLY valid JSON: {"title": "...", "abstract": "...", "summary": "..."}\n'
    'summary = 2-3 sentences in your own words. If a field is missing, use "".'
)


async def _llm_extract(text: str, model_client) -> dict:
    from autogen_agentchat.agents import AssistantAgent

    agent = AssistantAgent(
        name="MetadataExtractor",
        model_client=model_client,
        system_message=_LLM_PROMPT,
    )
    result = await agent.run(task=text[:6000])
    raw = result.messages[-1].content
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return {
            "title": data.get("title", "Untitled") or "Untitled",
            "abstract": data.get("abstract", ""),
            "summary": data.get("summary", ""),
        }
    except Exception:
        return _cheap_extract(text)


def build_overview(merged_jsonl: Path, work_dir: Path) -> Path | None:
    mode = config.document_metadata.mode
    if mode == "none":
        return None

    out_path = work_dir / _OVERVIEW_NAME
    title_map_path = work_dir / "title_map.json"
    mode_stamp = work_dir / "metadata_mode.txt"

    # regen if the mode changed since last time
    cached_mode = mode_stamp.read_text().strip() if mode_stamp.exists() else ""
    if out_path.exists() and title_map_path.exists() and cached_mode == mode:
        return out_path
    if not merged_jsonl.exists():
        return None

    docs = _load_documents(merged_jsonl)
    if not docs:
        return None

    entries = []
    if mode == "llm":
        import asyncio

        from .llm import get_model_client

        model_client = get_model_client()

        async def _run():
            out = []
            for name, text in docs:
                out.append((name, await _llm_extract(text, model_client)))
            return out

        entries = asyncio.run(_run())
    else:  # cheap
        entries = [(name, _cheap_extract(text)) for name, text in docs]

    lines = ["# Corpus overview\n"]
    title_map = {}
    for name, meta in entries:
        lines.append(f"## {meta['title']}")
        lines.append(f"- file: {name}")
        if meta["summary"]:
            lines.append(f"- summary: {meta['summary']}")
        lines.append("")
        title_map[name] = meta["title"]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    title_map_path.write_text(
        json.dumps(title_map, ensure_ascii=False), encoding="utf-8"
    )
    mode_stamp.write_text(mode, encoding="utf-8")
    return out_path


def load_title_map(work_dir: Path) -> dict:
    path = work_dir / "title_map.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
