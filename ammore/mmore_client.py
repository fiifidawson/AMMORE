from pathlib import Path

import requests

from .config import config

_TITLE_MAP: dict = {}


def set_title_map(mapping: dict) -> None:
    global _TITLE_MAP
    _TITLE_MAP = mapping or {}


def _source_label(r: dict) -> str:
    file_path = r.get("filePath")
    if file_path:
        name = Path(file_path).name
        return _TITLE_MAP.get(name, name)
    return r.get("fileId", "unknown")


def _shorten(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n[...truncated {len(text) - limit} chars...]\n{text[-tail:]}"


def retrieve(
    query: str, max_matches: int | None = None, min_similarity: float | None = None
) -> str:
    max_matches = (
        max_matches if max_matches is not None else config.retrieval.max_matches
    )
    min_similarity = (
        min_similarity
        if min_similarity is not None
        else config.retrieval.min_similarity
    )

    try:
        response = requests.post(
            config.mmore.retriever_url,
            json={
                "query": query,
                "fileIds": [],
                "maxMatches": max_matches,
                "minSimilarity": min_similarity,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        return "ERROR: can't connect to mmore."
    except requests.RequestException as e:
        return f"ERROR: {e}"

    results = response.json()
    if not results:
        return "No results found."

    chunks = []
    total = 0
    for i, r in enumerate(results, 1):
        label = _source_label(r)
        content = _shorten(
            r.get("content", "").strip(), config.retrieval.max_chunk_chars
        )
        block = f"[Chunk {i} | {label}]\n{content}"
        if (
            config.retrieval.max_total_chars
            and total + len(block) > config.retrieval.max_total_chars
        ):
            chunks.append(f"[... {len(results) - i + 1} more chunk(s) omitted ...]")
            break
        chunks.append(block)
        total += len(block)

    return "\n\n---\n\n".join(chunks)
